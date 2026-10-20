"""Where does the compiled K1 velocity reward chain produce -inf?

K1 velocity runs with ``reward.compile_terms=True`` die the same way on
every stack (CHTC L40, pal4): on one step ``upper_body_posture`` comes
out -inf for a single environment while every other term, the commands
and the joint state stay finite; the -inf enters the returns and the
policy is NaN from the next iteration on. The eager chain never does
this. ``upper_body_posture`` is ``-mean(err^2 / std^2)`` with ``std``
from ``_regime_blend``: a masked sum over three speed regimes whose
masks are exhaustive for a finite speed, so the only way to a -inf is a
std of zero, i.e. every mask false.

Hypothesis under test: inductor recomputes the cheap ``speed = |v_xy| +
|w_z|`` inside each fused kernel, and two recomputations can disagree by
an ULP around a threshold, so ``speed < wt`` and ``speed >= wt`` are
both false for that env. ``upright`` shares the blend but wraps it in
``exp(-x / std^2)``, which turns a zero std into 0, not -inf, so it
would not show.

Three phases, each independent of the others:

1. ``standalone``: a replica of the two blend terms (same formulas, same
   K1 std tables) compiled with ``fullgraph=True``, fed commands whose
   speed straddles the thresholds 0.05 / 1.0 / 1.5 by -ULPS..+ULPS in
   several compositions (x only, w only, x and y at a random heading, x
   plus w). Compiled vs eager, all-mask-zero rows counted. The same
   sweep also runs a complement-mask variant of the blend (``walking = 1
   - standing - running``) to see whether it closes the hole.
2. ``inject``: the live K1 env with the compiled chain, the velocity
   command tensor overwritten with the same sweep right before each
   reward call, every term compared against the eager chain on the same
   state (the parity gate's save/restore around the eager reference).
3. ``run``: the live env stepped with random actions for many steps, no
   injection; the first non-finite term value stops the run and dumps
   that env's commands, speed and the eager recomputation of the term.

Usage (GPU box, from the SimForge root):
    jaxpy -m jaxrlworld.scripts.diag.k1.regime_blend_compile_diag --phase standalone
    jaxpy -m jaxrlworld.scripts.diag.k1.regime_blend_compile_diag --phase inject --sim mujoco
    jaxpy -m jaxrlworld.scripts.diag.k1.regime_blend_compile_diag --phase run --sim genesis --steps 20000
"""

from __future__ import annotations

import argparse
import importlib
import math

import torch
import torch._dynamo

from jaxrlworld.rl.envs.managers.common.command_term import VelocityCommandTerm
from jaxrlworld.rl.envs.mdp.rewards import k1_velocity as k1_rf
from jaxrlworld.rl.runners import BaseRunner
from jaxrlworld.scripts.diag.gates.check_reward_compile_parity import TermState

_PRESET = "jaxrlworld.rl.configs.presets.k1_velocity.base:K1VelocityConfig"

# The K1 preset's blend thresholds (base.py: posture 0.05 / 1.0, upright 0.05 / 1.5).
_THRESHOLDS = (0.05, 1.0, 1.5)
# Per-joint std tables of upper_body_posture in the preset's joint order
# (Head yaw/pitch, L/R shoulder pitch/roll, L/R elbow pitch/yaw).
_STD_STANDING = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
_STD_WALKING = [0.05, 0.05, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15]
_STD_RUNNING = [0.05, 0.05, 0.5, 0.2, 0.5, 0.2, 0.35, 0.35, 0.35, 0.35]


# ── Threshold sweep ──────────────────────────────────────────────────


def _ulp_ladder(center: float, ulps: int, device) -> torch.Tensor:
    """``center`` and its ``ulps`` float32 neighbours on each side, ascending."""
    c = torch.tensor(center, dtype=torch.float32, device=device)
    up = [c]
    down = []
    hi = c
    lo = c
    inf = torch.tensor(math.inf, dtype=torch.float32, device=device)
    for _ in range(ulps):
        hi = torch.nextafter(hi, inf)
        lo = torch.nextafter(lo, -inf)
        up.append(hi)
        down.append(lo)
    return torch.stack(down[::-1] + up)


def sweep_commands(ulps: int, headings: int, device, gen: torch.Generator) -> torch.Tensor:
    """``(N, 3)`` velocity commands whose speed straddles every threshold.

    Compositions per threshold value ``t``: ``(t, 0, 0)``; ``(0, 0, t)``;
    ``(t cos h, t sin h, 0)`` for ``headings`` random headings; ``(a, 0,
    t - a)`` for a random split. The straddling values are the ULP
    ladder around ``t``; the compositions with a norm or a sum re-round,
    so the realised speed lands within a few ULPs of the ladder value.
    """
    rows = []
    for t in _THRESHOLDS:
        ladder = _ulp_ladder(t, ulps, device)
        n = ladder.numel()
        zeros = torch.zeros(n, device=device)
        rows.append(torch.stack([ladder, zeros, zeros], dim=1))
        rows.append(torch.stack([zeros, zeros, ladder], dim=1))
        for _ in range(headings):
            h = torch.rand((), generator=gen, device=device) * 2 * math.pi
            rows.append(torch.stack([ladder * torch.cos(h), ladder * torch.sin(h), zeros], dim=1))
            a = torch.rand((), generator=gen, device=device) * t
            rows.append(torch.stack([ladder * 0 + a, zeros, ladder - a], dim=1))
            # Negative components: |.| and the norm are what the terms see.
            rows.append(torch.stack([-ladder * torch.cos(h), ladder * torch.sin(h), zeros], dim=1))
            rows.append(torch.stack([zeros, zeros, -ladder], dim=1))
    return torch.cat(rows, dim=0)


def random_commands(n: int, device, gen: torch.Generator) -> torch.Tensor:
    """Commands from the K1 preset's ranges (x in [-1, 1.2], y and w in [-1, 1])."""
    u = torch.rand(n, 3, generator=gen, device=device)
    lo = torch.tensor([-1.0, -1.0, -1.0], device=device)
    hi = torch.tensor([1.2, 1.0, 1.0], device=device)
    return lo + u * (hi - lo)


# ── Phase 1: standalone replica ──────────────────────────────────────


def _speed(cmd: torch.Tensor) -> torch.Tensor:
    return torch.norm(cmd[:, :2], dim=1) + torch.abs(cmd[:, 2])


def _blend_masked_sum(speed, standing, walking, running, wt, rt):
    """``k1_velocity._regime_blend`` verbatim."""
    standing_mask = (speed < wt).float()
    walking_mask = ((speed >= wt) & (speed < rt)).float()
    running_mask = (speed >= rt).float()
    if isinstance(standing, torch.Tensor) and standing.dim() == 1:
        standing_mask = standing_mask.unsqueeze(1)
        walking_mask = walking_mask.unsqueeze(1)
        running_mask = running_mask.unsqueeze(1)
    return standing * standing_mask + walking * walking_mask + running * running_mask


def _blend_complement(speed, standing, walking, running, wt, rt):
    """Candidate: the walking mask as the complement of the other two."""
    standing_mask = (speed < wt).float()
    running_mask = (speed >= rt).float()
    walking_mask = 1.0 - standing_mask - running_mask
    if isinstance(standing, torch.Tensor) and standing.dim() == 1:
        standing_mask = standing_mask.unsqueeze(1)
        walking_mask = walking_mask.unsqueeze(1)
        running_mask = running_mask.unsqueeze(1)
    return standing * standing_mask + walking * walking_mask + running * running_mask


def _make_chain(blend):
    def chain(cmd, joint_err, gravity_xy_sq, stds):
        """The two blend terms as the reward chain fuses them.

        ``posture`` is ``upper_body_posture_penalty`` on ``joint_err =
        current - default``; ``upright`` is ``variable_upright`` with the
        preset's scalar stds; ``mask_sum`` is the number of true masks
        per env of the posture blend, which must be exactly one.
        """
        standing, walking, running = stds
        speed = _speed(cmd)
        std_p = blend(speed, standing, walking, running, 0.05, 1.0)
        posture = -torch.mean(torch.square(joint_err) / torch.square(std_p), dim=1)
        std_u = blend(speed, 0.5, 0.75, 1.0, 0.05, 1.5)
        upright = torch.exp(-gravity_xy_sq / torch.square(std_u))
        mask_sum = (speed < 0.05).float() + ((speed >= 0.05) & (speed < 1.0)).float() + (speed >= 1.0).float()
        return torch.stack([posture, upright], dim=0), mask_sum, speed

    return chain


def phase_standalone(args, device) -> int:
    gen = torch.Generator(device=device).manual_seed(args.seed)
    stds = tuple(torch.tensor(t, device=device) for t in (_STD_STANDING, _STD_WALKING, _STD_RUNNING))
    eager = _make_chain(_blend_masked_sum)
    compiled = torch.compile(_make_chain(_blend_masked_sum), fullgraph=True, dynamic=False)
    compiled_fix = torch.compile(_make_chain(_blend_complement), fullgraph=True, dynamic=False)

    sweep = sweep_commands(args.ulps, args.headings, device, gen)
    n_sweep = sweep.shape[0]
    print(
        f"standalone replica on {device}: sweep {n_sweep} commands x {args.rounds} rounds, random {args.random_per_round} per round"
    )
    print(
        f"  torch {torch.__version__}, cuda {torch.version.cuda}, device {torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu'}"
    )

    worst_rel = 0.0
    bad_rows = 0
    bad_fix_rows = 0
    mask_holes = 0
    nonfinite = 0
    examples: list[str] = []
    checked = 0
    for r in range(args.rounds):
        cmd = torch.cat([sweep, random_commands(args.random_per_round, device, gen)], dim=0)
        n = cmd.shape[0]
        joint_err = 0.3 * torch.randn(n, len(_STD_STANDING), generator=gen, device=device)
        gravity = 0.2 * torch.rand(n, generator=gen, device=device)
        ref, ref_masks, ref_speed = eager(cmd, joint_err, gravity, stds)
        out, masks, speed = compiled(cmd, joint_err, gravity, stds)
        out_fix, _, _ = compiled_fix(cmd, joint_err, gravity, stds)
        checked += n
        if not torch.isfinite(ref).all():
            raise RuntimeError("eager replica produced a non-finite value; the sweep itself is wrong")
        nonfinite += int((~torch.isfinite(out)).any(dim=0).sum())
        mask_holes += int((masks != 1).sum())
        rel = (out - ref).abs() / ref.abs().clamp_min(1e-6)
        worst_rel = max(worst_rel, float(rel.max()))
        bad = (rel > args.tol).any(dim=0)
        bad_rows += int(bad.sum())
        bad_fix_rows += int(((out_fix - ref).abs() / ref.abs().clamp_min(1e-6) > args.tol).any(dim=0).sum())
        if bad.any() and len(examples) < 12:
            for i in torch.nonzero(bad).flatten()[:4].tolist():
                examples.append(
                    f"    round {r} row {i}: cmd={cmd[i].tolist()} speed eager={float(ref_speed[i]):.9g} "
                    f"compiled={float(speed[i]):.9g} masks={float(masks[i]):g} "
                    f"posture eager={float(ref[0, i]):.6g} compiled={float(out[0, i]):.6g} "
                    f"upright eager={float(ref[1, i]):.6g} compiled={float(out[1, i]):.6g}"
                )
    print(f"  rows checked {checked}")
    print(f"  compiled rows with a non-finite term     {nonfinite}")
    print(f"  compiled rows with mask sum != 1          {mask_holes}")
    print(f"  compiled rows off eager by > {args.tol:g} rel   {bad_rows}   (worst {worst_rel:.3e})")
    print(f"  complement-mask compiled rows off eager   {bad_fix_rows}")
    for line in examples:
        print(line)
    if nonfinite or mask_holes or bad_rows:
        print("\nREPRODUCED: the compiled masked-sum blend disagrees with eager")
        return 1
    print("\nNOT REPRODUCED in the standalone replica")
    return 0


# ── Phases 2 and 3: the live env ────────────────────────────────────


def _build_env(sim: str, num_envs: int):
    mod_path, cls_name = _PRESET.split(":", 1)
    cfg_cls = getattr(importlib.import_module(mod_path), cls_name)
    cfgs = cfg_cls(sim_type=sim, num_envs=num_envs).build()
    cfgs.reward.compile_terms = True
    runner = BaseRunner.create_with_env(cfgs, use_wandb=False)
    return runner.env


def _velocity_term(env):
    for _name, term in env.command_manager.iter_terms():
        if isinstance(term, VelocityCommandTerm):
            return term
    raise RuntimeError("no velocity command term on the env")


def _blend_term_names(mgr) -> dict[str, str]:
    """Reward-term key of the posture and upright terms by their function name."""
    out = {}
    for name, term in mgr.reward_terms.items():
        fn_name = term.func.__name__
        if fn_name == "upper_body_posture_penalty":
            out["posture"] = name
        elif fn_name == "variable_upright":
            out["upright"] = name
    if set(out) != {"posture", "upright"}:
        raise RuntimeError(f"K1 blend terms not found among {list(mgr.reward_terms)}")
    return out


def phase_inject(args, device) -> int:
    torch.manual_seed(args.seed)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    env = _build_env(args.sim, args.num_envs)
    mgr = env.reward_manager
    names = list(mgr.reward_terms)
    blend_names = _blend_term_names(mgr)
    cmd_term = _velocity_term(env)
    sweep = sweep_commands(args.ulps, args.headings, device, gen)
    # Every step takes a random subset of the sweep (all of it when it
    # fits) and fills the remaining envs from the preset's ranges.
    n_sweep = min(sweep.shape[0], env.num_envs - env.num_envs // 8)
    print(
        f"inject ({args.sim}, {env.num_envs} envs): {n_sweep} of {sweep.shape[0]} sweep rows per step, "
        f"{env.num_envs - n_sweep} random from the preset ranges"
    )

    original = mgr.set_rewards
    stats = {"compared": 0, "bad": 0, "nonfinite": 0}
    worst = {name: 0.0 for name in names}
    examples: list[str] = []

    def checked_set_rewards(reward_buffer, reward_buffer_per_type):
        if mgr._view is None:
            original(reward_buffer=reward_buffer, reward_buffer_per_type=reward_buffer_per_type)
            return
        pick = torch.randperm(sweep.shape[0], generator=gen, device=device)[:n_sweep]
        cmd = torch.cat([sweep[pick], random_commands(env.num_envs - n_sweep, device, gen)], dim=0)
        cmd_term._command.copy_(cmd)
        state = TermState(mgr)
        ref = mgr._compute_stacked(mgr.env)
        state.restore()
        original(reward_buffer=reward_buffer, reward_buffer_per_type=reward_buffer_per_type)
        stats["compared"] += 1
        if not torch.isfinite(ref).all():
            raise RuntimeError("eager chain produced a non-finite value on the injected commands")
        for i, name in enumerate(names):
            got = reward_buffer_per_type[name]
            nonfinite = ~torch.isfinite(got)
            rel = (got - ref[i]).abs() / ref[i].abs().max().clamp_min(1e-6)
            worst[name] = max(worst[name], float(rel[~nonfinite].max()) if (~nonfinite).any() else 0.0)
            bad = nonfinite | (rel > args.tol)
            if bad.any():
                stats["bad"] += int(bad.sum())
                stats["nonfinite"] += int(nonfinite.sum())
                if len(examples) < 12:
                    for e in torch.nonzero(bad).flatten()[:3].tolist():
                        examples.append(
                            f"    step {stats['compared']} {name} env {e}: cmd={cmd[e].tolist()} "
                            f"speed={float(k1_rf._commanded_speed(env)[e]):.9g} eager={float(ref[i, e]):.6g} compiled={float(got[e]):.6g}"
                        )

    mgr.set_rewards = checked_set_rewards
    action = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    for step in range(args.steps):
        env.step(0.1 * torch.randn_like(action))
        if step == 2:
            torch._dynamo.config.error_on_recompile = True

    print(f"  steps compared {stats['compared']}")
    for name in names:
        tag = "  <- blend" if name in blend_names.values() else ""
        print(f"  {name:<34} max rel diff {worst[name]:.2e}{tag}")
    print(f"  env-steps off eager (> {args.tol:g} rel or non-finite): {stats['bad']}  non-finite: {stats['nonfinite']}")
    for line in examples:
        print(line)
    if stats["bad"]:
        print("\nREPRODUCED on the live env")
        return 1
    print("\nNOT REPRODUCED on the live env with injected threshold commands")
    return 0


def phase_run(args, device) -> int:
    torch.manual_seed(args.seed)
    env = _build_env(args.sim, args.num_envs)
    mgr = env.reward_manager
    names = list(mgr.reward_terms)
    blend_names = _blend_term_names(mgr)
    cmd_term = _velocity_term(env)
    print(
        f"run ({args.sim}, {env.num_envs} envs, {args.steps} steps, random actions): stop at the first non-finite term"
    )

    original = mgr.set_rewards
    hit: dict = {}

    def trapped_set_rewards(reward_buffer, reward_buffer_per_type):
        original(reward_buffer=reward_buffer, reward_buffer_per_type=reward_buffer_per_type)
        if mgr._view is None or hit:
            return
        stacked = torch.stack([reward_buffer_per_type[n] for n in names], dim=0)
        bad = ~torch.isfinite(stacked)
        if bad.any():
            envs = torch.nonzero(bad.any(dim=0)).flatten()
            cmd = cmd_term._command[envs].clone()
            speed = k1_rf._commanded_speed(env)[envs].clone()
            # Eager recomputation of every term on the same state.
            state = TermState(mgr)
            ref = mgr._compute_stacked(mgr.env)
            state.restore()
            hit.update(envs=envs, cmd=cmd, speed=speed, compiled=stacked[:, envs].clone(), eager=ref[:, envs].clone())

    mgr.set_rewards = trapped_set_rewards
    action = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    for step in range(args.steps):
        env.step(0.1 * torch.randn_like(action))
        if hit:
            print(f"\n  NON-FINITE at step {step} in {hit['envs'].numel()} env(s): {hit['envs'].tolist()}")
            for j, e in enumerate(hit["envs"].tolist()):
                print(f"    env {e}: cmd={hit['cmd'][j].tolist()} speed={float(hit['speed'][j]):.9g}")
                for i, name in enumerate(names):
                    c = float(hit["compiled"][i, j])
                    r = float(hit["eager"][i, j])
                    flag = "  <- blend" if name in blend_names.values() else ""
                    mark = "  **" if not math.isfinite(c) or abs(c - r) > args.tol * max(abs(r), 1e-6) else ""
                    print(f"      {name:<34} compiled {c:>14.6g}  eager {r:>14.6g}{flag}{mark}")
            print(
                "\nREPRODUCED: compiled chain non-finite where the eager chain is finite"
                if torch.isfinite(hit["eager"]).all()
                else "\nnon-finite in BOTH chains: not a compile fault"
            )
            return 1
        if step % 1000 == 0 and step:
            print(f"  step {step}: finite", flush=True)
    print(
        f"\nNOT REPRODUCED in {args.steps} steps x {env.num_envs} envs ({args.steps * env.num_envs / 1e6:.0f}M env-steps)"
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="standalone", choices=("standalone", "inject", "run"))
    ap.add_argument("--sim", default="mujoco", choices=("genesis", "newton", "mujoco"))
    ap.add_argument("--num-envs", type=int, default=8192)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ulps", type=int, default=64, help="float32 neighbours on each side of every threshold")
    ap.add_argument("--headings", type=int, default=8, help="random headings per threshold in the sweep")
    ap.add_argument("--rounds", type=int, default=200, help="standalone: sweep repetitions")
    ap.add_argument("--random-per-round", type=int, default=65536, help="standalone: random commands per round")
    ap.add_argument("--tol", type=float, default=1e-4)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.phase == "standalone":
        return phase_standalone(args, device)
    if args.phase == "inject":
        return phase_inject(args, device)
    return phase_run(args, device)


if __name__ == "__main__":
    raise SystemExit(main())
