"""feet_slip cross-sim forensics: where exactly does mujoco's number diverge?

Observed on g1 / g1 rough / go2 / k1: the mujoco cell's feet_slip
penalty is 10-50% smaller in magnitude than genesis/newton. The go2
audit (2026-08-25) proved all parameters equal there and attributed the
residual to engine-level tangential creep (finite-difference foot speed
gs 0.0189 / nt 0.0172 / mj 0.0162 m/s, whose squared ratios matched the
reward ratios). This diag generalizes that proof to ANY preset, and
re-verifies per preset the hypotheses that were only checked on go2:

  A. Source dump      — the term's actual function, params, resolved
                        selector (site vs body ids/names), the contact
                        group's tracked names, and each backend's
                        is_contact semantics. "Are the three sims even
                        reading the same things?"
  B. Geometry         — settled foot positions of the term's own
                        position source, cross-sim (a site-vs-link
                        measurement-point mismatch shows up here as an
                        offset), plus per-sim velocity-source integrity:
                        the term's velocity against a finite difference
                        of the term's own positions.
  C. Decomposition    — captured INSIDE the reward manager's own call
                        (same cache generation as training): raw term
                        value, contact fraction, mean squared planar
                        foot speed over contacted feet, and the command
                        gate. reward ≈ E[v² · contact] · gate, so the
                        cross-sim ratio factors into "contacts differ"
                        vs "slip speed differs".
  D. Creep fingerprint— zero action, commands untouched, no reward code:
                        finite-difference planar foot speed of contacted
                        feet at steady standing. This is the pure
                        constraint-solver residual. The verdict compares
                        creep² ratios against the measured reward ratios
                        — a match proves the divergence is the engines'
                        friction creep, not our plumbing.

Dead hypotheses from the go2 audit (do not re-chase without new
evidence): mjlab one-substep staleness (measured zero), go2 site-vs-link
measurement point (coincident there — phase B re-checks it per preset),
"the sims run different robots" (pair/param/dof parity diags).

Usage:
    jaxpy -m rlworld.scripts.diag.parity.feet_slip_forensics --preset go2_gait
    jaxpy -m rlworld.scripts.diag.parity.feet_slip_forensics --preset g1_29dof
    jaxpy -m rlworld.scripts.diag.parity.feet_slip_forensics --preset g1_29dof_rough
    jaxpy -m rlworld.scripts.diag.parity.feet_slip_forensics --preset k1_joystick
"""

from __future__ import annotations

import argparse
import importlib
import os

os.environ.setdefault("JAXRLWORLD_ALLOW_MULTI_SIM", "1")
os.environ.setdefault("JAXRLWORLD_PLAIN_LOG", "1")

import numpy as np
import torch

_SIMS = ("genesis", "newton", "mujoco")

# name -> (module, class, ctor kwargs)
_PRESETS: dict[str, tuple[str, str, dict]] = {
    "go2": ("rlworld.rl.configs.presets.go2.base", "Go2FlatConfig", {}),
    "g1_29dof": ("rlworld.rl.configs.presets.g1_29dof.base", "G1FlatConfig", {}),
    "g1_29dof_rough": ("rlworld.rl.configs.presets.g1_29dof.base", "G1FlatConfig", {"use_rough_terrain": True}),
    "k1_joystick": ("rlworld.rl.configs.presets.k1_joystick.base", "K1JoystickConfig", {}),
    "k1_g1_recipe": ("rlworld.rl.configs.presets.k1_joystick.g1_recipe", "K1G1RecipeConfig", {}),
}
_PER_SIM_PRESETS: dict[str, dict[str, tuple[str, str, dict]]] = {
    "go2_gait": {
        "genesis": ("rlworld.rl.configs.presets.go2.genesis.gait_conditioned", "Go2GaitConditionedGenesisConfig", {}),
        "newton": ("rlworld.rl.configs.presets.go2.newton.gait_conditioned", "Go2GaitConditionedNewtonConfig", {}),
        "mujoco": ("rlworld.rl.configs.presets.go2.mujoco.gait_conditioned", "Go2GaitConditionedMujocoConfig", {}),
    },
}


def _build_env(preset: str, sim: str, num_envs: int):
    from rlworld.rl.runners import BaseRunner

    if ":" in preset:
        mod_path, cls_name, kwargs = *preset.split(":", 1), {}
    elif preset in _PER_SIM_PRESETS:
        mod_path, cls_name, kwargs = _PER_SIM_PRESETS[preset][sim]
    else:
        mod_path, cls_name, kwargs = _PRESETS[preset]
    cfg_cls = getattr(importlib.import_module(mod_path), cls_name)
    cfgs = cfg_cls(sim_type=sim, num_envs=num_envs, **kwargs).build()
    runner = BaseRunner.create_with_env(cfgs)
    return runner.env


def _find_slip_term(env):
    """The feet_slip reward term registered on this env."""
    for name, term in env.reward_manager.reward_terms.items():
        if "feet_slip" in name or "foot_slip" in name:
            return name, term
    raise ValueError(f"No feet_slip term found. Terms: {list(env.reward_manager.reward_terms)}")


def _slip_sources(env, term):
    """(kind, asset_cfg, contact_group, contact_order, command_threshold).

    ``kind`` is "foot" for the shared penalize_feet_slip family (foot
    planar speed squared x contact) and "base" for K1's verbatim
    feet_slip_base_vel (base planar speed x contact count).
    """
    fn = env.reward_manager._resolved_fns[_find_slip_term(env)[0]]
    kind = "base" if "base_vel" in getattr(fn, "__name__", "") else "foot"
    params = term.params
    return (
        kind,
        params.get("asset_cfg"),
        params.get("contact_group", "feet_ground_contact"),
        params.get("contact_order"),
        params.get("command_threshold", 0.05 if kind == "base" else 0.01),
    )


def _make_foot_reader(env, asset_cfg, contact_group):
    """A zero-arg () -> (pos, vel) reader for the term's feet.

    Uses the term's own selector when it is resolved (sites on mujoco,
    bodies on newton/genesis — exactly what the reward reads). Terms
    that rely on the unresolved default selector (the WTW family
    resolves feet internally) fall back to the contact group's tracked
    body names, which name the same feet.
    """
    if asset_cfg is not None and ((asset_cfg.body_ids is not None) != (asset_cfg.site_ids is not None)):
        from rlworld.rl.envs.mdp.rewards.common.reward_terms import _foot_pos_vel as fpv

        return lambda: fpv(env, asset_cfg), "selector"

    tracked = list(env.contact_manager._groups[contact_group].tracked_names)
    rd = env.get_robot_data(env.robot_entity_name)
    ids = torch.tensor([rd.find_body_index(n) for n in tracked], device=env.device, dtype=torch.long)

    def read():
        return rd.body_pos_w_all[:, ids], rd.body_lin_vel_w_all[:, ids]

    return read, f"contact-group bodies {tracked}"


def _contact(env, contact_group, asset_cfg, contact_order):
    from rlworld.rl.envs.mdp.rewards.common.reward_terms import _feet_contact_order

    order = _feet_contact_order(asset_cfg, contact_order) if asset_cfg is not None else contact_order
    return env.contact_manager.is_contact(contact_group, order=order)


def _phase_a(env, sim, preset):
    name, term = _find_slip_term(env)
    fn = env.reward_manager._resolved_fns[name]
    kind, asset_cfg, group, order, cmd_thr = _slip_sources(env, term)
    print(f"\n  [{sim}] A. sources")
    print(f"    term            : {name}  ({fn.__module__}.{fn.__name__})  weight={term.weight}")
    print(f"    formula kind    : {kind}")
    print(f"    contact_group   : {group}  order={order}")
    print(f"    command_thresh  : {cmd_thr}")
    if asset_cfg is not None:
        src = "SITES" if asset_cfg.site_ids is not None else "BODIES"
        names = asset_cfg.site_names if asset_cfg.site_ids is not None else asset_cfg.body_names
        print(f"    velocity source : {src}  {list(names or [])}")
    tracked = env.contact_manager._groups[group].tracked_names
    print(f"    contact tracked : {list(tracked)}")
    print(f"    contact backend : {type(env.contact_manager).__name__}")
    return kind, asset_cfg, group, order, cmd_thr


def _phase_b(env, sim, kind, read_feet, settle_steps, fd_steps):
    """Settled foot geometry + velocity-source integrity (per sim)."""
    zero = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    for _ in range(settle_steps):
        env.step(zero)

    if kind == "base":
        print(f"  [{sim}] B. (base-velocity formula — foot geometry n/a)")
        return None

    pos, _ = read_feet()
    mean_pos = pos.mean(dim=0).cpu().numpy()  # (n_feet, 3)

    # Velocity integrity: the term's instantaneous velocity against the
    # finite difference of the term's own positions over control steps.
    diffs = []
    prev_pos, _ = read_feet()
    prev_pos = prev_pos.clone()
    for _ in range(fd_steps):
        env.step(zero)
        p, v = read_feet()
        v_fd = (p - prev_pos) / env.control_dt
        prev_pos = p.clone()
        num = (v[..., :2] - v_fd[..., :2]).norm(dim=-1)
        den = v_fd[..., :2].norm(dim=-1).clamp_min(1e-4)
        diffs.append((num / den).mean().item())
    print(f"  [{sim}] B. settled foot positions (mean over envs, per foot):")
    for i, p in enumerate(mean_pos):
        print(f"      foot[{i}]  x={p[0]:+.4f}  y={p[1]:+.4f}  z={p[2]:+.4f}")
    print(f"      v_term vs v_fd mean rel dev over {fd_steps} steps: {np.mean(diffs):.3f}")
    return mean_pos


def _phase_cd(env, sim, kind, read_feet, asset_cfg, group, order, cmd_thr, steps, creep_steps):
    """Decomposition at reward time + standing creep fingerprint."""
    from rlworld.rl.envs.mdp.rewards.common.reward_terms import _command_active

    mgr = env.reward_manager
    slip_name, _ = _find_slip_term(env)
    rec = {"raw": [], "contact_frac": [], "v2_contact": [], "gate": []}

    orig = mgr._compute_weighted_reward

    def patched(name, term_cfg):
        value = orig(name, term_cfg)
        if name == slip_name:
            # Same cache generation as the production reward read.
            contact = _contact(env, group, asset_cfg, order).float()
            rec["contact_frac"].append(contact.mean().item())
            rec["gate"].append(_command_active(env, cmd_thr).mean().item())
            if kind == "foot":
                _, v = read_feet()
                v2 = torch.sum(torch.square(v[..., :2]), dim=-1)
                m = contact.sum().clamp_min(1.0)
                rec["v2_contact"].append(((v2 * contact).sum() / m).item())
            else:
                rd = env.get_entity_data(asset_cfg.name if asset_cfg else "robot")
                rec["v2_contact"].append(rd.root_link_lin_vel_w[:, :2].norm(dim=1).mean().item())
            rec["raw"].append((value / (term_cfg.weight * env.control_dt)).mean().item())
        return value

    mgr._compute_weighted_reward = patched
    zero = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    try:
        for _ in range(steps):
            env.step(zero)
    finally:
        mgr._compute_weighted_reward = orig

    # D. standing creep: convention-free finite difference of the foot
    # positions, only over feet in contact, no reward code involved.
    creep = None
    if kind == "foot":
        speeds = []
        prev, _ = read_feet()
        prev = prev.clone()
        for _ in range(creep_steps):
            env.step(zero)
            p, _ = read_feet()
            v_fd = (p - prev)[..., :2].norm(dim=-1) / env.control_dt
            prev = p.clone()
            contact = _contact(env, group, asset_cfg, order).float()
            m = contact.sum().clamp_min(1.0)
            speeds.append(((v_fd * contact).sum() / m).item())
        creep = float(np.mean(speeds))

    summary = {
        "raw": float(np.mean(rec["raw"])),
        "contact_frac": float(np.mean(rec["contact_frac"])),
        "v2_contact": float(np.mean(rec["v2_contact"])),
        "gate": float(np.mean(rec["gate"])),
        "creep": creep,
    }
    print(f"  [{sim}] C. decomposition over {steps} steps (means):")
    print(f"      raw slip value      : {summary['raw']:+.6f}")
    print(f"      P(contact)          : {summary['contact_frac']:.4f}")
    key = "E[v_xy^2 | contact]" if kind == "foot" else "E[base |v_xy|]"
    print(f"      {key:<20}: {summary['v2_contact']:.6f}")
    print(f"      command gate        : {summary['gate']:.4f}")
    if creep is not None:
        print(f"  [{sim}] D. standing creep (FD, contacted feet): {creep:.5f} m/s")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="go2_gait")
    ap.add_argument("--sims", default="genesis,newton,mujoco")
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--settle-steps", type=int, default=50)
    ap.add_argument("--fd-steps", type=int, default=20)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--creep-steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sims = [s.strip() for s in args.sims.split(",")]
    results: dict[str, dict] = {}
    foot_geo: dict[str, np.ndarray] = {}

    for sim in sims:
        print(f"\n{'=' * 72}\nBuilding [{sim}] {args.preset!r} (num_envs={args.num_envs}) ...")
        torch.manual_seed(args.seed)
        env = _build_env(args.preset, sim, args.num_envs)
        env.reset()
        kind, asset_cfg, group, order, cmd_thr = _phase_a(env, sim, args.preset)
        read_feet, src_desc = (None, "n/a") if kind == "base" else _make_foot_reader(env, asset_cfg, group)
        print(f"    diag foot reader: {src_desc}")
        geo = _phase_b(env, sim, kind, read_feet, args.settle_steps, args.fd_steps)
        if geo is not None:
            foot_geo[sim] = geo
        results[sim] = _phase_cd(
            env, sim, kind, read_feet, asset_cfg, group, order, cmd_thr, args.steps, args.creep_steps
        )
        del env
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── verdict ──────────────────────────────────────────────────────
    print(f"\n{'=' * 72}\nVERDICT [{args.preset}]")
    if len(foot_geo) >= 2:
        sims_g = list(foot_geo)
        ref = foot_geo[sims_g[0]]
        print("  measurement-point offsets vs " + sims_g[0] + " (max |Δ| per axis, m):")
        for s in sims_g[1:]:
            d = np.abs(foot_geo[s] - ref).max(axis=0)
            print(f"    {s:<8} dx={d[0]:.4f}  dy={d[1]:.4f}  dz={d[2]:.4f}")

    ref_sim = sims[0]
    print(f"\n  ratios vs {ref_sim}:")
    print(
        f"  {'sim':<8} {'raw slip':>10} {'P(contact)':>11} {'slip speed':>11} {'gate':>7} {'creep':>8} {'creep^2':>9}"
    )
    for s in sims:
        r = results[s]
        ref = results[ref_sim]

        def ratio(a, b):
            return a / b if b not in (0.0, None) and a is not None else float("nan")

        creep_r = ratio(r["creep"], ref["creep"]) if r["creep"] is not None else float("nan")
        print(
            f"  {s:<8} {ratio(r['raw'], ref['raw']):>10.3f} {ratio(r['contact_frac'], ref['contact_frac']):>11.3f} "
            f"{ratio(r['v2_contact'], ref['v2_contact']):>11.3f} {ratio(r['gate'], ref['gate']):>7.3f} "
            f"{creep_r:>8.3f} {creep_r**2:>9.3f}"
        )
    print(
        "\n  Reading: if 'raw slip' ratio ≈ 'creep^2' ratio while P(contact),\n"
        "  gate, and the measurement points match, the divergence is the\n"
        "  engines' friction creep (constraint-solver residual), reproducing\n"
        "  the go2 audit on this preset. If instead P(contact) or the\n"
        "  geometry differs, THAT sim's contact/measurement plumbing is the\n"
        "  cause and the offending input is printed above."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
