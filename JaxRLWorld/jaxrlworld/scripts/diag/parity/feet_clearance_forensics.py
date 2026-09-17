"""feet_clearance cross-sim forensics: where does mujoco's number diverge?

    clearance = -sum_feet |z - target| * ||v_xy|| * command_gate

Every backend feeds the term its INSTANTANEOUS foot velocity
(``RobotData.body_lin_vel_w``). On mjlab that read is one substep stale:
``step = forward(); integrate()`` computes ``cvel`` before the last
integration and nothing refreshes it (feet_slip_forensics, phase F). At
touchdown the true planar speed collapses within ~solref[0] = 10 ms
while the stale read still shows the swing speed — and the height error
there is the full ``target``, so the term puts its largest weight
exactly where the read is most wrong. During swing the 5 ms lag is a
phase shift on a smooth signal and costs nothing.

One backend per process (cross-sim numbers must never share a
process), same CPU-seeded actions, captured INSIDE the reward manager's
own call (same cache generation as training). Per step and per foot:

    A    the term as written
    FD   the term with the finite difference of its own positions
         (convention-free ground truth, as feet_slip_fd uses)
    F    mujoco only: the term after ``mujoco_warp.forward`` re-derives
         cvel from the post-integration state — the true end-of-step
         velocity newton/genesis read

each split by foot phase from the contact group (swing / touchdown /
stance), plus the mean planar speed per phase for A and F, so the
stale factor at touchdown is a number, not an argument.

    jaxpy -m jaxrlworld.scripts.diag.parity.feet_clearance_forensics --sim mujoco
    jaxpy -m jaxrlworld.scripts.diag.parity.feet_clearance_forensics --sim newton
    jaxpy -m jaxrlworld.scripts.diag.parity.feet_clearance_forensics --sim genesis
"""

from __future__ import annotations

import argparse
import importlib

import mujoco_warp
import torch
import warp as wp

from jaxrlworld.rl.envs.mdp.rewards.common.reward_terms import _command_active, _foot_pos_vel
from jaxrlworld.rl.runners import BaseRunner

_PRESETS = {
    "k1_g1_recipe": "jaxrlworld.rl.configs.presets.k1_joystick.g1_recipe:K1G1RecipeConfig",
    "k1_joystick": "jaxrlworld.rl.configs.presets.k1_joystick.base:K1JoystickConfig",
    "g1_flat": "jaxrlworld.rl.configs.presets.g1_29dof.base:G1FlatConfig",
}
_PHASES = ("swing", "touchdown", "stance")


def _build_env(preset: str, sim: str, num_envs: int):
    spec = _PRESETS.get(preset, preset)
    mod_path, cls_name = spec.split(":", 1)
    cfg_cls = getattr(importlib.import_module(mod_path), cls_name)
    cfgs = cfg_cls(sim_type=sim, num_envs=num_envs).build()
    return BaseRunner.create_with_env(cfgs, use_wandb=False).env


def _find_term(env):
    for name, term in env.reward_manager.reward_terms.items():
        if "feet_clearance" in name:
            return name, term
    raise ValueError(f"No feet_clearance term. Terms: {list(env.reward_manager.reward_terms)}")


class _Fresh:
    """mujoco: re-derive cvel/xpos from the post-integration qpos/qvel.

    qpos/qvel are untouched, so the trajectory is preserved; only the
    derived fields and the solver warm-start are perturbed, acceptable
    in an open-loop measurement run.
    """

    def __init__(self, env):
        self._sim = env.scene_manager.sim

    def forward(self) -> None:
        with wp.ScopedDevice(self._sim.wp_device):
            mujoco_warp.forward(self._sim.wp_model, self._sim.wp_data)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="k1_g1_recipe", help="shorthand or module.path:ClassName")
    ap.add_argument("--sim", required=True, choices=("genesis", "newton", "mujoco"))
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=20, help="steps before capture starts")
    ap.add_argument("--actions", default="random", choices=("random", "zero"))
    ap.add_argument("--action-scale", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    env = _build_env(args.preset, args.sim, args.num_envs)
    mgr = env.reward_manager
    # The capture hangs on the per-term seam, which the compiled reward
    # chain bypasses; the terms are the same either way.
    mgr._compile_terms = False
    name, term = _find_term(env)
    fn = mgr._resolved_fns[name]
    asset_cfg = term.params["asset_cfg"]
    target = term.params["target_height"]
    threshold = term.params.get("command_threshold", 0.01)
    contact_order = list(asset_cfg.body_names) if asset_cfg.body_names is not None else None
    fresh = _Fresh(env) if args.sim == "mujoco" else None
    dt = env.control_dt
    dev = env.device

    print(f"[{args.sim}] term {name}: weight {term.weight}, target {target}, threshold {threshold}")
    print(f"  feet: {asset_cfg.body_names or asset_cfg.site_names}, contact group feet_ground_contact")

    # Accumulators: per source x per phase, summed over envs x feet x steps.
    sources = ("A", "FD") + (("F",) if fresh is not None else ())
    cost = {s: dict.fromkeys(_PHASES, 0.0) for s in sources}
    speed = {s: dict.fromkeys(_PHASES, 0.0) for s in sources}
    count = dict.fromkeys(_PHASES, 0)
    total = {s: 0.0 for s in sources}
    stats = {"steps": 0, "term_mismatch": 0.0, "fresh_delta": 0.0, "gate": 0.0}
    prev_pos = None
    step_i = 0
    original = mgr._compute_weighted_reward

    def captured(term_name, term_cfg):
        nonlocal prev_pos
        if term_name == name and step_i >= args.warmup:
            pos, vel = _foot_pos_vel(env, asset_cfg)
            gate = _command_active(env, threshold)
            delta = (pos[..., 2] - target).abs()
            contact = env.contact_manager.is_contact("feet_ground_contact", order=contact_order)
            prev_contact = env.contact_manager.prev_is_contact("feet_ground_contact", order=contact_order)
            phase = {
                "swing": ~contact,
                "touchdown": contact & ~prev_contact,
                "stance": contact & prev_contact,
            }
            speeds = {"A": vel[..., :2].norm(dim=-1)}
            if prev_pos is not None:
                fd = (pos - prev_pos) / dt
                fd = torch.where((env.episode_length_buf <= 1).view(-1, 1, 1), torch.zeros_like(fd), fd)
                speeds["FD"] = fd[..., :2].norm(dim=-1)
            if fresh is not None:
                fresh.forward()
                _, vel_f = _foot_pos_vel(env, asset_cfg)
                stats["fresh_delta"] += float((vel_f - vel).abs().mean())
                speeds["F"] = vel_f[..., :2].norm(dim=-1)
            # Sanity: the decomposition IS the term (raw, unweighted).
            own = fn(env, **term_cfg.params)
            mine = -(delta * speeds["A"]).sum(dim=1) * gate
            stats["term_mismatch"] = max(stats["term_mismatch"], float((own - mine).abs().max()))
            stats["gate"] += float(gate.mean())
            stats["steps"] += 1
            g = gate.view(-1, 1)
            for s, v in speeds.items():
                contrib = delta * v * g
                total[s] += float(contrib.sum())
                for p, mask in phase.items():
                    m = mask.float()
                    cost[s][p] += float((contrib * m).sum())
                    speed[s][p] += float((v * m).sum())
            for p, mask in phase.items():
                count[p] += int(mask.sum())
            prev_pos = pos.clone()
        elif term_name == name:
            prev_pos = _foot_pos_vel(env, asset_cfg)[0].clone()
        return original(term_name, term_cfg)

    mgr._compute_weighted_reward = captured
    gen = torch.Generator().manual_seed(args.seed)
    for step_i in range(args.steps):
        if args.actions == "random":
            actions = (torch.randn((env.num_envs, env.num_actions), generator=gen) * args.action_scale).to(dev)
        else:
            actions = torch.zeros(env.num_envs, env.num_actions, device=dev)
        env.step(actions)
    mgr._compute_weighted_reward = original

    n = stats["steps"] * env.num_envs
    print()
    print("=" * 78)
    print(
        f"FEET CLEARANCE  [{args.sim}]  {stats['steps']} captured steps x {env.num_envs} envs, actions={args.actions}"
    )
    print("=" * 78)
    print(f"  decomposition vs the term itself: max |diff| {stats['term_mismatch']:.2e} (must be ~0)")
    print(f"  command gate active: {stats['gate'] / stats['steps']:.2f} of envs")
    if fresh is not None:
        print(
            f"  fresh re-read changed the velocity by {stats['fresh_delta'] / stats['steps']:.3e} m/s mean (0 = cache blocked it)"
        )
    print()
    print(f"  {'source':<8}{'mean cost / env-step':>22}" + "".join(f"{p:>14}" for p in _PHASES))
    for s in sources:
        row = f"  {s:<8}{-total[s] / n:>22.5f}"
        row += "".join(f"{-cost[s][p] / n:>14.5f}" for p in _PHASES)
        print(row)
    print()
    print(
        f"  {'phase share':<8}{'':>22}" + "".join(f"{count[p] / max(sum(count.values()), 1):>14.3f}" for p in _PHASES)
    )
    print(f"  {'mean planar speed in phase (m/s)':<30}")
    for s in sources:
        print(f"    {s:<6}" + "".join(f"{speed[s][p] / max(count[p], 1):>14.4f}" for p in _PHASES))
    print("=" * 78)
    print("  A = term as written (instantaneous read); FD = finite-difference velocity;")
    print("  F = mujoco after mujoco_warp.forward (true end-of-step). Compare F/A at touchdown.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
