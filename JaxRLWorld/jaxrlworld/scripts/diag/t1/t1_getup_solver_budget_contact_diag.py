"""What does the solver budget change about a lying T1's contacts?

The T1 getup policy learns on MuJoCo at the reference's unconverged budget
(iterations 10 / ls 20) about as fast as on Genesis and Newton, and much more
slowly at the converged budget (20 / 50) -- while Newton, converged at 50 / 50,
learns fine. So the converged solve changes something about lying on the
ground that the policy has to work against. This script measures that, on
the same fallen states and the same actions, for several solver budgets of
one backend (one backend per process):

  A. Lying still (zero actions after the settle hold): the ground support
     ratio sum(Fz) / (m g), the tangential-to-normal force ratio, bodies in
     contact, the root's horizontal drift speed and, on MuJoCo Warp, the
     penetration depth of the active contacts and the fraction of substeps
     that exhausted a solver budget.
  B. Under the same random actions: the same quantities, plus how far the
     trajectories drift apart between the first budget and each other one
     (root height and joint-position RMS over the environments).
  D. The reward the task hands the policy on those same states: every term's
     mean per budget in both phases, and the self-contact force the
     self-collision penalty thresholds at 10 N (fraction of environment-steps
     over the threshold, p50 / p90 of the per-step maximum). A term that
     moves with the budget while the states do not is a learning-signal
     difference the contact statistics above cannot show.

``--start standing`` runs the same phases from the home pose instead (every
reset takes the standing branch): the feet's stiff condim-6 contacts are
where a line search that stops early should matter most, and the learning
curves point there too (orientation rises alike at every budget, the
upright-gated posture term is what stalls).

``--checkpoint`` replaces the zero / random actions with a trained policy's
deterministic actions from the first step, so the states are the ones the
policy actually produces (pushing up, balancing); the run then stays inside
one episode, phase A is the first 2 s after the settle hold and phase B the
rest, and section D additionally reports the fraction of environments
upright at the end of each phase.

Run one backend per process; the MuJoCo cells share states and actions
exactly (same seed, same backend RNG), the Newton cell is the reference::

    jaxpy -m jaxrlworld.scripts.diag.t1.t1_getup_solver_budget_contact_diag --sim mujoco --budgets 10/20,20/50,50/50
    jaxpy -m jaxrlworld.scripts.diag.t1.t1_getup_solver_budget_contact_diag --sim mujoco --budgets 10/20,20/50 --start standing
    jaxpy -m jaxrlworld.scripts.diag.t1.t1_getup_solver_budget_contact_diag --sim newton --budgets 50/50,10/20
"""

from __future__ import annotations

import argparse
import math

import mujoco
import torch
import warp as wp

from jaxrlworld.rl.algorithms.ppo import PPO
from jaxrlworld.rl.configs.presets.t1_getup.base import _SIM_TIMINGS, T1GetupConfig
from jaxrlworld.rl.configs.sensors import ContactMatch, ContactSensorCfg
from jaxrlworld.rl.runners import BaseRunner
from jaxrlworld.rl.runners.on_policy_runner import OnPolicyRunner
from jaxrlworld.rl.utils.jax_utils import jax_to_torch
from jaxrlworld.scripts.diag.perf.check_solver_convergence import SOLVER_BITS, _mjw, _override

GROUP = "body_ground"
UPRIGHT_ERR = 0.05
CONTACT_N = 1.0
GRAVITY = 9.81


def _add_ground_sensor(cfgs, sim: str, decimation: int) -> None:
    if sim == "mujoco":
        sensor = ContactSensorCfg(
            name=GROUP,
            primary=ContactMatch(mode="body", pattern=".*", entity="robot"),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force"),
            reduce="netforce",
            history_length=decimation,
            num_slots=1,
        )
        cfgs.scene.sensors = (*cfgs.scene.sensors, sensor)
    else:
        sensor = ContactSensorCfg(
            name=GROUP,
            primary=ContactMatch(mode="body", pattern=".*", entity="robot"),
            secondary=ContactMatch(mode="geom", pattern="ground_plane", entity="terrain"),
            fields=("found", "force"),
            history_length=decimation,
        )
        cfgs.scene.contact_sensors = [*cfgs.scene.contact_sensors, sensor]


def _penetration(data, num_envs: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-world mean and max penetration depth [m] of the active contacts (0 when none)."""
    nacon = int(wp.to_torch(data.nacon)[0].item())
    mean = torch.zeros(num_envs)
    worst = torch.zeros(num_envs)
    if nacon == 0:
        return mean, worst
    dist = wp.to_torch(data.contact.dist)[:nacon].cpu()
    world = wp.to_torch(data.contact.worldid)[:nacon].cpu().long()
    depth = (-dist).clamp(min=0.0)
    count = torch.zeros(num_envs).index_add_(0, world, torch.ones_like(depth))
    total = torch.zeros(num_envs).index_add_(0, world, depth)
    mean = torch.where(count > 0, total / count.clamp(min=1.0), mean)
    worst = worst.index_reduce_(0, world, depth, "amax")
    return mean, worst


def run_cell(
    sim: str,
    budget: str,
    num_envs: int,
    seed: int,
    settle_steps: int,
    lie_steps: int,
    random_steps: int,
    action_std: float,
    standing: bool,
    checkpoint: str | None,
):
    iterations, ls_iterations = (int(x) for x in budget.split("/"))
    cfg = T1GetupConfig(sim_type=sim, num_envs=num_envs)
    if standing:
        cfg.fallen_prob = 0.0
    cfgs = cfg.build()
    _override(cfgs, sim, iterations, ls_iterations)
    _add_ground_sensor(cfgs, sim, cfgs.env.decimation)
    if checkpoint is None:
        runner = BaseRunner.create_with_env(cfgs, use_wandb=False)
    else:
        runner = OnPolicyRunner.load_checkpoint(checkpoint, cfgs=cfgs, use_wandb=False)
        runner.set_eval_mode()
    env = runner.env
    model, data = _mjw(env, sim)
    rd = env.get_entity_data("robot")
    mass = float(mujoco.MjModel.from_xml_path(cfg.robot.mjcf_path).body_subtreemass[1])
    up = torch.tensor([0.0, 0.0, -1.0], device=env.device)
    overflow = wp.to_torch(data.overflow)

    torch.manual_seed(seed)
    obs, _ = env.reset()
    gen = torch.Generator().manual_seed(seed)
    steps = settle_steps + lie_steps + random_steps
    rec = {
        k: []
        for k in (
            "support",
            "tangential",
            "n_contact",
            "slip",
            "lying",
            "pen_mean",
            "pen_max",
            "root_z",
            "joint_pos",
            "overflow",
            "self_force",
        )
    }
    rewards: dict[str, list] = {}
    for t in range(steps):
        if checkpoint is not None:
            act_in = PPO.ActInput(runner._pack_obs(obs, "actor"), runner._pack_obs(obs, "critic"))
            actions = jax_to_torch(
                runner._process_action_for_env(runner.alg.act(act_in, deterministic=True)), runner.device
            )
        elif t < settle_steps + lie_steps:
            actions = torch.zeros((num_envs, env.num_actions), device=env.device)
        else:
            actions = (torch.randn((num_envs, env.num_actions), generator=gen) * action_std).to(env.device)
        overflow.zero_()
        obs, _, _, _, infos = env.step(actions)
        for name, value in infos["rewards_per_type"].items():
            rewards.setdefault(name, []).append(value.detach().cpu().clone())
        self_hist = env.contact_manager.contact_force_history("self_collision")  # (N, bodies, H, 3)
        rec["self_force"].append(self_hist.norm(dim=-1).amax(dim=(1, 2)).detach().cpu())
        bits = overflow.cpu()
        rec["overflow"].append(torch.stack([((bits & int(b)) != 0) for b in SOLVER_BITS.values()], dim=1))
        force = env.contact_manager.contact_force(GROUP).detach().cpu()  # (N, bodies, 3)
        mag = force.norm(dim=-1)
        # mjlab reports the force on the terrain, Newton the force on the body:
        # the vertical component's sign differs, its magnitude is the support.
        normal = force[:, :, 2].sum(dim=1).abs()
        rec["support"].append(normal / (mass * GRAVITY))
        rec["tangential"].append(force[:, :, :2].norm(dim=-1).sum(dim=1) / normal.clamp(min=1e-6))
        rec["n_contact"].append((mag > CONTACT_N).sum(dim=1).float())
        rec["slip"].append(rd.root_link_lin_vel_w[:, :2].norm(dim=-1).detach().cpu())
        rec["lying"].append((torch.sum(torch.square(up - rd.projected_gravity_b), dim=-1) >= UPRIGHT_ERR).cpu())
        pen_mean, pen_max = _penetration(data, num_envs)
        rec["pen_mean"].append(pen_mean)
        rec["pen_max"].append(pen_max)
        rec["root_z"].append(rd.root_link_pos_w[:, 2].detach().cpu())
        rec["joint_pos"].append(rd.joint_pos.detach().cpu())
    out = {k: torch.stack(v, dim=0) for k, v in rec.items()}
    out["rewards"] = {name: torch.stack(v, dim=0) for name, v in rewards.items()}
    out["budget_in_effect"] = (int(model.opt.iterations), int(model.opt.ls_iterations))
    out["mass"] = mass
    return out


def _phase_table(label: str, r: dict, sl: slice, standing: bool) -> None:
    lying = ~r["lying"][sl] if standing else r["lying"][sl]
    n_lying = int(lying.all(dim=0).sum())
    w = lying.float()
    denom = w.sum()

    def wmean(x):
        return float((x[sl] * w).sum() / denom.clamp(min=1.0))

    over = r["overflow"][sl].float().mean(dim=(0, 1))
    print(
        f"    {label:10s} {'upright' if standing else 'lying':7s} {n_lying:5d}  support {wmean(r['support']):.3f}  tang/norm {wmean(r['tangential']):.3f}  "
        f"bodies {wmean(r['n_contact']):.2f}  slip {wmean(r['slip']) * 100:.2f} cm/s  "
        f"pen mean {wmean(r['pen_mean']) * 1000:.2f} mm max {float((r['pen_max'][sl] * w).max()) * 1000:.1f} mm  "
        f"overflow ITER {over[0]:.1%} LS {over[1]:.1%}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sim", choices=("mujoco", "newton"), required=True)
    parser.add_argument("--budgets", default="10/20,20/50", help="comma-separated iterations/ls_iterations cells")
    parser.add_argument("--num_envs", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lie_steps", type=int, default=100)
    parser.add_argument("--random_steps", type=int, default=200)
    parser.add_argument("--action_std", type=float, default=0.3)
    parser.add_argument("--start", choices=("fallen", "standing"), default="fallen")
    parser.add_argument(
        "--checkpoint", default=None, help="drive the robot with this policy instead of zero/random actions"
    )
    args = parser.parse_args()
    standing = args.start == "standing"
    driven = args.checkpoint is not None
    if driven:
        # Stay inside one episode: the step that times out would reset every environment.
        timing = _SIM_TIMINGS[args.sim]
        episode_steps = math.ceil(T1GetupConfig().episode_length_s / (timing["dt"] * timing["decimation"]))
        args.random_steps = min(args.random_steps, episode_steps - 1 - T1GetupConfig().settle_steps - args.lie_steps)
    budgets = args.budgets.split(",")
    settle = T1GetupConfig().settle_steps

    results = {}
    for budget in budgets:
        r = run_cell(
            args.sim,
            budget,
            args.num_envs,
            args.seed,
            settle,
            args.lie_steps,
            args.random_steps,
            args.action_std,
            standing,
            args.checkpoint,
        )
        results[budget] = r
        print(f"built {args.sim} {budget}: budget in effect {r['budget_in_effect']}, robot mass {r['mass']:.2f} kg")

    lie = slice(settle, settle + args.lie_steps)
    rnd = slice(settle + args.lie_steps, settle + args.lie_steps + args.random_steps)
    pose = "policy, first phase" if driven else ("standing still" if standing else "lying still")
    kept = "upright" if standing else "lying"
    print(f"\n=== A. {args.sim}: {pose}, steps {lie.start}..{lie.stop} (envs {kept} the whole phase) ===")
    for budget, r in results.items():
        _phase_table(budget, r, lie, standing)
    phase_b = "policy, second phase" if driven else f"random actions N(0, {args.action_std})"
    print(f"\n=== B. {args.sim}: {phase_b}, steps {rnd.start}..{rnd.stop} ===")
    for budget, r in results.items():
        _phase_table(budget, r, rnd, standing)

    for phase_name, sl in (("A", lie), ("B", rnd)):
        names = [n for n in results[budgets[0]]["rewards"] if n != "total_reward"] + ["total_reward"]
        print(
            f"\n=== D{phase_name}. reward terms on the same states, steps {sl.start}..{sl.stop} (mean per env-step) ==="
        )
        print(
            f"    {'budget':10s} "
            + " ".join(f"{n[:14]:>14s}" for n in names)
            + f" {'self>10N':>9s} {'self p50':>9s} {'self p90':>9s}"
        )
        for budget, r in results.items():
            cells = [f"{float(r['rewards'][n][sl].mean()):14.5f}" for n in names]
            sf = r["self_force"][sl]
            upright_end = float((~r["lying"][sl.stop - 1]).float().mean())
            print(
                f"    {budget:10s} "
                + " ".join(cells)
                + f" {float((sf > 10.0).float().mean()):9.3f} {float(sf.quantile(0.5)):9.2f} {float(sf.quantile(0.9)):9.2f}"
                + f"   upright at end {upright_end:.3f}"
            )

    first = budgets[0]
    r0 = results[first]
    print(f"\n=== C. trajectory drift from {first} under identical actions (RMS over envs) ===")
    marks = [rnd.start + k for k in (25, 50, 100, 150, 200) if rnd.start + k < rnd.stop]
    print(
        f"    {'budget':10s} "
        + " ".join(f"{'step ' + str(m):>18s}" for m in marks)
        + "   (root z [cm] / joint pos [rad])"
    )
    for budget, r in results.items():
        if budget == first:
            continue
        cells = []
        for m in marks:
            dz = (r["root_z"][m] - r0["root_z"][m]).pow(2).mean().sqrt() * 100
            dq = (r["joint_pos"][m] - r0["joint_pos"][m]).pow(2).mean().sqrt()
            cells.append(f"{dz:8.2f} / {dq:6.3f}")
        print(f"    {budget:10s} " + " ".join(f"{c:>18s}" for c in cells))


if __name__ == "__main__":
    main()
