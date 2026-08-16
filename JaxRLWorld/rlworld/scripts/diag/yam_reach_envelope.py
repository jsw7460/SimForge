"""Where can this arm actually put its grasp point, over this table?

A lift task needs two ranges: where the cube may start, and where it may
be asked to go. Getting them by arithmetic — taking another scene's
numbers and adding the table height — produces goals the arm cannot
reach, and an unreachable goal is an episode the policy fails no matter
what it does. So they are measured.

Two different questions, answered separately because they have different
answers:

* **Kinematic reach.** Joint angles are written directly and the pose
  read back. This is the set of grasp-point positions the linkage admits,
  ignoring whether the arm could hold them.
* **Holdable reach.** The same poses are commanded through the action
  path and allowed to settle under gravity. This arm tracks with a
  standing error of ~0.2 rad, so the set it can HOLD is smaller than the
  set it can pass through — and a target is only useful if the arm can
  stay there.

Both are reported as an envelope, and then as the answer the task
actually needs: at the cube's own (x, y), how high can the grasp point
go, and how low?

    python -m rlworld.scripts.diag.yam_reach_envelope --num-envs 4096

The sample count is num_envs x rounds; the default gives ~80k poses.
"""

from __future__ import annotations

import argparse

import torch

from rlworld.rl.configs.presets.yam_arm.base import CUBE_HALF, TABLE_TOP_Z, YamArmConfig
from rlworld.rl.runners import BaseRunner

SITE = "grasp_site"

NEAR_XY = 0.05
"""m — how close to the cube's own (x, y) a sample must land to count as
answering "how high can it reach THERE". Wider than the cube, narrow
enough that the height it reports is the height above the cube."""


def _fmt(v) -> str:
    return "[" + ", ".join(f"{float(x):+.4f}" for x in v) + "]"


def _envelope(points: torch.Tensor) -> tuple[list[float], list[float]]:
    return (
        [round(float(v), 4) for v in points.min(dim=0).values],
        [round(float(v), 4) for v in points.max(dim=0).values],
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", default="newton", choices=("genesis", "newton", "mujoco"))
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--settle-steps", type=int, default=60)
    # mjlab's lift ranges, raised by the table height. Checked rather than
    # assumed: the point of this diag is that a shifted range is a guess
    # until some pose is shown to reach it.
    ap.add_argument("--target-x", type=float, nargs=2, default=(0.30, 0.50))
    ap.add_argument("--target-y", type=float, nargs=2, default=(-0.20, 0.20))
    ap.add_argument("--target-z", type=float, nargs=2, default=(0.60, 0.80))
    args = ap.parse_args()

    preset = YamArmConfig(sim_type=args.sim, num_envs=args.num_envs)
    env = BaseRunner._create_env_from_config(preset.build())
    env.reset()

    data = env.get_entity_data("robot")
    writer = env.get_robot_state_writer("robot")
    mid, half = env.act_manager.soft_joint_limits_of("robot")
    joint_names = list(env.entity_indexing("robot").joint_names)
    torch.manual_seed(0)

    # The table's own frame, so the numbers below are stated against the
    # thing the cube sits on rather than against the world origin, which
    # differs per backend (mjlab lays environments out on a grid).
    table = env.get_entity_data("table")
    table_xyz = table.root_link_pos_w[0]
    table_top = float(table_xyz[2]) + TABLE_TOP_Z / 2.0
    cube_xy = (
        torch.tensor(preset.cube_pos[:2], device=env.device)
        + table_xyz[:2]
        - torch.tensor(preset.table_pos[:2], device=env.device)
    )

    print("=" * 78)
    print(f"YAM REACH ENVELOPE  [sim={args.sim}  samples={args.num_envs * args.rounds}]")
    print("=" * 78)
    print(f"  joints        : {joint_names}")
    print(f"  soft limits   : lo={_fmt(mid - half)}")
    print(f"                  hi={_fmt(mid + half)}")
    print(f"  table top     : z = {table_top:.4f}")
    print(f"  cube rests at : z = {table_top + CUBE_HALF:.4f}, xy = {_fmt(cube_xy)}")
    print(f"  arm base      : {_fmt(data.body_pos_w(['arm'])[0, 0])}")

    kinematic: list[torch.Tensor] = []
    holdable: list[torch.Tensor] = []

    for round_idx in range(args.rounds):
        # Uniform over the soft range of every joint. Not a uniform sample
        # of the workspace — the mapping is nonlinear — but it covers the
        # configuration space, which is what bounds the workspace.
        q = mid.unsqueeze(0) + half.unsqueeze(0) * (
            torch.rand(env.num_envs, len(joint_names), device=env.device) * 2 - 1
        )

        # Kinematic: place the joints and look, no dynamics at all.
        writer.set_dof_positions(q)
        writer.set_dof_velocities(torch.zeros_like(q))
        writer.eval_fk()
        env._invalidate_cache()
        kinematic.append(data.site_pos_w([SITE])[:, 0].clone())

        # Holdable: ask for the same pose through the action path and let
        # it settle. A zero action holds home, so the request is the offset
        # from home divided by the term scale — the same inversion the dual
        # arm diag uses to command an angle rather than a raw number.
        env.reset()
        # ``_scale`` is the manager's per-action-dimension gain; on this
        # preset every action dimension is an actuated joint, so the
        # inversion is exact.
        action = (q - env.act_manager.offset) / env.act_manager._scale
        action = action.clamp(-100.0, 100.0)
        for _ in range(args.settle_steps):
            env.step(action)
        env._invalidate_cache()
        holdable.append(data.site_pos_w([SITE])[:, 0].clone())

        if round_idx == 0:
            print(f"\n  round 0 sample: kinematic {_fmt(kinematic[0][0])}  holdable {_fmt(holdable[0][0])}")

    kin = torch.cat(kinematic)
    hold = torch.cat(holdable)

    print()
    print("-" * 78)
    for label, pts in (("kinematic", kin), ("holdable", hold)):
        lo, hi = _envelope(pts)
        print(
            f"  {label:<10} envelope  x {lo[0]:+.4f} .. {hi[0]:+.4f}   y {lo[1]:+.4f} .. {hi[1]:+.4f}   z {lo[2]:+.4f} .. {hi[2]:+.4f}"
        )

    # The question the task actually asks: standing over the cube, how far
    # up and down does the grasp point go? Everything else in the envelope
    # is somewhere the cube is not.
    print()
    print(f"-- over the cube (within {NEAR_XY} m of its xy) --")
    summary: dict[str, tuple[float, float, int]] = {}
    for label, pts in (("kinematic", kin), ("holdable", hold)):
        near = (pts[:, :2] - cube_xy).norm(dim=-1) < NEAR_XY
        above = pts[:, 2] > table_top
        sel = near & above
        count = int(sel.sum())
        if count == 0:
            print(f"  {label:<10}: no sample landed over the cube — widen NEAR_XY or add rounds")
            continue
        z = pts[sel, 2]
        summary[label] = (float(z.min()), float(z.max()), count)
        print(
            f"  {label:<10}: z {float(z.min()):.4f} .. {float(z.max()):.4f}   "
            f"({count} samples, {100 * count / len(pts):.2f}% of the total)"
        )

    print()
    print("-" * 78)
    print("SUGGESTED RANGES for the lift task (holdable, over the cube)")
    if "holdable" in summary:
        lo_z, hi_z, _ = summary["holdable"]
        cube_z = table_top + CUBE_HALF
        # Margins, not the extremes: the boundary of the reachable set is
        # reachable in exactly one configuration, which is not a goal a
        # policy can be asked to hold.
        target_lo = max(cube_z, lo_z + 0.03)
        target_hi = hi_z - 0.05
        print(f"  cube start z : {cube_z:.4f}  (resting on the table)")
        print(f"  target z     : {target_lo:.4f} .. {target_hi:.4f}")
        print(f"  reachable z  : {lo_z:.4f} .. {hi_z:.4f}  (raw, over the cube)")
        if target_hi <= target_lo:
            print("  NO USABLE TARGET RANGE — the arm cannot hold its grasp point")
            print("  meaningfully above the cube here. The table may be too tall,")
            print("  or the cube too far out.")
    # ── Is the proposed target box actually inside the reachable set? ──
    # An envelope is a bounding box of a curved set, so a corner of it can
    # be empty. Ask directly: how many holdable samples land in the box,
    # and how far is the nearest one from each of its corners and centre?
    print()
    print("-" * 78)
    lo = torch.tensor([args.target_x[0], args.target_y[0], args.target_z[0]], device=env.device)
    hi = torch.tensor([args.target_x[1], args.target_y[1], args.target_z[1]], device=env.device)
    lo = lo + torch.cat(
        [table_xyz[:2] - torch.tensor(preset.table_pos[:2], device=env.device), torch.zeros(1, device=env.device)]
    )
    hi = hi + torch.cat(
        [table_xyz[:2] - torch.tensor(preset.table_pos[:2], device=env.device), torch.zeros(1, device=env.device)]
    )
    print(
        f"PROPOSED TARGET BOX  x {lo[0]:+.3f}..{hi[0]:+.3f}  y {lo[1]:+.3f}..{hi[1]:+.3f}  z {lo[2]:+.3f}..{hi[2]:+.3f}"
    )
    inside = ((hold >= lo) & (hold <= hi)).all(dim=-1)
    print(
        f"  holdable samples inside the box: {int(inside.sum())} of {len(hold)} ({100 * float(inside.float().mean()):.2f}%)"
    )

    probes = {
        "centre": 0.5 * (lo + hi),
        "low-near": torch.tensor([lo[0], 0.5 * (lo[1] + hi[1]), lo[2]], device=env.device),
        "high-far": torch.tensor([hi[0], 0.5 * (lo[1] + hi[1]), hi[2]], device=env.device),
        "corner -y": torch.tensor([hi[0], lo[1], hi[2]], device=env.device),
        "corner +y": torch.tensor([hi[0], hi[1], hi[2]], device=env.device),
    }
    worst = 0.0
    for label, point in probes.items():
        d = float((hold - point).norm(dim=-1).min())
        worst = max(worst, d)
        print(f"  nearest holdable sample to {label:<10} {_fmt(point)} : {d:.4f} m")
    # A goal the sampling never came within a few centimetres of is one no
    # sampled configuration reaches. With ~80k poses that is evidence of
    # unreachability rather than of thin sampling.
    print(f"  worst probe distance = {worst:.4f} m")
    print(f"  VERDICT: {'the box looks reachable' if worst < 0.05 else 'PART OF THE BOX IS OUT OF REACH'}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
