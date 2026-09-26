"""Is the motion-prior reset drawing exactly the validated reference states?

The AMP recipe starts every episode from a pool of reference-motion frames
(random planar offset and yaw, the frame's own velocities) that passed a
collision check. This diagnostic pins the pool to the clips and the env's
reset to the pool.

Sections:

* **A  pool** -- built from the preset's spec: size, acceptance rate,
  rejection causes.
* **B  provenance** -- every pool entry is its source frame plus an offset
  inside the configured ranges: joints identical, root offset in range, the
  orientation delta a pure yaw in range, velocities rotated by that yaw.
* **C  re-validation** -- every entry passes a fresh MuJoCo collision /
  foot-height check (no self-collision, ground touched by feet only).
* **D  env reset** (``--sim``) -- after ``env.reset()`` every env's state
  is one pool entry (matched by its joint vector): root pose and velocities
  read back as written, the ``amp`` observation equals the expert feature
  row of the source frame (features are yaw-invariant, so the offset must
  not show), and one zero-action step terminates almost no env.

Run on the GPU box, once per simulator::

    jaxpy -m jaxrlworld.scripts.diag.k1.amp_pose_pool_diag --sim mujoco
    jaxpy -m jaxrlworld.scripts.diag.k1.amp_pose_pool_diag --sim newton
    jaxpy -m jaxrlworld.scripts.diag.k1.amp_pose_pool_diag --sim genesis

Without ``--sim`` sections A-C run on the host alone.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from jaxrlworld.rl.algorithms.amp_ppo.expert_motion import amp_feature_layout, load_expert_motions
from jaxrlworld.rl.configs.common_config_classes import disable_corruption
from jaxrlworld.rl.configs.presets.k1_velocity.base import K1VelocityConfig
from jaxrlworld.rl.envs.mdp.events.motion_pose_pool import (
    _PoseValidator,
    _quat_mul,
    _quat_rotate,
    load_pool_clips,
    mjcf_hinge_joint_names,
    motion_pose_pool,
)
from jaxrlworld.rl.runners.base_runner import BaseRunner

TOL_STATE = 1e-5
TOL_VEL = 1e-4
TOL_FEATURE = 1e-4
MAX_TERMINATED_FRACTION = 0.02


class Checker:
    def __init__(self) -> None:
        self.fails: list[str] = []

    def __call__(self, name: str, ok: bool, detail: str = "") -> bool:
        ok = bool(ok)
        print(f"  [{'OK' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
        if not ok:
            self.fails.append(name)
        return ok


def _quat_conj(q: np.ndarray) -> np.ndarray:
    return q * np.asarray([1.0, -1.0, -1.0, -1.0])


def section_b_provenance(spec, pool, variants, chk: Checker) -> None:
    print("\n=== B. provenance: pool entry == source frame + offset ===")
    M = pool.size
    src_pos = np.stack([variants[v].root_pos[f] for v, f in zip(pool.source_variant, pool.source_frame)]).astype(
        np.float64
    )
    src_quat = np.stack([variants[v].root_quat_wxyz[f] for v, f in zip(pool.source_variant, pool.source_frame)]).astype(
        np.float64
    )
    src_lin = np.stack([variants[v].root_lin_vel_w[f] for v, f in zip(pool.source_variant, pool.source_frame)]).astype(
        np.float64
    )
    src_ang = np.stack([variants[v].root_ang_vel_w[f] for v, f in zip(pool.source_variant, pool.source_frame)]).astype(
        np.float64
    )
    src_q = np.stack([variants[v].joint_pos[f] for v, f in zip(pool.source_variant, pool.source_frame)])
    src_qd = np.stack([variants[v].joint_vel[f] for v, f in zip(pool.source_variant, pool.source_frame)])
    chk("joint positions identical to the source frame", np.array_equal(pool.joint_pos, src_q))
    chk("joint velocities identical to the source frame", np.array_equal(pool.joint_vel, src_qd))
    off = pool.root_pos.astype(np.float64) - src_pos
    chk(
        f"planar offset within {spec.xy_range}",
        off[:, 0].min() >= spec.xy_range[0] - 1e-6
        and off[:, 0].max() <= spec.xy_range[1] + 1e-6
        and off[:, 1].min() >= spec.xy_range[0] - 1e-6
        and off[:, 1].max() <= spec.xy_range[1] + 1e-6,
        f"x [{off[:, 0].min():+.3f}, {off[:, 0].max():+.3f}] y [{off[:, 1].min():+.3f}, {off[:, 1].max():+.3f}]",
    )
    chk(
        f"height offset within {spec.z_range}",
        off[:, 2].min() >= spec.z_range[0] - 1e-6 and off[:, 2].max() <= spec.z_range[1] + 1e-6,
        f"z [{off[:, 2].min():+.4f}, {off[:, 2].max():+.4f}]",
    )
    delta = _quat_mul(pool.root_quat_wxyz.astype(np.float64), _quat_conj(src_quat))  # q_pool = delta * q_src
    chk(
        "orientation delta is a pure yaw (no x/y component)",
        np.abs(delta[:, 1:3]).max() < 1e-5,
        f"max |x,y| {np.abs(delta[:, 1:3]).max():.1e}",
    )
    yaw = 2.0 * np.arctan2(delta[:, 3], delta[:, 0])
    chk(
        f"yaw within {spec.yaw_range}",
        yaw.min() >= spec.yaw_range[0] - 1e-5 and yaw.max() <= spec.yaw_range[1] + 1e-5,
        f"[{yaw.min():+.3f}, {yaw.max():+.3f}]",
    )
    lin_err = np.abs(_quat_rotate(delta, src_lin) - pool.root_lin_vel_w).max()
    ang_err = np.abs(_quat_rotate(delta, src_ang) - pool.root_ang_vel_w).max()
    chk(
        "velocities == source velocities rotated by the yaw",
        max(lin_err, ang_err) < 1e-5,
        f"lin {lin_err:.1e} ang {ang_err:.1e}",
    )
    counts = np.bincount(pool.source_variant, minlength=len(variants))
    print(
        f"    entries per variant: min {counts.min()} max {counts.max()} (mean {counts.mean():.1f}; equal mass per variant)"
    )
    chk("every clip variant is represented", counts.min() > 0)
    print(
        f"    root height of entries: [{pool.root_pos[:, 2].min():.3f}, {pool.root_pos[:, 2].max():.3f}]; |lin vel| max {np.linalg.norm(pool.root_lin_vel_w, axis=1).max():.2f} m/s"
    )


def section_c_revalidate(spec, pool, chk: Checker) -> None:
    print("\n=== C. re-validation of every entry with a fresh MuJoCo model ===")
    validator = _PoseValidator(spec, list(pool.joint_names))
    causes: dict[str, int] = {}
    for i in range(pool.size):
        cause = validator.check(
            pool.root_pos[i].astype(np.float64),
            pool.root_quat_wxyz[i].astype(np.float64),
            pool.joint_pos[i].astype(np.float64),
        )
        if cause is not None:
            causes[cause] = causes.get(cause, 0) + 1
    chk("no entry rejected", not causes, f"{causes}" if causes else f"{pool.size} entries valid")


def section_d_env(sim: str, num_envs: int, cfg: K1VelocityConfig, pool, chk: Checker) -> None:
    print(f"\n=== D. env reset on {sim}: every env starts at a pool entry ===")
    cfgs = cfg.build()
    for name in list(vars(cfgs.event)):
        if name.startswith("dr_") or name == "push":
            delattr(cfgs.event, name)
    disable_corruption(cfgs.observation)
    env = BaseRunner._create_env_from_config(cfgs)
    env.reset()
    device = env.device
    rd = env.get_entity_data("robot")
    joint_names = list(env.act_manager.actuated_joint_names)
    perm = [pool.joint_names.index(n) for n in joint_names]
    pool_q = pool.joint_pos[:, perm]
    # A joint vector is not unique in the pool: a frame can be drawn twice
    # (different offset and yaw), and a speed variant reproduces its source's
    # poses bit for bit where its grid hits a knot (frame 0 always) while its
    # velocities differ. Among the entries sharing the env's joint vector,
    # the one nearest in root position is the one that was written.
    lookup: dict[bytes, list[int]] = {}
    for i, row in enumerate(pool_q):
        lookup.setdefault(row.tobytes(), []).append(i)
    q = rd.joint_pos.detach().cpu().numpy().astype(np.float32)
    origins = env.scene_manager.env_origins.detach().cpu().numpy()
    root_rel = rd.root_link_pos_w.detach().cpu().numpy() - origins
    rows = np.full(num_envs, -1)
    for e, row in enumerate(q):
        candidates = lookup.get(row.tobytes(), [])
        if candidates:
            rows[e] = candidates[int(np.argmin(np.linalg.norm(pool.root_pos[candidates] - root_rel[e], axis=1)))]
    chk(
        "every env's joint vector is a pool entry (exact match)",
        (rows >= 0).all(),
        f"{int((rows >= 0).sum())} / {num_envs} matched; {sum(len(v) > 1 for v in lookup.values())} joint vectors occur more than once in the pool",
    )
    if not (rows >= 0).all():
        return
    pos_err = np.abs(rd.root_link_pos_w.detach().cpu().numpy() - origins - pool.root_pos[rows]).max()
    # q and -q are the same rotation; a backend may hand back the canonical sign.
    quat = rd.root_link_quat_w.detach().cpu().numpy()
    quat_err = np.minimum(
        np.abs(quat - pool.root_quat_wxyz[rows]).max(axis=1), np.abs(quat + pool.root_quat_wxyz[rows]).max(axis=1)
    ).max()
    lin_err = np.abs(rd.root_link_lin_vel_w.detach().cpu().numpy() - pool.root_lin_vel_w[rows]).max()
    ang_err = np.abs(rd.root_link_ang_vel_w.detach().cpu().numpy() - pool.root_ang_vel_w[rows]).max()
    qd_err = np.abs(rd.joint_vel.detach().cpu().numpy() - pool.joint_vel[rows][:, perm]).max()
    chk("root position == entry + env origin", pos_err < TOL_STATE, f"max |Δ| {pos_err:.1e}")
    chk("root orientation == entry (up to quaternion sign)", quat_err < TOL_STATE, f"max |Δ| {quat_err:.1e}")
    chk("root velocities == entry", max(lin_err, ang_err) < TOL_VEL, f"lin {lin_err:.1e} ang {ang_err:.1e}")
    chk("joint velocities == entry", qd_err < TOL_STATE, f"max |Δ| {qd_err:.1e}")

    # The amp observation after reset must be the expert feature row of the
    # source frame: the offset and yaw leave body-frame features unchanged.
    layout = amp_feature_layout(env.obs_manager, "amp")
    expert = load_expert_motions(
        cfg._amp_motion_files(),
        layout,
        joint_names,
        rd.default_joint_pos.detach().cpu().numpy(),
        cfg.robot.base_link_name,
        float(env.control_dt),
        mirror=True,
        speed_factors=cfg._AMP_SPEED_AUGMENTATIONS,
    )
    expert_rows = expert.clip_start[pool.source_variant[rows]] + pool.source_frame[rows]
    amp = env.obs_manager.obs_dict["amp"].detach().cpu().numpy().astype(np.float64)
    want = expert.features[expert_rows].astype(np.float64)
    starts = np.concatenate([[0], np.cumsum([t.width for t in layout])])
    for k, term in enumerate(layout):
        err = np.abs(amp[:, starts[k] : starts[k + 1]] - want[:, starts[k] : starts[k + 1]]).max()
        chk(f"amp[{term.name}] after reset == expert row of the source frame", err < TOL_FEATURE, f"max |Δ| {err:.1e}")
    print(
        f"    source clip variants in this reset: {len(set(pool.source_variant[rows].tolist()))} of {expert.num_clips}"
    )

    actions = torch.zeros((num_envs, env.num_actions), device=device)
    _, _, terminated, truncated, infos = env.step(actions)
    frac = float(terminated.float().mean())
    chk(
        f"one zero-action step from the pool terminates <= {MAX_TERMINATED_FRACTION:.0%} of envs",
        frac <= MAX_TERMINATED_FRACTION,
        f"{frac:.2%} terminated, {float(truncated.float().mean()):.2%} truncated",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", choices=["mujoco", "newton", "genesis"], default=None)
    parser.add_argument("--num_envs", type=int, default=1024)
    args = parser.parse_args()
    chk = Checker()

    cfg = K1VelocityConfig(sim_type=args.sim or "mujoco", num_envs=args.num_envs, use_amp=True)
    spec = cfg._build_event_config().reset_robot_from_motion.params["spec"]
    print("=== A. pool from the preset's spec ===")
    print(f"    {spec}")
    pool = motion_pose_pool(spec)
    drawn = pool.size + sum(pool.rejected.values())
    chk(
        f"pool holds {spec.pool_size} entries",
        pool.size == spec.pool_size,
        f"{pool.size}; {100.0 * pool.size / drawn:.1f}% of {drawn} candidates valid, rejected {pool.rejected}",
    )
    chk("joint columns are the MJCF hinge joints", list(pool.joint_names) == mjcf_hinge_joint_names(spec.mjcf_path))

    variants = load_pool_clips(spec)
    section_b_provenance(spec, pool, variants, chk)
    section_c_revalidate(spec, pool, chk)
    if args.sim is not None:
        section_d_env(args.sim, args.num_envs, cfg, pool, chk)
    print("\n=== RESULT:", "ALL OK" if not chk.fails else f"{len(chk.fails)} FAIL: {chk.fails}", "===")
    return 1 if chk.fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
