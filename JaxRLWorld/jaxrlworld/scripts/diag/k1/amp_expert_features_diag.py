"""Are the K1 motion-prior expert features exactly what the env's ``amp`` group reports?

The discriminator compares the expert clips against the policy's ``amp``
observation group. Any difference in how a feature is DEFINED on the two
sides (a joint order, a velocity frame, a sign, a default-pose offset) is a
cue the discriminator learns instead of the gait, and nothing in training
would flag it. This diagnostic pins the two sides to each other frame by
frame, for the clips as converted and for their augmented variants.

Sections:

* **A  clips** -- the converted NPZs: rate, finiteness, joint set, root
  height, hard/soft joint-limit violations per joint.
* **B  converter contract** -- the clip velocities are the central finite
  differences of the clip poses at the clip rate (what the loader relies on
  when it treats them as instantaneous velocities).
* **C  env identity** (``--sim``) -- every clip frame is written into the
  simulator through the robot-state writer, and the env's ``amp`` group is
  compared with the expert feature row of that frame, term by term, plus the
  critic's full joint vector against the full expert joint block (that is
  the permutation onto the canonical joint order). Not one frame is skipped.
* **D  history windows** -- the K-frame discriminator input: chronological
  order, clip-start backfill, no window straddling two clips.
* **E  feature statistics** -- per-term raw ranges and per-clip planar speed,
  so the numbers can be read against what a walk or run should show.
* **F  augmentation** -- the mirror operator the env's ``amp`` layout yields
  equals the one the expert layout yields (``--sim``); rendering a mirrored
  clip equals mirroring the rendered clip; the speed variants obey the
  converter contract; and (``--sim``) every frame of every MIRRORED clip,
  written into the simulator, reports the mirrored expert row.

Run on the GPU box, once per simulator::

    jaxpy -m jaxrlworld.scripts.diag.k1.amp_expert_features_diag --sim mujoco
    jaxpy -m jaxrlworld.scripts.diag.k1.amp_expert_features_diag --sim newton
    jaxpy -m jaxrlworld.scripts.diag.k1.amp_expert_features_diag --sim genesis

Without ``--sim`` the data sections (A, B, D, E, F without the env) run
against the MJCF alone.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import mujoco
import numpy as np
import torch

from jaxrlworld.rl.algorithms.amp_ppo.expert_motion import (
    AmpFeatureTerm,
    ClipState,
    ExpertMotionSet,
    amp_feature_layout,
    clip_features,
    clip_variants,
    history_windows,
    layout_mirror_operator,
    load_clip,
    load_expert_motions,
    mirror_clip,
    mirror_features,
)
from jaxrlworld.rl.algorithms.ppo.symmetry import group_mirror_operator, joint_mirror_perm_sign
from jaxrlworld.rl.configs.common_config_classes import disable_corruption
from jaxrlworld.rl.configs.presets.k1_velocity.base import K1VelocityConfig
from jaxrlworld.rl.configs.robots.k1 import K1Config
from jaxrlworld.rl.runners.base_runner import BaseRunner
from jaxrlworld.tools.motion.motion_loader import _so3_derivative

# Tolerances for section C / F. Joint state and the gravity direction are
# pure float32 reads / rotations of what was written; the base linear
# velocity passes through each backend's link-origin transfer.
TOL_EXACT = 1e-5
TOL_LIN_VEL = 1e-4
# Section B / F: velocities recomputed in float64 from float32 poses.
TOL_CENTRAL_DIFF = 1e-4
# Section A: a retargeted pose may sit a hair past the MJCF range; more than
# this is a retargeting defect the simulator would fight every frame.
HARD_LIMIT_SLACK = 1e-3
ROOT_Z_RANGE = (0.3, 0.7)
# The source recipe's expert augmentation: mirror x speed {+-10%, +-20%}.
SPEED_FACTORS = (1.1, 0.9, 1.2, 0.8)


class Checker:
    def __init__(self) -> None:
        self.fails: list[str] = []

    def __call__(self, name: str, ok: bool, detail: str = "") -> bool:
        ok = bool(ok)
        print(f"  [{'OK' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
        if not ok:
            self.fails.append(name)
        return ok


# ── inputs shared by both modes ──────────────────────────────────────


def motion_files(cfg: K1VelocityConfig) -> list[str]:
    files = sorted(str(p) for p in Path(cfg.amp_motion_dir).glob("*.npz"))
    if not files:
        raise SystemExit(f"no NPZ clips under {cfg.amp_motion_dir}; run convert_lafan_k1 first")
    return files


def mjcf_joint_table(robot: K1Config) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Hinge joint names in MJCF order with their hard limits."""
    model = mujoco.MjModel.from_xml_path(robot.mjcf_path)
    names, lo, hi = [], [], []
    for j in range(model.njnt):
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j))
        lo.append(float(model.jnt_range[j, 0]))
        hi.append(float(model.jnt_range[j, 1]))
    return names, np.asarray(lo), np.asarray(hi)


def default_pose_by_name(robot: K1Config, joint_names: list[str]) -> np.ndarray:
    q = np.zeros(len(joint_names), dtype=np.float32)
    for i, name in enumerate(joint_names):
        for pattern, value in robot.default_joint_angles.items():
            if re.fullmatch(pattern, name):
                q[i] = value
    return q


def env_free_layout(cfg: K1VelocityConfig, joint_names: list[str]) -> list[AmpFeatureTerm]:
    """The preset's ``amp`` group as it would resolve, without an env: legs in joint order."""
    ids = np.asarray(
        [i for i, n in enumerate(joint_names) if any(re.fullmatch(p, n) for p in cfg.amp_joint_patterns)],
        dtype=np.int64,
    )
    return [
        AmpFeatureTerm("joint_pos", "joint_pos_rel", len(ids), ids),
        AmpFeatureTerm("joint_vel", "joint_vel_rel", len(ids), ids),
        AmpFeatureTerm("base_lin_vel", "base_lin_vel", 3, None),
        AmpFeatureTerm("projected_gravity", "projected_gravity", 3, None),
    ]


def build_env(sim: str, num_envs: int):
    cfg = K1VelocityConfig(sim_type=sim, num_envs=num_envs, use_amp=True)
    cfgs = cfg.build()
    for name in list(vars(cfgs.event)):
        if name.startswith("dr_") or name == "push":
            delattr(cfgs.event, name)
    disable_corruption(cfgs.observation)
    env = BaseRunner._create_env_from_config(cfgs)
    env.reset()
    return cfg, env


def contract_errors(clip: ClipState) -> dict[str, float]:
    """Distance from the converter's contract: velocities are central differences of the poses."""
    dt = clip.dt
    return {
        "joint_vel": float(
            np.abs(np.gradient(clip.joint_pos.astype(np.float64), dt, axis=0) - clip.joint_vel)[1:-1].max()
        ),
        "root_lin_vel": float(
            np.abs(np.gradient(clip.root_pos.astype(np.float64), dt, axis=0) - clip.root_lin_vel_w)[1:-1].max()
        ),
        "root_ang_vel": float(
            np.abs(_so3_derivative(clip.root_quat_wxyz.astype(np.float64), dt) - clip.root_ang_vel_w)[1:-1].max()
        ),
        "quat_norm": float(np.abs(np.linalg.norm(clip.root_quat_wxyz, axis=1) - 1.0).max()),
    }


# ── sections ─────────────────────────────────────────────────────────


def section_a_clips(
    files: list[str],
    joint_names: list[str],
    hard_lo: np.ndarray,
    hard_hi: np.ndarray,
    soft_factor: float,
    control_dt: float,
    chk: Checker,
) -> None:
    print("\n=== A. clips ===")
    mid, half = 0.5 * (hard_lo + hard_hi), 0.5 * (hard_hi - hard_lo)
    soft_lo, soft_hi = mid - half * soft_factor, mid + half * soft_factor
    total = 0
    worst_hard = 0.0
    hard_frames = 0
    soft_frames = 0
    per_joint_hard = np.zeros(len(joint_names), dtype=np.int64)
    for path in files:
        data = np.load(path, allow_pickle=True)
        fps = float(np.asarray(data["fps"]).item())
        npz_joints = [str(n) for n in data["joint_names"].tolist()]
        q = np.asarray(data["joint_pos"], dtype=np.float64)
        qd = np.asarray(data["joint_vel"], dtype=np.float64)
        bodies = [str(n) for n in data["body_names"].tolist()]
        root = bodies.index("Trunk")
        z = np.asarray(data["body_pos_w"], dtype=np.float64)[:, root, 2]
        quat = np.asarray(data["body_quat_w"], dtype=np.float64)[:, root]
        T = q.shape[0]
        total += T
        name = Path(path).stem
        chk(f"{name}: rate {fps:g} Hz == control rate", abs(fps * control_dt - 1.0) < 1e-6, f"control dt {control_dt}")
        chk(
            f"{name}: joint set == MJCF hinge set",
            set(npz_joints) == set(joint_names) and len(npz_joints) == len(joint_names),
        )
        finite = all(
            np.isfinite(np.asarray(data[k])).all()
            for k in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")
        )
        chk(f"{name}: all arrays finite", finite)
        chk(f"{name}: root quaternion unit", np.abs(np.linalg.norm(quat, axis=1) - 1.0).max() < 1e-5)
        chk(
            f"{name}: root height in {ROOT_Z_RANGE}",
            ROOT_Z_RANGE[0] <= z.min() and z.max() <= ROOT_Z_RANGE[1],
            f"z in [{z.min():.3f}, {z.max():.3f}]",
        )
        perm = [npz_joints.index(n) for n in joint_names]
        qc = q[:, perm]
        over = np.maximum(qc - hard_hi[None, :], hard_lo[None, :] - qc)  # > 0 beyond the hard range
        worst_hard = max(worst_hard, float(over.max()))
        hard_frames += int((over > HARD_LIMIT_SLACK).any(axis=1).sum())
        per_joint_hard += (over > HARD_LIMIT_SLACK).sum(axis=0)
        soft_frames += int(((qc > soft_hi[None, :]) | (qc < soft_lo[None, :])).any(axis=1).sum())
        print(
            f"    {name:20s} T={T:5d} ({T / fps:6.2f} s)  z[{z.min():.3f},{z.max():.3f}]  |qd|max {np.abs(qd).max():5.2f} rad/s"
        )
    print(f"    total {total} frames ({total * control_dt:.1f} s) in {len(files)} clips")
    print(
        f"    frames with a joint beyond its HARD range by > {HARD_LIMIT_SLACK}: {hard_frames} / {total} (worst excess {worst_hard:.4f} rad)"
    )
    for i, n in enumerate(joint_names):
        if per_joint_hard[i]:
            print(f"      {n:22s} {per_joint_hard[i]} frames")
    print(
        f"    frames with a joint beyond its SOFT range (factor {soft_factor}): {soft_frames} / {total} (informational: the sim writes them as is)"
    )
    chk("no frame beyond a hard joint limit", hard_frames == 0, f"{hard_frames} frames")


def section_b_converter(clips: list[ClipState], chk: Checker) -> None:
    print("\n=== B. converter contract: velocities are central differences of the poses ===")
    worst = {k: 0.0 for k in ("joint_vel", "root_lin_vel", "root_ang_vel", "quat_norm")}
    for clip in clips:
        for k, v in contract_errors(clip).items():
            worst[k] = max(worst[k], v)
    chk(
        "joint_vel == central difference of joint_pos",
        worst["joint_vel"] < TOL_CENTRAL_DIFF,
        f"max |Δ| {worst['joint_vel']:.2e} rad/s",
    )
    chk(
        "root lin vel == central difference of root pos",
        worst["root_lin_vel"] < TOL_CENTRAL_DIFF,
        f"max |Δ| {worst['root_lin_vel']:.2e} m/s",
    )
    chk(
        "root ang vel == SO(3) central difference of root quat",
        worst["root_ang_vel"] < TOL_CENTRAL_DIFF,
        f"max |Δ| {worst['root_ang_vel']:.2e} rad/s",
    )


def env_identity(
    env, clips: list[ClipState], layout: list[AmpFeatureTerm], default: np.ndarray, expected: list[np.ndarray]
) -> tuple[dict[str, float], float, int]:
    """Write every frame of every clip into the sim; return the worst |amp - expected| per term.

    ``expected[k]`` is the ``(T_k, D)`` feature matrix clip ``k`` must
    produce. Also returns the worst error of the critic's full joint vector
    against ``clip.joint_pos - default`` and the number of frames checked.
    """
    device = env.device
    num_envs = env.num_envs
    writer = env.get_robot_state_writer("robot")
    origins = env.scene_manager.env_origins
    critic_terms = {name: func.__name__ for name, func, _ in env.obs_manager.term_layout("critic")}
    full_joint_term = next((n for n, f in critic_terms.items() if f == "dof_pos_nominal_difference"), None)
    if full_joint_term is None:
        raise RuntimeError("critic group has no dof_pos_nominal_difference term to check the joint permutation with")
    starts = np.concatenate([[0], np.cumsum([t.width for t in layout])])
    worst = {t.name: 0.0 for t in layout}
    worst_full = 0.0
    frames = 0
    to_dev = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)  # noqa: E731 - one-line local helper
    for clip, want_all in zip(clips, expected, strict=True):
        T = clip.num_frames
        for start in range(0, T, num_envs):
            n = min(num_envs, T - start)
            env_ids = torch.arange(n, device=device)
            sl = slice(start, start + n)
            writer.set_root_pose(
                to_dev(clip.root_pos[sl]) + origins[:n], to_dev(clip.root_quat_wxyz[sl]), env_ids=env_ids
            )
            writer.set_root_velocity(to_dev(clip.root_lin_vel_w[sl]), to_dev(clip.root_ang_vel_w[sl]), env_ids=env_ids)
            writer.set_dof_state(to_dev(clip.joint_pos[sl]), to_dev(clip.joint_vel[sl]), env_ids=env_ids)
            writer.eval_fk(env_ids=env_ids)
            env._post_reset_forward()
            env._invalidate_cache()
            env.obs_manager.process_observations(update_history=False)
            amp = env.obs_manager.obs_dict["amp"][:n].detach().cpu().numpy().astype(np.float64)
            want = want_all[sl].astype(np.float64)
            for k, term in enumerate(layout):
                err = float(np.abs(amp[:, starts[k] : starts[k + 1]] - want[:, starts[k] : starts[k + 1]]).max())
                worst[term.name] = max(worst[term.name], err)
            got = env.obs_manager.extract_term("critic", full_joint_term)[:n].detach().cpu().numpy().astype(np.float64)
            worst_full = max(worst_full, float(np.abs(got - (clip.joint_pos[sl] - default[None, :])).max()))
            frames += n
    return worst, worst_full, frames


def report_identity(
    layout: list[AmpFeatureTerm],
    worst: dict[str, float],
    worst_full: float,
    frames: int,
    total: int,
    chk: Checker,
    tag: str,
) -> None:
    chk(f"{tag}: every clip frame was checked", frames == total, f"{frames} / {total}")
    for term in layout:
        tol = TOL_LIN_VEL if term.func_name == "base_lin_vel" else TOL_EXACT
        chk(
            f"{tag}: amp[{term.name}] == expert[{term.name}]  (width {term.width})",
            worst[term.name] < tol,
            f"max |Δ| {worst[term.name]:.2e} (tol {tol:g})",
        )
    chk(
        f"{tag}: critic full joint_pos == expert full joint block (canonical permutation)",
        worst_full < TOL_EXACT,
        f"max |Δ| {worst_full:.2e}",
    )


def section_c_env_identity(
    env, clips: list[ClipState], layout: list[AmpFeatureTerm], default: np.ndarray, chk: Checker
) -> None:
    print("\n=== C. env identity: every clip frame written to the sim, amp group vs expert row ===")
    expected = [clip_features(c, layout, default) for c in clips]
    worst, worst_full, frames = env_identity(env, clips, layout, default, expected)
    report_identity(layout, worst, worst_full, frames, sum(c.num_frames for c in clips), chk, "original")


def section_d_history(expert: ExpertMotionSet, num_steps: int, chk: Checker) -> None:
    print(f"\n=== D. history windows (K={num_steps}) ===")
    D = expert.features.shape[1]
    win = history_windows(expert.features, expert.clip_start, num_steps)
    chk("window shape (T, K*D)", win.shape == (expert.num_frames, num_steps * D), f"{win.shape}")
    ok_chrono = True
    ok_backfill = True
    for clip in range(expert.num_clips):
        s, e = int(expert.clip_start[clip]), int(expert.clip_start[clip + 1])
        for t in (s, s + 1, s + num_steps - 1, s + num_steps, e - 1):
            if not (s <= t < e):
                continue
            for j in range(num_steps):
                src = max(t - num_steps + 1 + j, s)
                block = win[t, j * D : (j + 1) * D]
                if not np.array_equal(block, expert.features[src]):
                    ok_chrono = False
                if t - num_steps + 1 + j < s and not np.array_equal(block, expert.features[s]):
                    ok_backfill = False
    chk("block j of window t is frame t-K+1+j (chronological, last block = t)", ok_chrono)
    chk("blocks before the clip start repeat the clip's first frame", ok_backfill)
    crossing = 0
    for clip in range(1, expert.num_clips):
        s = int(expert.clip_start[clip])
        prev_last = expert.features[s - 1]
        for t in range(s, min(s + num_steps, expert.num_frames)):
            for j in range(num_steps):
                if np.array_equal(win[t, j * D : (j + 1) * D], prev_last) and not np.array_equal(
                    prev_last, expert.features[s]
                ):
                    crossing += 1
    chk("no window straddles two clips", crossing == 0, f"{crossing} blocks from the previous clip")


def section_e_stats(expert: ExpertMotionSet, layout: list[AmpFeatureTerm]) -> None:
    print("\n=== E. expert feature statistics (raw, per term) ===")
    starts = np.concatenate([[0], np.cumsum([t.width for t in layout])])
    for k, term in enumerate(layout):
        block = expert.features[:, starts[k] : starts[k + 1]]
        print(
            f"    {term.name:18s} w={term.width:2d}  mean {block.mean():+.3f}  std {block.std():.3f}  min {block.min():+.3f}  max {block.max():+.3f}"
        )
    lin = next((k for k, t in enumerate(layout) if t.func_name == "base_lin_vel"), None)
    if lin is not None:
        print("    per-clip base linear velocity (body frame):")
        for clip, name in enumerate(expert.clip_names):
            s, e = int(expert.clip_start[clip]), int(expert.clip_start[clip + 1])
            v = expert.features[s:e, starts[lin] : starts[lin + 1]]
            speed = np.linalg.norm(v[:, :2], axis=1)
            print(
                f"      {name:20s} vx mean {v[:, 0].mean():+.2f}  |v_xy| mean {speed.mean():.2f} max {speed.max():.2f} m/s"
            )


def section_f_augmentation(
    env,
    clips: list[ClipState],
    layout: list[AmpFeatureTerm],
    joint_names: list[str],
    default: np.ndarray,
    files: list[str],
    control_dt: float,
    chk: Checker,
) -> None:
    print("\n=== F. augmentation: mirror and speed variants ===")
    jperm, jsign = joint_mirror_perm_sign(joint_names)
    perm, sign = layout_mirror_operator(layout, joint_names)
    if env is not None:
        env_perm, env_sign = group_mirror_operator(env.obs_manager, "amp", joint_names)
        chk(
            "env amp-group mirror operator == expert layout mirror operator",
            np.array_equal(env_perm, perm) and np.array_equal(env_sign, sign),
        )
    print("    amp mirror perm:", perm.tolist())
    print("    amp mirror sign:", sign.tolist())

    mirrored = [mirror_clip(c, jperm, jsign) for c in clips]
    worst = 0.0
    for c, m in zip(clips, mirrored, strict=True):
        worst = max(
            worst,
            float(
                np.abs(
                    clip_features(m, layout, default) - mirror_features(clip_features(c, layout, default), perm, sign)
                ).max()
            ),
        )
    chk("features(mirror(clip)) == mirror(features(clip)) on every clip", worst < 1e-6, f"max |Δ| {worst:.1e}")
    inv = all(
        np.array_equal(mirror_clip(m, jperm, jsign).joint_pos, c.joint_pos)
        for c, m in zip(clips, mirrored, strict=True)
    )
    chk("mirror is an involution on the clips", inv)

    worst_contract = {k: 0.0 for k in ("joint_vel", "root_lin_vel", "root_ang_vel", "quat_norm")}
    n_variants = 0
    n_frames = 0
    for c in clips:
        variants = clip_variants(c, True, SPEED_FACTORS, jperm, jsign)
        n_variants += len(variants)
        n_frames += sum(v.num_frames for v in variants)
        for v in variants:
            for k, e in contract_errors(v).items():
                worst_contract[k] = max(worst_contract[k], e)
        for f in SPEED_FACTORS:
            v = variants[1 + SPEED_FACTORS.index(f)]
            chk(f"{c.name} x{f}: {v.num_frames} frames == int(T/f)", v.num_frames == int(c.num_frames / f))
    chk(
        f"{len(clips)} clips x (1 + mirror) x (1 + {len(SPEED_FACTORS)} speeds) variants",
        n_variants == len(clips) * 2 * (1 + len(SPEED_FACTORS)),
        f"{n_variants} variants, {n_frames} frames ({n_frames * control_dt:.0f} s)",
    )
    chk(
        "every variant obeys the converter contract",
        max(worst_contract[k] for k in ("joint_vel", "root_lin_vel", "root_ang_vel")) < TOL_CENTRAL_DIFF
        and worst_contract["quat_norm"] < 1e-5,
        f"{worst_contract}",
    )

    augmented = load_expert_motions(
        files, layout, joint_names, default, "Trunk", control_dt, mirror=True, speed_factors=SPEED_FACTORS
    )
    chk(
        "load_expert_motions(mirror, speeds) assembles the same variant set",
        augmented.num_clips == n_variants and augmented.num_frames == n_frames,
    )

    if env is not None:
        print("    writing every frame of every MIRRORED clip into the sim ...")
        expected = [mirror_features(clip_features(c, layout, default), perm, sign) for c in clips]
        worst_t, worst_full, frames = env_identity(env, mirrored, layout, default, expected)
        report_identity(layout, worst_t, worst_full, frames, sum(c.num_frames for c in clips), chk, "mirrored")


# ── main ─────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sim", choices=["mujoco", "newton", "genesis"], default=None, help="omit for the data-only sections"
    )
    parser.add_argument("--num_envs", type=int, default=512)
    parser.add_argument(
        "--history", type=int, default=10, help="discriminator frames per sample (the source's num_amp_obs_steps)"
    )
    args = parser.parse_args()

    robot = K1Config()
    mjcf_joints, hard_lo, hard_hi = mjcf_joint_table(robot)
    chk = Checker()

    if args.sim is None:
        cfg = K1VelocityConfig(use_amp=True)
        control_dt = 0.005 * 4
        joint_names = mjcf_joints
        default = default_pose_by_name(robot, joint_names)
        layout = env_free_layout(cfg, joint_names)
        env = None
        print(f"mode: data only (MJCF joint order); {len(layout[0].joint_ids)} discriminator joints")
    else:
        cfg, env = build_env(args.sim, args.num_envs)
        control_dt = float(env.control_dt)
        joint_names = list(env.act_manager.actuated_joint_names)
        default = env.get_entity_data("robot").default_joint_pos.detach().cpu().numpy().astype(np.float32)
        layout = amp_feature_layout(env.obs_manager, "amp")
        print(
            f"mode: {args.sim}, {args.num_envs} envs; amp group: " + ", ".join(f"{t.name}[{t.width}]" for t in layout)
        )
        print("  discriminator joints: " + ", ".join(joint_names[i] for i in layout[0].joint_ids))
        # The MJCF limits are keyed by name; reorder onto the canonical order.
        order = [mjcf_joints.index(n) for n in joint_names]
        hard_lo, hard_hi = hard_lo[order], hard_hi[order]

    files = motion_files(cfg)
    print(f"clips: {len(files)} under {cfg.amp_motion_dir}")
    clips = [load_clip(f, joint_names, robot.base_link_name) for f in files]
    section_a_clips(files, joint_names, hard_lo, hard_hi, robot.soft_joint_pos_limit_factor, control_dt, chk)
    section_b_converter(clips, chk)
    expert = load_expert_motions(files, layout, joint_names, default, robot.base_link_name, control_dt)
    print(f"\nexpert set: {expert.num_frames} frames x {expert.features.shape[1]} features, {expert.num_clips} clips")
    if env is not None:
        section_c_env_identity(env, clips, layout, default, chk)
    section_d_history(expert, args.history, chk)
    section_e_stats(expert, layout)
    section_f_augmentation(env, clips, layout, joint_names, default, files, control_dt, chk)
    print("\n=== RESULT:", "ALL OK" if not chk.fails else f"{len(chk.fails)} FAIL: {chk.fails}", "===")
    return 1 if chk.fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
