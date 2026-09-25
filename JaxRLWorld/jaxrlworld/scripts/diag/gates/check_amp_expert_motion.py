"""Gate: the AMP expert-motion pipeline, proven without a simulator.

Everything the discriminator's expert side does between an NPZ clip and a
feature row is checked here on a synthetic clip whose ground truth is known
analytically, so no data set and no physics backend is needed:

1. the NumPy inverse rotation matches the torch one the observation terms use;
2. the clip mirror is an involution, and rendering a mirrored clip gives
   exactly the feature-level mirror operator applied to the original's
   features (so the expert augmentation and the policy-side symmetry
   operator are the same reflection);
3. the joint-subset mirror rule is the restriction of the full rule, is the
   full rule on the full subset, and refuses a subset without a mirror image;
4. a speed variant re-reads the poses at the right times, re-derives
   velocities by the converter's contract, and scales velocities by the
   factor;
5. ``clip_variants`` expands ``{identity, mirror} x {identity, speeds}`` with
   the original first and refuses a unit speed;
6. history windows are chronological, backfilled at a clip start and never
   straddle clips;
7. ``load_clip`` permutes NPZ joint columns onto the canonical order by name;
8. ``load_expert_motions`` assembles the set end to end and refuses a clip
   at the wrong rate.

Run::

    python -m jaxrlworld.scripts.diag.gates.check_amp_expert_motion
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch

from jaxrlworld.rl.algorithms.amp_ppo.expert_motion import (
    AmpFeatureTerm,
    ClipState,
    _rotate_inverse_wxyz,
    clip_features,
    clip_variants,
    history_windows,
    layout_mirror_operator,
    load_clip,
    load_expert_motions,
    mirror_clip,
    mirror_features,
    resample_clip_speed,
)
from jaxrlworld.rl.algorithms.ppo.symmetry import _subset_joint_rule, joint_mirror_perm_sign
from jaxrlworld.rl.utils.quat_utils import quat_from_euler_xyz_wxyz, quat_rotate_inverse_wxyz
from jaxrlworld.tools.motion.motion_loader import _so3_derivative

JOINT_NAMES = [
    "Head_Yaw", "Head_Pitch",
    "Left_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw",
    "Right_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw",
    "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw", "Left_Knee_Pitch", "Left_Ankle_Pitch", "Left_Ankle_Roll",
    "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw", "Right_Knee_Pitch", "Right_Ankle_Pitch", "Right_Ankle_Roll",
]  # fmt: skip
LEGS = [i for i, n in enumerate(JOINT_NAMES) if any(k in n for k in ("Hip", "Knee", "Ankle"))]
FPS = 50.0
T = 400


class Checker:
    def __init__(self) -> None:
        self.fails: list[str] = []

    def __call__(self, name: str, ok: bool, detail: str = "") -> bool:
        ok = bool(ok)
        print(f"  [{'OK' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
        if not ok:
            self.fails.append(name)
        return ok


def synthetic_clip(seed: int = 0) -> ClipState:
    """A smooth clip (sinusoids under 1 Hz) with analytic poses and central-difference velocities."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / FPS
    t = np.arange(T) * dt
    J = len(JOINT_NAMES)
    amp = rng.uniform(0.1, 0.6, J)
    # Under 1 Hz: at 50 Hz and up to 1.2x speed, w*dt <= 0.15, so the central
    # difference bias ((w dt)^2 / 6), the linear interpolation of a velocity
    # ((w dt)^2 / 8) and the kink of a linearly interpolated pose under a
    # central difference ((w dt)^2 / 2) together stay under 1.5% of the peak.
    freq = rng.uniform(0.3, 1.0, J)
    phase = rng.uniform(0, 2 * np.pi, J)
    offset = rng.uniform(-0.5, 0.5, J)
    joint_pos = (offset + amp * np.sin(2 * np.pi * freq * t[:, None] + phase)).astype(np.float32)
    root_pos = np.stack(
        [0.8 * t, 0.05 * np.sin(2 * np.pi * 0.9 * t), 0.52 + 0.02 * np.sin(2 * np.pi * 1.8 * t)], 1
    ).astype(np.float32)
    roll = 0.08 * np.sin(2 * np.pi * 0.9 * t)
    pitch = 0.06 * np.sin(2 * np.pi * 1.8 * t + 0.4)
    yaw = 0.3 * np.sin(2 * np.pi * 0.2 * t)
    quat = (
        quat_from_euler_xyz_wxyz(torch.tensor(roll), torch.tensor(pitch), torch.tensor(yaw)).numpy().astype(np.float32)
    )
    return ClipState(
        name="synthetic",
        fps=FPS,
        joint_pos=joint_pos,
        joint_vel=np.gradient(joint_pos, dt, axis=0).astype(np.float32),
        root_pos=root_pos,
        root_quat_wxyz=quat,
        root_lin_vel_w=np.gradient(root_pos, dt, axis=0).astype(np.float32),
        root_ang_vel_w=_so3_derivative(quat, dt).astype(np.float32),
    )


def layout() -> list[AmpFeatureTerm]:
    ids = np.asarray(LEGS, dtype=np.int64)
    return [
        AmpFeatureTerm("joint_pos", "joint_pos_rel", len(ids), ids),
        AmpFeatureTerm("joint_vel", "joint_vel_rel", len(ids), ids),
        AmpFeatureTerm("base_lin_vel", "base_lin_vel", 3, None),
        AmpFeatureTerm("base_ang_vel", "base_ang_vel", 3, None),
        AmpFeatureTerm("projected_gravity", "projected_gravity", 3, None),
    ]


def contract_errors(clip: ClipState) -> dict[str, float]:
    """How far a clip is from the converter's contract (velocities are central differences)."""
    dt = clip.dt
    return {
        "joint_vel": float(np.abs(np.gradient(clip.joint_pos.astype(np.float64), dt, axis=0) - clip.joint_vel).max()),
        "root_lin_vel": float(
            np.abs(np.gradient(clip.root_pos.astype(np.float64), dt, axis=0) - clip.root_lin_vel_w).max()
        ),
        "root_ang_vel": float(
            np.abs(_so3_derivative(clip.root_quat_wxyz.astype(np.float64), dt) - clip.root_ang_vel_w).max()
        ),
        "quat_norm": float(np.abs(np.linalg.norm(clip.root_quat_wxyz, axis=1) - 1.0).max()),
    }


def main() -> int:
    chk = Checker()
    clip = synthetic_clip()
    lay = layout()
    default = np.zeros(len(JOINT_NAMES), dtype=np.float32)
    jperm, jsign = joint_mirror_perm_sign(JOINT_NAMES)

    print("=== 1. inverse rotation: NumPy vs torch ===")
    rng = np.random.default_rng(1)
    q = rng.normal(size=(1000, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    v = rng.normal(size=(1000, 3))
    ours = _rotate_inverse_wxyz(q, v)
    ref = quat_rotate_inverse_wxyz(torch.tensor(q), torch.tensor(v)).numpy()
    chk(
        "R(q)^T v matches quat_rotate_inverse_wxyz",
        np.abs(ours - ref).max() < 1e-9,
        f"max |Δ| {np.abs(ours - ref).max():.1e}",
    )

    print("\n=== 2. clip mirror vs feature-level mirror operator ===")
    perm, sign = layout_mirror_operator(lay, JOINT_NAMES)
    m = mirror_clip(clip, jperm, jsign)
    mm = mirror_clip(m, jperm, jsign)
    chk(
        "mirror_clip is an involution (bitwise)",
        all(
            np.array_equal(getattr(mm, f), getattr(clip, f))
            for f in ("joint_pos", "joint_vel", "root_pos", "root_quat_wxyz", "root_lin_vel_w", "root_ang_vel_w")
        ),
    )
    chk(
        "feature operator is an involution",
        np.array_equal(perm[perm], np.arange(len(perm))) and np.allclose(sign[perm] * sign, 1.0),
    )
    f_orig = clip_features(clip, lay, default)
    f_mir = clip_features(m, lay, default)
    err = np.abs(f_mir - mirror_features(f_orig, perm, sign)).max()
    chk("features(mirror(clip)) == mirror(features(clip))", err < 1e-6, f"max |Δ| {err:.1e} over {f_orig.shape}")
    # A mirrored root reflects across y = 0: heading-frame quantities read the same way.
    chk(
        "mirrored root height and forward speed unchanged",
        np.allclose(m.root_pos[:, 2], clip.root_pos[:, 2])
        and np.allclose(m.root_lin_vel_w[:, 0], clip.root_lin_vel_w[:, 0]),
    )

    print("\n=== 3. joint-subset mirror rule ===")
    sub_perm, sub_sign = _subset_joint_rule(LEGS, jperm, jsign)
    expect_perm = [LEGS.index(int(jperm[j])) for j in LEGS]
    expect_sign = [float(jsign[j]) for j in LEGS]
    chk("subset rule == restriction of the full rule", sub_perm == expect_perm and sub_sign == expect_sign)
    full_perm, full_sign = _subset_joint_rule(list(range(len(JOINT_NAMES))), jperm, jsign)
    chk("subset rule on the full arange == full rule", full_perm == jperm.tolist() and full_sign == jsign.tolist())
    left_only = [i for i in LEGS if "Left" in JOINT_NAMES[i]]
    try:
        _subset_joint_rule(left_only, jperm, jsign)
        chk("left-leg-only subset is refused", False)
    except ValueError as e:
        chk("left-leg-only subset is refused", "not closed" in str(e))

    print("\n=== 4. speed variants ===")
    same = resample_clip_speed(clip, 1.0)
    chk("factor 1: frame count kept", same.num_frames == clip.num_frames)
    chk(
        "factor 1: poses reproduced",
        np.abs(same.joint_pos - clip.joint_pos).max() < 1e-6
        and np.abs(same.root_quat_wxyz - clip.root_quat_wxyz).max() < 1e-6,
    )
    chk(
        "factor 1: velocities reproduced",
        np.abs(same.joint_vel - clip.joint_vel).max() < 1e-4
        and np.abs(same.root_ang_vel_w - clip.root_ang_vel_w).max() < 1e-4,
    )
    dur = (T - 1) / FPS
    t_src = np.arange(T) / FPS
    for f in (1.2, 0.8):
        var = resample_clip_speed(clip, f)
        n = var.num_frames
        chk(f"factor {f}: {n} frames == int(T/f)", n == int(T / f))
        t_new = np.linspace(0.0, dur, n)
        ref_pos = np.stack(
            [np.interp(t_new, t_src, clip.joint_pos[:, j].astype(np.float64)) for j in range(clip.joint_pos.shape[1])],
            1,
        )
        chk(
            f"factor {f}: joints == original read at t_k = k·dur/(n-1)",
            np.abs(var.joint_pos - ref_pos).max() < 1e-5,
            f"max |Δ| {np.abs(var.joint_pos - ref_pos).max():.1e}",
        )
        errs = contract_errors(var)
        chk(
            f"factor {f}: velocities re-derived by the contract",
            max(errs["joint_vel"], errs["root_lin_vel"], errs["root_ang_vel"]) < 1e-4 and errs["quat_norm"] < 1e-6,
            f"{errs}",
        )
        # Physical time in the variant runs f times faster: v'(k) ≈ f · v(t_k).
        # Interior frames only: np.gradient is one-sided (first order) at the
        # two ends, on the variant and on the original alike.
        ref_vel = f * np.stack(
            [np.interp(t_new, t_src, clip.joint_vel[:, j].astype(np.float64)) for j in range(clip.joint_vel.shape[1])],
            1,
        )
        rel = np.abs(var.joint_vel - ref_vel)[1:-1].max() / np.abs(clip.joint_vel).max()
        chk(
            f"factor {f}: joint velocities scale by {f} (interior frames, discretization bound 1.5% of peak)",
            rel < 0.015,
            f"{rel:.2%}",
        )
        ref_lin = f * np.stack(
            [np.interp(t_new, t_src, clip.root_lin_vel_w[:, j].astype(np.float64)) for j in range(3)], 1
        )
        rel = np.abs(var.root_lin_vel_w - ref_lin)[1:-1].max() / np.abs(clip.root_lin_vel_w).max()
        chk(f"factor {f}: root linear velocity scales by {f} (interior frames, bound 1.5%)", rel < 0.015, f"{rel:.2%}")

    print("\n=== 5. variant expansion ===")
    variants = clip_variants(clip, True, (1.1, 0.9), jperm, jsign)
    chk("{identity, mirror} x {identity, 1.1, 0.9} = 6 variants", len(variants) == 6, [v.name for v in variants])
    chk("original comes first, unchanged", variants[0] is clip)
    chk("no augmentation -> the clip alone", clip_variants(clip, False, (), jperm, jsign) == [clip])
    try:
        clip_variants(clip, False, (1.0,), jperm, jsign)
        chk("unit speed factor refused", False)
    except ValueError:
        chk("unit speed factor refused", True)

    print("\n=== 6. history windows ===")
    feats = np.arange(7 * 2, dtype=np.float32).reshape(7, 2)  # two clips: rows 0-3 and 4-6
    win = history_windows(feats, np.asarray([0, 4, 7]), 3)
    chk("shape", win.shape == (7, 6))
    chk("window 2 = [f0, f1, f2]", np.array_equal(win[2], feats[[0, 1, 2]].reshape(-1)))
    chk("window 0 backfilled = [f0, f0, f0]", np.array_equal(win[0], feats[[0, 0, 0]].reshape(-1)))
    chk("window 5 (second clip) = [f4, f4, f5], not [f3, f4, f5]", np.array_equal(win[5], feats[[4, 4, 5]].reshape(-1)))

    print("\n=== 7-8. NPZ round trip through load_clip / load_expert_motions ===")
    with tempfile.TemporaryDirectory() as tmp:
        shuffled = rng.permutation(len(JOINT_NAMES))
        bodies = ["world", "Trunk", "Left_Shank"]
        B = len(bodies)
        zeros = np.zeros((T, B, 3), np.float32)
        quat_all = np.tile(np.asarray([[1, 0, 0, 0]], np.float32), (T, B, 1))
        pos_all, lin_all, ang_all, quat_all = zeros.copy(), zeros.copy(), zeros.copy(), quat_all
        pos_all[:, 1] = clip.root_pos
        lin_all[:, 1] = clip.root_lin_vel_w
        ang_all[:, 1] = clip.root_ang_vel_w
        quat_all[:, 1] = clip.root_quat_wxyz
        path = str(Path(tmp) / "synthetic.npz")
        np.savez(
            path,
            joint_pos=clip.joint_pos[:, shuffled],
            joint_vel=clip.joint_vel[:, shuffled],
            body_pos_w=pos_all,
            body_quat_w=quat_all,
            body_lin_vel_w=lin_all,
            body_ang_vel_w=ang_all,
            body_names=np.asarray(bodies),
            joint_names=np.asarray([JOINT_NAMES[i] for i in shuffled]),
            fps=np.asarray(FPS, np.float32),
        )
        loaded = load_clip(path, JOINT_NAMES, "Trunk")
        chk(
            "load_clip restores the canonical joint order from shuffled columns",
            np.array_equal(loaded.joint_pos, clip.joint_pos) and np.array_equal(loaded.joint_vel, clip.joint_vel),
        )
        chk(
            "load_clip picks the root body by name",
            np.array_equal(loaded.root_quat_wxyz, clip.root_quat_wxyz)
            and np.array_equal(loaded.root_lin_vel_w, clip.root_lin_vel_w),
        )
        expert = load_expert_motions(
            [path], lay, JOINT_NAMES, default, "Trunk", 1.0 / FPS, mirror=True, speed_factors=(1.1, 0.9)
        )
        chk(
            "expert set: 6 variants, frames add up",
            expert.num_clips == 6 and expert.num_frames == sum(v.num_frames for v in variants),
        )
        chk("expert set: first clip == rendering of the original", np.array_equal(expert.features[:T], f_orig))
        try:
            load_expert_motions([path], lay, JOINT_NAMES, default, "Trunk", 0.01)
            chk("clip at the wrong control rate refused", False)
        except ValueError as e:
            chk("clip at the wrong control rate refused", "control rate" in str(e))

    print("\n=== RESULT:", "ALL OK" if not chk.fails else f"{len(chk.fails)} FAIL: {chk.fails}", "===")
    return 1 if chk.fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
