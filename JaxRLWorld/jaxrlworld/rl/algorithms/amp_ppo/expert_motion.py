"""Expert-side motion features for adversarial motion priors (AMP).

An AMP discriminator scores one feature vector on both sides: the env's
``amp`` observation group on the policy side, and this module's rendering of
a reference-motion clip on the expert side. The layout is READ from the env's
observation manager rather than restated, so the two sides cannot drift: each
term of the group names the quantity, the joint subset and the width, and the
expert block is derived from the clip by the same definition
(:func:`amp_feature_layout`, :func:`load_expert_motions`).

Clips are the NPZ files the motion-tracking tasks already consume
(``jaxrlworld.tools.motion.csv_to_npz`` / ``mujoco_replayer``): joint state
plus per-body world state at the control rate, read through the tracking
:class:`~jaxrlworld.rl.envs.mdp.commands.motion.MotionLoader`. There is no
AMP-specific clip format. A clip is held here as a :class:`ClipState`: the
root state and the canonical-order joint state, NumPy on the host.

Augmentations act on the clip, before rendering: :func:`mirror_clip` is the
left/right reflection of the whole state (the same reflection the policy-side
mirror operator encodes, see ``ppo.symmetry``), :func:`resample_clip_speed`
replays the clip at another speed. Both are proven against the feature-level
operators by ``scripts/diag/gates/check_amp_expert_motion``.

Velocity convention: the clip's joint and root velocities are the converter's
central finite differences of the resampled poses (the tracking convention),
so a velocity at frame ``t`` is centred on ``t`` like the simulator's
instantaneous read on the policy side. A forward difference would sit half a
step late and hand the discriminator a spurious cue. A speed variant is
re-differentiated the same way.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import NamedTuple, Sequence

import numpy as np
import torch

from jaxrlworld.rl.algorithms.ppo.symmetry import _FIXED_TERM_RULES, _subset_joint_rule, joint_mirror_perm_sign
from jaxrlworld.rl.envs.mdp.commands.motion import MotionLoader
from jaxrlworld.tools.motion.motion_loader import _lerp, _slerp, _so3_derivative

# Observation functions (by ``__name__``) with an expert-side definition.
_JOINT_POS_TERMS = frozenset({"joint_pos_rel", "dof_pos_nominal_difference"})
_JOINT_VEL_TERMS = frozenset({"joint_vel_rel", "dof_vel"})
_ROOT_TERMS = frozenset({"base_lin_vel", "base_ang_vel", "projected_gravity"})

# The left/right reflection of the base state, y = 0 plane (``ppo.symmetry``).
_Y_FLIP = np.asarray([1.0, -1.0, 1.0], dtype=np.float32)
_ROLL_YAW_FLIP = np.asarray([-1.0, 1.0, -1.0], dtype=np.float32)
_QUAT_FLIP = np.asarray([1.0, -1.0, 1.0, -1.0], dtype=np.float32)  # wxyz: R -> M R M


class AmpFeatureTerm(NamedTuple):
    """One column block of the AMP feature vector, in group order."""

    name: str
    """Term name inside the observation group."""
    func_name: str
    """``__name__`` of the term's resolved observation function."""
    width: int
    """Number of columns the term contributes."""
    joint_ids: np.ndarray | None
    """Canonical joint indices for a joint term; ``None`` for a root term."""


def amp_feature_layout(obs_manager, group: str = "amp") -> list[AmpFeatureTerm]:
    """The AMP group's column blocks, each with its expert-side definition.

    Raises on a term with no expert definition, on a history-bearing term
    (the discriminator history is assembled by the algorithm, on both sides,
    so a term-level history would be applied twice), on a term with a scale,
    clip, noise or delay (the expert side renders raw quantities, so any of
    these would make the same motion a different feature for the policy
    than for the expert), and on a width that disagrees with the term's
    joint selection.
    """
    terms = obs_manager._group_terms[group]
    layout: list[AmpFeatureTerm] = []
    for name, func, width in obs_manager.term_layout(group):
        cfg = terms[name]
        if cfg.history_length > 0:
            raise ValueError(
                f"AMP term {name!r} carries history_length={cfg.history_length}; the algorithm stacks the "
                "discriminator history itself, so AMP group terms must be single-frame."
            )
        if cfg.scale != 1.0 or cfg.clip is not None or cfg.noise is not None or cfg.delay_max_lag > 0:
            raise ValueError(
                f"AMP term {name!r} sets scale={cfg.scale}, clip={cfg.clip}, noise={cfg.noise}, "
                f"delay_max_lag={cfg.delay_max_lag}; the expert features are rendered raw, so AMP group "
                "terms must be raw as well."
            )
        func_name = func.__name__
        if func_name in _JOINT_POS_TERMS or func_name in _JOINT_VEL_TERMS:
            joint_ids = np.asarray(cfg.params["asset_cfg"].joint_ids.cpu().numpy(), dtype=np.int64)
            if joint_ids.shape[0] != width:
                raise ValueError(
                    f"AMP term {name!r} ({func_name}) is {width} wide but selects {joint_ids.shape[0]} joints"
                )
        elif func_name in _ROOT_TERMS:
            joint_ids = None
            if width != 3:
                raise ValueError(f"AMP term {name!r} ({func_name}) is {width} wide; a root term is 3")
        else:
            raise ValueError(
                f"AMP term {name!r} resolves to {func_name!r}, which has no expert-side definition; "
                f"supported: {sorted(_JOINT_POS_TERMS | _JOINT_VEL_TERMS | _ROOT_TERMS)}"
            )
        layout.append(AmpFeatureTerm(name, func_name, width, joint_ids))
    if not layout:
        raise ValueError(f"observation group {group!r} has no terms")
    return layout


def feature_dim(layout: Sequence[AmpFeatureTerm]) -> int:
    return sum(term.width for term in layout)


def layout_mirror_operator(
    layout: Sequence[AmpFeatureTerm], joint_names: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    """``(perm, sign)`` mirroring a feature vector of ``layout``.

    Built from the same per-term rules as the policy-side operator
    (``ppo.symmetry``): joint blocks take the L<->R permutation restricted to
    the term's joints, root blocks their fixed rule. ``group_mirror_operator``
    on the live observation manager must give the same arrays; the K1 diag
    checks that they do.
    """
    jperm, jsign = joint_mirror_perm_sign(joint_names)
    perm: list[int] = []
    sign: list[float] = []
    offset = 0
    for term in layout:
        if term.joint_ids is not None:
            lp, ls = _subset_joint_rule(term.joint_ids, jperm, jsign)
        else:
            lp, ls = _FIXED_TERM_RULES[term.func_name]
        perm.extend(offset + k for k in lp)
        sign.extend(ls)
        offset += term.width
    return np.asarray(perm, dtype=np.int64), np.asarray(sign, dtype=np.float32)


def mirror_features(features: np.ndarray, perm: np.ndarray, sign: np.ndarray) -> np.ndarray:
    return features[..., perm] * sign


# ── clips ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ClipState:
    """One reference clip at the control rate: root and joint state per frame.

    Joint columns are in the canonical actuated order; the root velocities
    are world-frame link-origin velocities, the quaternion is wxyz.
    """

    name: str
    fps: float
    joint_pos: np.ndarray
    """``(T, J)`` float32."""
    joint_vel: np.ndarray
    root_pos: np.ndarray
    """``(T, 3)``."""
    root_quat_wxyz: np.ndarray
    """``(T, 4)``."""
    root_lin_vel_w: np.ndarray
    root_ang_vel_w: np.ndarray

    @property
    def num_frames(self) -> int:
        return int(self.joint_pos.shape[0])

    @property
    def dt(self) -> float:
        return 1.0 / self.fps


def load_clip(path: str, joint_names: Sequence[str], root_body_name: str) -> ClipState:
    """Read one NPZ through the tracking loader, joint columns in canonical order."""
    loader = MotionLoader(path, (root_body_name,), joint_names_cfg=tuple(joint_names), device="cpu")
    if loader.fps <= 0.0:
        raise ValueError(f"{path}: NPZ carries no fps")
    arr = lambda t: t.detach().cpu().numpy().astype(np.float32)  # noqa: E731 - one-line local helper
    return ClipState(
        name=path.rsplit("/", 1)[-1].removesuffix(".npz"),
        fps=float(loader.fps),
        joint_pos=arr(loader.joint_pos),
        joint_vel=arr(loader.joint_vel),
        root_pos=arr(loader.body_pos_w[:, 0]),
        root_quat_wxyz=arr(loader.body_quat_w[:, 0]),
        root_lin_vel_w=arr(loader.body_lin_vel_w[:, 0]),
        root_ang_vel_w=arr(loader.body_ang_vel_w[:, 0]),
    )


def mirror_clip(clip: ClipState, joint_perm: np.ndarray, joint_sign: np.ndarray) -> ClipState:
    """The left/right reflection of a clip: the motion the other leg would lead.

    Joints take the L<->R permutation with the roll/yaw sign flip, the root
    reflects across the y = 0 plane (position y, linear velocity y,
    angular velocity roll/yaw and the quaternion's x/z flip), exactly the
    state-space mirror ``ppo.symmetry.mirror_qpos`` / ``mirror_qvel`` apply.
    Involutive.
    """
    return replace(
        clip,
        name=f"{clip.name}+mirror",
        joint_pos=clip.joint_pos[:, joint_perm] * joint_sign,
        joint_vel=clip.joint_vel[:, joint_perm] * joint_sign,
        root_pos=clip.root_pos * _Y_FLIP,
        root_quat_wxyz=clip.root_quat_wxyz * _QUAT_FLIP,
        root_lin_vel_w=clip.root_lin_vel_w * _Y_FLIP,
        root_ang_vel_w=clip.root_ang_vel_w * _ROLL_YAW_FLIP,
    )


def resample_clip_speed(clip: ClipState, factor: float) -> ClipState:
    """The clip played ``factor`` times faster, at the same frame rate.

    The poses are re-read at ``int(T / factor)`` evenly spaced points of the
    original timeline (LERP for positions and joints, SLERP for the root
    quaternion), and the velocities are re-derived from those poses by
    central differences at the clip rate, as the converter does; the stored
    velocities are not scaled, because a scaled velocity would belong to the
    original frames, not the new ones.
    """
    if factor <= 0.0:
        raise ValueError(f"speed factor must be positive, got {factor}")
    T = clip.num_frames
    if T < 2:
        raise ValueError(f"{clip.name}: {T} frames, cannot resample")
    n_out = max(2, int(T / factor))
    dt = clip.dt
    duration = (T - 1) * dt
    times = np.linspace(0.0, duration, n_out, dtype=np.float64)
    phase = times / duration
    idx_0 = np.minimum(np.floor(phase * (T - 1)).astype(np.int64), T - 1)
    idx_1 = np.minimum(idx_0 + 1, T - 1)
    blend = (phase * (T - 1) - idx_0)[:, None].astype(np.float32)

    joint_pos = _lerp(clip.joint_pos[idx_0], clip.joint_pos[idx_1], blend).astype(np.float32)
    root_pos = _lerp(clip.root_pos[idx_0], clip.root_pos[idx_1], blend).astype(np.float32)
    root_quat = _slerp(clip.root_quat_wxyz[idx_0], clip.root_quat_wxyz[idx_1], blend).astype(np.float32)
    return replace(
        clip,
        name=f"{clip.name}+speed{factor:.2f}",
        joint_pos=joint_pos,
        joint_vel=np.gradient(joint_pos, dt, axis=0).astype(np.float32),
        root_pos=root_pos,
        root_quat_wxyz=root_quat,
        root_lin_vel_w=np.gradient(root_pos, dt, axis=0).astype(np.float32),
        root_ang_vel_w=_so3_derivative(root_quat, dt).astype(np.float32),
    )


def clip_variants(
    clip: ClipState,
    mirror: bool,
    speed_factors: Sequence[float],
    joint_perm: np.ndarray,
    joint_sign: np.ndarray,
) -> list[ClipState]:
    """The clip and its augmented copies: ``{identity, mirror} x {identity, speeds...}``.

    The original is always first. A speed factor of exactly 1 is refused
    (it would duplicate the original).
    """
    for f in speed_factors:
        if f == 1.0:
            raise ValueError("speed factor 1.0 duplicates the original clip; leave it out")
    bases = [clip] + ([mirror_clip(clip, joint_perm, joint_sign)] if mirror else [])
    out: list[ClipState] = []
    for base in bases:
        out.append(base)
        out.extend(resample_clip_speed(base, f) for f in speed_factors)
    return out


# ── features ─────────────────────────────────────────────────────────


def _rotate_inverse_wxyz(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """``R(q)^T v`` for unit quaternions ``q`` (N, 4) wxyz and vectors ``v`` (N, 3), in float64."""
    q = q.astype(np.float64)
    v = v.astype(np.float64)
    w, xyz = q[:, :1], q[:, 1:]
    # R^T v = v - 2 w (xyz x v) + 2 xyz x (xyz x v)
    c = np.cross(xyz, v)
    return v - 2.0 * w * c + 2.0 * np.cross(xyz, c)


def clip_features(clip: ClipState, layout: Sequence[AmpFeatureTerm], default_joint_pos: np.ndarray) -> np.ndarray:
    """Render one clip into the ``(T, D)`` feature matrix of ``layout``."""
    default = np.asarray(default_joint_pos, dtype=np.float32)
    if default.shape != (clip.joint_pos.shape[1],):
        raise ValueError(f"default_joint_pos {default.shape} vs {clip.joint_pos.shape[1]} clip joints")
    blocks: list[np.ndarray] = []
    for term in layout:
        if term.func_name in _JOINT_POS_TERMS:
            blocks.append((clip.joint_pos - default[None, :])[:, term.joint_ids])
        elif term.func_name in _JOINT_VEL_TERMS:
            blocks.append(clip.joint_vel[:, term.joint_ids])
        elif term.func_name == "base_lin_vel":
            blocks.append(_rotate_inverse_wxyz(clip.root_quat_wxyz, clip.root_lin_vel_w))
        elif term.func_name == "base_ang_vel":
            blocks.append(_rotate_inverse_wxyz(clip.root_quat_wxyz, clip.root_ang_vel_w))
        elif term.func_name == "projected_gravity":
            gravity = np.tile(np.asarray([[0.0, 0.0, -1.0]]), (clip.num_frames, 1))
            blocks.append(_rotate_inverse_wxyz(clip.root_quat_wxyz, gravity))
        else:
            raise ValueError(f"unhandled AMP term {term.func_name!r}")
    return np.concatenate(blocks, axis=-1).astype(np.float32)


@dataclass(frozen=True)
class ExpertMotionSet:
    """Every reference clip (and augmented variant) rendered into AMP features, frame-major."""

    features: np.ndarray
    """``(T, D)`` float32; clip ``k`` occupies rows ``clip_start[k]:clip_start[k+1]``."""
    clip_start: np.ndarray
    """``(n_clips + 1,)`` int64 row offsets, ``clip_start[-1] == T``."""
    clip_names: tuple[str, ...]
    fps: float
    source_index: np.ndarray
    """``(n_clips,)`` index of the motion file each clip (variant) came from."""

    @property
    def num_frames(self) -> int:
        return int(self.features.shape[0])

    @property
    def num_clips(self) -> int:
        return len(self.clip_names)

    @property
    def clip_lengths(self) -> np.ndarray:
        return np.diff(self.clip_start)

    @property
    def clip_of_frame(self) -> np.ndarray:
        """``(T,)`` clip index of every row."""
        return np.repeat(np.arange(self.num_clips), self.clip_lengths)


def load_expert_motions(
    motion_files: Sequence[str],
    layout: Sequence[AmpFeatureTerm],
    joint_names: Sequence[str],
    default_joint_pos: torch.Tensor | np.ndarray,
    root_body_name: str,
    control_dt: float,
    mirror: bool = False,
    speed_factors: Sequence[float] = (),
) -> ExpertMotionSet:
    """Load every clip, expand its augmentations, and render the AMP features.

    Args:
        motion_files: NPZ clips (tracking format), each at the control rate.
        layout: From :func:`amp_feature_layout` on the training env.
        joint_names: Canonical actuated joint order
            (``env.act_manager.actuated_joint_names``); clip columns are
            permuted onto it by name, and the mirror permutation is derived
            from it.
        default_joint_pos: ``(J,)`` default joint positions in that order.
        root_body_name: The base link's body name as the NPZ lists it.
        control_dt: The env's control step; a clip whose ``fps`` is not its
            inverse is refused rather than resampled here, because the
            velocities baked into it belong to its own rate.
        mirror: Add the left/right reflection of every clip.
        speed_factors: Add a replay of every clip (and of its mirror) at
            each factor.
    """
    if not motion_files:
        raise ValueError("no AMP motion files")
    default = np.asarray(
        default_joint_pos.detach().cpu().numpy() if isinstance(default_joint_pos, torch.Tensor) else default_joint_pos,
        dtype=np.float32,
    )
    if default.shape != (len(joint_names),):
        raise ValueError(f"default_joint_pos {default.shape} vs {len(joint_names)} joints")
    jperm, jsign = joint_mirror_perm_sign(joint_names)
    features: list[np.ndarray] = []
    names: list[str] = []
    source: list[int] = []
    fps_seen: float | None = None
    for file_index, path in enumerate(motion_files):
        clip = load_clip(path, joint_names, root_body_name)
        if abs(clip.fps * control_dt - 1.0) > 1e-6:
            raise ValueError(
                f"{path}: clip rate {clip.fps} Hz is not the control rate {1.0 / control_dt:.6g} Hz; "
                "convert the clip at the control rate"
            )
        if fps_seen is None:
            fps_seen = clip.fps
        elif abs(fps_seen - clip.fps) > 1e-9:
            raise ValueError(f"{path}: fps {clip.fps} differs from the first clip's {fps_seen}")
        for variant in clip_variants(clip, mirror, speed_factors, jperm, jsign):
            feats = clip_features(variant, layout, default)
            if not np.isfinite(feats).all():
                raise ValueError(f"{variant.name}: non-finite AMP feature")
            features.append(feats)
            names.append(variant.name)
            source.append(file_index)
    lengths = np.asarray([f.shape[0] for f in features], dtype=np.int64)
    clip_start = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    return ExpertMotionSet(
        features=np.concatenate(features, axis=0),
        clip_start=clip_start,
        clip_names=tuple(names),
        fps=float(fps_seen),
        source_index=np.asarray(source, dtype=np.int64),
    )


def history_windows(features: np.ndarray, clip_start: np.ndarray, num_steps: int) -> np.ndarray:
    """``(T, K*D)`` discriminator inputs: window ``t`` is ``[f(t-K+1), ..., f(t)]``.

    Frames before a clip's first are replaced by that first frame, which is
    what the policy side does at an episode boundary (its history is
    backfilled with the reset observation), so an expert sample at a clip
    start and a policy sample right after a reset have the same shape of
    repetition and the discriminator cannot tell them apart by it.
    """
    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}")
    T, D = features.shape
    lengths = np.diff(clip_start)
    first = np.repeat(clip_start[:-1], lengths)  # (T,) first row of each row's clip
    idx = np.arange(T)[:, None] + np.arange(1 - num_steps, 1)[None, :]  # (T, K)
    idx = np.maximum(idx, first[:, None])
    return features[idx].reshape(T, num_steps * D)
