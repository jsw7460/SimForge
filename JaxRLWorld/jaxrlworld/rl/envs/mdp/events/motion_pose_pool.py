"""Reset from a pool of reference-motion poses (the motion prior's reset).

A policy trained against a motion prior should start episodes inside the
reference distribution, not at the home pose. This event draws every reset
state from a pool of clip frames that was validated once, on the host, with
MuJoCo: each candidate is a random frame of a random clip variant (the same
mirror / speed variants the discriminator's expert set uses), with a random
planar offset and yaw, and is kept only if the pose has no self-collision,
touches the ground with nothing but its feet, and keeps both foot sites
above the ground. Velocities come with the frame, rotated by the yaw.

The validation runs on the robot's MJCF with a ground plane added, through
plain ``mujoco`` on the host, so the pool is the same on every simulator
backend: the collision model is the MJCF's, not a backend's contact
pipeline (which on Newton and Genesis is only populated by a physics step).

Usage in a preset (replacing the root / joint reset events)::

    "reset_robot_from_motion": EventTermConfig(
        func=reset_from_motion_pose_pool,
        mode="reset",
        params={"spec": MotionPosePoolSpec(...)},
    )

The spec is plain data (it serializes with the config); the pool it names
is built on first use and cached for the process, keyed by the spec.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import numpy as np
import torch

from jaxrlworld.rl.algorithms.amp_ppo.expert_motion import ClipState, clip_variants, load_clip
from jaxrlworld.rl.algorithms.ppo.symmetry import joint_mirror_perm_sign
from jaxrlworld.rl.configs.scene.entity_selector import ResolvedEntity, SceneEntitySelector

if TYPE_CHECKING:
    from jaxrlworld.rl.envs.world import World

_DEFAULT_SELECTOR = SceneEntitySelector(name="robot")


@dataclass(frozen=True)
class MotionPosePoolSpec:
    """Everything that determines a pool; hashable, serializable, the cache key."""

    mjcf_path: str
    motion_files: tuple[str, ...]
    root_body_name: str
    """The base link as the clips' ``body_names`` list it."""
    mirror_augmentation: bool
    speed_augmentations: tuple[float, ...]
    pool_size: int
    seed: int
    xy_range: tuple[float, float] = (-0.5, 0.5)
    """Planar offset added to the frame's root position, per axis."""
    z_range: tuple[float, float] = (0.0, 0.05)
    """Height offset added to the frame's root position."""
    yaw_range: tuple[float, float] = (-3.14, 3.14)
    """Yaw applied on top of the frame's orientation (velocities follow)."""
    foot_site_names: tuple[str, ...] = ("left_foot", "right_foot")
    """Sites that must stay at or above ``min_foot_height``."""
    foot_body_names: tuple[str, ...] = ("left_foot_link", "right_foot_link")
    """The only bodies allowed to touch the ground."""
    min_foot_height: float = 0.0
    max_candidates_per_accepted: int = 8
    """Give up (raise) if more than this many candidates per accepted pose are needed."""


@dataclass(frozen=True)
class MotionPosePool:
    """Validated reset states; root position is relative to the env origin."""

    joint_names: tuple[str, ...]
    """Column order of the joint arrays (the MJCF's hinge order)."""
    root_pos: np.ndarray
    root_quat_wxyz: np.ndarray
    root_lin_vel_w: np.ndarray
    root_ang_vel_w: np.ndarray
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    source_variant: np.ndarray
    """``(M,)`` index into the clip-variant list the pool was drawn from."""
    source_frame: np.ndarray
    """``(M,)`` frame of that variant."""
    rejected: dict[str, int]
    """Candidates dropped per cause during the build."""

    @property
    def size(self) -> int:
        return int(self.root_pos.shape[0])


# ── quaternion helpers (numpy, wxyz) ────────────────────────────────


def _yaw_quat(yaw: np.ndarray) -> np.ndarray:
    return np.stack([np.cos(yaw / 2), np.zeros_like(yaw), np.zeros_like(yaw), np.sin(yaw / 2)], axis=-1)


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """``R(q) v`` for unit quaternions ``q`` (N, 4) and vectors ``v`` (N, 3)."""
    w, xyz = q[:, :1], q[:, 1:]
    c = np.cross(xyz, v)
    return v + 2.0 * w * c + 2.0 * np.cross(xyz, c)


# ── build ───────────────────────────────────────────────────────────


def mjcf_hinge_joint_names(mjcf_path: str) -> list[str]:
    model = mujoco.MjModel.from_xml_path(mjcf_path)
    return [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        for j in range(model.njnt)
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
    ]


def load_pool_clips(spec: MotionPosePoolSpec) -> list[ClipState]:
    """The clip variants the pool draws from, in the expert set's order."""
    joint_names = mjcf_hinge_joint_names(spec.mjcf_path)
    jperm, jsign = joint_mirror_perm_sign(joint_names)
    variants: list[ClipState] = []
    for path in spec.motion_files:
        clip = load_clip(path, joint_names, spec.root_body_name)
        variants.extend(clip_variants(clip, spec.mirror_augmentation, spec.speed_augmentations, jperm, jsign))
    return variants


class _PoseValidator:
    """The robot's MJCF plus a ground plane, forwarded per candidate."""

    def __init__(self, spec: MotionPosePoolSpec, joint_names: list[str]):
        mj_spec = mujoco.MjSpec.from_file(spec.mjcf_path)
        plane = mj_spec.worldbody.add_geom()
        plane.type = mujoco.mjtGeom.mjGEOM_PLANE
        plane.size = [0.0, 0.0, 1.0]
        self.model = mj_spec.compile()
        self.data = mujoco.MjData(self.model)
        self.plane_geom = self.model.ngeom - 1
        free = [j for j in range(self.model.njnt) if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
        if len(free) != 1:
            raise ValueError(f"{spec.mjcf_path}: expected one free joint, found {len(free)}")
        self.free_adr = int(self.model.jnt_qposadr[free[0]])
        self.joint_adr = np.asarray(
            [
                int(self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)])
                for n in joint_names
            ]
        )
        self.foot_bodies = set()
        for name in spec.foot_body_names:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                raise ValueError(f"foot body {name!r} not in {spec.mjcf_path}")
            self.foot_bodies.add(bid)
        self.foot_sites = []
        for name in spec.foot_site_names:
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
            if sid < 0:
                raise ValueError(f"foot site {name!r} not in {spec.mjcf_path}")
            self.foot_sites.append(sid)
        self.min_foot_height = spec.min_foot_height

    def check(self, root_pos: np.ndarray, root_quat: np.ndarray, joint_pos: np.ndarray) -> str | None:
        """``None`` if the pose is valid, else the rejection cause."""
        data = self.data
        data.qpos[:] = 0.0
        data.qpos[self.free_adr : self.free_adr + 3] = root_pos
        data.qpos[self.free_adr + 3 : self.free_adr + 7] = root_quat
        data.qpos[self.joint_adr] = joint_pos
        data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, data)
        self_collision = False
        terrain = False
        for i in range(data.ncon):
            g1, g2 = int(data.contact.geom1[i]), int(data.contact.geom2[i])
            if self.plane_geom in (g1, g2):
                other = g2 if g1 == self.plane_geom else g1
                if int(self.model.geom_bodyid[other]) not in self.foot_bodies:
                    terrain = True
            else:
                self_collision = True
        if self_collision:
            return "self_collision"
        if terrain:
            return "terrain_contact"
        if (data.site_xpos[self.foot_sites, 2] < self.min_foot_height).any():
            return "feet_below_ground"
        return None


def build_motion_pose_pool(spec: MotionPosePoolSpec, verbose: bool = True) -> MotionPosePool:
    """Draw and validate ``spec.pool_size`` reset states (deterministic in ``spec.seed``)."""
    joint_names = mjcf_hinge_joint_names(spec.mjcf_path)
    variants = load_pool_clips(spec)
    validator = _PoseValidator(spec, joint_names)
    rng = np.random.default_rng(spec.seed)
    lengths = np.asarray([v.num_frames for v in variants])
    started = time.time()

    kept: dict[str, list] = {
        k: [] for k in ("root_pos", "root_quat", "lin", "ang", "joint_pos", "joint_vel", "variant", "frame")
    }
    rejected = {"self_collision": 0, "terrain_contact": 0, "feet_below_ground": 0}
    accepted = 0
    drawn = 0
    limit = spec.pool_size * spec.max_candidates_per_accepted
    batch = 1024
    while accepted < spec.pool_size:
        if drawn >= limit:
            raise RuntimeError(
                f"motion pose pool: only {accepted}/{spec.pool_size} valid poses in {drawn} candidates "
                f"(rejected {rejected}); loosen the spec or check the clips"
            )
        n = min(batch, limit - drawn)
        variant = rng.integers(0, len(variants), n)  # equal mass per variant, as the expert set
        frame = (rng.random(n) * lengths[variant]).astype(np.int64)
        offset = np.stack(
            [
                rng.uniform(*spec.xy_range, n),
                rng.uniform(*spec.xy_range, n),
                rng.uniform(*spec.z_range, n),
            ],
            axis=1,
        )
        yaw = rng.uniform(*spec.yaw_range, n)
        q_yaw = _yaw_quat(yaw)
        for i in range(n):
            clip = variants[variant[i]]
            t = int(frame[i])
            root_pos = clip.root_pos[t].astype(np.float64) + offset[i]
            root_quat = _quat_mul(q_yaw[i], clip.root_quat_wxyz[t].astype(np.float64))
            cause = validator.check(root_pos, root_quat, clip.joint_pos[t].astype(np.float64))
            drawn += 1
            if cause is not None:
                rejected[cause] += 1
                continue
            kept["root_pos"].append(root_pos)
            kept["root_quat"].append(root_quat)
            kept["lin"].append(_quat_rotate(q_yaw[i : i + 1], clip.root_lin_vel_w[t : t + 1].astype(np.float64))[0])
            kept["ang"].append(_quat_rotate(q_yaw[i : i + 1], clip.root_ang_vel_w[t : t + 1].astype(np.float64))[0])
            kept["joint_pos"].append(clip.joint_pos[t])
            kept["joint_vel"].append(clip.joint_vel[t])
            kept["variant"].append(int(variant[i]))
            kept["frame"].append(t)
            accepted += 1
            if accepted == spec.pool_size:
                break
    if verbose:
        print(
            f"[motion_pose_pool] {accepted} poses from {drawn} candidates "
            f"({100.0 * accepted / drawn:.1f}% valid; rejected {rejected}) over {len(variants)} clip variants "
            f"in {time.time() - started:.1f} s"
        )
    f32 = lambda k: np.asarray(kept[k], dtype=np.float32)  # noqa: E731 - one-line local helper
    return MotionPosePool(
        joint_names=tuple(joint_names),
        root_pos=f32("root_pos"),
        root_quat_wxyz=f32("root_quat"),
        root_lin_vel_w=f32("lin"),
        root_ang_vel_w=f32("ang"),
        joint_pos=f32("joint_pos"),
        joint_vel=f32("joint_vel"),
        source_variant=np.asarray(kept["variant"], dtype=np.int64),
        source_frame=np.asarray(kept["frame"], dtype=np.int64),
        rejected=rejected,
    )


# ── the event ───────────────────────────────────────────────────────

_POOL_CACHE: dict[MotionPosePoolSpec, MotionPosePool] = {}
_DEVICE_CACHE: dict[tuple[MotionPosePoolSpec, str, tuple[str, ...]], dict[str, torch.Tensor]] = {}


def motion_pose_pool(spec: MotionPosePoolSpec) -> MotionPosePool:
    """The pool for ``spec``, built on first request."""
    pool = _POOL_CACHE.get(spec)
    if pool is None:
        pool = build_motion_pose_pool(spec)
        _POOL_CACHE[spec] = pool
    return pool


def _device_pool(
    spec: MotionPosePoolSpec, joint_names: tuple[str, ...], device: torch.device
) -> dict[str, torch.Tensor]:
    """Pool tensors on ``device`` with joint columns in ``joint_names`` order."""
    key = (spec, str(device), joint_names)
    tensors = _DEVICE_CACHE.get(key)
    if tensors is None:
        pool = motion_pose_pool(spec)
        missing = [n for n in joint_names if n not in pool.joint_names]
        if missing:
            raise ValueError(f"motion pose pool lacks joints {missing}")
        perm = [pool.joint_names.index(n) for n in joint_names]
        tensors = {
            "root_pos": torch.as_tensor(pool.root_pos, device=device),
            "root_quat": torch.as_tensor(pool.root_quat_wxyz, device=device),
            "lin": torch.as_tensor(pool.root_lin_vel_w, device=device),
            "ang": torch.as_tensor(pool.root_ang_vel_w, device=device),
            "joint_pos": torch.as_tensor(pool.joint_pos[:, perm], device=device),
            "joint_vel": torch.as_tensor(pool.joint_vel[:, perm], device=device),
        }
        _DEVICE_CACHE[key] = tensors
    return tensors


def reset_from_motion_pose_pool(
    env: World,
    env_ids: torch.Tensor,
    spec: MotionPosePoolSpec,
    asset_cfg: ResolvedEntity = _DEFAULT_SELECTOR,
) -> None:
    """Reset every env in ``env_ids`` to a random pool state (root pose, velocities, joints)."""
    if len(env_ids) == 0:
        return
    if asset_cfg.name != env.robot_entity_name:
        raise ValueError(
            f"reset_from_motion_pose_pool resets the driven robot {env.robot_entity_name!r}, not {asset_cfg.name!r}"
        )
    device = env.device
    pool = _device_pool(spec, tuple(env.act_manager.actuated_joint_names), device)
    idx = torch.randint(pool["root_pos"].shape[0], (len(env_ids),), device=device)
    writer = env.get_robot_state_writer(asset_cfg.name)
    writer.set_root_pose(
        pool["root_pos"][idx] + env.scene_manager.env_origins[env_ids], pool["root_quat"][idx], env_ids=env_ids
    )
    writer.set_root_velocity(pool["lin"][idx], pool["ang"][idx], env_ids=env_ids)
    writer.set_dof_state(pool["joint_pos"][idx], pool["joint_vel"][idx], env_ids=env_ids)
    writer.eval_fk(env_ids=env_ids)
