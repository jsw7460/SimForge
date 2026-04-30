"""MuJoCo forward-kinematics replayer.

Takes an interpolated motion + MJCF file and produces per-frame per-body
world-frame state (position, wxyz quaternion, linear velocity, angular
velocity) by writing qpos / qvel and calling ``mj_forward`` frame by frame.

Uses only the ``mujoco`` Python package — no mjlab, no Genesis, no Newton.
The resulting arrays are sim-agnostic and can be consumed by MotionCommand
on any of the three JaxRLWorld simulators.

Frame convention note: MuJoCo's free joint qvel is mixed-frame. Empirical
test (mj_objectVelocity flag=0 cross-check):

    qvel[0:3] linear  -> WORLD frame
    qvel[3:6] angular -> LOCAL (body) frame

The official docs are inconsistent on this, but tests confirm the above.
The motion preprocessor stores world-frame angular velocity (axis-angle
of relative quaternion in world frame), so we must rotate it into the
body's local frame before writing to qvel. Without this conversion the
forward-kinematics chain double-rotates the base angular velocity, which
gets amplified by ``omega cross r`` for distant bodies and produces
nonsense per-body velocities (e.g. T1 hand velocity 12 m/s for a clip
where the trunk barely moves).
"""
from __future__ import annotations

import numpy as np

# Module-level import (CLAUDE.md: NEVER in-function imports unless circular).
import mujoco

from rlworld.tools.motion.motion_loader import InterpolatedMotion


def _quat_rotate_inverse_wxyz_np(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate ``v`` by the inverse of unit quaternion ``q`` (wxyz).

    For a unit quaternion ``q``, the inverse equals the conjugate
    ``(w, -x, -y, -z)``. Using Rodrigues-style identity
    ``R(q) v = v + 2 qxyz x (qxyz x v + qw v)``.

    Args:
        q: ``(4,)`` unit quaternion in wxyz layout.
        v: ``(3,)`` vector.

    Returns:
        ``(3,)`` vector ``R(q)^T @ v``.
    """
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    cqxyz = np.array([-x, -y, -z], dtype=np.float64)
    t = 2.0 * np.cross(cqxyz, v.astype(np.float64))
    return (v.astype(np.float64) + w * t + np.cross(cqxyz, t)).astype(v.dtype)


def _free_joint_info(model) -> "tuple[int, int]":
    """Return ``(qpos_adr, qvel_adr)`` for the first free joint, or raise."""
    for jid in range(model.njnt):
        if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
            return int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])
    raise ValueError(
        "MJCF has no free joint; motion tracking requires a floating base."
    )


def _joint_adr_map(
    model, joint_names: "list[str] | None",
) -> "tuple[np.ndarray, np.ndarray, list[str]]":
    """Map a preset's ``joint_names`` to model qpos / qvel indices.

    If ``joint_names`` is ``None``, use every non-free 1-DoF joint in
    XML order. Returns ``(qpos_adr, qvel_adr, resolved_names)``.
    """
    if joint_names is None:
        resolved = []
        qpos_adr = []
        qvel_adr = []
        for jid in range(model.njnt):
            jtype = model.jnt_type[jid]
            if jtype == mujoco.mjtJoint.mjJNT_FREE:
                continue
            if jtype not in (
                mujoco.mjtJoint.mjJNT_HINGE,
                mujoco.mjtJoint.mjJNT_SLIDE,
            ):
                # Ball / other multi-DoF joints are not supported here.
                continue
            resolved.append(str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)))
            qpos_adr.append(int(model.jnt_qposadr[jid]))
            qvel_adr.append(int(model.jnt_dofadr[jid]))
        return np.asarray(qpos_adr, dtype=np.int64), np.asarray(qvel_adr, dtype=np.int64), resolved

    qpos_adr = []
    qvel_adr = []
    for name in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"Joint {name!r} not found in MJCF.")
        qpos_adr.append(int(model.jnt_qposadr[jid]))
        qvel_adr.append(int(model.jnt_dofadr[jid]))
    return np.asarray(qpos_adr, dtype=np.int64), np.asarray(qvel_adr, dtype=np.int64), list(joint_names)


def _list_body_names(model) -> list[str]:
    return [
        str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid))
        for bid in range(model.nbody)
    ]


def replay_motion(
    mjcf_path: str,
    motion: InterpolatedMotion,
    joint_names: "list[str] | None" = None,
    timestep: "float | None" = None,
) -> dict[str, np.ndarray]:
    """Forward-kinematics replay producing per-body world state.

    Args:
        mjcf_path: MuJoCo XML (MJCF) path for the robot. Must include a
            free joint on the root body.
        motion: Interpolated per-frame base + dof state.
        joint_names: Ordered list of actuated joint names to map
            ``motion.dof_pos[:, i]`` / ``motion.dof_vel[:, i]`` into
            ``data.qpos`` / ``data.qvel``. If ``None``, every 1-DoF
            non-free joint in MJCF XML order is used.
        timestep: Override the MJCF's ``opt.timestep``. Defaults to
            ``1 / motion.fps`` so ``mj_forward`` sees a consistent dt.

    Returns:
        Dict with arrays consumable by
        :class:`rlworld.rl.envs.mdp.commands.motion.MotionLoader`:
            - ``joint_pos`` ``(T, J)``
            - ``joint_vel`` ``(T, J)``
            - ``body_pos_w`` ``(T, B, 3)``
            - ``body_quat_w`` ``(T, B, 4)`` wxyz
            - ``body_lin_vel_w`` ``(T, B, 3)``
            - ``body_ang_vel_w`` ``(T, B, 3)``
            - ``body_names`` ``(B,)`` unicode
            - ``joint_names`` ``(J,)`` unicode
            - ``fps`` scalar
    """
    model = mujoco.MjModel.from_xml_path(mjcf_path)
    if timestep is not None:
        model.opt.timestep = float(timestep)
    else:
        model.opt.timestep = 1.0 / float(motion.fps)
    data = mujoco.MjData(model)

    free_qpos_adr, free_qvel_adr = _free_joint_info(model)
    joint_qpos_adr, joint_qvel_adr, resolved_joint_names = _joint_adr_map(
        model, joint_names,
    )
    body_names = _list_body_names(model)

    T = motion.dof_pos.shape[0]
    J = joint_qpos_adr.shape[0]
    B = model.nbody

    if motion.dof_pos.shape[1] != J:
        raise ValueError(
            f"Motion has {motion.dof_pos.shape[1]} DoFs but the joint "
            f"mapping resolved to {J} actuated joints. Pass explicit "
            f"--joint-names or verify the CSV column count."
        )

    out_joint_pos = np.zeros((T, J), dtype=np.float32)
    out_joint_vel = np.zeros((T, J), dtype=np.float32)
    out_body_pos = np.zeros((T, B, 3), dtype=np.float32)
    out_body_quat = np.zeros((T, B, 4), dtype=np.float32)
    out_body_lin_vel = np.zeros((T, B, 3), dtype=np.float32)
    out_body_ang_vel = np.zeros((T, B, 3), dtype=np.float32)

    vel_buf = np.zeros(6, dtype=np.float64)

    for t in range(T):
        # Clear state, write free-joint pose + dof_pos + free-joint vel + dof_vel,
        # then mj_forward to refresh data.xpos / xquat / cvel.
        data.qpos[:] = 0.0
        data.qvel[:] = 0.0
        data.qpos[free_qpos_adr : free_qpos_adr + 3] = motion.base_pos[t]
        data.qpos[free_qpos_adr + 3 : free_qpos_adr + 7] = motion.base_quat_wxyz[t]
        data.qpos[joint_qpos_adr] = motion.dof_pos[t]
        # Free-joint qvel layout (verified empirically):
        #   qvel[0:3] linear  in WORLD frame
        #   qvel[3:6] angular in LOCAL (body) frame
        # The motion stores world-frame angular velocity (from
        # ``_so3_derivative``), so rotate into local before writing.
        data.qvel[free_qvel_adr : free_qvel_adr + 3] = motion.base_lin_vel[t]
        ang_local = _quat_rotate_inverse_wxyz_np(
            motion.base_quat_wxyz[t], motion.base_ang_vel[t],
        )
        data.qvel[free_qvel_adr + 3 : free_qvel_adr + 6] = ang_local
        data.qvel[joint_qvel_adr] = motion.dof_vel[t]

        mujoco.mj_forward(model, data)

        out_joint_pos[t] = motion.dof_pos[t]
        out_joint_vel[t] = motion.dof_vel[t]
        # data.xpos / xquat are (nbody, 3) / (nbody, 4 wxyz) — world frame.
        out_body_pos[t] = data.xpos
        out_body_quat[t] = data.xquat
        # Per-body spatial velocity at link origin in world frame.
        for bid in range(B):
            mujoco.mj_objectVelocity(
                model, data, mujoco.mjtObj.mjOBJ_BODY, bid, vel_buf, 0,
            )
            # mj_objectVelocity output: [ang_x, ang_y, ang_z, lin_x, lin_y, lin_z].
            out_body_ang_vel[t, bid] = vel_buf[:3]
            out_body_lin_vel[t, bid] = vel_buf[3:]

        import warp as wp

    return {
        "joint_pos": out_joint_pos,
        "joint_vel": out_joint_vel,
        "body_pos_w": out_body_pos,
        "body_quat_w": out_body_quat,
        "body_lin_vel_w": out_body_lin_vel,
        "body_ang_vel_w": out_body_ang_vel,
        "body_names": np.asarray(body_names, dtype=np.str_),
        "joint_names": np.asarray(resolved_joint_names, dtype=np.str_),
        "fps": np.asarray(motion.fps, dtype=np.float32),
    }
