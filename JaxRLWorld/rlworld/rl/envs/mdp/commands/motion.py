"""MotionCommand — reference motion tracking for humanoid locomotion.

A sim-agnostic port of Mjlab's ``tasks/tracking/mdp/commands.py``. Reads
an NPZ motion clip and exposes time-indexed reference body / joint state
plus anchor-frame-relative body poses that the tracking reward /
observation / termination terms consume.

Reset behaviour is the standout feature: when the base-class
``CommandTerm.reset`` path fires (at env init and on every episode
reset), this term writes a reference state into the sim via the
:class:`RobotStateWriterProtocol`, so motion tracking presets should
not register an additional ``reset_fallen_or_standing`` event — motion
is the single source of initial state.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch

from rlworld.rl.envs.managers.common.command_term import CommandTerm, CommandTermCfg
from rlworld.rl.utils.quat_utils import (
    quat_from_euler_xyz_wxyz,
    quat_inv_wxyz,
    quat_mul_wxyz,
    quat_error_magnitude_wxyz,
    quat_rotate_wxyz,
    yaw_quat_wxyz,
)

if TYPE_CHECKING:
    from rlworld.rl.envs import World


def _sample_uniform(lo, hi, size, device) -> torch.Tensor:
    """Drop-in for mjlab's ``sample_uniform``. Accepts scalar / tensor bounds."""
    if isinstance(size, int):
        size = (size,)
    return torch.rand(*size, device=device) * (hi - lo) + lo


class MotionLoader:
    """Load an NPZ motion clip and subset its bodies to ``body_names_cfg``.

    The NPZ is expected to contain:
        - ``joint_pos``      ``[T, J]``
        - ``joint_vel``      ``[T, J]``
        - ``body_pos_w``     ``[T, B, 3]``
        - ``body_quat_w``    ``[T, B, 4]`` (wxyz)
        - ``body_lin_vel_w`` ``[T, B, 3]``
        - ``body_ang_vel_w`` ``[T, B, 3]``
        - ``body_names``     ``[B]`` string array — the replayer's body order
        - ``fps``            scalar (optional)

    ``body_names_cfg`` lists the subset of bodies (bare names, no sim
    prefix) that the preset actually tracks, in the order expected by
    downstream reward / observation code. Bodies are reordered to match
    this list; ``MotionLoader.body_pos_w[t, i]`` is the world position of
    ``body_names_cfg[i]`` at frame ``t``.
    """

    def __init__(
        self,
        motion_file: str,
        body_names_cfg: tuple[str, ...],
        joint_names_cfg: "tuple[str, ...] | None" = None,
        device: str | torch.device = "cpu",
    ) -> None:
        data = np.load(motion_file, allow_pickle=True)
        if "body_names" not in data.files:
            raise ValueError(
                f"Motion file {motion_file!r} lacks a 'body_names' array. "
                "Regenerate via rlworld.tools.motion.csv_to_npz."
            )
        npz_body_names = [str(n) for n in np.asarray(data["body_names"]).tolist()]

        try:
            motion_idx = [npz_body_names.index(n) for n in body_names_cfg]
        except ValueError as e:
            missing = [n for n in body_names_cfg if n not in npz_body_names]
            raise ValueError(
                f"Motion file {motion_file!r} is missing body names "
                f"{missing}. NPZ contains {npz_body_names}."
            ) from e

        joint_pos_raw = torch.tensor(
            data["joint_pos"], dtype=torch.float32, device=device,
        )
        joint_vel_raw = torch.tensor(
            data["joint_vel"], dtype=torch.float32, device=device,
        )

        # Permute joint columns to the canonical order
        # (``env.act_manager.actuated_joint_names``) expected by the
        # RobotStateWriter. The NPZ stores joint_pos / joint_vel in
        # MJCF XML joint order (the preprocessor's replayer), which
        # does NOT in general match Newton / Genesis / Mjlab's
        # actuator order even though all three load the same robot.
        # Without this permutation, dof values land on the wrong
        # joints on at least one simulator.
        if joint_names_cfg is not None:
            if "joint_names" not in data.files:
                raise ValueError(
                    f"Motion file {motion_file!r} lacks a 'joint_names' array but "
                    "the caller requested canonical-order permutation. Regenerate "
                    "the NPZ via rlworld.tools.motion.csv_to_npz."
                )
            npz_joint_names = [
                str(n) for n in np.asarray(data["joint_names"]).tolist()
            ]
            missing = [n for n in joint_names_cfg if n not in npz_joint_names]
            if missing:
                raise ValueError(
                    f"Motion file {motion_file!r} is missing joints {missing}. "
                    f"NPZ contains {npz_joint_names}."
                )
            joint_perm = torch.tensor(
                [npz_joint_names.index(n) for n in joint_names_cfg],
                dtype=torch.long,
                device=device,
            )
            self.joint_pos = joint_pos_raw[:, joint_perm]
            self.joint_vel = joint_vel_raw[:, joint_perm]
        else:
            # Backwards compat: caller promises NPZ column order matches
            # their canonical order. Dangerous across simulators; kept
            # only for tests that don't have an env handy.
            self.joint_pos = joint_pos_raw
            self.joint_vel = joint_vel_raw

        idx = torch.tensor(motion_idx, dtype=torch.long, device=device)
        bp = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
        bq = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
        bl = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
        ba = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)
        self.body_pos_w = bp[:, idx]
        self.body_quat_w = bq[:, idx]
        self.body_lin_vel_w = bl[:, idx]
        self.body_ang_vel_w = ba[:, idx]

        self.time_step_total = int(self.joint_pos.shape[0])
        self.fps = float(np.asarray(data["fps"]).item()) if "fps" in data.files else 0.0


class MotionCommand(CommandTerm):
    """Reference-motion tracking command. Sim-agnostic (writes via writer protocol)."""

    cfg: "MotionCommandCfg"

    def __init__(self, env: "World", cfg: "MotionCommandCfg"):
        super().__init__(env, cfg)
        self._rd = env.get_robot_data(cfg.entity_name)
        self._writer = env.get_robot_state_writer(cfg.entity_name)

        prefix = cfg.body_name_prefix
        self.robot_anchor_body_index = self._rd.find_body_index(
            prefix + cfg.anchor_body_name
        )
        self.motion_anchor_body_index = cfg.body_names.index(cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            [self._rd.find_body_index(prefix + n) for n in cfg.body_names],
            dtype=torch.long,
            device=self.device,
        )

        # Resolve the actuated-joint order expected by this sim's
        # RobotStateWriter. ``env.act_manager.actuated_joint_names`` is
        # the single canonical list (same contract for Newton, Genesis,
        # Mjlab). Strip the sim-specific entity prefix so the names line
        # up with the NPZ's bare-name joint list from
        # ``mujoco_replayer``.
        actuated = list(env.act_manager.actuated_joint_names)
        if prefix and all(n.startswith(prefix) for n in actuated):
            bare_joint_names = tuple(n[len(prefix):] for n in actuated)
        else:
            bare_joint_names = tuple(actuated)

        self.motion = MotionLoader(
            cfg.motion_file,
            cfg.body_names,
            joint_names_cfg=bare_joint_names,
            device=self.device,
        )

        self.time_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device,
        )
        self.body_pos_relative_w = torch.zeros(
            self.num_envs, len(cfg.body_names), 3, device=self.device,
        )
        self.body_quat_relative_w = torch.zeros(
            self.num_envs, len(cfg.body_names), 4, device=self.device,
        )
        self.body_quat_relative_w[:, :, 0] = 1.0

        # Base-class set_command writes into self._command indexed by
        # env_ids. We don't support external override (motion is
        # time-indexed), but keep a zero buffer so the attribute exists.
        num_cols = 2 * self.motion.joint_pos.shape[1]
        self._command_buf = torch.zeros(
            self.num_envs, num_cols, device=self.device,
        )

        # Adaptive-sampling buffers. One bin per real-time second of
        # motion, matching mjlab's convention.
        self.bin_count = int(
            self.motion.time_step_total // (1.0 / env.control_dt)
        ) + 1
        self.bin_failed_count = torch.zeros(self.bin_count, device=self.device)
        self._current_bin_failed = torch.zeros(self.bin_count, device=self.device)
        kernel = torch.tensor(
            [cfg.adaptive_lambda ** i for i in range(cfg.adaptive_kernel_size)],
            device=self.device,
            dtype=torch.float32,
        )
        self.kernel = kernel / kernel.sum()

        # Per-env error / sampling metrics, same keys as mjlab for parity.
        metric_keys = (
            "error_anchor_pos", "error_anchor_rot",
            "error_anchor_lin_vel", "error_anchor_ang_vel",
            "error_body_pos", "error_body_rot",
            "error_body_lin_vel", "error_body_ang_vel",
            "error_joint_pos", "error_joint_vel",
            "sampling_entropy", "sampling_top1_prob", "sampling_top1_bin",
        )
        self.metrics: dict[str, torch.Tensor] = {
            k: torch.zeros(self.num_envs, device=self.device) for k in metric_keys
        }

        # env_origins is mjlab-only (multi-env spatial offset baked into
        # scene). Newton / Genesis handle per-env isolation internally and
        # don't expose an env_origins tensor — fall back to no offset.
        scene_obj = getattr(env.scene_manager, "scene", None) or env.scene_manager
        self._env_origins = getattr(scene_obj, "env_origins", None)

    # ------------------------------------------------------------------
    # Command API — command() is derived from time_steps, not set_command.
    # ------------------------------------------------------------------
    @property
    def command(self) -> torch.Tensor:
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    def set_command(self, env_ids, values):
        raise NotImplementedError(
            "MotionCommand is time-indexed; external set_command is not supported."
        )

    # ------------------------------------------------------------------
    # Motion reference (world frame). env_origins offset applied for mjlab.
    # ------------------------------------------------------------------
    def _add_env_origins(self, pos: torch.Tensor, per_body: bool) -> torch.Tensor:
        if self._env_origins is None:
            return pos
        eo = self._env_origins
        return pos + (eo[:, None, :] if per_body else eo)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.motion.joint_pos[self.time_steps]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motion.joint_vel[self.time_steps]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._add_env_origins(
            self.motion.body_pos_w[self.time_steps], per_body=True,
        )

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        pos = self.motion.body_pos_w[self.time_steps, self.motion_anchor_body_index]
        return self._add_env_origins(pos, per_body=False)

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps, self.motion_anchor_body_index]

    # ------------------------------------------------------------------
    # Live robot state via RobotData protocol.
    # ------------------------------------------------------------------
    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self._rd.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self._rd.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self._rd.body_pos_w_all[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self._rd.body_quat_w_all[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self._rd.body_lin_vel_w_all[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self._rd.body_ang_vel_w_all[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self._rd.body_pos_w_all[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self._rd.body_quat_w_all[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self._rd.body_lin_vel_w_all[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self._rd.body_ang_vel_w_all[:, self.robot_anchor_body_index]

    # ------------------------------------------------------------------
    # Anchor-aligned (yaw-only) relative body poses.
    # ------------------------------------------------------------------
    def update_relative_body_poses(self) -> None:
        """Recompute ``body_pos_relative_w`` / ``body_quat_relative_w``.

        Expresses each motion-reference body in a frame whose origin is
        at ``(robot_anchor.x, robot_anchor.y, motion_anchor.z)`` and
        whose yaw is the delta yaw between robot and motion anchors.
        This lets relative rewards ignore absolute XY drift / yaw while
        still penalising pitch, roll, Z, and articulation errors.
        """
        num_bodies = len(self.cfg.body_names)
        anchor_pos_rep = self.anchor_pos_w[:, None, :].expand(-1, num_bodies, 3)
        anchor_quat_rep = self.anchor_quat_w[:, None, :].expand(-1, num_bodies, 4)
        robot_anchor_pos_rep = self.robot_anchor_pos_w[:, None, :].expand(
            -1, num_bodies, 3,
        )
        robot_anchor_quat_rep = self.robot_anchor_quat_w[:, None, :].expand(
            -1, num_bodies, 4,
        )

        delta_pos_w = robot_anchor_pos_rep.clone()
        delta_pos_w[..., 2] = anchor_pos_rep[..., 2]
        delta_ori_w = yaw_quat_wxyz(
            quat_mul_wxyz(robot_anchor_quat_rep, quat_inv_wxyz(anchor_quat_rep))
        )

        self.body_quat_relative_w = quat_mul_wxyz(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_rotate_wxyz(
            delta_ori_w, self.body_pos_w - anchor_pos_rep,
        )

    def _update_metrics(self) -> None:
        self.metrics["error_anchor_pos"] = torch.norm(
            self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1,
        )
        self.metrics["error_anchor_rot"] = quat_error_magnitude_wxyz(
            self.anchor_quat_w, self.robot_anchor_quat_w,
        )
        self.metrics["error_anchor_lin_vel"] = torch.norm(
            self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1,
        )
        self.metrics["error_anchor_ang_vel"] = torch.norm(
            self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1,
        )
        self.metrics["error_body_pos"] = torch.norm(
            self.body_pos_relative_w - self.robot_body_pos_w, dim=-1,
        ).mean(-1)
        self.metrics["error_body_rot"] = quat_error_magnitude_wxyz(
            self.body_quat_relative_w, self.robot_body_quat_w,
        ).mean(-1)
        self.metrics["error_body_lin_vel"] = torch.norm(
            self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1,
        ).mean(-1)
        self.metrics["error_body_ang_vel"] = torch.norm(
            self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1,
        ).mean(-1)
        self.metrics["error_joint_pos"] = torch.norm(
            self.joint_pos - self.robot_joint_pos, dim=-1,
        )
        self.metrics["error_joint_vel"] = torch.norm(
            self.joint_vel - self.robot_joint_vel, dim=-1,
        )

    # ------------------------------------------------------------------
    # Frame sampling strategies.
    # ------------------------------------------------------------------
    def _uniform_sampling(self, env_ids: torch.Tensor) -> None:
        self.time_steps[env_ids] = torch.randint(
            0, self.motion.time_step_total, (len(env_ids),), device=self.device,
        )
        self.metrics["sampling_entropy"][:] = 1.0
        self.metrics["sampling_top1_prob"][:] = 1.0 / max(self.bin_count, 1)
        self.metrics["sampling_top1_bin"][:] = 0.5

    def _adaptive_sampling(self, env_ids: torch.Tensor) -> None:
        # Record current-step failures into bins for the EMA update
        # that happens in _update_command.
        term_manager = getattr(self._env, "termination_manager", None)
        if term_manager is not None and hasattr(term_manager, "terminated"):
            episode_failed = term_manager.terminated[env_ids]
            if torch.any(episode_failed):
                current_bin_index = torch.clamp(
                    (self.time_steps * self.bin_count)
                    // max(self.motion.time_step_total, 1),
                    0,
                    self.bin_count - 1,
                )
                fail_bins = current_bin_index[env_ids][episode_failed]
                self._current_bin_failed[:] = torch.bincount(
                    fail_bins, minlength=self.bin_count,
                )

        probs = (
            self.bin_failed_count
            + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
        )
        probs = torch.nn.functional.pad(
            probs.unsqueeze(0).unsqueeze(0),
            (0, self.cfg.adaptive_kernel_size - 1),
            mode="replicate",
        )
        probs = torch.nn.functional.conv1d(probs, self.kernel.view(1, 1, -1)).view(-1)
        probs = probs / probs.sum()

        sampled_bins = torch.multinomial(probs, len(env_ids), replacement=True)
        self.time_steps[env_ids] = (
            (sampled_bins + torch.rand(len(env_ids), device=self.device))
            / self.bin_count
            * (self.motion.time_step_total - 1)
        ).long()

        H = -(probs * (probs + 1e-12).log()).sum()
        H_norm = H / math.log(self.bin_count) if self.bin_count > 1 else torch.tensor(1.0)
        pmax, imax = probs.max(dim=0)
        self.metrics["sampling_entropy"][:] = H_norm
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count

    # ------------------------------------------------------------------
    # Write reference state through the sim-agnostic writer protocol.
    # ------------------------------------------------------------------
    def _write_reference_state_to_sim(
        self,
        env_ids: torch.Tensor,
        root_pos: torch.Tensor,
        root_quat_wxyz: torch.Tensor,
        root_lin_vel: torch.Tensor,
        root_ang_vel: torch.Tensor,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> None:
        lo, hi = self._rd.soft_joint_pos_limits
        joint_pos = torch.clip(joint_pos, lo.unsqueeze(0), hi.unsqueeze(0))
        self._writer.set_dof_positions(joint_pos, env_ids=env_ids)
        self._writer.set_dof_velocities(joint_vel, env_ids=env_ids)
        self._writer.set_root_pose(root_pos, root_quat_wxyz, env_ids=env_ids)
        self._writer.set_root_velocity(root_lin_vel, root_ang_vel, env_ids=env_ids)
        self._writer.eval_fk(env_ids=env_ids)

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        # 1. Sample a new motion frame per env.
        if self.cfg.sampling_mode == "start":
            self.time_steps[env_ids] = 0
        elif self.cfg.sampling_mode == "uniform":
            self._uniform_sampling(env_ids)
        else:
            assert self.cfg.sampling_mode == "adaptive", (
                f"Unknown sampling_mode: {self.cfg.sampling_mode!r}"
            )
            self._adaptive_sampling(env_ids)

        # 2. Reference root state at sampled frame. The root follows
        # the body_names[0] convention used by mjlab: the first listed
        # body is the floating-base body whose pose/velocity seeds the
        # root state write.
        root_pos = self.body_pos_w[env_ids, 0].clone()
        root_quat = self.body_quat_w[env_ids, 0].clone()
        root_lin_vel = self.body_lin_vel_w[env_ids, 0].clone()
        root_ang_vel = self.body_ang_vel_w[env_ids, 0].clone()

        # 3. Reference State Initialization (RSI) perturbations.
        pose_ranges = torch.tensor(
            [
                self.cfg.pose_range.get(k, (0.0, 0.0))
                for k in ("x", "y", "z", "roll", "pitch", "yaw")
            ],
            device=self.device,
        )
        pose_delta = _sample_uniform(
            pose_ranges[:, 0], pose_ranges[:, 1],
            (len(env_ids), 6), device=self.device,
        )
        root_pos = root_pos + pose_delta[:, 0:3]
        root_quat = quat_mul_wxyz(
            quat_from_euler_xyz_wxyz(
                pose_delta[:, 3], pose_delta[:, 4], pose_delta[:, 5],
            ),
            root_quat,
        )

        vel_ranges = torch.tensor(
            [
                self.cfg.velocity_range.get(k, (0.0, 0.0))
                for k in ("x", "y", "z", "roll", "pitch", "yaw")
            ],
            device=self.device,
        )
        vel_delta = _sample_uniform(
            vel_ranges[:, 0], vel_ranges[:, 1],
            (len(env_ids), 6), device=self.device,
        )
        root_lin_vel = root_lin_vel + vel_delta[:, :3]
        root_ang_vel = root_ang_vel + vel_delta[:, 3:]

        joint_pos = self.joint_pos[env_ids].clone()
        joint_vel = self.joint_vel[env_ids]
        joint_pos = joint_pos + _sample_uniform(
            self.cfg.joint_position_range[0],
            self.cfg.joint_position_range[1],
            joint_pos.shape,
            device=self.device,
        )

        # 4. Write through the sim-agnostic writer protocol (eval_fk
        # is a no-op on mjlab/Genesis, triggers FK on Newton).
        self._write_reference_state_to_sim(
            env_ids, root_pos, root_quat, root_lin_vel, root_ang_vel,
            joint_pos, joint_vel,
        )

    # ------------------------------------------------------------------
    # Per-step update (called every CommandTerm.compute).
    # ------------------------------------------------------------------
    def _update_command(self) -> None:
        self.time_steps += 1
        rollover = torch.where(self.time_steps >= self.motion.time_step_total)[0]
        if rollover.numel() > 0:
            self._resample_command(rollover)

        self.update_relative_body_poses()

        if self.cfg.sampling_mode == "adaptive":
            self.bin_failed_count = (
                self.cfg.adaptive_alpha * self._current_bin_failed
                + (1.0 - self.cfg.adaptive_alpha) * self.bin_failed_count
            )
            self._current_bin_failed.zero_()

    # ------------------------------------------------------------------
    # Reset hook — base class schedules a resample for env_ids; we also
    # refresh the relative body buffers so first-step reads are correct.
    # ------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        super().reset(env_ids)
        self.update_relative_body_poses()

    def reset_to_frame(self, env_ids: torch.Tensor, frame: int) -> None:
        """Deterministic reset to an exact motion frame (no RSI)."""
        self.time_steps[env_ids] = int(frame)
        self._write_reference_state_to_sim(
            env_ids,
            self.body_pos_w[env_ids, 0],
            self.body_quat_w[env_ids, 0],
            self.body_lin_vel_w[env_ids, 0],
            self.body_ang_vel_w[env_ids, 0],
            self.joint_pos[env_ids],
            self.joint_vel[env_ids],
        )
        self.update_relative_body_poses()


@dataclass(kw_only=True)
class MotionCommandCfg(CommandTermCfg):
    """Configuration for :class:`MotionCommand`."""

    motion_file: str
    """NPZ path. Produced by ``rlworld.tools.motion.csv_to_npz``."""

    anchor_body_name: str
    """Body whose world pose defines the anchor frame. Must appear in
    ``body_names``. The anchor drives the yaw-aligned relative-frame
    rewards and the bad-anchor termination checks."""

    body_names: tuple[str, ...]
    """Bodies to track (bare names — no simulator prefix). The NPZ must
    contain all of them. The first entry is the floating-base body whose
    pose/velocity seeds the root state written on reset. For humanoids
    this is typically the pelvis or torso."""

    entity_name: str = "robot"
    """Scene entity name; passed to
    ``env.get_robot_data()`` / ``env.get_robot_state_writer()``."""

    body_name_prefix: str = ""
    """Prefix prepended to ``body_names`` / ``anchor_body_name`` when
    resolving against the simulator's body namespace. Newton prepends the
    entity prefix (e.g. ``"T1/"``); mjlab / Genesis use ``""``."""

    pose_range: dict[str, tuple[float, float]] = field(default_factory=dict)
    """Per-axis RSI ranges for root pose. Keys: ``x`` / ``y`` / ``z``
    (meters), ``roll`` / ``pitch`` / ``yaw`` (radians). Missing keys
    default to ``(0, 0)``."""

    velocity_range: dict[str, tuple[float, float]] = field(default_factory=dict)
    """Per-axis RSI ranges for root velocity. Same keys as ``pose_range``;
    linear in m/s, angular in rad/s."""

    joint_position_range: tuple[float, float] = (-0.52, 0.52)
    """Symmetric RSI range added to every actuated joint position (radians)."""

    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001

    sampling_mode: Literal["adaptive", "uniform", "start"] = "adaptive"
    """Motion-frame sampling strategy on reset / rollover:

    - ``"adaptive"``: failure-weighted curriculum (default for training)
    - ``"uniform"``: uniform over ``[0, motion_length)``
    - ``"start"``: always frame 0 (for deterministic playback / eval)
    """

    # Motion rollover drives resampling; disable the base class's
    # timer-based resample by setting a huge interval.
    resampling_time_range: tuple[float, float] = (1e9, 1e9)

    def build(self, env: "World") -> MotionCommand:
        return MotionCommand(env, self)
