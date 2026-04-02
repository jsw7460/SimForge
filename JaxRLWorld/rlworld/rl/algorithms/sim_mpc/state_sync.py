"""
GenesisStateSync: Training env → Planning env state synchronization.

Forks a single training environment's physics + manager state and
broadcasts it to all S planning environments for MPPI rollouts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from rlworld.rl.envs.genesis.genesis_env import GenesisEnv


class GenesisStateSync:
    """Synchronizes state from a training env to a planning env."""

    def __init__(self, training_env: GenesisEnv, planning_env: GenesisEnv):
        self.train_env = training_env
        self.plan_env = planning_env
        # Cached states to avoid repeated get_state() calls during planning
        self._cached_plan_state = None
        self._cached_train_slices: dict[int, dict[str, list]] = {}

    def begin_planning(self, train_env_idx: int) -> None:
        """Cache training env state once per planning call.

        Call this before the first fork_state in a planning sequence.
        Avoids repeated scene.get_state() calls during CEM iterations.
        """
        train_state = self.train_env.scene.get_state()
        self._cached_plan_state = self.plan_env.scene.get_state()

        # Pre-slice and detach training state for this env idx
        slices = []
        for train_ss, plan_ss in zip(
            train_state.solvers_state, self._cached_plan_state.solvers_state
        ):
            solver_slices = {}
            for attr_name in [
                "qpos", "dofs_vel", "dofs_acc",
                "links_pos", "links_quat",
                "i_pos_shift", "mass_shift", "friction_ratio",
            ]:
                src = getattr(train_ss, attr_name, None)
                dst = getattr(plan_ss, attr_name, None)
                if src is None or dst is None:
                    continue
                solver_slices[attr_name] = src[train_env_idx].sceneless()
            slices.append(solver_slices)
        self._cached_train_slices[train_env_idx] = slices

    def fork_state(self, train_env_idx: int) -> None:
        """Broadcast training_env[idx] physics state to all planning envs.

        Uses cached state if begin_planning() was called.
        """
        if self._cached_plan_state is not None and train_env_idx in self._cached_train_slices:
            # Fast path: use cached slices
            plan_state = self._cached_plan_state
            slices = self._cached_train_slices[train_env_idx]
            for solver_idx, plan_ss in enumerate(plan_state.solvers_state):
                solver_slices = slices[solver_idx]
                for attr_name, sliced in solver_slices.items():
                    dst = getattr(plan_ss, attr_name)
                    dst[:] = sliced.unsqueeze(0).expand_as(dst)
        else:
            # Fallback: get state fresh
            train_state = self.train_env.scene.get_state()
            plan_state = self.plan_env.scene.get_state()
            for train_ss, plan_ss in zip(
                train_state.solvers_state, plan_state.solvers_state
            ):
                for attr_name in [
                    "qpos", "dofs_vel", "dofs_acc",
                    "links_pos", "links_quat",
                    "i_pos_shift", "mass_shift", "friction_ratio",
                ]:
                    src = getattr(train_ss, attr_name, None)
                    dst = getattr(plan_ss, attr_name, None)
                    if src is None or dst is None:
                        continue
                    sliced = src[train_env_idx].sceneless()
                    dst[:] = sliced.unsqueeze(0).expand_as(dst)

        self.plan_env.scene.reset(state=plan_state)

    def end_planning(self) -> None:
        """Clear cached state after planning is done."""
        self._cached_plan_state = None
        self._cached_train_slices.clear()

    @staticmethod
    def _detach(t: torch.Tensor) -> torch.Tensor:
        """Detach tensor from Genesis scene graph if needed."""
        if hasattr(t, "sceneless"):
            return t.sceneless()
        return t

    def sync_managers(self, train_env_idx: int) -> None:
        """Sync reward-relevant manager state from training env[idx] to planning env."""
        S = self.plan_env.num_envs
        idx = train_env_idx
        _d = self._detach

        # --- Command Manager ---
        self.plan_env.command_manager._commands_tensor[:] = (
            _d(self.train_env.command_manager._commands_tensor[idx])
            .unsqueeze(0)
            .expand(S, -1)
        )

        # --- Action Manager (action history for action_rate reward) ---
        for i in range(self.train_env.act_manager._action_history_len):
            self.plan_env.act_manager._raw_action_history[i][:] = (
                _d(self.train_env.act_manager._raw_action_history[i][idx])
                .unsqueeze(0)
                .expand(S, -1)
            )
            self.plan_env.act_manager._processed_action_history[i][:] = (
                _d(self.train_env.act_manager._processed_action_history[i][idx])
                .unsqueeze(0)
                .expand(S, -1)
            )

        # --- Contact Manager (per-group buffers) ---
        for group_name in self.train_env.contact_manager.group_names():
            train_group = self.train_env.contact_manager._get_group(group_name)
            plan_group = self.plan_env.contact_manager._get_group(group_name)
            for attr in [
                "current_air_time", "current_contact_time",
                "last_air_time", "last_contact_time", "_prev_is_contact",
            ]:
                src = getattr(train_group, attr, None)
                if src is None:
                    continue
                dst = getattr(plan_group, attr)
                dst[:] = _d(src[idx]).unsqueeze(0).expand_as(dst)

        # --- Termination Manager ---
        self.plan_env.termination_manager.episode_length_buf[:] = (
            _d(self.train_env.termination_manager.episode_length_buf[idx])
        )

        # --- Gait Manager (if LocomotionEnv) ---
        if hasattr(self.train_env, "gait_manager") and hasattr(self.plan_env, "gait_manager"):
            self.plan_env.gait_manager.gait_timer[:] = (
                _d(self.train_env.gait_manager.gait_timer[idx])
            )

        # --- Stateful Reward Terms (e.g., feet_swing_height_mjlab.peak_heights) ---
        self._sync_stateful_reward_terms(idx)

    def _sync_stateful_reward_terms(self, train_env_idx: int) -> None:
        """Sync internal state of stateful reward term instances."""
        S = self.plan_env.num_envs
        idx = train_env_idx
        _d = self._detach

        train_instances = self.train_env.reward_manager._instances
        plan_instances = self.plan_env.reward_manager._instances

        for name in train_instances:
            if name not in plan_instances:
                continue
            train_inst = train_instances[name]
            plan_inst = plan_instances[name]

            # Force lazy initialization on plan_inst if train_inst is initialized
            if hasattr(train_inst, "_initialized") and train_inst._initialized:
                if hasattr(plan_inst, "_initialized") and not plan_inst._initialized:
                    if hasattr(plan_inst, "_lazy_init"):
                        plan_inst._lazy_init(self.plan_env)

            # Sync all tensor attributes that have batch dimension (num_envs, ...)
            for attr_name in vars(train_inst):
                if attr_name.startswith("_"):
                    continue
                src = getattr(train_inst, attr_name, None)
                if not isinstance(src, torch.Tensor):
                    continue
                if src.ndim == 0 or src.shape[0] != self.train_env.num_envs:
                    continue
                dst = getattr(plan_inst, attr_name, None)
                if not isinstance(dst, torch.Tensor) or dst.shape[0] != S:
                    # dst doesn't exist or wrong size — create it
                    new_val = _d(src[idx]).unsqueeze(0).expand(S, *src.shape[1:]).contiguous()
                    setattr(plan_inst, attr_name, new_val)
                else:
                    dst[:] = _d(src[idx]).unsqueeze(0).expand_as(dst)

    def fork_and_sync(self, train_env_idx: int) -> None:
        """Convenience: fork physics state + sync all managers."""
        self.fork_state(train_env_idx)
        self.sync_managers(train_env_idx)
