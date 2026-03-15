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

    def fork_state(self, train_env_idx: int) -> None:
        """Broadcast training_env[idx] physics state to all planning envs."""
        train_state = self.train_env.scene.get_state()
        plan_state = self.plan_env.scene.get_state()

        S = self.plan_env.num_envs

        for train_solver_state, plan_solver_state in zip(
            train_state.solvers_state, plan_state.solvers_state
        ):
            for attr_name in [
                "qpos", "dofs_vel", "dofs_acc",
                "links_pos", "links_quat",
                "i_pos_shift", "mass_shift", "friction_ratio",
            ]:
                src = getattr(train_solver_state, attr_name, None)
                dst = getattr(plan_solver_state, attr_name, None)
                if src is None or dst is None:
                    continue
                # src[train_env_idx] → broadcast to all S planning envs
                # .sceneless() detaches the tensor from its Scene to allow cross-scene copy
                sliced = src[train_env_idx].sceneless()  # remove batch dim + detach
                dst[:] = sliced.unsqueeze(0).expand_as(dst)

        self.plan_env.scene.reset(state=plan_state)

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
        self.plan_env.act_manager._prev_processed_actions[:] = (
            _d(self.train_env.act_manager._prev_processed_actions[idx])
            .unsqueeze(0)
            .expand(S, -1)
        )
        self.plan_env.act_manager._processed_actions[:] = (
            _d(self.train_env.act_manager._processed_actions[idx])
            .unsqueeze(0)
            .expand(S, -1)
        )
        self.plan_env.act_manager._prev_raw_actions[:] = (
            _d(self.train_env.act_manager._prev_raw_actions[idx])
            .unsqueeze(0)
            .expand(S, -1)
        )
        self.plan_env.act_manager._raw_actions[:] = (
            _d(self.train_env.act_manager._raw_actions[idx])
            .unsqueeze(0)
            .expand(S, -1)
        )

        # --- Contact Manager ---
        for attr in [
            "current_air_time", "current_contact_time",
            "last_air_time", "last_contact_time", "_prev_is_contact",
        ]:
            src = getattr(self.train_env.contact_manager, attr, None)
            if src is None:
                continue
            dst = getattr(self.plan_env.contact_manager, attr)
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
                if dst is None or not isinstance(dst, torch.Tensor):
                    continue
                if dst.shape[0] != S:
                    continue
                dst[:] = _d(src[idx]).unsqueeze(0).expand_as(dst)

    def fork_and_sync(self, train_env_idx: int) -> None:
        """Convenience: fork physics state + sync all managers."""
        self.fork_state(train_env_idx)
        self.sync_managers(train_env_idx)
