"""Policy inference wrappers for evaluation."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import jax
import numpy as np
import torch

from rlworld.rl.utils.jax_utils import torch_to_jax, jax_to_torch

if TYPE_CHECKING:
    from rlworld.rl.envs.multi_sim_world import _JointPermutation
    from rlworld.rl.runners import BaseRunner


class PolicyWrapper(ABC):
    """
    Base inference wrapper for evaluation.

    Subclasses implement get_action for different algorithm families:
    - ModelPolicyWrapper: PPO / SAC / FastTD3 (JIT-compiled model inference)
    - MPCPolicyWrapper:   TD-MPC2 / ScaffoldedTDMPC2 (MPPI planning)

    Use PolicyWrapper.from_runner() factory to create the appropriate subclass.
    """

    def __init__(
        self,
        runner: "BaseRunner",
        device: torch.device,
        joint_perm: "_JointPermutation | None" = None,
    ):
        self.device = device
        self.is_squashed = runner.squash_output
        if self.is_squashed:
            self.action_scale = runner.action_scale
            self.action_bias = runner.action_bias
        self._joint_perm = joint_perm

    @classmethod
    def from_runner(
        cls,
        runner: "BaseRunner",
        device: torch.device,
        joint_perm: "_JointPermutation | None" = None,
    ) -> "PolicyWrapper":
        """Factory: returns appropriate subclass based on algorithm type."""
        if hasattr(runner.alg, 'act_with_t0'):
            return MPCPolicyWrapper(runner, device, joint_perm=joint_perm)
        return ModelPolicyWrapper(runner, device, joint_perm=joint_perm)

    def _process_action(self, actions: jax.Array) -> jax.Array:
        """Apply action rescaling for squashed policies."""
        if self.is_squashed:
            return actions * self.action_scale + self.action_bias
        return actions

    def _permute_obs_to_canonical(
        self, env_obs: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Reorder observations from eval sim's joint order to canonical."""
        if self._joint_perm is None:
            return env_obs
        return self._joint_perm.permute_obs(env_obs)

    def _permute_actions_to_sim(self, actions: torch.Tensor) -> torch.Tensor:
        """Reorder actions from canonical order to eval sim's joint order."""
        if self._joint_perm is None:
            return actions
        return self._joint_perm.permute_actions(actions)

    @abstractmethod
    def get_action(
        self,
        env_obs: dict[str, torch.Tensor],
        robot_states: torch.Tensor,
        deterministic: bool = True,
    ) -> torch.Tensor:
        ...

    def notify_reset(self, reset_mask: np.ndarray) -> None:
        """Called when environments reset. Override in subclasses if needed."""
        pass


class ModelPolicyWrapper(PolicyWrapper):
    """JIT-compiled batched inference for PPO / SAC / FastTD3."""

    def __init__(
        self,
        runner: "BaseRunner",
        device: torch.device,
        joint_perm: "_JointPermutation | None" = None,
    ):
        super().__init__(runner, device, joint_perm=joint_perm)
        model = runner.alg.train_state.model
        self._key = jax.random.PRNGKey(0)

        def _single(obs, key):
            action, _ = model.act_inference(obs, key=key)
            return action

        self._inference_fn = jax.jit(
            jax.vmap(_single, in_axes=(0, None))
        )

    def get_action(self, env_obs, robot_states, deterministic=True):
        env_obs = self._permute_obs_to_canonical(env_obs)
        actor_obs = torch_to_jax(env_obs["actor"])
        action_jax = self._inference_fn(actor_obs, self._key)
        actions = jax_to_torch(self._process_action(action_jax), self.device)
        return self._permute_actions_to_sim(actions)


class MPCPolicyWrapper(PolicyWrapper):
    """MPPI planning for TD-MPC2 / ScaffoldedTDMPC2."""

    def __init__(
        self,
        runner: "BaseRunner",
        device: torch.device,
        joint_perm: "_JointPermutation | None" = None,
    ):
        super().__init__(runner, device, joint_perm=joint_perm)
        self._runner = runner
        self._t0_mask = np.ones(
            runner.alg._prev_mean.shape[0], dtype=bool
        )

    def get_action(self, env_obs, robot_states, deterministic=True):
        env_obs = self._permute_obs_to_canonical(env_obs)
        actor_obs = torch_to_jax(env_obs["actor"])
        action_jax = self._runner.alg.act_with_t0(
            obs=actor_obs, t0_mask=self._t0_mask, eval_mode=True,
        )
        self._t0_mask[:] = False
        actions = jax_to_torch(self._process_action(action_jax), self.device)
        return self._permute_actions_to_sim(actions)

    def notify_reset(self, reset_mask: np.ndarray) -> None:
        """Mark environments that need MPPI warm-start reset."""
        self._t0_mask[reset_mask] = True
