"""
Privileged Model-Based Runner for Scaffolded TD-MPC2 + ABD-Net.

Inherits ModelBasedRunner and overrides methods to support:
  - Privileged observations (s+ = [s-, priv])
  - Dual world models (target on s-, scaffolded on s+)
  - Exploration / target policy branching via explore_ratio
  - ScaffoldedSequenceReplayBuffer with privileged storage

Compatible with teacher-student, asymmetric actor-critic,
and sensory scaffolding paradigms.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

import jax
import jax.numpy as jnp

from rlworld.rl.algorithms.scaffolded_tdmpc2 import ScaffoldedTDMPC2
from rlworld.rl.configs import ConfigsForRun
from rlworld.rl.configs.robots.kinematic_tree import KinematicTree
from rlworld.rl.envs import World
from rlworld.rl.runners.model_based_runner import ModelBasedRunner
from rlworld.rl.utils.jax_utils import torch_to_jax, jax_to_torch


class PrivilegedModelBasedRunner(ModelBasedRunner):
    """
    Model-based runner with privileged observation support.

    Overrides from ModelBasedRunner:
      1. _init_training_modules  - build KinematicTree
      2. _init_algorithm         - ScaffoldedTDMPC2 with ABD-Net
      3. _init_storage           - privileged_obs_dim in replay buffer
      4. _collect_experience     - privileged obs + explore/target branching
      5. _get_initial_obs        - split obs dict {"actor", "privileged"}

    Inherited unchanged:
      - learn(), _run_training_iteration(), checkpoint, logging, post_iteration
    """

    algorithm_name: str = "scaffolded_tdmpc2"
    alg: ScaffoldedTDMPC2

    def __init__(
        self,
        env: World,
        cfgs: ConfigsForRun,
        use_wandb: bool = True,
        seed: int = 0,
    ):
        # Extract privileged config before super().__init__,
        # since overridden _init_training_modules / _init_algorithm
        # are called inside super().__init__.
        alg_cfg = cfgs.algorithm
        self.explore_ratio: float = getattr(alg_cfg, "explore_ratio", 0.5)

        # privileged_obs_dim computed from obs_manager in _init_training_modules
        self.privileged_obs_dim: int = 0

        # Kinematic tree from robot config in scene
        robot_cfg = cfgs.scene.robot_cfg
        urdf_path = getattr(robot_cfg, "urdf_path", None)
        mjcf_path = getattr(robot_cfg, "mjcf_path", None)

        self.kinematic_tree = KinematicTree(
            urdf_path=urdf_path,
            mjcf_path=mjcf_path,
        )

        super().__init__(env, cfgs, use_wandb=use_wandb, seed=seed)

    # ------------------------------------------------------------------
    # Override 1: Training modules
    # ------------------------------------------------------------------

    def _init_training_modules(self) -> None:
        """
        Build training modules. Called by BaseRunner.__init__.

        Computes privileged_obs_dim from obs_manager, then delegates
        to parent for world model construction.
        """
        obs_dim = self.env.obs_manager.calculate_obs_dim()
        self.privileged_obs_dim = obs_dim.get("privileged", 0)
        super()._init_training_modules()

    # ------------------------------------------------------------------
    # Override 2: Algorithm construction
    # ------------------------------------------------------------------

    def _init_algorithm(self) -> ScaffoldedTDMPC2:
        """Build ScaffoldedTDMPC2 with ABD-Net dynamics."""
        alg_cfg = self.cfgs.algorithm

        obs_dim = self.obs_dim
        action_dim = self.num_actions_dim

        return ScaffoldedTDMPC2(
            kinematic_tree=self.kinematic_tree,
            obs_dim=obs_dim,
            action_dim=action_dim,
            privileged_obs_dim=self.privileged_obs_dim,
            num_envs=self.env.num_envs,
            # Base TD-MPC2
            gamma=alg_cfg.gamma,
            episode_length=self.env.max_episode_length,
            lr=alg_cfg.lr,
            pi_lr=getattr(alg_cfg, "pi_lr", alg_cfg.lr),
            tau=getattr(alg_cfg, "tau", 0.01),
            mpc=getattr(alg_cfg, "mpc", True),
            horizon=getattr(alg_cfg, "horizon", 3),
            num_samples=getattr(alg_cfg, "num_samples", 512),
            num_pi_trajs=getattr(alg_cfg, "num_pi_trajs", 24),
            num_elites=getattr(alg_cfg, "num_elites", 64),
            temperature=getattr(alg_cfg, "temperature", 0.5),
            consistency_coef=getattr(alg_cfg, "consistency_coef", 2.0),
            reward_coef=getattr(alg_cfg, "reward_coef", 0.5),
            value_coef=getattr(alg_cfg, "value_coef", 0.1),
            entropy_coef=getattr(alg_cfg, "entropy_coef", 1e-4),
            batch_size=getattr(alg_cfg, "batch_size", 256),
            grad_clip_norm=getattr(alg_cfg, "grad_clip_norm", 20.0),
            # MLP
            mlp_dim=getattr(alg_cfg, "mlp_dim", 512),
            num_q=getattr(alg_cfg, "num_q", 5),
            # ABD-Net
            link_channels=getattr(alg_cfg, "link_channels", 8),
            spatial_dim=getattr(alg_cfg, "spatial_dim", 6),
            learnable_contribution_weight=getattr(
                alg_cfg, "learnable_contribution_weight", False
            ),
            use_positive_constraint=getattr(
                alg_cfg, "use_positive_constraint", True
            ),
            residual_scale_init=getattr(alg_cfg, "residual_scale_init", 0.1),
            ortho_coef=getattr(alg_cfg, "ortho_coef", 0.01),
            # Scaffolding
            explore_ratio=self.explore_ratio,
            key=self.key,
        )

    # ------------------------------------------------------------------
    # Override 3: Storage initialization
    # ------------------------------------------------------------------

    def _init_storage(self) -> None:
        """Initialize ScaffoldedSequenceReplayBuffer with privileged_obs_dim."""
        obs_dim = self.env.obs_manager.calculate_obs_dim()
        alg_cfg = self.cfgs.algorithm
        size_per_env = alg_cfg.get("buffer_size", 1_000_000) // self.env.num_envs

        self.alg.init_storage({
            "num_envs": self.env.num_envs,
            "obs_dim": obs_dim["actor"],
            "action_dim": self.env.num_actions,
            "privileged_obs_dim": self.privileged_obs_dim,
            "size_per_env": size_per_env,
        })

    # ------------------------------------------------------------------
    # Override 4: Experience collection
    # ------------------------------------------------------------------

    def _collect_experience(
        self,
        obs: jax.Array,
        ep_infos: List[Dict],
        iteration: int,
    ) -> Dict[str, Any]:
        """
        Collect experience with privileged obs and explore/target branching.

        Overrides ModelBasedRunner._collect_experience to:
          - Maintain obs as {"actor": ..., "privileged": ...} dict
          - Branch between exploration policy (s+) and target policy (s-)
          - Pass privileged_obs / next_privileged_obs to store_transition
        """
        start_time = time.time()
        infos = {}

        # obs is a jax.Array from parent; split into privileged dict
        # On first call, obs comes from _get_initial_obs (already split).
        # On subsequent calls, obs comes from collection_data["last_obs"].
        last_obs = self._ensure_split_obs(obs)

        for step in range(self.num_steps_per_env):
            actor_obs = last_obs["actor"]

            # Warmup: random actions
            if self.total_timesteps < self.cfgs.algorithm.learning_starts:
                self.key, subkey = jax.random.split(self.key)
                warmup_std = getattr(self.cfgs.algorithm, "warmup_std", 1.0)
                actions = jax.random.normal(
                    subkey,
                    shape=(self.env.num_envs, self.env.num_actions),
                ) * warmup_std
                actions = jnp.clip(actions, -1.0, 1.0)
            else:
                # Target policy on s- (MPPI or pi)
                actions = self.alg.act_with_t0(
                    actor_obs,
                    t0_mask=self.env.reset_buf.cpu().numpy(),
                    eval_mode=False,
                )

            # Process and step
            actions_for_env = self._process_action_for_env(actions)
            actions_torch = jax_to_torch(actions_for_env, self.device)

            obs_dict, rewards, terminated, truncated, infos = self.env.step(actions_torch)

            # Split next obs
            next_obs = self._split_obs_from_env(obs_dict)
            next_actor_obs = next_obs["actor"]

            rewards_jax = torch_to_jax(rewards)
            terminated_jax = jnp.asarray(terminated.cpu().numpy())
            truncated_jax = jnp.asarray(truncated.cpu().numpy())

            # Handle final_observation at episode boundaries
            next_actor_for_buffer = next_actor_obs
            next_priv_for_buffer = next_obs["privileged"]
            final_obs = infos.get("final_observation")
            if final_obs is not None:
                final_split = self._split_obs_from_env(final_obs)
                mask = (terminated | truncated).cpu().numpy()
                for i in range(self.env.num_envs):
                    if mask[i]:
                        next_actor_for_buffer = next_actor_for_buffer.at[i].set(
                            final_split["actor"][i]
                        )
                        next_priv_for_buffer = next_priv_for_buffer.at[i].set(
                            final_split["privileged"][i]
                        )

            # Store with privileged obs
            self.alg.store_transition(
                obs=actor_obs,
                action=actions,
                reward=rewards_jax,
                next_obs=next_actor_for_buffer,
                terminated=terminated_jax,
                truncated=truncated_jax,
                privileged_obs=last_obs["privileged"],
                next_privileged_obs=next_priv_for_buffer,
            )

            # Update reward statistics
            dones = terminated | truncated
            self.reward_statistics.update(
                reward_info=infos["rewards_per_type"],
                dones=dones,
                success=infos.get("success", None),
            )

            # Continue with reset obs
            last_obs = next_obs

        # Build collection data (same structure as parent)
        from copy import deepcopy
        cur_return = self.reward_statistics.get_current_returns()
        return_buffer = self.reward_statistics.get_returns_buffer()
        length_buffer = self.reward_statistics.get_length_buffer()
        reward_breakdown_stats = self.reward_statistics.get_reward_stats_per_type()

        collection_data = dict(infos)
        collection_data.update({
            "cur_return": cur_return,
            "return_buffer": deepcopy(return_buffer),
            "length_buffer": deepcopy(length_buffer),
            "reward_breakdown_stats": deepcopy(reward_breakdown_stats),
            "success_rate": self.reward_statistics.get_success_rate(),
            "contact_force": infos.get("recent_contact_force", None),
            "collection_time": time.time() - start_time,
            "last_obs": last_obs,  # dict: {"actor": ..., "privileged": ...}
        })

        return collection_data

    # ------------------------------------------------------------------
    # Override 5: Initial observation
    # ------------------------------------------------------------------

    def _get_initial_obs(self) -> Dict[str, jax.Array]:
        """
        Get initial observation as split dict.

        Returns:
            {"actor": jax.Array, "privileged": jax.Array}
        """
        obs_dict = self.env.obs_manager.get_observation()
        return self._split_obs_from_env(obs_dict)

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def _split_obs_from_env(self, obs_dict) -> Dict[str, jax.Array]:
        """
        Split env observation dict into actor and privileged jax arrays.

        Expects obs_dict to have "actor" key (torch.Tensor).
        Privileged obs comes from "privileged" key if present,
        otherwise splits the actor obs at self.obs_dim boundary.
        """
        actor = torch_to_jax(obs_dict["actor"])

        if "privileged" in obs_dict:
            privileged = torch_to_jax(obs_dict["privileged"])
        else:
            # Fallback: split flat obs
            privileged = actor[..., self.obs_dim:]
            actor = actor[..., :self.obs_dim]

        return {"actor": actor, "privileged": privileged}

    def _ensure_split_obs(self, obs) -> Dict[str, jax.Array]:
        """Ensure obs is a split dict. Pass through if already split."""
        if isinstance(obs, dict):
            return obs
        # Fallback: flat jax array
        return {
            "actor": obs[..., :self.obs_dim],
            "privileged": obs[..., self.obs_dim:],
        }

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _build_wandb_extra(self, metrics) -> Dict[str, float]:
        """Extract scaffolding-specific metrics for wandb logging."""
        extra = {}
        if metrics is None:
            return extra

        scaff_fields = [
            "scaff_consistency_loss", "scaff_reward_loss",
            "scaff_value_loss", "scaff_total_loss",
            "explore_entropy",
            "target_ortho_loss", "scaff_ortho_loss",
        ]
        for field in scaff_fields:
            val = getattr(metrics, field, None)
            if val is not None:
                extra[f"scaffolding/{field}"] = float(val)

        return extra

