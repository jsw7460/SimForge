# """
# Model-Based Runner for model-based RL algorithms.
#
# Supports TD-MPC2 and other MBRL algorithms that use:
# - Sequence replay buffers
# - Planning-based action selection
# - World model + policy joint training
#
# Follows the same runner pattern as OnPolicyRunner and OffPolicyRunner.
# """
#
# import os
# import pickle
# import time
# from copy import deepcopy
# from typing import Dict, List, Any, Union
#
# import jax
# import jax.numpy as jnp
# import numpy as np
# import torch
#
# from rlworld.rl.algorithms.tdmpc2 import TDMPC2
# from rlworld.rl.configs import ConfigsForRun, configs_from_dict
# from rlworld.rl.envs import World
# from rlworld.rl.envs.utils import LearningIterationObserver
# from rlworld.rl.modules.policies.tdmpc2_world_model import TDMPC2WorldModel
# from rlworld.rl.modules.utils import print_model_summary, count_parameters
# from rlworld.rl.runners.base_runner import BaseRunner
# from rlworld.rl.utils.jax_utils import torch_to_jax, jax_to_torch
#
#
# class ModelBasedRunner(BaseRunner):
#     """
#     Runner for model-based RL algorithms (e.g., TD-MPC2).
#
#     Features:
#     - Sequence replay buffer for trajectory-chunk sampling
#     - Planning-based action selection (MPPI)
#     - Configurable UTD ratio
#     - Checkpoint save/load
#     """
#
#     alg: TDMPC2
#     is_distributed_runner: bool = False
#
#     def __init__(
#         self,
#         env: World,
#         cfgs: ConfigsForRun,
#         use_wandb: bool = True,
#         seed: int = 0,
#     ):
#         self.algorithm_name = cfgs.algorithm.algorithm_name
#         super().__init__(env=env, cfgs=cfgs, use_wandb=use_wandb, seed=seed)
#
#     # ==================== Initialization ====================
#
#     def _init_training_modules(self) -> None:
#         """Initialize world model."""
#         obs_dim = self.env.obs_manager.calculate_obs_dim()
#         self.obs_dim = obs_dim["actor"]  # TD-MPC2 uses single obs (no actor/critic split)
#         self.num_actions_dim = self.env.num_actions
#
#         alg_cfg = self.cfgs.algorithm
#         self.key, subkey = jax.random.split(self.key)
#
#         # Build world model
#         self.world_model = TDMPC2WorldModel(
#             obs_dim=self.obs_dim,
#             action_dim=self.num_actions_dim,
#             latent_dim=alg_cfg.latent_dim,
#             mlp_dim=alg_cfg.mlp_dim,
#             num_enc_layers=alg_cfg.num_enc_layers,
#             num_q=alg_cfg.num_q,
#             num_bins=alg_cfg.num_bins,
#             simnorm_dim=alg_cfg.simnorm_dim,
#             dropout=alg_cfg.dropout,
#             log_std_min=alg_cfg.log_std_min,
#             log_std_max=alg_cfg.log_std_max,
#             key=subkey,
#         )
#
#         self.training_modules = {"world_model": self.world_model}
#
#         # Action scaling
#         self.squash_output = True  # TD-MPC2 outputs in [-1, 1]
#         self._init_action_scaling()
#
#         # Print model info
#         print_model_summary(self.world_model, "TDMPC2WorldModel")
#         if self.use_wandb:
#             self._log_model_parameters()
#
#     def _init_algorithm(self) -> TDMPC2:
#         """Initialize TD-MPC2 algorithm."""
#         alg_cfg = self.cfgs.algorithm
#         self.key, subkey = jax.random.split(self.key)
#
#         self.alg = TDMPC2(
#             world_model=self.world_model,
#             num_envs=self.env.num_envs,
#             gamma=alg_cfg.gamma,
#             episode_length=alg_cfg.episode_length,
#             discount_min=alg_cfg.discount_min,
#             discount_max=alg_cfg.discount_max,
#             discount_denom=alg_cfg.discount_denom,
#             lr=alg_cfg.lr,
#             pi_lr=alg_cfg.pi_lr,
#             tau=alg_cfg.tau,
#             mpc=alg_cfg.mpc,
#             horizon=alg_cfg.horizon,
#             num_samples=alg_cfg.num_samples,
#             num_pi_trajs=alg_cfg.num_pi_trajs,
#             num_elites=alg_cfg.num_elites,
#             num_iterations=alg_cfg.num_iterations,
#             temperature=alg_cfg.temperature,
#             min_std=alg_cfg.min_std,
#             max_std=alg_cfg.max_std,
#             consistency_coef=alg_cfg.consistency_coef,
#             reward_coef=alg_cfg.reward_coef,
#             value_coef=alg_cfg.value_coef,
#             entropy_coef=alg_cfg.entropy_coef,
#             rho=alg_cfg.rho,
#             num_bins=alg_cfg.num_bins,
#             vmin=alg_cfg.vmin,
#             vmax=alg_cfg.vmax,
#             batch_size=alg_cfg.batch_size,
#             grad_clip_norm=alg_cfg.grad_clip_norm,
#             max_grad_norm=alg_cfg.max_grad_norm,
#             key=subkey,
#         )
#
#         return self.alg
#
#     def _init_storage(self) -> None:
#         """Initialize sequence replay buffer."""
#         obs_dim = self.env.obs_manager.calculate_obs_dim()
#         alg_cfg = self.cfgs.algorithm
#         size_per_env = alg_cfg.get("buffer_size", 1_000_000) // self.env.num_envs
#
#         cfg = {
#             "num_envs": self.env.num_envs,
#             "obs_dim": obs_dim["actor"],
#             "action_dim": self.env.num_actions,
#             "size_per_env": size_per_env,
#         }
#         self.alg.init_storage(cfg)
#
#     def _log_model_parameters(self) -> None:
#         """Log model parameters to wandb."""
#         import wandb
#         total = count_parameters(self.world_model)
#         wandb.summary["model/total_parameters"] = total
#
#     # ==================== Data Collection ====================
#
#     def _get_initial_obs(self) -> jax.Array:
#         """Get initial observation as JAX array."""
#         obs = self.env.obs_manager.get_observation()
#         return torch_to_jax(obs["actor"])
#
#     def _collect_experience(
#         self,
#         obs: jax.Array,
#         ep_infos: List[Dict],
#         iteration: int,
#     ) -> Dict[str, Any]:
#         """Collect experience from the environment."""
#         start_time = time.time()
#         infos = {}
#         actor_obs = obs
#
#         for step in range(self.num_steps_per_env):
#             # Warmup: random actions
#             if self.total_timesteps < self.cfgs.algorithm.get("learning_starts", 0):
#                 self.key, subkey = jax.random.split(self.key)
#                 actions = jax.random.uniform(
#                     subkey,
#                     shape=(self.env.num_envs, self.env.num_actions),
#                     minval=-1.0, maxval=1.0,
#                 )
#             else:
#                 # Plan/act with MPPI or policy
#                 actions = self.alg.act_with_t0(
#                     actor_obs,
#                     t0_mask=self.env.reset_buf.cpu().numpy(),
#                     eval_mode=False,
#                 )
#
#             # Process action for environment
#             actions_for_env = self._process_action_for_env(actions)
#             actions_torch = jax_to_torch(actions_for_env, self.device)
#
#             # Environment step
#             obs_dict, rewards, terminated, truncated, infos = self.env.step(actions_torch)
#
#             # Convert to JAX
#             next_actor_obs = torch_to_jax(obs_dict["actor"])  # reset obs (post-autoreset)
#             rewards_jax = torch_to_jax(rewards)
#             # NOTE: DO NOT USE DLPACK HERE. DLPACK DOESN'T SUPPORT BOOLEAN
#             terminated_jax = jnp.asarray(terminated.cpu().numpy())
#             truncated_jax = jnp.asarray(truncated.cpu().numpy())
#
#             # For buffer: use final_observation at truncation boundaries
#             next_obs_for_buffer = next_actor_obs
#             final_obs = infos.get("final_observation")
#             if final_obs is not None:
#                 final_actor = torch_to_jax(final_obs["actor"])
#                 truncated_mask = truncated.cpu().numpy()
#                 terminated_mask = terminated.cpu().numpy()
#                 for i in range(self.env.num_envs):
#                     if (truncated_mask[i] or terminated_mask[i]) and final_obs is not None:
#                         next_obs_for_buffer = next_obs_for_buffer.at[i].set(final_actor[i])
#
#             # Store transition (buffer gets terminal obs, loop continues with reset obs)
#             self.alg.store_transition(
#                 obs=actor_obs,
#                 action=actions,
#                 reward=rewards_jax,
#                 next_obs=next_obs_for_buffer,       # Terminal observation
#                 terminated=terminated_jax,
#                 truncated=truncated_jax,
#             )
#
#             # Update reward statistics
#             dones = terminated | truncated
#             self.reward_statistics.update(
#                 reward_info=infos["rewards_per_type"],
#                 dones=dones,
#                 success=infos.get("success", None),
#             )
#
#             # Continue with reset obs for next step
#             actor_obs = next_actor_obs
#
#         # Collect statistics
#         cur_return = self.reward_statistics.get_current_returns()
#         return_buffer = self.reward_statistics.get_returns_buffer()
#         length_buffer = self.reward_statistics.get_length_buffer()
#         reward_breakdown_stats = self.reward_statistics.get_reward_stats_per_type()
#
#         collection_data = dict(infos)
#         collection_data.update({
#             "cur_return": cur_return,
#             "return_buffer": deepcopy(return_buffer),
#             "length_buffer": deepcopy(length_buffer),
#             "reward_breakdown_stats": deepcopy(reward_breakdown_stats),
#             "success_rate": self.reward_statistics.get_success_rate(),
#             "contact_force": infos.get("recent_contact_force", None),
#             "collection_time": time.time() - start_time,
#             "last_obs": actor_obs,
#         })
#
#         return collection_data
#
#     # ==================== Training ====================
#
#     def _run_training_iteration(
#         self,
#         obs: jax.Array,
#         iteration: int,
#         ep_infos: List[Dict] = None,
#     ) -> Dict[str, Any]:
#         """Execute a single training iteration."""
#         # Collect experience
#         collection_data = self._collect_experience(
#             obs=obs, ep_infos=ep_infos, iteration=iteration,
#         )
#
#         # Train if we have enough data
#         training_data = {"learning_time": 0.0}
#         batch_size = self.cfgs.algorithm.batch_size
#
#         min_buffer_size = max(
#             self.cfgs.algorithm.learning_starts,
#             batch_size,
#         )
#
#         if self.alg.replay_buffer.size >= min_buffer_size:
#             training_start = time.time()
#
#             # Pretrain on seed data (matches author: num_updates = seed_steps)
#             if not getattr(self, '_pretrained', False):
#                 num_updates = self.cfgs.algorithm.learning_starts // self.env.num_envs
#                 print(f'Pretraining agent on seed data ({num_updates} updates)...')
#                 self._pretrained = True
#             else:
#                 utd_ratio = self.cfgs.algorithm.utd_ratio
#                 num_updates = max(1, utd_ratio)
#
#             update_data = {}
#             for i in range(num_updates):
#                 self.key, subkey = jax.random.split(self.key)
#                 batch = self.alg.sample_batch(batch_size, subkey)
#                 is_last = (i == num_updates - 1)
#                 metrics = self.alg.update(batch, build_metrics=is_last)
#
#             if metrics is not None:
#                 update_data = metrics.to_full_dict()
#                 update_data["metrics"] = metrics
#
#             learning_time = time.time() - training_start
#             fps = (self.num_steps_per_env * self.env.num_envs) / (
#                 collection_data["collection_time"] + learning_time
#             )
#             reward_stats = self.reward_statistics.get_reward_stats_per_type()
#
#             training_data.update({
#                 "iteration": iteration,
#                 "learning_time": learning_time,
#                 "fps": fps,
#                 "buffer_size": self.alg.replay_buffer.size,
#                 "reward_stats": reward_stats,
#                 **update_data,
#             })
#
#             LearningIterationObserver.on_iteration_update(iteration)
#
#         return {
#             **collection_data,
#             **training_data,
#             "action_distribution": self._get_action_statistics(),
#         }
#
#     def learn(
#         self,
#         num_learning_iterations: int,
#         init_at_random_ep_len: bool = False,
#     ):
#         """Main training loop."""
#         if init_at_random_ep_len:
#             if hasattr(self.env, "termination_manager"):
#                 self.env.termination_manager.episode_length_buf = torch.randint_like(
#                     self.env.episode_length_buf, high=int(self.env.max_episode_length)
#                 )
#
#         obs = self._get_initial_obs()
#         ep_infos: List[Dict] = []
#
#         total_iter = self.current_learning_iteration + num_learning_iterations
#         self.initial_learning_iteration = self.current_learning_iteration
#
#         for it in range(self.initial_learning_iteration, total_iter + 1):
#             self.it = it
#
#             training_data = self._run_training_iteration(
#                 obs=obs, iteration=it, ep_infos=ep_infos,
#             )
#
#             obs = training_data["last_obs"]
#             self.post_iteration(training_data, total_iter, it)
#
#     def _get_action_statistics(self) -> Dict[str, Any]:
#         """Extract recent action statistics from replay buffer."""
#         n_recent = min(
#             self.num_steps_per_env * self.env.num_envs,
#             self.alg.replay_buffer.size,
#         )
#         if n_recent == 0:
#             return {}
#         actions = self.alg.replay_buffer.get_recent_actions(n_recent)
#         return self._compute_action_distribution_stats(np.array(actions))
#
#     # ==================== Checkpoint ====================
#
#     @classmethod
#     def load_checkpoint(
#         cls,
#         checkpoint_path: str,
#         cfgs: ConfigsForRun = None,
#         env: World = None,
#         use_wandb: bool = True,
#     ) -> "ModelBasedRunner":
#         """Load runner from checkpoint."""
#         metadata_path = os.path.join(checkpoint_path, "metadata.pkl")
#         with open(metadata_path, "rb") as f:
#             metadata = pickle.load(f)
#
#         if cfgs is None:
#             cfgs = configs_from_dict(metadata["config"])
#         if env is None:
#             env = cls._create_env_from_config(cfgs)
#
#         runner = cls(env=env, cfgs=cfgs, use_wandb=use_wandb)
#         runner.alg.load_train_state(checkpoint_path, metadata)
#
#         runner.current_learning_iteration = metadata.get(
#             "current_learning_iteration", metadata["iteration"]
#         )
#         runner.total_timesteps = metadata["total_timesteps"]
#         runner.total_time = metadata.get("total_time", 0)
#         runner.key = jnp.array(metadata["jax_key"])
#
#         print(f"Loaded checkpoint from {checkpoint_path}")
#         print(f"  Algorithm: {runner.algorithm_name}")
#         print(f"  Iteration: {runner.current_learning_iteration}")
#         print(f"  Timesteps: {runner.total_timesteps}")
#
#         return runner


"""
Model-Based Runner for model-based RL algorithms.

Supports TD-MPC2 and other MBRL algorithms that use:
- Sequence replay buffers
- Planning-based action selection
- World model + policy joint training

Follows the same runner pattern as OnPolicyRunner and OffPolicyRunner.
"""

import os
import pickle
import time
from copy import deepcopy
from typing import Dict, List, Any, Union

import jax
import jax.numpy as jnp
import numpy as np
import torch

from rlworld.rl.algorithms.tdmpc2 import TDMPC2
from rlworld.rl.configs import ConfigsForRun, configs_from_dict
from rlworld.rl.envs import World
from rlworld.rl.modules.policies.tdmpc2_world_model import TDMPC2WorldModel
from rlworld.rl.modules.utils import print_model_summary, count_parameters
from rlworld.rl.runners.base_runner import BaseRunner
from rlworld.rl.utils.jax_utils import torch_to_jax, jax_to_torch


class ModelBasedRunner(BaseRunner):
    """
    Runner for model-based RL algorithms (e.g., TD-MPC2).

    Features:
    - Sequence replay buffer for trajectory-chunk sampling
    - Planning-based action selection (MPPI)
    - Configurable UTD ratio
    - Checkpoint save/load
    """

    alg: TDMPC2
    is_distributed_runner: bool = False

    def __init__(
        self,
        env: World,
        cfgs: ConfigsForRun,
        use_wandb: bool = True,
        seed: int = 0,
    ):
        self.algorithm_name = cfgs.algorithm.algorithm_name
        super().__init__(env=env, cfgs=cfgs, use_wandb=use_wandb, seed=seed)

    # ==================== Initialization ====================

    def _init_training_modules(self) -> None:
        """Initialize world model."""
        obs_dim = self.env.obs_manager.calculate_obs_dim()
        self.obs_dim = obs_dim["actor"]  # TD-MPC2 uses single obs (no actor/critic split)
        self.num_actions_dim = self.env.num_actions

        alg_cfg = self.cfgs.algorithm
        self.key, subkey = jax.random.split(self.key)

        # Determine squash_action from config (default: True for backward compatibility)
        squash_action = alg_cfg.squash_action

        # Action bounds from environment (used when squash_action=False)
        action_low = self.env.action_low.cpu().numpy()
        action_high = self.env.action_high.cpu().numpy()

        # Build world model
        self.world_model = TDMPC2WorldModel(
            obs_dim=self.obs_dim,
            action_dim=self.num_actions_dim,
            latent_dim=alg_cfg.latent_dim,
            mlp_dim=alg_cfg.mlp_dim,
            num_enc_layers=alg_cfg.num_enc_layers,
            num_q=alg_cfg.num_q,
            num_bins=alg_cfg.num_bins,
            simnorm_dim=alg_cfg.simnorm_dim,
            dropout=alg_cfg.dropout,
            log_std_min=alg_cfg.log_std_min,
            log_std_max=alg_cfg.log_std_max,
            squash_action=squash_action,
            action_low=tuple(action_low.tolist()),
            action_high=tuple(action_high.tolist()),
            key=subkey,
        )

        self.training_modules = {"world_model": self.world_model}

        # Action scaling: driven by world model's squash_action setting
        # squash_action=True  -> squash_output=True  -> _process_action_for_env scales [-1,1] to env range
        # squash_action=False -> squash_output=False -> _process_action_for_env clips to env range
        self.squash_output = self.world_model.squash_action
        self._init_action_scaling()

        # Print model info
        print_model_summary(self.world_model, "TDMPC2WorldModel")
        if self.use_wandb:
            self._log_model_parameters()

    def _init_algorithm(self) -> TDMPC2:
        """Initialize TD-MPC2 algorithm."""
        alg_cfg = self.cfgs.algorithm
        self.key, subkey = jax.random.split(self.key)

        self.alg = TDMPC2(
            world_model=self.world_model,
            num_envs=self.env.num_envs,
            gamma=alg_cfg.gamma,
            episode_length=alg_cfg.episode_length,
            discount_min=alg_cfg.discount_min,
            discount_max=alg_cfg.discount_max,
            discount_denom=alg_cfg.discount_denom,
            lr=alg_cfg.lr,
            pi_lr=alg_cfg.pi_lr,
            tau=alg_cfg.tau,
            mpc=alg_cfg.mpc,
            horizon=alg_cfg.horizon,
            num_samples=alg_cfg.num_samples,
            num_pi_trajs=alg_cfg.num_pi_trajs,
            num_elites=alg_cfg.num_elites,
            num_iterations=alg_cfg.num_iterations,
            temperature=alg_cfg.temperature,
            min_std=alg_cfg.min_std,
            max_std=alg_cfg.max_std,
            consistency_coef=alg_cfg.consistency_coef,
            reward_coef=alg_cfg.reward_coef,
            value_coef=alg_cfg.value_coef,
            entropy_coef=alg_cfg.entropy_coef,
            rho=alg_cfg.rho,
            num_bins=alg_cfg.num_bins,
            vmin=alg_cfg.vmin,
            vmax=alg_cfg.vmax,
            batch_size=alg_cfg.batch_size,
            grad_clip_norm=alg_cfg.grad_clip_norm,
            max_grad_norm=alg_cfg.max_grad_norm,
            key=subkey,
        )

        return self.alg

    def _init_storage(self) -> None:
        """Initialize sequence replay buffer."""
        obs_dim = self.env.obs_manager.calculate_obs_dim()
        alg_cfg = self.cfgs.algorithm
        size_per_env = alg_cfg.get("buffer_size", 1_000_000) // self.env.num_envs

        cfg = {
            "num_envs": self.env.num_envs,
            "obs_dim": obs_dim["actor"],
            "action_dim": self.env.num_actions,
            "size_per_env": size_per_env,
        }
        self.alg.init_storage(cfg)

    def _log_model_parameters(self) -> None:
        """Log model parameters to wandb."""
        import wandb
        total = count_parameters(self.world_model)
        wandb.summary["model/total_parameters"] = total

    # ==================== Data Collection ====================

    def _get_initial_obs(self) -> jax.Array:
        """Get initial observation as JAX array."""
        obs = self.env.obs_manager.get_observation()
        return torch_to_jax(obs["actor"])

    def _collect_experience(
        self,
        obs: jax.Array,
        ep_infos: List[Dict],
        iteration: int,
    ) -> Dict[str, Any]:
        """Collect experience from the environment."""
        start_time = time.time()
        infos = {}
        actor_obs = obs

        for step in range(self.num_steps_per_env):
            # Warmup: random actions
            if self.total_timesteps < self.cfgs.algorithm.get("learning_starts", 0):
                self.key, subkey = jax.random.split(self.key)
                if self.world_model.squash_action:
                    # Original behavior: sample in [-1, 1] with scalar bounds
                    actions = jax.random.uniform(
                        subkey,
                        shape=(self.env.num_envs, self.env.num_actions),
                        minval=-1.0,
                        maxval=1.0,
                    )
                else:
                    # Raw action mode: sample in [action_low, action_high]
                    action_low = jnp.array(self.world_model.action_low_tuple)
                    action_high = jnp.array(self.world_model.action_high_tuple)
                    actions = jax.random.uniform(
                        subkey,
                        shape=(self.env.num_envs, self.env.num_actions),
                        minval=action_low,
                        maxval=action_high,
                    )
            else:
                # Plan/act with MPPI or policy
                actions = self.alg.act_with_t0(
                    actor_obs,
                    t0_mask=self.env.reset_buf.cpu().numpy(),
                    eval_mode=False,
                )

            # Process action for environment
            actions_for_env = self._process_action_for_env(actions)
            actions_torch = jax_to_torch(actions_for_env, self.device)

            # Environment step
            obs_dict, rewards, terminated, truncated, infos = self.env.step(actions_torch)

            # Convert to JAX
            next_actor_obs = torch_to_jax(obs_dict["actor"])  # reset obs (post-autoreset)
            rewards_jax = torch_to_jax(rewards)
            # NOTE: DO NOT USE DLPACK HERE. DLPACK DOESN'T SUPPORT BOOLEAN
            terminated_jax = jnp.asarray(terminated.cpu().numpy())
            truncated_jax = jnp.asarray(truncated.cpu().numpy())

            # For buffer: use final_observation at truncation boundaries
            next_obs_for_buffer = next_actor_obs
            final_obs = infos.get("final_observation")
            if final_obs is not None:
                final_actor = torch_to_jax(final_obs["actor"])
                truncated_mask = truncated.cpu().numpy()
                terminated_mask = terminated.cpu().numpy()
                for i in range(self.env.num_envs):
                    if (truncated_mask[i] or terminated_mask[i]) and final_obs is not None:
                        next_obs_for_buffer = next_obs_for_buffer.at[i].set(final_actor[i])

            # Store transition (buffer gets terminal obs, loop continues with reset obs)
            self.alg.store_transition(
                obs=actor_obs,
                action=actions,
                reward=rewards_jax,
                next_obs=next_obs_for_buffer,       # Terminal observation
                terminated=terminated_jax,
                truncated=truncated_jax,
            )

            # Update reward statistics
            dones = terminated | truncated
            self.reward_statistics.update(
                reward_info=infos["rewards_per_type"],
                dones=dones,
                success=infos.get("success", None),
            )

            # Continue with reset obs for next step
            actor_obs = next_actor_obs

        # Collect statistics
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
            "last_obs": actor_obs,
        })

        return collection_data

    # ==================== Training ====================

    def _run_training_iteration(
        self,
        obs: jax.Array,
        iteration: int,
        ep_infos: List[Dict] = None,
    ) -> Dict[str, Any]:
        """Execute a single training iteration."""
        # Collect experience
        collection_data = self._collect_experience(
            obs=obs, ep_infos=ep_infos, iteration=iteration,
        )

        # Train if we have enough data
        training_data = {"learning_time": 0.0}
        batch_size = self.cfgs.algorithm.batch_size

        min_buffer_size = max(
            self.cfgs.algorithm.learning_starts,
            batch_size,
        )

        if self.alg.replay_buffer.size >= min_buffer_size:
            training_start = time.time()

            # Pretrain on seed data (matches author: num_updates = seed_steps)
            if not getattr(self, '_pretrained', False):
                num_updates = self.cfgs.algorithm.learning_starts // self.env.num_envs
                print(f'Pretraining agent on seed data ({num_updates} updates)...')
                self._pretrained = True
            else:
                utd_ratio = self.cfgs.algorithm.utd_ratio
                num_updates = max(1, utd_ratio)

            update_data = {}
            for i in range(num_updates):
                self.key, subkey = jax.random.split(self.key)
                batch = self.alg.sample_batch(batch_size, subkey)
                is_last = (i == num_updates - 1)
                metrics = self.alg.update(batch, build_metrics=is_last)

            if metrics is not None:
                update_data = metrics.to_full_dict()
                update_data["metrics"] = metrics

            learning_time = time.time() - training_start
            fps = (self.num_steps_per_env * self.env.num_envs) / (
                collection_data["collection_time"] + learning_time
            )
            reward_stats = self.reward_statistics.get_reward_stats_per_type()

            training_data.update({
                "iteration": iteration,
                "learning_time": learning_time,
                "fps": fps,
                "buffer_size": self.alg.replay_buffer.size,
                "reward_stats": reward_stats,
                **update_data,
            })

        return {
            **collection_data,
            **training_data,
            "action_distribution": self._get_action_statistics(),
        }

    def learn(
        self,
        num_learning_iterations: int,
        init_at_random_ep_len: bool = False,
    ):
        """Main training loop."""
        if init_at_random_ep_len:
            if hasattr(self.env, "termination_manager"):
                self.env.termination_manager.episode_length_buf = torch.randint_like(
                    self.env.episode_length_buf, high=int(self.env.max_episode_length)
                )

        obs = self._get_initial_obs()
        ep_infos: List[Dict] = []

        total_iter = self.current_learning_iteration + num_learning_iterations
        self.initial_learning_iteration = self.current_learning_iteration

        for it in range(self.initial_learning_iteration, total_iter + 1):
            self.it = it

            training_data = self._run_training_iteration(
                obs=obs, iteration=it, ep_infos=ep_infos,
            )

            obs = training_data["last_obs"]
            self.post_iteration(training_data, total_iter, it)

    def _get_action_statistics(self) -> Dict[str, Any]:
        """Extract recent action statistics from replay buffer."""
        n_recent = min(
            self.num_steps_per_env * self.env.num_envs,
            self.alg.replay_buffer.size,
        )
        if n_recent == 0:
            return {}
        actions = self.alg.replay_buffer.get_recent_actions(n_recent)
        return self._compute_action_distribution_stats(np.array(actions))

    # ==================== Checkpoint ====================

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_path: str,
        cfgs: ConfigsForRun = None,
        env: World = None,
        use_wandb: bool = True,
    ) -> "ModelBasedRunner":
        """Load runner from checkpoint."""
        metadata_path = os.path.join(checkpoint_path, "metadata.pkl")
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)

        if cfgs is None:
            cfgs = configs_from_dict(metadata["config"])
        if env is None:
            env = cls._create_env_from_config(cfgs)

        runner = cls(env=env, cfgs=cfgs, use_wandb=use_wandb)
        runner.alg.load_train_state(checkpoint_path, metadata)

        runner.current_learning_iteration = metadata.get(
            "current_learning_iteration", metadata["iteration"]
        )
        runner.total_timesteps = metadata["total_timesteps"]
        runner.total_time = metadata.get("total_time", 0)
        runner.key = jnp.array(metadata["jax_key"])

        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"  Algorithm: {runner.algorithm_name}")
        print(f"  Iteration: {runner.current_learning_iteration}")
        print(f"  Timesteps: {runner.total_timesteps}")

        return runner