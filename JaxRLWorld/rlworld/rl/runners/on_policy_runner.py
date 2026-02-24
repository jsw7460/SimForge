import os
import pickle
import time
from copy import deepcopy
from typing import Dict, List, Any

import jax
import jax.numpy as jnp
import numpy as np
import torch

from rlworld.rl.algorithms.ppo import PPO
from rlworld.rl.algorithms.ppo_dr3 import PPODR3
from rlworld.rl.configs import ConfigsForRun, configs_from_dict
from rlworld.rl.configs.algorithms import PPOConfig, PPODR3Config
from rlworld.rl.envs import World
from rlworld.rl.envs.utils import LearningIterationObserver
from rlworld.rl.modules.policies.ppo_ac import PPOActorCritic
from rlworld.rl.modules.policies.ppo_dr3_ac import PPODR3ActorCritic
from rlworld.rl.modules.utils import print_model_summary, count_parameters
from rlworld.rl.runners.base_runner import BaseRunner
from rlworld.rl.utils.jax_utils import torch_to_jax, jax_to_torch


class OnPolicyRunner(BaseRunner):
    """
    On-policy runner using JAX PPO with PyTorch environment.

    Features:
    - Rollout storage for experience collection
    - GAE advantage estimation
    - Checkpoint save/load support
    """

    alg: PPO | PPODR3
    actor_critic: PPOActorCritic | PPODR3ActorCritic
    is_distributed_runner: bool = False

    def __init__(
        self,
        env: World,
        cfgs: ConfigsForRun,
        use_wandb: bool = True,
        seed: int = 0,
    ):
        """Initialize the runner with environment and configuration."""
        self.algorithm_name = cfgs.algorithm.algorithm_name
        super().__init__(env=env, cfgs=cfgs, use_wandb=use_wandb, seed=seed)

    def _init_training_modules(self) -> None:
        """Initialize actor-critic model based on algorithm type."""
        obs_dim = self.env.calculate_obs_dim()
        self.actor_obs_dim = obs_dim["actor"]
        self.critic_obs_dim = obs_dim["critic"]
        self.num_actions_dim = self.env.num_actions

        policy_cfg = self.cfgs.nn.policy
        actor_kwargs = policy_cfg.get("actor_kwargs", {})
        critic_kwargs = policy_cfg.get("critic_kwargs", {})

        self.key, subkey = jax.random.split(self.key)

        if self.algorithm_name == "PPODR3":
            self._init_ppodr3_actor_critic(policy_cfg, actor_kwargs, critic_kwargs, subkey)
        else:  # PPO
            self._init_ppo_actor_critic(policy_cfg, actor_kwargs, critic_kwargs, subkey)

        self.training_modules = {"actor_critic": self.actor_critic}

        self.squash_output = self.actor_critic.is_squashed
        self._init_action_scaling()

        # Print model info
        model_name = "PPODR3ActorCritic" if self.algorithm_name == "PPODR3" else "PPOActorCritic"
        print_model_summary(self.actor_critic, model_name)

        if self.use_wandb:
            self._log_model_parameters()

    def _init_algorithm(self) -> PPO | PPODR3:
        """Initialize algorithm based on type."""
        alg_cfg = self.cfgs.algorithm

        self.key, subkey = jax.random.split(self.key)

        if self.algorithm_name == "PPODR3":
            self.alg = self._init_ppodr3_algorithm(alg_cfg, subkey)
        else:  # PPO
            self.alg = self._init_ppo_algorithm(alg_cfg, subkey)

        return self.alg

    def _init_ppo_algorithm(self, alg_cfg: PPOConfig, key: jax.Array) -> PPO:
        """Initialize PPO algorithm."""
        return PPO(
            actor_critic=self.actor_critic,
            num_learning_epochs=alg_cfg.num_learning_epochs,
            num_mini_batches=alg_cfg.num_mini_batches,
            clip_param=alg_cfg.clip_param,
            gamma=alg_cfg.gamma,
            lam=alg_cfg.lam,
            value_loss_coef=alg_cfg.value_loss_coef,
            entropy_coef=alg_cfg.entropy_coef,
            actor_lr=alg_cfg.actor_lr,
            critic_lr=alg_cfg.critic_lr,
            max_grad_norm=alg_cfg.max_grad_norm,
            use_clipped_value_loss=alg_cfg.use_clipped_value_loss,
            schedule=alg_cfg.schedule,
            desired_kl=alg_cfg.desired_kl,
            use_reward_scaling=alg_cfg.use_reward_scaling,
            use_early_stop=alg_cfg.use_early_stop,
            key=key,
        )

    def _init_ppodr3_algorithm(self, alg_cfg: PPODR3Config, key: jax.Array) -> PPODR3:
        """Initialize PPO-DR3 algorithm."""
        return PPODR3(
            actor_critic=self.actor_critic,
            num_learning_epochs=alg_cfg.num_learning_epochs,
            num_mini_batches=alg_cfg.num_mini_batches,
            clip_param=alg_cfg.clip_param,
            gamma=alg_cfg.gamma,
            lam=alg_cfg.lam,
            value_loss_coef=alg_cfg.value_loss_coef,
            entropy_coef=alg_cfg.entropy_coef,
            dr3_coef=alg_cfg.dr3_coef,
            actor_lr=alg_cfg.actor_lr,
            critic_lr=alg_cfg.critic_lr,
            max_grad_norm=alg_cfg.max_grad_norm,
            use_clipped_value_loss=alg_cfg.use_clipped_value_loss,
            schedule=alg_cfg.schedule,
            desired_kl=alg_cfg.desired_kl,
            use_reward_scaling=alg_cfg.use_reward_scaling,
            use_early_stop=alg_cfg.use_early_stop,
            key=key,
        )

    def _init_ppo_actor_critic(
        self,
        policy_cfg: Dict,
        actor_kwargs: Dict,
        critic_kwargs: Dict,
        key: jax.Array,
    ) -> None:
        """Initialize PPO actor-critic."""

        if hasattr(self.env, "scene_manager"):
            kinematic_tree = self.env.scene_manager.trees.get("robot", None)
        else:
            kinematic_tree = None

        self.actor_critic = PPOActorCritic(
            num_actor_obs=self.actor_obs_dim,
            num_critic_obs=self.critic_obs_dim,
            num_actions=self.num_actions_dim,
            actor_class_name=policy_cfg["actor_class_name"],
            init_noise_std=policy_cfg["init_noise_std"],
            std_type=policy_cfg["std_type"],
            distribution_type=policy_cfg["distribution_type"],
            kinematic_tree=kinematic_tree,
            key=key,
            actor_kwargs=actor_kwargs,
            critic_kwargs=critic_kwargs,
            obs_normalization=self.cfgs.algorithm.obs_normalization
        )

    def _init_ppodr3_actor_critic(
        self,
        policy_cfg: Dict,
        actor_kwargs: Dict,
        critic_kwargs: Dict,
        key: jax.Array,
    ) -> None:
        """Initialize PPO-DR3 actor-critic with DR3Critic."""

        if hasattr(self.env, "scene_manager"):
            kinematic_tree = self.env.scene_manager.trees.get("robot", None)
        else:
            kinematic_tree = None

        self.actor_critic = PPODR3ActorCritic(
            num_actor_obs=self.actor_obs_dim,
            num_critic_obs=self.critic_obs_dim,
            num_actions=self.num_actions_dim,
            actor_class_name=policy_cfg["actor_class_name"],
            init_noise_std=policy_cfg["init_noise_std"],
            std_type=policy_cfg["std_type"],
            distribution_type=policy_cfg["distribution_type"],
            kinematic_tree=kinematic_tree,
            key=key,
            actor_kwargs=actor_kwargs,
            critic_kwargs=critic_kwargs,
        )

    def _log_model_parameters(self) -> None:
        """Log model parameters to wandb."""
        import wandb

        actor_params = count_parameters(self.actor_critic.actor)
        critic_params = count_parameters(self.actor_critic.critic)
        std_params = count_parameters(self.actor_critic.std_module)

        wandb.summary["model/actor_parameters"] = actor_params
        wandb.summary["model/critic_parameters"] = critic_params
        wandb.summary["model/std_parameters"] = std_params
        wandb.summary["model/total_parameters"] = actor_params + critic_params + std_params

        if self.algorithm_name == "PPODR3":
            wandb.summary["model/critic_feature_dim"] = self.actor_critic.critic_feature_dim

    def _init_storage(self):
        """Initialize the experience storage."""
        obs_dim = self.env.calculate_obs_dim()
        cfg = {
            "num_envs": self.env.num_envs,
            "num_transitions_per_env": self.cfgs.algorithm.num_steps_per_env,
            "actor_obs_shape": [obs_dim["actor"]],
            "critic_obs_shape": [obs_dim["critic"]],
            "actions_shape": [self.env.num_actions],
            "robot_state_shape": [obs_dim.get("robot_state", 0)],
            "estimator_obs_shape": [obs_dim.get("estimator", 0)],
        }
        self.alg.init_storage(cfg)

    def _get_initial_obs(self) -> PPO.ActInput:
        """Get initial observation as JAX arrays."""
        obs = self.env.get_observation()
        actor_obs = torch_to_jax(obs["actor"])
        critic_obs = torch_to_jax(obs["critic"])
        return PPO.ActInput(actor_obs, critic_obs)

    def _collect_experience(
        self,
        obs: PPO.ActInput,
        ep_infos: List[Dict],
    ) -> Dict[str, Any]:
        """Collect experience from the environment."""
        start_time = time.time()

        actor_obs = obs.actor_obs
        critic_obs = obs.critic_obs
        infos = {}

        for _step_i in range(self.num_steps_per_env):

            # Get action
            actions = self.alg.act(PPO.ActInput(actor_obs, critic_obs))
            actions_for_env = self._process_action_for_env(actions)
            actions_torch = jax_to_torch(actions_for_env, self.device)

            # Environment step
            obs_dict, rewards, terminated, truncated, infos = self.env.step(actions_torch)
            dones = terminated | truncated

            # Convert to JAX
            actor_obs = torch_to_jax(obs_dict["actor"])
            critic_obs = torch_to_jax(obs_dict["critic"])
            rewards_jax = torch_to_jax(rewards)
            # NOTE: DO NOT USE DLPACK HERE. DLPACK DOESN'T SUPPORT BOOLEAN
            terminated_jax = jnp.asarray(terminated.cpu().numpy())
            truncated_jax = jnp.asarray(truncated.cpu().numpy())

            # Process step
            infos_jax = {}
            if infos.get("final_observation") is not None:
                infos_jax["final_observation"] = {
                    "actor": torch_to_jax(infos["final_observation"]["actor"]),
                    "critic": torch_to_jax(infos["final_observation"]["critic"]),
                }
            self.alg.process_env_step(
                rewards_jax,
                terminated_jax,
                truncated_jax,
                infos_jax,
                next_actor_obs=actor_obs,
                next_critic_obs=critic_obs
            )

            # Update statistics
            self.reward_statistics.update(
                reward_info=infos["rewards_per_type"],
                dones=dones,
                success=infos.get("success", None),
            )

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
            "last_obs": {
                "actor_obs": actor_obs,
                "critic_obs": critic_obs,
            },
        })

        return collection_data

    def _run_training_iteration(
        self,
        obs: PPO.ActInput,
        iteration: int,
        ep_infos: List[Dict] = None,
    ) -> Dict[str, Any]:
        """Execute a single training iteration."""
        # Collect experience
        collection_data = self._collect_experience(obs=obs, ep_infos=ep_infos)

        # Update policy
        start_time = time.time()
        self.alg.compute_returns(collection_data["last_obs"]["critic_obs"])
        action_stats = self._get_action_statistics()

        metrics = self.alg.update()
        training_data = metrics.to_full_dict()
        training_data["metrics"] = metrics

        training_data.update(collection_data)
        learning_time = time.time() - start_time

        fps = (self.num_steps_per_env * self.env.num_envs) / (
            collection_data["collection_time"] + learning_time
        )
        reward_stats = self.reward_statistics.get_reward_stats_per_type()

        # Get current std for logging
        sample_obs = collection_data["last_obs"]["actor_obs"][:1]
        current_std = float(self.alg.model.std_module(sample_obs).mean())

        training_data.update({
            "iteration": iteration,
            "total_timesteps": self.total_timesteps,
            "learning_time": learning_time,
            "collection_time": collection_data["collection_time"],
            "action_std": current_std,
            "action_mean": None,
            "fps": fps,
            "reward_stats": reward_stats,
            "action_distribution": action_stats,
        })

        return training_data

    def learn(
        self,
        num_learning_iterations: int,
        init_at_random_ep_len: bool = False,
    ):
        """Main training loop."""
        # Initialize random episode length
        if init_at_random_ep_len:
            self.env.termination_manager.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # Training state
        obs = self._get_initial_obs()
        ep_infos: List[Dict] = []

        # Main training loop
        total_iter = self.current_learning_iteration + num_learning_iterations
        self.initial_learning_iteration = self.current_learning_iteration

        for it in range(self.initial_learning_iteration, total_iter + 1):
            self.it = it
            training_data = self._run_training_iteration(
                obs=obs,
                iteration=it,
                ep_infos=ep_infos,
            )
            LearningIterationObserver.on_iteration_update(self.it)

            # Update obs
            obs = PPO.ActInput(
                actor_obs=training_data["last_obs"]["actor_obs"],
                critic_obs=training_data["last_obs"]["critic_obs"],
            )
            self.post_iteration(training_data, total_iter, it)

    def _get_action_statistics(self) -> Dict[str, Any]:
        """Extract from rollout storage."""
        actions = np.array(self.alg.storage._storage["actions"])  # (num_steps, num_envs, action_dim)
        return self._compute_action_distribution_stats(actions)

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_path: str,
        cfgs: ConfigsForRun = None,
        env: World = None,
        use_wandb: bool = True,
    ) -> "OnPolicyRunner":
        """
        Load runner from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint directory
            cfgs: Configuration (if None, load from checkpoint metadata)
            env: Environment (if None, create from config)
            use_wandb: Whether to use WandB logging

        Returns:
            Loaded OnPolicyRunner instance
        """
        # Load metadata
        metadata_path = os.path.join(checkpoint_path, "metadata.pkl")
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)

        # Use saved config if not provided
        if cfgs is None:
            cfgs = configs_from_dict(metadata["config"])

        # Create env if not provided
        if env is None:
            env = cls._create_env_from_config(cfgs)

        # Create runner (this initializes fresh model and algorithm)
        runner = cls(env=env, cfgs=cfgs, use_wandb=use_wandb)

        # Delegate model loading to algorithm
        runner.alg.load_train_state(checkpoint_path, metadata)

        # Restore runner state
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
        print(f"  Total time: {runner.total_time:.2f}s")
        print(f"  Note: Optimizer state re-initialized (momentum reset)")

        return runner
