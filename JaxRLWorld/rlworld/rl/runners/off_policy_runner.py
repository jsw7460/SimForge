import os

import time
from typing import Dict, List, Any, Union

import jax
import jax.numpy as jnp
import numpy as np
import torch

from rlworld.rl.algorithms.base import ActInput
from rlworld.rl.algorithms import SAC, TD3, FastTD3
from rlworld.rl.configs.algorithms import TD3Config, FastTD3Config, SACConfig
from rlworld.rl.configs import ConfigsForRun, configs_from_dict
from rlworld.rl.envs import World
from rlworld.rl.modules.policies.sac_ac import SACActorCritic
from rlworld.rl.modules.policies.td3_ac import TD3ActorCritic
from rlworld.rl.modules.policies.fast_td3_ac import FastTD3ActorCritic
from rlworld.rl.modules.utils import print_model_summary, count_parameters
from rlworld.rl.runners.base_runner import BaseRunner
from rlworld.rl.runners.iteration_data import IterationData
from rlworld.rl.utils.jax_utils import torch_to_jax, jax_to_torch


class OffPolicyRunner(BaseRunner):
    """
    Off-policy runner supporting SAC and TD3 algorithms.

    Features:
    - Replay buffer management
    - Configurable update-to-data (UTD) ratio
    - Checkpoint save/load support
    - Supports both SAC (stochastic) and TD3 (deterministic) policies
    """

    alg: SAC | TD3 | FastTD3
    actor_critic: SACActorCritic | TD3ActorCritic | FastTD3ActorCritic
    is_distributed_runner: bool = False

    def __init__(
        self,
        env: World,
        cfgs: ConfigsForRun,
        use_wandb: bool = True,
        seed: int = 0,
    ):
        """Initialize the runner with environment and configuration."""
        # Determine algorithm type
        self.algorithm_name = cfgs.algorithm.algorithm_name
        super().__init__(env=env, cfgs=cfgs, use_wandb=use_wandb, seed=seed)

    def _init_training_modules(self) -> None:
        """Initialize actor-critic model based on algorithm type."""
        obs_dim = self.env.obs_manager.calculate_obs_dim()
        self.actor_obs_dim = obs_dim["actor"]
        self.critic_obs_dim = obs_dim["critic"]
        self.num_actions_dim = self.env.num_actions

        policy_cfg = self.cfgs.nn.policy

        self.key, subkey = jax.random.split(self.key)

        if self.algorithm_name == "SAC":
            self._init_sac_actor_critic(policy_cfg, subkey)
        elif self.algorithm_name == "TD3":
            self._init_td3_actor_critic(policy_cfg, subkey)
        elif self.algorithm_name == "FastTD3":
            self._init_fast_td3_actor_critic(policy_cfg, subkey)
        else:
            raise NotImplementedError(f"Unknown algorithm: {self.algorithm_name}")

        self.training_modules = {"actor_critic": self.actor_critic}

        self.squash_output = self.actor_critic.is_squashed
        self._init_action_scaling()

        # Print model info
        model_name = "SACActorCritic" if self.algorithm_name == "SAC" else "TD3ActorCritic"
        print_model_summary(self.actor_critic, model_name)

        if self.use_wandb:
            self._log_model_parameters()

    def _init_fast_td3_actor_critic(self, policy_cfg, key: jax.Array) -> None:
        """Initialize FastTD3 actor-critic with distributional critics."""
        alg_cfg = self.cfgs.algorithm

        self.actor_critic = FastTD3ActorCritic(
            num_actor_obs=self.actor_obs_dim,
            num_critic_obs=self.critic_obs_dim,
            num_actions=self.num_actions_dim,
            num_atoms=alg_cfg.num_atoms,
            v_min=alg_cfg.v_min,
            v_max=alg_cfg.v_max,
            actor_class_name=policy_cfg.actor_class_name,
            kinematic_tree=self.env.scene_manager.trees.get("robot", None),
            key=key,
            is_squashed=alg_cfg.is_squashed,
            actor_kwargs=policy_cfg.actor_kwargs,
            critic_kwargs=policy_cfg.critic_kwargs,
            obs_normalization=alg_cfg.obs_normalization,
        )

    def _init_sac_actor_critic(self, policy_cfg, key: jax.Array) -> None:
        """Initialize SAC actor-critic."""
        self.actor_critic = SACActorCritic(
            num_actor_obs=self.actor_obs_dim,
            num_critic_obs=self.critic_obs_dim,
            num_actions=self.num_actions_dim,
            actor_class_name=policy_cfg.actor_class_name,
            distribution_type=policy_cfg.distribution_type,
            init_noise_std=policy_cfg.init_noise_std,
            log_std_min=policy_cfg.log_std_min,
            log_std_max=policy_cfg.log_std_max,
            kinematic_tree=self.env.scene_manager.trees.get("robot", None),
            obs_normalization=self.cfgs.algorithm.obs_normalization,
            key=key,
            actor_kwargs=policy_cfg.actor_kwargs,
            critic_kwargs=policy_cfg.critic_kwargs,
        )

    def _init_td3_actor_critic(self, policy_cfg, key: jax.Array) -> None:
        """Initialize TD3 actor-critic."""
        self.actor_critic = TD3ActorCritic(
            num_actor_obs=self.actor_obs_dim,
            num_critic_obs=self.critic_obs_dim,
            num_actions=self.num_actions_dim,
            actor_class_name=policy_cfg.actor_class_name,
            kinematic_tree=self.env.scene_manager.trees.get("robot", None),
            obs_normalization=self.cfgs.algorithm.obs_normalization,
            key=key,
            actor_kwargs=policy_cfg.actor_kwargs,
            critic_kwargs=policy_cfg.critic_kwargs,
        )

    def _log_model_parameters(self) -> None:
        """Log model parameters to wandb."""
        import wandb

        actor_params = count_parameters(self.actor_critic.actor)
        critic1_params = count_parameters(self.actor_critic.critic1)
        critic2_params = count_parameters(self.actor_critic.critic2)

        wandb.summary["model/actor_parameters"] = actor_params
        wandb.summary["model/critic1_parameters"] = critic1_params
        wandb.summary["model/critic2_parameters"] = critic2_params

        # SAC has additional log_std_net
        if self.algorithm_name == "SAC":
            log_std_params = count_parameters(self.actor_critic.log_std_net)
            wandb.summary["model/log_std_parameters"] = log_std_params
            wandb.summary["model/total_parameters"] = (
                actor_params + critic1_params + critic2_params + log_std_params
            )
        else:
            wandb.summary["model/total_parameters"] = (
                actor_params + critic1_params + critic2_params
            )

    def _init_algorithm(self) -> Union[SAC, TD3]:
        """Initialize algorithm based on type."""
        alg_cfg = self.cfgs.algorithm

        self.key, subkey = jax.random.split(self.key)

        if self.algorithm_name == "SAC":
            self.alg = self._init_sac_algorithm(alg_cfg, subkey)
        elif self.algorithm_name == "TD3":
            self.alg = self._init_td3_algorithm(alg_cfg, subkey)
        elif self.algorithm_name == "FastTD3":
            self.alg = self._init_fast_td3_algorithm(alg_cfg, subkey)
        else:
            raise NotImplementedError(f"Unknown algorithm: {self.algorithm_name}")

        # Compute training info
        num_gradient_steps = alg_cfg.get("num_gradient_steps", 1)
        num_transitions = self.env.num_envs * self.cfgs.algorithm.num_steps_per_env

        print(f"\n🔧 JAX {self.algorithm_name} initialized")
        print(f"  Learning rate: actor={alg_cfg.actor_lr}, critic={alg_cfg.critic_lr}")
        print(f"  Tau: {alg_cfg.tau}")
        print(f"  Gamma: {alg_cfg.gamma}")
        print(f"  Batch size: {alg_cfg.batch_size}")
        print(f"\n  Training config:")
        print(f"    - num_envs: {self.env.num_envs}")
        print(f"    - num_steps_per_env: {self.cfgs.algorithm.num_steps_per_env}")
        print(f"    - transitions per iter: {num_transitions}")
        print(f"    - gradient steps per iter: {num_gradient_steps}")

        return self.alg

    def _init_sac_algorithm(self, alg_cfg: SACConfig, key: jax.Array) -> SAC:
        """Initialize SAC algorithm."""
        return SAC(
            actor_critic=self.actor_critic,
            actor_lr=alg_cfg.actor_lr,
            critic_lr=alg_cfg.critic_lr,
            alpha_lr=alg_cfg.get("alpha_lr", 3e-4),
            gamma=alg_cfg.gamma,
            tau=alg_cfg.tau,
            batch_size=alg_cfg.batch_size,
            ent_coef=alg_cfg.get("ent_coef", "auto"),
            target_entropy=alg_cfg.get("target_entropy", "auto"),
            policy_delay=alg_cfg.get("policy_delay", 1),
            max_grad_norm=alg_cfg.get("max_grad_norm", 10.0),
            key=key,
        )

    def _init_td3_algorithm(self, alg_cfg: TD3Config, key: jax.Array) -> TD3:
        """Initialize TD3 algorithm."""
        return TD3(
            actor_critic=self.actor_critic,
            actor_lr=alg_cfg.actor_lr,
            critic_lr=alg_cfg.critic_lr,
            gamma=alg_cfg.gamma,
            tau=alg_cfg.tau,
            batch_size=alg_cfg.batch_size,
            policy_delay=alg_cfg.get("policy_delay", 2),
            exploration_noise=alg_cfg.get("exploration_noise", 0.1),
            target_policy_noise=alg_cfg.get("target_policy_noise", 0.2),
            target_noise_clip=alg_cfg.get("target_noise_clip", 0.5),
            max_grad_norm=alg_cfg.get("max_grad_norm", 10.0),
            key=key,
        )

    def _init_fast_td3_algorithm(self, alg_cfg: FastTD3Config, key: jax.Array) -> FastTD3:
        """Initialize FastTD3 algorithm."""
        return FastTD3(
            actor_critic=self.actor_critic,
            num_envs=self.env.num_envs,
            actor_lr=alg_cfg.actor_lr,
            critic_lr=alg_cfg.critic_lr,
            gamma=alg_cfg.gamma,
            tau=alg_cfg.tau,
            batch_size=alg_cfg.batch_size,
            policy_delay=alg_cfg.get("policy_delay", 2),
            noise_min=alg_cfg.get("noise_min", 0.05),
            noise_max=alg_cfg.get("noise_max", 0.4),
            target_policy_noise=alg_cfg.get("target_policy_noise", 0.2),
            target_noise_clip=alg_cfg.get("target_noise_clip", 0.5),
            use_cdq=alg_cfg.get("use_cdq", True),
            use_target_actor=alg_cfg.get("use_target_actor", False),
            max_grad_norm=alg_cfg.get("max_grad_norm", 10.0),
            key=key,
        )

    def _init_storage(self):
        """Initialize the replay buffer."""
        obs_dim = self.env.obs_manager.calculate_obs_dim()
        size_per_env = self.cfgs.algorithm.buffer_size // self.env.num_envs

        cfg = {
            "num_envs": self.env.num_envs,
            "actor_obs_shape": [obs_dim["actor"]],
            "critic_obs_shape": [obs_dim["critic"]],
            "actions_shape": [self.env.num_actions],
            "size_per_env": size_per_env,
            "n_steps": self.cfgs.algorithm.n_steps,
        }
        self.alg.init_storage(cfg)

    def _get_initial_obs(self) -> ActInput:
        """Get initial observation as JAX arrays."""
        obs = self.env.obs_manager.get_observation()
        actor_obs = torch_to_jax(obs["actor"])
        critic_obs = torch_to_jax(obs["critic"])
        return ActInput(actor_obs, critic_obs)

    def _collect_experience(
        self,
        obs: ActInput,
        ep_infos: List[Dict],
        iteration: int,
    ) -> Dict[str, Any]:
        """Collect experience from the environment and store in replay buffer."""
        start_time = time.time()

        total_steps = self.num_steps_per_env
        infos = {}
        actor_obs = obs.actor_obs
        critic_obs = obs.critic_obs

        for step in range(total_steps):

            # ========== Warmup: use random actions ==========
            if self.total_timesteps < self.cfgs.algorithm.learning_starts:
                # Random uniform action in [-1, 1]
                self.key, subkey = jax.random.split(self.key)
                actions = jax.random.uniform(
                    subkey,
                    shape=(self.env.num_envs, self.env.num_actions),
                    minval=-1.0,
                    maxval=1.0,
                )

            else:
                # Get action from policy (with exploration noise for TD3, stochastic for SAC)
                actions = self.alg.act(ActInput(actor_obs, critic_obs), deterministic=False)

            # Process action for environment
            actions_for_env = self._process_action_for_env(actions)
            actions_torch = jax_to_torch(actions_for_env, self.device)

            # Environment step
            obs_dict, rewards, terminated, truncated, infos = self.env.step(actions_torch)

            # Convert to JAX
            next_actor_obs = torch_to_jax(obs_dict["actor"])
            next_critic_obs = torch_to_jax(obs_dict["critic"])

            # Handle truncated episodes: use final_observation for bootstrap
            final_obs = infos.get("final_observation")
            if final_obs is not None:
                final_actor = torch_to_jax(final_obs["actor"])
                final_critic = torch_to_jax(final_obs["critic"])

                # Replace next_obs for truncated (not terminated) envs
                truncated_jax_mask = jnp.asarray(truncated.cpu().numpy())
                terminated_jax_mask = jnp.asarray(terminated.cpu().numpy())
                truncated_only = (truncated_jax_mask & ~terminated_jax_mask)[:, None]
                next_actor_obs = jnp.where(truncated_only, final_actor, next_actor_obs)
                next_critic_obs = jnp.where(truncated_only, final_critic, next_critic_obs)

            rewards_jax = torch_to_jax(rewards)
            # NOTE: DO NOT USE DLPACK HERE. DLPACK DOESN'T SUPPORT BOOLEAN
            terminated_jax = jnp.asarray(terminated.cpu().numpy())
            truncated_jax = jnp.asarray(truncated.cpu().numpy())

            # Store transition in replay buffer
            self.alg.store_transition(
                actor_obs=actor_obs,
                critic_obs=critic_obs,
                action=actions,
                reward=rewards_jax,
                next_actor_obs=next_actor_obs,
                next_critic_obs=next_critic_obs,
                terminated=terminated_jax,
                truncated=truncated_jax,
            )

            # Process env step (noise resampling, etc.)
            self.alg.process_env_step(rewards_jax, terminated_jax, truncated_jax, {})

            # Update reward statistics
            dones = terminated | truncated
            self._update_reward_stats(
                reward_info=infos["rewards_per_type"],
                dones=dones,
                success=infos.get("success", None),
            )

            # Update observations
            actor_obs = next_actor_obs
            critic_obs = next_critic_obs

        return {
            "collection_time": time.time() - start_time,
            "last_obs": {
                "actor_obs": actor_obs,
                "critic_obs": critic_obs,
            },
        }

    def _run_training_iteration(
        self,
        obs: ActInput,
        iteration: int,
        ep_infos: List[Dict] = None,
    ) -> IterationData:
        """Execute a single training iteration."""
        # Collect experience
        collection_data = self._collect_experience(
            obs=obs, ep_infos=ep_infos, iteration=iteration
        )

        collection_time = collection_data["collection_time"]
        metrics = None
        learning_time = 0.0
        fps = 0.0
        buffer_size = None
        batch_size = self.cfgs.algorithm.batch_size

        if self.alg.replay_buffer.size >= self.cfgs.algorithm.learning_starts:
            training_start_time = time.time()

            num_updates = max(1, self.cfgs.algorithm.get("num_gradient_steps", 1))
            for _ in range(num_updates):
                self.key, subkey = jax.random.split(self.key)
                batch = self.alg.sample_batch(batch_size, subkey)
                metrics = self.alg.update(batch)
                del batch

            learning_time = time.time() - training_start_time
            fps = (self.num_steps_per_env * self.env.num_envs) / (
                collection_time + learning_time
            )
            buffer_size = self.alg.replay_buffer.size

        # Only compute action statistics on log intervals
        action_dist = {}
        if iteration % self.runner_cfg.log_interval == 0:
            action_dist = self._get_action_statistics()

        return IterationData(
            collection_time=collection_time,
            learning_time=learning_time,
            fps=fps,
            episode_stats=self._build_episode_stats(),
            metrics=metrics,
            last_obs=collection_data["last_obs"],
            action_distribution=action_dist,
            buffer_size=buffer_size,
            iteration=iteration,
        )

    def learn(
        self,
        num_learning_iterations: int,
        init_at_random_ep_len: bool = False,
    ):
        """Main training loop."""
        # Initialize random episode length if requested

        if init_at_random_ep_len:
            self.env.termination_manager.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # Get initial observation
        obs = self._get_initial_obs()
        ep_infos: List[Dict] = []

        # Main training loop
        total_iter = self.current_learning_iteration + num_learning_iterations
        self.initial_learning_iteration = self.current_learning_iteration

        for it in range(self.initial_learning_iteration, total_iter + 1):
            self.it = it

            data = self._run_training_iteration(
                obs=obs,
                iteration=it,
                ep_infos=ep_infos,
            )

            obs = ActInput(
                actor_obs=data.last_obs["actor_obs"],
                critic_obs=data.last_obs["critic_obs"],
            )

            self.post_iteration(data, total_iter, it)

    def _get_action_statistics(self) -> Dict[str, Any]:
        """Extract recent actions from replay buffer."""
        n_recent = min(
            self.num_steps_per_env * self.env.num_envs,
            self.alg.replay_buffer.size
        )
        actions = self.alg.replay_buffer.get_recent_actions(n_recent)
        return self._compute_action_distribution_stats(actions)

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_path: str,
        cfgs: ConfigsForRun = None,
        env: World = None,
        use_wandb: bool = True,
    ) -> "OffPolicyRunner":
        """
        Load runner from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint directory
            cfgs: Config (if None, load from checkpoint)
            env: Environment (if None, create from config)
            use_wandb: Whether to use WandB
        """
        # Load metadata (YAML)
        from rlworld.rl.utils.checkpoint import load_checkpoint_metadata
        metadata = load_checkpoint_metadata(checkpoint_path)

        # Use saved config if not provided
        if cfgs is None:
            from rlworld.rl.utils.checkpoint import load_config_from_checkpoint
            cfgs = load_config_from_checkpoint(metadata)

        # Create env if not provided (reuse BaseRunner logic)
        if env is None:
            env = cls._create_env_from_config(cfgs)

        # Create runner
        runner = cls(env=env, cfgs=cfgs, use_wandb=use_wandb)

        # Load algorithm state
        runner.alg.load_train_state(checkpoint_path, metadata)

        # Restore runner state
        runner.current_learning_iteration = metadata.get(
            "current_learning_iteration", metadata["iteration"]
        )
        runner.total_timesteps = metadata["total_timesteps"]
        runner.total_time = metadata.get("total_time", 0)
        runner.key = jnp.array(metadata["jax_key"], dtype=jnp.uint32)

        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"  Algorithm: {runner.algorithm_name}")
        print(f"  Iteration: {runner.current_learning_iteration}")
        print(f"  Timesteps: {runner.total_timesteps}")

        return runner
