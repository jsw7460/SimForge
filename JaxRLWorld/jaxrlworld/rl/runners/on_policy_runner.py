import time
from typing import Any, Dict, List

import jax
import jax.numpy as jnp
import numpy as np
import torch

from jaxrlworld.rl.algorithms.ppo import PPO
from jaxrlworld.rl.algorithms.ppo.symmetry import build_mirror_spec
from jaxrlworld.rl.configs import ConfigsForRun
from jaxrlworld.rl.configs.algorithms import PPOConfig
from jaxrlworld.rl.configs.common_config_classes import VisionActorCfg, VisionCriticCfg
from jaxrlworld.rl.envs import World
from jaxrlworld.rl.modules.policies.ppo_ac import PPOActorCritic
from jaxrlworld.rl.modules.utils import count_parameters, print_model_summary
from jaxrlworld.rl.runners.base_runner import BaseRunner
from jaxrlworld.rl.runners.iteration_data import IterationData
from jaxrlworld.rl.utils.jax_utils import jax_to_torch, torch_to_jax, torch_to_jax_many


class OnPolicyRunner(BaseRunner):
    """
    On-policy runner using JAX PPO with PyTorch environment.

    Features:
    - Rollout storage for experience collection
    - GAE advantage estimation
    - Checkpoint save/load support
    """

    alg: PPO
    actor_critic: PPOActorCritic
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
        # The on-policy path reads only the critic's groups from
        # ``final_observation`` (PPO truncation bootstrap; see
        # ``_pack_obs``), so terminal steps skip computing the rest.
        # Adapter envs without the attribute simply carry it inertly.
        self.env.terminal_obs_groups = ("critic", *self.critic_image_groups)

    def _init_training_modules(self) -> None:
        """Initialize actor-critic model based on algorithm type."""
        obs_dim = self.env.calculate_obs_dim()
        self.actor_obs_dim = obs_dim["actor"]
        self.critic_obs_dim = obs_dim["critic"]
        self.num_actions_dim = self.env.num_actions

        policy_cfg = self.cfgs.nn.policy

        # A vision policy reads image groups next to the state vector, so
        # observations travel as a dict of groups instead of one array.
        self.obs_shapes = self.env.obs_manager.calculate_obs_shapes()
        self.actor_image_groups = (
            tuple(policy_cfg.actor.image_groups) if isinstance(policy_cfg.actor, VisionActorCfg) else ()
        )
        self.critic_image_groups = (
            tuple(policy_cfg.critic.image_groups) if isinstance(policy_cfg.critic, VisionCriticCfg) else ()
        )
        self.image_groups_by_role = {"actor": self.actor_image_groups, "critic": self.critic_image_groups}

        self.key, subkey = jax.random.split(self.key)

        self._init_ppo_actor_critic(policy_cfg, subkey)

        self.training_modules = {"actor_critic": self.actor_critic}

        self.squash_output = self.actor_critic.is_squashed
        self._init_action_scaling()

        print_model_summary(self.actor_critic, "PPOActorCritic")

        if self.use_wandb:
            self._log_model_parameters()

    def _init_algorithm(self) -> PPO:
        """Initialize algorithm based on type."""
        alg_cfg = self.cfgs.algorithm

        self.key, subkey = jax.random.split(self.key)

        self.alg = self._init_ppo_algorithm(alg_cfg, subkey)

        return self.alg

    def _init_ppo_algorithm(self, alg_cfg: PPOConfig, key: jax.Array) -> PPO:
        """Initialize PPO algorithm."""
        symmetry_spec = None
        symmetry_coef = 0.0
        sc = alg_cfg.symmetry_cfg
        if sc is not None and sc.use_mirror_loss:
            if self.actor_image_groups:
                raise ValueError(
                    "The mirror loss permutes observation entries, which has no meaning for an image: "
                    "mirroring a camera view means flipping pixels and re-deriving what the flipped "
                    "scene should look like. Turn off symmetry_cfg for a vision policy."
                )
            symmetry_spec = build_mirror_spec(self.env.obs_manager, list(self.env.act_manager.actuated_joint_names))
            symmetry_coef = sc.mirror_loss_coeff
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
            use_value_normalization=alg_cfg.use_value_normalization,
            use_early_stop=alg_cfg.use_early_stop,
            optimizer_eps=alg_cfg.optimizer_eps,
            normalize_advantage_per_minibatch=alg_cfg.normalize_advantage_per_minibatch,
            symmetry_spec=symmetry_spec,
            symmetry_coef=symmetry_coef,
            key=key,
        )

    def _init_ppo_actor_critic(self, policy_cfg, key: jax.Array) -> None:
        """Initialize PPO actor-critic."""

        if hasattr(self.env, "scene_manager"):
            kinematic_tree = self.env.scene_manager.trees.get("robot", None)
        else:
            kinematic_tree = None

        actuated_joint_names = (
            list(self.env.act_manager.actuated_joint_names) if hasattr(self.env, "act_manager") else None
        )

        self.actor_critic = PPOActorCritic(
            num_actor_obs=self.actor_obs_dim,
            num_critic_obs=self.critic_obs_dim,
            num_actions=self.num_actions_dim,
            actor_cfg=policy_cfg.actor,
            critic_cfg=policy_cfg.critic,
            init_noise_std=policy_cfg.init_noise_std,
            std_type=policy_cfg.std_type,
            distribution_type=policy_cfg.distribution_type,
            kinematic_tree=kinematic_tree,
            actuated_joint_names=actuated_joint_names,
            key=key,
            obs_normalization=self.cfgs.algorithm.obs_normalization,
            obs_shapes=self.obs_shapes,
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

    def _init_storage(self):
        """Initialize the experience storage."""
        obs_dim = self.env.calculate_obs_dim()
        cfg = {
            "num_envs": self.env.num_envs,
            "num_transitions_per_env": self.cfgs.algorithm.num_steps_per_env,
            "actor_obs_shape": self._obs_shape_spec("actor"),
            "critic_obs_shape": self._obs_shape_spec("critic"),
            "actions_shape": [self.env.num_actions],
            "robot_state_shape": [obs_dim.get("robot_state", 0)],
            "estimator_obs_shape": [obs_dim.get("estimator", 0)],
        }
        self.alg.init_storage(cfg)

    def _obs_shape_spec(self, role: str):
        """Storage shape for one model: a plain tuple, or one per group."""
        image_groups = self.image_groups_by_role[role]
        if not image_groups:
            return [self.obs_shapes[role][0]]
        spec = {role: tuple(self.obs_shapes[role])}
        spec.update({group: tuple(self.obs_shapes[group]) for group in image_groups})
        return spec

    def _pack_obs(self, obs_dict, role: str):
        """One model's observation, converted to JAX.

        Without image groups this is the state vector alone, exactly as
        before. With them it is a dict keyed by group name — the same
        keys the model was built against.

        One conversion, so one wait. The collection loop instead splits
        this into :meth:`_obs_sources` / :meth:`_assemble_obs` and pays a
        single wait for every tensor of the step (see
        ``torch_to_jax_many``).
        """
        vector = torch_to_jax(obs_dict[role])
        image_groups = self.image_groups_by_role[role]
        if not image_groups:
            return vector
        packed = {role: vector}
        packed.update({group: torch_to_jax(obs_dict[group]) for group in image_groups})
        return packed

    def _obs_sources(self, obs_dict, role: str, prefix: str) -> Dict[str, Any]:
        """The torch tensors one model's observation is built from.

        Keys are prefixed so the actor's, the critic's and a terminal
        observation's groups can share one conversion batch. Two roles
        naming the same image group collapse to one entry, which is
        correct: it is the same tensor.
        """
        sources = {f"{prefix}{role}": obs_dict[role]}
        sources.update({f"{prefix}{group}": obs_dict[group] for group in self.image_groups_by_role[role]})
        return sources

    def _assemble_obs(self, converted: Dict[str, Any], role: str, prefix: str):
        """Rebuild :meth:`_pack_obs`'s shape from a converted batch."""
        vector = converted[f"{prefix}{role}"]
        image_groups = self.image_groups_by_role[role]
        if not image_groups:
            return vector
        packed = {role: vector}
        packed.update({group: converted[f"{prefix}{group}"] for group in image_groups})
        return packed

    def _get_initial_obs(self) -> PPO.ActInput:
        """Get initial observation as JAX arrays."""
        obs = self.env.get_observation()
        return PPO.ActInput(
            self._pack_obs(obs, "actor"),
            self._pack_obs(obs, "critic"),
        )

    def _postprocess_step_reward(self, rewards, actions, obs_dict, step_i):
        """Per-step reward-shaping hook; identity by default.

        Override to add an externally-computed reward term before it
        enters the algorithm. ``rewards`` is the torch reward tensor from
        ``env.step``; return a tensor of the same shape/device. ``self.env``
        is accessible for reading state. Called every rollout step.
        """
        return rewards

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

            # Reward-shaping hook (default: identity). Subclasses may add
            # an externally-computed reward term — one that the env's
            # reward manager cannot produce because it depends on a window
            # of steps or a separate evaluation env (e.g. a per-segment
            # information-gain reward) — before it enters the algorithm.
            rewards = self._postprocess_step_reward(
                rewards,
                actions_torch,
                obs_dict,
                _step_i,
            )

            # Convert to JAX. Every tensor of the step crosses in ONE
            # batch, because each conversion has to wait for its copy to
            # run before the source can be released (see
            # ``torch_to_jax_many``) and one wait for the step is far
            # cheaper than one per tensor. The uint8 casts below have to
            # be built into the batch rather than converted separately:
            # the mapping is what keeps those temporaries alive across
            # the wait.
            #
            # Bool tensors must not go through DLPack as BOOL (its bool
            # dtype exchange is what once produced rare random bit flips).
            # Crossing as uint8 sidesteps that entirely: the device-side
            # cast allocates a fresh 0/1 buffer, DLPack carries a
            # first-class dtype, and JAX reconstructs bool with a defined
            # nonzero->True cast. Adoption was gated on
            # check_bool_dlpack_bridge.
            terminal_obs = infos.get("final_observation")
            bootstrap_mask = infos.get("bootstrap_mask")

            sources = self._obs_sources(obs_dict, "actor", "")
            sources.update(self._obs_sources(obs_dict, "critic", ""))
            sources["reward"] = rewards
            sources["terminated"] = terminated.to(torch.uint8)
            sources["truncated"] = truncated.to(torch.uint8)
            if terminal_obs is not None:
                # Only the critic's terminal observation is consumed
                # (bootstrap value); nothing reads an "actor" entry.
                sources.update(self._obs_sources(terminal_obs, "critic", "final_"))
                # Bootstrap mask (truncations + non-absorbing terminations).
                # Only meaningful on done steps (final_observation present);
                # absent -> PPO falls back to ``truncated & ~terminated``.
                if bootstrap_mask is not None:
                    sources["bootstrap_mask"] = bootstrap_mask.to(torch.uint8)

            converted = torch_to_jax_many(sources)

            actor_obs = self._assemble_obs(converted, "actor", "")
            critic_obs = self._assemble_obs(converted, "critic", "")
            rewards_jax = converted["reward"]
            terminated_jax = converted["terminated"].astype(jnp.bool_)
            truncated_jax = converted["truncated"].astype(jnp.bool_)

            # Process step
            infos_jax = {}
            if terminal_obs is not None:
                infos_jax["final_observation"] = {
                    "critic": self._assemble_obs(converted, "critic", "final_"),
                }
                if bootstrap_mask is not None:
                    infos_jax["bootstrap_mask"] = converted["bootstrap_mask"].astype(jnp.bool_)
            self.alg.process_env_step(
                rewards_jax,
                terminated_jax,
                truncated_jax,
                infos_jax,
                next_actor_obs=actor_obs,
                next_critic_obs=critic_obs,
            )

            # Update statistics
            self._update_reward_stats(
                reward_info=infos["rewards_per_type"],
                dones=dones,
                success=infos.get("success", None),
            )

        return {
            "collection_time": time.time() - start_time,
            "last_obs": {
                "actor_obs": actor_obs,
                "critic_obs": critic_obs,
            },
        }

    def _run_training_iteration(
        self,
        obs: PPO.ActInput,
        iteration: int,
        ep_infos: List[Dict] = None,
    ) -> IterationData:
        """Execute a single training iteration."""
        # Collect experience
        collection_data = self._collect_experience(obs=obs, ep_infos=ep_infos)

        # Update policy
        start_time = time.time()
        self.alg.compute_returns(collection_data["last_obs"]["critic_obs"])
        action_stats = self._get_action_statistics()

        metrics = self.alg.update()
        learning_time = time.time() - start_time

        collection_time = collection_data["collection_time"]
        fps = (self.num_steps_per_env * self.env.num_envs) / (collection_time + learning_time)

        return IterationData(
            collection_time=collection_time,
            learning_time=learning_time,
            fps=fps,
            episode_stats=self._build_episode_stats(),
            metrics=metrics,
            last_obs=collection_data["last_obs"],
            action_distribution=action_stats,
            iteration=iteration,
        )

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
            data = self._run_training_iteration(
                obs=obs,
                iteration=it,
                ep_infos=ep_infos,
            )
            # Update obs
            obs = PPO.ActInput(
                actor_obs=data.last_obs["actor_obs"],
                critic_obs=data.last_obs["critic_obs"],
            )
            self.post_iteration(data, total_iter, it)

    def _get_action_statistics(self) -> Dict[str, Any]:
        """Extract action stats from rollout storage.

        Gated BEFORE the device transfer: with both action-logging flags
        off (the default) the old path still paid a blocking D2H of the
        whole rollout's actions every iteration — and, sitting between
        compute_returns and update(), charged it to learning_time.
        """
        logging_cfg = getattr(self.runner_cfg, "logging", None)
        if not (getattr(logging_cfg, "action_dist", False) or getattr(logging_cfg, "action_histogram", False)):
            return {}
        # Returns flattened [num_steps * num_envs, action_dim]; the helper
        # reshape internally if it needs the (T, N, D) layout.
        actions = np.array(self.alg.storage.get_flat_actions())
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
        # Load metadata (YAML)
        from jaxrlworld.rl.utils.checkpoint import load_checkpoint_metadata

        metadata = load_checkpoint_metadata(checkpoint_path)

        # Use saved config if not provided
        if cfgs is None:
            from jaxrlworld.rl.utils.checkpoint import load_config_from_checkpoint

            cfgs = load_config_from_checkpoint(metadata)

        # Create env if not provided
        if env is None:
            env = cls._create_env_from_config(cfgs)

        # Create runner (this initializes fresh model and algorithm)
        runner = cls(env=env, cfgs=cfgs, use_wandb=use_wandb)

        # Delegate model loading to algorithm
        runner.alg.load_train_state(checkpoint_path, metadata)

        # Restore runner state
        runner.current_learning_iteration = metadata.get("current_learning_iteration", metadata["iteration"])
        runner.total_timesteps = metadata["total_timesteps"]
        runner.total_time = metadata.get("total_time", 0)
        runner.key = jnp.array(metadata["jax_key"], dtype=jnp.uint32)

        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"  Algorithm: {runner.algorithm_name}")
        print(f"  Iteration: {runner.current_learning_iteration}")
        print(f"  Timesteps: {runner.total_timesteps}")
        print(f"  Total time: {runner.total_time:.2f}s")
        print("  Note: Optimizer state re-initialized (momentum reset)")

        return runner
