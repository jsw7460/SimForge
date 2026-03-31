import os
import pickle
import shutil
import statistics
import time
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Dict, Any, Optional, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import torch

from rlworld.rl.algorithms.base import RLAlgorithm, ActInput
from rlworld.rl.configs import ConfigsForRun
from rlworld.rl.configs.observations import ObservationTermConfig
from rlworld.rl.envs import World, EpisodeStatsCollector
# from rlworld.rl.envs.curriculum import Go2Curriculum
from rlworld.rl.runners.iteration_data import IterationData, EpisodeStats
from rlworld.rl.utils import setup_log_dir
from rlworld.rl.utils.dynamics_dataset import DynamicsDataset
from rlworld.rl.utils.jax_utils import torch_to_jax, jax_to_torch
from rlworld.rl.utils.console import GREEN, YELLOW, RED, RESET
from rlworld.rl.utils.logger import WandbLogger, ConsoleWriter


# ==================== Base Runner ====================

class BaseRunner(ABC):
    is_distributed_runner: bool = False
    algorithm_name: str

    @classmethod
    def _create_env_from_config(
        cls,
        cfgs: ConfigsForRun,
    ) -> World:
        """Create environment from config.

        Dispatches on ``cfgs.sim_type`` when available (new path), falling
        back to ``cfgs.env.env_name`` for backward compatibility and for
        non-physics environments (Gymnasium, ManiSkill).
        """
        from gymnasium.vector import SyncVectorEnv, AutoresetMode
        import gymnasium as gym
        from rlworld.rl import envs
        from rlworld.rl.envs import GymnasiumEnv

        env_class_name = cfgs.env.env_name
        env_class = getattr(envs, env_class_name)

        sim_type = getattr(cfgs, "sim_type", None)

        if sim_type in ("genesis", "newton", "mujoco") or env_class.sim_name in (
            "Genesis", "Newton", "Mujoco"
        ):
            kwargs = dict(
                num_envs=cfgs.env.num_envs,
                env_cfg=cfgs.env,
                scene_cfg=cfgs.scene,
                visualization_cfg=cfgs.visualization,
                obs_cfg=cfgs.observation,
                act_cfg=cfgs.action,
                reward_cfg=cfgs.reward,
                command_cfg=cfgs.command,
                event_cfg=cfgs.event,
            )
            gait_cfg = getattr(cfgs, "gait", None)
            if gait_cfg is not None:
                kwargs["gait_cfg"] = gait_cfg
            env = env_class(**kwargs)

        elif env_class.sim_name == "ManiSkill":
            from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
            from rlworld.rl.envs import ManiSkillEnv

            env_kwargs = dict(
                obs_mode="state",
                render_mode="rgb_array",
                sim_backend="physx_cuda",
            )
            env_kwargs.update(cfgs.env.gym_make_kwargs)

            env = gym.make(cfgs.env.task_name, num_envs=cfgs.env.num_envs, **env_kwargs)
            env = ManiSkillVectorEnv(env, cfgs.env.num_envs, auto_reset=True, ignore_terminations=False)
            env = ManiSkillEnv(
                env,
                env_cfg=cfgs.env,
                scene_cfg=cfgs.scene,
                obs_cfg=cfgs.observation,
                act_cfg=cfgs.action,
                reward_cfg=cfgs.reward,
                command_cfg=cfgs.command,
                seed=cfgs.env.seed
            )

        elif env_class.sim_name == "Gymnasium":
            def make_env(seed):
                def _init():
                    env = gym.make(cfgs.env.task_name)
                    env.action_space.seed(seed)
                    env.observation_space.seed(seed)
                    return env

                return _init

            env_gym = SyncVectorEnv(
                [make_env(i) for i in range(cfgs.env.num_envs)],
                autoreset_mode=AutoresetMode.SAME_STEP
            )
            env = GymnasiumEnv(
                env_gym,
                env_cfg=cfgs.env,
                scene_cfg=cfgs.scene,
                obs_cfg=cfgs.observation,
                act_cfg=cfgs.action,
                reward_cfg=cfgs.reward,
                command_cfg=cfgs.command,
                seed=cfgs.env.seed
            )

        else:
            raise NotImplementedError(f"{env_class_name} is not implemented.")

        return env

    @classmethod
    def create_with_env(cls, configs: ConfigsForRun, use_wandb: bool = True, seed: int = 0) -> "BaseRunner":
        from rlworld.rl.algorithms import get_runner_class

        runner_cls = get_runner_class(configs.algorithm.algorithm_name)
        env = cls._create_env_from_config(configs)

        if configs.runner.resume_path is None:
            return runner_cls(env, configs, use_wandb=use_wandb, seed=seed)
        else:
            return runner_cls.load_checkpoint(
                checkpoint_path=configs.runner.resume_path,
                cfgs=configs,
                env=env,
                use_wandb=use_wandb,
            )

    def __init__(
        self,
        env: World,
        cfgs: ConfigsForRun,
        use_wandb: bool = True,
        seed: int = 0,
    ):
        """
        Initialize the runner.

        Args:
            env: The environment
            cfgs: Configuration
            use_wandb: Whether to use WandB logging (console always enabled)
            seed: Random seed for JAX
        """
        super().__init__()
        device = env.device

        self.env = env
        self.cfgs = cfgs
        self.runner_cfg = cfgs.runner
        self.device = device

        # JAX random key
        self.jax_seed = seed
        self.key = jax.random.PRNGKey(seed)

        # Logging setup
        self.model_log_dir, self.wandb_log_dir = setup_log_dir(output_dir=self.runner_cfg.output_dir)

        # WandB logger is optional
        self.wandb_logger = None
        self.wandb_url = None
        self.use_wandb = use_wandb
        if use_wandb:
            group_name = self.runner_cfg.run_name
            run_name = self.runner_cfg.run_name + f"_seed{self.env.seed}"
            self.wandb_logger = WandbLogger(
                project_name=self.runner_cfg.wandb_project,
                group_name=group_name,
                run_name=run_name,
                log_dir=self.wandb_log_dir,
                cfg=self.cfgs.recursive_to_dict()
            )
            self.wandb_url = self.wandb_logger.wandb_url

        # Initialize curriculum manager
        # self.curriculum_manager = Go2Curriculum(runner=self, curriculum_cfg=self.cfgs.curriculum)

        # Training parameters
        self.save_interval = self.runner_cfg.save_interval
        self.squash_output: bool | None = None

        # Initialize training modules
        self.training_modules: Dict[str, Any] = dict()
        self._init_training_modules()

        self.alg = self._init_algorithm()

        # Setup console writer
        self.console_writer = ConsoleWriter()

        self.env.reset()
        # JAX version uses jax array for _last_dones
        self.alg._last_dones = jnp.ones(env.num_envs, dtype=jnp.bool_)

        # Initialize storage
        self._init_storage()

        # Training state
        self.initial_learning_iteration = 0
        self.it = 0
        self.total_timesteps = 0
        self.total_time = 0
        self.current_learning_iteration = 0
        self.num_steps_per_env = self.cfgs.algorithm.num_steps_per_env

        # Last eval stats (persisted across iterations for console display)
        self._last_eval_stats: Optional[Dict[str, Any]] = None

        # Initialize environment
        self.reward_statistics = EpisodeStatsCollector(
            num_envs=self.env.num_envs,
            max_episode_length=self.env.max_episode_length,
            device=self.device,
            gamma=self.cfgs.algorithm.gamma
        )

        # Per-sim stats collectors for MultiSimWorld
        from rlworld.rl.envs.multi_sim_world import MultiSimWorld
        if isinstance(self.env, MultiSimWorld):
            self._per_sim_collectors: dict[str, EpisodeStatsCollector] = {}
            for sub_env in self.env.envs:
                self._per_sim_collectors[sub_env.sim_name] = EpisodeStatsCollector(
                    num_envs=sub_env.num_envs,
                    max_episode_length=sub_env.max_episode_length,
                    device=self.device,
                    gamma=self.cfgs.algorithm.gamma,
                )

    def _update_reward_stats(
        self,
        reward_info: dict[str, torch.Tensor],
        dones: torch.Tensor,
        success: torch.Tensor | None = None,
    ) -> None:
        """Update reward statistics including per-sim collectors."""
        self.reward_statistics.update(reward_info=reward_info, dones=dones, success=success)

        if hasattr(self, '_per_sim_collectors'):
            offset = 0
            for sub_env in self.env.envs:
                n = sub_env.num_envs
                collector = self._per_sim_collectors[sub_env.sim_name]
                sub_reward_info = {
                    k: v[offset:offset + n] for k, v in reward_info.items()
                }
                sub_dones = dones[offset:offset + n]
                sub_success = success[offset:offset + n] if success is not None else None
                collector.update(
                    reward_info=sub_reward_info,
                    dones=sub_dones,
                    success=sub_success,
                )
                offset += n

    def _init_action_scaling(self) -> None:
        """Initialize action scaling parameters."""
        action_low = self.env.action_low.cpu().numpy()
        action_high = self.env.action_high.cpu().numpy()
        self.action_low_jax = jnp.array(action_low)
        self.action_high_jax = jnp.array(action_high)
        self.action_scale = (self.action_high_jax - self.action_low_jax) / 2.0
        self.action_bias = (self.action_high_jax + self.action_low_jax) / 2.0

    def _process_action_for_env(self, actions: jax.Array) -> jax.Array:
        """Process actions for environment (SB3-compatible)."""
        if self.squash_output:
            return actions * self.action_scale + self.action_bias
        else:
            return jnp.clip(actions, self.action_low_jax, self.action_high_jax)

    def log_training_data(self, data: IterationData, total_iter: int):
        """Log training data to console and optionally to WandB."""
        # Enrich with runner-level context
        data.total_timesteps = self.total_timesteps
        data.iteration = self.it - self.initial_learning_iteration
        data.total_time = self.total_time

        context = {
            "total_iterations": total_iter,
            "log_dir": self.model_log_dir,
            "simulator": self.env.sim_name,
            "task_name": self.env.task_name,
            "wandb_run_name": self.runner_cfg.run_name,
        }
        if self.wandb_url:
            context["wandb_url"] = self.wandb_url
        if self.wandb_logger:
            context["wandb_run_path"] = self.wandb_logger.run.path

        # Print to console
        self.console_writer.write_iteration(
            data=data,
            context=context,
            last_eval_stats=self._last_eval_stats,
        )

        # Optionally log to WandB
        if self.wandb_logger:
            self.wandb_logger.log_iteration(data=data, step=self.total_timesteps)

    def _build_episode_stats(self) -> EpisodeStats:
        """Build EpisodeStats from reward_statistics, including per-sim stats if MultiSim."""
        stats = self.reward_statistics.snapshot()
        if hasattr(self, '_per_sim_collectors'):
            stats.per_sim_stats = {
                name: collector.snapshot()
                for name, collector in self._per_sim_collectors.items()
            }
        return stats

    def close(self):
        """Clean up resources."""
        if self.wandb_logger:
            self.wandb_logger.close()

    @abstractmethod
    def _init_storage(self):
        """Initialize the experience storage or replay buffer."""
        pass

    @abstractmethod
    def _run_training_iteration(self, obs, iteration: int, **kwargs) -> IterationData:
        """Execute a single training iteration."""
        pass

    @abstractmethod
    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        """Main training loop."""
        pass

    @abstractmethod
    def _init_training_modules(self) -> None:
        """Initialize all trainable modules."""
        pass

    @abstractmethod
    def _init_algorithm(self) -> RLAlgorithm:
        """Initialize the algorithm."""
        pass

    def _get_training_modules(self) -> Dict[str, Any]:
        """Get training modules for the algorithm."""
        return {"actor_critic": self.actor_critic}

    def get_actor_critic_obs(self, actor_obs, critic_obs, *args, **kwargs):
        """
        Process observations for actor and critic.
        Can be overridden by subclasses to add custom processing.
        """
        return actor_obs, critic_obs

    def set_eval_mode(self):
        """Set all components to evaluation mode (no-op for JAX)."""
        if hasattr(self.alg, 'test_mode'):
            self.alg.test_mode()

    def set_train_mode(self):
        """Set all components to training mode (no-op for JAX)."""
        if hasattr(self.alg, 'train_mode'):
            self.alg.train_mode()

    # ==================== In-Training Evaluation ====================

    def _create_eval_configs(self) -> ConfigsForRun:
        """Create evaluation config by modifying a copy of training config."""
        eval_cfgs = deepcopy(self.cfgs)

        # Use fewer envs for evaluation
        eval_cfgs.env.num_envs = self.runner_cfg.eval_num_envs

        # Disable observation noise
        if self.runner_cfg.eval_disable_noise:
            eval_cfgs.observation.enable_noise = False

        # Remove interval events (external forces, disturbances)
        if self.runner_cfg.eval_disable_interval_events and hasattr(eval_cfgs, 'event'):
            eval_cfgs.event.event_terms = [
                t for t in eval_cfgs.event.event_terms
                if t.mode != "interval"
            ]

        # Disable viewer
        eval_cfgs.visualization.show_viewer = False
        eval_cfgs.visualization.record_video = False

        return eval_cfgs

    def _get_or_create_eval_env(self) -> World:
        """Lazily create eval environment on first use."""
        if not hasattr(self, '_eval_env') or self._eval_env is None:
            eval_cfgs = self._create_eval_configs()
            self._eval_env = self._create_env_from_config(eval_cfgs)
        return self._eval_env

    def _run_evaluation(self) -> Dict[str, Any]:
        """Run deterministic evaluation episodes and return statistics."""
        eval_env = self._get_or_create_eval_env()
        eval_start = time.time()

        num_envs = eval_env.num_envs
        target_episodes = self.runner_cfg.eval_num_episodes
        deterministic = self.runner_cfg.eval_deterministic

        # Reset eval env
        eval_env.reset()
        obs_dict = eval_env.obs_manager.get_observation()

        # Per-env tracking
        episode_returns = torch.zeros(num_envs, device=self.device)
        episode_lengths = torch.zeros(num_envs, device=self.device, dtype=torch.long)

        completed_returns: list[float] = []
        completed_lengths: list[float] = []

        # Per-reward-type tracking
        reward_type_sums: dict[str, torch.Tensor] = {}
        completed_reward_breakdowns: dict[str, list[float]] = {}

        max_steps = int(eval_env.max_episode_length) * 2  # Safety limit
        step = 0

        while len(completed_returns) < target_episodes and step < max_steps:
            # Policy inference
            actor_obs = torch_to_jax(obs_dict["actor"])
            critic_obs = torch_to_jax(obs_dict["critic"])
            actions = self.alg.act(
                ActInput(actor_obs, critic_obs),
                deterministic=deterministic,
            )

            # Process actions
            actions_for_env = self._process_action_for_env(actions)
            actions_torch = jax_to_torch(actions_for_env, self.device)

            # Step
            obs_dict, rewards, terminated, truncated, infos = eval_env.step(actions_torch)
            dones = terminated | truncated

            # Accumulate returns
            episode_returns += rewards
            episode_lengths += 1

            # Per-reward-type accumulation
            rewards_per_type = infos.get("rewards_per_type", {})
            for rname, rval in rewards_per_type.items():
                if rname not in reward_type_sums:
                    reward_type_sums[rname] = torch.zeros(num_envs, device=self.device)
                    completed_reward_breakdowns[rname] = []
                reward_type_sums[rname] += rval

            # Collect completed episodes
            for i in range(num_envs):
                if dones[i] and len(completed_returns) < target_episodes:
                    completed_returns.append(episode_returns[i].item())
                    completed_lengths.append(episode_lengths[i].item())
                    for rname in reward_type_sums:
                        completed_reward_breakdowns[rname].append(
                            reward_type_sums[rname][i].item()
                        )

            # Reset tracking for done envs
            episode_returns[dones] = 0
            episode_lengths[dones] = 0
            for rname in reward_type_sums:
                reward_type_sums[rname][dones] = 0

            step += 1

        eval_time = time.time() - eval_start

        # Build results
        eval_stats = {
            "eval/mean_return": np.mean(completed_returns) if completed_returns else 0.0,
            "eval/std_return": np.std(completed_returns) if completed_returns else 0.0,
            "eval/min_return": np.min(completed_returns) if completed_returns else 0.0,
            "eval/max_return": np.max(completed_returns) if completed_returns else 0.0,
            "eval/mean_episode_length": np.mean(completed_lengths) if completed_lengths else 0.0,
            "eval/num_episodes": len(completed_returns),
            "eval/time": eval_time,
        }

        # Per-reward-type eval stats (per-step average, matching training display)
        for rname, vals in completed_reward_breakdowns.items():
            if vals:
                per_step = [v / l for v, l in zip(vals, completed_lengths)]
                eval_stats[f"eval/reward/{rname}"] = np.mean(per_step)

        return eval_stats

    def _log_eval_stats(self, eval_stats: Dict[str, Any], it: int) -> None:
        """Store eval stats for persistent console display and log to wandb."""
        eval_stats["eval/iteration"] = it
        self._last_eval_stats = eval_stats

        # Immediate console summary
        mean_ret = eval_stats["eval/mean_return"]
        std_ret = eval_stats["eval/std_return"]
        mean_len = eval_stats["eval/mean_episode_length"]
        n_eps = eval_stats["eval/num_episodes"]
        eval_time = eval_stats["eval/time"]

        print(f"\n  {GREEN}[Eval @ iter {it}]{RESET} "
              f"return={mean_ret:.2f} ± {std_ret:.2f}  "
              f"length={mean_len:.1f}  "
              f"episodes={n_eps}  "
              f"time={eval_time:.1f}s")

        if self.wandb_logger:
            self.wandb_logger.log_eval_data(eval_stats, step=self.total_timesteps)

    # ==================== End Evaluation ====================

    def _get_action_statistics(self) -> Dict[str, Any]:
        """Extract action statistics from storage. Override in subclasses."""
        raise NotImplementedError

    def _compute_action_distribution_stats(self, actions: np.ndarray) -> Dict[str, Any]:
        """Compute action distribution statistics from raw actions."""
        actions_flat = actions.reshape(-1, actions.shape[-1])

        return {
            "mean": actions_flat.mean(axis=0),
            "std": actions_flat.std(axis=0),
            "min": actions_flat.min(axis=0),
            "max": actions_flat.max(axis=0),
            "raw": actions_flat,
            "correlation": np.corrcoef(actions_flat.T),
        }

    def act(
        self,
        obs: dict[str, torch.Tensor],
        robot_states: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Get action from policy (torch interface for compatibility)."""
        actor_obs = torch_to_jax(obs["actor"])
        critic_obs = torch_to_jax(obs["critic"])
        action_jax = self.alg.act(self.alg.ActInput(actor_obs, critic_obs), deterministic)
        return jax_to_torch(action_jax, self.device)

    # def update_curriculum(self, training_data: Dict[str, Any]):
    #     """Update curriculum difficulty based on training data."""
    #     self.curriculum_manager.update_difficulty(training_data)

    @classmethod
    @abstractmethod
    def load_checkpoint(
        cls,
        checkpoint_path: str,
        cfgs: ConfigsForRun = None,
        env: World = None,
        use_wandb: bool = True,
    ) -> "BaseRunner":
        """
        Load runner from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint directory
            cfgs: Configuration (if None, load from checkpoint metadata)
            env: Environment (if None, create from config)
            use_wandb: Whether to use WandB logging

        Returns:
            Loaded runner instance
        """
        pass

    def checkpoint(self, iteration: int) -> str:
        """Save a checkpoint of the current state."""
        checkpoint_dir = os.path.join(self.model_log_dir, f"checkpoint_{iteration}")
        self._save_checkpoint_to(checkpoint_dir, iteration)
        print(f"Saved checkpoint to {checkpoint_dir}")
        if self.runner_cfg.upload_checkpoint and self.wandb_logger is not None:
            self._upload_checkpoint(checkpoint_dir, iteration)
        return checkpoint_dir

    def _upload_checkpoint(self, checkpoint_dir: str, iteration: int) -> None:
        """Upload checkpoint to wandb as an artifact. Never interrupts training on failure."""
        try:
            self.wandb_logger.upload_checkpoint_artifact(
                checkpoint_dir=checkpoint_dir,
                iteration=iteration,
                metadata={"iteration": iteration, "total_timesteps": self.total_timesteps},
            )
            print(f"Uploaded checkpoint to wandb (iteration {iteration})")
            if self.runner_cfg.delete_local_after_upload:
                shutil.rmtree(checkpoint_dir)
                print(f"Deleted local checkpoint: {checkpoint_dir}")
        except Exception as e:
            print(f"WARNING: Failed to upload checkpoint to wandb: {e}")

    def _save_checkpoint_to(self, checkpoint_dir: str, iteration: int) -> None:
        """Save checkpoint to specified directory."""
        os.makedirs(checkpoint_dir, exist_ok=True)

        alg_metadata = self.alg.save_train_state(checkpoint_dir)
        metadata = {
            "runner_class": self.__class__.__name__,
            "algorithm_name": self.algorithm_name,
            "sim_type": self.cfgs.sim_type,
            "iteration": iteration,
            "total_timesteps": self.total_timesteps,
            "total_time": self.total_time,
            "current_learning_iteration": self.current_learning_iteration,
            "jax_key": np.array(self.key),
            "config": self.cfgs.recursive_to_dict(),
            "wandb_run_path": self.wandb_logger.run.path if self.wandb_logger else None,
            **alg_metadata,
        }

        # Save canonical joint names and training sim info for cross-sim eval.
        try:
            from rlworld.rl.envs.multi_sim_world import MultiSimWorld
            if isinstance(self.env, MultiSimWorld):
                metadata["canonical_joint_names"] = list(
                    self.env.envs[0].act_manager.actuated_joint_names
                )
                metadata["train_sim_names"] = [e.sim_name for e in self.env.envs]
            elif hasattr(self.env, "act_manager"):
                metadata["canonical_joint_names"] = list(
                    self.env.act_manager.actuated_joint_names
                )
        except Exception:
            pass

        metadata_path = os.path.join(checkpoint_dir, "metadata.pkl")
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)

    def _save_latest_checkpoint(self, iteration: int) -> None:
        """Save latest checkpoint (overwritten every iteration)."""
        latest_dir = os.path.join(self.model_log_dir, "checkpoint_latest")
        if os.path.exists(latest_dir):
            shutil.rmtree(latest_dir)
        self._save_checkpoint_to(latest_dir, iteration)

    def post_iteration(self, data: IterationData, total_iter: int, it: int = 0):
        """Post-iteration processing."""
        self.current_learning_iteration += 1
        self.total_timesteps += self.num_steps_per_env * self.env.num_envs
        self.total_time += (data.collection_time + data.learning_time)

        if it % self.runner_cfg.log_interval == 0:
            self.log_training_data(data, total_iter=total_iter)

        # In-training evaluation
        eval_interval = self.runner_cfg.eval_interval
        if eval_interval > 0 and it > 0 and it % eval_interval == 0:
            eval_stats = self._run_evaluation()
            self._log_eval_stats(eval_stats, it=it)

        if it % self.runner_cfg.save_interval == 0:
            self.checkpoint(it)

        # Save latest every iteration
        self._save_latest_checkpoint(it)

    def collect_dynamics_dataset(
        self,
        num_samples: int,
        use_random_policy: bool = False,
        auxiliary_terms: Optional[List[ObservationTermConfig]] = None,
        progress_interval: int = 100
    ) -> DynamicsDataset:
        """
        Collect dynamics dataset with optional auxiliary observations.

        Args:
            num_samples: Number of transitions to collect
            use_random_policy: If True, use random actions; if False, use current policy
            auxiliary_terms: List of observation terms to compute but NOT include in policy obs.
            progress_interval: Print progress every N samples

        Returns:
            DynamicsDataset with collected transitions and auxiliary observations
        """
        print(f"\n{'=' * 60}")
        print(f"Collecting Dynamics Dataset")
        print(f"{'=' * 60}")
        print(f"Target samples: {num_samples}")
        print(f"Policy: {'Random' if use_random_policy else 'Current Policy'}")
        print(f"Num environments: {self.env.num_envs}")

        if auxiliary_terms:
            print(f"\nAuxiliary observations to collect:")
            for i, term in enumerate(auxiliary_terms):
                term_name = getattr(term.func, '__name__', f'term_{i}')
                print(f"  [{i + 1}] {term_name} (scale={term.scale})")
        else:
            print(f"\nNo auxiliary observations requested")

        # Storage for policy observations
        observations_list = []
        actions_list = []
        next_observations_list = []
        dones_list = []

        # Storage for auxiliary observations
        auxiliary_obs_dict = {}
        next_auxiliary_obs_dict = {}

        if auxiliary_terms:
            for i, term in enumerate(auxiliary_terms):
                term_name = getattr(term.func, '__name__', f'term_{i}')
                auxiliary_obs_dict[term_name] = []
                next_auxiliary_obs_dict[term_name] = []

        # Reset environment
        obs_dict, info = self.env.reset()

        robot_states = self.env.get_robot_state()

        collected = 0
        episode_count = 0

        # Episode return tracking
        episode_returns = []
        current_returns = torch.zeros(self.env.num_envs, device=self.device)

        if not use_random_policy:
            self.set_eval_mode()

        with torch.no_grad():
            while collected < num_samples:
                actor_obs = obs_dict["actor"]

                # Compute auxiliary observations (CURRENT state)
                if auxiliary_terms:
                    for term in auxiliary_terms:
                        term_name = getattr(term.func, '__name__', 'unknown')
                        aux_value = term.func(self.env, **term.params)
                        aux_value = aux_value * term.scale
                        auxiliary_obs_dict[term_name].append(aux_value.clone().cpu())

                # Choose action
                if use_random_policy:
                    actions = torch.randn(
                        self.env.num_envs,
                        self.env.num_actions,
                        device=self.device
                    )
                    actions = torch.clamp(actions, -1.0, 1.0)
                else:
                    actions = self.act(obs_dict, robot_states, deterministic=True)

                # Step environment
                next_obs_dict, _, rewards, dones, infos = self.env.step(actions)
                next_robot_states = self.env.get_robot_state()
                next_actor_obs = next_obs_dict["actor"]

                # Update episode returns
                current_returns += rewards

                if dones.any():
                    completed_returns = current_returns[dones].cpu().tolist()
                    episode_returns.extend(completed_returns)
                    current_returns[dones] = 0.0

                # Compute auxiliary observations (NEXT state)
                if auxiliary_terms:
                    for term in auxiliary_terms:
                        term_name = getattr(term.func, '__name__', 'unknown')
                        next_aux_value = term.func(self.env, **term.params)
                        next_aux_value = next_aux_value * term.scale
                        next_auxiliary_obs_dict[term_name].append(next_aux_value.clone().cpu())

                # Store policy observations and actions
                observations_list.append(actor_obs.clone().cpu())
                actions_list.append(actions.clone().cpu())
                next_observations_list.append(next_actor_obs.clone().cpu())
                dones_list.append(dones.clone().cpu())

                # Update
                obs_dict = next_obs_dict
                robot_states = next_robot_states
                collected += self.env.num_envs

                episode_count += dones.sum().item()

                if collected % progress_interval == 0:
                    if episode_returns:
                        mean_return = sum(episode_returns) / len(episode_returns)
                        recent_returns = episode_returns[-100:]
                        recent_mean = sum(recent_returns) / len(recent_returns)
                        min_ret = min(episode_returns)
                        max_ret = max(episode_returns)
                        print(
                            f"Collected: {collected}/{num_samples} | "
                            f"Episodes: {episode_count} | "
                            f"Return: {mean_return:.2f} (recent: {recent_mean:.2f}) | "
                            f"Range: [{min_ret:.2f}, {max_ret:.2f}]"
                        )
                    else:
                        print(f"Collected: {collected}/{num_samples} | Episodes: {episode_count}")

        if not use_random_policy:
            self.set_train_mode()

        # Helper function to reorder parallel data
        def reorder_parallel_data(data_list: List[torch.Tensor], num_envs: int, target_size: int) -> torch.Tensor:
            data = torch.cat(data_list, dim=0)
            num_steps = len(data_list)

            if data.dim() == 1:
                data = data.reshape(num_steps, num_envs)
                data = data.permute(1, 0)
                data = data.reshape(-1)[:target_size]
            else:
                dim = data.shape[1]
                data = data.reshape(num_steps, num_envs, dim)
                data = data.permute(1, 0, 2)
                data = data.reshape(-1, dim)[:target_size]

            return data

        # Reorder policy observations
        observations = reorder_parallel_data(observations_list, self.env.num_envs, num_samples)
        actions = reorder_parallel_data(actions_list, self.env.num_envs, num_samples)
        next_observations = reorder_parallel_data(next_observations_list, self.env.num_envs, num_samples)
        dones = reorder_parallel_data(dones_list, self.env.num_envs, num_samples)

        # Reorder auxiliary observations
        auxiliary_obs = {}
        next_auxiliary_obs = {}

        for term_name, values in auxiliary_obs_dict.items():
            if values:
                auxiliary_obs[term_name] = reorder_parallel_data(values, self.env.num_envs, num_samples)

        for term_name, values in next_auxiliary_obs_dict.items():
            if values:
                next_auxiliary_obs[term_name] = reorder_parallel_data(values, self.env.num_envs, num_samples)

        # Create dataset with metadata
        metadata = {
            'collection_iteration': self.current_learning_iteration,
            'collection_timesteps': self.total_timesteps,
            'policy_type': 'random' if use_random_policy else 'trained',
            'num_envs': self.env.num_envs,
            'episodes_collected': episode_count,
            'env_name': self.cfgs.env.env_name,
            'obs_dim': observations.shape[1],
            'action_dim': actions.shape[1],
            'auxiliary_terms': list(auxiliary_obs.keys()) if auxiliary_obs else [],
            'mean_return': sum(episode_returns) / len(episode_returns) if episode_returns else 0.0,
            'std_return': (sum((r - sum(episode_returns) / len(episode_returns)) ** 2 for r in episode_returns) / len(
                episode_returns)) ** 0.5 if len(episode_returns) > 1 else 0.0,
            'min_return': min(episode_returns) if episode_returns else 0.0,
            'max_return': max(episode_returns) if episode_returns else 0.0,
            'total_episodes': len(episode_returns),
        }

        dataset = DynamicsDataset(
            observations=observations,
            actions=actions,
            next_observations=next_observations,
            dones=dones,
            auxiliary_obs=auxiliary_obs,
            next_auxiliary_obs=next_auxiliary_obs,
            metadata=metadata
        )

        print(f"\n{GREEN}✓ Dataset collection complete!{RESET}")
        print(f"  - Collected: {len(dataset)} transitions")
        print(f"  - Episodes: {episode_count}")
        print(f"  - Episode ends: {dones.sum().item()}")
        print(f"  - Policy obs shape: {observations.shape}")
        print(f"  - Action shape: {actions.shape}")

        if episode_returns:
            print(f"\n  - Return Statistics:")
            print(f"    * Mean: {metadata['mean_return']:.2f}")
            print(f"    * Std:  {metadata['std_return']:.2f}")
            print(f"    * Min:  {metadata['min_return']:.2f}")
            print(f"    * Max:  {metadata['max_return']:.2f}")

        if auxiliary_obs:
            print(f"\n  - Auxiliary observations:")
            for key, tensor in auxiliary_obs.items():
                print(f"    * {key}: {tensor.shape}")

        return dataset

    def save_dataset_checkpoint(
        self,
        dataset: DynamicsDataset,
        save_name: Optional[str] = None,
        include_policy: bool = True
    ) -> str:
        """Save dataset checkpoint with runner state."""
        from rlworld.rl.utils.dataset_manager import DatasetCheckpointHandler

        if save_name is None:
            save_name = f"dataset_iter{self.current_learning_iteration}_size{len(dataset)}.pt"

        save_path = os.path.join(self.model_log_dir, save_name)

        DatasetCheckpointHandler.save_dataset_checkpoint(
            runner=self,
            dataset=dataset,
            path=save_path,
            include_policy=include_policy
        )

        return save_path

    def collect_and_save_dataset(
        self,
        num_samples: int,
        save_name: Optional[str] = None,
        use_random_policy: bool = False,
        auxiliary_terms: Optional[List[ObservationTermConfig]] = None,
        include_policy: bool = True
    ) -> Tuple[DynamicsDataset, str]:
        """Convenience method to collect and save dataset in one call."""
        dataset = self.collect_dynamics_dataset(
            num_samples=num_samples,
            use_random_policy=use_random_policy,
            auxiliary_terms=auxiliary_terms
        )

        save_path = self.save_dataset_checkpoint(
            dataset=dataset,
            save_name=save_name,
            include_policy=include_policy
        )

        return dataset, save_path
