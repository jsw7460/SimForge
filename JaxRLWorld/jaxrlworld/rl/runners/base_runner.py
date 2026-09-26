import os
import shutil
import time
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Callable, Dict

if TYPE_CHECKING:
    import gymnasium as gym

import jax
import jax.numpy as jnp
import numpy as np
import torch

from jaxrlworld.rl.algorithms.base import ActInput, RLAlgorithm
from jaxrlworld.rl.configs import ConfigsForRun
from jaxrlworld.rl.envs import EpisodeStatsCollector, World
from jaxrlworld.rl.envs.utils.lazy_import_check import assert_single_sim_loaded
from jaxrlworld.rl.runners.iteration_data import EpisodeStats, IterationData
from jaxrlworld.rl.utils import setup_log_dir
from jaxrlworld.rl.utils.console import GREEN, RESET
from jaxrlworld.rl.utils.jax_utils import jax_to_torch, torch_to_jax
from jaxrlworld.rl.utils.logger import ConsoleWriter, WandbLogger

# ==================== Base Runner ====================


class BaseRunner(ABC):
    is_distributed_runner: bool = False
    algorithm_name: str

    @classmethod
    def _create_env_from_config(
        cls,
        cfgs: ConfigsForRun,
        gym_env_factory: "Callable[[int], gym.Env] | None" = None,
    ) -> World:
        """Create environment from config.

        Dispatches on ``env_class.sim_name`` (resolved from
        ``cfgs.env.env_name`` via ``jaxrlworld.rl.envs``).  ``cfgs.sim_type``
        is intentionally not consulted here: it labels the *config family*
        (a ``GenesisConfigsForRun`` keeps ``sim_type="genesis"`` even when
        ``env_name="GymnasiumEnv"``), so dispatching on it would route a
        Gymnasium env through the physics-sim kwargs branch.

        ``gym_env_factory`` (Gymnasium path only): user-provided
        ``(seed) -> gym.Env`` callable.  When set, replaces the default
        bare ``gym.make(task_name)`` factory so the runner-built eval
        env shares the same wrapper chain as the user-built training
        env (action repeat, ``FlattenObservation``, ...).  Set via
        :attr:`gym_env_factory` on the runner instance — see
        :class:`jaxrlworld.rl.envs.gymnasium.make_dmc_env_factory` for the
        DMC reproduction pattern.
        """
        from jaxrlworld.rl import envs

        env_class_name = cfgs.env.env_name
        # ``envs.__getattr__`` lazily imports the matching sim package only
        # when the class is actually requested — no other simulator gets
        # dragged in.
        env_class = getattr(envs, env_class_name)

        # Dispatch on the env class's own ``sim_name`` only — ``cfgs.sim_type``
        # tracks the *config family* (a ``GenesisConfigsForRun`` keeps
        # ``sim_type="genesis"`` even when ``env_name="GymnasiumEnv"``), so
        # using it here would force the Genesis sim kwargs onto a Gymnasium
        # env and raise ``TypeError: unexpected keyword argument 'num_envs'``.
        if env_class.sim_name in ("Genesis", "Newton", "Mujoco"):
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
                curriculum_cfg=cfgs.curriculum,
            )
            gait_cfg = getattr(cfgs, "gait", None)
            if gait_cfg is not None:
                kwargs["gait_cfg"] = gait_cfg
            env = env_class(**kwargs)

        elif env_class.sim_name == "ManiSkill":
            import gymnasium as gym
            from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

            from jaxrlworld.rl.envs import ManiSkillEnv

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
                seed=cfgs.env.seed,
            )

        elif env_class.sim_name == "Gymnasium":
            import gymnasium as gym
            from gymnasium.vector import AutoresetMode, SyncVectorEnv

            from jaxrlworld.rl.envs import GymnasiumEnv

            if gym_env_factory is not None:
                # User-supplied factory carries the full wrapper chain
                # (action repeat, FlattenObservation, seeding).  Reuse it
                # verbatim so train and eval envs are structurally
                # identical.
                def make_env(seed):
                    return lambda: gym_env_factory(seed)
            else:
                # Bare fallback for the legacy / wrapper-less Gymnasium
                # path.  Used by scripts that don't set
                # ``runner.gym_env_factory`` — fine for simple tasks
                # whose raw observation is already a flat ``Box``.
                def make_env(seed):
                    def _init():
                        env = gym.make(cfgs.env.task_name)
                        env.action_space.seed(seed)
                        env.observation_space.seed(seed)
                        return env

                    return _init

            env_gym = SyncVectorEnv(
                [make_env(i) for i in range(cfgs.env.num_envs)], autoreset_mode=AutoresetMode.SAME_STEP
            )
            env = GymnasiumEnv(
                env_gym,
                env_cfg=cfgs.env,
                scene_cfg=cfgs.scene,
                obs_cfg=cfgs.observation,
                act_cfg=cfgs.action,
                reward_cfg=cfgs.reward,
                command_cfg=cfgs.command,
                seed=cfgs.env.seed,
            )

        else:
            raise NotImplementedError(f"{env_class_name} is not implemented.")

        return env

    @classmethod
    def create_with_env(cls, configs: ConfigsForRun, use_wandb: bool = True, seed: int = 0) -> "BaseRunner":
        from jaxrlworld.rl.algorithms import get_runner_class
        from jaxrlworld.rl.utils.wandb_checkpoint import resolve_checkpoint_path

        runner_cls = get_runner_class(configs.algorithm.algorithm_name)
        env = cls._create_env_from_config(configs)
        # Guard the per-process single-backend invariant — see
        # ``lazy_import_check`` for context. Set JAXRLWORLD_ALLOW_MULTI_SIM=1
        # in diag/cross-sim scripts that build multiple backends on purpose.
        assert_single_sim_loaded()

        if configs.runner.resume_path is None:
            return runner_cls(env, configs, use_wandb=use_wandb, seed=seed)
        else:
            # resume_path may be a local dir OR a wandb run path
            # ("entity/project/run_id"); the latter downloads the latest
            # checkpoint (re-resolved to the newest upload each run).
            checkpoint_path = resolve_checkpoint_path(configs.runner.resume_path)
            return runner_cls.load_checkpoint(
                checkpoint_path=checkpoint_path,
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
            self.wandb_logger = WandbLogger(
                project_name=self.runner_cfg.wandb_project,
                log_dir=self.wandb_log_dir,
                cfg=self.cfgs.recursive_to_dict(),
                **self._wandb_identity(),
            )
            self.wandb_url = self.wandb_logger.wandb_url
            self.wandb_logger.record_run_location(self.model_log_dir)

        # Training parameters
        self.save_interval = self.runner_cfg.save_interval
        self.squash_output: bool | None = None

        # Gymnasium-only: user-supplied ``(seed) -> gym.Env`` factory
        # used by ``_create_env_from_config`` when building the eval
        # env, so the runner-built eval env shares the same wrapper
        # chain as the user-built training env.  ``None`` falls back
        # to a bare ``gym.make(task_name)`` factory (legacy path).
        self.gym_env_factory: Callable[[int], gym.Env] | None = None

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
        self._last_eval_stats: Dict[str, Any] | None = None

        # Initialize environment
        self.reward_statistics = EpisodeStatsCollector(
            num_envs=self.env.num_envs,
            max_episode_length=self.env.max_episode_length,
            device=self.device,
            gamma=self.cfgs.algorithm.gamma,
        )

    def _wandb_identity(self) -> dict:
        """The W&B axes for this run: group, name, job type, tags, notes.

        The run config is uploaded whole, so nothing here needs to encode a
        comparison. These are only what the UI groups and filters on before
        anyone opens the config: which batch a run belongs to, which
        simulator produced it, and which preset it came from.

        ``WANDB_RUN_GROUP`` and ``WANDB_NOTES`` are read by W&B itself; they
        are read here too so the values also reach the run name and so a
        config field can override them.
        """
        cfg = self.runner_cfg
        # "...configs.presets.k1_velocity.amp" -> ("k1_velocity", "amp")
        preset_path = self.cfgs.preset_module or ""
        _, _, preset_tail = preset_path.partition(".presets.")
        preset_tags = [part for part in preset_tail.split(".") if part and part != "base"]

        group = cfg.wandb_group or os.environ.get("WANDB_RUN_GROUP") or cfg.run_name
        tags = [self.cfgs.sim_type, *preset_tags, *cfg.wandb_tags]

        return {
            "group_name": group,
            "run_name": f"{cfg.run_name}-s{self.env.seed}",
            "job_type": cfg.wandb_job_type or self.cfgs.sim_type,
            "tags": list(dict.fromkeys(tag for tag in tags if tag)),
            "notes": os.environ.get("WANDB_NOTES") or None,
        }

    def _update_reward_stats(
        self,
        reward_info: dict[str, torch.Tensor],
        dones: torch.Tensor,
        success: torch.Tensor | None = None,
    ) -> None:
        """Update reward statistics."""
        self.reward_statistics.update(reward_info=reward_info, dones=dones, success=success)

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

            # Curriculum state: logged as its own ``Curriculum/`` wandb
            # namespace separate from the reward breakdown so that
            # physically-scaled values (e.g. ``energy_threshold`` in
            # Watts) don't get averaged alongside unit-less reward
            # terms. Only finite scalar fields are logged — infinite
            # placeholders (``float("inf")`` before the first stage
            # fires) are skipped so wandb plots cleanly.
            curriculum_manager = getattr(self.env, "curriculum_manager", None)
            if curriculum_manager is not None and curriculum_manager.state:
                import math as _math

                import wandb as _wandb

                curr_log: dict[str, float] = {}
                for term_name, field_dict in curriculum_manager.state.items():
                    if not isinstance(field_dict, dict):
                        continue
                    for field_name, value in field_dict.items():
                        if not isinstance(value, int | float):
                            continue
                        if not _math.isfinite(value):
                            continue
                        curr_log[f"Curriculum/{term_name}/{field_name}"] = float(value)
                if curr_log:
                    _wandb.log(curr_log, step=self.total_timesteps)

            # Per-term termination-cause ratios, aggregated over the
            # iteration window by the termination manager's internal
            # accumulator. ``consume_episode_stats`` returns keys of the
            # form ``Episode_Termination/<name>`` and clears the window
            # so successive iterations report disjoint episodes.
            term_mgr = getattr(self.env, "termination_manager", None)
            if term_mgr is not None and hasattr(term_mgr, "consume_episode_stats"):
                term_log = term_mgr.consume_episode_stats()
                if term_log:
                    import wandb as _wandb

                    _wandb.log(term_log, step=self.total_timesteps)

    def _build_episode_stats(self) -> EpisodeStats:
        """Build EpisodeStats from reward_statistics."""
        return self.reward_statistics.snapshot()

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
        self.alg.test_mode()

    def set_train_mode(self):
        """Set all components to training mode (no-op for JAX)."""
        self.alg.train_mode()

    # ==================== In-Training Evaluation ====================

    def _create_eval_configs(self) -> ConfigsForRun:
        """Create evaluation config by modifying a copy of training config."""
        eval_cfgs = deepcopy(self.cfgs)

        # Use fewer envs for evaluation
        eval_cfgs.env.num_envs = self.runner_cfg.eval_num_envs

        # Disable observation noise on every group
        if self.runner_cfg.eval_disable_noise:
            from jaxrlworld.rl.configs.common_config_classes import disable_corruption

            disable_corruption(eval_cfgs.observation)

        # Disable interval events (external forces, disturbances) and the
        # interval-timer domain randomization by setting them to None.
        if self.runner_cfg.eval_disable_interval_events and hasattr(eval_cfgs, "event"):
            from jaxrlworld.rl.configs.base_config import iter_terms
            from jaxrlworld.rl.configs.events.event_term_config import EventTermConfig

            for name, term in iter_terms(eval_cfgs.event, EventTermConfig).items():
                if term.mode in ("interval", "interval_dr"):
                    setattr(eval_cfgs.event, name, None)

        # Disable viewer
        eval_cfgs.visualization.show_viewer = False
        eval_cfgs.visualization.record_video = False

        return eval_cfgs

    def _get_or_create_eval_env(self) -> World:
        """Lazily create eval environment on first use.

        For Gymnasium envs, threads ``self.gym_env_factory`` (when set
        by the user script) into ``_create_env_from_config`` so the
        eval env's wrapper chain matches the training env's exactly.
        """
        if not hasattr(self, "_eval_env") or self._eval_env is None:
            eval_cfgs = self._create_eval_configs()
            self._eval_env = self._create_env_from_config(
                eval_cfgs,
                gym_env_factory=self.gym_env_factory,
            )
        return self._eval_env

    def _pack_obs(self, obs_dict: Dict[str, torch.Tensor], role: str):
        """One model's observation, converted to JAX.

        A model that reads a single observation group gets that group's
        array, which is every algorithm here except a vision policy —
        ``OnPolicyRunner`` overrides this to hand back a dict of groups.
        Every inference path must go through it, training and evaluation
        alike, or the two disagree about what the model is fed.
        """
        return torch_to_jax(obs_dict[role])

    def _run_evaluation(self) -> Dict[str, Any]:
        """Deterministic eval on the default (training-motion) eval env."""
        return self._evaluate_env(self._get_or_create_eval_env())

    def _evaluate_env(self, eval_env: "World") -> Dict[str, Any]:
        """Run the deterministic eval loop on ``eval_env`` and return stats."""
        eval_start = time.time()

        # An eval env is a separate world whose global step counter
        # walks at its own (much slower) pace, so anything scheduled on
        # ``env_step_counter`` — curricula, observation anneals — would
        # run at a stale phase: an annealed privileged term would sit in
        # its hold phase forever and the eval would score a policy the
        # training no longer produces. Mirror the training clock so
        # evaluation happens at the training run's phase.
        eval_env.env_step_counter = self.env.env_step_counter

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
        # Optional per-episode success, only for envs that report
        # ``infos["success"]`` (e.g. ManiSkill). Stays empty for sims that don't,
        # so no ``eval/success_rate`` key is emitted and their logging is
        # unchanged.
        completed_successes: list[float] = []
        success_available = False

        # Per-reward-type tracking
        reward_type_sums: dict[str, torch.Tensor] = {}
        completed_reward_breakdowns: dict[str, list[float]] = {}

        max_steps = int(eval_env.max_episode_length) * 2  # Safety limit
        step = 0

        while len(completed_returns) < target_episodes and step < max_steps:
            # Policy inference
            actor_obs = self._pack_obs(obs_dict, "actor")
            critic_obs = self._pack_obs(obs_dict, "critic")
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

            # Per-episode success (terminal value at the done step), when the env
            # reports it. Absent for sims without a success criterion.
            success_t = infos.get("success", None)
            if success_t is not None:
                success_available = True

            # Collect completed episodes.  One nonzero + one batched
            # transfer: the old per-env ``if dones[i]`` / ``.item()``
            # loop was num_envs blocking syncs per eval step.
            done_idx = dones.nonzero(as_tuple=False).flatten()
            if len(done_idx) > 0 and len(completed_returns) < target_episodes:
                rnames = list(reward_type_sums)
                rows = [episode_returns[done_idx].float(), episode_lengths[done_idx].float()]
                if success_t is not None:
                    rows.append(success_t[done_idx].float())
                rows.extend(reward_type_sums[r][done_idx].float() for r in rnames)
                stacked = torch.stack(rows).cpu().numpy()
                off = 3 if success_t is not None else 2
                for j in range(stacked.shape[1]):
                    if len(completed_returns) >= target_episodes:
                        break
                    completed_returns.append(float(stacked[0, j]))
                    completed_lengths.append(int(stacked[1, j]))
                    if success_t is not None:
                        completed_successes.append(float(stacked[2, j]))
                    for k, rname in enumerate(rnames):
                        completed_reward_breakdowns[rname].append(float(stacked[off + k, j]))

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

        # Success rate -- only emitted when the env reports success, so no key is
        # added (and downstream console/wandb logging is unchanged) for sims
        # without a success criterion.
        if success_available and completed_successes:
            eval_stats["eval/success_rate"] = float(np.mean(completed_successes))

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

        sr = eval_stats.get("eval/success_rate", None)
        sr_str = f"success={sr * 100:.1f}%  " if sr is not None else ""
        print(
            f"\n  {GREEN}[Eval @ iter {it}]{RESET} "
            f"return={mean_ret:.2f} ± {std_ret:.2f}  "
            f"length={mean_len:.1f}  "
            f"{sr_str}"
            f"episodes={n_eps}  "
            f"time={eval_time:.1f}s"
        )

        if self.wandb_logger:
            self.wandb_logger.log_eval_data(eval_stats, step=self.total_timesteps)

    def _get_or_create_heldout_eval_envs(self) -> Dict[str, World]:
        """Lazily build one deterministic eval env per held-out motion set.

        Reads ``runner_cfg.eval_extra_motion_files`` (label -> motion-file
        tuple). Each env reuses ``_create_eval_configs`` (small num_envs,
        noise / interval events / viewer off) and only swaps the motion
        command's ``motion_files`` to the held-out set (uniform sampling,
        since adaptive is single-motion only). Returns ``{}`` when nothing
        is configured or the preset has no "motion" command term.
        """
        if hasattr(self, "_heldout_eval_envs"):
            return self._heldout_eval_envs

        self._heldout_eval_envs: Dict[str, World] = {}
        extra = self.runner_cfg.eval_extra_motion_files
        for label, files in extra.items():
            if not files:
                continue
            cfgs = self._create_eval_configs()
            motion = cfgs.command.terms.get("motion")
            if motion is None:
                continue  # non-tracking preset: nothing to swap
            motion.motion_files = tuple(files)
            motion.sampling_mode = "uniform"
            self._heldout_eval_envs[label] = self._create_env_from_config(cfgs)
        return self._heldout_eval_envs

    def _log_heldout_eval_stats(self, eval_stats: Dict[str, Any], label: str, it: int) -> None:
        """Console + wandb logging for one held-out set, under the
        ``Eval/heldout_<label>/`` namespace (mirrors ``log_eval_data``)."""
        mean_ret = eval_stats.get("eval/mean_return", 0.0)
        mean_len = eval_stats.get("eval/mean_episode_length", 0.0)
        n_eps = eval_stats.get("eval/num_episodes", 0)
        print(
            f"  {GREEN}[Eval/heldout_{label} @ iter {it}]{RESET} "
            f"return={mean_ret:.2f}  length={mean_len:.1f}  episodes={n_eps}"
        )
        if not self.wandb_logger:
            return
        import wandb

        prefix = f"Eval/heldout_{label}"
        core = {
            "eval/mean_return",
            "eval/std_return",
            "eval/min_return",
            "eval/max_return",
            "eval/mean_episode_length",
            "eval/num_episodes",
            "eval/time",
        }
        log_dict: Dict[str, Any] = {}
        for key, val in eval_stats.items():
            if key in core:
                log_dict[f"{prefix}/{key.split('eval/', 1)[-1]}"] = val
            elif key.startswith("eval/reward/"):
                log_dict[f"{prefix}/Rewards/{key.split('eval/reward/')[1]}"] = val
        wandb.log(log_dict, step=self.total_timesteps)

    # ==================== End Evaluation ====================

    def _get_action_statistics(self) -> Dict[str, Any]:
        """Extract action statistics from storage. Override in subclasses."""
        raise NotImplementedError

    def _compute_action_distribution_stats(self, actions: np.ndarray) -> Dict[str, Any]:
        """Compute action distribution statistics from raw actions.

        Gated by ``runner_cfg.logging.action_dist`` (scalar per-dim
        ``mean/std/min/max``) and ``runner_cfg.logging.action_histogram``
        (raw-action array used by the logger to build
        ``wandb.Histogram`` per dim). When both are off the method
        returns an empty dict and callers pass it through unchanged —
        the logger's existing ``if data.action_distribution:`` guard
        naturally skips the entire ``ActionDist/*`` block.

        The two flags are independent because scalars are cheap
        (~100 floats) while histograms copy the full
        ``(num_steps * num_envs, action_dim)`` array into wandb and
        dominate the per-iteration logging cost.
        """
        want_scalars = self.runner_cfg.logging.action_dist
        want_hist = self.runner_cfg.logging.action_histogram

        if not (want_scalars or want_hist):
            return {}

        actions_flat = actions.reshape(-1, actions.shape[-1])
        stats: Dict[str, Any] = {}
        if want_scalars:
            stats["mean"] = actions_flat.mean(axis=0)
            stats["std"] = actions_flat.std(axis=0)
            stats["min"] = actions_flat.min(axis=0)
            stats["max"] = actions_flat.max(axis=0)
        if want_hist:
            stats["raw"] = actions_flat
        return stats

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
        from jaxrlworld.rl.utils.yaml_io import dump_yaml

        os.makedirs(checkpoint_dir, exist_ok=True)

        # 1. Algorithm saves weights (.eqx / .pt files)
        alg_metadata = self.alg.save_train_state(checkpoint_dir)

        # 2. Config → YAML (callables auto-converted to strings)
        dump_yaml(
            os.path.join(checkpoint_dir, "config.yaml"),
            self.cfgs.recursive_to_dict(),
        )

        # 3. Train state → YAML (scalars + metadata only)
        train_state = {
            "runner_class": self.__class__.__name__,
            "algorithm_name": self.algorithm_name,
            "sim_type": self.cfgs.sim_type,
            "iteration": iteration,
            "total_timesteps": self.total_timesteps,
            "total_time": self.total_time,
            "current_learning_iteration": self.current_learning_iteration,
            # The env clock drives curricula, reward schedules and observation
            # anneals; a resumed run continues it rather than restarting at 0.
            "env_step_counter": self.env.env_step_counter,
            "jax_key": np.array(self.key).tolist(),
            "wandb_run_path": self.wandb_logger.run.path if self.wandb_logger else None,
            **alg_metadata,
        }

        # Save the training joint order for cross-sim eval.
        if hasattr(self.env, "act_manager"):
            train_state["canonical_joint_names"] = list(self.env.act_manager.actuated_joint_names)

        dump_yaml(
            os.path.join(checkpoint_dir, "train_state.yaml"),
            train_state,
        )

    def _restore_train_state(self, metadata: dict) -> None:
        """Restore the runner-level state a checkpoint's ``train_state.yaml`` holds.

        Iteration, timestep and wall-time counters, the JAX key, and the
        env clock. A checkpoint written before the env clock was saved
        cannot be resumed faithfully by a preset whose curricula or
        schedules read that clock, so the key is required; add
        ``env_step_counter`` to its ``train_state.yaml`` by hand (0 is
        exact for a preset with nothing clock-driven).
        """
        self.current_learning_iteration = metadata.get("current_learning_iteration", metadata["iteration"])
        self.total_timesteps = metadata["total_timesteps"]
        self.total_time = metadata.get("total_time", 0)
        self.key = jnp.array(metadata["jax_key"], dtype=jnp.uint32)
        if "env_step_counter" not in metadata:
            raise KeyError(
                "train_state.yaml has no 'env_step_counter'; this checkpoint predates the env clock being "
                "saved. Add the key (the number of env steps taken when it was written; 0 if nothing in "
                "the preset is clock-driven) to resume from it."
            )
        self.env.env_step_counter = metadata["env_step_counter"]

    def _save_latest_checkpoint(self, iteration: int) -> None:
        """Save the rolling ``checkpoint_latest``.

        Written to a temp dir first and swapped in with two renames, so a
        complete checkpoint is on disk at every instant: ``checkpoint_latest``
        is absent only between the two renames, during which the previous
        one is intact under ``checkpoint_latest.old``. Leftovers of a run
        interrupted inside this routine (``.tmp``, ``.old``) are cleared
        first, since a non-empty ``.old`` would block the first rename.
        """
        latest_dir = os.path.join(self.model_log_dir, "checkpoint_latest")
        tmp_dir = latest_dir + ".tmp"
        old_dir = latest_dir + ".old"
        for leftover in (tmp_dir, old_dir):
            if os.path.exists(leftover):
                shutil.rmtree(leftover)
        self._save_checkpoint_to(tmp_dir, iteration)
        if os.path.exists(latest_dir):
            os.rename(latest_dir, old_dir)
        os.rename(tmp_dir, latest_dir)
        if os.path.exists(old_dir):
            shutil.rmtree(old_dir)

    def _save_final_checkpoint(self) -> None:
        """Checkpoint the last iteration of learn() when it missed the save grid.

        Iterations run over ``[start, start + N)``, so the last one is on
        the ``save_interval`` grid only by coincidence; a run still ends
        with its final policy on disk.
        """
        last = self.current_learning_iteration - 1
        if last >= self.initial_learning_iteration and last % self.runner_cfg.save_interval != 0:
            self.checkpoint(last)

    def post_iteration(self, data: IterationData, total_iter: int, it: int = 0):
        """Post-iteration processing."""
        self.current_learning_iteration += 1
        self.total_timesteps += self.num_steps_per_env * self.env.num_envs
        self.total_time += data.collection_time + data.learning_time

        if it % self.runner_cfg.log_interval == 0:
            self.log_training_data(data, total_iter=total_iter)

        # In-training evaluation
        eval_interval = self.runner_cfg.eval_interval
        if eval_interval > 0 and it > 0 and it % eval_interval == 0:
            eval_stats = self._run_evaluation()
            self._log_eval_stats(eval_stats, it=it)
            # Held-out (generalization) eval on any configured motion sets.
            for label, hold_env in self._get_or_create_heldout_eval_envs().items():
                self._log_heldout_eval_stats(self._evaluate_env(hold_env), label, it)

        if it % self.runner_cfg.save_interval == 0:
            self.checkpoint(it)

        # The rolling latest checkpoint is a full-parameter D2H plus
        # synchronous disk I/O — every iteration was pure overhead.
        if it % self.runner_cfg.latest_checkpoint_interval == 0:
            self._save_latest_checkpoint(it)
