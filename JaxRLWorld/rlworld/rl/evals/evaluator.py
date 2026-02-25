import atexit
import json
import os
import signal
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Union, TYPE_CHECKING

import gymnasium as gym
import jax
import numpy as np
import torch

from rlworld.rl.envs import (
    GenesisEnv,
    GymnasiumEnv,
    EpisodeStatsCollector
)
from rlworld.rl.envs.utils import NumStepCallsObserver
from rlworld.rl.utils import compare_dicts
from rlworld.rl.utils.checkpoint import load_runner, load_checkpoint_metadata
from rlworld.rl.utils.jax_utils import torch_to_jax, jax_to_torch

if TYPE_CHECKING:
    from rlworld.rl.runners import BaseRunner

ManiSkillEnv = None


# ANSI color codes
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RESET = '\033[0m'


def print_header(text: str, width: int = 70):
    """Print a formatted header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'═' * width}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text.center(width)}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'═' * width}{Colors.RESET}")


def print_subheader(text: str, width: int = 70):
    """Print a formatted subheader."""
    print(f"\n{Colors.BOLD}{Colors.YELLOW}{'─' * width}{Colors.RESET}")
    print(f"{Colors.BOLD}{text}{Colors.RESET}")
    print(f"{Colors.YELLOW}{'─' * width}{Colors.RESET}")


def print_key_value(key: str, value: Any, key_width: int = 20):
    """Print a key-value pair with formatting."""
    print(f"  {Colors.DIM}{key:<{key_width}}{Colors.RESET}: {Colors.GREEN}{value}{Colors.RESET}")


def print_path(label: str, path: str, key_width: int = 20):
    """Print a path with both relative and absolute versions."""
    abs_path = os.path.abspath(path)
    print(f"  {Colors.DIM}{label:<{key_width}}{Colors.RESET}: {Colors.GREEN}{path}{Colors.RESET}")
    print(f"  {Colors.DIM}{'':<{key_width}}{Colors.RESET}  {Colors.DIM}({abs_path}){Colors.RESET}")


def print_success(text: str):
    """Print a success message."""
    print(f"{Colors.GREEN}✓ {text}{Colors.RESET}")


def print_warning(text: str):
    """Print a warning message."""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.RESET}")


def print_error(text: str):
    """Print an error message."""
    print(f"{Colors.RED}✗ {text}{Colors.RESET}")


def print_info(text: str):
    """Print an info message."""
    print(f"{Colors.CYAN}ℹ {text}{Colors.RESET}")


def print_progress(current: int, total: int, prefix: str = "", suffix: str = "", width: int = 40):
    """Print a progress bar."""
    percent = current / total
    filled = int(width * percent)
    bar = '█' * filled + '░' * (width - filled)
    print(f"\r  {prefix} {Colors.CYAN}│{bar}│{Colors.RESET} {percent * 100:5.1f}% {suffix}", end='', flush=True)


# ==================== Policy Wrappers ====================


class PolicyWrapper(ABC):
    """
    Base inference wrapper for evaluation.

    Subclasses implement get_action for different algorithm families:
    - ModelPolicyWrapper: PPO / SAC / FastTD3 (JIT-compiled model inference)
    - MPCPolicyWrapper:   TD-MPC2 / ScaffoldedTDMPC2 (MPPI planning)

    Use PolicyWrapper.from_runner() factory to create the appropriate subclass.
    """

    def __init__(self, runner: "BaseRunner", device: torch.device):
        self.device = device
        self.is_squashed = runner.squash_output
        if self.is_squashed:
            self.action_scale = runner.action_scale
            self.action_bias = runner.action_bias

    @classmethod
    def from_runner(
        cls, runner: "BaseRunner", device: torch.device,
    ) -> "PolicyWrapper":
        """Factory: returns appropriate subclass based on algorithm type."""
        if hasattr(runner.alg, 'act_with_t0'):
            return MPCPolicyWrapper(runner, device)
        return ModelPolicyWrapper(runner, device)

    def _process_action(self, actions: jax.Array) -> jax.Array:
        """Apply action rescaling for squashed policies."""
        if self.is_squashed:
            return actions * self.action_scale + self.action_bias
        return actions

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

    def __init__(self, runner: "BaseRunner", device: torch.device):
        super().__init__(runner, device)
        model = runner.alg.train_state.model
        self._key = jax.random.PRNGKey(0)

        def _single(obs, key):
            action, _ = model.act_inference(obs, key=key)
            return action

        self._inference_fn = jax.jit(
            jax.vmap(_single, in_axes=(0, None))
        )

    def get_action(self, env_obs, robot_states, deterministic=True):
        actor_obs = torch_to_jax(env_obs["actor"])
        action_jax = self._inference_fn(actor_obs, self._key)
        return jax_to_torch(self._process_action(action_jax), self.device)


class MPCPolicyWrapper(PolicyWrapper):
    """MPPI planning for TD-MPC2 / ScaffoldedTDMPC2."""

    def __init__(self, runner: "BaseRunner", device: torch.device):
        super().__init__(runner, device)
        self._runner = runner
        self._t0_mask = np.ones(
            runner.alg._prev_mean.shape[0], dtype=bool
        )

    def get_action(self, env_obs, robot_states, deterministic=True):
        actor_obs = torch_to_jax(env_obs["actor"])
        action_jax = self._runner.alg.act_with_t0(
            obs=actor_obs, t0_mask=self._t0_mask, eval_mode=True,
        )
        self._t0_mask[:] = False
        return jax_to_torch(self._process_action(action_jax), self.device)

    def notify_reset(self, reset_mask: np.ndarray) -> None:
        """Mark environments that need MPPI warm-start reset."""
        self._t0_mask[reset_mask] = True


class PolicyEvaluator(NumStepCallsObserver):
    """
    Evaluates trained policies by loading checkpoints and running episodes.
    Supports Genesis, Newton, ManiSkill, and Gymnasium environments.
    """

    def __init__(
        self,
        eval_env_cfgs: dict | None,
        policy_path: str,
        num_evals: int = 5,
        seed: int = 42,
        use_logging: bool = True,
        show_viewer: bool = False,
        record_video: bool = False,
        save_data: bool = True,
        record_steps: int | None = 1000,
        video_dir: str | None = None,
        extra_overrides: dict = None,
        use_rich_display: bool = True
    ):
        """
        Initialize the evaluator.

        Args:
            eval_env_cfgs: Environment configurations for evaluation
            policy_path: Path to the saved policy checkpoint directory
            num_evals: Number of evaluation episodes to run
            seed: Random seed
            use_logging: Whether to use logging
            show_viewer: Whether to show viewer
            record_video: Whether to record video
            save_data: Whether to save evaluation data
            record_steps: Number of steps to record
            video_dir: Video directory path
            extra_overrides: Extra config overrides
            use_rich_display: Whether to use rich display
        """
        super().__init__()
        self.policy_path = policy_path
        self.num_evals = num_evals
        self.seed = seed

        self.use_logging = use_logging
        self.show_viewer = show_viewer
        self.record_video = record_video
        self.use_rich_display = use_rich_display
        self.video_dir = None
        self.record_steps = record_steps
        self.recording_started = False

        # Load metadata to determine simulator type
        metadata = load_checkpoint_metadata(policy_path)
        self.sim_type = self._detect_sim_type(metadata)

        # Initialize device based on simulator
        self.device = self._init_device()

        if record_video:
            self.video_dir = self._generate_video_dir(policy_path, record_video, video_dir)
            atexit.register(self.cleanup)
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)

        # Load and prepare configurations
        self.eval_cfgs = self._prepare_configs(policy_path, eval_env_cfgs, extra_overrides, metadata)

        # Initialize environment
        self.env = self._init_environment()
        self.env.reset()

        # Load runner and build policy wrapper
        self.runner = load_runner(
            env=self.env,
            checkpoint_path=policy_path,
            cfgs=self.eval_cfgs,
            use_wandb=False,
        )
        self.runner.set_eval_mode()
        self._resize_mppi_state()
        self.policy = PolicyWrapper.from_runner(self.runner, self.device)

        # Episode tracker
        self.episode_tracker = EpisodeStatsCollector(
            num_envs=self.env.num_envs,
            max_episode_length=self.env.max_episode_length,
            device=self.env.device,
            gamma=self.eval_cfgs.algorithm.gamma,
            window_size=self.env.num_envs + 100
        )

        # Setup evaluation tools
        self.save_data = save_data
        self.eval_results_dir = None
        self.episode_data = []
        self._setup_evaluation_tools()

    def _resize_mppi_state(self) -> None:
        """Resize MPPI prev_mean to match eval num_envs (may differ from training)."""
        if not hasattr(self.runner.alg, '_prev_mean'):
            return
        _, horizon, action_dim = self.runner.alg._prev_mean.shape
        self.runner.alg._prev_mean = np.zeros(
            (self.env.num_envs, horizon, action_dim), dtype=np.float32,
        )

    def _detect_sim_type(self, metadata: dict) -> str:
        """Detect simulator type from checkpoint metadata."""
        env_name = metadata.get('config', {}).get('env', {}).get('env_name', '')
        if "Genesis" in env_name:
            return "Genesis"
        elif "Newton" in env_name:
            return 'Newton'
        elif "MjlabEnv" in env_name:
            return "MjlabEnv"
        elif env_name == 'Maniskill':
            return 'ManiSkill'
        elif env_name == 'Gymnasium':
            return 'Gymnasium'
        else:
            return 'Genesis'

    def _init_device(self) -> torch.device:
        if self.sim_type == 'Newton':
            import warp as wp
            from warp.torch import device_to_torch
            return device_to_torch(wp.get_device())
        elif self.sim_type == 'MjlabEnv':
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            import genesis as gs
            return gs.device

    def _signal_handler(self, sig, frame):
        """Handle signals by cleaning up and exiting."""
        print_warning(f"Signal {sig} received. Saving video and exiting...")
        self.cleanup()
        exit(0)

    def cleanup(self):
        """Perform cleanup tasks - save video etc."""
        if self.sim_type == 'ManiSkill':
            return
        elif self.sim_type == 'Genesis':
            if hasattr(self, 'env') and self.recording_started:
                print_info("Saving video before exit...")
                print(f"[DEBUG] Recorded frames: {len(self.env.vis_manager._recorded_frames)}")
                try:
                    self.env.vis_manager.stop_recording()
                    print_success("Video saved successfully!")
                except Exception as e:
                    print_error(f"Error saving video: {e}")
        elif self.sim_type == 'Newton':
            if hasattr(self, 'env') and hasattr(self.env, 'vis_manager'):
                print_info("Closing Newton viewer...")
                try:
                    self.env.vis_manager.close()
                    print_success("Newton viewer closed!")
                except Exception as e:
                    print_error(f"Error closing viewer: {e}")

        elif self.sim_type == 'MjlabEnv':
            if hasattr(self, 'env') and hasattr(self.env, 'visualization_manager'):
                print_info("Closing Mjlab viewer...")
                try:
                    self.env.visualization_manager.close()
                    print_success("Mjlab viewer closed!")
                except Exception as e:
                    print_error(f"Error closing viewer: {e}")

    def _generate_video_dir(self, policy_path: str, record_video: bool, video_dir: str | None) -> str:
        """Generate the video directory path based on policy_path if needed."""
        if record_video and video_dir is None:
            dir_path = os.path.dirname(policy_path)
            filename = os.path.basename(policy_path)
            step_str = ''.join(filter(str.isdigit, filename))
            video_subdir = os.path.join(dir_path, "videos")
            os.makedirs(video_subdir, exist_ok=True)

            # Newton uses .bin, Genesis uses .mp4
            if self.sim_type == 'Newton':
                video_filename = f"step{step_str}.bin"
            else:
                video_filename = f"step{step_str}.mp4"

            video_dir = os.path.join(video_subdir, video_filename)
        return video_dir

    def _prepare_configs(
        self,
        policy_path: str,
        eval_env_cfgs: dict,
        extra_overrides: dict,
        metadata: dict
    ):
        """Prepare evaluation configurations by loading saved configs and updating env configs."""
        if self.sim_type == 'Newton':
            from rlworld.rl.configs.newton_config_classes import NewtonConfigsForRun, NewtonEnvConfig
            train_cfgs = NewtonConfigsForRun.from_dict(metadata['config'])
            EnvConfigClass = NewtonEnvConfig

        elif self.sim_type == "Genesis":
            from rlworld.rl.configs.genesis_config_classes import GenesisConfigsForRun, EnvConfig
            train_cfgs = GenesisConfigsForRun.from_dict(metadata['config'])
            EnvConfigClass = EnvConfig

        elif self.sim_type == 'MjlabEnv':
            from rlworld.rl.configs.mujoco_config_classes import MujocoConfigsForRun, MujocoEnvConfig
            train_cfgs = MujocoConfigsForRun.from_dict(metadata['config'])
            EnvConfigClass = MujocoEnvConfig

        if eval_env_cfgs is not None:
            compare_dicts(eval_env_cfgs, train_cfgs.env.to_dict(), "eval_env_cfgs", "train_cfgs.env")
            eval_cfgs = train_cfgs
            eval_cfgs.env = EnvConfigClass.from_dict(eval_env_cfgs)
        else:
            eval_cfgs = train_cfgs

        if extra_overrides is not None:
            eval_cfgs.apply_overrides(**extra_overrides)

        eval_cfgs.visualization.show_viewer = self.show_viewer
        eval_cfgs.visualization.record_video = self.record_video
        eval_cfgs.visualization.video_dir = self.video_dir

        return eval_cfgs

    def _init_environment(self) -> Union[GenesisEnv, "NewtonEnv", "ManiSkillEnv", GymnasiumEnv]:
        """Initialize the evaluation environment with prepared configs."""
        if self.sim_type == 'Newton':
            return self._init_newton_env()
        elif self.sim_type == 'MjlabEnv':
            return self._init_mjlab_env()
        elif self.sim_type == 'ManiSkill':
            return self._init_maniskill_env()
        elif self.sim_type == 'Gymnasium':
            return self._init_gymnasium_env()
        else:
            return self._init_genesis_env()

    def _init_newton_env(self):
        """Initialize Newton environment."""
        from rlworld.rl.envs import NewtonEnv, NewtonLocomotionEnv
        return NewtonLocomotionEnv(
            num_envs=self.eval_cfgs.env.num_envs,
            env_cfg=self.eval_cfgs.env,
            scene_cfg=self.eval_cfgs.scene,
            visualization_cfg=self.eval_cfgs.visualization,
            obs_cfg=self.eval_cfgs.observation,
            act_cfg=self.eval_cfgs.action,
            reward_cfg=self.eval_cfgs.reward,
            command_cfg=self.eval_cfgs.command,
            event_cfg=self.eval_cfgs.event,
        )

    def _init_genesis_env(self):
        """Initialize Genesis environment."""
        from rlworld.rl import envs
        env_class_name = self.eval_cfgs.env.env_name

        if hasattr(envs, env_class_name):
            env_class = getattr(envs, env_class_name)
            return env_class(
                num_envs=self.eval_cfgs.env.num_envs,
                env_cfg=self.eval_cfgs.env,
                scene_cfg=self.eval_cfgs.scene,
                visualization_cfg=self.eval_cfgs.visualization,
                obs_cfg=self.eval_cfgs.observation,
                act_cfg=self.eval_cfgs.action,
                reward_cfg=self.eval_cfgs.reward,
                command_cfg=self.eval_cfgs.command,
                event_cfg=self.eval_cfgs.event,
            )
        raise NotImplementedError(f"Undefined env class name {env_class_name}")

    def _init_mjlab_env(self):
        """Initialize MjlabEnv environment."""
        from rlworld.rl.envs import MjlabEnv
        if self.eval_cfgs.scene.mjlab_scene_cfg is None:
            raise ValueError(
                "mjlab_scene_cfg is required for MjlabEnv evaluation but was not found "
                "in the checkpoint. Provide it via extra_overrides, e.g.:\n"
                "  PolicyEvaluator(\n"
                "      ...,\n"
                "      extra_overrides={'scene': {'mjlab_scene_cfg': your_config.scene.mjlab_scene_cfg}},\n"
                "  )"
            )
        return MjlabEnv(
            num_envs=self.eval_cfgs.env.num_envs,
            env_cfg=self.eval_cfgs.env,
            scene_cfg=self.eval_cfgs.scene,
            visualization_cfg=self.eval_cfgs.visualization,
            obs_cfg=self.eval_cfgs.observation,
            act_cfg=self.eval_cfgs.action,
            reward_cfg=self.eval_cfgs.reward,
            command_cfg=self.eval_cfgs.command,
            event_cfg=self.eval_cfgs.event,
        )

    def _init_maniskill_env(self):
        """Initialize ManiSkill environment."""
        from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
        from mani_skill.utils.wrappers.record import RecordEpisode
        from rlworld.rl.envs import ManiSkillEnv

        env_kwargs = self.eval_cfgs.env.gym_make_kwargs
        env = gym.make(self.eval_cfgs.env.task_name, num_envs=self.eval_cfgs.env.num_envs, **env_kwargs)

        if self.record_video:
            video_dir_only = os.path.dirname(self.video_dir)
            env = RecordEpisode(
                env,
                output_dir=video_dir_only,
                save_trajectory=False,
                save_video=True,
                max_steps_per_video=self.record_steps,
                video_fps=30
            )

        env = ManiSkillVectorEnv(env, self.eval_cfgs.env.num_envs, auto_reset=True, ignore_terminations=False)
        env = ManiSkillEnv(
            env,
            env_cfg=self.eval_cfgs.env,
            scene_cfg=self.eval_cfgs.scene,
            obs_cfg=self.eval_cfgs.observation,
            act_cfg=self.eval_cfgs.action,
            reward_cfg=self.eval_cfgs.reward,
            command_cfg=self.eval_cfgs.command,
            seed=self.seed
        )
        return env

    def _init_gymnasium_env(self):
        """Initialize Gymnasium environment."""
        from rlworld.rl.envs import GymnasiumEnv
        from gymnasium.vector import SyncVectorEnv

        def make_env(seed):
            def _init():
                return gym.make(self.eval_cfgs.env.task_name, max_episode_steps=100)

            return _init

        num_envs = self.eval_cfgs.env.num_envs
        env_gym = SyncVectorEnv([make_env(i) for i in range(num_envs)])
        env = GymnasiumEnv(
            env_gym,
            env_cfg=self.eval_cfgs.env,
            scene_cfg=self.eval_cfgs.scene,
            obs_cfg=self.eval_cfgs.observation,
            act_cfg=self.eval_cfgs.action,
            reward_cfg=self.eval_cfgs.reward,
            command_cfg=self.eval_cfgs.command,
            seed=self.seed
        )
        return env

    def _setup_evaluation_tools(self):
        """Initialize logger and results directory."""
        if self.policy_path:
            base_dir = Path(self.policy_path).parent / "eval_results"
        else:
            base_dir = Path("eval_results")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = base_dir / timestamp
        results_dir.mkdir(parents=True, exist_ok=True)
        self.eval_results_dir = results_dir

    def evaluate(self):
        """Entry point that dispatches to appropriate evaluator."""
        if self.sim_type == 'ManiSkill':
            return self._evaluate_maniskill()
        elif self.sim_type in ('Genesis', 'Newton', 'MjlabEnv'):
            return self._evaluate_physics_sim()
        else:
            raise NotImplementedError(f"Unsupported sim_type: {self.sim_type}")

    def _print_eval_config(self):
        """Print evaluation configuration."""
        print_subheader("Configuration")
        print_key_value("Simulator", self.sim_type)
        print_key_value("Num Environments", self.env.num_envs)
        print_key_value("Max Steps", self.env.max_episode_length)
        print_key_value("Seed", self.seed)

        print_subheader("Paths")
        print_path("Policy", self.policy_path)
        print_path("Results Dir", str(self.eval_results_dir))
        if self.record_video and self.video_dir:
            video_dir_only = os.path.dirname(self.video_dir)
            print_path("Video Dir", video_dir_only)

    def _evaluate_maniskill(self):
        """ManiSkill environment evaluation (RecordEpisode handles video)."""
        print_header("ManiSkill Evaluation")
        self._print_eval_config()

        obs = self.env.obs_manager.get_observation()
        robot_states = self.env.get_robot_state()

        try:
            print_subheader("Running Evaluation")
            new_obs, new_robot_states, eval_stats = self._run_evaluation_loop(
                obs, robot_states, track_success=True
            )

            self._print_evaluation_summary(eval_stats)

            if self.save_data:
                self._save_evaluation_results(eval_stats)

            if self.record_video:
                print_info("Saving ManiSkill video...")
                self.env.gym_env.close()
                print_success("Video saved!")

            print_header("Evaluation Complete")
            return eval_stats

        except Exception as e:
            print_error(f"Exception during evaluation: {e}")
            raise

    def _evaluate_physics_sim(self):
        """Genesis/Newton environment evaluation."""
        print_header(f"{self.sim_type} Evaluation")
        self._print_eval_config()

        obs = self.env.obs_manager.get_observation()
        robot_states = self.env.get_robot_state()

        if self.record_video:
            if self.sim_type == 'Genesis':
                self.env.vis_manager.start_recording()
                self.recording_started = True
                print_info("Video recording started")
            elif self.sim_type == 'Newton':
                # Newton ViewerFile records automatically, nothing to do
                print_info("Newton recording active")

        try:
            print_subheader("Running Evaluation")
            new_obs, new_robot_states, eval_stats = self._run_evaluation_loop(
                obs, robot_states, track_success=False
            )

            self._print_evaluation_summary(eval_stats)

            if self.save_data:
                self._save_evaluation_results(eval_stats)

            # Stop recording here
            if self.record_video and self.sim_type == 'Newton':
                self.env.vis_manager.stop_recording()
                print_success("Newton recording saved!")

            print_header("Evaluation Complete")
            return eval_stats

        except Exception as e:
            print_error(f"Exception during evaluation: {e}")
            self.cleanup()
            raise

    @torch.no_grad()
    def _run_evaluation_loop(
        self,
        obs: torch.Tensor,
        robot_states: torch.Tensor | None,
        track_success: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, Any]]:
        """
        Unified evaluation loop for all physics simulators.
        Each environment completes 1 episode.
        """
        current_step = 0
        eval_dones = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.device)

        if track_success:
            episode_success = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.device)

        while not torch.all(eval_dones):
            action = self.policy.get_action(obs, robot_states)

            obs, rewards, terminated, truncated, infos = self.env.step(action)
            reset_idx = terminated | truncated
            robot_states = self.env.get_robot_state()
            current_step += 1

            reward_info = infos["rewards_per_type"]
            new_dones = reset_idx & ~eval_dones

            self.episode_tracker.update(reward_info, new_dones)

            if new_dones.any():
                if track_success and "success" in infos:
                    episode_success[new_dones] = infos["success"][new_dones]
                eval_dones[new_dones] = True
                self.policy.notify_reset(new_dones.cpu().numpy())

            if current_step % 50 == 0:
                completed = torch.sum(eval_dones).item()
                print_progress(
                    completed, self.env.num_envs,
                    prefix="Progress",
                    suffix=f"({current_step} steps)"
                )

            # Handle video recording stop for Genesis
            if (self.record_steps is not None and
                self.sim_type == 'Genesis' and
                self.env_step_calls >= self.record_steps - 1):
                self.env.vis_manager.stop_recording()
                self.record_steps = None
                print_success("Video recording stopped")

        print()  # New line after progress bar
        eval_stats = self._extract_evaluation_stats()

        if track_success:
            success_list = episode_success.cpu().tolist()
            eval_stats['successes'] = success_list
            eval_stats['success_rate'] = float(np.mean(success_list))

        return obs, robot_states, eval_stats

    def _extract_evaluation_stats(self) -> dict[str, Any]:
        """Extract statistics from completed episodes across all environments."""
        returns = self.episode_tracker.get_return_history()
        discounted_returns = self.episode_tracker.get_discounted_return_history()
        lengths = self.episode_tracker.get_length_history()

        returns_per_type: dict[str, list[float]] = {}
        for reward_type in self.episode_tracker.return_history_per_type.keys():
            returns_per_type[reward_type] = self.episode_tracker.get_return_history_per_type(
                reward_type
            )

        if len(returns) > 0:
            mean_return = float(np.mean(returns))
            std_return = float(np.std(returns))
            min_return = float(np.min(returns))
            max_return = float(np.max(returns))

            mean_discounted_return = float(np.mean(discounted_returns))
            std_discounted_return = float(np.std(discounted_returns))
            min_discounted_return = float(np.min(discounted_returns))
            max_discounted_return = float(np.max(discounted_returns))

            mean_length = float(np.mean(lengths))
            std_length = float(np.std(lengths))
            min_length = float(np.min(lengths))
            max_length = float(np.max(lengths))
        else:
            mean_return = std_return = min_return = max_return = 0.0
            mean_discounted_return = std_discounted_return = min_discounted_return = max_discounted_return = 0.0
            mean_length = std_length = min_length = max_length = 0.0

        reward_breakdown: dict[str, dict[str, float]] = {}
        for reward_type, type_returns in returns_per_type.items():
            if len(type_returns) > 0:
                reward_breakdown[reward_type] = {
                    'mean': float(np.mean(type_returns)),
                    'std': float(np.std(type_returns)),
                    'min': float(np.min(type_returns)),
                    'max': float(np.max(type_returns))
                }

        return {
            'mean_return': mean_return,
            'std_return': std_return,
            'min_return': min_return,
            'max_return': max_return,
            'mean_discounted_return': mean_discounted_return,
            'std_discounted_return': std_discounted_return,
            'min_discounted_return': min_discounted_return,
            'max_discounted_return': max_discounted_return,
            'gamma': self.episode_tracker.gamma,
            'mean_length': mean_length,
            'std_length': std_length,
            'min_length': min_length,
            'max_length': max_length,
            'num_episodes': len(returns),
            'num_envs': self.env.num_envs,
            'returns': returns,
            'discounted_returns': discounted_returns,
            'lengths': lengths,
            'reward_breakdown': reward_breakdown
        }

    def _print_evaluation_summary(self, stats: dict[str, Any]) -> None:
        """Print evaluation summary with colors."""
        print_subheader("Results Summary")

        # Episode stats
        print(f"\n  {Colors.BOLD}Episodes{Colors.RESET}")
        print(f"    Completed: {Colors.GREEN}{stats['num_episodes']}{Colors.RESET} / {stats['num_envs']}")

        # Success rate (if available)
        if 'success_rate' in stats:
            success_pct = stats['success_rate'] * 100
            print(f"\n  {Colors.BOLD}Success Rate{Colors.RESET}")
            print(
                f"    Rate: {Colors.GREEN}{success_pct:5.1f}%{Colors.RESET} "
                f"({sum(stats['successes'])}/{len(stats['successes'])})"
            )

        # Return stats
        print(f"\n  {Colors.BOLD}Return{Colors.RESET}")
        print(f"    Mean ± Std: {Colors.GREEN}{stats['mean_return']:8.2f}{Colors.RESET} ± {stats['std_return']:.2f}")
        print(
            f"    Range:      [{Colors.CYAN}{stats['min_return']:8.2f}{Colors.RESET}, "
            f"{Colors.CYAN}{stats['max_return']:8.2f}{Colors.RESET}]"
        )

        print(f"\n  {Colors.BOLD}Discounted Return (γ={stats['gamma']}){Colors.RESET}")
        print(
            f"    Mean ± Std: {Colors.GREEN}{stats['mean_discounted_return']:8.2f}{Colors.RESET} "
            f"± {stats['std_discounted_return']:.2f}"
        )
        print(
            f"    Range:      [{Colors.CYAN}{stats['min_discounted_return']:8.2f}{Colors.RESET}, "
            f"{Colors.CYAN}{stats['max_discounted_return']:8.2f}{Colors.RESET}]"
        )

        # Length stats
        print(f"\n  {Colors.BOLD}Episode Length{Colors.RESET}")
        print(f"    Mean ± Std: {Colors.GREEN}{stats['mean_length']:8.1f}{Colors.RESET} ± {stats['std_length']:.1f}")
        print(
            f"    Range:      [{Colors.CYAN}{stats['min_length']:8.0f}{Colors.RESET}, "
            f"{Colors.CYAN}{stats['max_length']:8.0f}{Colors.RESET}]"
        )

        # Reward breakdown
        if stats['reward_breakdown']:
            print(f"\n  {Colors.BOLD}Reward Breakdown{Colors.RESET}")
            print(f"    {'Type':<25} {'Mean':>10} {'± Std':>10} {'Min':>10} {'Max':>10}")
            print(f"    {'-' * 65}")
            for reward_type, type_stats in sorted(stats['reward_breakdown'].items()):
                print(
                    f"    {reward_type:<25} {Colors.GREEN}{type_stats['mean']:>10.2f}{Colors.RESET} "
                    f"± {type_stats['std']:>7.2f} "
                    f"{Colors.DIM}{type_stats['min']:>10.2f} {type_stats['max']:>10.2f}{Colors.RESET}"
                )

    def _save_evaluation_results(self, stats: dict[str, Any]) -> None:
        """Save evaluation results to file."""
        if self.eval_results_dir is None:
            return

        results_file = self.eval_results_dir / "evaluation_results.json"

        # Convert numpy types to Python native types
        def convert_to_native(obj):
            if isinstance(obj, (np.floating, np.integer)):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, list):
                return [convert_to_native(x) for x in obj]
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            return obj

        save_data = {
            'summary': {
                'simulator': self.sim_type,
                # Undiscounted return
                'mean_return': stats['mean_return'],
                'std_return': stats['std_return'],
                'min_return': stats['min_return'],
                'max_return': stats['max_return'],
                # Discounted return
                'mean_discounted_return': stats.get('mean_discounted_return'),
                'std_discounted_return': stats.get('std_discounted_return'),
                'min_discounted_return': stats.get('min_discounted_return'),
                'max_discounted_return': stats.get('max_discounted_return'),
                'gamma': stats.get('gamma'),
                # Episode length
                'mean_length': stats['mean_length'],
                'std_length': stats['std_length'],
                'min_length': stats['min_length'],
                'max_length': stats['max_length'],
                # Counts
                'num_episodes': stats['num_episodes'],
                'num_envs': stats['num_envs'],
                # Success rate
                'success_rate': stats.get('success_rate'),
                'num_successes': sum(stats['successes']) if 'successes' in stats else None,
            },
            'reward_breakdown': stats['reward_breakdown'],
            'episode_returns': stats['returns'],
            'episode_discounted_returns': stats.get('discounted_returns'),
            'episode_lengths': stats['lengths'],
            'episode_successes': stats.get('successes'),
            'metadata': {
                'policy_path': os.path.abspath(self.policy_path),
                'timestamp': datetime.now().isoformat(),
                'seed': self.seed
            }
        }

        save_data = convert_to_native(save_data)

        with open(results_file, 'w') as f:
            json.dump(save_data, f, indent=2)

        print_success(f"Results saved to: {results_file}")
        print(f"         {Colors.DIM}({os.path.abspath(results_file)}){Colors.RESET}")
