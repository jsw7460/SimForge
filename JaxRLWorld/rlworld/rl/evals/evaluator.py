import atexit
import json
import os
import signal
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rlworld.rl.envs import EpisodeStatsCollector
from rlworld.rl.envs.utils import NumStepCallsObserver
from rlworld.rl.evals.policy_wrappers import PolicyWrapper
from rlworld.rl.evals.sim_initializers import detect_sim_type, get_initializer
from rlworld.rl.utils.checkpoint import load_runner, load_checkpoint_metadata
from rlworld.rl.utils.console import (
    Colors,
    print_header,
    print_subheader,
    print_key_value,
    print_path,
    print_success,
    print_warning,
    print_error,
    print_info,
    print_progress,
)


class PolicyEvaluator(NumStepCallsObserver):
    """
    Evaluates trained policies by loading checkpoints and running episodes.
    Supports Genesis, Newton, MjlabEnv, ManiSkill, and Gymnasium environments.
    """

    def __init__(
        self,
        eval_env_cfgs: dict | None,
        policy_path: str | None = None,
        wandb_run_path: str | None = None,
        wandb_checkpoint_iter: int | None = None,
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
        super().__init__()

        # Resolve policy_path from wandb if needed
        if wandb_run_path is not None:
            from rlworld.rl.utils.wandb_checkpoint import get_wandb_checkpoint
            policy_path, was_cached = get_wandb_checkpoint(
                wandb_run_path=wandb_run_path,
                iteration=wandb_checkpoint_iter,
            )
            status = "cached" if was_cached else "downloaded"
            print_info(f"Using {status} wandb checkpoint: {policy_path}")

        if policy_path is None:
            raise ValueError("Either policy_path or wandb_run_path must be provided.")

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
        self.sim_type = detect_sim_type(metadata)
        self._init = get_initializer(self.sim_type)

        # Initialize device based on simulator
        self.device = self._init.init_device()

        if record_video:
            self.video_dir = self._generate_video_dir(policy_path, record_video, video_dir)
            atexit.register(self.cleanup)
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)

        # Load and prepare configurations
        self.eval_cfgs = self._init.prepare_configs(
            policy_path=policy_path,
            eval_env_cfgs=eval_env_cfgs,
            extra_overrides=extra_overrides,
            metadata=metadata,
            show_viewer=show_viewer,
            record_video=record_video,
            video_dir=self.video_dir,
        )

        # Apply eval-mode defaults: disable obs noise, remove interval events
        self._apply_eval_defaults()

        # Initialize environment
        self.env = self._init.init_environment(
            self.eval_cfgs,
            record_video=record_video,
            video_dir=self.video_dir,
            record_steps=record_steps,
            seed=seed,
        )
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

    def _apply_eval_defaults(self) -> None:
        """Apply evaluation-mode defaults to configs.

        - Disables observation noise (enable_noise = False)
        - Removes interval events (e.g. push_by_setting_velocity)
        """
        # Disable observation noise
        if hasattr(self.eval_cfgs, 'observation'):
            self.eval_cfgs.observation.enable_noise = False

        # Remove interval events (external forces, etc.)
        if hasattr(self.eval_cfgs, 'event'):
            event_cfg = self.eval_cfgs.event
            if hasattr(event_cfg, 'event_terms'):
                event_cfg.event_terms = [
                    t for t in event_cfg.event_terms
                    if t.mode != "interval"
                ]

    def _resize_mppi_state(self) -> None:
        """Resize MPPI prev_mean to match eval num_envs (may differ from training)."""
        if not hasattr(self.runner.alg, '_prev_mean'):
            return
        _, horizon, action_dim = self.runner.alg._prev_mean.shape
        self.runner.alg._prev_mean = np.zeros(
            (self.env.num_envs, horizon, action_dim), dtype=np.float32,
        )

    def _signal_handler(self, sig, frame):
        """Handle signals by cleaning up and exiting."""
        print_warning(f"Signal {sig} received. Saving video and exiting...")
        self.cleanup()
        exit(0)

    def cleanup(self):
        """Perform cleanup tasks — delegate to initializer."""
        if hasattr(self, 'env') and hasattr(self, '_init'):
            self._init.cleanup(self.env)

    def _generate_video_dir(self, policy_path: str, record_video: bool, video_dir: str | None) -> str:
        """Generate the video directory path based on policy_path if needed."""
        if record_video and video_dir is None:
            dir_path = os.path.dirname(policy_path)
            filename = os.path.basename(policy_path)
            step_str = ''.join(filter(str.isdigit, filename))
            video_subdir = os.path.join(dir_path, "videos")
            os.makedirs(video_subdir, exist_ok=True)

            ext = self._init.video_extension
            video_filename = f"step{step_str}{ext}"
            video_dir = os.path.join(video_subdir, video_filename)
        return video_dir

    def _setup_evaluation_tools(self):
        """Initialize results directory."""
        if self.policy_path:
            base_dir = Path(self.policy_path).parent / "eval_results"
        else:
            base_dir = Path("eval_results")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = base_dir / timestamp
        results_dir.mkdir(parents=True, exist_ok=True)
        self.eval_results_dir = results_dir

    def evaluate(self):
        """Run evaluation."""
        print_header(f"{self.sim_type} Evaluation")
        self._print_eval_config()

        obs = self.env.obs_manager.get_observation()
        robot_states = self.env.get_robot_state()

        if self.record_video:
            self._init.start_recording(self.env)
            self.recording_started = True

        try:
            print_subheader("Running Evaluation")
            new_obs, new_robot_states, eval_stats = self._run_evaluation_loop(
                obs, robot_states,
                track_success=self._init.supports_success_tracking,
            )

            self._print_evaluation_summary(eval_stats)

            if self.save_data:
                self._save_evaluation_results(eval_stats)

            if self.record_video:
                self._init.stop_recording(self.env)

            print_header("Evaluation Complete")
            return eval_stats

        except Exception as e:
            print_error(f"Exception during evaluation: {e}")
            self.cleanup()
            raise

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

    @torch.no_grad()
    def _run_evaluation_loop(
        self,
        obs: torch.Tensor,
        robot_states: torch.Tensor | None,
        track_success: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, Any]]:
        """
        Unified evaluation loop for all simulators.
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

            # Handle mid-episode recording stop for Genesis
            if (self.record_steps is not None
                and self.sim_type == 'Genesis'
                and self.env_step_calls >= self.record_steps - 1):
                self._init.stop_recording(self.env)
                self.record_steps = None

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
