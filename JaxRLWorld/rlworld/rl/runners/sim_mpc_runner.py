"""
SimMPCRunner: Runner for SimMPC (real-simulator MPPI + policy training).

Creates a planning environment with the same config as the training env
but with num_envs = num_samples, then runs MPPI planning each step
and trains policy + Q-ensemble from collected experience.
"""

import os

import time
from typing import Dict, List, Any

import numpy as np
import torch

from rlworld.rl.algorithms.sim_mpc import SimMPC
from rlworld.rl.configs import ConfigsForRun, configs_from_dict
from rlworld.rl.envs import World
from rlworld.rl.runners.base_runner import BaseRunner
from rlworld.rl.runners.iteration_data import IterationData


class SimMPCRunner(BaseRunner):
    """Runner for SimMPC (MPPI with real simulator + learned policy)."""

    alg: SimMPC
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
        """Create the planning environment and state sync."""
        from rlworld.rl.algorithms.sim_mpc.state_sync import GenesisStateSync

        alg_cfg = self.cfgs.algorithm

        # Create planning env with same config but num_envs = num_samples
        plan_cfgs = deepcopy(self.cfgs)
        plan_cfgs.env.num_envs = alg_cfg.num_samples
        plan_cfgs.visualization.show_viewer = False
        plan_cfgs.visualization.record_video = False

        self.planning_env = self._create_env_from_config(plan_cfgs)
        self.state_sync = GenesisStateSync(self.env, self.planning_env)

        # Compute obs dim
        obs_dims = self.env.obs_manager.calculate_obs_dim()
        self.obs_dim = obs_dims["actor"]
        self.action_dim = self.env.num_actions

        self.training_modules = {}

    def _init_algorithm(self) -> SimMPC:
        """Initialize SimMPC algorithm with policy, Q-network, planner."""
        alg_cfg = self.cfgs.algorithm

        self.alg = SimMPC(
            planning_env=self.planning_env,
            state_sync=self.state_sync,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            # Planning
            horizon=alg_cfg.horizon,
            num_samples=alg_cfg.num_samples,
            num_pi_trajs=alg_cfg.num_pi_trajs,
            num_elites=alg_cfg.num_elites,
            num_iterations=alg_cfg.num_iterations,
            temperature=alg_cfg.temperature,
            min_std=alg_cfg.min_std,
            max_std=alg_cfg.max_std,
            gamma=alg_cfg.gamma,
            num_train_envs=self.env.num_envs,
            # Training
            lr=alg_cfg.lr,
            pi_lr=alg_cfg.pi_lr,
            tau=alg_cfg.tau,
            # Networks
            hidden_dims=alg_cfg.hidden_dims,
            num_q=alg_cfg.num_q,
            squash_policy=alg_cfg.squash_policy,
        )

        return self.alg

    def _init_storage(self) -> None:
        """Initialize replay buffer."""
        alg_cfg = self.cfgs.algorithm
        self.alg.init_storage(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            buffer_size=alg_cfg.buffer_size,
        )

    # ==================== Data Collection ====================

    def _get_initial_obs(self) -> Dict[str, torch.Tensor]:
        """Get initial observation."""
        return self.env.obs_manager.get_observation()

    def _collect_experience(
        self,
        obs: Dict[str, torch.Tensor],
        iteration: int,
    ) -> Dict[str, Any]:
        """Collect experience using MPPI planning and store transitions."""
        start_time = time.time()
        alg_cfg = self.cfgs.algorithm

        for step in range(self.num_steps_per_env):
            actor_obs = obs["actor"]  # [N, obs_dim]

            # Warmup: random actions
            if self.total_timesteps < alg_cfg.learning_starts:
                actions = torch.rand(
                    self.env.num_envs, self.action_dim, device=self.device
                )
                action_low = self.env.action_low
                action_high = self.env.action_high
                actions = actions * (action_high - action_low) + action_low
            else:
                # MPPI + policy mix
                t0_mask = self.env.reset_buf
                actions = self.alg.act(
                    self.env, t0_mask,
                    eval_mode=False,
                    mppi_ratio=alg_cfg.mppi_ratio,
                    obs=actor_obs,
                )

            # Step training environment
            obs_dict, rewards, terminated, truncated, infos = self.env.step(actions)
            dones = terminated | truncated

            # Store transition
            next_actor_obs = obs_dict["actor"]
            self.alg.store_transition(
                obs=actor_obs,
                action=actions,
                reward=rewards,
                next_obs=next_actor_obs,
                done=dones.float(),
            )

            # Update reward statistics
            self._update_reward_stats(
                reward_info=infos["rewards_per_type"],
                dones=dones,
                success=infos.get("success", None),
            )

            obs = obs_dict

        return {
            "collection_time": time.time() - start_time,
            "last_obs": obs,
        }

    # ==================== Training ====================

    def _run_training_iteration(
        self,
        obs: Dict[str, torch.Tensor],
        iteration: int,
        ep_infos: List[Dict] = None,
    ) -> IterationData:
        """Collect experience + train policy and Q-networks."""
        collection_data = self._collect_experience(obs=obs, iteration=iteration)

        collection_time = collection_data["collection_time"]
        learning_time = 0.0
        fps = 0.0
        buffer_size = None
        alg_cfg = self.cfgs.algorithm

        min_buffer_size = max(alg_cfg.learning_starts, alg_cfg.batch_size)

        if (self.alg.replay_buffer is not None
                and self.alg.replay_buffer.size >= min_buffer_size):
            training_start = time.time()

            if not getattr(self, '_pretrained', False):
                num_updates = 1
                print(f'Pretraining policy on seed data ({num_updates} updates)...')
                self._pretrained = True
            else:
                num_updates = max(1, alg_cfg.num_gradient_steps)

            if not self.alg.planner.use_terminal_q:
                self.alg.planner.use_terminal_q = True
                print("Enabled terminal Q-value for MPPI planning.")

            for i in range(num_updates):
                self.alg.update(batch_size=alg_cfg.batch_size)

            learning_time = time.time() - training_start
            fps = (self.num_steps_per_env * self.env.num_envs) / max(
                collection_time + learning_time, 1e-6
            )
            buffer_size = self.alg.replay_buffer.size
        else:
            fps = (self.num_steps_per_env * self.env.num_envs) / max(
                collection_time, 1e-6
            )

        return IterationData(
            collection_time=collection_time,
            learning_time=learning_time,
            fps=fps,
            episode_stats=self._build_episode_stats(),
            metrics=None,  # SimMPC doesn't use BaseMetrics
            last_obs=collection_data["last_obs"],
            action_distribution={},
            buffer_size=buffer_size,
            iteration=iteration,
        )

    def learn(
        self,
        num_learning_iterations: int,
        init_at_random_ep_len: bool = False,
    ):
        """Main loop: MPPI-planned actions + policy training."""
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

            data = self._run_training_iteration(
                obs=obs, iteration=it, ep_infos=ep_infos,
            )

            obs = data.last_obs
            self.post_iteration(data, total_iter, it)

    def _get_action_statistics(self) -> Dict[str, Any]:
        """No action statistics for now."""
        return {}

    # ==================== Evaluation ====================

    def _run_evaluation(self) -> Dict[str, Any]:
        """Evaluate using learned policy only (no MPPI planning)."""
        eval_env = self._get_or_create_eval_env()
        eval_start = time.time()

        num_envs = eval_env.num_envs
        target_episodes = self.runner_cfg.eval_num_episodes
        max_steps = int(eval_env.max_episode_length) * 2

        eval_env.reset()
        obs_dict = eval_env.obs_manager.get_observation()

        episode_returns = torch.zeros(num_envs, device=self.device)
        episode_lengths = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        completed_returns: list[float] = []
        completed_lengths: list[float] = []
        reward_type_sums: dict[str, torch.Tensor] = {}
        completed_reward_breakdowns: dict[str, list[float]] = {}

        step = 0
        while len(completed_returns) < target_episodes and step < max_steps:
            actor_obs = obs_dict["actor"]
            with torch.no_grad():
                actions, _ = self.alg.policy(actor_obs, deterministic=True)
                actions = actions.clamp(self.env.action_low, self.env.action_high)

            obs_dict, rewards, terminated, truncated, infos = eval_env.step(actions)
            dones = terminated | truncated

            episode_returns += rewards
            episode_lengths += 1

            rewards_per_type = infos.get("rewards_per_type", {})
            for rname, rval in rewards_per_type.items():
                if rname not in reward_type_sums:
                    reward_type_sums[rname] = torch.zeros(num_envs, device=self.device)
                    completed_reward_breakdowns[rname] = []
                reward_type_sums[rname] += rval

            for i in range(num_envs):
                if dones[i] and len(completed_returns) < target_episodes:
                    completed_returns.append(episode_returns[i].item())
                    completed_lengths.append(episode_lengths[i].item())
                    for rname in reward_type_sums:
                        completed_reward_breakdowns[rname].append(
                            reward_type_sums[rname][i].item()
                        )

            episode_returns[dones] = 0
            episode_lengths[dones] = 0
            for rname in reward_type_sums:
                reward_type_sums[rname][dones] = 0

            step += 1

        eval_time = time.time() - eval_start

        eval_stats = {
            "eval/mean_return": np.mean(completed_returns) if completed_returns else 0.0,
            "eval/std_return": np.std(completed_returns) if completed_returns else 0.0,
            "eval/min_return": np.min(completed_returns) if completed_returns else 0.0,
            "eval/max_return": np.max(completed_returns) if completed_returns else 0.0,
            "eval/mean_episode_length": np.mean(completed_lengths) if completed_lengths else 0.0,
            "eval/num_episodes": len(completed_returns),
            "eval/time": eval_time,
        }

        for rname, vals in completed_reward_breakdowns.items():
            if vals:
                per_step = [v / l for v, l in zip(vals, completed_lengths)]
                eval_stats[f"eval/reward/{rname}"] = np.mean(per_step)

        return eval_stats

    # ==================== Checkpoint ====================

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_path: str,
        cfgs: ConfigsForRun = None,
        env: World = None,
        use_wandb: bool = True,
    ) -> "SimMPCRunner":
        """Load runner from checkpoint."""
        from rlworld.rl.utils.checkpoint import load_checkpoint_metadata
        metadata = load_checkpoint_metadata(checkpoint_path)

        if cfgs is None:
            from rlworld.rl.utils.checkpoint import load_config_from_checkpoint
            cfgs = load_config_from_checkpoint(metadata)
        if env is None:
            env = cls._create_env_from_config(cfgs)

        runner = cls(env=env, cfgs=cfgs, use_wandb=use_wandb)
        runner.alg.load_train_state(checkpoint_path, metadata)

        runner.current_learning_iteration = metadata.get(
            "current_learning_iteration", metadata["iteration"]
        )
        runner.total_timesteps = metadata["total_timesteps"]
        runner.total_time = metadata.get("total_time", 0)

        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"  Algorithm: {runner.algorithm_name}")
        print(f"  Iteration: {runner.current_learning_iteration}")
        print(f"  Timesteps: {runner.total_timesteps}")

        return runner
