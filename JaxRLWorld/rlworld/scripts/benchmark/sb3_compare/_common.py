"""Stable-Baselines3 vs JaxRLWorld on identical Gymnasium tasks.

One launcher file per (framework, algorithm, task) so each cell of the
comparison runs on its own; every launcher funnels into :func:`run_sb3`
or :func:`run_jrw` here, so both frameworks read the SAME hyperparameter
table and the same fairness contract:

- Same environment: the plain Gymnasium task (``HalfCheetah-v5`` /
  ``Swimmer-v5``), same number of parallel envs per algorithm
  (PPO: 16, SAC/TD3: 1 — the SB3 convention).
- Same budget: a fixed number of environment steps per algorithm
  (constants below, sized for roughly 10-15 minutes). Equal samples on
  both sides; the wall-clock difference is visible on wandb's relative
  time axis, where the faster framework's curve simply ends earlier.
- Same metric: both sides log ``Train/mean_return`` to the wandb
  project ``SB3_vs_JRW``. Compare with the x-axis set to *relative
  time*; the env-step count is logged alongside as ``Train/env_steps``
  (SB3) / the native step axis (JRW).
- Same hyperparameters where the concepts map 1:1 (tables below,
  values = SB3 defaults so SB3 runs untuned-but-canonical).

Known, accepted asymmetries (kept because removing them would compare
something other than the frameworks as they are actually used):

- JRW's learner is JAX (run launchers with ``jaxpy``), SB3's is torch
  (run with plain ``python``); the physics is CPU Gymnasium in both.
- SB3 has no value-loss clipping by default; JRW's flag is turned off
  here to match, and JRW's adaptive-KL LR schedule is fixed to SB3's
  constant LR. PPO init and Adam epsilon are pinned to SB3's values
  (hidden ortho gain sqrt(2), eps 1e-5).

Usage (each launcher takes no arguments):
    python  rlworld/scripts/benchmark/sb3_compare/sb3_ppo_halfcheetah.py
    jaxpy   rlworld/scripts/benchmark/sb3_compare/jrw_ppo_halfcheetah.py
"""

from __future__ import annotations

import math
import os
import time

PROJECT = "SB3_vs_JRW"

# Fixed budgets, sized for roughly 10-15 minutes; edit here to change.
PPO_ITERS = 1000  # x 16 envs x 128 steps = ~2.0M env steps
OFFPOLICY_ITERS = 100_000  # 1 env step + 1 gradient step each

TASKS = {
    "halfcheetah": "HalfCheetah-v5",
    "swimmer": "Swimmer-v5",
}

# ── shared hyperparameter tables (values = SB3 defaults) ─────────────

PPO_HP = {
    "n_envs": 16,
    "n_steps": 128,  # per env -> rollout 2048
    "minibatch_size": 64,  # -> 32 minibatches
    "n_epochs": 10,
    "lr": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "net": [64, 64],  # MlpPolicy default, tanh
}

SAC_HP = {
    "n_envs": 1,
    "lr": 3e-4,
    "buffer_size": 1_000_000,
    "batch_size": 256,
    "tau": 0.005,
    "gamma": 0.99,
    "learning_starts": 100,
    "gradient_steps": 1,
    "net": [256, 256],  # relu
}

TD3_HP = {
    "n_envs": 1,
    "lr": 1e-3,
    "buffer_size": 1_000_000,
    "batch_size": 256,
    "tau": 0.005,
    "gamma": 0.99,
    "learning_starts": 100,
    "policy_delay": 2,
    "target_policy_noise": 0.2,
    "target_noise_clip": 0.5,
    "exploration_noise": 0.1,
    "net": [256, 256],  # kept identical on both sides (SB3's own default is [400, 300])
}


def _seed() -> int:
    """Seed from BENCH_SEED (default 0) — the .bash drivers sweep it."""
    return int(os.environ.get("BENCH_SEED", "0"))


def _run_name(framework: str, algo: str, task_key: str, seed: int) -> str:
    return f"{framework}_{algo}_{task_key}_s{seed}"


# ── Stable-Baselines3 side ───────────────────────────────────────────


def run_sb3(algo: str, task_key: str) -> None:
    try:
        import stable_baselines3 as sb3
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.env_util import make_vec_env
    except ImportError as e:
        raise SystemExit("stable-baselines3 is not installed: pip install stable-baselines3") from e

    import gymnasium as gym
    import numpy as np
    import torch

    import wandb

    task = TASKS[task_key]
    seed = _seed()

    class BenchCallback(BaseCallback):
        """Log Train/mean_return every few seconds."""

        def __init__(self, log_every_s: float = 5.0):
            super().__init__()
            self._log_every_s = log_every_s
            self._next_log = 0.0

        def _on_step(self) -> bool:
            now = time.monotonic()
            if now >= self._next_log and len(self.model.ep_info_buffer) > 0:
                self._next_log = now + self._log_every_s
                returns = [info["r"] for info in self.model.ep_info_buffer]
                # step = env steps, matching JRW's wandb step axis
                # (its logger logs with step=total_timesteps).
                wandb.log({"Train/mean_return": float(np.mean(returns))}, step=int(self.num_timesteps))
            return True

    if algo == "ppo":
        hp = PPO_HP
        env = make_vec_env(task, n_envs=hp["n_envs"], seed=seed)
        model = sb3.PPO(
            "MlpPolicy",
            env,
            n_steps=hp["n_steps"],
            batch_size=hp["minibatch_size"],
            n_epochs=hp["n_epochs"],
            learning_rate=hp["lr"],
            gamma=hp["gamma"],
            gae_lambda=hp["gae_lambda"],
            clip_range=hp["clip_range"],
            ent_coef=hp["ent_coef"],
            vf_coef=hp["vf_coef"],
            max_grad_norm=hp["max_grad_norm"],
            policy_kwargs={"net_arch": list(hp["net"]), "activation_fn": torch.nn.Tanh},
            seed=seed,
            verbose=1,
        )
    elif algo == "sac":
        hp = SAC_HP
        env = gym.make(task)
        model = sb3.SAC(
            "MlpPolicy",
            env,
            learning_rate=hp["lr"],
            buffer_size=hp["buffer_size"],
            batch_size=hp["batch_size"],
            tau=hp["tau"],
            gamma=hp["gamma"],
            learning_starts=hp["learning_starts"],
            train_freq=1,
            gradient_steps=hp["gradient_steps"],
            policy_kwargs={"net_arch": list(hp["net"])},
            seed=seed,
            verbose=1,
        )
    elif algo == "td3":
        from stable_baselines3.common.noise import NormalActionNoise

        hp = TD3_HP
        env = gym.make(task)
        n_act = env.action_space.shape[0]
        model = sb3.TD3(
            "MlpPolicy",
            env,
            learning_rate=hp["lr"],
            buffer_size=hp["buffer_size"],
            batch_size=hp["batch_size"],
            tau=hp["tau"],
            gamma=hp["gamma"],
            learning_starts=hp["learning_starts"],
            policy_delay=hp["policy_delay"],
            target_policy_noise=hp["target_policy_noise"],
            target_noise_clip=hp["target_noise_clip"],
            action_noise=NormalActionNoise(np.zeros(n_act), hp["exploration_noise"] * np.ones(n_act)),
            policy_kwargs={"net_arch": list(hp["net"])},
            seed=seed,
            verbose=1,
        )
    else:
        raise ValueError(f"Unknown algo {algo!r}")

    total_steps = PPO_ITERS * PPO_HP["n_envs"] * PPO_HP["n_steps"] if algo == "ppo" else OFFPOLICY_ITERS
    wandb.init(project=PROJECT, name=_run_name("SB3", algo, task_key, seed), config={"task": task, "seed": seed, **hp})
    model.learn(total_timesteps=total_steps, callback=BenchCallback())
    wandb.finish()


# ── JaxRLWorld side ──────────────────────────────────────────────────


def _jrw_base_cfg(algo: str, task_key: str, num_envs: int):
    """The shared JRW config skeleton: gym task, wandb, no eval/ckpt spam."""
    # The genesis asset-dir monkeypatch mirrors every existing gym
    # benchmark script in this tree — get_config("genesis") imports
    # genesis, which insists on resolving its asset directory.
    custom_assets = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ppo", "assets"))
    import genesis.utils.terrain

    genesis.utils.misc.get_assets_dir = lambda: custom_assets
    genesis.utils.terrain.get_assets_dir = lambda: custom_assets

    from rlworld.rl.configs.presets.go2.mlp import get_config

    seed = _seed()
    cfgs = get_config(sim="genesis")
    cfgs.runner.run_name = _run_name("JRW", algo, task_key, seed)
    cfgs.runner.wandb_project = PROJECT
    cfgs.runner.save_interval = 10**9
    cfgs.runner.eval_interval = 0
    cfgs.env.num_envs = num_envs
    cfgs.env.env_name = "GymnasiumEnv"
    cfgs.env.task_name = TASKS[task_key]
    cfgs.env.seed = seed
    return cfgs


def run_jrw(algo: str, task_key: str) -> None:
    from rlworld.rl.configs.common_config_classes import (
        Activation,
        DefaultInit,
        DistributionType,
        MLPActorCfg,
        MLPCriticCfg,
        OrthoInit,
    )

    if algo == "ppo":
        from rlworld.rl.configs.algorithms.ppo import PPOConfig
        from rlworld.rl.runners import BaseRunner

        hp = PPO_HP
        cfgs = _jrw_base_cfg(algo, task_key, num_envs=hp["n_envs"])
        cfgs.runner.log_interval = 5
        # SB3 parity: orthogonal init with hidden gain sqrt(2) on every
        # hidden layer (SB3 uses sqrt(2) regardless of activation; our
        # default for tanh would be 1.0) and SB3's head gains.
        cfgs.nn.policy.actor = MLPActorCfg(
            hidden_dims=list(hp["net"]),
            activation=Activation.TANH,
            init=OrthoInit(output_gain=0.01, hidden_gain=math.sqrt(2.0)),
        )
        cfgs.nn.policy.critic = MLPCriticCfg(
            hidden_dims=list(hp["net"]),
            activation=Activation.TANH,
            init=OrthoInit(output_gain=1.0, hidden_gain=math.sqrt(2.0)),
        )
        cfgs.nn.policy.distribution_type = DistributionType.GAUSSIAN
        rollout = hp["n_envs"] * hp["n_steps"]
        cfgs.algorithm = PPOConfig(
            actor_lr=hp["lr"],
            critic_lr=hp["lr"],
            gamma=hp["gamma"],
            lam=hp["gae_lambda"],
            clip_param=hp["clip_range"],
            entropy_coef=hp["ent_coef"],
            value_loss_coef=hp["vf_coef"],
            max_grad_norm=hp["max_grad_norm"],
            num_steps_per_env=hp["n_steps"],
            num_learning_epochs=hp["n_epochs"],
            num_mini_batches=rollout // hp["minibatch_size"],
            # SB3 parity: constant LR (no adaptive-KL schedule), no value
            # clipping, no SDE, no obs normalization (SB3 runs without
            # VecNormalize by default).
            schedule="fixed",
            desired_kl=None,
            use_clipped_value_loss=False,
            use_sde=False,
            obs_normalization=False,
            optimizer_eps=1e-5,  # SB3's PPO Adam epsilon
        )
        cfgs.runner.max_iterations = PPO_ITERS
        runner = BaseRunner.create_with_env(cfgs, seed=cfgs.env.seed)
        runner.learn(num_learning_iterations=PPO_ITERS, init_at_random_ep_len=False)
        return

    # SAC / TD3: mirror the existing off-policy gym scripts — a manual
    # SyncVectorEnv wrapped in GymnasiumEnv, fed to OffPolicyRunner.
    import gymnasium as gym
    from gymnasium.vector import AutoresetMode, SyncVectorEnv

    from rlworld.rl.envs import GymnasiumEnv
    from rlworld.rl.runners import OffPolicyRunner

    if algo == "sac":
        from rlworld.rl.configs import SACPolicyConfig
        from rlworld.rl.configs.algorithms import SACConfig

        hp = SAC_HP
        cfgs = _jrw_base_cfg(algo, task_key, num_envs=hp["n_envs"])
        cfgs.algorithm = SACConfig(
            actor_lr=hp["lr"],
            critic_lr=hp["lr"],
            alpha_lr=hp["lr"],
            gamma=hp["gamma"],
            tau=hp["tau"],
            batch_size=hp["batch_size"],
            buffer_size=hp["buffer_size"],
            learning_starts=hp["learning_starts"],
            num_gradient_steps=hp["gradient_steps"],
        )
        cfgs.nn.policy = cfgs.nn.policy.to(SACPolicyConfig)
        policy_cfg = cfgs.nn.policy
    elif algo == "td3":
        from rlworld.rl.configs import TD3PolicyConfig
        from rlworld.rl.configs.algorithms import TD3Config

        hp = TD3_HP
        cfgs = _jrw_base_cfg(algo, task_key, num_envs=hp["n_envs"])
        cfgs.algorithm = TD3Config(
            actor_lr=hp["lr"],
            critic_lr=hp["lr"],
            gamma=hp["gamma"],
            tau=hp["tau"],
            batch_size=hp["batch_size"],
            buffer_size=hp["buffer_size"],
            learning_starts=hp["learning_starts"],
            policy_delay=hp["policy_delay"],
            target_policy_noise=hp["target_policy_noise"],
            target_noise_clip=hp["target_noise_clip"],
            exploration_noise=hp["exploration_noise"],
        )
        cfgs.nn.policy = cfgs.nn.policy.to(TD3PolicyConfig)
        policy_cfg = cfgs.nn.policy
    else:
        raise ValueError(f"Unknown algo {algo!r}")

    cfgs.runner.log_interval = 500
    # An off-policy iteration is one environment step, so the rolling
    # checkpoint's default cadence (every 10 iterations) would write the
    # full parameter set to disk 10,000 times over this run. SB3 writes
    # none while learning; keep the comparison about learning.
    cfgs.runner.latest_checkpoint_interval = OFFPOLICY_ITERS
    policy_cfg.actor = MLPActorCfg(hidden_dims=list(hp["net"]), activation=Activation.RELU, init=DefaultInit())
    policy_cfg.critic = MLPCriticCfg(hidden_dims=list(hp["net"]), activation=Activation.RELU, init=DefaultInit())

    def make_env(env_seed):
        def _init():
            e = gym.make(cfgs.env.task_name)
            e.action_space.seed(env_seed)
            e.observation_space.seed(env_seed)
            return e

        return _init

    env_gym = SyncVectorEnv(
        [make_env(cfgs.env.seed * 1000 + i) for i in range(cfgs.env.num_envs)],
        autoreset_mode=AutoresetMode.SAME_STEP,
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
    cfgs.runner.max_iterations = OFFPOLICY_ITERS
    runner = OffPolicyRunner(env=env, cfgs=cfgs, use_wandb=True, seed=cfgs.env.seed)
    runner.learn(num_learning_iterations=OFFPOLICY_ITERS, init_at_random_ep_len=False)
