"""ManiSkill state-based PPO preset (all tasks).

One config class encodes the common state-based PPO recipe used by ManiSkill's
official baseline (``ManiSkill/examples/baselines/ppo/examples.sh``); per-task
differences live in the ``STATE_PPO_TASKS`` table. ``get_config(task)`` merges a
table entry onto the defaults and returns a built ``GymnasiumConfigsForRun``.

``build()`` stamps ``preset_module`` / ``preset_class_name`` / ``preset_kwargs``
(non-default fields only), so eval reloads via the standard
``load_config_from_checkpoint`` (preset re-run) path and recovers the task
automatically -- no per-task eval entry point is needed.

Env-construction params (obs_mode / control_mode / sim_backend) live in
``cfgs.env.gym_make_kwargs`` so they are persisted in the checkpoint and both
training and eval build the identical ManiSkill env.
"""

from __future__ import annotations

import math
from dataclasses import MISSING, dataclass, fields
from typing import Any, Dict

from rlworld.rl.configs import GymnasiumConfigsForRun
from rlworld.rl.configs.algorithms.ppo import PPOConfig
from rlworld.rl.configs.common_config_classes import Activation, MLPActorCfg, MLPCriticCfg, OrthoInit
from rlworld.rl.configs.gymnasium_config_classes import GymnasiumEnvConfig

# Per-task overrides relative to the common state-based recipe below. Values are
# taken from ManiSkill's examples.sh; only fields that differ are listed.
STATE_PPO_TASKS: Dict[str, Dict[str, Any]] = {
    "PickCube-v1": dict(total_timesteps=10_000_000),
    "StackCube-v1": dict(total_timesteps=25_000_000),
    "PushT-v1": dict(total_timesteps=25_000_000, num_steps=100, gamma=0.99),
    "PickSingleYCB-v1": dict(total_timesteps=25_000_000),
    "PegInsertionSide-v1": dict(total_timesteps=250_000_000, num_steps=100),
    "TwoRobotPickCube-v1": dict(total_timesteps=20_000_000, num_steps=100),
    "TwoRobotStackCube-v1": dict(total_timesteps=40_000_000, num_steps=100),
    "TriFingerRotateCubeLevel0-v1": dict(num_envs=128, total_timesteps=50_000_000, num_steps=250),
    "TriFingerRotateCubeLevel1-v1": dict(num_envs=128, total_timesteps=50_000_000, num_steps=250),
    "TriFingerRotateCubeLevel2-v1": dict(num_envs=128, total_timesteps=50_000_000, num_steps=250),
    "TriFingerRotateCubeLevel3-v1": dict(num_envs=128, total_timesteps=50_000_000, num_steps=250),
    "TriFingerRotateCubeLevel4-v1": dict(num_envs=1024, total_timesteps=500_000_000, num_steps=250),
    "PokeCube-v1": dict(total_timesteps=5_000_000, num_steps=20, eval_freq=10),
    "MS-CartpoleBalance-v1": dict(total_timesteps=4_000_000, num_steps=250, gamma=0.99, gae_lambda=0.95, eval_freq=5),
}


@dataclass
class ManiSkillStatePPOConfig:
    # Defaults = the common state-based recipe (examples.sh shared args + the
    # PickCube-v1 line); per-task overrides come from STATE_PPO_TASKS.
    task: str = "PickCube-v1"
    num_envs: int = 1024
    obs_mode: str = "state"
    control_mode: str = "pd_joint_delta_pos"
    sim_backend: str = "physx_cuda"
    num_steps: int = 50
    gamma: float = 0.8
    gae_lambda: float = 0.9
    update_epochs: int = 8
    num_minibatches: int = 32
    total_timesteps: int = 10_000_000
    eval_freq: int = 25  # in iterations
    seed: int = 0
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4

    def build(self) -> GymnasiumConfigsForRun:
        cfgs = GymnasiumConfigsForRun()
        cfgs.env = GymnasiumEnvConfig(
            env_name="ManiSkillEnv",
            task_name=self.task,
            num_envs=self.num_envs,
            seed=self.seed,
            gym_make_kwargs={
                "obs_mode": self.obs_mode,
                "control_mode": self.control_mode,
                "sim_backend": self.sim_backend,
            },
        )

        # Tanh MLP [256,256,256], orthogonal init, near-zero actor output layer,
        # initial log-std of -0.5 (std ~= 0.61).
        cfgs.nn.policy.actor = MLPActorCfg(
            hidden_dims=[256, 256, 256], activation=Activation.TANH, init=OrthoInit(output_gain=0.01)
        )
        cfgs.nn.policy.critic = MLPCriticCfg(
            hidden_dims=[256, 256, 256], activation=Activation.TANH, init=OrthoInit(output_gain=1.0)
        )
        cfgs.nn.policy.init_noise_std = math.exp(-0.5)

        cfgs.algorithm = PPOConfig(
            gamma=self.gamma,
            lam=self.gae_lambda,
            num_steps_per_env=self.num_steps,
            num_mini_batches=self.num_minibatches,
            num_learning_epochs=self.update_epochs,
            clip_param=0.2,
            entropy_coef=0.0,
            value_loss_coef=0.5,
            max_grad_norm=0.5,
            actor_lr=self.actor_lr,
            critic_lr=self.critic_lr,
            schedule="fixed",
            use_early_stop=True,
            desired_kl=0.1,
            use_clipped_value_loss=False,
            use_value_normalization=False,
            obs_normalization=False,
            use_sde=False,
        )

        # total_timesteps -> iterations (env-steps per iteration = num_envs * num_steps).
        max_iterations = max(1, self.total_timesteps // (self.num_envs * self.num_steps))
        cfgs.runner.run_name = f"ManiSkill_{self.task.replace('-', '_')}_PPO"
        cfgs.runner.max_iterations = max_iterations
        cfgs.runner.log_interval = 10
        cfgs.runner.save_interval = max(1, max_iterations // 10)
        cfgs.runner.eval_interval = self.eval_freq

        cfgs.preset_module = type(self).__module__
        cfgs.preset_class_name = type(self).__name__
        cfgs.preset_kwargs = self._get_preset_kwargs()
        return cfgs

    def _get_preset_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if f.default is not MISSING:
                default = f.default
            elif f.default_factory is not MISSING:  # type: ignore[misc]
                default = f.default_factory()  # type: ignore[misc]
            else:
                kwargs[f.name] = value
                continue
            if value != default:
                kwargs[f.name] = value
        return kwargs


def get_config(task: str = "PickCube-v1", **overrides) -> GymnasiumConfigsForRun:
    """Build the state-based PPO config for ``task`` (table overrides + kwargs)."""
    if task not in STATE_PPO_TASKS:
        raise KeyError(f"Unknown ManiSkill task {task!r}. Known: {sorted(STATE_PPO_TASKS)}")
    merged = {**STATE_PPO_TASKS[task], **overrides}
    return ManiSkillStatePPOConfig(task=task, **merged).build()
