"""ManiSkill PickCube-v1 PPO preset.

Single-target preset for ManiSkill via the Gymnasium config family. Mirrors the
preset convention of the physics-sim presets (go2/g1/crazyflie): ``build()``
returns a fully populated ``GymnasiumConfigsForRun`` and stamps
``preset_module`` / ``preset_class_name`` / ``preset_kwargs`` so eval reload goes
through the *standard* ``load_config_from_checkpoint`` (preset re-run) path --
the same logic every other sim uses, with no ManiSkill-specific deserialization.

Hyperparameters are aligned 1:1 with ManiSkill's official state-based PPO
baseline (``ManiSkill/examples/baselines/ppo/ppo.py``, PickCube-v1 defaults).

The env-construction params (obs_mode / control_mode / sim_backend) live in
``cfgs.env.gym_make_kwargs`` so they are persisted in the checkpoint and both
training and eval build the ManiSkill env from the same single source.
"""

from __future__ import annotations

import math
from dataclasses import MISSING, dataclass, fields
from typing import Any, Dict

from rlworld.rl.configs import GymnasiumConfigsForRun
from rlworld.rl.configs.algorithms.ppo import PPOConfig
from rlworld.rl.configs.common_config_classes import Activation, MLPActorCfg, MLPCriticCfg, OrthoInit
from rlworld.rl.configs.gymnasium_config_classes import GymnasiumEnvConfig


@dataclass
class ManiSkillPickCubeConfig:
    task: str = "PickCube-v1"
    num_envs: int = 1024
    obs_mode: str = "state"
    control_mode: str = "pd_joint_delta_pos"
    sim_backend: str = "physx_cuda"
    seed: int = 0
    max_iterations: int = 400  # ~10M timesteps at 512 envs x 50 steps
    eval_interval: int = 25
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4

    def build(self) -> GymnasiumConfigsForRun:
        cfgs = GymnasiumConfigsForRun()
        cfgs.env = GymnasiumEnvConfig(
            env_name="ManiSkillEnv",
            task_name=self.task,
            num_envs=self.num_envs,
            seed=self.seed,
            # Single source of truth for env construction; persisted in the
            # checkpoint so eval rebuilds the identical ManiSkill env.
            gym_make_kwargs={
                "obs_mode": self.obs_mode,
                "control_mode": self.control_mode,
                "sim_backend": self.sim_backend,
            },
        )

        # Network: Tanh MLP [256,256,256], orthogonal init, near-zero actor
        # output layer (ManiSkill uses std=0.01 on the final actor layer).
        cfgs.nn.policy.actor = MLPActorCfg(
            hidden_dims=[256, 256, 256],
            activation=Activation.TANH,
            init=OrthoInit(output_gain=0.01),
        )
        cfgs.nn.policy.critic = MLPCriticCfg(
            hidden_dims=[256, 256, 256],
            activation=Activation.TANH,
            init=OrthoInit(output_gain=1.0),
        )
        cfgs.nn.policy.init_noise_std = math.exp(-0.5)  # ManiSkill actor_logstd init = -0.5

        cfgs.algorithm = PPOConfig(
            gamma=0.8,
            lam=0.9,
            num_steps_per_env=50,
            num_mini_batches=32,
            num_learning_epochs=8,
            clip_param=0.2,
            entropy_coef=0.0,
            value_loss_coef=0.5,
            max_grad_norm=0.5,
            actor_lr=self.actor_lr,
            critic_lr=self.critic_lr,
            schedule="fixed",  # ManiSkill anneal_lr=False (constant LR)
            use_early_stop=True,  # epoch break on KL, like ManiSkill target_kl
            desired_kl=0.1,  # target_kl
            use_clipped_value_loss=False,
            use_value_normalization=False,
            obs_normalization=False,
            use_sde=False,
        )

        cfgs.runner.run_name = f"ManiSkill_{self.task.replace('-', '_')}_PPO"
        cfgs.runner.max_iterations = self.max_iterations
        cfgs.runner.log_interval = 10
        cfgs.runner.save_interval = 100
        cfgs.runner.eval_interval = self.eval_interval

        # Preset metadata (same convention as the physics-sim presets) so eval
        # reload re-runs this exact preset via load_config_from_checkpoint.
        cfgs.preset_module = type(self).__module__
        cfgs.preset_class_name = type(self).__name__
        cfgs.preset_kwargs = self._get_preset_kwargs()
        return cfgs

    def _get_preset_kwargs(self) -> Dict[str, Any]:
        """Constructor kwargs needed to reconstruct this config at eval time.

        Only fields whose value differs from the dataclass default are
        included so the dict stays small and forward-compatible.
        """
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


def get_config(**kwargs) -> GymnasiumConfigsForRun:
    return ManiSkillPickCubeConfig(**kwargs).build()
