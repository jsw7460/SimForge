"""Go2 Newton config: SAC velocity-tracking walking on flat ground.

Off-policy counterpart of the PPO ``Go2FlatConfig`` family. The scene,
rewards, observations, commands, DR and terminations are inherited
verbatim from :class:`Go2FlatConfig`; only three things change:

* ``use_joint_limit_action=True`` — the action manager maps the
  policy's tanh-bounded [-1, 1] output to the full soft joint-limit
  range per joint (``scale="joint_limit"`` + ``offset="joint_limit_center"``),
  replacing the fixed ``GO2_ACTION_SCALE`` mapping that assumes an
  unbounded Gaussian (PPO) policy.
* ``_build_algorithm_config`` returns a :class:`SACConfig` with the
  hyperparameters from "Bridging the Gap: Enabling Soft Actor Critic
  for High Performance Legged Locomotion" (RSL-RL-SAC): n-step
  returns, timeout-aware bootstrapping (both already native to the
  JaxRLWorld SAC stack), delayed policy updates and a physically
  calibrated entropy target.
* ``_build_nn_config`` returns the SAC actor/critic ([1024, 512, 256],
  state-dependent std, sigma_0 = 0.15, small-output init).

``target_entropy`` note: the reference work computes the policy
log-probability in physical action units, i.e. with a ``-sum(log c_j)``
affine-scaling correction (its Eq. 6). Here the policy distribution
lives in the normalized [-1, 1] space (the affine map is applied by
the action manager, outside the distribution), so that correction is
a constant and is folded into the entropy target instead::

    target_entropy = -0.167 * 12 - sum_j log(soft_half_j)
                   = -2.004 - 2.39721 = -4.40121

with ``soft_half_j = (upper_j - lower_j) / 2 * 0.9`` from the go2 MJCF
joint ranges (hip ±1.0472, front thigh [-1.5708, 3.4907], back thigh
[-0.5236, 4.5379], knee [-2.7227, -0.83776]). Recompute when the MJCF
joint ranges or ``joint_limit_soft_factor`` change.
"""

from dataclasses import dataclass

from rlworld.rl.configs import NewtonConfigsForRun
from rlworld.rl.configs.algorithms.sac import SACConfig
from rlworld.rl.configs.common_config_classes import (
    Activation,
    MLPActorCfg,
    MLPCriticCfg,
    NNConfig,
    SACPolicyConfig,
)
from rlworld.rl.configs.presets.go2.base import Go2FlatConfig


@dataclass
class Go2SACNewtonConfig(Go2FlatConfig):
    sim_type: str = "newton"
    run_name: str = "Go2_SAC_Newton"
    algorithm_name: str = "SAC"
    use_joint_limit_action: bool = True

    def _build_algorithm_config(self) -> SACConfig:
        return SACConfig(
            actor_lr=2e-4,
            critic_lr=2e-4,
            alpha_lr=2e-5,
            gamma=0.97,
            tau=0.003,
            n_steps=5,
            ent_coef="auto_0.001",
            # -0.167 * num_actions - sum(log soft_half_j); see module
            # docstring for the derivation and the per-joint values.
            target_entropy=-4.40121,
            obs_normalization=True,
            max_grad_norm=1.0,
            batch_size=8192,
            buffer_size=5_000_000,
            # Transitions before updates start; one 24-step collection
            # pass at 8192 envs stores 196,608 transitions, so updates
            # begin from the second iteration.
            learning_starts=200_000,
            policy_delay=1,
            num_gradient_steps=200,
            num_steps_per_env=24,
        )

    def _build_nn_config(self) -> NNConfig:
        return NNConfig(
            policy=SACPolicyConfig(
                actor=MLPActorCfg(
                    activation=Activation.SILU,
                    hidden_dims=[1024, 512, 256],
                ),
                critic=MLPCriticCfg(
                    activation=Activation.SILU,
                    hidden_dims=[1024, 512, 256],
                ),
                init_noise_std=0.15,
                small_output_init=True,
            ),
        )


def get_config() -> NewtonConfigsForRun:
    return Go2SACNewtonConfig().build()
