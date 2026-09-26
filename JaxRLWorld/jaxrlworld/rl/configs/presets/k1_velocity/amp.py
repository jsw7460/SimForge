"""The K1 velocity task with the adversarial motion prior (the source's "Flat-Amp").

Same MDP as :class:`~jaxrlworld.rl.configs.presets.k1_velocity.base.K1VelocityConfig`
with the prior switched on: ``AMP_PPO``, the ``amp`` observation group the
discriminator scores, pose-pool resets from the LAFAN1 clips, and the
source's standing-pose weight.
"""

from __future__ import annotations

from dataclasses import dataclass

from jaxrlworld.rl.configs.presets.k1_velocity.base import K1VelocityConfig


@dataclass
class K1AmpConfig(K1VelocityConfig):
    use_amp: bool = True
    algorithm_name: str = "AMP_PPO"
