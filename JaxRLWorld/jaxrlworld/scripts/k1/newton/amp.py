"""Train the K1 velocity task with the adversarial motion prior on newton.

``--style-weight`` sets ``w`` in ``(1 - w) r_task + w dt r_style`` (the
preset's 0.3 if omitted); dotted overrides still pass through, e.g.
``runner.max_iterations=100000 algorithm.amp.discriminator_lr=1e-4``.
"""

import argparse

from jaxrlworld.rl.configs.presets.k1_velocity.amp import K1AmpConfig
from jaxrlworld.rl.runners import BaseRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="K1 velocity + motion prior on newton")
    parser.add_argument("--style-weight", type=float, default=None)
    args, _overrides = parser.parse_known_args()

    preset = K1AmpConfig(sim_type="newton")
    if args.style_weight is not None:
        preset.amp_style_reward_weight = args.style_weight
    cfgs = preset.build().with_cli_overrides()
    runner = BaseRunner.create_with_env(cfgs)
    runner.learn(num_learning_iterations=cfgs.runner.max_iterations)


if __name__ == "__main__":
    main()
