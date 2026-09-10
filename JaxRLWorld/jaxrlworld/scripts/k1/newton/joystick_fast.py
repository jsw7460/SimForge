"""Train the K1 fast-locomotion task (velocity curriculum to 2 m/s) on newton."""

from jaxrlworld.rl.configs.presets.k1_joystick.fast import K1FastConfig
from jaxrlworld.rl.runners import BaseRunner


def main() -> None:
    cfgs = K1FastConfig(sim_type="newton").build().with_cli_overrides()
    runner = BaseRunner.create_with_env(cfgs)
    runner.learn(num_learning_iterations=cfgs.runner.max_iterations)


if __name__ == "__main__":
    main()
