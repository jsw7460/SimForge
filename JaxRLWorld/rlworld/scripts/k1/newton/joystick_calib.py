"""Train the K1 joystick task on Newton in the calibrated-plant world."""

from rlworld.rl.configs.presets.k1_joystick.calib import K1CalibConfig
from rlworld.rl.runners import BaseRunner


def main() -> None:
    cfgs = K1CalibConfig(sim_type="newton").build().with_cli_overrides()
    runner = BaseRunner.create_with_env(cfgs)
    runner.learn(num_learning_iterations=cfgs.runner.max_iterations)


if __name__ == "__main__":
    main()
