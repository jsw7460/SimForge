"""Train the K1 joystick task on mujoco."""

from jaxrlworld.rl.configs.presets.k1_joystick.base import K1JoystickConfig
from jaxrlworld.rl.runners import BaseRunner


def main() -> None:
    cfgs = K1JoystickConfig(sim_type="mujoco").build().with_cli_overrides()
    runner = BaseRunner.create_with_env(cfgs)
    runner.learn(num_learning_iterations=cfgs.runner.max_iterations)


if __name__ == "__main__":
    main()
