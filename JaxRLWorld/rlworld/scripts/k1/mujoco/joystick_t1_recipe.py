"""Train the K1 joystick task on mujoco with the T1-flat recipe (robot_lab)."""

from rlworld.rl.configs.presets.k1_joystick.t1_recipe import K1T1RecipeConfig
from rlworld.rl.runners import BaseRunner


def main() -> None:
    cfgs = K1T1RecipeConfig(sim_type="mujoco").build().with_cli_overrides()
    runner = BaseRunner.create_with_env(cfgs)
    runner.learn(num_learning_iterations=cfgs.runner.max_iterations)


if __name__ == "__main__":
    main()
