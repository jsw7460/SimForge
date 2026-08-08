"""Train the K1 joystick task on newton with the G1-flat recipe + mirror symmetry."""

from rlworld.rl.configs.presets.k1_joystick.g1_recipe_mirror import K1G1RecipeMirrorConfig
from rlworld.rl.runners import BaseRunner


def main() -> None:
    cfgs = K1G1RecipeMirrorConfig(sim_type="newton").build().with_cli_overrides()
    runner = BaseRunner.create_with_env(cfgs)
    runner.learn(num_learning_iterations=cfgs.runner.max_iterations)


if __name__ == "__main__":
    main()
