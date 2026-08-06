"""Train the K1 joystick task on mujoco in the dynamic-loss actuator world."""

from rlworld.rl.configs.presets.k1_joystick.dyn_loss import K1DynLossConfig
from rlworld.rl.runners import BaseRunner


def main() -> None:
    cfgs = K1DynLossConfig(sim_type="mujoco").build().with_cli_overrides()
    runner = BaseRunner.create_with_env(cfgs)
    runner.learn(num_learning_iterations=cfgs.runner.max_iterations)


if __name__ == "__main__":
    main()
