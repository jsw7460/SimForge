"""Train the K1 getup task (fall recovery on the velocity policy contract) on newton."""

from jaxrlworld.rl.configs.presets.k1_getup.base import K1GetupConfig
from jaxrlworld.rl.runners import BaseRunner


def main() -> None:
    cfgs = K1GetupConfig(sim_type="newton").build().with_cli_overrides()
    runner = BaseRunner.create_with_env(cfgs)
    runner.learn(num_learning_iterations=cfgs.runner.max_iterations)


if __name__ == "__main__":
    main()
