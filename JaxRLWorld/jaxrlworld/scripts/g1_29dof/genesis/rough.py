from jaxrlworld.rl.configs.presets.g1_29dof.genesis.rough import G1RoughGenesisConfig
from jaxrlworld.rl.runners import BaseRunner


def main():
    config = G1RoughGenesisConfig().build()
    cfgs_for_run = config.with_cli_overrides()
    runner = BaseRunner.create_with_env(cfgs_for_run)

    runner.learn(
        num_learning_iterations=cfgs_for_run.runner.max_iterations,
        init_at_random_ep_len=cfgs_for_run.runner.init_at_random_ep_len,
    )


if __name__ == "__main__":
    main()
