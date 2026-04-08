from rlworld.rl.configs.presets.go2_flat.newton.mlp import get_config
from rlworld.rl.runners import BaseRunner


def main():
    cfgs_for_run = get_config().with_cli_overrides()
    runner = BaseRunner.create_with_env(cfgs_for_run)

    # Start training
    runner.learn(
        num_learning_iterations=cfgs_for_run.runner.max_iterations,
        init_at_random_ep_len=cfgs_for_run.runner.init_at_random_ep_len
    )


if __name__ == "__main__":
    main()
