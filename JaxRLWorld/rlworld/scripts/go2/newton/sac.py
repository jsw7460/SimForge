from rlworld.rl.configs.presets.go2.newton.sac import Go2SACNewtonConfig
from rlworld.rl.runners import BaseRunner


def main():
    config = Go2SACNewtonConfig().build()
    cfgs_for_run = config.with_cli_overrides()
    runner = BaseRunner.create_with_env(cfgs_for_run)

    # Start training
    runner.learn(
        num_learning_iterations=cfgs_for_run.runner.max_iterations,
        init_at_random_ep_len=cfgs_for_run.runner.init_at_random_ep_len,
    )


if __name__ == "__main__":
    main()
