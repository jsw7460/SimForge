from rlworld.rl.runners import BaseRunner
from rlworld.rl.configs.algorithms import FastTD3Config
from rlworld.rl.configs.presets.go2_flat.genesis.mlp import get_config


def main():
    cfgs_for_run = get_config().with_cli_overrides()

    cfgs_for_run.env.num_envs = 4096
    fasttd3_config = FastTD3Config(
        batch_size=4096 * 2,
        buffer_size=4096 * 1100,
        learning_starts=10000,
        is_squashed=True,
        use_cdq=False,
        num_gradient_steps=16,
    )
    cfgs_for_run.algorithm = fasttd3_config

    cfgs_for_run.action.clip_actions = "joint_limit"
    cfgs_for_run.action.action_scale = 1.0

    runner = BaseRunner.create_with_env(cfgs_for_run)

    # Start training
    runner.learn(
        num_learning_iterations=cfgs_for_run.runner.max_iterations,
        init_at_random_ep_len=cfgs_for_run.runner.init_at_random_ep_len
    )


if __name__ == "__main__":
    main()
