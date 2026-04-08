from rlworld.rl.runners import BaseRunner
from rlworld.rl.configs.presets.g1_29dof.mlp import get_config


def main():
    cfgs_for_run = get_config(sim="mujoco").with_cli_overrides()
    runner = BaseRunner.create_with_env(cfgs_for_run)

    # Start training
    runner.learn(
        num_learning_iterations=cfgs_for_run.runner.max_iterations,
        init_at_random_ep_len=cfgs_for_run.runner.init_at_random_ep_len,
    )


if __name__ == "__main__":
    main()
