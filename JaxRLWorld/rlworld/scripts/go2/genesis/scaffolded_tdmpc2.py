from rlworld.rl.configs import GenesisConfigsForRun
from rlworld.rl.runners import BaseRunner
from rlworld.rl.configs.presets.go2_flat.genesis.scaffolded_tdmpc2 import get_config

def main():
    # Get complete config from preset
    configs_dict = get_config()

    # Create configs and runner
    cfgs_for_run = GenesisConfigsForRun.from_dict_with_overrides(configs_dict)

    cfgs_for_run.env.num_envs = 1024
    cfgs_for_run.algorithm.buffer_size = 5000000
    cfgs_for_run.algorithm.learning_starts = 5000
    cfgs_for_run.algorithm.batch_size = 10000
    cfgs_for_run.algorithm.utd_ratio = 8
    cfgs_for_run.algorithm.warmup_std = 0.1
    cfgs_for_run.algorithm.entropy_coef = 1e-3
    cfgs_for_run.algorithm.lr = 1e-4
    cfgs_for_run.algorithm.pi_lr = 1e-4

    cfgs_for_run.runner.max_iterations = 100000
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
