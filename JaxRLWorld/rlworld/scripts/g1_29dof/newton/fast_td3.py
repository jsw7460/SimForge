from rlworld.rl.configs import NewtonConfigsForRun
from rlworld.rl.configs.presets.g1_29dof.newton.mlp import get_config
from rlworld.rl.configs.algorithms import FastTD3Config
from rlworld.rl.runners import BaseRunner


def main():
    # Get complete config from preset
    configs_dict = get_config()

    # Create configs and runner
    cfgs_for_run = NewtonConfigsForRun.from_dict_with_overrides(configs_dict)
    cfgs_for_run.env.num_envs = 1024
    fasttd3_config = FastTD3Config(
        batch_size=32768,
        buffer_size=1024 * 1024 * 10,
        learning_starts=10,
        obs_normalization=True,
        is_squashed=True,
        use_cdq=True,
        gamma=0.97,
        utd_ratio=4,
        v_min=-10.0,
        v_max=10.0,
        num_atoms=101,
        target_policy_noise=0.001,
        noise_min=0.001,
        noise_max=0.4,
        n_steps=1,
        tau=0.1,
    )
    cfgs_for_run.algorithm = fasttd3_config
    cfgs_for_run.nn.policy["actor_kwargs"].update(
        {
            "activation": "relu",
            "output_gain": 0.01,
            "hidden_dims": [512, 256, 128]
        },
    )

    cfgs_for_run.nn.policy["critic_kwargs"].update(
        {
            "activation": "relu",
            "output_gain": 0.01,
            "hidden_dims": [1024, 512, 256]
        }
    )

    cfgs_for_run.action.clip_actions = (-1.0, 1.0)
    cfgs_for_run.action.action_scale = 0.5


    runner = BaseRunner.create_with_env(cfgs_for_run)
    # Start training
    runner.learn(
        num_learning_iterations=cfgs_for_run.runner.max_iterations,
        init_at_random_ep_len=cfgs_for_run.runner.init_at_random_ep_len
    )


if __name__ == "__main__":
    main()
