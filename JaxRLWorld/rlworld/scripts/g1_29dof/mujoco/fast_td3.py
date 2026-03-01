from rlworld.rl.configs import MujocoConfigsForRun
from rlworld.rl.runners import BaseRunner
from rlworld.rl.configs.presets.g1_29dof.mujoco.mlp import get_config
from rlworld.rl.configs.algorithms import FastTD3Config


def main():
    # Get complete config from preset
    configs_dict = get_config()

    # Create configs and runner
    cfgs_for_run = MujocoConfigsForRun.from_dict_with_overrides(configs_dict)
    cfgs_for_run.env.num_envs = 1024
    fasttd3_config = FastTD3Config(
        batch_size=32768,
        buffer_size=1024 * 1000 * 50,
        learning_starts=10,
        is_squashed=True,
        use_cdq=True,
        utd_ratio=4,
        v_min=-10.0,
        v_max=10.0,
        num_atoms=101,
        noise_min=0.001,
        noise_max=0.4,
        n_steps=8,
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
    cfgs_for_run.algorithm.obs_normalization = True

    runner = BaseRunner.create_with_env(cfgs_for_run)

    # Start training
    runner.learn(
        num_learning_iterations=cfgs_for_run.runner.max_iterations,
        init_at_random_ep_len=cfgs_for_run.runner.init_at_random_ep_len,
    )


if __name__ == "__main__":
    main()
