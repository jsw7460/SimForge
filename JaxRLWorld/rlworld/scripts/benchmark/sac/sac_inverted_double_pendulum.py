import os

os.environ['__NV_PRIME_RENDER_OFFLOAD'] = '1'
os.environ['__GLX_VENDOR_LIBRARY_NAME'] = 'nvidia'

custom_assets = os.path.abspath(os.path.join(os.path.dirname(__file__), 'assets'))
import genesis.utils.terrain

genesis.utils.misc.get_assets_dir = lambda: custom_assets
genesis.utils.terrain.get_assets_dir = lambda: custom_assets

from rlworld.rl.configs.algorithms.sac import SACConfig
from rlworld.rl.configs import GenesisConfigsForRun
from rlworld.rl.runners import OffPolicyRunner
from rlworld.rl.configs.presets.go2_flat.genesis.mlp import get_config

import gymnasium as gym


def main():
    # Get complete config from preset
    configs_dict = get_config()
    configs_dict["runner"]["algorithm_class_name"] = "SAC"
    configs_dict["runner"]["policy_class_name"] = "SACActorCritic"

    # Create configs and runner
    cfgs_for_run = GenesisConfigsForRun.from_dict_with_overrides(configs_dict)

    sac_config = SACConfig(
        actor_lr=3e-4,
        critic_lr=3e-4,
        tau=0.005,
        batch_size=256,
        buffer_size=1_000_000,
        learning_starts=100
    )

    cfgs_for_run.algorithm = sac_config
    cfgs_for_run.runner.num_steps_per_env = 1
    cfgs_for_run.runner.log_interval = 100
    cfgs_for_run.runner.max_iterations = 1000000
    cfgs_for_run.runner.save_interval = 100000
    cfgs_for_run.runner.run_name = "SACBenchmarkInvertedDoublePendulum"

    cfgs_for_run.nn.policy["std_type"] = "state_dependent"
    # cfgs_for_run.nn.policy["distribution_type"] = "squashed_gaussian"
    cfgs_for_run.nn.policy["distribution_type"] = "gaussian"
    cfgs_for_run.nn.policy["actor_kwargs"].update({
        "hidden_dims": [128, 128],
        "activation": "tanh"
    })
    cfgs_for_run.nn.policy["critic_kwargs"].update({
        "hidden_dims": [256, 128],
        "activation": "tanh"
    })

    from rlworld.rl.envs import GymnasiumEnv
    from gymnasium.vector import SyncVectorEnv
    def make_env(seed):
        def _init():
            return gym.make("InvertedDoublePendulum-v4")

        return _init

    num_envs = 1
    env_gym = SyncVectorEnv([make_env(i) for i in range(num_envs)])
    env = GymnasiumEnv(
        env_gym,
        env_cfg=cfgs_for_run.env,
        scene_cfg=cfgs_for_run.scene,
        obs_cfg=cfgs_for_run.observation,
        act_cfg=cfgs_for_run.action,
        reward_cfg=cfgs_for_run.reward,
        command_cfg=cfgs_for_run.command,
        seed=42
    )

    runner = OffPolicyRunner(env=env, cfgs=cfgs_for_run, use_wandb=True)

    # Start training
    runner.learn(
        num_learning_iterations=cfgs_for_run.runner.max_iterations,
        init_at_random_ep_len=cfgs_for_run.runner.init_at_random_ep_len
    )


if __name__ == "__main__":
    main()
