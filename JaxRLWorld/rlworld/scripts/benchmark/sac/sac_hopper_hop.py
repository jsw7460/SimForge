import os

os.environ['__NV_PRIME_RENDER_OFFLOAD'] = '1'
os.environ['__GLX_VENDOR_LIBRARY_NAME'] = 'nvidia'

custom_assets = os.path.abspath(os.path.join(os.path.dirname(__file__), 'assets'))
import genesis.utils.terrain

genesis.utils.misc.get_assets_dir = lambda: custom_assets
genesis.utils.terrain.get_assets_dir = lambda: custom_assets

from rlworld.rl.configs import GenesisConfigsForRun, SACPolicyConfig
from rlworld.rl.runners import OffPolicyRunner
from rlworld.rl.configs.presets.go2_flat.genesis.mlp import get_config
from rlworld.rl.configs.algorithms import SACConfig

import gymnasium as gym


def main():
    # Get complete config from preset
    configs_dict = get_config()

    # Create configs and runner
    cfgs_for_run = GenesisConfigsForRun.from_dict_with_overrides(configs_dict)

    cfgs_for_run.algorithm = SACConfig()
    cfgs_for_run.algorithm.obs_normalization = False
    cfgs_for_run.algorithm.actor_lr = 1e-4
    cfgs_for_run.algorithm.buffer_size = 1_000_000
    cfgs_for_run.algorithm.batch_size = 256
    cfgs_for_run.algorithm.tau = 0.005
    cfgs_for_run.algorithm.num_steps_per_env = 1
    cfgs_for_run.nn.policy = cfgs_for_run.nn.policy.to(SACPolicyConfig)
    cfgs_for_run.nn.policy.actor_kwargs.update({
        "hidden_dims": [64, 64, 32],
        "activation": "relu"
    })
    cfgs_for_run.nn.policy.critic_kwargs.update({
        "hidden_dims": [128, 128, 128],
        "activation": "relu"
    })
    cfgs_for_run.runner.log_interval = 100
    cfgs_for_run.runner.max_iterations = 1000000
    cfgs_for_run.runner.save_interval = 100000
    cfgs_for_run.runner.run_name = "SACBenchmarkHopper_Hop"
    cfgs_for_run.runner.eval_interval = 0       # Do not change this

    from rlworld.rl.envs import GymnasiumEnv
    from gymnasium.vector import SyncVectorEnv, AutoresetMode
    def make_env(seed):
        def _init():
            return gym.make("Hopper-v5")

        return _init

    num_envs = 1
    env_gym = SyncVectorEnv([make_env(i) for i in range(num_envs)], autoreset_mode=AutoresetMode.SAME_STEP)
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
