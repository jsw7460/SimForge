import os

os.environ['__NV_PRIME_RENDER_OFFLOAD'] = '1'
os.environ['__GLX_VENDOR_LIBRARY_NAME'] = 'nvidia'

custom_assets = os.path.abspath(os.path.join(os.path.dirname(__file__), 'assets'))
import genesis.utils.terrain

genesis.utils.misc.get_assets_dir = lambda: custom_assets
genesis.utils.terrain.get_assets_dir = lambda: custom_assets

from rlworld.rl.configs import GenesisConfigsForRun
from rlworld.rl.runners import OnPolicyRunner
from rlworld.rl.configs.presets.inverted_double_pendulum.mlp import get_config
import gymnasium as gym


def main():
    # Get complete config from preset
    configs_dict = get_config()

    # Create configs and runner
    cfgs_for_run = GenesisConfigsForRun.from_dict_with_overrides(configs_dict)
    from rlworld.rl.envs import GymnasiumEnv
    from gymnasium.vector import SyncVectorEnv
    def make_env(seed):
        def _init():
            return gym.make("InvertedDoublePendulum-v4")

        return _init

    num_envs = 128
    env_gym = SyncVectorEnv([make_env(i) for i in range(num_envs)])
    env = GymnasiumEnv(env_gym, seed=42)

    runner = OnPolicyRunner(env=env, cfgs=cfgs_for_run, use_wandb=True)

    # Start training
    runner.learn(
        num_learning_iterations=cfgs_for_run.runner.max_iterations,
        init_at_random_ep_len=cfgs_for_run.runner.init_at_random_ep_len
    )


if __name__ == "__main__":
    main()
