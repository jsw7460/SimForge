import os

os.environ['__NV_PRIME_RENDER_OFFLOAD'] = '1'
os.environ['__GLX_VENDOR_LIBRARY_NAME'] = 'nvidia'

custom_assets = os.path.abspath(os.path.join(os.path.dirname(__file__), 'assets'))
import genesis.utils.terrain

import gymnasium as gym

genesis.utils.misc.get_assets_dir = lambda: custom_assets
genesis.utils.terrain.get_assets_dir = lambda: custom_assets

from rlworld.rl.configs import GenesisConfigsForRun
from rlworld.rl.runners import OnPolicyRunner
from rlworld.rl.envs import GymnasiumEnv
from rlworld.rl.configs.presets.go2_terrain.aba import get_config


def main():
    # Get complete config from preset
    configs_dict = get_config()

    # Create configs and runner
    cfgs_for_run = GenesisConfigsForRun.from_dict_with_overrides(configs_dict)
    cfgs_for_run.env.num_envs = 128
    cfgs_for_run.nn.policy.actor_kwargs.update(
        {
            # "encoder_type": "GatedDeepPhysicsEncoder",
            # "encoder_type": "GatedPhysicsEncoder",
            "encoder_type": "HybridDynamicsKinematicsEncoder",
            "joint_channels": 4,
            "link_channels": 6,
            "num_rodrigues_blocks": 1,
            "hidden_dim": 32,
            "embed_dim": 32,
            "spatial_dim": 6,
            "aba_link_channels": 8,
            "decoder_hidden_dim": 64,
            "gate_hidden_dim": 16,
            "use_auxiliary_loss": True,
        }
    )

    env = gym.make_vec("InvertedDoublePendulum-v4", num_envs=cfgs_for_run.env.num_envs, vectorization_mode="sync")
    env = GymnasiumEnv(env, seed=cfgs_for_run.env.seed)

    runner = OnPolicyRunner(
        env=env, cfgs=cfgs_for_run, use_wandb=True
    )

    # Start training
    runner.learn(
        num_learning_iterations=cfgs_for_run.runner.max_iterations,
        init_at_random_ep_len=cfgs_for_run.runner.init_at_random_ep_len
    )


if __name__ == "__main__":
    main()
