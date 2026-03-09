import os

os.environ['__NV_PRIME_RENDER_OFFLOAD'] = '1'
os.environ['__GLX_VENDOR_LIBRARY_NAME'] = 'nvidia'

custom_assets = os.path.abspath(os.path.join(os.path.dirname(__file__), 'assets'))
import genesis.utils.terrain

genesis.utils.misc.get_assets_dir = lambda: custom_assets
genesis.utils.terrain.get_assets_dir = lambda: custom_assets

from rlworld.rl.configs import GenesisConfigsForRun
from rlworld.rl.runners import OnPolicyRunner
from rlworld.rl.configs.presets.t1_flat.crba import get_config


def main():
    # Get complete config from preset
    configs_dict = get_config()

    # Create configs and runner
    cfgs_for_run = GenesisConfigsForRun.from_dict_with_overrides(configs_dict)

    cfgs_for_run.nn.policy.actor_kwargs.update({
        "encoder_type": "CRBAAttentionBiasedEncoder",
        "latent_dim": 64,
        "spatial_dim": 5,
        "num_layers": 4,
        "use_auxiliary_loss": False,
        "use_adjacency_mask": False,
        "interleave_mask": False,
    })

    runner = OnPolicyRunner.create_with_env(cfgs_for_run, show_viewer=False)

    # Start training
    runner.learn(
        num_learning_iterations=cfgs_for_run.runner.max_iterations,
        init_at_random_ep_len=cfgs_for_run.runner.init_at_random_ep_len
    )


if __name__ == "__main__":
    main()
