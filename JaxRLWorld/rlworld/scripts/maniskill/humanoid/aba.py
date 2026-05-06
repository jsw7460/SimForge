import os

os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"
os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"

custom_assets = os.path.abspath(os.path.join(os.path.dirname(__file__), "assets"))
import genesis.utils.terrain

genesis.utils.misc.get_assets_dir = lambda: custom_assets
genesis.utils.terrain.get_assets_dir = lambda: custom_assets

from rlworld.rl.configs import GenesisConfigsForRun
from rlworld.rl.configs.presets.maniskill.humanoid.aba import get_config
from rlworld.rl.runners import OnPolicyRunner


def main():
    # Get complete config from preset
    configs_dict = get_config()

    # Create configs and runner
    cfgs_for_run = GenesisConfigsForRun.from_dict_with_overrides(configs_dict)

    cfgs_for_run.nn.policy.actor_kwargs.update(
        {
            "encoder_type": "DualBiasedEncoder",
            "hidden_dim": 32,
            "embed_dim": 32,
            "aba_link_channels": 8,
            "aba_spatial_dim": 3,
            "use_auxiliary_loss": True,
            "num_layers": 4,
            "num_heads": 2,
            "dim_feedforward": 32,
            "rodrigues_use_stable_init": True,
            "decoder_hidden_dim": 64,
            "interleave_mask": True,
            "use_adjacency_mask": True,
        }
    )
    runner = OnPolicyRunner.create_with_env(cfgs_for_run)

    # Start training
    runner.learn(
        num_learning_iterations=cfgs_for_run.runner.max_iterations,
        init_at_random_ep_len=cfgs_for_run.runner.init_at_random_ep_len,
    )


if __name__ == "__main__":
    main()
