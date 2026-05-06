import os

os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"
os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"

custom_assets = os.path.abspath(os.path.join(os.path.dirname(__file__), "assets"))
import genesis.utils.terrain

genesis.utils.misc.get_assets_dir = lambda: custom_assets
genesis.utils.terrain.get_assets_dir = lambda: custom_assets

from rlworld.rl.configs import GenesisConfigsForRun
from rlworld.rl.configs.presets.gymnasium.hopper.aba import get_config
from rlworld.rl.runners import OnPolicyRunner

medium = {
    "encoder_type": "ABARodriguesEncoder",
    "hidden_dim": 64,
    "embed_dim": 64,
    "aba_link_channels": 8,
    "aba_spatial_dim": 3,
    "use_auxiliary_loss": True,
    "num_layers": 2,
    "num_heads": 4,
    "dim_feedforward": 256,
    "aba_orth_loss_weight": 1.0,
    "rodrigues_joint_channels": 3,
    "rodrigues_link_channels": 6,
    "rodrigues_spatial_dim": 3,
    "rodrigues_parent_contribution": 0.1,
    "rodrigues_use_stable_init": True,
    "rodrigues_use_global_layer_norm": True,
    "decoder_hidden_dim": 128,
    "interleave_mask": False,
    "use_adjacency_mask": False,
}


def main():
    # Get complete config from preset
    configs_dict = get_config()

    # Create configs and runner
    cfgs_for_run = GenesisConfigsForRun.from_dict_with_overrides(configs_dict)

    # num params: 1.5m
    cfgs_for_run.nn.policy.actor_kwargs.update(**medium)

    runner = OnPolicyRunner.create_with_env(cfgs_for_run)

    # Start training
    runner.learn(
        num_learning_iterations=cfgs_for_run.runner.max_iterations,
        init_at_random_ep_len=cfgs_for_run.runner.init_at_random_ep_len,
    )


if __name__ == "__main__":
    main()
