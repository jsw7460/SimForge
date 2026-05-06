import os

os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"
os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"

custom_assets = os.path.abspath(os.path.join(os.path.dirname(__file__), "assets"))
import genesis.utils.terrain

genesis.utils.misc.get_assets_dir = lambda: custom_assets
genesis.utils.terrain.get_assets_dir = lambda: custom_assets

from rlworld.rl.configs import GenesisConfigsForRun
from rlworld.rl.configs.presets.maniskill.hopper_hop.aba import get_config
from rlworld.rl.runners import OnPolicyRunner

# 아래: vD0
medium = {
    "encoder_type": "ABDEncoder",
    "aba_link_channels": 9,
    "aba_spatial_dim": 6,
    "use_auxiliary_loss": False,
    "aba_orth_loss_weight": 1.0,
    "aba_orth_loss_decay": 1.0,
    "mlp_hidden_mult": 1,
    "decoder_hidden_dim": 128,
    "interleave_mask": True,
    "use_adjacency_mask": True,
    "rodrigues_use_global_layer_norm": True,
    "use_positive_constraint": True,
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
