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

verysmall = {
    "encoder_type": "ABAEncoder",
    "hidden_dim": 32,
    "embed_dim": 32,
    "aba_link_channels": 4,
    "aba_spatial_dim": 2,
    "use_auxiliary_loss": True,
    "aba_orth_loss_weight": 1.0,
    "aba_orth_loss_decay": 1.0,
    "decoder_hidden_dim": 64,
    "use_positive_constraint": True,
}

# medium = {
#     "encoder_type": "ABAEncoder",
#
#     "hidden_dim": 256,
#     "embed_dim": 512,
#
#     "aba_link_channels": 12,
#     "aba_spatial_dim": 6,
#     "use_auxiliary_loss": True,
#     "aba_orth_loss_weight": 1.0,
#     "aba_orth_loss_decay": 1.0,
#
#     "decoder_hidden_dim": 128,
#     "use_positive_constraint": True,
# }


def main():
    # Get complete config from preset
    configs_dict = get_config()

    # Create configs and runner
    cfgs_for_run = GenesisConfigsForRun.from_dict_with_overrides(configs_dict)

    # num params: 1.5m
    cfgs_for_run.nn.policy.actor_kwargs.update(**verysmall)

    runner = OnPolicyRunner.create_with_env(cfgs_for_run)

    # Start training
    runner.learn(
        num_learning_iterations=cfgs_for_run.runner.max_iterations,
        init_at_random_ep_len=cfgs_for_run.runner.init_at_random_ep_len,
    )


if __name__ == "__main__":
    main()
