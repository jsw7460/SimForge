import os

os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"
os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"

custom_assets = os.path.abspath(os.path.join(os.path.dirname(__file__), "assets"))
import genesis.utils.terrain

genesis.utils.misc.get_assets_dir = lambda: custom_assets
genesis.utils.terrain.get_assets_dir = lambda: custom_assets

from rlworld.rl.configs import GenesisConfigsForRun
from rlworld.rl.configs.presets.maniskill.anymal_spin.aba import get_config
from rlworld.rl.runners import OnPolicyRunner

default_encoder_params = {  # 1.8m
    "joint_channels": 3,
    "link_channels": 3,
    "num_rodrigues_blocks": 1,
    "hidden_dim": 32,
    "embed_dim": 32,
    "spatial_dim": 6,
    "aba_link_channels": 12,
    "decoder_hidden_dim": 64,
    "gate_hidden_dim": 64,
    "use_auxiliary_loss": True,
    "interleave_mask": True,
}


def main():
    # Get complete config from preset
    configs_dict = get_config()

    # Create configs and runner
    cfgs_for_run = GenesisConfigsForRun.from_dict_with_overrides(configs_dict)
    cfgs_for_run.nn.policy.actor_kwargs.update(
        {"encoder_type": "HybridDynamicsKinematicsEncoder", **default_encoder_params}
    )

    runner = OnPolicyRunner.create_with_env(cfgs_for_run, use_wandb=True)

    # Start training
    runner.learn(
        num_learning_iterations=cfgs_for_run.runner.max_iterations,
        init_at_random_ep_len=cfgs_for_run.runner.init_at_random_ep_len,
    )


if __name__ == "__main__":
    main()
