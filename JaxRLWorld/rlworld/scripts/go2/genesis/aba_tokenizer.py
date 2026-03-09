import os

os.environ['__NV_PRIME_RENDER_OFFLOAD'] = '1'
os.environ['__GLX_VENDOR_LIBRARY_NAME'] = 'nvidia'

custom_assets = os.path.abspath(os.path.join(os.path.dirname(__file__), 'assets'))
import genesis.utils.terrain

genesis.utils.misc.get_assets_dir = lambda: custom_assets
genesis.utils.terrain.get_assets_dir = lambda: custom_assets

from rlworld.rl.configs import GenesisConfigsForRun
from rlworld.rl.runners import OnPolicyRunner
from rlworld.rl.configs.presets.go2_terrain.aba import get_config

medium = {
    "encoder_type": "ABATransformerEncoder",

    "hidden_dim": 72,
    "embed_dim": 72,

    "aba_link_channels": 8,
    "aba_spatial_dim": 6,
    "aba_learnable_contribution_weight": False,
    "aba_orth_loss_weight": 1.0,
    "aba_global_layer_norm": False,
    "use_auxiliary_loss": True,

    "pe_type": "traversal",
    "re_type": "graph",
    "use_laplacian": True,
    "use_spd": True,
    "use_ppr": True,
    "ppr_alpha": 0.15,

    "num_layers": 10,
    "num_heads": 4,
    "dim_feedforward": 512,

    "use_adjacency_mask": False,
    "interleave_mask": False,

    "decoder_hidden_dim": 128,
}

medium2 = {
    "encoder_type": "ABATransformerEncoder",

    "hidden_dim": 200,
    "embed_dim": 200,

    "aba_link_channels": 10,
    "aba_spatial_dim": 4,
    "aba_learnable_contribution_weight": False,
    "aba_orth_loss_weight": 1.0,
    "aba_global_layer_norm": False,
    "use_auxiliary_loss": True,

    "pe_type": "traversal",
    "re_type": "graph",
    "use_laplacian": True,
    "use_spd": True,
    "use_ppr": True,
    "ppr_alpha": 0.15,

    "num_layers": 3,
    "num_heads": 4,
    "dim_feedforward": 256,

    "use_adjacency_mask": False,
    "interleave_mask": False,

    "decoder_hidden_dim": 128,

}


def main():
    # Get complete config from preset
    configs_dict = get_config()

    # Create configs and runner
    cfgs_for_run = GenesisConfigsForRun.from_dict_with_overrides(configs_dict)

    # num params: 1.5m
    cfgs_for_run.nn.policy.actor_kwargs.update(**medium2)

    runner = OnPolicyRunner.create_with_env(cfgs_for_run)

    # Start training
    runner.learn(
        num_learning_iterations=cfgs_for_run.runner.max_iterations,
        init_at_random_ep_len=cfgs_for_run.runner.init_at_random_ep_len
    )


if __name__ == "__main__":
    main()
