import os

os.environ['__NV_PRIME_RENDER_OFFLOAD'] = '1'
os.environ['__GLX_VENDOR_LIBRARY_NAME'] = 'nvidia'

custom_assets = os.path.abspath(os.path.join(os.path.dirname(__file__), 'assets'))
import genesis.utils.terrain

genesis.utils.misc.get_assets_dir = lambda: custom_assets
genesis.utils.terrain.get_assets_dir = lambda: custom_assets

from rlworld.rl.runners import BaseRunner
from rlworld.rl.configs.presets.go1.genesis.abdnet import get_config


medium = {
    "encoder_type": "ABAEncoder",
    "link_channels": 12,
    "spatial_dim": 3,
    "use_auxiliary_loss": True,
}
large = [512, 256, 128]


def main():
    cfgs_for_run = get_config().with_cli_overrides()

    cfgs_for_run.nn.policy.actor_kwargs.update(**medium)
    cfgs_for_run.nn.policy.critic_kwargs.update({
        "hidden_dims": large,
    })
    runner = BaseRunner.create_with_env(cfgs_for_run)

    # Start training
    runner.learn(
        num_learning_iterations=cfgs_for_run.runner.max_iterations,
        init_at_random_ep_len=cfgs_for_run.runner.init_at_random_ep_len
    )


if __name__ == "__main__":
    main()
