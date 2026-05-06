import os

os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"
os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"

custom_assets = os.path.abspath(os.path.join(os.path.dirname(__file__), "assets"))
import genesis.utils.terrain

genesis.utils.misc.get_assets_dir = lambda: custom_assets
genesis.utils.terrain.get_assets_dir = lambda: custom_assets

from rlworld.rl.configs import GenesisConfigsForRun
from rlworld.rl.configs.presets.branched_multilink.mlp import get_config
from rlworld.rl.runners import OnPolicyRunner

# small = [700, 400]      #
# medium = [800, 512, 256]        # Actor + Critic: 112k
large = [1024, 800, 512]
# extreme_large = [2000, 2000, 2000]


def main():
    # Get complete config from preset
    configs_dict = get_config()

    # Create configs and runner
    cfgs_for_run = GenesisConfigsForRun.from_dict_with_overrides(configs_dict)
    cfgs_for_run.nn.policy.actor_kwargs.update({"hidden_dims": large})
    runner = OnPolicyRunner.create_with_env(cfgs_for_run, show_viewer=True)

    # Start training
    runner.learn(
        num_learning_iterations=cfgs_for_run.runner.max_iterations,
        init_at_random_ep_len=cfgs_for_run.runner.init_at_random_ep_len,
    )


if __name__ == "__main__":
    main()
