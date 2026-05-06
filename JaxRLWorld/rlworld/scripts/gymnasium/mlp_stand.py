import os

os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"
os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"

custom_assets = os.path.abspath(os.path.join(os.path.dirname(__file__), "assets"))
import genesis.utils.terrain
import gymnasium as gym

genesis.utils.misc.get_assets_dir = lambda: custom_assets
genesis.utils.terrain.get_assets_dir = lambda: custom_assets

from rlworld.rl.configs import GenesisConfigsForRun
from rlworld.rl.configs.presets.go2_terrain.mlp import get_config
from rlworld.rl.envs import GymnasiumEnv
from rlworld.rl.runners import OnPolicyRunner

actor_medium = [1024, 800, 512]  # Actor 160k
critic_medium = [512, 512, 256]  # critic: 58k


def main():
    # Get complete config from preset
    configs_dict = get_config()
    configs_dict["runner"]["run_name"] = "MLPStandTest"

    # Create configs and runner
    cfgs_for_run = GenesisConfigsForRun.from_dict_with_overrides(configs_dict)
    cfgs_for_run.env.num_envs = 1024
    cfgs_for_run.nn.policy.actor_kwargs.update({"hidden_dims": actor_medium})
    cfgs_for_run.nn.policy.critic_kwargs.update({"hidden_dims": critic_medium})

    env = gym.vector.SyncVectorEnv(
        [lambda: gym.make("HumanoidStandup-v4", max_episode_steps=1000) for _ in range(cfgs_for_run.env.num_envs)]
    )
    env = GymnasiumEnv(env)

    runner = OnPolicyRunner(env=env, cfgs=cfgs_for_run, use_wandb=True)

    # Start training
    runner.learn(
        num_learning_iterations=cfgs_for_run.runner.max_iterations,
        init_at_random_ep_len=cfgs_for_run.runner.init_at_random_ep_len,
    )


if __name__ == "__main__":
    main()
