from rlworld.rl.configs import NewtonConfigsForRun
from rlworld.rl.configs.presets.g1_29dof.newton.mlp import get_config
from rlworld.rl.runners import BaseRunner
# import jax
# jax.config.update("jax_debug_nans", True)

def main():
    # Get complete config from preset
    configs_dict = get_config()

    # Create configs and runner
    cfgs_for_run = NewtonConfigsForRun.from_dict_with_overrides(configs_dict)

    runner = BaseRunner.create_with_env(cfgs_for_run)

    # Start training
    runner.learn(
        num_learning_iterations=cfgs_for_run.runner.max_iterations,
        init_at_random_ep_len=cfgs_for_run.runner.init_at_random_ep_len
    )


if __name__ == "__main__":
    main()
