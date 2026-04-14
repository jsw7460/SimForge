"""Train T1 fall-recovery (getup) policy in Newton."""

from rlworld.rl.configs.presets.t1_getup.base import T1GetupConfig
from rlworld.rl.runners import BaseRunner


def main():
    cfgs_for_run = T1GetupConfig(sim_type="newton").build().with_cli_overrides()

    runner = BaseRunner.create_with_env(cfgs_for_run)

    runner.learn(
        num_learning_iterations=cfgs_for_run.runner.max_iterations,
        init_at_random_ep_len=cfgs_for_run.runner.init_at_random_ep_len,
    )


if __name__ == "__main__":
    main()
