from rlworld.rl.actuators import IdealPDActuatorCfg
from rlworld.rl.configs.presets.go2_flat.mlp import get_config
from rlworld.rl.configs.robots.go2 import Go2Config
from rlworld.rl.runners import BaseRunner


def main():
    cfgs_for_run = get_config(sim="genesis").with_cli_overrides()
    cfgs_for_run.action.actuator_cfg = IdealPDActuatorCfg(
        stiffness=Go2Config().p_gains,
        damping=Go2Config().d_gains,
    )

    runner = BaseRunner.create_with_env(cfgs_for_run)

    # Start training
    runner.learn(
        num_learning_iterations=cfgs_for_run.runner.max_iterations,
        init_at_random_ep_len=cfgs_for_run.runner.init_at_random_ep_len,
    )


if __name__ == "__main__":
    main()
