from rlworld.rl.configs.algorithms import TDMPC2Config
from rlworld.rl.configs.presets.g1_29dof.mujoco.mlp import get_config
from rlworld.rl.runners import BaseRunner
from rlworld.rl.envs.mdp.configs import (
    TerminationTermConfig,
    CommandTermConfig,
)
from rlworld.rl.envs.mdp.terminations.common import terminations as tf
from rlworld.rl.envs.mdp.terminations.common import max_episode_exceed


def main():
    cfgs_for_run = get_config().with_cli_overrides()

    tdmpc2_config = TDMPC2Config(
        vmin=-5.0,
        vmax=10.0,
        num_bins=101,
        num_samples=512,
        num_pi_trajs=128,
        num_elites=64,
        num_iterations=6,
        buffer_size=1024 * 1024 * 100,
        num_gradient_steps=8,
        batch_size=40000,
        learning_starts=5000
    )

    cfgs_for_run.env.num_envs = 1024
    cfgs_for_run.env.termination_criteria = [
                TerminationTermConfig(
                    tf.roll_pitch_violation,
                    {"roll_threshold_degree": 45.0, "pitch_threshold_degree": 45.0}
                ),
                TerminationTermConfig(max_episode_exceed),
        ]
    cfgs_for_run.runner.max_iterations = 100000
    cfgs_for_run.action.clip_actions = "joint_limit"
    cfgs_for_run.action.action_scale = 0.5
    # cfgs_for_run.action.clip_actions = (-5.0, 5.0)
    cfgs_for_run.algorithm = tdmpc2_config
    cfgs_for_run.runner.run_name = "G1_NT_TDMPC2"


    runner = BaseRunner.create_with_env(cfgs_for_run)

    # Start training
    runner.learn(
        num_learning_iterations=cfgs_for_run.runner.max_iterations,
        init_at_random_ep_len=cfgs_for_run.runner.init_at_random_ep_len
    )


if __name__ == "__main__":
    main()
