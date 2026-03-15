from rlworld.rl.configs.algorithms import SimMPCConfig
from rlworld.rl.configs.presets.g1_29dof.genesis.mlp import get_config
from rlworld.rl.envs.mdp.configs import (
    TerminationTermConfig,
)
from rlworld.rl.envs.mdp.terminations.common import max_episode_exceed
from rlworld.rl.envs.mdp.terminations.common import terminations as tf
from rlworld.rl.runners import BaseRunner


def main():
    cfgs_for_run = get_config().with_cli_overrides()

    sim_mpc_config = SimMPCConfig(
        horizon=5,
        num_samples=512,
        num_pi_trajs=64,
        num_elites=64,
        num_iterations=6,
        temperature=0.5,
        min_std=0.05,
        max_std=2.0,
        gamma=0.99,
        lr=3e-4,
        pi_lr=3e-4,
        tau=0.005,
        num_q=5,
        hidden_dims=(512, 256),
        batch_size=4096,
        buffer_size=1_000_000,
        learning_starts=1000,
        num_gradient_steps=8,
    )

    cfgs_for_run.env.num_envs = 1
    cfgs_for_run.reward.reward_terms.pop("raw_action_rate_l2_mjlab")
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
    cfgs_for_run.algorithm = sim_mpc_config
    cfgs_for_run.runner.run_name = "G1_SimMPC"
    cfgs_for_run.runner.eval_interval = 0
    cfgs_for_run.runner.log_interval = 1

    runner = BaseRunner.create_with_env(cfgs_for_run)

    # Start running
    runner.learn(
        num_learning_iterations=cfgs_for_run.runner.max_iterations,
        init_at_random_ep_len=cfgs_for_run.runner.init_at_random_ep_len,
    )


if __name__ == "__main__":
    main()
