"""Multi-Simulator PPO training for G1 29-DOF.

Trains a single PPO policy using rollouts from Genesis, Newton, and MuJoCo
simultaneously — a form of "simulator randomization" that goes beyond
per-simulator domain randomization.

All environments share the same observation terms (common proprioception
functions via the RobotData protocol) so that obs/action dimensions are
guaranteed to match across simulators.

Usage:
    python -m rlworld.scripts.g1_29dof.multi_sim.ppo_mlp

    # Override per-simulator env counts (default: 1024 each)
    python -m rlworld.scripts.g1_29dof.multi_sim.ppo_mlp \\
        --genesis_num_envs 2048 --newton_num_envs 512 --mujoco_num_envs 512
"""

import argparse
import random

import numpy as np
import torch

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =====================================================================
# Common observation terms (RobotData protocol — works on ALL sims)
# =====================================================================


def _build_common_obs():
    """Build unified obs terms that work identically on Genesis/Newton/MuJoCo.

    Actor obs: ang_vel, gravity, command, dof_pos, dof_vel, prev_actions
    Critic obs: actor obs + base_lin_vel + base_height
    """
    from rlworld.rl.configs.observations import ObservationTermConfig
    from rlworld.rl.configs.observations.noise import UniformNoiseConfig as Unoise
    from rlworld.rl.envs.mdp.observations.common.proprioception import (
        base_ang_vel,
        base_height,
        base_lin_vel,
        command,
        dof_pos,
        dof_vel,
        prev_processed_actions,
        projected_gravity,
    )

    actor_terms = [
        ObservationTermConfig(func=base_ang_vel, scale=1.0, noise=Unoise(-0.2, 0.2)),
        ObservationTermConfig(func=projected_gravity, scale=1.0, noise=Unoise(-0.05, 0.05)),
        ObservationTermConfig(func=command, scale=1.0),
        ObservationTermConfig(func=dof_pos, scale=1.0, noise=Unoise(-0.01, 0.01)),
        ObservationTermConfig(func=dof_vel, scale=1.0, noise=Unoise(-1.5, 1.5)),
        ObservationTermConfig(func=prev_processed_actions, scale=1.0),
    ]

    critic_terms = [
        ObservationTermConfig(func=base_lin_vel, scale=1.0, noise=Unoise(-0.5, 0.5)),
        ObservationTermConfig(func=base_ang_vel, scale=1.0, noise=Unoise(-0.2, 0.2)),
        ObservationTermConfig(func=projected_gravity, scale=1.0, noise=Unoise(-0.05, 0.05)),
        ObservationTermConfig(func=command, scale=1.0),
        ObservationTermConfig(func=dof_pos, scale=1.0, noise=Unoise(-0.01, 0.01)),
        ObservationTermConfig(func=dof_vel, scale=1.0, noise=Unoise(-1.5, 1.5)),
        ObservationTermConfig(func=prev_processed_actions, scale=1.0),
        ObservationTermConfig(func=base_height, scale=1.0),
    ]

    return {"actor": actor_terms, "critic": critic_terms}


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Sim PPO for G1 29-DOF")
    parser.add_argument("--genesis_num_envs", type=int, default=1024)
    parser.add_argument("--newton_num_envs", type=int, default=1024)
    parser.add_argument("--mujoco_num_envs", type=int, default=1024)
    parser.add_argument("--max_iterations", type=int, default=30000)
    parser.add_argument("--no_wandb", action="store_true")
    args, _ = parser.parse_known_args()
    return args


def main():
    args = parse_args()

    # ── Import configs ──
    from rlworld.rl.configs.presets.g1_29dof.mlp import get_config
    from rlworld.rl.envs.multi_sim_world import MultiSimWorld
    from rlworld.rl.runners import BaseRunner
    from rlworld.rl.runners.on_policy_runner import OnPolicyRunner

    # ── Common obs terms (identical across all 3 sims) ──
    common_obs = _build_common_obs()

    # ── Build per-simulator configs ──
    genesis_cfg = get_config(sim="genesis")
    genesis_cfg.env.num_envs = args.genesis_num_envs
    genesis_cfg.algorithm.obs_normalization = True
    genesis_cfg.observation.obs_group = common_obs

    newton_cfg = get_config(sim="newton")
    newton_cfg.env.num_envs = args.newton_num_envs
    newton_cfg.algorithm.obs_normalization = True
    newton_cfg.observation.obs_group = common_obs

    mujoco_cfg = get_config(sim="mujoco")
    mujoco_cfg.env.num_envs = args.mujoco_num_envs
    mujoco_cfg.algorithm.obs_normalization = True
    mujoco_cfg.observation.obs_group = common_obs

    # ── Create individual environments ──
    print("Creating Genesis environment...")
    genesis_env = BaseRunner._create_env_from_config(genesis_cfg)

    print("Creating Newton environment...")
    newton_env = BaseRunner._create_env_from_config(newton_cfg)

    print("Creating MuJoCo environment...")
    mujoco_env = BaseRunner._create_env_from_config(mujoco_cfg)

    # ── Wrap in MultiSimWorld ──
    multi_env = MultiSimWorld([genesis_env, newton_env, mujoco_env])
    print(f"\n{multi_env}")
    print(f"Total environments: {multi_env.num_envs}")
    print(f"Obs dims: {multi_env.calculate_obs_dim()}")
    print(f"Action dim: {multi_env.num_actions}\n")

    # ── Use genesis config as the "primary" config for algorithm/nn/runner ──
    primary_cfg = genesis_cfg
    primary_cfg.runner.run_name = "G1_29Dof_MultiSim_PPO"
    primary_cfg.runner.max_iterations = args.max_iterations

    # ── Create runner directly with the multi-env ──
    runner = OnPolicyRunner(
        env=multi_env,
        cfgs=primary_cfg,
        use_wandb=not args.no_wandb,
    )

    # ── Train ──
    runner.learn(
        num_learning_iterations=primary_cfg.runner.max_iterations,
        init_at_random_ep_len=primary_cfg.runner.init_at_random_ep_len,
    )


if __name__ == "__main__":
    main()
