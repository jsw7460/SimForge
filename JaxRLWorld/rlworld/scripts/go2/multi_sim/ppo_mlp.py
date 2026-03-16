"""Multi-Simulator PPO training for Go2.

Trains a single PPO policy using rollouts from Genesis and Newton
simultaneously — "simulator randomization" for quadruped locomotion.

Usage:
    python -m rlworld.scripts.go2.multi_sim.ppo_mlp

    # Override per-simulator env counts (default: 2048 each)
    python -m rlworld.scripts.go2.multi_sim.ppo_mlp \\
        --genesis_num_envs 2048 --newton_num_envs 2048
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
# Common observation terms (RobotData protocol — works on both sims)
# =====================================================================

def _build_common_obs():
    """Build unified obs terms for Go2 Genesis+Newton.

    Matches the existing Go2 preset obs structure:
    - Actor: ang_vel, gravity, command, dof_pos, dof_vel, prev_actions
      (no base_lin_vel, matching both presets' include_base_lin_vel=False)
    - Critic: actor obs + base_lin_vel + base_height
    """
    from rlworld.rl.configs.observations import ObservationTermConfig
    from rlworld.rl.configs.observations.noise import UniformNoiseConfig as Unoise
    from rlworld.rl.envs.mdp.observations.common.proprioception import (
        base_lin_vel,
        base_ang_vel,
        base_height,
        projected_gravity,
        dof_pos,
        dof_vel,
        prev_processed_actions,
    )
    from rlworld.rl.envs.mdp.observations.genesis.exteroception import command

    actor_terms = [
        ObservationTermConfig(func=base_ang_vel, scale=0.25, noise=Unoise(-0.2, 0.2)),
        ObservationTermConfig(func=projected_gravity, scale=1.0, noise=Unoise(-0.05, 0.05)),
        ObservationTermConfig(func=command, scale=1.0),
        ObservationTermConfig(func=dof_pos, scale=1.0, noise=Unoise(-0.01, 0.01)),
        ObservationTermConfig(func=dof_vel, scale=0.05, noise=Unoise(-1.5, 1.5)),
        ObservationTermConfig(func=prev_processed_actions, scale=1.0),
    ]

    critic_terms = [
        ObservationTermConfig(func=base_lin_vel, scale=1.0),
        ObservationTermConfig(func=base_ang_vel, scale=0.25, noise=Unoise(-0.2, 0.2)),
        ObservationTermConfig(func=projected_gravity, scale=1.0, noise=Unoise(-0.05, 0.05)),
        ObservationTermConfig(func=command, scale=1.0),
        ObservationTermConfig(func=dof_pos, scale=1.0, noise=Unoise(-0.01, 0.01)),
        ObservationTermConfig(func=dof_vel, scale=0.05, noise=Unoise(-1.5, 1.5)),
        ObservationTermConfig(func=prev_processed_actions, scale=1.0),
        ObservationTermConfig(func=base_height, scale=1.0),
    ]

    return {"actor": actor_terms, "critic": critic_terms}


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Sim PPO for Go2")
    parser.add_argument("--genesis_num_envs", type=int, default=2048)
    parser.add_argument("--newton_num_envs", type=int, default=2048)
    parser.add_argument("--max_iterations", type=int, default=6000)
    parser.add_argument("--no_wandb", action="store_true")
    args, _ = parser.parse_known_args()
    return args


def main():
    args = parse_args()

    # ── Import configs ──
    from rlworld.rl.configs.presets.go2_flat.genesis.mlp import (
        get_config as get_genesis_config,
    )
    from rlworld.rl.configs.presets.go2_flat.newton.mlp import (
        get_config as get_newton_config,
    )
    from rlworld.rl.envs.multi_sim_world import MultiSimWorld
    from rlworld.rl.runners import BaseRunner
    from rlworld.rl.runners.on_policy_runner import OnPolicyRunner

    # ── Common obs terms ──
    common_obs = _build_common_obs()

    # ── Build per-simulator configs ──
    genesis_cfg = get_genesis_config()
    genesis_cfg.env.num_envs = args.genesis_num_envs
    genesis_cfg.algorithm.obs_normalization = True
    genesis_cfg.observation.obs_group = common_obs

    newton_cfg = get_newton_config()
    newton_cfg.env.num_envs = args.newton_num_envs
    newton_cfg.algorithm.obs_normalization = True
    newton_cfg.observation.obs_group = common_obs

    # ── Create individual environments ──
    print("Creating Genesis environment...")
    genesis_env = BaseRunner._create_env_from_config(genesis_cfg)

    print("Creating Newton environment...")
    newton_env = BaseRunner._create_env_from_config(newton_cfg)

    # ── Wrap in MultiSimWorld ──
    multi_env = MultiSimWorld([genesis_env, newton_env])
    print(f"\n{multi_env}")
    print(f"Total environments: {multi_env.num_envs}")
    print(f"Obs dims: {multi_env.calculate_obs_dim()}")
    print(f"Action dim: {multi_env.num_actions}\n")

    # ── Use genesis config as primary (for algorithm/nn/runner) ──
    primary_cfg = genesis_cfg
    primary_cfg.runner.run_name = "Go2_MultiSim_PPO"
    primary_cfg.runner.max_iterations = args.max_iterations

    # ── Create runner ──
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
