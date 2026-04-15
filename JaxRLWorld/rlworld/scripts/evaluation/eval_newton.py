import argparse

from rlworld.rl.evals import PolicyEvaluator

import newton
from newton import ShapeFlags
from consysid.sysid.param_terms.newton import (
    apply_contact_friction,
    apply_joint_friction,
)
import numpy as np
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Newton evaluation")
    parser.add_argument("--eval", action="store_true", help="Run batch evaluation instead of interactive viewer")
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--port", type=int, default=2026, help="Viser viewer port")
    args = parser.parse_args()

    overrides = {
        "env": {
            "num_envs": 1,
            "episode_length_s": 10e+9,
        },
        # "command": {
        #     "rel_standing_envs": 0.3,
        # },
    }

    if args.eval:
        overrides["visualization"] = {
            "viser_port": args.port,
            "viewer_type": "viser",
        }

    evaluator = PolicyEvaluator(
        # policy_path="./outputs/models/2026-04-14/21-56-39/checkpoint_latest/",
        wandb_run_path="jsw7460/RLArchitecture/2rg9mo51",
        seed=42,
        num_evals=100000000,
        record_video=args.record_video,
        record_steps=None,
        video_dir=None,
        extra_overrides=overrides,
    )
    #
    # # ============================== DEBUG ==============================
    # # Direct Newton model write via the same SysID apply functions, so the
    # # visual behaviour matches exactly what Stage 1 sees during CMA-ES.
    # env = evaluator.env
    # num_envs = env.num_envs
    # env_ids = torch.arange(num_envs, device=env.device)
    #
    # # Foot contact friction: (B, 1).
    # foot_mu = np.full((num_envs, 1), 0.3, dtype=np.float32)
    # apply_contact_friction(env, foot_mu, env_ids, body_pattern=".*foot$")
    #
    # # Leg-joint Coulomb friction: (B, 12). The default regex targets
    # # every joint whose name ends in "_joint" which on Go2 is exactly
    # # the 12 actuated leg joints (floating_base is skipped).
    # joint_tau = np.full((num_envs, 12), 0.8, dtype=np.float32)
    # apply_joint_friction(
    #     env, joint_tau, env_ids,
    #     joint_patterns=(r".*_joint$",),
    # )
    #
    # # Sanity read-back from the robot_view so we know the write landed.
    # import warp as wp
    # view = env.scene_manager.robot_view
    # jf_back = wp.to_torch(view.get_attribute("joint_friction", env.scene_manager.model))
    # print(f"[DEBUG] joint_friction view shape = {tuple(jf_back.shape)}")
    # print(f"[DEBUG] joint_friction[env0] = {jf_back[0, 0].tolist()}")
    # print(f"[DEBUG] joint_dof_names      = {view.joint_dof_names}")
    # # ============================== /DEBUG =============================

    if args.eval:
        evaluator.evaluate()
    else:
        evaluator.play(port=args.port)
