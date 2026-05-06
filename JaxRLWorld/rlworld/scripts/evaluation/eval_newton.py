import argparse

import numpy as np
import torch
from consysid.sysid.param_terms.newton import (
    apply_contact_friction,
)

from rlworld.rl.evals import PolicyEvaluator


def apply_contact_friction(
    env,
    values: np.ndarray,
    env_ids: torch.Tensor,
    body_pattern: str = ".*",
) -> None:
    """Set contact friction coefficient. values: (B, 1).

    Sets per-env friction on the matched body shapes (e.g. feet) AND on
    the ground/terrain geoms in the MuJoCo solver. Both are required
    because the Newton MuJoCo wrapper combines pair friction with max(),
    so the ground's friction would otherwise override the foot value.
    """
    import warp as wp
    from newton.solvers import SolverNotifyFlags

    from rlworld.rl.envs.utils.newton.body_cache import get_cache

    cache = get_cache(env)
    model = env.scene_manager.model
    body_indices = cache.get_body_indices(body_pattern)

    # shape_material_mu may contain trailing global shapes (e.g. ground) that
    # don't divide evenly by num_envs. Slice off those trailing entries before
    # reshaping into per-env layout.
    shapes_per_env = model.shape_count // env.num_envs
    n_robot_shapes = env.num_envs * shapes_per_env
    flat_mu = wp.to_torch(model.shape_material_mu)
    shape_mu = flat_mu[:n_robot_shapes].reshape(env.num_envs, shapes_per_env)

    mu_val = torch.tensor(values[:, 0], dtype=torch.float32, device=env.device)
    for body_idx in body_indices:
        shape_indices = model.body_shapes[body_idx]
        for si in shape_indices:
            shape_mu[env_ids, si] = mu_val

    wp.copy(model.shape_material_mu, wp.from_torch(flat_mu, dtype=wp.float32))
    env.scene_manager.solver.notify_model_changed(SolverNotifyFlags.SHAPE_PROPERTIES)

    # Also set ground/terrain geom friction directly on mjw_model so that
    # max(foot_mu, ground_mu) = foot_mu in the contact pair.
    solver = env.scene_manager.solver
    mj_model = solver.mj_model
    mjw_friction = wp.to_torch(solver.mjw_model.geom_friction)  # [nworld, ngeom, 3]

    if not hasattr(env, "_ground_geom_indices"):
        ground_indices = []
        for i in range(mj_model.ngeom):
            name = mj_model.geom(i).name.lower()
            if "terrain" in name or "ground" in name or "plane" in name:
                ground_indices.append(i)
        env._ground_geom_indices = ground_indices

    for gi in env._ground_geom_indices:
        mjw_friction[env_ids, gi, 0] = mu_val


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Newton evaluation")
    parser.add_argument("--eval", action="store_true", help="Run batch evaluation instead of interactive viewer")
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--port", type=int, default=2026, help="Viser viewer port")
    args = parser.parse_args()

    overrides = {
        "env": {
            "num_envs": 1,
            "episode_length_s": 10e9,
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
        policy_path="./outputs/models/2026-04-22/22-26-13/checkpoint_latest/",
        # wandb_run_path="jsw7460/T1_Tracking/gx62rsbe",
        seed=42,
        num_evals=100000000,
        record_video=args.record_video,
        record_steps=None,
        video_dir=None,
        extra_overrides=overrides,
    )

    # apply_contact_friction(
    #     env=evaluator.env,
    #     values=np.array([[0.1],]),
    #     env_ids=torch.tensor([0]),
    #     body_pattern=".*foot$"
    # )

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
