import argparse

from rlworld.rl.evals import PolicyEvaluator

import newton
from newton import ShapeFlags
from consysid.sysid.param_terms.newton import apply_contact_friction
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
        # policy_path="./outputs/models/2026-04-07/13-20-03/checkpoint_latest/",
        wandb_run_path="jsw7460/RLArchitecture/6be952e4",
        seed=42,
        num_evals=100000000,
        record_video=args.record_video,
        record_steps=None,
        video_dir=None,
        extra_overrides=overrides,
    )

    env = evaluator.env

    # 1. body 이름 확인
    from rlworld.rl.envs.utils.newton.body_cache import get_cache

    cache = get_cache(env)
    print(f"[DEBUG] body names: {cache.body_names}")

    # 2. foot body match 확인
    foot_indices = cache.get_body_indices(".*foot$")
    print(f"[DEBUG] matched foot body indices: {foot_indices}")

    # 3. friction 적용 전 값
    import warp as wp

    shape_mu_before = wp.to_torch(env.scene_manager.model.shape_material_mu).clone()
    print(f"[DEBUG] mu before: min={shape_mu_before.min()}, max={shape_mu_before.max()}, sample={shape_mu_before[:10]}")

    # 4. friction 적용
    import torch
    import numpy as np
    from consysid.sysid.param_terms.newton import apply_contact_friction

    num_envs = env.num_envs
    values = np.full((num_envs, 1), 0.05, dtype=np.float32)
    env_ids = torch.arange(num_envs, device=env.device)
    apply_contact_friction(env, values, env_ids, body_pattern=".*foot$")

    # 5. friction 적용 후 값
    shape_mu_after = wp.to_torch(env.scene_manager.model.shape_material_mu).clone()
    print(f"[DEBUG] mu after: min={shape_mu_after.min()}, max={shape_mu_after.max()}, sample={shape_mu_after[:10]}")
    print(f"[DEBUG] changed entries: {(shape_mu_before != shape_mu_after).sum()}")

    import ipdb; ipdb.set_trace()

    if args.eval:
        evaluator.evaluate()
    else:
        evaluator.play(port=args.port)
