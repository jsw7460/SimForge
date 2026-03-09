import warp as wp

from rlworld.rl.configs.robots.g1_29dof import G1MjlabConfig
from rlworld.rl.evals import PolicyEvaluator

if __name__ == '__main__':
    evaluator = PolicyEvaluator(
        eval_env_cfgs=None,
        # wandb_run_path="jsw7460/RLArchitecture/2jzsqo16",
        policy_path=f"./outputs/models/2026-03-07/18-47-17/checkpoint_latest/",
        seed=42,
        num_evals=100000000,
        show_viewer=True,
        record_video=True,
        record_steps=None,
        video_dir=None,
        extra_overrides={
            "env": {
                "num_envs": 1,
                # "base_init_pos": [0.0, 0.0, g1_29dof.base_init_height],
                "episode_length_s": 10e+9,
                # "termination_criteria": [],
            },
            "visualization": {
                # "viser_share": True,
                "viser_port": 2026,
                "viewer_type": "viser",
            },
            # "scene": {
            #     "entities": entities,
            #     "robot_cfg": g1_29dof
            # },
            "command": {
                "rel_standing_envs": 0.5,
            }
        },
    )
    evaluator.evaluate()
