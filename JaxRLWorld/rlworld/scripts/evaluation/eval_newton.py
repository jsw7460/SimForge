from rlworld.rl.evals import PolicyEvaluator

if __name__ == '__main__':
    evaluator = PolicyEvaluator(
        policy_path=f"./outputs/models/2026-03-16/14-25-49/checkpoint_latest/",  # Newton
        seed=42,
        num_evals=100000000,
        show_viewer=True,
        record_video=True,
        record_steps=None,
        video_dir=None,
        extra_overrides={
            "env": {
                "num_envs": 1,
                "episode_length_s": 10e+9,
            },
            "visualization": {
                "viser_port": 2026,
                "viewer_type": "viser",
            },
            "command": {
                "rel_standing_envs": 0.3
            }
        },
    )
    evaluator.evaluate()
