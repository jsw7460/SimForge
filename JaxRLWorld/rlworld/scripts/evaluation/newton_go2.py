from rlworld.rl.evals import PolicyEvaluator
from rlworld.rl.envs.mdp.configs.commands.command_term_config import CommandTermConfig
from rlworld.rl.envs.mdp.commands import command_terms as cf
from rlworld.rl.configs.robots.go2 import Go2Config

if __name__ == '__main__':
    evaluator = PolicyEvaluator(
        eval_env_cfgs=None,
        # policy_path=f"./outputs/models/2026-01-16/11-07-56/checkpoint_500/",  # ABAGNN
        policy_path=f"./outputs/models/2026-02-23/22-08-19/checkpoint_latest/",  # ABAGNN
        # policy_path=f"./outputs/models/2026-01-10/23-36-14/checkpoint_6000/",       # MLP
        num_evals=1,
        seed=42,
        show_viewer=True,
        record_video=True,
        record_steps=None,
        video_dir=None,
        extra_overrides={
            "env": {
                "num_envs": 1,
                "episode_length_s": 10e9,
                # "seed": 43,
                # "termination_criteria": [],
            },
            "scene": {
                "robot_cfg": Go2Config()
            },
            "visualization": {
                "viewer_type": "viser",
                "viser_port": 8080,
            },
            # "command": {
            #     "sampler": [
            #         CommandTermConfig(cf.lin_vel_x, params={"range": (1.0, 1.0)}),
            #         CommandTermConfig(cf.lin_vel_y, params={"range": (1.0, 1.0)}),
            #         CommandTermConfig(cf.ang_vel, params={"range": (-0.0, 0.0)}),
            #         CommandTermConfig(cf.base_height, params={"range": (0.34, 0.34)})
            #     ]
            #     # "resampling_time_s": (5.0, 5.0),
            #     # "sampler": [
            #     #     CommandTermConfig(cf.lin_vel_x, params={"range": (-1.0, 1.0)}),
            #     #     CommandTermConfig(cf.lin_vel_y, params={"range": (-1.0, 1.0)}),
            #     #     CommandTermConfig(cf.ang_vel, params={"range": (-0.0, 0.0)}),
            #     #     CommandTermConfig(cf.base_height, params={"range": (0.34, 0.34)})
            #     # ]
            # }
        },
    )
    evaluator.evaluate()
