from rlworld.rl.evals import PolicyEvaluator
from rlworld.rl.envs.mdp.configs.commands.command_term_config import CommandTermConfig
from rlworld.rl.envs.mdp.commands import command_terms as cf
from rlworld.rl.configs.robots.go2 import Go2Config

if __name__ == '__main__':
    evaluator = PolicyEvaluator(
        eval_env_cfgs=None,
        # wandb_run_path="jsw7460/RLArchitecture/b80sk0ys",
        policy_path="./outputs/models/2026-03-08/12-10-17/checkpoint_latest/",
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
            "visualization": {
                "viewer_type": "viser",
                "viser_port": 2028,
            },
            "command": {
                "rel_standing_envs": 0.3
            }
        },
    )
    evaluator.evaluate()
