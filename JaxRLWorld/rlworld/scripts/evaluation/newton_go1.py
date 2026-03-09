import warp as wp

# mjlab imports for scene configuration
from mjlab.terrains import TerrainImporterCfg
from mjlab.terrains import TerrainImporterCfg
from rlworld.rl.configs.events import EventTermConfig
from rlworld.rl.configs.robots.go1 import Go1Config
from rlworld.rl.envs.mdp.events import mujoco_event_terms as ef
from rlworld.rl.envs.mdp.events.mujoco_event_terms import EntityCfg
from rlworld.rl.evals import PolicyEvaluator

robot = Go1Config()

if __name__ == '__main__':
    quat = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi * 0.5)

    evaluator = PolicyEvaluator(
        eval_env_cfgs=None,
        policy_path=f"./outputs/models/2026-03-08/11-58-07/checkpoint_latest/",  # Newton
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
                # "termination_criteria": [],
            },
            "visualization": {
                # "viser_share": True,
                "viser_port": 2026,
                "viewer_type": "viser",
            },
            "command": {
                "rel_standing_envs": 0.3
            }
        },
    )
    evaluator.evaluate()
