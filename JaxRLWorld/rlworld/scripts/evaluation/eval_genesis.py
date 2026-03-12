import genesis as gs
from rlworld.rl.evals import PolicyEvaluator
from rlworld.rl.envs.mdp.commands import command_terms as cf
from rlworld.rl.envs.mdp.configs import CommandTermConfig
from rlworld.rl.vis.overlays.hud_items import LinkPositionItem, LinkPositionItemConfig

from rlworld.rl.configs.robots.g1_29dof import G1MjlabConfig

if __name__ == '__main__':
    g1_29dof = G1MjlabConfig()
    link_pos_item = LinkPositionItem(
        LinkPositionItemConfig(link_patterns=("left_ankle_roll_link", "right_ankle_roll_link")),
    )

    evaluator = PolicyEvaluator(
        eval_env_cfgs=None,
        policy_path=f"outputs/models/2026-03-11/18-17-32/checkpoint_latest/",
        num_evals=1,
        seed=42,
        show_viewer=False,
        record_video=True,
        record_steps=None,
        video_dir=None,
        extra_overrides={
            "env": {
                "num_envs": 1,
                "episode_length_s": 10e+9,
            },
            "scene": {
                "vis_options": gs.options.VisOptions(
                    background_color=(0.4, 0.5, 0.6),
                    ambient_light=(0.7, 0.7, 0.7),
                    shadow=True,
                    plane_reflection=True,
                    lights=[
                        {"type": "directional", "dir": (-1, -1, -1), "color": (1.0, 1.0, 1.0), "intensity": 10.0},
                        {"type": "directional", "dir": (1, 0.5, -1), "color": (1.0, 1.0, 1.0), "intensity": 8.0},
                        {"type": "directional", "dir": (0, 1, -1), "color": (1.0, 1.0, 1.0), "intensity": 5.0},
                    ],
                ),
            },
            "visualization": {
                "viewer_type": "viser",
                "viser_port": 2026,
                "extra_hud_items": [link_pos_item, ]
            },
            "command": {
                "sampler": [
                    CommandTermConfig(cf.lin_vel_x, params={"range": (-1.0, 1.5)}),
                    CommandTermConfig(cf.lin_vel_y, params={"range": (-0.5, 0.5)}),
                    CommandTermConfig(cf.ang_vel, params={"range": (-0.5, 0.5)})
                ],
                "rel_standing_envs": 0.3,
            }
        },
    )
    evaluator.evaluate()
