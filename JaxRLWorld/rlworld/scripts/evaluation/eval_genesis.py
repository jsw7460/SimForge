import argparse

import genesis as gs
from rlworld.rl.configs.robots.g1_29dof import G1MujocoConfig
from rlworld.rl.evals import PolicyEvaluator
from rlworld.rl.vis.overlays.hud_items import LinkPositionItem, LinkPositionItemConfig

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Genesis evaluation")
    parser.add_argument("--eval", action="store_true", help="Run batch evaluation instead of interactive viewer")
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--port", type=int, default=2026, help="Viser viewer port")
    args = parser.parse_args()

    g1_29dof = G1MujocoConfig()
    link_pos_item = LinkPositionItem(
        LinkPositionItemConfig(link_patterns=("left_ankle_roll_link", "right_ankle_roll_link")),
    )

    overrides = {
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
    }

    if args.eval:
        overrides["visualization"] = {
            "viewer_type": "viser",
            "viser_port": args.port,
            "extra_hud_items": [link_pos_item],
        }

    evaluator = PolicyEvaluator(
        policy_path="outputs/models/2026-04-01/17-55-54/checkpoint_latest/",
        num_evals=1,
        seed=42,
        record_video=args.record_video,
        record_steps=None,
        video_dir=None,
        extra_overrides=overrides,
    )

    if args.eval:
        evaluator.evaluate()
    else:
        evaluator.play(port=args.port)
