import os

os.environ['__NV_PRIME_RENDER_OFFLOAD'] = '1'
os.environ['__GLX_VENDOR_LIBRARY_NAME'] = 'nvidia'

import genesis as gs
from rlworld.rl.evals import PolicyEvaluator
from rlworld.rl.configs.scene import EntityConfig
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
        # wandb_run_path="jsw7460/RLArchitecture/p79hdfkt",
        policy_path=f"outputs/models/2026-03-04/06-36-44/checkpoint_latest/",  # MLP
        # policy_path=f"/home/sangwoo/workspace/model_zoo/dynann/g1_29dof/mlp/checkpoint_latest/",  # MLP
        num_evals=1,
        seed=42,
        show_viewer=False,
        record_video=True,
        record_steps=None,
        video_dir=None,
        extra_overrides={
            "env": {
                "num_envs": 1,
                # "episode_length_s": 5.0,
                "seed": 42,
                # "env_name": "Maniskill",
                # "episode_length_s": 1.0,
                # "gym_make_kwargs": {
                #     "obs_mode": "state",
                #     "render_mode": "rgb_array",
                #     "sim_backend": "physx_cuda",
                #     "reward_mode": "sparse"
                # }
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
            #     "entities": [
            #         EntityConfig(
            #             entity_name="base_entity",
            #             morph=gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True),
            #         ),
            #         EntityConfig(
            #             entity_name="robot",
            #             morph=gs.morphs.URDF(
            #                 file=g1_29dof.urdf_path,
            #                 links_to_keep=[g1_29dof.base_link_name],
            #                 convexify=True,
            #             ),
            #             visualize_contact=True,
            #             p_gain=g1_29dof.p_gains,
            #             d_gain=g1_29dof.d_gains,
            #             armature=g1_29dof.armature,
            #         )
            #     ]
            },
            "visualization": {
                "extra_hud_items": [link_pos_item, ]
            },
            "command": {
                "sampler": [
                    CommandTermConfig(cf.lin_vel_x, params={"range": (-1.0, 1.0)}),
                    CommandTermConfig(cf.lin_vel_y, params={"range": (-0.3, 0.3)}),
                    CommandTermConfig(cf.ang_vel, params={"range": (-0.0, 0.0)})
                ]
            }
        },
    )
    evaluator.evaluate()
