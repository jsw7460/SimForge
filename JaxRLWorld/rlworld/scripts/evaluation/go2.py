import os

from rlworld.rl.envs.mdp.configs import CommandTermConfig

os.environ['__NV_PRIME_RENDER_OFFLOAD'] = '1'
os.environ['__GLX_VENDOR_LIBRARY_NAME'] = 'nvidia'

import genesis as gs
from rlworld.rl.evals import PolicyEvaluator
from rlworld.rl.configs.scene import EntityConfig
from rlworld.rl.envs.mdp.commands import command_terms as cf

if __name__ == '__main__':
    evaluator = PolicyEvaluator(
        eval_env_cfgs=None,
        # policy_path=f"./outputs/models/2026-02-26/12-58-51/checkpoint_latest/",
        wandb_run_path="jsw7460/RLArchitecture/m24kgrku",
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
                # "entities": [
                #     EntityConfig(
                #         entity_name="base_entity",
                #         morph=gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True),
                #     ),
                #     EntityConfig(
                #         entity_name="robot",
                #         morph=gs.morphs.URDF(
                #             # file="./rlworld/assets/go1_description/urdf/go1.urdf",
                #             file="./rlworld/assets/go1_model_clean/urdf/go1_simplified_stl.urdf",
                #             convexify=False,
                #             links_to_keep=("FR_foot", "FL_foot", "RR_foot", "RL_foot")
                #         ),
                #         visualize_contact=True,
                #         surface=gs.surfaces.Metal(color=(0.4, 0.4, 0.45)),
                #         p_gain={"FL.*": 20.0, "FR.*": 20.0, "RL.*": 20.0, "RR.*": 20.0},
                #         d_gain={"FL.*": 0.5, "FR.*": 0.5, "RL.*": 0.5, "RR.*": 0.5},
                #     )
                # ]
            },
            # "command": {
            #     "resampling_time_s": (5.0, 5.0),
            #     "sampler": [
            #         CommandTermConfig(cf.lin_vel_x, params={"range": (-1.0, 1.0)}),
            #         CommandTermConfig(cf.lin_vel_y, params={"range": (-1.0, 1.0)}),
            #         CommandTermConfig(cf.ang_vel, params={"range": (-0.0, 0.0)}),
            #         CommandTermConfig(cf.base_height, params={"range": (0.34, 0.34)})
            #     ]
            # }
        },
    )
    evaluator.evaluate()
