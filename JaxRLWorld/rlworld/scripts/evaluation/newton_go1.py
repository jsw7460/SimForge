from rlworld.rl.evals import PolicyEvaluator
from rlworld.rl.envs.mdp.configs.commands.command_term_config import CommandTermConfig
from rlworld.rl.envs.mdp.commands import command_terms as cf

import newton
from rlworld.rl.configs.scene import NewtonEntityConfig
import warp as wp

from rlworld.rl.configs.robots.go1 import Go1Config

robot = Go1Config()

if __name__ == '__main__':
    quat = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi * 0.5)
    # entities = [
    #     NewtonEntityConfig(
    #         entity_name="ground",
    #         entity_type="ground_plane",
    #         shape_cfg=newton.ModelBuilder.ShapeConfig(
    #             kf=1.0e3,
    #             mu=0.75,
    #         ),
    #         floating=False
    #     ),
    #     NewtonEntityConfig(
    #         entity_name="robot",
    #         urdf_path="./rlworld/assets/go1_model_clean/urdf/go1_simplified_stl.urdf",
    #         transform=wp.transform(
    #             wp.vec3(0.0, 0.0, robot.base_init_height),
    #             quat
    #         ),
    #         floating=True,
    #         joint_cfg=newton.ModelBuilder.JointDofConfig(
    #             armature=0.1,
    #             target_ke=20.0,
    #             target_kd=0.5
    #         ),
    #         shape_cfg=newton.ModelBuilder.ShapeConfig(
    #             kf=1.0e3,
    #             mu=0.75,
    #         ),
    #         joint_target_ke_map=robot.p_gains,
    #         joint_target_kd_map=robot.d_gains,
    #         joint_armature_map=robot.armature,
    #         sites={"imu_site_base": "base"},
    #     ),
    # ]

    evaluator = PolicyEvaluator(
        eval_env_cfgs=None,
        policy_path=f"./outputs/models/2026-01-30/19-52-51/checkpoint_latest/",  # Newton
        num_evals=1,
        seed=42,
        show_viewer=False,
        record_video=True,
        record_steps=None,
        video_dir=None,
        extra_overrides={
            "env": {
                "num_envs": 1,
                # "episode_length_s": 10.0,
                # "termination_criteria": [],
            },
            "visualization": {
                # "viser_share": True,
                # "viser_port": 8081,
                "viewer_type": "viser",
            },
            "scene": {
                "robot_cfg": robot
            },
            # "command": {
                # "sampler": [
                #     CommandTermConfig(cf.lin_vel_x, params={"range": (0.5, 0.5)}),
                #     CommandTermConfig(cf.lin_vel_y, params={"range": (0.0, 0.0)}),
                #     CommandTermConfig(cf.ang_vel, params={"range": (-0.0, 0.0)}),
                #     CommandTermConfig(cf.base_height, params={"range": (0.34, 0.34)})
                # ]
                # "resampling_time_s": (5.0, 5.0),
                # "sampler": [
                #     CommandTermConfig(cf.lin_vel_x, params={"range": (-1.0, 1.0)}),
                #     CommandTermConfig(cf.lin_vel_y, params={"range": (-1.0, 1.0)}),
                #     CommandTermConfig(cf.ang_vel, params={"range": (-0.0, 0.0)}),
                #     CommandTermConfig(cf.base_height, params={"range": (0.34, 0.34)})
                # ]
            # }
        },
    )
    evaluator.evaluate()
