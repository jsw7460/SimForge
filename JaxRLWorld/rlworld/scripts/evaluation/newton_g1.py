from rlworld.rl.evals import PolicyEvaluator
from rlworld.rl.envs.mdp.configs.commands.command_term_config import CommandTermConfig
from rlworld.rl.envs.mdp.commands import command_terms as cf

import newton
from rlworld.rl.configs.scene import NewtonEntityConfig
from rlworld.rl.configs.robots.g1_29dof import G1MjlabConfig
import warp as wp

if __name__ == '__main__':
    quat = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi * 0.5)

    g1_29dof = G1MjlabConfig()

    entities = [
        NewtonEntityConfig(
            entity_name="ground",
            entity_type="ground_plane",
            shape_cfg=newton.ModelBuilder.ShapeConfig(
                ke=2.0e3,
                kd=1.0e2,
                kf=1.0e3,
                mu=1.0,
            ),
            floating=False
        ),
        NewtonEntityConfig(
            entity_name="robot",
            entity_type="urdf",
            body_label_prefix=g1_29dof.name,
            urdf_path=g1_29dof.urdf_path,
            transform=wp.transform(
                wp.vec3(0.0, 0.0, g1_29dof.base_init_height),
                quat
            ),
            floating=True,
            joint_cfg=newton.ModelBuilder.JointDofConfig(
                armature=0.1,
                target_ke=400.0,
                target_kd=5.0
            ),
            shape_cfg=newton.ModelBuilder.ShapeConfig(
                ke=2.0e3,
                kd=1.0e2,
                kf=1.0e3,
                mu=1.0,
            ),
            joint_target_ke_map=g1_29dof.prefixed_p_gains,
            joint_target_kd_map=g1_29dof.prefixed_d_gains,
            joint_armature_map=g1_29dof.prefixed_armature,
            sites={"imu_site_base": g1_29dof.base_link_name},
            enable_self_collisions=False
        ),
    ]
    evaluator = PolicyEvaluator(
        eval_env_cfgs=None,
        policy_path=f"./outputs/models/2026-02-23/22-54-09/checkpoint_latest/",  # Newton
        num_evals=1,
        seed=42,
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
                "viser_port": 8081,
                "viewer_type": "viser",
            },
            # "scene": {
            #     "entities": entities,
            #     "robot_cfg": g1_29dof
            # },
            # "command": {
            #     "sampler": [
            #         CommandTermConfig(cf.lin_vel_x, params={"range": (0.7, 0.7)}),
            #         CommandTermConfig(cf.lin_vel_y, params={"range": (0.0, 0.0)}),
            #         CommandTermConfig(cf.ang_vel, params={"range": (-0.0, 0.0)}),
            #     ]
            #     # ]
            # }
        },
    )
    evaluator.evaluate()
