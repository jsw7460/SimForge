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
        policy_path=f"./outputs/models/2026-03-06/15-38-52/checkpoint_latest/",  # Newton
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
            # "event": {
            #     "event_terms": [
            #         # Reset events
            #         EventTermConfig(
            #             func=ef.reset_root_state_uniform,
            #             mode="reset",
            #             params={
            #                 "pose_range": {
            #                     "x": (-0.5, 0.5),
            #                     "y": (-0.5, 0.5),
            #                     "z": (0.01, 0.05),
            #                     "yaw": (-3.14, 3.14),
            #                 },
            #                 "velocity_range": {},
            #             },
            #         ),
            #         EventTermConfig(
            #             func=ef.reset_joints_by_offset,
            #             mode="reset",
            #             params={
            #                 "position_range": (0.0, 0.0),
            #                 "velocity_range": (0.0, 0.0),
            #                 "entity_cfg": EntityCfg(name="robot", joint_names=(".*",)),
            #             },
            #         ),
            #
            #         # Startup events (domain randomization)
            #         EventTermConfig(
            #             func=ef.randomize_geom_friction,
            #             mode="startup",
            #             params={
            #                 "ranges": (0.3, 1.2),
            #                 "operation": "abs",
            #                 "shared_random": True,
            #                 "entity_cfg": EntityCfg(
            #                     name="robot",
            #                     geom_names=("FR_foot_collision", "FL_foot_collision",
            #                                 "RR_foot_collision", "RL_foot_collision"),
            #                 ),
            #             },
            #         ),
            #         EventTermConfig(
            #             func=ef.randomize_encoder_bias,
            #             mode="startup",
            #             params={
            #                 "bias_range": (-0.015, 0.015),
            #                 "entity_cfg": EntityCfg(name="robot"),
            #             },
            #         ),
            #         EventTermConfig(
            #             func=ef.randomize_body_com_offset,
            #             mode="startup",
            #             params={
            #                 "ranges": {
            #                     0: (-0.025, 0.025),
            #                     1: (-0.025, 0.025),
            #                     2: (-0.03, 0.03),
            #                 },
            #                 "operation": "add",
            #                 "entity_cfg": EntityCfg(name="robot", body_names=("trunk",)),
            #             },
            #         ),
            #     ]
            # },
            "command": {
                "rel_standing_envs": 1.0
            }
        },
    )
    evaluator.evaluate()
