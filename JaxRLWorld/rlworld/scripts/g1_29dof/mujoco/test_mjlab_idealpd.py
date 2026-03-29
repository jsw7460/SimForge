"""Test G1 MuJoCo with mjlab's native IdealPdActuator.

This bypasses JaxRLWorld's actuator system entirely and uses mjlab's
own IdealPdActuatorCfg to verify whether motor actuator + G1 armature
is stable in MuJoCo.

If this works → our torque delivery pipeline has an issue.
If this also NaN → MuJoCo + motor actuator + G1 armature is fundamentally unstable.
"""
import os

os.environ['__NV_PRIME_RENDER_OFFLOAD'] = '1'
os.environ['__GLX_VENDOR_LIBRARY_NAME'] = 'nvidia'

custom_assets = os.path.abspath(os.path.join(os.path.dirname(__file__), 'assets'))
import genesis.utils.terrain
genesis.utils.misc.get_assets_dir = lambda: custom_assets
genesis.utils.terrain.get_assets_dir = lambda: custom_assets

from rlworld.rl.runners import BaseRunner
from rlworld.rl.configs.presets.g1_29dof.mujoco.mlp import get_config
from rlworld.rl.actuators import ImplicitActuatorCfg


def main():
    cfgs_for_run = get_config().with_cli_overrides()

    # Override: use mjlab's native IdealPdActuator instead of BuiltinPositionActuator.
    # This creates motor actuators but mjlab computes PD torques in write_data_to_sim().
    from mjlab.actuator.pd_actuator import IdealPdActuatorCfg as MjlabIdealPdCfg
    from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_spec as g1_get_spec, FULL_COLLISION
    from mjlab.entity import EntityCfg as MjlabEntityCfg, EntityArticulationInfoCfg
    from mjlab.scene import SceneCfg
    from mjlab.terrains import TerrainEntityCfg
    from rlworld.rl.configs.robots.g1_29dof import G1MjlabConfig

    robot = G1MjlabConfig()

    # Build mjlab actuators from robot gains — one per gain group
    mjlab_actuators = []
    for pattern, kp in robot.p_gains.items():
        kd = robot.d_gains.get(pattern, 0.0)
        arm = robot.armature.get(pattern, 0.0)
        mjlab_actuators.append(MjlabIdealPdCfg(
            target_names_expr=(pattern,),
            stiffness=kp,
            damping=kd,
            armature=arm,
        ))

    mjlab_entity = MjlabEntityCfg(
        init_state=MjlabEntityCfg.InitialStateCfg(
            pos=(0, 0, robot.base_init_height),
            joint_pos=robot.default_joint_angles,
        ),
        spec_fn=g1_get_spec,
        articulation=EntityArticulationInfoCfg(
            actuators=tuple(mjlab_actuators),
        ),
        collisions=(FULL_COLLISION,),
        sort_actuators=True,
    )

    # Replace the scene entities with our mjlab-native entity
    cfgs_for_run.scene.mjlab_scene_cfg = SceneCfg(
        num_envs=cfgs_for_run.scene.num_envs,
        env_spacing=2.0,
        terrain=TerrainEntityCfg(terrain_type="plane"),
        entities={"robot": mjlab_entity},
        sensors=cfgs_for_run.scene.sensors,
    )
    # Disable unified entities so scene manager uses mjlab_scene_cfg directly
    cfgs_for_run.scene.entities = None

    runner = BaseRunner.create_with_env(cfgs_for_run)
    runner.learn(
        num_learning_iterations=cfgs_for_run.runner.max_iterations,
        init_at_random_ep_len=cfgs_for_run.runner.init_at_random_ep_len,
    )


if __name__ == "__main__":
    main()