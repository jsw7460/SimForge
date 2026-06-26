"""Multi-entity rigid-object (RigidObjectCfg) read-path smoke / verification.

Verifies the multi-entity read path on a chosen backend end-to-end: declare a
passive rigid object (a free-floating cube) in ``scene.rigid_objects``
alongside the Go2 robot, build the env, and read the cube's root pose/velocity
via ``env.get_rigid_object_data("cube")`` (the RigidObjectData root-only API),
while confirming the robot path (``get_robot_data("robot")``) is unaffected.

Run (GPU box):
    jaxpy rlworld/scripts/diag/rigid_object_smoke.py --sim genesis
    jaxpy rlworld/scripts/diag/rigid_object_smoke.py --sim newton
    jaxpy rlworld/scripts/diag/rigid_object_smoke.py --sim mujoco
"""

from __future__ import annotations

import argparse
import tempfile

import torch

from rlworld.rl.configs.events import EventTermConfig
from rlworld.rl.configs.presets.go2.base import Go2FlatConfig
from rlworld.rl.configs.scene import RigidObjectCfg
from rlworld.rl.configs.scene.entity_selector import SceneEntitySelector
from rlworld.rl.configs.scene.unified_entity_config import InitialStateCfg
from rlworld.rl.envs.mdp.events import common as common_ef
from rlworld.rl.runners import BaseRunner

CUBE_SPAWN = (0.3, 0.0, 0.6)

CUBE_URDF = """<?xml version="1.0"?>
<robot name="cube">
  <link name="cube">
    <inertial>
      <origin xyz="0 0 0"/>
      <mass value="0.2"/>
      <inertia ixx="1e-4" ixy="0" ixz="0" iyy="1e-4" iyz="0" izz="1e-4"/>
    </inertial>
    <visual><origin xyz="0 0 0"/><geometry><box size="0.06 0.06 0.06"/></geometry></visual>
    <collision><origin xyz="0 0 0"/><geometry><box size="0.06 0.06 0.06"/></geometry></collision>
  </link>
</robot>
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", choices=["genesis", "newton", "mujoco"], default="genesis")
    args = ap.parse_args()

    with tempfile.NamedTemporaryFile("w", suffix=".urdf", delete=False) as f:
        f.write(CUBE_URDF)
        cube_urdf = f.name

    cfg = Go2FlatConfig(sim_type=args.sim, num_envs=1)
    cfgs = cfg.build()
    # Inject one passive rigid object into the (separate) rigid_objects registry.
    cfgs.scene.rigid_objects = {
        "cube": RigidObjectCfg(
            urdf_path=cube_urdf,
            floating=True,
            init_state=InitialStateCfg(pos=CUBE_SPAWN),
        )
    }

    # Place the cube at reset via the SAME event the robot uses
    # (reset_root_state_uniform), targeting it through a SceneEntitySelector.
    # No perturbation → it should land exactly at CUBE_SPAWN on every backend
    # (verifies the rigid-object state writer + polymorphic root-writer lookup).
    cfgs.event.reset_cube = EventTermConfig(
        func=common_ef.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {},
            "velocity_range": {},
            "default_pos": CUBE_SPAWN,
            "default_quat_wxyz": (1.0, 0.0, 0.0, 0.0),
            "asset_cfg": SceneEntitySelector(name="cube"),
        },
    )

    runner = BaseRunner.create_with_env(cfgs)
    env = runner.env
    env.reset()

    print("=" * 70)
    print(f"RIGID-OBJECT READ SMOKE  [sim={args.sim}]")
    print("=" * 70)

    results: dict[str, bool] = {}

    # 1) Robot path still works (regression check for the multi-entity change).
    try:
        rd = env.get_robot_data("robot")
        rpos = rd.root_link_pos_w[0].tolist()
        rjoints = tuple(rd.joint_pos.shape)
        print(f"[robot]  root_link_pos_w = {rpos}  joint_pos.shape = {rjoints}")
        results["robot_read"] = len(rpos) == 3 and rd.joint_pos.shape[-1] > 0
    except Exception as e:  # noqa: BLE001
        print(f"[robot]  ERROR: {type(e).__name__}: {e}")
        results["robot_read"] = False

    # 2) Rigid object read via get_rigid_object_data — the new path.
    try:
        od = env.get_rigid_object_data("cube")
        opos = od.root_link_pos_w[0].tolist()
        oquat = od.root_link_quat_w[0].tolist()
        ovel = od.root_link_lin_vel_w[0].tolist()
        print(f"[cube]   root_link_pos_w  = {opos}")
        print(f"[cube]   root_link_quat_w = {oquat}")
        print(f"[cube]   root_link_lin_vel_w = {ovel}")
        finite = all(map(_isfinite, opos + oquat + ovel))
        results["object_read"] = len(opos) == 3 and len(oquat) == 4 and len(ovel) == 3 and finite
    except Exception as e:  # noqa: BLE001
        print(f"[cube]   ERROR: {type(e).__name__}: {e}")
        results["object_read"] = False

    # 2b) Object placed at reset by the event (root writer path). On every
    #     backend the cube must sit at CUBE_SPAWN right after reset — this is
    #     what was broken on mjlab before the rigid-object writer existed
    #     (it spawned at the origin because reset_data ignores the keyframe).
    try:
        od = env.get_rigid_object_data("cube")
        opos = od.root_link_pos_w[0].tolist()
        err = max(abs(opos[i] - CUBE_SPAWN[i]) for i in range(3))
        print(f"[cube]   post-reset pos = {opos}  target = {list(CUBE_SPAWN)}  max_err = {err:.4f}")
        results["object_placed_by_event"] = err < 1e-3
    except Exception as e:  # noqa: BLE001
        print(f"[cube]   PLACEMENT ERROR: {type(e).__name__}: {e}")
        results["object_placed_by_event"] = False

    # 3) Object has NO joint API (it's a RigidObjectData, not RobotData).
    has_joint_attr = hasattr(env.get_rigid_object_data("cube"), "joint_pos")
    print(f"[cube]   exposes joint_pos? {has_joint_attr} (expect False — RigidObjectData)")
    results["object_is_rigid_only"] = not has_joint_attr

    # 4) Dynamics dump (informational): step and watch the free cube move.
    print("-" * 70)
    zeros = torch.zeros((env.num_envs, env.num_actions), device=env.device)
    z0 = float(env.get_rigid_object_data("cube").root_link_pos_w[0, 2])
    for s in range(20):
        env.step(zeros)
        if s % 5 == 0 or s == 19:
            z = float(env.get_rigid_object_data("cube").root_link_pos_w[0, 2])
            print(f"  step {s:3d}  cube_z = {z:.4f}")
    z1 = float(env.get_rigid_object_data("cube").root_link_pos_w[0, 2])
    print(f"  cube z: {z0:.4f} -> {z1:.4f}  (moved={abs(z1 - z0) > 1e-4})")

    print("=" * 70)
    print("VERDICT")
    for k, v in results.items():
        print(f"  {k:22s}: {'PASS' if v else 'FAIL'}")
    ok = all(results.values())
    print(f"  OVERALL               : {'PASS' if ok else 'FAIL'}")
    print("=" * 70)
    return 0 if ok else 1


def _isfinite(x: float) -> bool:
    return x == x and x not in (float("inf"), float("-inf"))


if __name__ == "__main__":
    raise SystemExit(main())
