"""Root-velocity write/read round trip — the CoM/origin reference-point guard.

WHY THIS EXISTS. ``set_root_velocity`` takes the LINK-ORIGIN velocity and
``root_link_lin_vel_w`` reads it back; between them, Newton stores the root
velocity CoM-referenced (its documented free-joint convention — FK kernel and
both mjwarp bridge kernels transfer with ``v_C = v_O + omega x (R @ c)``).
Commit ``0a95fb2`` unified the READ side across the three sims but left both
Newton WRITERS un-transferred, so for 3.5 months an injected ``(v, omega)``
with ``omega != 0`` read back as ``v - omega x (R @ c)`` on Newton only —
caught by the K1 cross-sim parity diag, fixed in the articulation writer and
then the rigid-object writer. This diag pins the whole contract so it cannot
silently regress on ANY backend:

  A. articulation round trip (K1 joystick robot), omega = 0 and != 0,
     upright and tilted poses, subset env_ids;
  B. rigid-object round trip on a synthetic prop whose inertial origin is
     DELIBERATELY offset (c != 0) — the only configuration where the
     transfer does anything;
  C. rigid-object round trip on the stock cube (c = 0) — proves the fix is
     an exact identity for every existing prop;
  D. CoM cross-check: after writing (v, omega), the independently derived
     ``root_com_lin_vel_w`` must equal ``v + omega x (R @ c)``;
  E. physics smoke on the offset prop: inject a spin whose ORIGIN velocity
     is ``-omega x (R @ c)`` so the CoM velocity is exactly zero — free
     rigid-body dynamics then keeps the CoM put in xy while the origin
     orbits it, and nothing NaNs. (Injecting v_origin = 0 instead is NOT a
     pinned-CoM state: the CoM starts at ``omega x c`` and coasts.)

WHAT EACH BACKEND RUN HAS CAUGHT SO FAR (this is a 3-sim CONTRACT guard,
not a Newton regression test): newton — the missing writer transfer
itself; mujoco — velocity reads are cvel-derived and stale without a
forward pass (a diag pitfall, now handled below); genesis — ``align``
resolved to True for basic rigid objects and silently REFRAMED the prop
so its link origin sat at the CoM, putting an off-CoM prop c apart from
the other backends (fixed by pinning ``align=False`` in the Genesis
scene manager's morph kwargs).

Run per backend (one process per sim)::

    python -m jaxrlworld.scripts.diag.engine.newton_root_velocity_roundtrip_diag --sim newton
    python -m jaxrlworld.scripts.diag.engine.newton_root_velocity_roundtrip_diag --sim mujoco
    python -m jaxrlworld.scripts.diag.engine.newton_root_velocity_roundtrip_diag --sim genesis
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import torch

from jaxrlworld.rl.runners.base_runner import BaseRunner
from jaxrlworld.rl.utils.quat_utils import quat_from_angle_axis_wxyz, quat_mul_wxyz, quat_rotate_wxyz

TOL = 1e-5
NUM_ENVS = 3

# (name, lin_vel, ang_vel) — the omega != 0 rows are the ones the old writer corrupted.
VELOCITY_CASES = [
    ("lin_only", (0.4, -0.2, 0.1), (0.0, 0.0, 0.0)),
    ("spin_only", (0.0, 0.0, 0.0), (0.8, -0.5, 1.2)),
    ("mixed", (0.3, 0.2, -0.1), (0.5, 0.4, -0.9)),
    ("zero", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
]
POSE_CASES = [
    ("upright", (0.0, 0.0, 0.0)),
    ("tilted", (0.3, -0.4, 1.1)),
]

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'OK' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def rpy_quat(rpy, device, n):
    r, p, y = rpy
    ones = torch.ones(n, device=device)
    ax = lambda v: torch.tensor(v, device=device)
    return quat_mul_wxyz(
        quat_mul_wxyz(
            quat_from_angle_axis_wxyz(y * ones, ax((0.0, 0.0, 1.0))),
            quat_from_angle_axis_wxyz(p * ones, ax((0.0, 1.0, 0.0))),
        ),
        quat_from_angle_axis_wxyz(r * ones, ax((1.0, 0.0, 0.0))),
    )


def roundtrip(env, entity: str, base_pos_z: float, com_expected: torch.Tensor | None) -> None:
    """Write pose+velocity, read back, and cross-check the CoM velocity."""
    device = env.device
    rd = env.get_entity_data(entity)
    writer = env.get_root_state_writer(entity)

    for pose_name, rpy in POSE_CASES:
        for env_ids in (torch.arange(env.num_envs, device=device), torch.tensor([1], device=device)):
            sel = "all" if len(env_ids) == env.num_envs else "subset"
            n = len(env_ids)
            pos = env.scene_manager.env_origins[env_ids] + torch.tensor([0.0, 0.0, base_pos_z], device=device)
            quat = rpy_quat(rpy, device, n)
            for vel_name, lin, ang in VELOCITY_CASES:
                lin_t = torch.tensor(lin, device=device).repeat(n, 1)
                ang_t = torch.tensor(ang, device=device).repeat(n, 1)
                writer.set_root_pose(pos, quat, env_ids=env_ids)
                writer.set_root_velocity(lin_t, ang_t, env_ids=env_ids)
                writer.eval_fk(env_ids=env_ids)
                # mjlab's velocity reads come off derived data (cvel), which
                # only refreshes on a forward pass; Newton/Genesis read the
                # written coordinates directly and are unaffected.
                env._post_reset_forward()
                env._invalidate_cache()

                got_lin = rd.root_link_lin_vel_w[env_ids]
                got_ang = rd.root_link_ang_vel_w[env_ids]
                dl = float((got_lin - lin_t).abs().max())
                da = float((got_ang - ang_t).abs().max())
                check(f"{entity}/{pose_name}/{sel}/{vel_name}: lin round trip", dl <= TOL, f"maxdiff {dl:.2e}")
                check(f"{entity}/{pose_name}/{sel}/{vel_name}: ang round trip", da <= TOL, f"maxdiff {da:.2e}")

                # Independent CoM cross-check: v_com = v_origin + omega x (R @ c).
                if com_expected is not None:
                    r_world = quat_rotate_wxyz(quat, com_expected.to(device).expand_as(lin_t))
                    want_com = lin_t + torch.cross(ang_t, r_world, dim=-1)
                    got_com = rd.root_com_lin_vel_w[env_ids]
                    dc = float((got_com - want_com).abs().max())
                    check(f"{entity}/{pose_name}/{sel}/{vel_name}: CoM cross-check", dc <= TOL, f"maxdiff {dc:.2e}")


def offset_cube_urdf(tmpdir: str, c=(0.08, -0.05, 0.03)) -> str:
    """A 5 cm cube whose inertial origin (CoM) is deliberately off the link
    origin — the configuration that exposes a missing reference-point
    transfer. Inertia is the plain cube inertia about the CoM (URDF inertia
    is about the inertial origin, so no parallel-axis term belongs here)."""
    path = Path(tmpdir) / "offset_com_cube.urdf"
    path.write_text(f"""<?xml version="1.0"?>
<robot name="offset_com_cube">
  <link name="cube">
    <inertial>
      <origin xyz="{c[0]} {c[1]} {c[2]}"/>
      <mass value="0.05"/>
      <inertia ixx="2.0833e-5" ixy="0" ixz="0" iyy="2.0833e-5" iyz="0" izz="2.0833e-5"/>
    </inertial>
    <collision>
      <origin xyz="0 0 0"/>
      <geometry><box size="0.05 0.05 0.05"/></geometry>
    </collision>
  </link>
</robot>
""")
    return str(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", choices=["newton", "mujoco", "genesis"], required=True)
    args = ap.parse_args()

    from jaxrlworld.rl.configs.presets.yam_arm.base import YamArmConfig

    tmpdir = tempfile.mkdtemp(prefix="root_vel_roundtrip_")
    offset_urdf = offset_cube_urdf(tmpdir)

    cfgs = YamArmConfig(sim_type=args.sim, num_envs=NUM_ENVS).build()
    # Second, deliberately-offset-CoM prop next to the stock cube.
    stock_cube = cfgs.scene.rigid_objects["cube"]
    offset_cube = type(stock_cube)(
        urdf_path=offset_urdf,
        floating=True,
        init_state=type(stock_cube.init_state)(pos=(0.45, 0.25, 0.85)),
    )
    cfgs.scene.rigid_objects["offset_cube"] = offset_cube
    env = BaseRunner._create_env_from_config(cfgs)
    env.reset()

    print(f"\n=== root-velocity round trip ({args.sim}) ===")

    # A. articulation writer — the robot. (YAM is bench-mounted/fixed-base,
    # so the articulation check runs only when the robot is floating; the
    # K1 cross-sim parity diag covers the floating-robot case end to end.)
    if env.get_entity_data("robot").is_fixed_base:
        print("  [--] robot is fixed-base here; articulation round trip covered by the K1 parity diag")
    else:
        roundtrip(env, "robot", 0.6, com_expected=None)

    # B. rigid object with c != 0 — the exposing configuration.
    c = torch.tensor((0.08, -0.05, 0.03))
    roundtrip(env, "offset_cube", 0.9, com_expected=c)

    # C. stock cube, c = 0: transfer must be an exact identity.
    roundtrip(env, "cube", 0.9, com_expected=torch.zeros(3))

    # E. physics smoke: pure spin on the offset prop -> CoM holds in xy
    # while the origin orbits it; nothing NaNs.
    device = env.device
    rd = env.get_entity_data("offset_cube")
    writer = env.get_root_state_writer("offset_cube")
    ids = torch.arange(env.num_envs, device=device)
    pos = env.scene_manager.env_origins + torch.tensor([0.0, 0.0, 1.5], device=device)
    quat = rpy_quat((0.0, 0.0, 0.0), device, env.num_envs)
    omega = torch.tensor([0.0, 0.0, 6.0], device=device).repeat(env.num_envs, 1)
    # v_com = v_origin + omega x (R @ c) == 0  =>  v_origin = -omega x (R @ c).
    r_world = quat_rotate_wxyz(quat, c.to(device).expand_as(omega))
    v_origin = -torch.cross(omega, r_world, dim=-1)
    writer.set_root_pose(pos, quat, env_ids=ids)
    writer.set_root_velocity(v_origin, omega, env_ids=ids)
    writer.eval_fk(env_ids=ids)
    env._post_reset_forward()
    env._invalidate_cache()
    com0 = rd.root_com_pos_w.clone()
    origin0 = rd.root_link_pos_w.clone()
    env._step_physics()
    env._invalidate_cache()
    com_drift = float((rd.root_com_pos_w - com0)[:, :2].abs().max())
    origin_move = float((rd.root_link_pos_w - origin0)[:, :2].abs().max())
    finite = bool(torch.isfinite(rd.root_link_pos_w).all() and torch.isfinite(rd.root_link_lin_vel_w).all())
    check("spin physics: finite state", finite)
    check(
        "spin physics: CoM pinned, origin orbits",
        origin_move > 5.0 * max(com_drift, 1e-9) and origin_move > 1e-4,
        f"origin xy moved {origin_move:.2e} m vs CoM drift {com_drift:.2e} m over one control step",
    )

    print("\n=== RESULT:", "ALL OK" if not FAILS else f"{len(FAILS)} FAIL: {FAILS[:6]}", "===")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
