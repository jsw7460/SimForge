"""What a Genesis state or control write costs by the form of its indices.

``env_step_breakdown --sync-debug`` counts 52 host synchronisations per
control step on the Genesis G1 flat preset, almost all raised inside
Genesis setters the managers call with index TENSORS: the per-substep
``control_dofs_position`` (two syncs each), the reset's
``set_dofs_position`` / ``set_dofs_velocity`` / ``set_pos`` / ``set_quat``
(five for the position write alone) and the push's ``set_dofs_velocity``.
Genesis documents the alternative itself (``genesis/utils/misc.py``,
``indices_to_mask``): a ``range`` of DOFs becomes a slice and takes the
zero-copy path, an index tensor takes advanced indexing, and a boolean env
mask is written on the device while an index tensor of env ids is
converted and bounds-handled on the host.

This times the exact writes the Genesis managers make, on a bare G1 scene
at ``--num-envs``, in every index form Genesis accepts, and counts the host
syncs each form raises. It is the measurement behind changing how the
managers address Genesis; nothing here depends on JaxRLWorld.

    jaxpy -m jaxrlworld.scripts.diag.perf.genesis_setter_index_bench --num-envs 4096
"""

from __future__ import annotations

import argparse
import time
import warnings
from collections.abc import Callable
from pathlib import Path

import genesis as gs
import torch

_G1_XML = Path(__file__).resolve().parents[3] / "assets" / "g1" / "g1.xml"
_DT = 0.005


def build_scene(num_envs: int):
    gs.init(backend=gs.gpu, logging_level="warning", performance_mode=True)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=_DT, substeps=1),
        rigid_options=gs.options.RigidOptions(
            dt=_DT,
            integrator=gs.integrator.implicitfast,
            constraint_solver=gs.constraint_solver.Newton,
            iterations=20,
            ls_iterations=50,
            tolerance=1e-5,
            constraint_timeconst=0.02,
            enable_collision=True,
            enable_self_collision=True,
            enable_joint_limit=True,
            max_collision_pairs=100,
            batch_dofs_info=True,
            batch_links_info=True,
        ),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(gs.morphs.MJCF(file=str(_G1_XML), pos=(0.0, 0.0, 0.8)))
    scene.build(n_envs=num_envs)
    return scene, robot


def measure(label: str, fn: Callable[[], None], calls: int, warmup: int = 5) -> None:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    syncs = 0

    def count(message, category, filename, lineno, file=None, line=None) -> None:
        nonlocal syncs
        syncs += 1

    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.showwarning = count
        torch.cuda.set_sync_debug_mode("warn")
        try:
            fn()
        finally:
            torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(calls):
        fn()
    torch.cuda.synchronize()
    us = (time.perf_counter() - t0) / calls * 1e6
    print(f"{us:10.1f} us/call  {syncs:2d} syncs/call  {label}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--reset-envs", type=int, default=8, help="envs written per reset-style call")
    ap.add_argument("--calls", type=int, default=200)
    args = ap.parse_args()

    scene, robot = build_scene(args.num_envs)
    solver = scene.rigid_solver
    dev = gs.device
    n_envs, n_dofs = args.num_envs, robot.n_dofs
    joint_dofs = list(range(6, n_dofs))
    joint_dofs_tensor = torch.tensor(joint_dofs, device=dev)
    joint_dofs_range = range(6, n_dofs)
    root_dofs_tensor = torch.tensor(list(range(6)), device=dev)
    root_dofs_range = range(0, 6)
    ids = torch.arange(0, args.reset_envs, device=dev) * (n_envs // args.reset_envs)
    mask = torch.zeros(n_envs, dtype=torch.bool, device=dev)
    mask[ids] = True
    q_default = robot.get_dofs_position()
    print(f"[bench] genesis {gs.__version__}, n_envs {n_envs}, n_dofs {n_dofs}, reset envs {args.reset_envs}")

    targets = q_default[:, 6:].clone()
    joint_rows = q_default[ids, 6:].clone()
    zeros_rows = torch.zeros(args.reset_envs, len(joint_dofs), device=dev)
    zeros_rows_all = torch.zeros(n_envs, len(joint_dofs), device=dev)
    root_vel_rows = torch.zeros(args.reset_envs, 6, device=dev)
    root_vel_rows_all = torch.zeros(n_envs, 6, device=dev)
    pos_rows = robot.get_pos()[ids].clone()
    quat_rows = robot.get_quat()[ids].clone()
    pos_all = robot.get_pos().clone()
    quat_all = robot.get_quat().clone()
    qpos_root_all = torch.cat([pos_all, quat_all], dim=-1)

    print("== control (every substep, all envs)")
    measure(
        "control_dofs_position, dofs TENSOR (action manager today)",
        lambda: robot.control_dofs_position(targets, joint_dofs_tensor),
        args.calls,
    )
    measure(
        "control_dofs_position, dofs RANGE",
        lambda: robot.control_dofs_position(targets, joint_dofs_range),
        args.calls,
    )

    print("== reset joint state (writer.set_dof_state today = velocity skip_forward + position)")
    measure(
        "set_dofs_velocity, env ids TENSOR, dofs TENSOR",
        lambda: robot.set_dofs_velocity(zeros_rows, joint_dofs_tensor, ids, skip_forward=True),
        args.calls,
    )
    measure(
        "set_dofs_velocity, env ids TENSOR, dofs RANGE",
        lambda: robot.set_dofs_velocity(zeros_rows, joint_dofs_range, ids, skip_forward=True),
        args.calls,
    )
    measure(
        "set_dofs_velocity, env MASK (every row), dofs RANGE",
        lambda: robot.set_dofs_velocity(zeros_rows_all, joint_dofs_range, mask, skip_forward=True),
        args.calls,
    )
    measure(
        "set_dofs_position, env ids TENSOR, dofs TENSOR, zero_velocity=False",
        lambda: robot.set_dofs_position(joint_rows, joint_dofs_tensor, ids, zero_velocity=False),
        args.calls,
    )
    measure(
        "set_dofs_position, env ids TENSOR, dofs RANGE, zero_velocity=False",
        lambda: robot.set_dofs_position(joint_rows, joint_dofs_range, ids, zero_velocity=False),
        args.calls,
    )
    measure(
        "set_dofs_position, env MASK (selected rows), dofs RANGE, zero_velocity=False",
        lambda: robot.set_dofs_position(joint_rows, joint_dofs_range, mask, zero_velocity=False),
        args.calls,
    )

    print("== reset root pose (writer.set_root_pose today = set_pos skip_forward + set_quat)")
    measure(
        "set_pos(skip_forward) + set_quat, env ids TENSOR",
        lambda: (
            robot.set_pos(pos_rows, envs_idx=ids, zero_velocity=False, skip_forward=True),
            robot.set_quat(quat_rows, envs_idx=ids, zero_velocity=False),
        ),
        args.calls,
    )
    measure(
        "set_pos(skip_forward) + set_quat, env MASK (every row)",
        lambda: (
            robot.set_pos(pos_all, envs_idx=mask, zero_velocity=False, skip_forward=True),
            robot.set_quat(quat_all, envs_idx=mask, zero_velocity=False),
        ),
        args.calls,
    )
    measure(
        "solver.set_qpos root 7 q, qs RANGE, env MASK (every row)",
        lambda: solver.set_qpos(qpos_root_all, range(0, 7), mask),
        args.calls,
    )

    print("== push / root velocity (writer.set_root_velocity today = get_quat + set_dofs_velocity)")
    measure(
        "get_quat(env ids TENSOR) + set_dofs_velocity root, env ids TENSOR, dofs TENSOR",
        lambda: (robot.get_quat(envs_idx=ids), robot.set_dofs_velocity(root_vel_rows, root_dofs_tensor, ids)),
        args.calls,
    )
    measure(
        "get_quat(all) + set_dofs_velocity root, env MASK (every row), dofs RANGE",
        lambda: (robot.get_quat(), robot.set_dofs_velocity(root_vel_rows_all, root_dofs_range, mask)),
        args.calls,
    )

    print("== reads")
    measure(
        "get_contacts(as_tensor, to_torch)",
        lambda: solver.collider.get_contacts(as_tensor=True, to_torch=True),
        args.calls,
    )
    measure("get_links_pos()", lambda: robot.get_links_pos(), args.calls)
    measure("get_dofs_position(dofs TENSOR)", lambda: robot.get_dofs_position(joint_dofs_tensor), args.calls)
    measure("get_dofs_position(dofs RANGE)", lambda: robot.get_dofs_position(joint_dofs_range), args.calls)

    print("== physics for scale")
    measure("scene.step()", scene.step, args.calls // 4)


if __name__ == "__main__":
    main()
