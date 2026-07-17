"""G1 step-time benchmark: JaxRLWorld env.step vs raw simulator stepping.

Answers ONE question: is the Genesis G1 slowdown caused by the JaxRLWorld
wrapper (managers/obs/reward/event machinery) or by the simulator itself?

12 cells = 3 sims x 2 modes x 2 env counts:

    mode "jaxrlworld": full G1FlatConfig env (the thing training runs);
        zero-action env.step() timed.
    mode "raw":        the simulator's OWN API only — no JaxRLWorld import
        anywhere in the cell.  Same G1 MJCF asset, same physics timing
        (dt=0.005, substeps=1; one measured "control step" = 4 physics
        steps = 0.02 s, matching the presets' decimation), solver options
        mirrored from the g1_29dof preset builders (documented inline).
        Passive dynamics (no control) — the robot free-falls and settles
        on the ground plane, giving a contact-rich steady state.

Genesis-only extra modes (state-dependent solver cost):
    mode "rawsens":    raw + the preset's native contact-sensor set —
        isolates the engine-side sensor cost.
    mode "rawact":     raw + ACTIVE load mirroring the wrapped cell —
        Genesis internal PD to the default pose with the preset's
        per-joint gains, plus a periodic full state reset.  The passive
        raw cell measures a collapsed, statically-resting pile, which the
        tolerance-early-exit Newton solver finishes far faster than the
        standing, PD-actuated scene the wrapped cell measures; this mode
        removes that asymmetry.
    mode "rawsensact": rawsens + the same active load — the raw cell
        whose physics workload matches the wrapped cell most closely;
        wrapped minus rawsensact ≈ true JaxRLWorld-side cost.

Fairness notes:
    * raw Newton replays a CUDA graph over the decimation loop — this is
      both what Newton's own example_robot_g1.py does and what the wrapped
      NewtonEnv does (scene_manager.capture()).
    * raw mjlab's Simulation captures step graphs internally.
    * raw Genesis is driven exactly like Genesis's own locomotion examples
      (scene.step() per physics step).
    * Timing excludes build + warmup (JIT/kernel compile, graph capture);
      those are reported separately.  Every timed segment is bracketed by
      a device sync.

Each cell runs in its own subprocess (one sim backend per process).
Report: ms per control step, env-steps/s, wrapped/raw overhead ratio per
sim, and raw-vs-raw engine ratios.

Usage (GPU box):
    python -m rlworld.scripts.diag.g1_step_benchmark
    python -m rlworld.scripts.diag.g1_step_benchmark --env-counts 1,4096 --steps 100
    python -m rlworld.scripts.diag.g1_step_benchmark --cells genesis:raw:4096
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_MODULE = "rlworld.scripts.diag.g1_step_benchmark"

# Timing shared by all three presets (g1_29dof/base.py _SIM_TIMINGS):
# physics dt 0.005 s, substeps 1, decimation 4 -> control step = 0.02 s.
_DT = 0.005
_DECIMATION = 4

_SIMS = ("genesis", "newton", "mujoco")
_MODES = ("jaxrlworld", "raw")
# Genesis-only diagnostic modes (rendered in the table when measured).
_GENESIS_EXTRA_MODES = (
    "rawsens",
    "rawact",
    "rawsensact",
    "rawsensactdyn",
    "rawfull",
    "rawfullspaced",
    "rawcontactlist",
    "wrappedbare",
    "wrappedbarenodr",
    "wrappedbisect",
)

# Active-load raw cells: control steps between full state resets (~1 s of
# sim time, approximating the wrapped cell's episodic resets).
_RESET_EVERY = 50

# G1 MJCF shared by every backend (rlworld/assets/g1/g1.xml — resolved from
# this file's location so the raw cells need no JaxRLWorld import).
_G1_XML = Path(__file__).resolve().parents[2] / "assets" / "g1" / "g1.xml"
_SPAWN_Z = 0.8


def _stage(msg: str) -> None:
    print(f"[STAGE] {msg}", flush=True)


# ── raw cells (no JaxRLWorld) ───────────────────────────────────────


def _probe_scene(scene, robot, tag: str, drive=None, start_step: int = 0) -> None:
    """Dump a scene fingerprint + measured contact counts.

    Steps 10 extra control steps OUTSIDE any timed window and samples the
    collider's per-env contact counter each step.  Used to diff the
    wrapped-built scene against the raw mirror: if entity/geom counts or
    live contact counts diverge, the bare-step time difference is physics
    workload, not overhead.
    """
    import numpy as np

    solver = scene.rigid_solver
    _stage(
        f"{tag} fingerprint: robot n_links={robot.n_links} n_geoms={robot.n_geoms} "
        f"n_dofs={robot.n_dofs}; scene n_geoms={solver.n_geoms} n_links={solver.n_links}"
    )
    samples = []
    for k in range(10):
        if drive is not None:
            drive(start_step + k)
        for _ in range(_DECIMATION):
            scene.step()
        samples.append(solver.collider._collider_state.n_contacts.to_numpy())
    arr = np.stack(samples)
    _stage(
        f"{tag} contacts/env: mean {arr.mean():.2f}  p99 {np.percentile(arr, 99):.0f}  "
        f"max {arr.max()}  (10 control steps, {arr.shape[1]} envs)"
    )


def _setup_active_pd(robot):
    """Drive a raw Genesis scene exactly like the wrapped zero-action cell.

    The wrapped benchmark steps with zero actions: every actuated joint is
    PD-servoed to the XML default pose while the robots stand (with
    episodic resets).  This reproduces that load with Genesis's native API
    only — the preset's per-joint gains are imported as PURE DATA from the
    robot config module (no JaxRLWorld runtime touches the cell):
    ``set_dofs_kp/kv`` + ``control_dofs_position`` to the default pose +
    a full state reset every ``_RESET_EVERY`` control steps.

    Returns ``drive(ctrl_step)`` to call once per control step.
    """
    import re

    from rlworld.rl.configs.robots.g1_29dof import G1MujocoConfig

    gains = G1MujocoConfig()
    dof_ids: list[int] = []
    kp: list[float] = []
    kv: list[float] = []
    for joint in robot.joints:
        if joint.n_dofs != 1:
            continue  # free root joint
        p = next((v for pat, v in gains.p_gains.items() if re.fullmatch(pat, joint.name)), None)
        d = next((v for pat, v in gains.d_gains.items() if re.fullmatch(pat, joint.name)), None)
        if p is None or d is None:
            continue  # not in the preset's actuated set (same regex tables)
        dof_ids.append(joint.dofs_idx_local[0])
        kp.append(p)
        kv.append(d)
    if not dof_ids:
        raise RuntimeError("active PD setup matched no joints — gain tables / joint names diverged")
    robot.set_dofs_kp(kp, dof_ids)
    robot.set_dofs_kv(kv, dof_ids)

    pos0 = robot.get_pos().clone()
    quat0 = robot.get_quat().clone()
    dofs0 = robot.get_dofs_position(dof_ids).clone()

    def drive(ctrl_step: int) -> None:
        if ctrl_step % _RESET_EVERY == 0:
            robot.set_pos(pos0, zero_velocity=True)
            robot.set_quat(quat0, zero_velocity=True)
            robot.set_dofs_position(dofs0, dof_ids, zero_velocity=True)
        robot.control_dofs_position(dofs0, dof_ids)

    return drive


def _mirror_dynamics(robot) -> None:
    """Mirror what ``GenesisSceneManager._configure_robot_dynamics`` applies
    for the g1 preset's explicit-PD actuator and the raw XML lacks:
    per-joint armature (preset tables, pure data import) and
    ``frictionloss=0.3`` on every actuated dof (the preset builder's
    hardcoded value).  Joint frictionloss adds a friction constraint per
    dof to every Newton solve — a physics-cost difference, not overhead.
    """
    import re

    from rlworld.rl.configs.robots.g1_29dof import G1MujocoConfig

    gains = G1MujocoConfig()
    dof_ids: list[int] = []
    arm: list[float] = []
    for joint in robot.joints:
        if joint.n_dofs != 1:
            continue
        a = next((v for pat, v in gains.armature.items() if re.fullmatch(pat, joint.name)), None)
        if a is None:
            continue
        dof_ids.append(joint.dofs_idx_local[0])
        arm.append(a)
    if not dof_ids:
        raise RuntimeError("dynamics mirror matched no joints — armature tables / joint names diverged")
    robot.set_dofs_armature(arm, dof_ids)
    robot.set_dofs_frictionloss([0.3] * len(dof_ids), dof_ids)


def raw_genesis(num_envs: int, steps: int, warmup: int, active: bool = False) -> dict:
    """Genesis native: mirrors Genesis's own locomotion-example driving and
    the g1_29dof genesis preset's SimOptions/RigidOptions."""
    import genesis as gs

    t_build0 = time.perf_counter()
    gs.init(backend=gs.gpu, logging_level="warning")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=_DT, substeps=1),
        # Mirrors presets/g1_29dof/_genesis_builders.py build_scene().
        rigid_options=gs.options.RigidOptions(
            dt=_DT,
            constraint_solver=gs.constraint_solver.Newton,
            iterations=10,
            ls_iterations=20,
            tolerance=1e-5,
            constraint_timeconst=0.02,
            enable_collision=True,
            enable_self_collision=True,
            enable_joint_limit=True,
            max_collision_pairs=100,
            batch_dofs_info=True,
        ),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(gs.morphs.MJCF(file=str(_G1_XML), pos=(0.0, 0.0, _SPAWN_Z)))
    scene.build(n_envs=num_envs)
    drive = _setup_active_pd(robot) if active else None

    def sync() -> None:
        # Device->host readback forces completion of all queued kernels.
        robot.get_dofs_position().contiguous().cpu()

    for k in range(warmup):
        if drive is not None:
            drive(k)
        for _ in range(_DECIMATION):
            scene.step()
    sync()
    build_s = time.perf_counter() - t_build0
    _stage(f"genesis raw{'act' if active else ''} built+warm (build+warmup {build_s:.1f}s)")

    t0 = time.perf_counter()
    for k in range(steps):
        if drive is not None:
            drive(warmup + k)
        for _ in range(_DECIMATION):
            scene.step()
    sync()
    elapsed = time.perf_counter() - t0
    return {"elapsed_s": elapsed, "build_warmup_s": build_s}


def raw_genesis_sensors(
    num_envs: int,
    steps: int,
    warmup: int,
    active: bool = False,
    mirror_dynamics: bool = False,
    with_imu: bool = False,
    spaced: bool = False,
) -> dict:
    """raw_genesis PLUS the g1 preset's two contact-sensor groups, built with
    Genesis's native API only (one gs.sensors.Contact + ContactForce pair per
    primary link — the FORMER GenesisContactSensor design, kept as the
    baseline the contact-list backend is measured against: 2 feet links vs
    ground + EVERY robot link for self-collision).

    Isolates the engine-side cost of the sensor configuration: if this cell
    jumps from raw_genesis's level toward the wrapped level, the wrapped
    slowdown lives in the sensors, not in the manager stack.
    """
    import genesis as gs

    t_build0 = time.perf_counter()
    gs.init(backend=gs.gpu, logging_level="warning")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=_DT, substeps=1),
        rigid_options=gs.options.RigidOptions(
            dt=_DT,
            constraint_solver=gs.constraint_solver.Newton,
            iterations=10,
            ls_iterations=20,
            tolerance=1e-5,
            constraint_timeconst=0.02,
            enable_collision=True,
            enable_self_collision=True,
            enable_joint_limit=True,
            max_collision_pairs=100,
            batch_dofs_info=True,
        ),
        show_viewer=False,
    )
    ground = scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(gs.morphs.MJCF(file=str(_G1_XML), pos=(0.0, 0.0, _SPAWN_Z)))

    n_links = len(robot.links)
    feet_local = [l.idx_local for l in robot.links if "ankle_roll" in l.name]
    hist = _DECIMATION

    # filter_link_idx is a GLOBAL-link-index blacklist (matches
    # GenesisContactSensor, which builds it from link_start/link_end).
    # Group 1 — feet vs ground (secondary: terrain -> blacklist = all robot links).
    feet_filter = tuple(sorted(l.idx for l in robot.links))
    # Group 2 — self collision (secondary: self -> blacklist = ground links).
    self_filter = tuple(sorted(l.idx for l in ground.links))

    n_sensors = 0
    for l in feet_local:
        scene.add_sensor(
            gs.sensors.Contact(entity_idx=robot.idx, link_idx_local=l, filter_link_idx=feet_filter, history_length=hist)
        )
        scene.add_sensor(
            gs.sensors.ContactForce(
                entity_idx=robot.idx, link_idx_local=l, filter_link_idx=feet_filter, history_length=hist
            )
        )
        n_sensors += 2
    for l in range(n_links):
        scene.add_sensor(
            gs.sensors.Contact(entity_idx=robot.idx, link_idx_local=l, filter_link_idx=self_filter, history_length=hist)
        )
        scene.add_sensor(
            gs.sensors.ContactForce(
                entity_idx=robot.idx, link_idx_local=l, filter_link_idx=self_filter, history_length=hist
            )
        )
        n_sensors += 2
    _stage(
        f"genesis rawsens: {n_sensors} native sensors attached ({len(feet_local)} feet + {n_links} self-collision links)"
    )
    if with_imu:
        # Mirrors the preset's SensorConfig(gs.sensors.IMU) on the base link.
        pelvis = next(l for l in robot.links if l.name == "pelvis")
        scene.add_sensor(gs.sensors.IMU(entity_idx=robot.idx, link_idx_local=pelvis.idx_local))
        _stage("genesis rawsens: + IMU on pelvis")

    if spaced:
        # Mirror the wrapped genesis build: 20 m grid spacing, not centered
        # (genesis_config_classes.py default).  At 4096 envs this puts
        # robots up to ~1.3 km from the origin — isolates the fp32
        # large-coordinate cost that the origin-stacked raw cells avoid.
        scene.build(n_envs=num_envs, env_spacing=(20.0, 20.0), center_envs_at_origin=False)
    else:
        scene.build(n_envs=num_envs)
    if mirror_dynamics:
        _mirror_dynamics(robot)
    drive = _setup_active_pd(robot) if active else None

    def sync() -> None:
        robot.get_dofs_position().contiguous().cpu()

    for k in range(warmup):
        if drive is not None:
            drive(k)
        for _ in range(_DECIMATION):
            scene.step()
    sync()
    build_s = time.perf_counter() - t_build0
    _stage(f"genesis rawsens{'act' if active else ''} built+warm (build+warmup {build_s:.1f}s)")

    t0 = time.perf_counter()
    for k in range(steps):
        if drive is not None:
            drive(warmup + k)
        for _ in range(_DECIMATION):
            scene.step()
    sync()
    elapsed = time.perf_counter() - t0
    _probe_scene(scene, robot, "genesis raw(sens/act)", drive=drive, start_step=warmup + steps)
    return {"elapsed_s": elapsed, "build_warmup_s": build_s}


def raw_newton(num_envs: int, steps: int, warmup: int) -> dict:
    """Newton native: mirrors newton/examples/robot/example_robot_g1.py
    (replicate + SolverMuJoCo + CUDA-graph over the substep loop) with the
    g1_29dof newton preset's solver budget."""
    import newton
    import warp as wp

    t_build0 = time.perf_counter()
    robot_b = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(robot_b)
    robot_b.add_mjcf(
        str(_G1_XML),
        xform=wp.transform(wp.vec3(0.0, 0.0, _SPAWN_Z), wp.quat_identity()),
        floating=True,
        enable_self_collisions=True,
        collapse_fixed_joints=False,
    )
    scene_b = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(scene_b)
    scene_b.add_ground_plane()
    scene_b.replicate(robot_b, num_envs)
    model = scene_b.finalize()

    # Mirrors presets/g1_29dof/_newton_builders.py SolverMuJoCoCfg (flat).
    solver = newton.solvers.SolverMuJoCo(
        model,
        solver="newton",
        integrator="implicitfast",
        cone="elliptic",
        iterations=50,
        ls_iterations=50,
        ccd_iterations=50,
        njmax=1500,
        nconmax=128,
        use_mujoco_contacts=True,
    )
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    def substep_loop() -> None:
        nonlocal state_0, state_1
        for _ in range(_DECIMATION):
            state_0.clear_forces()
            solver.step(state_0, state_1, control, None, _DT)
            state_0, state_1 = state_1, state_0

    # CUDA graph over one control step — same as the official example and
    # the wrapped NewtonEnv (scene_manager.capture()).
    with wp.ScopedCapture() as capture:
        substep_loop()
    graph = capture.graph

    for _ in range(warmup):
        wp.capture_launch(graph)
    wp.synchronize_device()
    build_s = time.perf_counter() - t_build0
    _stage(f"newton raw built+warm (build+warmup {build_s:.1f}s)")

    t0 = time.perf_counter()
    for _ in range(steps):
        wp.capture_launch(graph)
    wp.synchronize_device()
    elapsed = time.perf_counter() - t0
    return {"elapsed_s": elapsed, "build_warmup_s": build_s}


def raw_mujoco(num_envs: int, steps: int, warmup: int) -> dict:
    """mjlab native: Simulation (mujoco-warp + internal CUDA graphs) fed the
    same G1 MJCF plus a ground plane; solver options mirror the g1_29dof
    mujoco preset."""
    import mujoco
    import warp as wp
    from mjlab.sim.sim import MujocoCfg, Simulation, SimulationCfg

    t_build0 = time.perf_counter()
    spec = mujoco.MjSpec.from_file(str(_G1_XML))
    floor = spec.worldbody.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [0.0, 0.0, 1.0]
    mj_model = spec.compile()
    if mj_model.nq >= 7:
        mj_model.qpos0[2] = _SPAWN_Z

    # Mirrors presets/g1_29dof/_mujoco_builders.py build_scene().
    cfg = SimulationCfg(
        nconmax=100,
        mujoco=MujocoCfg(
            timestep=_DT,
            iterations=50,
            ls_iterations=50,
            ccd_iterations=50,
            cone="elliptic",
        ),
    )
    sim = Simulation(num_envs=num_envs, cfg=cfg, model=mj_model, device="cuda:0")

    for _ in range(warmup * _DECIMATION):
        sim.step()
    wp.synchronize_device()
    build_s = time.perf_counter() - t_build0
    _stage(f"mujoco raw built+warm (build+warmup {build_s:.1f}s)")

    t0 = time.perf_counter()
    for _ in range(steps * _DECIMATION):
        sim.step()
    wp.synchronize_device()
    elapsed = time.perf_counter() - t0
    return {"elapsed_s": elapsed, "build_warmup_s": build_s}


# ── jaxrlworld cells ────────────────────────────────────────────────


def wrapped(sim: str, num_envs: int, steps: int, warmup: int) -> dict:
    import torch

    from rlworld.rl.configs.presets.g1_29dof.base import G1FlatConfig
    from rlworld.rl.evals.sim_initializers import get_initializer

    t_build0 = time.perf_counter()
    cfgs = G1FlatConfig(sim_type=sim, num_envs=num_envs, seed=0).build()
    sim_key = {"genesis": "Genesis", "newton": "Newton", "mujoco": "MujocoEnv"}[sim]
    env = get_initializer(sim_key).init_environment(cfgs)
    env.reset()

    actions = torch.zeros((num_envs, env.num_actions), device=env.device)

    def sync() -> None:
        torch.cuda.synchronize()
        # Backend queues (taichi / warp) complete on a device->host readback.
        env.get_robot_data().root_link_pos_w[0].detach().cpu()
        torch.cuda.synchronize()

    for _ in range(warmup):
        env.step(actions)
    sync()
    build_s = time.perf_counter() - t_build0
    _stage(f"{sim} jaxrlworld built+warm (build+warmup {build_s:.1f}s)")

    t0 = time.perf_counter()
    for _ in range(steps):
        env.step(actions)
    sync()
    elapsed = time.perf_counter() - t0
    return {"elapsed_s": elapsed, "build_warmup_s": build_s}


def raw_genesis_contact_list(num_envs: int, steps: int, warmup: int) -> dict:
    """Candidate-4 cost probe: NO native sensors — instead, after every
    substep, read the solver's global contact list once
    (``collider.get_contacts(as_tensor=True, to_torch=True)``) and compute
    the same quantities our contact groups need (per-link ``found`` bool +
    per-link 3-vec force, for both the feet-vs-ground and self-collision
    groups) with torch masking.  This measures the FULL replacement
    workload of the per-link native-sensor design; compare against:

        rawact      (no sensors)       -> the floor
        rawsensact  (68 native sensors) -> the current design's cost
    """
    import genesis as gs
    import torch

    t_build0 = time.perf_counter()
    gs.init(backend=gs.gpu, logging_level="warning")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=_DT, substeps=1),
        rigid_options=gs.options.RigidOptions(
            dt=_DT,
            constraint_solver=gs.constraint_solver.Newton,
            iterations=10,
            ls_iterations=20,
            tolerance=1e-5,
            constraint_timeconst=0.02,
            enable_collision=True,
            enable_self_collision=True,
            enable_joint_limit=True,
            max_collision_pairs=100,
            batch_dofs_info=True,
        ),
        show_viewer=False,
    )
    ground = scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(gs.morphs.MJCF(file=str(_G1_XML), pos=(0.0, 0.0, _SPAWN_Z)))
    scene.build(n_envs=num_envs)
    drive = _setup_active_pd(robot)

    dev = torch.device("cuda:0")
    ground_links = torch.tensor([l.idx for l in ground.links], device=dev)
    robot_links = torch.tensor([l.idx for l in robot.links], device=dev)
    feet_links = torch.tensor([l.idx for l in robot.links if "ankle_roll" in l.name], device=dev)

    def group_read(link_a, link_b, force_a, valid, primary, counterpart):
        """Per-primary-link found + net force vs a counterpart link set."""
        a_is_p = (link_a.unsqueeze(-1) == primary).any(-1)
        b_is_p = (link_b.unsqueeze(-1) == primary).any(-1)
        a_is_c = (link_a.unsqueeze(-1) == counterpart).any(-1)
        b_is_c = (link_b.unsqueeze(-1) == counterpart).any(-1)
        pair = valid & ((a_is_p & b_is_c) | (b_is_p & a_is_c))
        # (n_envs, C, P) one-hot of which primary link each pair touches
        pmask_a = (link_a.unsqueeze(-1) == primary) & pair.unsqueeze(-1)
        pmask_b = (link_b.unsqueeze(-1) == primary) & pair.unsqueeze(-1)
        found = (pmask_a | pmask_b).any(1)  # (n_envs, P)
        # net force on the primary side: force_a where primary is side a, -force_a where side b
        f = torch.einsum("ncp,nci->npi", (pmask_a.float() - pmask_b.float()), force_a)
        return found, f

    def read_all():
        # Public documented path (rigid_entity.get_contacts): one batched
        # collider readback + torch masking.  NOTE for the real
        # implementation: rows past each env's live n_contacts can be
        # stale on the zero-copy path — production code must mask with
        # collider n_contacts, not rely on valid_mask alone.
        cd = robot.get_contacts()
        link_a, link_b = cd["link_a"], cd["link_b"]
        valid = cd["valid_mask"]
        force_a = cd["force_a"]
        feet = group_read(link_a, link_b, force_a, valid, feet_links, ground_links)
        selfc = group_read(link_a, link_b, force_a, valid, robot_links, robot_links)
        return feet, selfc

    def sync() -> None:
        robot.get_dofs_position().contiguous().cpu()

    for k in range(warmup):
        drive(k)
        for _ in range(_DECIMATION):
            scene.step()
            read_all()
    sync()
    build_s = time.perf_counter() - t_build0
    _stage(f"genesis rawcontactlist built+warm (build+warmup {build_s:.1f}s)")

    t0 = time.perf_counter()
    for k in range(steps):
        drive(warmup + k)
        for _ in range(_DECIMATION):
            scene.step()
            read_all()
    sync()
    elapsed = time.perf_counter() - t0

    # State dump: one extra driven control step, then inspect everything —
    # base height (are the robots even standing?), the collider's raw
    # per-env contact counter, and the group reads themselves.
    drive(warmup + steps)
    for _ in range(_DECIMATION):
        scene.step()
    (feet_found, feet_force), (self_found, _self_force) = read_all()
    z = robot.get_pos()[:, 2]
    nc = scene.rigid_solver.collider._collider_state.n_contacts.to_numpy()
    cd = robot.get_contacts()
    _stage(
        f"rawcontactlist state: base z mean {z.mean():.3f} min {z.min():.3f} max {z.max():.3f}; "
        f"collider n_contacts mean {nc.mean():.2f} max {nc.max()}; "
        f"feet found/env {feet_found.float().sum(1).mean():.2f}, |F| mean {feet_force.norm(dim=-1).mean():.2f}; "
        f"self found/env {self_found.float().sum(1).mean():.2f}"
    )
    _stage(
        f"rawcontactlist buffer: n_contacts axis={cd['link_a'].shape[1]} "
        f"(reads/ctrl-step={_DECIMATION}, groups=2: feet[{len(feet_links)}] self[{len(robot_links)}])"
    )
    return {"elapsed_s": elapsed, "build_warmup_s": build_s}


def wrapped_bare_scene(num_envs: int, steps: int, warmup: int, strip_dr: bool = False) -> dict:
    """Build the FULL wrapped genesis env (identical scene construction,
    startup DR, warmup env.step traffic), then time ONLY bare
    ``scene.step()`` on that already-built scene — no managers, no action
    application, no sensor reads.

    Separates the two remaining hypotheses for wrapped scene.step being
    slower than rawsensact:
      - ≈ wrapped's phys:scene.step  -> the BUILT SCENE itself is slower
        (construction/DR difference vs the raw mirror), independent of any
        per-step JaxRLWorld work;
      - ≈ rawsensact                 -> the scene is fine and the cost
        comes from per-step interleaving.
    """
    import torch

    from rlworld.rl.configs.presets.g1_29dof.base import G1FlatConfig
    from rlworld.rl.evals.sim_initializers import get_initializer

    t_build0 = time.perf_counter()
    cfgs = G1FlatConfig(sim_type="genesis", num_envs=num_envs, seed=0).build()
    if strip_dr:
        # Remove every DR event term (startup + reset + interval) while
        # keeping state-reset events intact, to isolate whether the
        # per-env DR writes (friction ratio / mass shift / COM shift /
        # gains) are what makes the wrapped-built scene's bare physics
        # slower than the raw mirror.
        from rlworld.rl.configs.base_config import iter_terms
        from rlworld.rl.configs.events.event_term_config import EventTermConfig

        stripped = []
        for name, _term in list(iter_terms(cfgs.event, EventTermConfig).items()):
            if "randomize" in name:
                setattr(cfgs.event, name, None)
                stripped.append(name)
        _stage(f"genesis wrappedbarenodr: stripped DR terms {stripped}")
    env = get_initializer("Genesis").init_environment(cfgs)
    env.reset()

    actions = torch.zeros((num_envs, env.num_actions), device=env.device)
    for _ in range(warmup):
        env.step(actions)
    # Fresh reset so every env is in a clean standing state BEFORE the PD
    # drive captures its reset anchor — capturing post-warmup poses would
    # pin fallen/pushed envs into permanent high-contact states and make
    # the cell incomparable to the raw cells' clean spawn anchor.
    env.reset()
    scene = env.scene_manager.scene
    robot = env.scene_manager.robot
    # Same standing PD drive as the raw active cells (the wrapped scene's
    # internal kp/kv are unset because the preset actuator is explicit),
    # so this cell is directly comparable to rawfull: the delta is pure
    # scene-construction difference, not pose/contact-state skew.
    drive = _setup_active_pd(robot)

    def sync() -> None:
        torch.cuda.synchronize()
        robot.get_dofs_position().contiguous().cpu()

    sync()
    build_s = time.perf_counter() - t_build0
    _stage(f"genesis wrappedbare built+warm (build+warmup {build_s:.1f}s)")

    t0 = time.perf_counter()
    for k in range(steps):
        drive(k)
        for _ in range(_DECIMATION):
            scene.step()
    sync()
    elapsed = time.perf_counter() - t0
    _probe_scene(scene, robot, "genesis wrappedbare", drive=drive, start_step=steps)
    return {"elapsed_s": elapsed, "build_warmup_s": build_s}


def wrapped_bisect(num_envs: int, steps: int, warmup: int) -> dict:
    """In-process bisect of the wrapped-built scene's bare-step slowdown.

    Builds the full wrapped env once, then times bare PD-driven
    ``scene.step()`` chunks back-to-back, NEUTRALIZING one class of
    per-env write between chunks (friction ratio -> 1, mass/COM shift ->
    0, dof frictionloss -> 0, force range -> +/-1e6).  Whichever
    neutralization collapses the step time toward the raw mirror is the
    write responsible.  A final baseline-order re-run is NOT possible
    (writes are destructive), so read deltas chunk-to-chunk.
    """
    import torch

    from rlworld.rl.configs.presets.g1_29dof.base import G1FlatConfig
    from rlworld.rl.evals.sim_initializers import get_initializer

    t_build0 = time.perf_counter()
    cfgs = G1FlatConfig(sim_type="genesis", num_envs=num_envs, seed=0).build()
    env = get_initializer("Genesis").init_environment(cfgs)
    env.reset()
    actions = torch.zeros((num_envs, env.num_actions), device=env.device)
    for _ in range(warmup):
        env.step(actions)
    env.reset()
    scene = env.scene_manager.scene
    robot = env.scene_manager.robot
    solver = scene.rigid_solver
    drive = _setup_active_pd(robot)

    def sync() -> None:
        torch.cuda.synchronize()
        robot.get_dofs_position().contiguous().cpu()

    sync()
    build_s = time.perf_counter() - t_build0
    _stage(f"genesis wrappedbisect built+warm (build+warmup {build_s:.1f}s)")

    step_counter = [0]

    def timed_chunk(label: str) -> float:
        sync()
        t0 = time.perf_counter()
        for _ in range(steps):
            drive(step_counter[0])
            step_counter[0] += 1
            for _ in range(_DECIMATION):
                scene.step()
        sync()
        ms = (time.perf_counter() - t0) / steps * 1e3
        _stage(f"wrappedbisect [{label}]: {ms:.3f} ms/ctrl-step")
        return ms

    n_links = robot.n_links
    link_ids = list(range(n_links))
    dof_ids = list(range(6, robot.n_dofs))
    dev = env.device

    results: dict[str, float] = {}
    results["as_built"] = timed_chunk("as-built (all wrapped writes live)")

    robot.set_friction_ratio(torch.ones(num_envs, n_links, device=dev), link_ids)
    results["friction_ratio=1"] = timed_chunk("friction_ratio -> 1.0")

    robot.set_mass_shift(torch.zeros(num_envs, n_links, device=dev), link_ids)
    robot.set_COM_shift(torch.zeros(num_envs, n_links, 3, device=dev), link_ids)
    results["shifts=0"] = timed_chunk("mass/COM shift -> 0")

    robot.set_dofs_frictionloss([0.0] * len(dof_ids), dof_ids)
    results["frictionloss=0"] = timed_chunk("dof frictionloss -> 0")

    robot.set_dofs_force_range([-1e6] * len(dof_ids), [1e6] * len(dof_ids), dof_ids)
    results["force_range=inf"] = timed_chunk("dof force range -> +/-1e6")

    for label, ms in results.items():
        print(f"[bisect] {label:<28} {ms:>8.3f} ms/ctrl-step", flush=True)
    return {"elapsed_s": results["as_built"] * steps / 1e3, "build_warmup_s": build_s}


# ── cell runner (child) ─────────────────────────────────────────────


def run_cell(sim: str, mode: str, num_envs: int, steps: int, warmup: int) -> dict:
    _stage(f"cell start: {sim}:{mode}:{num_envs} (steps={steps}, warmup={warmup})")
    if mode == "raw":
        fn = {"genesis": raw_genesis, "newton": raw_newton, "mujoco": raw_mujoco}[sim]
        r = fn(num_envs, steps, warmup)
    elif mode in _GENESIS_EXTRA_MODES:
        assert sim == "genesis", f"{mode} mode is genesis-only"
        r = {
            "rawsens": lambda: raw_genesis_sensors(num_envs, steps, warmup),
            "rawact": lambda: raw_genesis(num_envs, steps, warmup, active=True),
            "rawsensact": lambda: raw_genesis_sensors(num_envs, steps, warmup, active=True),
            "rawsensactdyn": lambda: raw_genesis_sensors(num_envs, steps, warmup, active=True, mirror_dynamics=True),
            "rawfull": lambda: raw_genesis_sensors(
                num_envs, steps, warmup, active=True, mirror_dynamics=True, with_imu=True
            ),
            "rawfullspaced": lambda: raw_genesis_sensors(
                num_envs, steps, warmup, active=True, mirror_dynamics=True, with_imu=True, spaced=True
            ),
            "rawcontactlist": lambda: raw_genesis_contact_list(num_envs, steps, warmup),
            "wrappedbare": lambda: wrapped_bare_scene(num_envs, steps, warmup),
            "wrappedbarenodr": lambda: wrapped_bare_scene(num_envs, steps, warmup, strip_dr=True),
            "wrappedbisect": lambda: wrapped_bisect(num_envs, steps, warmup),
        }[mode]()
    else:
        r = wrapped(sim, num_envs, steps, warmup)

    ms_per_ctrl = r["elapsed_s"] / steps * 1e3
    env_steps_per_s = num_envs * steps / r["elapsed_s"]
    _stage(f"cell done: {ms_per_ctrl:.3f} ms/ctrl-step, {env_steps_per_s:,.0f} env-steps/s")
    return {
        "sim": sim,
        "mode": mode,
        "num_envs": num_envs,
        "steps": steps,
        "ms_per_ctrl_step": ms_per_ctrl,
        "env_steps_per_s": env_steps_per_s,
        "build_warmup_s": r["build_warmup_s"],
    }


# ── parent ──────────────────────────────────────────────────────────


def run_parent(args) -> int:
    out_path = Path(args.out).resolve()
    log_dir = out_path.parent / (out_path.stem + "_logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    env_counts = [int(x) for x in args.env_counts.split(",")]
    if args.cells:
        specs = [c.strip() for c in args.cells.split(",")]
    else:
        specs = [f"{s}:{m}:{n}" for s in _SIMS for m in _MODES for n in env_counts]

    results: dict[str, dict] = {}
    for spec in specs:
        sim, mode, n = spec.split(":")
        tag = spec.replace(":", "_")
        log_path = log_dir / f"{tag}.log"
        result_path = log_dir / f"{tag}.json"
        if result_path.exists():
            result_path.unlink()
        print(f"[bench] running {spec} ...", flush=True)
        t0 = time.perf_counter()
        with open(log_path, "w") as lf:
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    _MODULE,
                    "--cell",
                    spec,
                    "--result-json",
                    str(result_path),
                    "--steps",
                    str(args.steps),
                    "--warmup",
                    str(args.warmup),
                ],
                stdout=lf,
                stderr=subprocess.STDOUT,
                env=os.environ.copy(),
            )
        wall = time.perf_counter() - t0
        if result_path.exists():
            d = json.loads(result_path.read_text())
            results[spec] = d
            print(f"[bench]   -> {d['ms_per_ctrl_step']:.3f} ms/ctrl-step ({wall:.0f}s incl. build)", flush=True)
        else:
            results[spec] = {"sim": sim, "mode": mode, "num_envs": int(n), "error": True}
            print(f"[bench]   -> CRASH (rc={proc.returncode}, see {log_path})", flush=True)

    # ── report ────────────────────────────────────────────────────
    def get(sim: str, mode: str, n: int) -> dict | None:
        return results.get(f"{sim}:{mode}:{n}")

    lines: list[str] = []
    lines.append("=" * 110)
    lines.append("G1 step-time benchmark — JaxRLWorld env.step vs raw simulator")
    lines.append("=" * 110)
    lines.append(f"asset: {_G1_XML}")
    lines.append(
        f"1 control step = {_DECIMATION} x dt {_DT}s physics steps (0.02 s sim time); timed steps: {args.steps}"
    )
    lines.append("")
    header = f"{'sim':<10}{'mode':<13}" + "".join(
        f"{f'n={n} ms/ctrl':>16}{f'n={n} envsteps/s':>20}" for n in env_counts
    )
    lines.append(header + f"{'build+warm(s)':>16}")
    lines.append("-" * 110)
    for sim in _SIMS:
        modes = _MODES + (_GENESIS_EXTRA_MODES if sim == "genesis" else ())
        for mode in modes:
            if mode in _GENESIS_EXTRA_MODES and not any(get(sim, mode, n) for n in env_counts):
                continue
            row = f"{sim:<10}{mode:<13}"
            build = ""
            for n in env_counts:
                d = get(sim, mode, n)
                if d is None or d.get("error"):
                    row += f"{'—':>16}{'—':>20}"
                else:
                    row += f"{d['ms_per_ctrl_step']:>16.3f}{d['env_steps_per_s']:>20,.0f}"
                    build = f"{d['build_warmup_s']:>16.1f}"
            lines.append(row + build)
    lines.append("")
    lines.append("[wrapper overhead]  jaxrlworld / raw  (>1 = JaxRLWorld-side cost dominates)")
    for sim in _SIMS:
        for n in env_counts:
            w, r = get(sim, "jaxrlworld", n), get(sim, "raw", n)
            if w and r and not w.get("error") and not r.get("error"):
                lines.append(f"  {sim:<10} n={n:<6} x{w['ms_per_ctrl_step'] / r['ms_per_ctrl_step']:.2f}")
    lines.append("")
    lines.append("[genesis decomposition] ms/ctrl-step by load level; wrapped - rawsensact = true JaxRLWorld-side cost")
    for n in env_counts:
        cells = {
            m: get("genesis", m, n)
            for m in ("raw", "rawact", "rawsens", "rawsensact", "rawsensactdyn", "rawfull", "wrappedbare", "jaxrlworld")
        }
        parts = [f"{m}={d['ms_per_ctrl_step']:.1f}" for m, d in cells.items() if d and not d.get("error")]
        if not parts:
            continue
        lines.append(f"  n={n:<6} " + "  ".join(parts))
        w, rsa = cells["jaxrlworld"], cells["rawsensact"]
        if w and rsa and not w.get("error") and not rsa.get("error"):
            delta = w["ms_per_ctrl_step"] - rsa["ms_per_ctrl_step"]
            lines.append(f"          -> wrapped - rawsensact = {delta:.1f} ms JaxRLWorld-side")
    lines.append("")
    lines.append("[engine comparison] raw ms/ctrl-step relative to newton raw (=1.00)")
    for n in env_counts:
        ref = get("newton", "raw", n)
        if not ref or ref.get("error"):
            continue
        for sim in _SIMS:
            d = get(sim, "raw", n)
            if d and not d.get("error"):
                lines.append(f"  n={n:<6} {sim:<10} x{d['ms_per_ctrl_step'] / ref['ms_per_ctrl_step']:.2f}")
    lines.append("")
    lines.append("[Reading] genesis wrapped/raw >> other sims' ratio -> JaxRLWorld genesis path is the problem;")
    lines.append("          genesis raw >> newton/mujoco raw          -> the engine itself is the problem.")

    report = "\n".join(lines)
    out_path.write_text(report + "\n")
    print()
    print(report)
    print(f"\nReport written to: {out_path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", default=None, help="internal: run one cell 'sim:mode:num_envs'")
    ap.add_argument("--result-json", default=None, help="internal: child result path")
    ap.add_argument("--cells", default=None, help="comma-separated subset, e.g. 'genesis:raw:4096'")
    ap.add_argument("--env-counts", default="1,4096")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--out", default="g1_step_benchmark.txt")
    args = ap.parse_args()

    if args.cell is not None:
        sim, mode, n = args.cell.split(":")
        result = run_cell(sim, mode, int(n), args.steps, args.warmup)
        Path(args.result_json).write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return 0

    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
