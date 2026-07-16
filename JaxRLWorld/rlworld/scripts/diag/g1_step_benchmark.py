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

# G1 MJCF shared by every backend (rlworld/assets/g1/g1.xml — resolved from
# this file's location so the raw cells need no JaxRLWorld import).
_G1_XML = Path(__file__).resolve().parents[2] / "assets" / "g1" / "g1.xml"
_SPAWN_Z = 0.8


def _stage(msg: str) -> None:
    print(f"[STAGE] {msg}", flush=True)


# ── raw cells (no JaxRLWorld) ───────────────────────────────────────


def raw_genesis(num_envs: int, steps: int, warmup: int) -> dict:
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

    def sync() -> None:
        # Device->host readback forces completion of all queued kernels.
        robot.get_dofs_position().contiguous().cpu()

    for _ in range(warmup * _DECIMATION):
        scene.step()
    sync()
    build_s = time.perf_counter() - t_build0
    _stage(f"genesis raw built+warm (build+warmup {build_s:.1f}s)")

    t0 = time.perf_counter()
    for _ in range(steps * _DECIMATION):
        scene.step()
    sync()
    elapsed = time.perf_counter() - t0
    return {"elapsed_s": elapsed, "build_warmup_s": build_s}


def raw_genesis_sensors(num_envs: int, steps: int, warmup: int) -> dict:
    """raw_genesis PLUS the g1 preset's two contact-sensor groups, built with
    Genesis's native API only (one gs.sensors.Contact + ContactForce pair per
    primary link, exactly what GenesisContactSensor.create_native_sensors
    does: 2 feet links vs ground + EVERY robot link for self-collision).

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
    ground_links = {l.idx_local for l in ground.links}
    feet_local = [l.idx_local for l in robot.links if "ankle_roll" in l.name]
    hist = _DECIMATION

    # Group 1 — feet vs ground (primary: 2 ankle_roll links, secondary:
    # everything except the terrain -> blacklist = all robot links).
    feet_filter = tuple(sorted({l.idx_local for l in robot.links}))
    # Group 2 — self collision (primary: every link, secondary: self ->
    # blacklist = ground links).
    self_filter = tuple(sorted(ground_links))

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

    scene.build(n_envs=num_envs)

    def sync() -> None:
        robot.get_dofs_position().contiguous().cpu()

    for _ in range(warmup * _DECIMATION):
        scene.step()
    sync()
    build_s = time.perf_counter() - t_build0
    _stage(f"genesis rawsens built+warm (build+warmup {build_s:.1f}s)")

    t0 = time.perf_counter()
    for _ in range(steps * _DECIMATION):
        scene.step()
    sync()
    elapsed = time.perf_counter() - t0
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


# ── cell runner (child) ─────────────────────────────────────────────


def run_cell(sim: str, mode: str, num_envs: int, steps: int, warmup: int) -> dict:
    _stage(f"cell start: {sim}:{mode}:{num_envs} (steps={steps}, warmup={warmup})")
    if mode == "raw":
        fn = {"genesis": raw_genesis, "newton": raw_newton, "mujoco": raw_mujoco}[sim]
        r = fn(num_envs, steps, warmup)
    elif mode == "rawsens":
        assert sim == "genesis", "rawsens mode is genesis-only"
        r = raw_genesis_sensors(num_envs, steps, warmup)
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
        for mode in _MODES:
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
