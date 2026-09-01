"""Verify the reset-path fixes: solver-internal clearing, curriculum ordering, substep forces.

One command covers everything::

    jaxpy -m jaxrlworld.scripts.diag.perf.reset_path_diag

The parent process spawns one child per case (a simulator backend cannot share
a process with another — see ``envs/utils/lazy_import_check.py``) and prints a
PASS/FAIL matrix at the end. Two of the cases are deliberately-broken CONTROL
runs: they re-enable the old code path and are expected to FAIL. If a control
run passes, the corresponding check is not actually discriminating and the
"fixed" result next to it means nothing.

Checks
------
A1  Newton clears MuJoCo solver-internal state on reset.
    ``SolverMuJoCo`` carries ``qacc_warmstart`` (the constraint solver's
    initial guess), ``qfrc_applied`` / ``xfrc_applied``, ``act`` and ``ctrl``
    across ``step`` calls. Writing a fresh joint pose touches none of them, so
    a reset environment warm-started from the previous episode — and a single
    NaN survived every subsequent reset, leaving that world permanently dead
    (newton-physics/newton#1266). ``NewtonEnv._reset_scene`` now calls
    ``solver.reset(state_0, world_mask=<reset envs>, flags=0)``.
      T1  reset worlds' input buffers are zeroed; NON-reset worlds keep theirs.
      T2  reset worlds' ``qacc_warmstart`` no longer holds the pre-reset value.
      T3  a world poisoned with NaN becomes finite again after a reset.

A2  The curriculum manager observes the ENDING episode's terminal state.
    ``MujocoEnv`` used to run ``scene_manager.reset`` (``mjwarp.reset_data``,
    which snaps ``qpos`` back to ``qpos0``) as the first statement of its
    ``_reset_idx`` override — ahead of ``curriculum_manager.compute``. A
    curriculum term on MuJoCo therefore saw the spawn pose while Genesis and
    Newton saw the real one. The simulator-side reset now goes through the
    ``World._reset_scene`` hook, which the base ``_reset_idx`` invokes after
    the curriculum. Run on all three backends: the point is that they agree.

A5  An external link wrench acts for the FULL physics step at any substep count.
    ``NewtonEnv._step_physics`` writes ``state_0.body_f`` once per decimation
    iteration. Under the old double-buffered substep loop the two ``State``
    references were swapped after every substep, so from substep 1 onward the
    solver's *input* was a buffer nobody had written the wrench into. Running
    SolverMuJoCo in single-state mode removes the swap.

    Method — momentum bookkeeping, fully analytic. The robot is placed high in
    the air (no contact) and stepped once, twice: once with no wrench, once
    with a known constant ``F_z``. Total linear momentum ``p = sum_i m_i v_i``
    is read before and after each run. Gravity, PD joint torques and every
    internal constraint force are identical between the two runs and cancel in
    the difference, so ``(dp_force - dp_free)[z] == F_z * control_dt`` exactly,
    independent of the mass distribution. Only meaningful at ``substeps >= 2``;
    the ``substeps=1`` row is reported for reference.

Other useful invocations::

    jaxpy -m jaxrlworld.scripts.diag.perf.reset_path_diag --only newton
    jaxpy -m jaxrlworld.scripts.diag.perf.reset_path_diag --case newton   # run one child inline
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

_MODULE = "jaxrlworld.scripts.diag.perf.reset_path_diag"

# Sentinel written into the solver input buffers before the A1 reset. Large and
# obviously non-physical so a surviving value is unmistakable.
_SENTINEL = 12345.0
_RESET_ENVS = (0, 3)
_SPAWN_Z = 5.0  # A5: high enough that one control step of free fall cannot reach the ground
_FORCE_TOL = 0.02  # A5: |ratio - 1| bound; mjwarp parallel reductions are not bit-exact
_TERMINAL_Z = 1.7315  # A2: distinctive height, far from any preset spawn pose and from qpos0
_Z_TOL = 1e-3


class Case:
    """One child invocation: a backend configuration plus the checks it runs."""

    def __init__(
        self, name, sim, checks, *, substeps=1, no_clear=False, legacy_substep=False, expect_pass=True, note=""
    ):
        self.name = name
        self.sim = sim
        self.checks = checks
        self.substeps = substeps
        self.no_clear = no_clear
        self.legacy_substep = legacy_substep
        self.expect_pass = expect_pass
        self.note = note


CASES: tuple[Case, ...] = (
    Case(
        "newton",
        "newton",
        ("A1", "A2", "A5"),
        substeps=1,
        note="all Newton checks; the A5 row here is reference-only (no swap at substeps=1)",
    ),
    Case("newton-sub2", "newton", ("A5",), substeps=2, note="the discriminating A5 case"),
    Case(
        "newton-sub2-legacy",
        "newton",
        ("A5",),
        substeps=2,
        legacy_substep=True,
        expect_pass=False,
        note="CONTROL: restores the double-buffered swap; A5 must FAIL (ratio < 1)",
    ),
    Case(
        "newton-noclear",
        "newton",
        ("A1",),
        substeps=1,
        no_clear=True,
        expect_pass=False,
        note="CONTROL: disables _reset_scene; A1 must FAIL",
    ),
    Case("mujoco", "mujoco", ("A2",)),
    Case("genesis", "genesis", ("A2",)),
)

_BY_NAME = {c.name: c for c in CASES}


def _fmt(x) -> str:
    return f"{float(x):+.6e}"


# ══════════════════════════════════════════════════════════════════════════
# Curriculum recorder (A2) — registered on every child so it is always live
# ══════════════════════════════════════════════════════════════════════════

_RECORD: dict = {}


def record_terminal_state(env, env_ids):
    """Curriculum term: snapshot what the curriculum can see at compute time."""
    rd = env.get_robot_data("robot")
    _RECORD["z"] = rd.root_link_pos_w[env_ids, 2].detach().clone()
    _RECORD["ep_len"] = env.episode_length_buf[env_ids].detach().clone()
    _RECORD["calls"] = _RECORD.get("calls", 0) + 1
    return {}


# ══════════════════════════════════════════════════════════════════════════
# Child: environment construction
# ══════════════════════════════════════════════════════════════════════════


def _build_env(case: Case, num_envs: int):
    from jaxrlworld.rl.configs.curriculums import CurriculumTermConfig
    from jaxrlworld.rl.configs.presets.go2.base import Go2FlatConfig
    from jaxrlworld.rl.runners import BaseRunner

    cfgs = Go2FlatConfig(sim_type=case.sim, num_envs=num_envs).build()
    # iter_terms() discovers instance attributes, so a plain assignment
    # registers the term without needing a preset subclass.
    cfgs.curriculum.record_terminal_state = CurriculumTermConfig(func=record_terminal_state)
    if case.sim == "newton":
        cfgs.scene.substeps = case.substeps

    env = BaseRunner.create_with_env(cfgs).env
    env.reset()

    if case.no_clear:
        # Pre-fix reproduction: NewtonEnv._reset_scene becomes a no-op again.
        env._reset_scene = lambda env_ids: None  # noqa: ARG005
    if case.legacy_substep:
        # Pre-fix reproduction: restore the double-buffered swap. The captured
        # CUDA graph recorded the single-state loop, so drop it and run the
        # substep loop eagerly, otherwise the flag would have no effect.
        sm = env.scene_manager
        sm._use_single_state = False
        sm.use_cuda_graph = False
        sm.graph = None
    return env


def _print_banner(case: Case, env) -> None:
    print("=" * 92)
    print(f"RESET-PATH DIAG   case={case.name}   sim={case.sim}")
    if case.note:
        print(f"  {case.note}")
    print("=" * 92)
    print(f"[cfg] num_envs   = {env.num_envs}   decimation = {env.decimation}")
    print(f"[cfg] physics_dt = {env.physics_dt}   control_dt = {env.control_dt}")
    print(f"[cfg] checks     = {list(case.checks)}   expect_pass = {case.expect_pass}")
    if case.sim == "newton":
        sm = env.scene_manager
        solver = sm.solver
        print(f"[cfg] solver_type    = {sm.config.solver_type}   use_mujoco_cpu = {solver.use_mujoco_cpu}")
        print(f"[cfg] use_mjc_contacts = {sm._use_mujoco_contacts}   single_state = {sm._use_single_state}")
        print(f"[cfg] substeps       = {sm.config.substeps}   substep_dt = {sm.substep_dt}")
        print(f"[cfg] cuda_graph     = {sm.use_cuda_graph}   graph_captured = {sm.graph is not None}")
        print(
            f"[cfg] mjw nworld/nv/na = {int(solver.mjw_data.nworld)} / "
            f"{int(solver.mj_model.nv)} / {int(solver.mj_model.na)}"
        )
    print(f"[cfg] contact groups = {env.contact_manager.group_names()}")
    print(f"[cfg] reset-scene override active: no_clear={case.no_clear} legacy_substep={case.legacy_substep}")
    print()


# ══════════════════════════════════════════════════════════════════════════
# A5 — external wrench impulse across substeps
# ══════════════════════════════════════════════════════════════════════════


def _base_link_name(env) -> str:
    """Bare name of body 0 — the floating base of the single articulation."""
    model = env.scene_manager.model
    bodies_per_world = model.body_count // model.world_count
    return list(model.body_label)[:bodies_per_world][0].rsplit("/", 1)[-1]


def _force_run(env, force_z: float, victim: int) -> dict:
    """Place the robot in the air, step once, return the momentum delta + telemetry."""
    import torch
    import warp as wp

    rd = env.get_robot_data("robot")
    writer = env.get_robot_state_writer("robot")
    model = env.scene_manager.model
    bodies_per_world = model.body_count // model.world_count
    mass = wp.to_torch(model.body_mass).view(env.num_envs, bodies_per_world)  # (E, B)

    n, dev = env.num_envs, env.device
    pos = torch.zeros(n, 3, device=dev)
    pos[:, 2] = _SPAWN_Z
    quat = torch.zeros(n, 4, device=dev)
    quat[:, 0] = 1.0
    zeros3 = torch.zeros(n, 3, device=dev)
    default_q = rd.default_joint_pos.unsqueeze(0).expand(n, -1).contiguous()

    writer.set_root_pose(pos, quat)
    writer.set_root_velocity(zeros3, zeros3)
    writer.set_dof_positions(default_q)
    writer.set_dof_velocities(torch.zeros_like(default_q))
    writer.eval_fk()
    env._invalidate_cache()

    if force_z != 0.0:
        env.set_external_wrench(_base_link_name(env), torch.tensor([0.0, 0.0, force_z], device=dev), victim)
    else:
        env.clear_external_wrench()

    def momentum():
        return (mass.unsqueeze(-1) * rd.body_com_lin_vel_w_all).sum(dim=1)  # (E, 3)

    z_before = float(rd.root_link_pos_w[victim, 2])
    p0 = momentum().clone()
    _obs, _rew, terminated, truncated, _extras = env.step(torch.zeros(n, env.num_actions, device=dev))
    p1 = momentum().clone()

    result = {
        "dp": (p1 - p0)[victim].clone(),
        "z_before": z_before,
        "z_after": float(env.get_robot_data("robot").root_link_pos_w[victim, 2]),
        "contact": bool(env.contact_manager.is_contact("feet_ground_contact")[victim].any()),
        "did_reset": bool((terminated | truncated)[victim]),
        "total_mass": float(mass[victim].sum()),
    }
    env.clear_external_wrench()
    return result


def check_a5(env, force: float, victim: int) -> dict:
    substeps = env.scene_manager.config.substeps
    expected = force * env.control_dt

    print("-" * 92)
    print(f"A5  external wrench impulse   (substeps={substeps}, force_z={force} N, env {victim})")
    print("-" * 92)

    free = _force_run(env, 0.0, victim)
    forced = _force_run(env, force, victim)
    impulse = float((forced["dp"] - free["dp"])[2])
    ratio = impulse / expected if expected != 0 else float("nan")

    print(f"  total_mass         = {free['total_mass']:.6f} kg")
    print(
        f"  free   run: z {free['z_before']:.4f} -> {free['z_after']:.4f}  "
        f"contact={free['contact']} reset={free['did_reset']}  dp={[round(float(v), 6) for v in free['dp']]}"
    )
    print(
        f"  forced run: z {forced['z_before']:.4f} -> {forced['z_after']:.4f}  "
        f"contact={forced['contact']} reset={forced['did_reset']}  dp={[round(float(v), 6) for v in forced['dp']]}"
    )
    print(f"  measured impulse_z = {_fmt(impulse)} N.s")
    print(f"  expected impulse_z = {_fmt(expected)} N.s   (= force * control_dt)")
    print(f"  ratio              = {ratio:.6f}   (tolerance |ratio-1| <= {_FORCE_TOL})")

    invalid = free["contact"] or forced["contact"] or free["did_reset"] or forced["did_reset"]
    if invalid:
        print("\n  A5: INCONCLUSIVE — contact or a reset happened during the measured step.")
        print("      Lower --force or raise the spawn height.")
        return {"status": "INCONCLUSIVE", "ratio": ratio, "substeps": substeps}

    ok = abs(ratio - 1.0) <= _FORCE_TOL
    if substeps == 1:
        print("\n  NOTE: substeps=1 has no swap on either code path, so this row cannot")
        print("        distinguish fixed from broken. It is reported for reference.")
    print(f"\n  A5: {'PASS' if ok else 'FAIL'}\n")
    return {"status": "PASS" if ok else "FAIL", "ratio": ratio, "substeps": substeps}


# ══════════════════════════════════════════════════════════════════════════
# A2 — curriculum sees the terminal state
# ══════════════════════════════════════════════════════════════════════════


def check_a2(env, warmup_steps: int) -> dict:
    import torch

    print("-" * 92)
    print("A2  curriculum observes the terminal state")
    print("-" * 92)
    print(f"  curriculum terms = {env.curriculum_manager.active_terms}")
    if not env.curriculum_manager.active_terms:
        print("\n  A2: FAIL — the recording curriculum term was not registered.\n")
        return {"status": "FAIL", "reason": "term not registered"}

    zero_act = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    for _ in range(warmup_steps):
        env.step(zero_act)

    victim = 0
    rd = env.get_robot_data("robot")
    writer = env.get_robot_state_writer("robot")
    ids = torch.tensor([victim], device=env.device, dtype=torch.long)

    pos = rd.root_link_pos_w[ids].clone()
    pos[:, 2] = _TERMINAL_Z
    writer.set_root_pose(pos, rd.root_link_quat_w[ids].clone(), env_ids=ids)
    writer.eval_fk(ids)
    # mjlab's write_root_link_pose_to_sim lands in qpos directly, but
    # root_link_pos_w reads data.xpos, so MuJoCo needs a forward pass before the
    # read-back. _post_reset_forward() is exactly that on MuJoCo and a no-op on
    # Newton (FK ran above) and Genesis (writes go straight through).
    env._post_reset_forward()
    env._invalidate_cache()

    rd = env.get_robot_data("robot")
    true_z = float(rd.root_link_pos_w[victim, 2])
    true_ep_len = int(env.episode_length_buf[victim])
    print(f"  wrote root z = {_TERMINAL_Z}, read back {true_z:.6f}   (ep_len={true_ep_len})")
    if abs(true_z - _TERMINAL_Z) > _Z_TOL:
        print("\n  A2: INCONCLUSIVE — the teleport did not land, so the state-write path is")
        print("      the problem, not the curriculum ordering.\n")
        return {"status": "INCONCLUSIVE", "reason": "teleport did not land"}

    _RECORD.clear()
    env._reset_idx(ids)
    if "z" not in _RECORD:
        print("\n  A2: FAIL — curriculum_manager.compute() was never called during _reset_idx.\n")
        return {"status": "FAIL", "reason": "compute not called"}

    seen_z = float(_RECORD["z"][0])
    seen_ep_len = int(_RECORD["ep_len"][0])
    delta = abs(seen_z - true_z)

    print(f"\n  {'quantity':<28} {'true (pre-reset)':>18} {'seen by curriculum':>20}")
    print(f"  {'root_link_pos_w[z]':<28} {true_z:>18.6f} {seen_z:>20.6f}")
    print(f"  {'episode_length_buf':<28} {true_ep_len:>18d} {seen_ep_len:>20d}")
    print(f"  {'|delta z|':<28} {'':>18} {delta:>20.3e}")
    print(f"  {'compute calls':<28} {'':>18} {_RECORD['calls']:>20d}")

    ok = delta <= _Z_TOL and seen_ep_len == true_ep_len
    if not ok:
        print("\n  A2: FAIL — a value near the preset's spawn height means the simulator")
        print("      reset ran before curriculum_manager.compute().")
    else:
        print("\n  A2: PASS")
    print()
    return {"status": "PASS" if ok else "FAIL", "true_z": true_z, "seen_z": seen_z, "delta": delta}


# ══════════════════════════════════════════════════════════════════════════
# A1 — Newton clears solver-internal state on reset
# ══════════════════════════════════════════════════════════════════════════


def check_a1(env, steps: int) -> dict:
    import torch
    import warp as wp

    solver = env.scene_manager.solver
    mjw = solver.mjw_data
    num_envs = env.num_envs

    reset_ids = torch.tensor(list(_RESET_ENVS), device=env.device, dtype=torch.long)
    keep_ids = torch.tensor([i for i in range(num_envs) if i not in _RESET_ENVS], device=env.device, dtype=torch.long)

    candidates = {
        "qfrc_applied": wp.to_torch(mjw.qfrc_applied),
        "xfrc_applied": wp.to_torch(mjw.xfrc_applied),
        "ctrl": wp.to_torch(mjw.ctrl),
        "act": wp.to_torch(mjw.act),
    }
    # Zero-width buffers carry no information and would blow up the reductions
    # below. ``ctrl`` is empty (nu=0) whenever the preset drives every joint
    # through an EXPLICIT actuator: Newton then leaves ``joint_target_mode`` at
    # NONE and mjwarp builds no actuators at all. ``act`` is empty unless some
    # actuator is stateful (na>0).
    buffers = {name: buf for name, buf in candidates.items() if buf.numel() > 0}
    empty = {name: tuple(buf.shape) for name, buf in candidates.items() if buf.numel() == 0}
    warmstart = wp.to_torch(mjw.qacc_warmstart)

    print("-" * 92)
    print("A1  solver-internal state cleared on reset")
    print("-" * 92)
    print(f"  reset envs = {list(_RESET_ENVS)}   keep envs = {keep_ids.tolist()}")
    print(
        f"  mj_model nu/na = {int(solver.mj_model.nu)} / {int(solver.mj_model.na)}   "
        f"explicit_actuators = {env.act_manager.has_explicit_actuators}"
    )
    for name, buf in buffers.items():
        print(f"  [buf] {name:<14} shape={tuple(buf.shape)}")
    for name, shape in empty.items():
        print(f"  [buf] {name:<14} shape={shape}  -> SKIPPED (empty)")
    print(f"  [buf] {'qacc_warmstart':<14} shape={tuple(warmstart.shape)}")
    if not buffers:
        print("\n  A1: INCONCLUSIVE — every solver input buffer is empty, so T1 has nothing")
        print("      to check. Run against a preset with implicit actuators.\n")
        return {"status": "INCONCLUSIVE", "reason": "all input buffers empty"}

    # ── T1 + T2: poison every world, reset a subset, inspect ──
    for buf in buffers.values():
        buf.fill_(_SENTINEL)
    warmstart.fill_(_SENTINEL)
    env._reset_idx(reset_ids)

    print(f"\n  T1  {'buffer':<16}{'max|reset|':>18}{'min|keep|':>18}{'cleared':>10}{'kept':>7}")
    t1_ok = True
    for name, buf in buffers.items():
        flat = buf.reshape(buf.shape[0], -1)
        reset_max, keep_min = flat[reset_ids].abs().max(), flat[keep_ids].abs().min()
        cleared, kept = bool(reset_max == 0), bool(keep_min == _SENTINEL)
        t1_ok &= cleared and kept
        print(f"      {name:<16}{_fmt(reset_max):>18}{_fmt(keep_min):>18}{str(cleared):>10}{str(kept):>7}")
    print(f"      -> T1 {'PASS' if t1_ok else 'FAIL'} (want cleared=True kept=True on every row)")

    ws_flat = warmstart.reshape(warmstart.shape[0], -1)
    still_sentinel = bool((ws_flat[reset_ids] == _SENTINEL).any())
    t2_ok = not still_sentinel
    print(f"\n  T2  reset-env qacc_warmstart max|.| = {_fmt(ws_flat[reset_ids].abs().max())}")
    print(f"      reset-env qacc_warmstart == sentinel({_SENTINEL}) anywhere : {still_sentinel}")
    print(f"      keep-env  qacc_warmstart max|.| = {_fmt(ws_flat[keep_ids].abs().max())}")
    print("      NOTE: contact_manager.refresh_after_reset() runs mujoco_warp.forward() for")
    print("            ALL worlds after the clear, and MuJoCo's fwdConstraint rewrites")
    print("            qacc_warmstart from the freshly solved qacc. A non-zero value is")
    print("            expected — what matters is that it is no longer the PRE-reset value.")
    print(f"      -> T2 {'PASS' if t2_ok else 'FAIL'}")

    # ── T3: NaN recovery ──
    env.reset()
    zero_act = torch.zeros(num_envs, env.num_actions, device=env.device)
    victim = 0
    warmstart[victim].fill_(float("nan"))
    print(f"\n  T3  poisoned qacc_warmstart[env {victim}] with NaN")

    env.step(zero_act)
    rd = env.get_robot_data()

    def finite(e: int) -> bool:
        return bool(
            torch.isfinite(rd.joint_pos[e]).all()
            and torch.isfinite(rd.joint_vel[e]).all()
            and torch.isfinite(rd.root_link_pos_w[e]).all()
            and torch.isfinite(rd.root_link_lin_vel_w[e]).all()
        )

    print(
        f"      after 1 step: env {victim} finite={finite(victim)} (want False)   "
        f"others all finite={all(finite(e) for e in range(1, num_envs))} (want True)"
    )
    if finite(victim):
        print("      -> T3 INCONCLUSIVE — the NaN never propagated, so this cannot")
        print("         distinguish fixed from broken.")
        return {"status": "INCONCLUSIVE", "T1": t1_ok, "T2": t2_ok, "T3": "INCONCLUSIVE"}

    env._reset_idx(torch.tensor([victim], device=env.device, dtype=torch.long))
    print(f"      reset env {victim}")
    for i in range(steps):
        env.step(zero_act)
        rd = env.get_robot_data()
        jv = torch.nan_to_num(rd.joint_vel[victim].abs(), nan=float("inf")).max()
        ws = torch.nan_to_num(warmstart[victim].abs(), nan=float("inf")).max()
        print(
            f"        step {i + 1}: finite={finite(victim)}  root_z={_fmt(rd.root_link_pos_w[victim, 2])}  "
            f"max|jv|={_fmt(jv)}  max|ws|={_fmt(ws)}"
        )
    t3_ok = finite(victim)
    print(f"      -> T3 {'PASS' if t3_ok else 'FAIL'} (env {victim} finite after reset)")

    ok = t1_ok and t2_ok and t3_ok
    print(f"\n  A1: {'PASS' if ok else 'FAIL'}\n")
    return {"status": "PASS" if ok else "FAIL", "T1": t1_ok, "T2": t2_ok, "T3": t3_ok}


# ══════════════════════════════════════════════════════════════════════════
# Child driver
# ══════════════════════════════════════════════════════════════════════════


def run_case(case: Case, args) -> dict:
    env = _build_env(case, args.num_envs)
    _print_banner(case, env)

    results: dict[str, dict] = {}
    # Order matters: A5 needs the cleanest state (its momentum cancellation
    # assumes no reset redraws the domain-randomized masses between its two
    # runs), and A1 is destructive (poisons buffers, injects NaN), so it runs
    # last.
    if "A5" in case.checks:
        results["A5"] = check_a5(env, args.force, args.victim)
    if "A2" in case.checks:
        results["A2"] = check_a2(env, args.warmup_steps)
    if "A1" in case.checks:
        results["A1"] = check_a1(env, args.recovery_steps)

    statuses = [r["status"] for r in results.values()]
    if "INCONCLUSIVE" in statuses:
        verdict = "INCONCLUSIVE"
    elif "FAIL" in statuses:
        verdict = "FAIL"
    else:
        verdict = "PASS"

    print("=" * 92)
    print(f"case {case.name}: {verdict}   " + "  ".join(f"{k}={v['status']}" for k, v in results.items()))
    print("=" * 92)
    return {"case": case.name, "verdict": verdict, "checks": results}


# ══════════════════════════════════════════════════════════════════════════
# Parent driver
# ══════════════════════════════════════════════════════════════════════════


def _child_argv(case: Case, args, result_json: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        _MODULE,
        "--case",
        case.name,
        "--result-json",
        str(result_json),
        "--num-envs",
        str(args.num_envs),
        "--force",
        str(args.force),
        "--victim",
        str(args.victim),
        "--warmup-steps",
        str(args.warmup_steps),
        "--recovery-steps",
        str(args.recovery_steps),
    ]


def run_parent(args) -> int:
    cases = [c for c in CASES if args.only is None or c.name in args.only]
    if not cases:
        print(f"No case matches --only {args.only}. Available: {[c.name for c in CASES]}")
        return 2

    summary: list[tuple[Case, str, dict]] = []
    tracebacks: dict[str, str] = {}
    with tempfile.TemporaryDirectory(prefix="reset_path_diag_") as tmp:
        for case in cases:
            out = Path(tmp) / f"{case.name}.json"
            cmd = _child_argv(case, args, out)
            print("\n" + "#" * 92)
            print(f"# $ {' '.join(cmd)}")
            print("#" * 92, flush=True)
            # stdout streams live so progress is visible; stderr is captured so a
            # crashed child's traceback can be replayed under the summary instead
            # of being buried thousands of lines up in the scrollback.
            proc = subprocess.run(cmd, stderr=subprocess.PIPE, text=True)
            if proc.stderr:
                print(proc.stderr, file=sys.stderr, end="")
            if out.exists():
                payload = json.loads(out.read_text())
                summary.append((case, payload["verdict"], payload["checks"]))
            else:
                summary.append((case, f"CRASHED(rc={proc.returncode})", {}))
                tracebacks[case.name] = proc.stderr or "<no stderr captured>"

    print("\n" + "=" * 92)
    print("SUMMARY")
    print("=" * 92)
    print(f"{'case':<22}{'sim':<9}{'expected':<16}{'observed':<16}{'per-check'}")
    print("-" * 92)
    all_ok = True
    for case, verdict, checks in summary:
        expected = "PASS" if case.expect_pass else "FAIL (control)"
        per_check = "  ".join(f"{k}={v['status']}" for k, v in checks.items()) or "-"
        matched = verdict == ("PASS" if case.expect_pass else "FAIL")
        all_ok &= matched
        flag = "" if matched else "   <-- UNEXPECTED"
        print(f"{case.name:<22}{case.sim:<9}{expected:<16}{verdict:<16}{per_check}{flag}")
    print("-" * 92)
    print("Control rows are expected to FAIL: they restore the old code path. A control row")
    print("that PASSes means the check does not discriminate, so the fixed row above it is")
    print("not evidence of anything.")

    if tracebacks:
        print("\n" + "=" * 92)
        print("CRASHED CASES — last 30 stderr lines each")
        print("=" * 92)
        for name, err in tracebacks.items():
            print(f"\n--- {name} " + "-" * (88 - len(name)))
            tail = [ln for ln in err.splitlines() if ln.strip()][-30:]
            print("\n".join(tail) if tail else "<empty>")

    print(f"\nOVERALL: {'PASS' if all_ok else 'FAIL'}")
    print("=" * 92)
    return 0 if all_ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--case",
        choices=sorted(_BY_NAME),
        default=None,
        help="Run a single case inline (child mode). Omit to run all cases as subprocesses.",
    )
    ap.add_argument("--only", nargs="+", default=None, help="Parent mode: subset of case names to run.")
    ap.add_argument("--result-json", default=None, help="Child mode: where to write this case's result.")
    ap.add_argument("--num-envs", type=int, default=8)
    ap.add_argument("--force", type=float, default=500.0, help="A5: constant +z force [N] on the base link.")
    ap.add_argument("--victim", type=int, default=0, help="A5: env index the wrench is applied to.")
    ap.add_argument("--warmup-steps", type=int, default=3, help="A2: steps before the teleport.")
    ap.add_argument("--recovery-steps", type=int, default=5, help="A1 T3: steps after the reset.")
    args = ap.parse_args()

    if args.case is None:
        return run_parent(args)

    payload = run_case(_BY_NAME[args.case], args)
    if args.result_json:
        Path(args.result_json).write_text(json.dumps(payload, default=str))
    return 0 if payload["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
