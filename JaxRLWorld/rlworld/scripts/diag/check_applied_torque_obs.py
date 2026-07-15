"""Verify the ``applied_torque`` observation for BOTH actuator modes.

Supersedes ``applied_torque_smoke.py`` (explicit-only, nonzero/ballpark
checks) with value-level verification. Builds the Go2 preset twice on
the selected backend — once with its explicit IdealPD actuators, once
with the actuators swapped for ``ImplicitActuatorCfg`` carrying the
SAME stiffness/damping/effort limits — and checks that the
``applied_torque`` observation term (→ ``RobotData.applied_torque``,
MuJoCo ``qfrc_actuator`` semantics) returns the physically correct
torque in each mode:

 1. obs term identity — the term returns exactly
    ``RobotData.applied_torque``, shaped ``(num_envs, num_actions)``
 2. settled-state PD formula — after settling under a zero action
    (target = default pose), per-joint
    ``tau = kp * (target - q) - kd * qd`` must match the observation
    (state is stationary, so substep-timing skew vanishes)
 3. effort-limit clipping — under a saturating action the observed
    torque must respect the per-joint effort limit in both modes
 4. cross-mode agreement — implicit and explicit runs settle to the
    same stance, so their observed torques must agree per joint
 5. finiteness / non-triviality throughout

DR events are stripped from both builds so randomized PD gains cannot
invalidate the hand-computed formula.

A single invocation covers ALL backends: the backends must stay
import-isolated (one sim per process), so the no-argument run spawns
one child process per simulator and collects everything into a log
file (default ``applied_torque_obs_diag.txt``), printing only per-sim
PASS/FAIL to the console.

Run once (GPU box):
    jaxpy rlworld/scripts/diag/check_applied_torque_obs.py
"""

from __future__ import annotations

import argparse
import contextlib
import os
import re
import subprocess
import sys
import tempfile

# Diag runs never log to wandb; set before any rlworld import so the
# runner's wandb.init() becomes a no-op.
os.environ.setdefault("WANDB_MODE", "disabled")

import torch

from rlworld.rl.actuators import ImplicitActuatorCfg
from rlworld.rl.configs.common_config_classes import EventConfig
from rlworld.rl.configs.presets.go2.base import Go2FlatConfig
from rlworld.rl.envs.mdp.observations.common.proprioception import applied_torque
from rlworld.rl.runners import BaseRunner

SETTLE_STEPS = 150
FORMULA_TOL_NM = 0.5
CROSS_MODE_TOL_NM = 1.0
CLIP_TOL_NM = 1e-3
# Raw action for the effort-limit test. After scale (~0.17-0.33) this puts
# the position target several radians outside the reachable joint range, so
# the PD error cannot be tracked away within the decimation window and the
# torque stays pinned at the effort limit at every substep.
SAT_ACTION = 30.0


@contextlib.contextmanager
def _quiet():
    """Silence env-construction noise (warp/Genesis banners, rich tables).

    File-descriptor-level redirect so C-extension prints are caught too.
    The captured text is replayed only if construction fails.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    saved_out, saved_err = os.dup(1), os.dup(2)
    with tempfile.TemporaryFile(mode="w+") as tmp:
        os.dup2(tmp.fileno(), 1)
        os.dup2(tmp.fileno(), 2)
        try:
            yield
        except BaseException:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved_out, 1)
            os.dup2(saved_err, 2)
            tmp.seek(0)
            print("--- captured build output (construction failed) ---")
            print(tmp.read())
            raise
        else:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved_out, 1)
            os.dup2(saved_err, 2)
        finally:
            os.close(saved_out)
            os.close(saved_err)


failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(f"{label}: {detail}")


def _resolve_per_joint(value, patterns: tuple[str, ...], joint_names: list[str], device) -> torch.Tensor:
    """Expand a float | dict[str,float] actuator-cfg field to per-joint values.

    A joint gets the value only if it matches this actuator group's
    ``target_names_expr``; the caller merges groups.
    """
    out = torch.full((len(joint_names),), float("nan"), device=device)
    for j, name in enumerate(joint_names):
        bare = name.rsplit("/", 1)[-1]
        if not any(re.fullmatch(p, bare) or re.fullmatch(p, name) for p in patterns):
            continue
        if isinstance(value, dict):
            hits = {v for pat, v in value.items() if re.fullmatch(pat, bare) or re.fullmatch(pat, name)}
            if len(hits) > 1:
                raise ValueError(f"ambiguous per-joint value for {name}: {hits}")
            out[j] = hits.pop() if hits else float("nan")
        elif value is not None:
            out[j] = float(value)
    return out


def _merge_groups(cfgs, field: str, joint_names: list[str], device) -> torch.Tensor:
    merged = torch.full((len(joint_names),), float("nan"), device=device)
    for a in cfgs:
        vals = _resolve_per_joint(getattr(a, field), tuple(a.target_names_expr), joint_names, device)
        mask = ~torch.isnan(vals)
        merged[mask] = vals[mask]
    if torch.isnan(merged).any():
        missing = [joint_names[i] for i in torch.isnan(merged).nonzero().flatten().tolist()]
        raise ValueError(f"actuator cfgs leave {field} unresolved for joints: {missing}")
    return merged


def build_env(sim: str, implicit: bool):
    cfg = Go2FlatConfig(sim_type=sim, num_envs=2, use_ideal_pd_actuator=True)
    with _quiet():
        cfgs = cfg.build()  # imports the sim backend (warp init banner etc.)
        # Strip every event term (incl. reset_dr PD-gain randomization) so
        # the hand-computed PD formula below sees the nominal gains.
        cfgs.event = EventConfig()

        entity = cfgs.scene.entities["robot"]
        actuator_cfgs = tuple(entity.articulation.actuators)
        if implicit:
            entity.articulation.actuators = tuple(
                ImplicitActuatorCfg(
                    target_names_expr=a.target_names_expr,
                    stiffness=a.stiffness,
                    damping=a.damping,
                    effort_limit=a.effort_limit,
                    velocity_limit=a.velocity_limit,
                    armature=a.armature,
                    frictionloss=a.frictionloss,
                )
                for a in actuator_cfgs
            )
        env = BaseRunner.create_with_env(cfgs).env
        env.reset()
    return env, entity.articulation.actuators


def run_mode(sim: str, implicit: bool) -> dict:
    mode = "implicit" if implicit else "explicit"
    print(f"\n=== mode={mode} " + "=" * 50)
    env, actuator_cfgs = build_env(sim, implicit)
    expected_explicit = not implicit
    check(
        f"{mode}: act_manager mode as intended",
        env.act_manager.has_explicit_actuators == expected_explicit,
        f"has_explicit={env.act_manager.has_explicit_actuators}",
    )

    joint_names = list(env.act_manager.actuated_joint_names)
    device = env.device
    kp = _merge_groups(actuator_cfgs, "stiffness", joint_names, device)
    kd = _merge_groups(actuator_cfgs, "damping", joint_names, device)
    effort = _merge_groups(actuator_cfgs, "effort_limit", joint_names, device)
    print(
        f"  joints({len(joint_names)}): kp range [{kp.min():.1f}, {kp.max():.1f}], "
        f"kd range [{kd.min():.2f}, {kd.max():.2f}], effort range [{effort.min():.1f}, {effort.max():.1f}]"
    )

    # ── 1. settle under zero action (target = default pose) ─────────
    actions = torch.zeros((env.num_envs, env.num_actions), device=device)
    for _ in range(SETTLE_STEPS):
        env.step(actions)

    rd = env.get_robot_data("robot")
    tau_obs = applied_torque(env)
    check(f"{mode}: obs term == RobotData.applied_torque", torch.equal(tau_obs, rd.applied_torque))
    check(
        f"{mode}: shape == (num_envs, num_actions)",
        tuple(tau_obs.shape) == (env.num_envs, env.num_actions),
        f"{tuple(tau_obs.shape)}",
    )
    check(f"{mode}: finite", bool(torch.isfinite(tau_obs).all()))
    check(
        f"{mode}: non-trivial under gravity load (|mean| > 0.1 Nm)",
        bool(tau_obs.abs().mean() > 0.1),
        f"{tau_obs.abs().mean():.4f}",
    )

    # ── 2. settled-state PD formula ──────────────────────────────────
    q = rd.joint_pos
    qd = rd.joint_vel
    target = env.act_manager.offset  # zero action → target = offset (default pose)
    tau_expected = (kp * (target - q) - kd * qd).clamp(-effort, effort)
    err = (tau_obs - tau_expected).abs()
    qd_norm = qd.abs().max()
    print(f"  settled |qd|max = {qd_norm:.4f} rad/s (formula check assumes stationarity)")
    print(f"  {'joint':<22}{'tau_obs':>10}{'tau_pd':>10}{'|err|':>9}")
    for j, name in enumerate(joint_names):
        print(f"  {name.rsplit('/', 1)[-1]:<22}{tau_obs[0, j]:>10.3f}{tau_expected[0, j]:>10.3f}{err[0, j]:>9.4f}")
    check(
        f"{mode}: settled torque matches PD formula (max err < {FORMULA_TOL_NM} Nm)",
        bool(err.max() < FORMULA_TOL_NM),
        f"max err {err.max():.4f}",
    )

    # ── 3. effort-limit clipping under saturating action ────────────
    for _ in range(3):
        env.step(torch.full_like(actions, SAT_ACTION))
    tau_sat = applied_torque(env)
    over = (tau_sat.abs() - effort).max()
    print(f"  saturated: |tau|max per joint vs limit — worst overshoot {over:.5f} Nm")
    check(f"{mode}: |tau| respects per-joint effort limit", bool(over <= CLIP_TOL_NM), f"overshoot {over:.5f}")
    check(
        f"{mode}: saturation actually reached on some joint",
        bool(((tau_sat.abs() - effort).abs() < 1e-2).any()),
        "no joint near its limit — clipping untested",
    )

    settled = {"tau": tau_obs.detach().clone(), "q": q.detach().clone(), "names": joint_names}
    return settled


def run_single(sim: str) -> int:
    """Run both actuator modes + cross-mode checks for one backend."""
    print("=" * 74)
    print(f"APPLIED-TORQUE OBSERVATION DIAG  [sim={sim}]")
    print("=" * 74)

    explicit = run_mode(sim, implicit=False)
    implicit = run_mode(sim, implicit=True)

    print("\n=== cross-mode agreement " + "=" * 40)
    q_diff = (explicit["q"] - implicit["q"]).abs().max()
    tau_diff = (explicit["tau"] - implicit["tau"]).abs().max()
    print(f"  settled stance diff |dq|max = {q_diff:.4f} rad; torque diff |dtau|max = {tau_diff:.4f} Nm")
    check("both modes settle to the same stance (|dq| < 0.05 rad)", bool(q_diff < 0.05), f"{q_diff:.4f}")
    check(
        f"cross-mode torque agreement (|dtau| < {CROSS_MODE_TOL_NM} Nm)",
        bool(tau_diff < CROSS_MODE_TOL_NM),
        f"{tau_diff:.4f}",
    )

    print("\n=== Result " + "=" * 55)
    if failures:
        print(f"  {len(failures)} FAILURE(S):")
        for f in failures:
            print(f"   - {f}")
        return 1
    print("  ALL CHECKS PASSED")
    return 0


def run_all(out_path: str) -> int:
    """Run every backend as a subprocess and collect output into one file.

    One process per simulator is mandatory: the backends are kept
    import-isolated (Genesis / warp / mjwarp cannot share a process),
    which is why this dispatcher re-invokes the script with ``--sim``.
    """
    sims = ["newton", "mujoco", "genesis"]
    results: dict[str, bool] = {}
    with open(out_path, "w") as f:
        for sim in sims:
            print(f"[{sim}] running (output -> {out_path}) ...", flush=True)
            proc = subprocess.run(
                [sys.executable, os.path.abspath(__file__), "--sim", sim],
                capture_output=True,
                text=True,
            )
            f.write(f"\n{'#' * 74}\n# sim = {sim} (exit {proc.returncode})\n{'#' * 74}\n")
            f.write(proc.stdout)
            if proc.stderr:
                f.write(f"\n--- stderr ({sim}) ---\n{proc.stderr}")
            f.flush()
            results[sim] = proc.returncode == 0
            print(f"[{sim}] {'PASS' if results[sim] else 'FAIL'}", flush=True)

    print("-" * 40)
    for sim, ok in results.items():
        print(f"  {sim:10s}: {'PASS' if ok else 'FAIL'}")
    ok_all = all(results.values())
    print(f"  OVERALL   : {'PASS' if ok_all else 'FAIL'}")
    print(f"  full output: {out_path}")
    return 0 if ok_all else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sim",
        choices=["genesis", "newton", "mujoco"],
        default=None,
        help="Internal per-backend child mode; omit to run ALL backends and write a log file.",
    )
    ap.add_argument("--out", default="applied_torque_obs_diag.txt", help="Log file for the all-backends run.")
    args = ap.parse_args()

    if args.sim is None:
        return run_all(args.out)
    return run_single(args.sim)


if __name__ == "__main__":
    raise SystemExit(main())
