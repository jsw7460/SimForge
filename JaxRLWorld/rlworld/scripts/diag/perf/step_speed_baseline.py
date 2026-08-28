"""Step-speed baseline, for before/after comparison across a refactor.

A standalone measurement, deliberately not framework instrumentation: the
env-var-gated profilers that used to live inside the managers were removed
on purpose, and nothing here runs unless this script is invoked.

Prints milliseconds per control step and per reset for a preset, so a
refactor that claims "no performance change" can be held to a number. Run
it before the change, keep the output, run it again after.

    python -m rlworld.scripts.diag.perf.step_speed_baseline --num-envs 4096

or one backend, or another preset:

    python -m rlworld.scripts.diag.perf.step_speed_baseline --sim newton --num-envs 4096
    python -m rlworld.scripts.diag.perf.step_speed_baseline --preset yam_lift --num-envs 4096

or with a solver budget the presets do not carry, to find out whether the
budget is what a cross-backend gap is made of:

    python -m rlworld.scripts.diag.perf.step_speed_baseline --preset yam_lift --iterations 10 --ls-iterations 20

Episode ends are spread out by default, so the timed loop pays for the
terminations and resets a training loop pays for. Without that the numbers
flatter any preset whose episodes outlast the measurement: go2's are 1000
steps and this runs 200, which read 13.0 ms against the 17.7 the trainer
sees. ``--no-stagger`` restores the old behaviour.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch

from rlworld.rl.configs.presets.go2.base import Go2FlatConfig
from rlworld.rl.configs.presets.k1_joystick.base import K1JoystickConfig
from rlworld.rl.configs.presets.k1_joystick.g1_recipe import K1G1RecipeConfig
from rlworld.rl.configs.presets.yam_lift.base import YamLiftConfig
from rlworld.rl.runners import BaseRunner

_SIMS = ("genesis", "newton", "mujoco")

# Preset by name. A cross-backend gap is a property of the SCENE, not of
# the framework: a locomotion robot on flat ground and an arm holding a
# contact-rich grasp stress entirely different parts of a simulator, so a
# ratio measured on one says nothing about the other.
#
# The three here span the interesting axis, which is NOT joint count: a
# 23-DOF humanoid, a 12-DOF quadruped and a 7-DOF arm. If the arm is the
# slowest of the three, the time is going somewhere other than the
# articulation — the scene it sits in, or the terms read off it.
_PRESETS = {
    "go2": Go2FlatConfig,
    "k1_joystick": K1JoystickConfig,
    # The config the K1 training scripts actually run. Its scene and
    # timing are inherited unchanged, so physics numbers match
    # k1_joystick; the policy, reward and DR cadence differ.
    "k1_g1_recipe": K1G1RecipeConfig,
    "yam_lift": YamLiftConfig,
}


def _sync() -> None:
    """Block until the GPU has actually finished the queued work.

    Without this every launch is timed as ~0 and the total lands on
    whichever call happens to synchronize.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _override_solver_iterations(sim: str, scene, iterations: int, ls_iterations: int) -> None:
    """Set the solver's iteration budget, whatever the backend calls it.

    The three presets carry different budgets — mujoco 10/20 (mjlab's own
    numbers for this task), genesis 30/40, newton 50/50 — and the budget
    is the first thing to rule in or out when one backend is slower than
    another. Overriding it here answers that with a measurement instead of
    an edit to three preset files that then has to be undone.
    """
    if sim == "newton":
        scene.solver_cfg.iterations = iterations
        scene.solver_cfg.ls_iterations = ls_iterations
    elif sim == "mujoco":
        scene.solver_iterations = iterations
        scene.solver_ls_iterations = ls_iterations
    elif sim == "genesis":
        scene.rigid_options.iterations = iterations
        scene.rigid_options.ls_iterations = ls_iterations
    else:
        raise ValueError(f"Unknown sim {sim!r}")


def _effective_solver_iterations(sim: str, env) -> tuple[int, int]:
    """The iteration budget the LIVE simulator ended up with.

    Read from the built solver, never from the config that was handed to
    it. A config field that no longer reaches the backend still reads back
    fine, and the timing would then be flat for a reason that has nothing
    to do with physics — a null result and a broken knob look identical
    from the outside.
    """
    if sim == "newton":
        opt = env.scene_manager.solver.mj_model.opt
        return int(opt.iterations), int(opt.ls_iterations)
    if sim == "mujoco":
        opt = env.scene_manager.sim.mj_model.opt
        return int(opt.iterations), int(opt.ls_iterations)
    if sim == "genesis":
        constraint_solver = env.scene_manager.scene.rigid_solver.constraint_solver
        return int(constraint_solver._n_iterations), int(constraint_solver.ls_iterations)
    raise ValueError(f"Unknown sim {sim!r}")


def _apply_rigid_overrides(sim: str, scene, overrides: list[str]) -> dict:
    """Set Genesis ``RigidOptions`` fields named on the command line.

    The collision and solver knobs a preset leaves unset default to
    Genesis's own values, and finding which of them a slow scene is
    paying for means changing one at a time. Doing that here keeps the
    preset — which several other runs depend on — untouched between
    measurements.

    Values are parsed as Python literals, so ``None``, ``True``, ``150``
    and ``0.02`` all arrive as the right type, and the field must already
    exist: a typo raises rather than being silently ignored.
    """
    if not overrides:
        return {}
    if sim != "genesis":
        raise ValueError(f"--rigid-option applies to genesis rigid_options, not {sim!r}.")
    applied = {}
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Expected FIELD=VALUE, got {override!r}.")
        field, _, raw = override.partition("=")
        field = field.strip()
        if not hasattr(scene.rigid_options, field):
            raise ValueError(f"RigidOptions has no field {field!r}.")
        value = ast.literal_eval(raw.strip())
        setattr(scene.rigid_options, field, value)
        applied[field] = value
    return applied


def run_single(
    preset: str,
    sim: str,
    num_envs: int,
    warmup: int,
    steps: int,
    resets: int,
    iterations: int | None = None,
    ls_iterations: int | None = None,
    rigid_options: list[str] | None = None,
    stagger: bool = True,
) -> dict:
    config = _PRESETS[preset](sim_type=sim, num_envs=num_envs)
    cfgs = config.build()
    if (iterations is None) != (ls_iterations is None):
        raise ValueError("Pass --iterations and --ls-iterations together; one alone is half an experiment.")
    if iterations is not None:
        _override_solver_iterations(sim, cfgs.scene, iterations, ls_iterations)
    applied_rigid = _apply_rigid_overrides(sim, cfgs.scene, rigid_options or [])
    env = BaseRunner._create_env_from_config(cfgs)

    live_iterations, live_ls_iterations = _effective_solver_iterations(sim, env)
    if iterations is not None and (live_iterations, live_ls_iterations) != (iterations, ls_iterations):
        raise RuntimeError(
            f"[{sim}] asked for {iterations}/{ls_iterations} but the solver runs "
            f"{live_iterations}/{live_ls_iterations}. The override did not reach the backend, so any "
            "timing measured here would say nothing about the iteration budget."
        )

    env.reset()

    # Spread the episode ends out, as ``learn(init_at_random_ep_len=)`` does.
    # Without it this measures only the cheap case: straight after a reset
    # every environment is the same age, so nothing terminates for a whole
    # episode and the step never pays for a terminal observation, a reset or
    # a post-reset forward. A preset with 20 s episodes at 50 Hz needs 1000
    # steps before its first timeout, and this loop runs 200 — so go2 read
    # 13.0 ms here against 17.7 in the loop training actually runs.
    if stagger:
        env.termination_manager.episode_length_buf = torch.randint_like(
            env.episode_length_buf, high=int(env.max_episode_length)
        )

    n_act = env.act_manager.num_actions
    torch.manual_seed(0)
    action = torch.empty(env.num_envs, n_act, device=env.device).uniform_(-0.3, 0.3)

    for _ in range(warmup):
        env.step(action)
    _sync()

    # Per-step samples rather than one total: a single mean hides a
    # periodic stall, which is exactly what a manager-loop regression
    # would look like.
    per_step: list[float] = []
    for _ in range(steps):
        t0 = time.perf_counter()
        env.step(action)
        _sync()
        per_step.append((time.perf_counter() - t0) * 1e3)

    all_ids = torch.arange(env.num_envs, device=env.device)
    per_reset: list[float] = []
    for _ in range(resets):
        t0 = time.perf_counter()
        env._reset_idx(all_ids)
        _sync()
        per_reset.append((time.perf_counter() - t0) * 1e3)

    result = {
        "preset": preset,
        "sim": sim,
        "num_envs": num_envs,
        "iterations": live_iterations,
        "ls_iterations": live_ls_iterations,
        "step_mean_ms": round(statistics.mean(per_step), 4),
        "step_median_ms": round(statistics.median(per_step), 4),
        "step_p95_ms": round(sorted(per_step)[int(0.95 * len(per_step))], 4),
        "step_max_ms": round(max(per_step), 4),
        "reset_mean_ms": round(statistics.mean(per_reset), 4),
        "num_actions": n_act,
    }

    print("=" * 78)
    # The LIVE budget, so the header cannot claim a setting the solver
    # is not running.
    budget = f"{live_iterations}/{live_ls_iterations}"
    episodes = "staggered" if stagger else "all fresh"
    print(
        f"STEP SPEED  [preset={preset}  sim={sim}  num_envs={num_envs}  actions={n_act}  "
        f"solver={budget}  episodes={episodes}]"
    )
    if applied_rigid:
        print(f"  rigid_options overridden: {applied_rigid}")
    print("=" * 78)
    print(f"  step   mean {result['step_mean_ms']:8.3f} ms   median {result['step_median_ms']:8.3f} ms")
    print(f"         p95  {result['step_p95_ms']:8.3f} ms   max    {result['step_max_ms']:8.3f} ms")
    print(f"  reset  mean {result['reset_mean_ms']:8.3f} ms   over {resets} full resets")
    print("=" * 78)
    return result


def run_all(
    preset: str,
    num_envs: int,
    warmup: int,
    steps: int,
    resets: int,
    iterations: int | None,
    ls_iterations: int | None,
    rigid_options: list[str],
    stagger: bool,
) -> int:
    tmp = Path(tempfile.mkdtemp(prefix="step_speed_"))
    out: dict[str, dict] = {}
    env_vars = dict(os.environ, JAXRLWORLD_ALLOW_MULTI_SIM="1")

    for sim in _SIMS:
        path = tmp / f"{sim}.json"
        cmd = [
            sys.executable,
            "-m",
            "rlworld.scripts.diag.perf.step_speed_baseline",
            "--preset",
            preset,
            "--sim",
            sim,
            "--result-json",
            str(path),
            "--num-envs",
            str(num_envs),
            "--warmup",
            str(warmup),
            "--steps",
            str(steps),
            "--resets",
            str(resets),
        ]
        if iterations is not None:
            cmd += ["--iterations", str(iterations), "--ls-iterations", str(ls_iterations)]
        for override in rigid_options:
            cmd += ["--rigid-option", override]
        if not stagger:
            cmd += ["--no-stagger"]
        print()
        print("#" * 78)
        print(f"# $ {' '.join(cmd)}")
        print("#" * 78)
        subprocess.run(cmd, env=env_vars, check=False)
        if path.exists():
            out[sim] = json.loads(path.read_text())

    if not out:
        print("No backend produced a result.")
        return 1

    print()
    print("=" * 78)
    print(f"BASELINE SUMMARY  (preset={preset}  num_envs={num_envs})")
    print("=" * 78)
    print(f"{'metric':<20}" + "".join(f"{s:>14}" for s in _SIMS))
    print("-" * 78)
    # The budget each backend actually ran at, so a flat comparison cannot
    # be read as "the budget does not matter" when it never changed.
    row = f"{'solver_iters':<20}"
    for sim in _SIMS:
        result = out.get(sim)
        cell = "—" if result is None else f"{result['iterations']}/{result['ls_iterations']}"
        row += f"{cell:>14}"
    print(row)
    for key in ("step_mean_ms", "step_median_ms", "step_p95_ms", "step_max_ms", "reset_mean_ms"):
        row = f"{key:<20}"
        for s in _SIMS:
            v = out.get(s, {}).get(key)
            row += f"{'—' if v is None else f'{v:.3f}':>14}"
        print(row)
    print("=" * 78)
    print("Keep this table. Re-run after the change and compare.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", choices=list(_SIMS), default=None, help="Run one backend. Omit to run all three.")
    ap.add_argument("--preset", choices=list(_PRESETS), default="go2")
    ap.add_argument("--result-json", default=None)
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--resets", type=int, default=20)
    ap.add_argument(
        "--no-stagger",
        action="store_true",
        help="Start every episode together, so the timed loop never resets.",
    )
    ap.add_argument("--iterations", type=int, default=None, help="Override the solver iteration budget.")
    ap.add_argument("--ls-iterations", type=int, default=None, help="Override the line-search budget.")
    ap.add_argument(
        "--rigid-option",
        action="append",
        default=[],
        metavar="FIELD=VALUE",
        help="Genesis RigidOptions field to override, repeatable (e.g. box_box_detection=False).",
    )
    args = ap.parse_args()

    if args.sim is None:
        return run_all(
            args.preset,
            args.num_envs,
            args.warmup,
            args.steps,
            args.resets,
            args.iterations,
            args.ls_iterations,
            args.rigid_option,
            not args.no_stagger,
        )

    result = run_single(
        args.preset,
        args.sim,
        args.num_envs,
        args.warmup,
        args.steps,
        args.resets,
        args.iterations,
        args.ls_iterations,
        args.rigid_option,
        not args.no_stagger,
    )
    if args.result_json:
        Path(args.result_json).write_text(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
