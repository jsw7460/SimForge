"""mjlab per-env DR — training-faithful matrix diagnostic.

Verifies the setup-time model-field expansion (``MjlabEnv._pre_manager_setup``,
called between scene build and manager creation, mirroring mjlab's own
``ManagerBasedRlEnv`` ordering) under the EXACT conditions where the previous
``_post_setup`` placement crashed with ``CUDA_ERROR_ILLEGAL_ADDRESS``:

    - preset:   flat vs rough (terrain heightfield)
    - jax:      policy stack initialized via BaseRunner (on) vs env-only (off)
    - num_envs: 8 vs 256

Each cell runs in its OWN subprocess (a CUDA illegal access poisons the
process context, so cells must be isolated) with ``CUDA_LAUNCH_BLOCKING=1``
so crash stacktraces point at the faulting launch.  Per cell:

    1. build config -> build env (+ optionally the full JAX runner, which
       resets the env exactly like training does)
    2. check every nominated mjlab field is expanded to ``(num_envs, ...)``
    3. check per-env variance (env-axis std > 0) after the reset-time DR
    4. run N control steps (policy actions on the jax path, random actions
       otherwise) and check observations stay finite
    5. reset again (a second pass through the reset-time DR write path)

The aggregated report is written to a plain .txt file; raw per-cell logs
(including warp/mujoco-warp noise and crash tracebacks) go to a sibling
``*_logs/`` directory.  The parent prints only the report.

Usage (GPU box):
    python -m rlworld.scripts.diag.mjlab_dr_training_matrix_diag
    python -m rlworld.scripts.diag.mjlab_dr_training_matrix_diag --cells rough:on:256
    python -m rlworld.scripts.diag.mjlab_dr_training_matrix_diag --num-steps 50

num_envs stays far below the GPU box's 8192 OOM ceiling by design.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_MODULE = "rlworld.scripts.diag.mjlab_dr_training_matrix_diag"

_PRESETS = ("flat", "rough")
_JAX = ("on", "off")
_NUM_ENVS = (8, 256)

# Unified DR function name -> the mjlab wp_model field(s) it writes.  Must
# stay in sync with ``MjlabEnv._pre_manager_setup`` and the
# ``_mujoco_*_backend`` implementations in ``events/dr/unified.py``.
_UNIFIED_TERM_FIELDS: dict[str, tuple[str, ...]] = {
    "randomize_friction": ("geom_friction",),
    "randomize_body_mass": ("body_mass",),
    "randomize_body_com_offset": ("body_ipos",),
    "randomize_pd_gains": ("actuator_gainprm", "actuator_biasprm"),
    "randomize_joint_armature": ("dof_armature",),
    "randomize_joint_friction": ("dof_frictionloss",),
}


# ── Child: run one cell ─────────────────────────────────────────────


def _stage(msg: str) -> None:
    print(f"[STAGE] {msg}", flush=True)


def _build_cfgs(preset: str, num_envs: int, seed: int):
    if preset == "flat":
        from rlworld.rl.configs.presets.g1_29dof.base import G1FlatConfig

        return G1FlatConfig(sim_type="mujoco", num_envs=num_envs, seed=seed).build()
    from rlworld.rl.configs.presets.g1_29dof.mujoco.rough import G1RoughMujocoConfig

    cfgs = G1RoughMujocoConfig().build()
    cfgs.apply_overrides(**{"env": {"num_envs": num_envs, "seed": seed}})
    return cfgs


def _used_fields(env) -> list[str]:
    """mjlab fields the preset's unified DR terms write."""
    from rlworld.rl.configs.base_config import iter_terms
    from rlworld.rl.configs.events.event_term_config import EventTermConfig

    fields: set[str] = set()
    for _name, term in iter_terms(env.event_cfg, EventTermConfig).items():
        fn = term.func
        mod = getattr(fn, "__module__", "") or ""
        name = getattr(fn, "__name__", "") or ""
        if mod.endswith(".dr.unified") and name in _UNIFIED_TERM_FIELDS:
            fields.update(_UNIFIED_TERM_FIELDS[name])
    return sorted(fields)


def run_cell(preset: str, jax_on: bool, num_envs: int, num_steps: int, seed: int) -> dict:
    import torch
    import warp as wp

    _stage(f"cell start: preset={preset} jax={'on' if jax_on else 'off'} num_envs={num_envs}")
    torch.manual_seed(seed)

    cfgs = _build_cfgs(preset, num_envs, seed)
    _stage("config built")

    if jax_on:
        # Full training entry: builds the env, initializes the JAX policy
        # stack, and calls env.reset() — identical to rough.py training.
        from rlworld.rl.runners import BaseRunner

        runner = BaseRunner.create_with_env(cfgs, use_wandb=False, seed=seed)
        env = runner.env
        _stage("runner built (env + JAX policy) + reset done")
    else:
        from rlworld.rl.evals.sim_initializers.mjlab import MujocoInitializer

        runner = None
        env = MujocoInitializer().init_environment(cfgs)
        _stage("env built (no JAX)")
        env.reset()
        _stage("reset done")

    sim = env.scene_manager.sim
    fields = _used_fields(env)
    expand: dict[str, bool] = {}
    variance: dict[str, float] = {}
    for f in fields:
        arr = getattr(sim.wp_model, f)
        expand[f] = int(arr.shape[0]) == num_envs
        t = wp.to_torch(arr).detach().float()
        flat = t.reshape(t.shape[0], -1)
        variance[f] = float(flat.std(dim=0).mean().item()) if t.shape[0] > 1 else 0.0
    _stage(f"expand/variance checks done (fields={fields})")

    obs = env.obs_manager.get_observation()
    obs_finite = True
    if jax_on:
        from rlworld.rl.algorithms.base import ActInput
        from rlworld.rl.utils.jax_utils import jax_to_torch, torch_to_jax

        for k in range(num_steps):
            actions = runner.alg.act(
                ActInput(torch_to_jax(obs["actor"]), torch_to_jax(obs["critic"])),
                deterministic=False,
            )
            a = jax_to_torch(runner._process_action_for_env(actions), env.device)
            obs, _rew, _term, _trunc, _infos = env.step(a)
            obs_finite &= bool(torch.isfinite(obs["actor"]).all().item())
    else:
        for k in range(num_steps):
            a = torch.empty((num_envs, env.num_actions), device=env.device).uniform_(-1.0, 1.0)
            obs, _rew, _term, _trunc, _infos = env.step(a)
            obs_finite &= bool(torch.isfinite(obs["actor"]).all().item())
    _stage(f"{num_steps} control steps done (obs_finite={obs_finite})")

    # Second full reset: another pass through the reset-time DR writes,
    # now after CUDA-graph replays from the step loop.
    env.reset()
    torch.cuda.synchronize()
    _stage("second reset done")

    return {
        "preset": preset,
        "jax": "on" if jax_on else "off",
        "num_envs": num_envs,
        "fields": fields,
        "expand": expand,
        "expand_ok": all(expand.values()) if expand else None,
        "variance": variance,
        "variance_ok": all(v > 0.0 for v in variance.values()) if variance else None,
        "steps_done": num_steps,
        "obs_finite": obs_finite,
        "second_reset": True,
    }


# ── Parent: orchestrate the matrix ──────────────────────────────────


def _cell_verdict(data: dict | None, returncode: int) -> tuple[str, str]:
    """(verdict, detail)"""
    if data is None or returncode != 0:
        return "FAIL", "crashed (see log)"
    if not data["fields"]:
        return "PASS", "no unified mujoco DR terms in preset"
    problems = []
    if data["expand_ok"] is False:
        bad = [f for f, ok in data["expand"].items() if not ok]
        problems.append(f"not expanded: {bad}")
    if data["variance_ok"] is False:
        bad = [f for f, v in data["variance"].items() if v <= 0.0]
        problems.append(f"env-shared (std=0): {bad}")
    if not data["obs_finite"]:
        problems.append("non-finite observations during steps")
    if problems:
        return "FAIL", "; ".join(problems)
    stds = ", ".join(f"{f}={v:.2e}" for f, v in data["variance"].items())
    return "PASS", stds


def _last_stage(log_path: Path) -> str:
    stage = "(none)"
    for line in log_path.read_text(errors="replace").splitlines():
        if line.startswith("[STAGE] "):
            stage = line[len("[STAGE] ") :]
    return stage


def _error_excerpt(log_path: Path, n: int = 12) -> list[str]:
    lines = log_path.read_text(errors="replace").splitlines()
    return lines[-n:]


def _axis_analysis(results: dict[str, str]) -> list[str]:
    """For each axis, find cell pairs differing only in that axis with
    opposite verdicts — those axes are the failure triggers."""
    out: list[str] = []
    axes = {"preset": _PRESETS, "jax": _JAX, "num_envs": [str(n) for n in _NUM_ENVS]}
    for axis_idx, (axis_name, values) in enumerate(axes.items()):
        flips = []
        for spec, verdict in results.items():
            parts = spec.split(":")
            for other in values:
                if str(parts[axis_idx]) == str(other):
                    continue
                alt = parts.copy()
                alt[axis_idx] = str(other)
                alt_spec = ":".join(alt)
                if alt_spec in results and results[alt_spec] != verdict:
                    pair = tuple(sorted([spec, alt_spec]))
                    if pair not in flips:
                        flips.append(pair)
        for a, b in flips:
            out.append(f"  {axis_name:<9} flips the outcome: {a} = {results[a]}  vs  {b} = {results[b]}")
    if not out:
        out.append("  (no single axis flips the outcome — verdicts are uniform or multi-factor)")
    return out


def run_parent(args) -> int:
    out_path = Path(args.out).resolve()
    log_dir = out_path.parent / (out_path.stem + "_logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    if args.cells:
        specs = [c.strip() for c in args.cells.split(",") if c.strip()]
    else:
        specs = [f"{p}:{j}:{n}" for p in _PRESETS for j in _JAX for n in _NUM_ENVS]

    child_env = {
        **os.environ,
        # Synchronous launches: the crash stacktrace points at the
        # faulting kernel instead of a random later op.
        "CUDA_LAUNCH_BLOCKING": "1",
        # Match the jaxpy alias used for training runs.
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    }

    rows: list[tuple[str, str, str, float]] = []
    verdicts: dict[str, str] = {}
    excerpts: dict[str, list[str]] = {}
    for spec in specs:
        preset, jax, ne = spec.split(":")
        tag = f"{preset}_{jax}_{ne}"
        log_path = log_dir / f"{tag}.log"
        result_path = log_dir / f"{tag}.json"
        if result_path.exists():
            result_path.unlink()

        print(f"[matrix] running cell {spec} ...", flush=True)
        t0 = time.time()
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
                    "--num-steps",
                    str(args.num_steps),
                    "--seed",
                    str(args.seed),
                ],
                stdout=lf,
                stderr=subprocess.STDOUT,
                env=child_env,
            )
        elapsed = time.time() - t0

        data = json.loads(result_path.read_text()) if result_path.exists() else None
        verdict, detail = _cell_verdict(data, proc.returncode)
        if verdict == "FAIL" and data is None:
            detail = f"crashed after stage: {_last_stage(log_path)}"
            excerpts[spec] = _error_excerpt(log_path)
        rows.append((spec, verdict, detail, elapsed))
        verdicts[spec] = verdict
        print(f"[matrix]   -> {verdict} ({elapsed:.0f}s)", flush=True)

    # ── Report ────────────────────────────────────────────────────
    lines: list[str] = []
    lines.append("=" * 110)
    lines.append("mjlab per-env DR — training-faithful matrix diagnostic")
    lines.append("=" * 110)
    lines.append("expand hook:  MjlabEnv._pre_manager_setup (scene built -> expand -> managers)")
    lines.append(f"num steps:    {args.num_steps} per cell   seed: {args.seed}")
    lines.append(f"cell logs:    {log_dir}")
    lines.append("")
    lines.append(f"{'cell (preset:jax:num_envs)':<30}{'verdict':<10}{'elapsed':<10}detail")
    lines.append("-" * 110)
    for spec, verdict, detail, elapsed in rows:
        lines.append(f"{spec:<30}{verdict:<10}{f'{elapsed:.0f}s':<10}{detail}")
    lines.append("")
    lines.append("[Axis analysis] which variable flips PASS/FAIL")
    lines.extend(_axis_analysis(verdicts))
    for spec, tail in excerpts.items():
        lines.append("")
        lines.append(f"[Crash excerpt] {spec}  (full log: {log_dir}/{spec.replace(':', '_')}.log)")
        lines.extend(f"    {ln}" for ln in tail)
    lines.append("")
    n_fail = sum(1 for _s, v, _d, _e in rows if v == "FAIL")
    overall = "PASS" if n_fail == 0 else f"FAIL ({n_fail}/{len(rows)} cells)"
    lines.append("=" * 110)
    lines.append(f"OVERALL: {overall}")
    lines.append("=" * 110)

    report = "\n".join(lines)
    out_path.write_text(report + "\n")
    print()
    print(report)
    print(f"\nReport written to: {out_path}")
    return 0 if n_fail == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", default=None, help="internal: run a single cell 'preset:jax:num_envs'")
    ap.add_argument("--result-json", default=None, help="internal: child result path")
    ap.add_argument("--cells", default=None, help="comma-separated subset, e.g. 'rough:on:256,flat:on:256'")
    ap.add_argument("--num-steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="mjlab_dr_training_matrix.txt")
    args = ap.parse_args()

    if args.cell is not None:
        preset, jax, ne = args.cell.split(":")
        result = run_cell(preset, jax == "on", int(ne), args.num_steps, args.seed)
        Path(args.result_json).write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return 0

    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
