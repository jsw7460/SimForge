"""mjlab per-env DR diagnostic.

Verifies that the unified DR terms actually randomize mjlab model fields
per-env (not silently env-shared) with the setup-time expand path in
``MjlabEnv._pre_manager_setup`` (scene built -> expand -> managers).

The report is written to a plain .txt file (default:
``mjlab_dr_per_env_diag.txt`` in cwd).  All environment-construction and
reset stdout/stderr — including C-level prints from warp / mujoco_warp
kernel load — are silenced.  Only the final report is printed to
console.

Two checks per mjlab DR field the unified backend writes:
    1. shape[0] == num_envs                  (setup-time expand succeeded)
    2. std across the env axis > 0           (reset actually diversified)

A field only passes check 2 when the preset uses the corresponding
unified DR term; unused fields are reported as SKIP.

Usage:
    python -m rlworld.scripts.diag.dr.check_mjlab_dr_per_env
    python -m rlworld.scripts.diag.dr.check_mjlab_dr_per_env --out my.txt
    python -m rlworld.scripts.diag.dr.check_mjlab_dr_per_env --num-envs 16
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

import torch
import warp as wp

# Preset used for the diagnostic — g1_29dof mujoco exercises every unified
# DR term below (friction, body_mass, body_com_offset, pd_gains,
# joint_armature, joint_friction).
_PRESET_MODULE = "rlworld.rl.configs.presets.g1_29dof.base"
_PRESET_CLASS = "G1FlatConfig"

# Unified DR function -> the mjlab wp_model field(s) it writes.  Must stay
# in sync with the ``_mujoco_*_backend`` implementations in
# ``rl/envs/mdp/events/dr/unified.py`` and with the field map inside
# ``MjlabEnv._pre_manager_setup``.
_UNIFIED_TERM_FIELDS: dict[str, tuple[str, ...]] = {
    "randomize_friction": ("geom_friction",),
    "randomize_body_mass": ("body_mass",),
    "randomize_body_com_offset": ("body_ipos",),
    "randomize_pd_gains": ("actuator_gainprm", "actuator_biasprm"),
    "randomize_joint_armature": ("dof_armature",),
    "randomize_joint_friction": ("dof_frictionloss",),
}

# All fields the diagnostic inspects.  Sorted so the report is stable
# regardless of dict insertion order.
_ALL_FIELDS = sorted({f for fields in _UNIFIED_TERM_FIELDS.values() for f in fields})


class _SilenceFD:
    """Silence stdout+stderr at the file-descriptor level.

    A plain ``contextlib.redirect_stdout`` only captures Python-side
    writes; warp / mujoco_warp emit CUDA-module load lines and errors
    from C.  Redirecting fd 1 and 2 to /dev/null catches both.
    """

    def __enter__(self):
        self._devnull = open(os.devnull, "w")
        self._saved_out = os.dup(1)
        self._saved_err = os.dup(2)
        os.dup2(self._devnull.fileno(), 1)
        os.dup2(self._devnull.fileno(), 2)
        return self

    def __exit__(self, *_):
        os.dup2(self._saved_out, 1)
        os.dup2(self._saved_err, 2)
        os.close(self._saved_out)
        os.close(self._saved_err)
        self._devnull.close()


def _build_env(num_envs: int, seed: int):
    from rlworld.rl.runners import BaseRunner

    cfg_cls = getattr(importlib.import_module(_PRESET_MODULE), _PRESET_CLASS)
    cfgs = cfg_cls(sim_type="mujoco", num_envs=num_envs, seed=seed).build()
    runner = BaseRunner.create_with_env(cfgs)
    return runner.env


def _preset_unified_terms(env) -> set[str]:
    """Return the set of unified DR term names the preset uses.

    Detected by term.func.__module__ ending in ``.dr.unified`` and
    term.func.__name__ being one of ``_UNIFIED_TERM_FIELDS``.
    """
    from rlworld.rl.configs.base_config import iter_terms
    from rlworld.rl.configs.events.event_term_config import EventTermConfig

    used: set[str] = set()
    for _name, term in iter_terms(env.event_cfg, EventTermConfig).items():
        fn = term.func
        if fn is None:
            continue
        mod = getattr(fn, "__module__", "") or ""
        name = getattr(fn, "__name__", "") or ""
        if mod.endswith(".dr.unified") and name in _UNIFIED_TERM_FIELDS:
            used.add(name)
    return used


def _fields_used_by_preset(used_terms: set[str]) -> set[str]:
    out: set[str] = set()
    for term in used_terms:
        out.update(_UNIFIED_TERM_FIELDS[term])
    return out


def _field_shape(sim, field: str) -> tuple | None:
    arr = getattr(sim.wp_model, field, None)
    if arr is None:
        return None
    return tuple(arr.shape)


def _field_tensor(sim, field: str) -> torch.Tensor | None:
    arr = getattr(sim.wp_model, field, None)
    if arr is None:
        return None
    t = wp.to_torch(arr)
    return t.detach().float()


def _env_axis_std(t: torch.Tensor) -> float:
    """Mean std across the env axis (dim 0).

    Aggregates over all trailing dims (per-body / per-dof / per-geom).
    A single scalar > 0 means at least one entry varies across envs.
    """
    if t.shape[0] < 2:
        return 0.0
    flat = t.reshape(t.shape[0], -1)  # (num_envs, K)
    per_col_std = flat.std(dim=0)  # (K,)
    return float(per_col_std.mean().item())


def _fmt_row(*cols: str, widths=(28, 24, 14, 40)) -> str:
    return "  ".join(c.ljust(w) for c, w in zip(cols, widths))


def _hline(widths=(28, 24, 14, 40)) -> str:
    return "-" * (sum(widths) + 2 * (len(widths) - 1))


def run(num_envs: int, seed: int, out_path: Path) -> int:
    lines: list[str] = []
    lines.append("=" * 100)
    lines.append("mjlab per-env DR diagnostic")
    lines.append("=" * 100)
    lines.append(f"Preset:      {_PRESET_CLASS}  (sim_type=mujoco)")
    lines.append(f"num_envs:    {num_envs}")
    lines.append(f"Seed:        {seed}")
    lines.append("")

    # ── Silence env construction + first reset ────────────────────────
    with _SilenceFD():
        torch.manual_seed(seed)
        env = _build_env(num_envs=num_envs, seed=seed)
        sim = env.scene_manager.sim
        used_terms = _preset_unified_terms(env)
        used_fields = _fields_used_by_preset(used_terms)

        # env.reset() triggers reset_dr events which run the unified DR
        # backends we want to validate.
        env.reset()

    # ── Test 1: setup-time expand shape check ─────────────────────────
    lines.append("[Test 1] Setup-time expand shape check")
    lines.append("  Expected: for every field the preset randomizes, shape[0] == num_envs.")
    lines.append(_hline())
    lines.append(_fmt_row("field", "shape", "expanded?", "verdict"))
    lines.append(_hline())

    n_test1_total = 0
    n_test1_pass = 0
    for field in _ALL_FIELDS:
        shape = _field_shape(sim, field)
        if shape is None:
            lines.append(_fmt_row(field, "<missing>", "-", "SKIP (no such wp_model attr)"))
            continue
        in_use = field in used_fields
        expanded = shape[0] == num_envs
        if not in_use:
            verdict = "SKIP (preset does not randomize)"
            tag = "-"
        else:
            n_test1_total += 1
            if expanded:
                verdict = "PASS"
                tag = "YES"
                n_test1_pass += 1
            else:
                verdict = f"FAIL (shape[0]={shape[0]}, expected {num_envs})"
                tag = "NO"
        lines.append(_fmt_row(field, str(shape), tag, verdict))
    lines.append(_hline())
    lines.append(f"  Test 1 summary: {n_test1_pass}/{n_test1_total} fields expanded per-env")
    lines.append("")

    # ── Test 2: per-env variance after reset ──────────────────────────
    lines.append("[Test 2] Per-env variance after reset() (unified DR must vary field per env)")
    lines.append("  Expected: for every field the preset randomizes, std across env axis > 0.")
    lines.append(_hline())
    lines.append(_fmt_row("field", "shape", "env-axis std", "verdict"))
    lines.append(_hline())

    n_test2_total = 0
    n_test2_pass = 0
    for field in _ALL_FIELDS:
        t = _field_tensor(sim, field)
        if t is None:
            lines.append(_fmt_row(field, "<missing>", "-", "SKIP (no such wp_model attr)"))
            continue
        in_use = field in used_fields
        if not in_use:
            lines.append(_fmt_row(field, str(tuple(t.shape)), "-", "SKIP (preset does not randomize)"))
            continue
        n_test2_total += 1
        std = _env_axis_std(t)
        if std > 1e-12:
            verdict = f"PASS (std={std:.6e})"
            n_test2_pass += 1
        else:
            verdict = f"FAIL (std={std:.6e} = env-shared)"
        lines.append(_fmt_row(field, str(tuple(t.shape)), f"{std:.6e}", verdict))
    lines.append(_hline())
    lines.append(f"  Test 2 summary: {n_test2_pass}/{n_test2_total} fields vary per env")
    lines.append("")

    # ── Preset introspection block ────────────────────────────────────
    lines.append("[Info] Unified DR terms detected in preset event_cfg")
    lines.append(_hline())
    if used_terms:
        for term_name in sorted(used_terms):
            fields = ", ".join(_UNIFIED_TERM_FIELDS[term_name])
            lines.append(f"  {term_name:30s} -> {fields}")
    else:
        lines.append("  (none)")
    lines.append("")

    # ── Overall verdict ───────────────────────────────────────────────
    all_pass = n_test1_pass == n_test1_total and n_test2_pass == n_test2_total and n_test1_total > 0
    lines.append("=" * 100)
    lines.append(f"OVERALL: {'PASS' if all_pass else 'FAIL'}")
    lines.append("=" * 100)

    report = "\n".join(lines) + "\n"
    out_path.write_text(report)

    # Only the report goes to console; env construction noise is already silenced.
    sys.stdout.write(report)
    sys.stdout.write(f"\nReport written to: {out_path.resolve()}\n")
    return 0 if all_pass else 1


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="mjlab per-env DR diagnostic")
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--out",
        type=Path,
        default=Path("mjlab_dr_per_env_diag.txt"),
        help="Report output path (default: mjlab_dr_per_env_diag.txt in cwd).",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    sys.exit(run(num_envs=args.num_envs, seed=args.seed, out_path=args.out))
