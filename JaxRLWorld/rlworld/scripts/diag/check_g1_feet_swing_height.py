"""Cross-sim feet_swing_height parity diagnostic for g1_29dof.

The ``feet_swing_height`` reward uses a stateful
:class:`FeetSwingHeightTracker` (see
``rlworld/rl/envs/mdp/rewards/common/reward_terms.py``) that:

  1. Reads foot z heights every call.
  2. Reads ``is_contact`` bool per foot.
  3. Updates ``peak_heights[i, f] = max(peak_heights[i, f], foot_z[i, f])``
     wherever ``~is_contact`` (foot in swing).
  4. Reads ``first_contact`` bool per foot.
  5. Computes ``error = peak_heights / target_h - 1``, squared then
     summed * ``cmd_active`` to give per-env cost.
  6. Resets ``peak_heights[i, f] = 0`` where ``first_contact``.

So the reward's output depends on:
  * The foot z time series (per sim's body kinematics).
  * The ``is_contact`` time series — affects when peak updates run.
  * The ``first_contact`` time series — affects when cost triggers and
    when peak resets.

The user's observation: enabling Genesis ``contact_pruning_tolerance``
fixed ``soft_landing`` parity but broke ``feet_swing_height``. This
diag reveals where the divergence lives by capturing every input the
tracker sees, step by step, and replaying the same algorithm against
both sims' streams.

Captured per step (after a settle phase, during a random-action phase):
  * ``foot_z[i, f]`` for every (env, foot)
  * ``foot_vz[i, f]``
  * ``is_contact[i, f]``
  * ``first_contact[i, f]``  (computed via the contact manager — this
    is what the tracker sees)
  * ``peak_heights[i, f]`` (this diag's own replay state)
  * Per-step cost contribution

Aggregated:
  * Landing event list: ``(step, env, foot, peak, target, err^2)``
  * Distribution of peak heights at landing
  * Number of landings per sim
  * Mean per-foot cost

Driver mode (default): spawns Genesis and MuJoCo (and optionally
Newton) as fresh subprocesses; reads back JSON dumps; prints
comparison.

Usage::

    python -m rlworld.scripts.diag.check_g1_feet_swing_height
    python -m rlworld.scripts.diag.check_g1_feet_swing_height \\
        --num-envs 16 --capture-steps 200 --action-scale 0.3
    python -m rlworld.scripts.diag.check_g1_feet_swing_height \\
        --sims genesis,mujoco,newton
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

os.environ.setdefault("JAXRLWORLD_ALLOW_MULTI_SIM", "1")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_list(t) -> list:
    """Detach + cpu + tolist for tensors; passthrough for already-list values."""
    try:
        import torch

        if isinstance(t, torch.Tensor):
            t2 = t.detach().cpu()
            if t2.dtype == torch.bool:
                return t2.tolist()
            return t2.float().tolist()
    except ImportError:
        pass
    if hasattr(t, "tolist"):
        return t.tolist()
    return t


# ---------------------------------------------------------------------------
# Per-sim capture (runs inside subprocess)
# ---------------------------------------------------------------------------


def _capture_sim(
    sim: str,
    num_envs: int,
    seed: int,
    settle_steps: int,
    capture_steps: int,
    action_scale: float,
    target_height: float = 0.1,
    command_threshold: float = 0.05,
) -> dict:
    """Run g1_29dof for ``settle_steps + capture_steps`` and dump all
    inputs feet_swing_height sees, plus our own replay state.

    The reward tracker is stateful; rather than fishing the runtime
    tracker out of the reward manager, we replay the same algorithm
    here against the same env streams (foot z, is_contact). Same
    inputs => same output, so any divergence we observe in the
    runtime tracker is reproduced in our replay.
    """
    import torch

    from rlworld.rl.configs.presets.g1_29dof.base import G1FlatConfig
    from rlworld.rl.configs.scene.entity_selector import SceneEntitySelector
    from rlworld.rl.envs.mdp.rewards.common.reward_terms import _command_active
    from rlworld.rl.runners import BaseRunner

    out: dict = {
        "sim": sim,
        "num_envs": num_envs,
        "seed": seed,
        "settle_steps": settle_steps,
        "capture_steps": capture_steps,
        "action_scale": action_scale,
        "target_height": target_height,
        "command_threshold": command_threshold,
    }

    # ── Build env ────────────────────────────────────────────────────
    cfg = G1FlatConfig(sim_type=sim, num_envs=num_envs, seed=seed)
    cfgs = cfg.build()
    runner = BaseRunner.create_with_env(cfgs)
    env = runner.env

    # The Genesis preset reads ``feet_links`` (= ankle_roll_link) for
    # foot_height obs; Newton/Genesis presets use ``left_foot_frame``/
    # ``right_foot_frame`` for the *reward* asset_cfg. Use the reward's
    # selector exactly so we match the production tracker.
    feet_selector = env.resolve_selector(
        SceneEntitySelector(
            name="robot",
            body_names=("left_foot_frame", "right_foot_frame"),
            preserve_order=True,
        )
    )
    # The contact_order used by the reward (see preset):
    contact_order = ["left_ankle_roll_link", "right_ankle_roll_link"]
    foot_ids = feet_selector.body_ids
    out["asset_cfg_body_names"] = list(feet_selector.body_names) if feet_selector.body_names else None
    out["contact_order"] = list(contact_order)

    # ── Reset + settle (zero action) ──────────────────────────────────
    env.reset()
    n_act = env.num_actions
    zero = torch.zeros(env.num_envs, n_act, device=env.device)
    for _ in range(settle_steps):
        env.step(zero)

    # ── Replay state: per-env per-foot peak_heights ──────────────────
    n_feet = len(foot_ids)
    peak_heights = torch.zeros((num_envs, n_feet), device=env.device, dtype=torch.float32)

    # Per-step capture (only env 0..min(4, num_envs) detailed; aggregate stats over all envs)
    timeseries: list[dict] = []
    landings: list[dict] = []  # (step, env, foot, peak, error, err_squared)
    cumulative_cost = torch.zeros(num_envs, device=env.device, dtype=torch.float32)

    # Deterministic RNG for action. Generate on CPU regardless of
    # default-device setting (Genesis sets default_device=cuda which
    # breaks ``torch.rand(generator=cpu_rng)`` without explicit
    # ``device='cpu'``).
    rng = torch.Generator(device="cpu").manual_seed(seed)

    for s in range(capture_steps):
        # Small uniform random action so feet may swing.
        if action_scale > 0:
            act_cpu = (torch.rand(num_envs, n_act, generator=rng, device="cpu") * 2 - 1) * action_scale
            act = act_cpu.to(env.device)
        else:
            act = zero
        env.step(act)

        # ── Inputs the tracker sees ──────────────────────────────────
        rd = env.get_robot_data()
        foot_pos = rd.body_pos_w_by_ids(foot_ids).detach()  # (B, 2, 3)
        foot_z = foot_pos[..., 2]  # (B, 2)
        foot_lin_vel = rd.body_lin_vel_w_by_ids(foot_ids).detach()  # (B, 2, 3)
        foot_vz = foot_lin_vel[..., 2]  # (B, 2)
        is_contact = env.contact_manager.is_contact("feet_ground_contact", order=contact_order)  # (B, 2) bool
        first_contact = env.contact_manager.compute_first_contact(
            "feet_ground_contact", order=contact_order
        )  # (B, 2) bool

        # ── Replay the tracker logic ─────────────────────────────────
        in_air = ~is_contact
        peak_before = peak_heights.clone()
        peak_heights = torch.where(in_air, torch.maximum(peak_heights, foot_z), peak_heights)

        error = peak_heights / target_height - 1.0
        err_term = error.square()
        cmd_active = _command_active(env, command_threshold)  # (B,)
        per_step_cost = (err_term * first_contact.float()).sum(dim=1) * cmd_active
        cumulative_cost += per_step_cost

        # Record landing events.
        if bool(first_contact.any()):
            fc_cpu = first_contact.cpu()
            eh = peak_heights.cpu()
            er_sq = err_term.cpu()
            for i in range(num_envs):
                for f in range(n_feet):
                    if bool(fc_cpu[i, f]):
                        landings.append(
                            {
                                "step": s,
                                "env": i,
                                "foot": f,
                                "peak": float(eh[i, f].item()),
                                "ratio": float((eh[i, f] / target_height).item()),
                                "err_sq": float(er_sq[i, f].item()),
                            }
                        )

        # Reset peaks where first_contact.
        peak_heights = torch.where(first_contact, torch.zeros_like(peak_heights), peak_heights)

        # Time series (small payload; capture every step but env 0 only).
        timeseries.append(
            {
                "step": s,
                "foot_z_env0": foot_z[0].cpu().tolist(),
                "foot_vz_env0": foot_vz[0].cpu().tolist(),
                "is_contact_env0": is_contact[0].cpu().tolist(),
                "first_contact_env0": first_contact[0].cpu().tolist(),
                "peak_heights_env0_before": peak_before[0].cpu().tolist(),
                "peak_heights_env0_after": peak_heights[0].cpu().tolist(),
                "per_step_cost_env0": float(per_step_cost[0].item()),
                # Aggregate stats.
                "is_contact_fraction": float(is_contact.float().mean().item()),
                "first_contact_fraction": float(first_contact.float().mean().item()),
                "in_air_fraction": float(in_air.float().mean().item()),
                "foot_z_mean": float(foot_z.mean().item()),
                "foot_z_max": float(foot_z.max().item()),
                "foot_z_min": float(foot_z.min().item()),
                "peak_heights_mean": float(peak_heights.mean().item()),
                "peak_heights_max": float(peak_heights.max().item()),
                "per_step_cost_mean": float(per_step_cost.mean().item()),
                "cmd_active_fraction": float(cmd_active.float().mean().item()),
            }
        )

    out["timeseries"] = timeseries
    out["landings"] = landings
    out["cumulative_cost_mean"] = float(cumulative_cost.mean().item())
    out["cumulative_cost_per_env"] = cumulative_cost.cpu().tolist()
    out["final_peak_heights_env0"] = peak_heights[0].cpu().tolist()
    out["n_landings"] = len(landings)

    # Aggregate landings stats.
    if landings:
        peaks = [l["peak"] for l in landings]
        ratios = [l["ratio"] for l in landings]
        err_sqs = [l["err_sq"] for l in landings]
        out["landings_stats"] = {
            "n": len(landings),
            "peak_mean": sum(peaks) / len(peaks),
            "peak_min": min(peaks),
            "peak_max": max(peaks),
            "ratio_mean": sum(ratios) / len(ratios),
            "ratio_min": min(ratios),
            "ratio_max": max(ratios),
            "err_sq_mean": sum(err_sqs) / len(err_sqs),
            "err_sq_sum": sum(err_sqs),
        }
    return out


# ---------------------------------------------------------------------------
# Driver + comparison
# ---------------------------------------------------------------------------


def _spawn_one(
    sim: str,
    num_envs: int,
    seed: int,
    out_dir: Path,
    settle_steps: int,
    capture_steps: int,
    action_scale: float,
) -> Path | None:
    out_json = out_dir / f"g1_swing_{sim}.json"
    cmd = [
        sys.executable,
        "-m",
        "rlworld.scripts.diag.check_g1_feet_swing_height",
        "--sim",
        sim,
        "--num-envs",
        str(num_envs),
        "--seed",
        str(seed),
        "--settle-steps",
        str(settle_steps),
        "--capture-steps",
        str(capture_steps),
        "--action-scale",
        str(action_scale),
        "--out-json",
        str(out_json),
        "--no-driver",
    ]
    print(f"\n┃ launching subprocess: {' '.join(cmd)}\n")
    res = subprocess.run(cmd, check=False)
    if res.returncode != 0 or not out_json.exists():
        print(f"  ✗ {sim} subprocess failed (rc={res.returncode})")
        return None
    return out_json


def _fmt(v: Any, prec: int = 6) -> str:
    if isinstance(v, float):
        return f"{v:.{prec}g}"
    if isinstance(v, bool):
        return "T" if v else "F"
    if isinstance(v, list):
        return "[" + ", ".join(_fmt(x, prec) for x in v) + "]"
    if v is None:
        return "—"
    return str(v)


def _print_compare(per_sim: dict[str, dict], lg) -> None:
    sims = list(per_sim)

    def row(label: str, lookup):
        cells = []
        for s in sims:
            try:
                v = lookup(per_sim[s])
            except Exception:
                v = "—"
            cells.append(_fmt(v))
        col_width = 32
        line = f"  {label:<48s}" + " | ".join(f"{c:<{col_width}s}" for c in cells)
        lg(line)

    def section(title: str):
        lg("")
        lg(f"━━━ {title} " + "━" * max(0, 70 - len(title)))

    # Header
    lg("\n" + "=" * 120)
    lg(
        f"  g1_29dof feet_swing_height parity  |  sims = {sims}  |  "
        f"num_envs = {per_sim[sims[0]]['num_envs']}  |  capture_steps = {per_sim[sims[0]]['capture_steps']}"
    )
    lg("=" * 120)
    col_width = 32
    header = f"  {'metric':<48s}" + " | ".join(f"{s:<{col_width}s}" for s in sims)
    lg(header)
    lg("  " + "-" * (48 + (col_width + 3) * len(sims)))

    section("Config")
    row("num_envs", lambda d: d["num_envs"])
    row("seed", lambda d: d["seed"])
    row("settle_steps", lambda d: d["settle_steps"])
    row("capture_steps", lambda d: d["capture_steps"])
    row("action_scale", lambda d: d["action_scale"])
    row("target_height", lambda d: d["target_height"])
    row("command_threshold", lambda d: d["command_threshold"])
    row("asset_cfg.body_names", lambda d: d["asset_cfg_body_names"])
    row("contact_order", lambda d: d["contact_order"])

    section("Landing-event aggregate stats")
    row("n_landings", lambda d: d["n_landings"])
    row("cumulative_cost.mean (over all envs)", lambda d: d["cumulative_cost_mean"])
    row("landings.peak_mean", lambda d: d["landings_stats"]["peak_mean"])
    row("landings.peak_min", lambda d: d["landings_stats"]["peak_min"])
    row("landings.peak_max", lambda d: d["landings_stats"]["peak_max"])
    row("landings.ratio_mean (peak/target)", lambda d: d["landings_stats"]["ratio_mean"])
    row("landings.ratio_min", lambda d: d["landings_stats"]["ratio_min"])
    row("landings.ratio_max", lambda d: d["landings_stats"]["ratio_max"])
    row("landings.err_sq_mean", lambda d: d["landings_stats"]["err_sq_mean"])
    row("landings.err_sq_sum (= total cost contribution)", lambda d: d["landings_stats"]["err_sq_sum"])

    # ── Per-step time-series alignment ────────────────────────────────
    section("Per-step time series — aggregate (all envs)")
    ts_lengths = [len(per_sim[s]["timeseries"]) for s in sims]
    n_show = min(ts_lengths)
    # Sample evenly-spaced steps.
    sample_idx = sorted(set(list(range(min(10, n_show))) + [n_show * k // 8 for k in range(1, 9)]))
    sample_idx = [i for i in sample_idx if i < n_show]
    lg(f"  showing {len(sample_idx)} sampled steps out of {n_show}")
    lg("")
    for sample in sample_idx:
        row(
            f"step {sample}: is_contact.fraction",
            lambda d, sample=sample: d["timeseries"][sample]["is_contact_fraction"],
        )
    lg("")
    for sample in sample_idx:
        row(
            f"step {sample}: first_contact.fraction",
            lambda d, sample=sample: d["timeseries"][sample]["first_contact_fraction"],
        )
    lg("")
    for sample in sample_idx:
        row(f"step {sample}: foot_z.max (all envs)", lambda d, sample=sample: d["timeseries"][sample]["foot_z_max"])
    lg("")
    for sample in sample_idx:
        row(f"step {sample}: foot_z.mean (all envs)", lambda d, sample=sample: d["timeseries"][sample]["foot_z_mean"])
    lg("")
    for sample in sample_idx:
        row(
            f"step {sample}: peak_heights.max (all envs)",
            lambda d, sample=sample: d["timeseries"][sample]["peak_heights_max"],
        )
    lg("")
    for sample in sample_idx:
        row(
            f"step {sample}: per_step_cost.mean (all envs)",
            lambda d, sample=sample: d["timeseries"][sample]["per_step_cost_mean"],
        )

    # ── Env-0 trace (every step, foot z + contacts) ───────────────────
    section("env 0 trace — foot_z[left, right], is_contact, first_contact, peak (before→after)")
    lg(
        f"  {'step':<6s} {'sim':<10s} {'foot_z':<28s} {'is_contact':<14s} {'first_contact':<16s} {'peak_before':<22s} {'peak_after':<22s} {'cost':<8s}"
    )
    for sample in sample_idx:
        for s in sims:
            ts = per_sim[s]["timeseries"][sample]
            lg(
                f"  {sample:<6d} {s:<10s} "
                f"{str(ts['foot_z_env0']):<28s} "
                f"{str(ts['is_contact_env0']):<14s} "
                f"{str(ts['first_contact_env0']):<16s} "
                f"{str(ts['peak_heights_env0_before']):<22s} "
                f"{str(ts['peak_heights_env0_after']):<22s} "
                f"{ts['per_step_cost_env0']:<8.4f}"
            )

    # ── First N landing events of env 0 (per sim) ────────────────────
    section("First 16 landing events per sim — (step, env, foot, peak, ratio, err²)")
    for s in sims:
        lg(f"  [{s}]")
        for l in (per_sim[s]["landings"] or [])[:16]:
            lg(
                f"    step={l['step']:<4d} env={l['env']:<3d} foot={l['foot']:<2d} "
                f"peak={l['peak']:.4f}  ratio={l['ratio']:.4f}  err²={l['err_sq']:.4f}"
            )

    # ── Ratios ────────────────────────────────────────────────────────
    if len(sims) >= 2:
        section("Ratios (relative to last sim)")
        ref = sims[-1]
        for s in sims[:-1]:
            for label, key in (
                ("cumulative_cost_mean", "cumulative_cost_mean"),
                ("n_landings", "n_landings"),
            ):
                try:
                    a = per_sim[s][key]
                    b = per_sim[ref][key]
                    if isinstance(b, int | float) and b != 0:
                        lg(f"  ({s}) / ({ref})  {label:<28s} = {a / b:.4g}")
                except Exception:
                    pass
            try:
                a = per_sim[s]["landings_stats"]["err_sq_sum"]
                b = per_sim[ref]["landings_stats"]["err_sq_sum"]
                lg(f"  ({s}) / ({ref})  total_err²_sum            = {a / b if b else 'inf':.4g}")
            except Exception:
                pass
            try:
                a = per_sim[s]["landings_stats"]["peak_mean"]
                b = per_sim[ref]["landings_stats"]["peak_mean"]
                lg(f"  ({s}) / ({ref})  peak_mean                 = {a / b if b else 'inf':.4g}")
            except Exception:
                pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--sims",
        type=str,
        default="genesis,mujoco",
        help="Comma-separated subset of {genesis, mujoco, newton}. Default: genesis,mujoco.",
    )
    ap.add_argument(
        "--sim",
        type=str,
        default=None,
        choices=("genesis", "mujoco", "newton"),
        help="Single-sim inline mode (with --no-driver) — used by driver subprocess.",
    )
    ap.add_argument("--num-envs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--settle-steps", type=int, default=20, help="Zero-action steps to land the robot before capture.")
    ap.add_argument(
        "--capture-steps",
        type=int,
        default=120,
        help="Random-action capture steps (need enough to see multiple landing events).",
    )
    ap.add_argument(
        "--action-scale", type=float, default=0.3, help="Uniform random action magnitude during capture (0 = static)."
    )
    ap.add_argument("--out-json", type=str, default=None, help="Inline single-sim mode: JSON destination.")
    ap.add_argument("--out-dir", type=str, default=".", help="Driver mode: where per-sim JSONs + comparison go.")
    ap.add_argument("--no-driver", action="store_true")
    args = ap.parse_args()

    # ── Inline mode ───────────────────────────────────────────────────
    if args.no_driver:
        if args.sim is None:
            print("inline mode requires --sim", file=sys.stderr)
            return 2
        try:
            data = _capture_sim(
                args.sim,
                args.num_envs,
                args.seed,
                args.settle_steps,
                args.capture_steps,
                args.action_scale,
            )
        except Exception as e:
            traceback.print_exc()
            print(f"capture failed for {args.sim}: {e!r}", file=sys.stderr)
            return 1
        target = Path(args.out_json) if args.out_json else Path(f"./g1_swing_{args.sim}.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(data, indent=2))
        print(f"\n✓ wrote {target}")
        return 0

    # ── Driver mode ───────────────────────────────────────────────────
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    sims = [s.strip() for s in args.sims.split(",") if s.strip()]
    unknown = [s for s in sims if s not in ("genesis", "mujoco", "newton")]
    if unknown:
        print(f"unknown sim(s): {unknown}", file=sys.stderr)
        return 2

    per_sim: dict[str, dict] = {}
    for sim in sims:
        path = _spawn_one(
            sim, args.num_envs, args.seed, out_dir, args.settle_steps, args.capture_steps, args.action_scale
        )
        if path is None:
            continue
        per_sim[sim] = json.loads(path.read_text())

    if not per_sim:
        print("no sim produced a valid JSON — aborting")
        return 1

    lines: list[str] = []

    def lg(s: str = "") -> None:
        print(s)
        lines.append(s)

    _print_compare(per_sim, lg)
    compare_path = out_dir / "g1_swing_compare.txt"
    compare_path.write_text("\n".join(lines))
    print(f"\n✓ wrote comparison to {compare_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
