"""End-to-end diagnostic for the Genesis ContactForce filter in t1_getup.

This script verifies that the user-authored Genesis contact-sensor filter
(``rlworld/rl/envs/managers/genesis/contact_sensor.py``) actually
produces the intended self-collision force for the Booster T1 humanoid
get-up preset.

The script dumps **every plausibly-relevant measurement on first run** —
no incremental observability cycles. It records:

  1. Run config (preset name, num_envs, seed, sim timing).
  2. Scene topology — every link in the rigid solver, with its global
     index, name, owning entity, geom count, and inertia mass.
  3. Filter resolution — for each registered ContactSensorCfg group,
     the resolved primary link indices, the resolved ``filter_link_idx``
     blacklist, and a correctness check (``blacklist == all - primary_entity_links``).
  4. Per-step contact force — for each frame in the "standing settle"
     and "fallen settle" phases:
       a. native sensor output (``sensor.compute().force``)
       b. native ContactForce ``found`` bool
       c. Genesis raw ``entity.get_contacts(with_entity=entity)`` output
          (which natively returns self-collision pairs only)
       d. manually-aggregated per-link force from the raw contacts
       e. per-link MAGNITUDE delta — the **acceptance test** for the
          filter. (Per-component delta is also dumped for visibility,
          but Genesis's ContactForceSensor returns force in each link's
          LOCAL frame while the raw API returns world-frame force, so
          per-component values diverge by link rotation while
          magnitudes — which is all the reward consumes — must agree.)
  5. History buffer — ``compute_history()`` shape; cross-check that
     history[..., 0, :] equals the current frame (under the
     "newest-first" docstring assumption).
  6. Edge-case safety — NaN/Inf scan; ``num_envs`` shape check; reward
     call (``rf_genesis.wtw_collision``) returns a finite (E,) tensor.
  7. Verdict — PASS only when all per-frame |sensor - manual| < eps.

The point is *not* to be a self-contained unit test — it relies on the
Genesis simulator being installed and a GPU available. It IS a complete
log file you can hand to a teammate to diagnose whether the filter
holds water for t1_getup.

Usage:
    python -m rlworld.scripts.diag.check_t1_getup_genesis_contact_filter
    python -m rlworld.scripts.diag.check_t1_getup_genesis_contact_filter \\
        --num-envs 2 --settle 20 --steps 40 \\
        --output /tmp/t1_getup_genesis_filter.txt
    python -m rlworld.scripts.diag.check_t1_getup_genesis_contact_filter \\
        --fallen-prob 1.0  # force every env to spawn fallen, so self-contacts occur

Defaults are sized for a quick smoke run; bump --num-envs / --steps for
heavier sampling. The verdict line is the single source of truth.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path

# Multi-sim guard: only Genesis is used here, but the project's default
# is to refuse non-default backend launches when JAXRLWORLD_SIM is set
# inconsistently. Force-enable Genesis-only mode.
os.environ.setdefault("JAXRLWORLD_ALLOW_MULTI_SIM", "1")


# Float-format helper used throughout the report.
def _fmt(x: float, width: int = 9) -> str:
    if x != x or x in (float("inf"), float("-inf")):
        return f"{str(x):>{width}}"
    if abs(x) >= 100:
        return f"{x:{width}.2f}"
    if abs(x) >= 1:
        return f"{x:{width}.4f}"
    return f"{x:{width}.6f}"


def _hr(char: str = "=", n: int = 78) -> str:
    return char * n


def _hdr(title: str) -> str:
    return f"\n{_hr()}\n  {title}\n{_hr()}"


def _subhdr(title: str) -> str:
    return f"\n{_hr('-')}\n  {title}\n{_hr('-')}"


# ──────────────────────────────────────────────────────────────────────
# 1. Build env
# ──────────────────────────────────────────────────────────────────────


def _build_env(num_envs: int, fallen_prob: float | None, seed: int):
    """Construct the Genesis t1_getup env at the requested num_envs / seed."""
    import torch  # noqa: F401  (eager-load before genesis to surface CUDA issues early)

    from rlworld.rl.configs.presets.t1_getup.base import T1GetupConfig
    from rlworld.rl.runners import BaseRunner

    overrides = {}
    if fallen_prob is not None:
        overrides["fallen_prob"] = fallen_prob
    cfg = T1GetupConfig(sim_type="genesis", num_envs=num_envs, seed=seed, **overrides)
    cfgs = cfg.build()
    runner = BaseRunner.create_with_env(cfgs)
    return runner.env, cfg


# ──────────────────────────────────────────────────────────────────────
# 2. Scene topology dump
# ──────────────────────────────────────────────────────────────────────


def dump_scene_topology(env) -> dict[int, dict]:
    """Print every rigid link with idx / name / owning entity / mass.

    Returns a dict keyed by global link index for cross-reference in
    later sections.
    """
    print(_hdr("[2] SCENE TOPOLOGY"))
    scene = env.scene_manager.scene
    rigid_solver = scene.sim.rigid_solver
    n_links = rigid_solver.n_links
    print(f"  rigid_solver.n_links = {n_links}")

    # Build a flat (idx → (name, entity_name)) map by walking all entities.
    by_idx: dict[int, dict] = {}
    try:
        entities = list(scene.entities)
    except Exception:
        entities = []
    print(f"  scene.entities (n={len(entities)}):")
    for ent in entities:
        ent_name = getattr(ent, "name", None) or repr(ent)
        ent_idx = getattr(ent, "idx", "?")
        link_start = getattr(ent, "link_start", None)
        link_end = getattr(ent, "link_end", None)
        n_geoms = "?"
        if hasattr(ent, "n_geoms"):
            try:
                n_geoms = ent.n_geoms
            except Exception:
                pass
        print(
            f"    idx={ent_idx} name={ent_name!r} link_range=[{link_start}, {link_end}) "
            f"n_links={(link_end - link_start) if (link_start is not None and link_end is not None) else '?'} "
            f"n_geoms={n_geoms}"
        )
        if link_start is None or link_end is None:
            continue
        for li in range(link_start, link_end):
            try:
                link = rigid_solver.links[li]
            except Exception:
                link = None
            lname = getattr(link, "name", None) if link is not None else None
            mass = "?"
            try:
                mass = float(link.inertial_mass)
            except Exception:
                try:
                    mass = float(rigid_solver.links_info.inertial_mass[li])
                except Exception:
                    pass
            by_idx[li] = {"name": lname, "entity": ent_name, "mass": mass}

    print(f"\n  per-link map (n={len(by_idx)}):")
    print(f"    {'global_idx':>10}  {'entity':<14}  {'mass':>9}  name")
    for li in sorted(by_idx):
        e = by_idx[li]
        mass_s = _fmt(e["mass"]) if isinstance(e["mass"], int | float) else str(e["mass"])
        print(f"    {li:>10}  {str(e['entity']):<14}  {mass_s}  {e['name']!r}")
    return by_idx


# ──────────────────────────────────────────────────────────────────────
# 3. Filter resolution
# ──────────────────────────────────────────────────────────────────────


def dump_filter_resolution(env, link_map: dict[int, dict]) -> dict[str, dict]:
    """For each Genesis contact sensor, dump the resolved blacklist.

    Returns a dict keyed by group name with introspection data for later
    cross-checks.
    """
    print(_hdr("[3] FILTER RESOLUTION (per ContactSensorCfg group)"))
    mgr = env.contact_manager
    groups = getattr(mgr, "_sensors", None)
    if groups is None:
        print("  ERROR: env.contact_manager has no _sensors dict (unexpected backend?)")
        return {}

    report: dict[str, dict] = {}
    rigid_solver = env.scene_manager.scene.sim.rigid_solver
    n_links = rigid_solver.n_links
    all_links = set(range(n_links))

    for gname, sensor in groups.items():
        print(_subhdr(f"group: {gname!r}"))
        cfg = sensor.cfg
        print(
            f"  cfg.primary   = mode={cfg.primary.mode!r} pattern={cfg.primary.pattern!r} entity={cfg.primary.entity!r}"
        )
        sec = cfg.secondary
        if sec is None:
            print("  cfg.secondary = None  (no filter — every contact counts)")
        else:
            print(f"  cfg.secondary = mode={sec.mode!r} pattern={sec.pattern!r} entity={sec.entity!r}")
        print(f"  cfg.history_length = {cfg.history_length}")
        print(f"  cfg.reduce         = {cfg.reduce!r}")
        print(f"  cfg.num_slots      = {cfg.num_slots}")
        print(f"  cfg.fields         = {tuple(cfg.fields)!r}")

        # Resolved primary link indices (local to entity).
        primary_local = list(sensor._link_ids_local)
        primary_names = list(sensor._tracked_names)
        # Local → global via entity's link_start.
        entity = sensor._entity
        ls = entity.link_start
        primary_global = [ls + l for l in primary_local]
        print(f"\n  resolved primary links (n={len(primary_local)}):")
        print(f"    {'local':>6} {'global':>6}  name")
        for lo, go, nm in zip(primary_local, primary_global, primary_names):
            print(f"    {lo:>6} {go:>6}  {nm!r}")

        blacklist = list(sensor._filter_link_idx)
        print(f"\n  resolved filter_link_idx (blacklist, length={len(blacklist)}):")
        print(f"    {blacklist}")
        # Annotate each blacklist entry with link map info.
        if link_map:
            print("\n    blacklist link details:")
            for bi in blacklist:
                e = link_map.get(bi, {})
                print(f"      idx={bi}  entity={e.get('entity', '?')!r}  name={e.get('name', '?')!r}")

        # CORRECTNESS CHECK: for secondary.entity == "self", blacklist
        # should be exactly all_links minus the primary entity's links.
        if sec is not None and sec.entity == "self":
            expected = sorted(all_links - set(range(entity.link_start, entity.link_end)))
            actual = sorted(blacklist)
            match = expected == actual
            print(
                f"\n  CORRECTNESS CHECK (secondary='self'): blacklist == (all_links − robot_links) ? "
                f"{'PASS' if match else 'FAIL'}"
            )
            if not match:
                missing = set(expected) - set(actual)
                extra = set(actual) - set(expected)
                if missing:
                    print(f"    missing from blacklist: {sorted(missing)}")
                if extra:
                    print(f"    extra in blacklist:    {sorted(extra)}")

        report[gname] = {
            "primary_local": primary_local,
            "primary_global": primary_global,
            "primary_names": primary_names,
            "blacklist": blacklist,
            "entity_link_range": (entity.link_start, entity.link_end),
            "entity_idx": entity.idx,
            "history_length": cfg.history_length,
            "sec_is_self": (sec is not None and sec.entity == "self"),
        }
    return report


# ──────────────────────────────────────────────────────────────────────
# 4. Per-step contact force with raw-contact cross-check
# ──────────────────────────────────────────────────────────────────────


def _capture_raw_self_contacts(entity, num_envs: int):
    """Return dict of raw self-collision contacts for the most recent step.

    Output keys (all torch tensors):
        link_a, link_b   : (num_envs, n_contacts) global link indices
        force_a, force_b : (num_envs, n_contacts, 3) world-frame forces on A / B
        valid_mask       : (num_envs, n_contacts) bool
    """
    return entity.get_contacts(with_entity=entity)


def _manual_per_link_force(raw, primary_global: list[int], num_envs: int, device):
    """Aggregate raw self-contact forces into per-primary-link net Fz tensor.

    Genesis's ContactForce sensor reports the net force on the primary
    link. The raw API gives ``force_a`` on geom A and ``force_b = -force_a``
    on geom B. For each contact (A=link_a, B=link_b), the primary link
    L gets:
        +force_a  if  L == link_a
        +force_b  if  L == link_b
    Summed over all contacts in the frame.

    Returns ``(num_envs, len(primary_global), 3)`` torch tensor.
    """
    import torch

    out = torch.zeros(num_envs, len(primary_global), 3, device=device)

    link_a = raw["link_a"]  # (E, K) or (K,) when num_envs==0
    link_b = raw["link_b"]
    force_a = raw["force_a"]  # (E, K, 3) or (K, 3)
    force_b = raw["force_b"]
    valid_mask = raw.get("valid_mask")  # parallel scenes

    # Normalize shape: ensure (E, K, ...).
    if link_a.dim() == 1:
        link_a = link_a.unsqueeze(0)
        link_b = link_b.unsqueeze(0)
        force_a = force_a.unsqueeze(0)
        force_b = force_b.unsqueeze(0)
    if valid_mask is None:
        valid_mask = torch.ones(link_a.shape, dtype=torch.bool, device=link_a.device)
    elif valid_mask.dim() == 1:
        valid_mask = valid_mask.unsqueeze(0)

    # primary_global → index in the (num_envs, N, 3) output tensor.
    p2idx = {g: i for i, g in enumerate(primary_global)}

    for ei in range(num_envs):
        K = link_a.shape[1]
        for ki in range(K):
            if not bool(valid_mask[ei, ki]):
                continue
            la = int(link_a[ei, ki])
            lb = int(link_b[ei, ki])
            if la in p2idx:
                out[ei, p2idx[la]] += force_a[ei, ki]
            if lb in p2idx:
                out[ei, p2idx[lb]] += force_b[ei, ki]
    return out


def per_step_dump(env, group_report: dict[str, dict], settle: int, steps: int, phase_label: str):
    """Run the env with zero action and capture per-step contact force.

    Compares (a) the user's filter output via ``sensor.compute()`` to
    (b) a manually-aggregated cross-check from
    ``entity.get_contacts(with_entity=entity)``.

    Returns a dict keyed by group with delta stats (for the verdict).
    """
    import torch

    print(_hdr(f"[4-{phase_label}] PER-STEP CONTACT FORCE — {phase_label.upper()} PHASE"))
    print(f"  zero-action settle: {settle} steps; capture: {steps} steps")
    print(f"  num_envs = {env.num_envs}   device = {env.device}")

    zero = torch.zeros(env.num_envs, env.num_actions, device=env.device)

    print("\n  -- settling --")
    for _ in range(settle):
        env.step(zero)

    delta_records: dict[str, dict] = {}
    for gname in group_report:
        delta_records[gname] = {"max_abs_delta": 0.0, "max_rel_delta": 0.0, "per_frame": []}

    for t in range(steps):
        env.step(zero)
        print(_subhdr(f"frame {t} ({phase_label})"))
        for gname, info in group_report.items():
            sensor = env.contact_manager._sensors[gname]
            data = sensor.compute()
            sf = data.force.detach()  # (E, N, 3)
            sfound = data.found.detach()  # (E, N) bool

            print(f"  group={gname!r}")
            print(
                f"    sensor.force  shape={tuple(sf.shape)}  "
                f"|F| max={float(torch.linalg.norm(sf, dim=-1).max()):.4f} N  "
                f"|F| mean={float(torch.linalg.norm(sf, dim=-1).mean()):.4f} N"
            )
            print(
                f"    sensor.found  any={bool(sfound.any())}  "
                f"true-count per env (first 4 envs): {sfound[: min(4, sfound.shape[0])].sum(dim=1).tolist()}"
            )
            # NaN/Inf scan.
            n_nan = int(torch.isnan(sf).sum())
            n_inf = int(torch.isinf(sf).sum())
            if n_nan or n_inf:
                print(f"    !! sensor.force has NaN={n_nan} Inf={n_inf}")

            # Cross-check vs. raw get_contacts() — only meaningful for
            # secondary='self' (raw API returns only self contacts in that mode).
            if info["sec_is_self"]:
                entity = sensor._entity
                try:
                    raw = _capture_raw_self_contacts(entity, env.num_envs)
                except Exception as e:
                    print(f"    raw get_contacts() FAILED: {e}")
                    continue
                # Raw stats.
                la = raw["link_a"]
                if la.dim() == 1:
                    la = la.unsqueeze(0)
                vm = raw.get("valid_mask")
                if vm is None:
                    n_valid_per_env = [la.shape[1]] * env.num_envs
                else:
                    if vm.dim() == 1:
                        vm = vm.unsqueeze(0)
                    n_valid_per_env = vm.sum(dim=1).tolist()
                print(f"    raw self-contacts per env: {n_valid_per_env}")

                manual = _manual_per_link_force(raw, info["primary_global"], env.num_envs, env.device)

                # ── Acceptance test: per-link MAGNITUDE comparison ──
                #
                # Genesis ContactForceSensor returns the per-link force in
                # the LINK's LOCAL frame (genesis/engine/sensors/contact_force.py
                # line 307: ``inv_transform_by_quat(sensors_force, sensors_quat)``).
                # The raw ``entity.get_contacts()`` API returns force in
                # WORLD frame. So per-component values legitimately differ
                # by the link's rotation — but magnitudes must agree
                # because the reward consumer (``penalize_contact_force_count``)
                # uses ``torch.norm(force, dim=-1)``, which is frame-invariant.
                sf_mag = torch.linalg.norm(sf, dim=-1)  # (E, N)
                man_mag = torch.linalg.norm(manual, dim=-1)  # (E, N)
                mag_delta = (sf_mag - man_mag).abs()  # (E, N)
                max_mag_abs = float(mag_delta.max())
                max_mag_val = max(float(sf_mag.max()), float(man_mag.max()), 1e-6)
                max_mag_rel = max_mag_abs / max_mag_val

                # Per-component diff (informational; expected non-zero
                # because of the world-vs-local frame mismatch).
                comp_delta = (sf - manual).abs()
                max_comp_abs = float(comp_delta.max())

                print(f"    sensor_max|F|={float(sf_mag.max()):.4f}  manual_max|F|={float(man_mag.max()):.4f}")
                print(
                    f"    [ACCEPTANCE] max | |sensor| − |manual| | per (env,link) = {max_mag_abs:.4f} N  "
                    f"rel={max_mag_rel:.4%}"
                )
                print(
                    f"    [informational] max |sensor − manual| per element = {max_comp_abs:.4f} N  "
                    f"(expected non-zero: sensor is in link local frame, manual in world frame; "
                    f"magnitudes above are the meaningful check)"
                )
                # Per-link breakdown (top-3 by |sensor.force|).
                sf_mag_mean = sf_mag.mean(dim=0)  # (N,)
                topk = torch.topk(sf_mag_mean, k=min(3, sf_mag_mean.numel())).indices.tolist()
                for k in topk:
                    nm = info["primary_names"][k]
                    s_link = float(sf_mag_mean[k])
                    m_link = float(man_mag.mean(dim=0)[k])
                    print(f"      top |F| link[{k}]={nm!r}: sensor={s_link:.4f}  manual={m_link:.4f}")

                rec = delta_records[gname]
                rec["max_abs_delta"] = max(rec["max_abs_delta"], max_mag_abs)
                rec["max_rel_delta"] = max(rec["max_rel_delta"], max_mag_rel)
                rec["per_frame"].append({"t": t, "max_abs": max_mag_abs, "max_rel": max_mag_rel})

    return delta_records


# ──────────────────────────────────────────────────────────────────────
# 5. History buffer test
# ──────────────────────────────────────────────────────────────────────


def history_buffer_test(env, group_report: dict[str, dict]):
    """Verify shape of compute_history() and the newest-first claim."""

    print(_hdr("[5] HISTORY BUFFER TEST"))
    for gname, info in group_report.items():
        sensor = env.contact_manager._sensors[gname]
        H = sensor.cfg.history_length
        print(_subhdr(f"group={gname!r} cfg.history_length={H}"))
        if H <= 0:
            print("  history disabled — skipping")
            continue
        hist = sensor.compute_history()
        if hist is None:
            print("  compute_history() returned None despite history_length > 0 — UNEXPECTED")
            continue
        cur = sensor.compute().force
        print(
            f"  compute_history() shape = {tuple(hist.shape)}  (expected: (E={env.num_envs}, N={len(info['primary_global'])}, H={H}, 3))"
        )
        # Compare slot 0 and slot -1 against compute().force.
        d0 = (hist[:, :, 0, :] - cur).abs().max().item()
        d_last = (hist[:, :, -1, :] - cur).abs().max().item()
        print(f"  max|history[..., 0, :] − current|     = {d0:.4f}  (≈0 if newest-first as docstring claims)")
        print(f"  max|history[..., -1, :] − current|    = {d_last:.4f}  (≈0 if newest-last)")
        if d0 < d_last:
            print("  → ring order looks like NEWEST-FIRST (matches docstring)")
        elif d_last < d0:
            print("  → ring order looks like NEWEST-LAST (docstring is WRONG)")
        else:
            print("  → inconclusive (both ends differ from current)")


# ──────────────────────────────────────────────────────────────────────
# 6. Reward call sanity
# ──────────────────────────────────────────────────────────────────────


def reward_sanity(env):
    """Call the reward function the preset uses and report finite-ness."""
    import torch

    from rlworld.rl.envs.mdp.rewards.genesis.reward_terms import wtw_collision

    print(_hdr("[6] REWARD CALL SANITY (rf_genesis.wtw_collision, force_threshold=10.0)"))
    try:
        rew = wtw_collision(env, contact_group="self_collision", force_threshold=10.0)
        print(f"  reward shape = {tuple(rew.shape)}  expected ({env.num_envs},)")
        print(f"  reward dtype = {rew.dtype}  device = {rew.device}")
        print(f"  reward min/mean/max = {float(rew.min()):.4f} / {float(rew.mean()):.4f} / {float(rew.max()):.4f}")
        print(f"  NaN count = {int(torch.isnan(rew).sum())}   Inf count = {int(torch.isinf(rew).sum())}")
        # Per-env reward sample (first 8 envs).
        sample = rew[: min(8, rew.shape[0])].detach().cpu().tolist()
        print(f"  per-env sample (first 8): {sample}")
    except Exception as e:
        print(f"  wtw_collision FAILED: {e}")
        traceback.print_exc()


# ──────────────────────────────────────────────────────────────────────
# 7. Verdict
# ──────────────────────────────────────────────────────────────────────


def verdict(deltas: list[dict[str, dict]], abs_eps: float, rel_eps: float) -> int:
    print(_hdr("[7] VERDICT"))
    worst_abs = 0.0
    worst_rel = 0.0
    worst_group: str | None = None
    worst_frame: dict | None = None
    for phase_deltas in deltas:
        for gname, rec in phase_deltas.items():
            if rec["max_abs_delta"] > worst_abs:
                worst_abs = rec["max_abs_delta"]
                worst_group = gname
                if rec["per_frame"]:
                    worst_frame = max(rec["per_frame"], key=lambda r: r["max_abs"])
            worst_rel = max(worst_rel, rec["max_rel_delta"])
    print("  ACCEPTANCE METRIC: per-(env, link) magnitude delta  | |sensor|_2 − |manual|_2 |")
    print(f"  worst absolute delta across all groups/phases: {worst_abs:.6f} N")
    print(f"  worst relative delta:                          {worst_rel:.4%}")
    print(f"  thresholds: abs < {abs_eps:.4f} N  AND  rel < {rel_eps:.4%}")
    if worst_group:
        print(f"  worst group: {worst_group!r}")
        if worst_frame:
            print(
                f"  worst frame: t={worst_frame['t']}  abs={worst_frame['max_abs']:.4f}  rel={worst_frame['max_rel']:.4%}"
            )
    passed = worst_abs < abs_eps and worst_rel < rel_eps
    print(f"\n  STATUS: {'PASS' if passed else 'FAIL'}")
    if not passed:
        print(
            "\n  Likely root causes if FAIL (per-link MAGNITUDE diverges — filter is broken):\n"
            "    (1) Genesis filter_link_idx semantics differ from what\n"
            "        GenesisContactSensor assumes (e.g. it filters by primary\n"
            "        side instead of the counterpart side).\n"
            "    (2) Genesis link indexing for the robot entity is non-contiguous;\n"
            "        range(link_start, link_end) misses some robot links or\n"
            "        wrongly includes non-robot links.\n"
            "    (3) Filter pattern under-/over-matches link names due to a regex\n"
            "        mismatch (e.g. the secondary='self' alias resolves to the\n"
            "        wrong entity)."
        )
    else:
        print(
            "\n  Filter is producing per-link forces that match a manual aggregation\n"
            "  over Genesis's raw self-contacts. The per-component delta dumped above\n"
            "  is informational only — Genesis's ContactForceSensor returns force in\n"
            "  each link's LOCAL frame while the raw API returns world-frame force,\n"
            "  so per-axis values legitimately differ by the link's rotation. The\n"
            "  reward consumer (``penalize_contact_force_count``) takes ``torch.norm``\n"
            "  of the force vector, which is frame-invariant, so the per-component\n"
            "  divergence has no effect on reward correctness."
        )
    return 0 if passed else 1


# ──────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num-envs", type=int, default=2, help="Genesis env count (keep small).")
    ap.add_argument("--settle", type=int, default=10, help="Zero-action settle steps before each capture phase.")
    ap.add_argument("--steps", type=int, default=10, help="Captured steps per phase.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--fallen-prob",
        type=float,
        default=None,
        help="Override T1GetupConfig.fallen_prob. 1.0 = every env spawns fallen "
        "(forces self-contacts to occur, exercising the filter). Default uses "
        "preset value 0.6.",
    )
    ap.add_argument(
        "--abs-eps",
        type=float,
        default=0.5,
        help="Verdict: max |sensor − manual| in N to declare PASS (default 0.5).",
    )
    ap.add_argument(
        "--rel-eps",
        type=float,
        default=0.05,
        help="Verdict: max relative delta to declare PASS (default 5%%).",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/check_t1_getup_genesis_contact_filter.txt"),
        help="Output .txt file (full stdout redirect). Default /tmp/.",
    )
    args = ap.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f, redirect_stdout(f):
        # Section 0: header
        print(_hdr("T1_GETUP GENESIS CONTACT-FILTER DIAG"))
        print(f"  generated: {datetime.now().isoformat(timespec='seconds')}")
        print(f"  args:      {vars(args)}")
        print(f"  python:    {sys.executable}")
        print(f"  cwd:       {os.getcwd()}")

        # Build env, dump preset config.
        try:
            env, cfg = _build_env(args.num_envs, args.fallen_prob, args.seed)
        except Exception as e:
            print(f"\n  FATAL: env build failed: {e}\n")
            traceback.print_exc()
            return 2

        print(_hdr("[1] PRESET / ENV"))
        print(f"  preset class:  {type(cfg).__name__}")
        print(f"  sim_type:      {cfg.sim_type}")
        print(f"  num_envs:      {cfg.num_envs}")
        print(f"  seed:          {cfg.seed}")
        print(f"  fallen_prob:   {cfg.fallen_prob}")
        print(f"  episode_len_s: {cfg.episode_length_s}")
        print(f"  action_scale:  {cfg.action_scale}")
        print(f"  settle_steps:  {cfg.settle_steps}")
        # Timing comes from base._SIM_TIMINGS — re-read it for record.
        from rlworld.rl.configs.presets.t1_getup.base import _SIM_TIMINGS

        timing = _SIM_TIMINGS["genesis"]
        print(f"  timing:        dt={timing['dt']}  substeps={timing['substeps']}  decimation={timing['decimation']}")
        print(
            f"  control_dt:    {timing['dt'] * timing['decimation']} s ({1.0 / (timing['dt'] * timing['decimation']):.1f} Hz)"
        )
        print(f"  env.num_envs:  {env.num_envs}")
        print(f"  env.device:    {env.device}")
        print(f"  env.num_actions: {env.num_actions}")

        # Initial reset.
        try:
            env.reset()
        except Exception as e:
            print(f"\n  WARN: env.reset() raised {e!r} — continuing")

        link_map = dump_scene_topology(env)
        group_report = dump_filter_resolution(env, link_map)

        if not group_report:
            print("\n  no contact groups registered — nothing to verify. Aborting.")
            return 2

        # Two phases: standing and (forced-)fallen. Standing should
        # report ≈0 self-collision; fallen will exercise the filter.
        all_deltas: list[dict[str, dict]] = []
        try:
            env.reset()
            all_deltas.append(per_step_dump(env, group_report, args.settle, args.steps, phase_label="standing"))
        except Exception as e:
            print(f"\n  standing-phase FAILED: {e}")
            traceback.print_exc()

        # For the fallen phase, only useful if the env config has
        # fallen_prob > 0. We don't re-build — the reset distribution
        # already includes fallen pose at rate ``fallen_prob``, so a
        # fresh reset will spawn fallen envs with that probability.
        try:
            env.reset()
            all_deltas.append(per_step_dump(env, group_report, args.settle, args.steps, phase_label="post-reset"))
        except Exception as e:
            print(f"\n  post-reset-phase FAILED: {e}")
            traceback.print_exc()

        try:
            history_buffer_test(env, group_report)
        except Exception as e:
            print(f"\n  history-buffer test FAILED: {e}")
            traceback.print_exc()

        try:
            reward_sanity(env)
        except Exception as e:
            print(f"\n  reward sanity FAILED: {e}")
            traceback.print_exc()

        code = verdict(all_deltas, args.abs_eps, args.rel_eps)
        print(_hr())

    # Echo path for user convenience (this print goes to real stdout).
    print(f"[diag] wrote {args.output}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
