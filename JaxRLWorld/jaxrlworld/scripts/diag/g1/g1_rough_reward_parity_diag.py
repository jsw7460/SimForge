"""G1 29-DOF ROUGH-terrain reward parity: first-step term divergence hunt.

Symptom: at the very first training step on rough terrain, these terms
report very different values across the three simulators:

    self_collision_cost, feet_slip, feet_clearance, flat_orientation,
    soft_landing, track_ang_vel

Rough terrain adds failure modes that no flat-ground diag can catch —
if the three backends disagree on the HEIGHTFIELD (sampling, tiling,
spawn origins, vertical offset), every contact/orientation term diverges
at step one for reasons that have nothing to do with the reward code.
So besides the usual shimmed first-call capture, this diag dumps the
spawn/terrain ground truth per sim:

- per-env root spawn position (first 8 exact + distribution stats):
  spawn-origin or terrain-offset mismatches show up here immediately
- feet height / ground-contact state at post_reset and first_step
- tilt angle (the flat_orientation input) distribution
- self_collision group wiring (tracked-pair count!) + firing stats:
  a group that tracks a different pair set explains self_collision_cost
  outright
- per-term observation parity + NaN/Inf counts

Phases: post_reset -> 1 zero-action step (first reward call captured by
shims) -> 50-step settle -> 300 bit-identical random-action steps with
per-50-step-block term/kinematics means.

Run (server, from SimForge root):
    python -m jaxrlworld.scripts.diag.g1.g1_rough_reward_parity_diag --num-envs 4096
"""

from __future__ import annotations

import argparse
import functools
import importlib
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

_MODULE = "jaxrlworld.scripts.diag.g1.g1_rough_reward_parity_diag"
_SIMS = ("genesis", "newton", "mujoco")
_SIM_KEY = {"genesis": "Genesis", "newton": "Newton", "mujoco": "MujocoEnv"}
_FOCUS = (
    "self_collision_cost",
    "feet_slip",
    "feet_clearance",
    "flat_orientation",
    "soft_landing",
    "track_ang_vel",
    "track_lin_vel",
    "feet_swing_height",
)


def _stage(msg: str) -> None:
    print(f"[STAGE] {msg}", flush=True)


# term name -> (module, function) per sim family
def _term_map(sim: str) -> dict[str, tuple[str, str]]:
    common = "jaxrlworld.rl.envs.mdp.rewards.common.reward_terms"
    if sim == "mujoco":
        mod = "jaxrlworld.rl.envs.mdp.rewards.mujoco.reward_terms"
        return {
            "self_collision_cost": (mod, "self_collision_cost"),
            "feet_slip": (mod, "feet_slip"),
            "feet_clearance": (mod, "feet_clearance"),
            "flat_orientation": (mod, "flat_orientation"),
            "soft_landing": (mod, "soft_landing"),
            "feet_swing_height": (mod, "feet_swing_height"),
            "track_ang_vel": (common, "track_ang_vel"),
            "track_lin_vel": (common, "track_lin_vel"),
        }
    wtw = f"jaxrlworld.rl.envs.mdp.rewards.{sim}.reward_terms"
    mj = f"jaxrlworld.rl.envs.mdp.rewards.{sim}.mjlab_rewards"
    return {
        "self_collision_cost": (wtw, "wtw_collision"),
        "feet_slip": (mj, "feet_slip_mjlab"),
        "feet_clearance": (mj, "feet_clearance_mjlab"),
        "flat_orientation": (common, "flat_orientation"),
        "soft_landing": (mj, "soft_landing_mjlab"),
        "feet_swing_height": (mj, "feet_swing_height_mjlab"),
        "track_ang_vel": (common, "track_ang_vel"),
        "track_lin_vel": (common, "track_lin_vel"),
    }


# ── Child: one sim per process ──────────────────────────────────────


def run_cell(sim: str, num_envs: int, settle_steps: int, seed: int) -> dict:
    import torch

    torch.manual_seed(seed)
    _stage(f"cell start: {sim} num_envs={num_envs}")

    records: dict[str, list] = defaultdict(list)
    term_meta: dict[str, dict] = {}
    env_holder: dict = {}

    def _q(t: torch.Tensor) -> list[float]:
        qs = torch.tensor([0.1, 0.5, 0.9], device=t.device, dtype=torch.float32)
        return [round(float(v), 5) for v in torch.quantile(t.float(), qs).tolist()]

    def _record(term: str, out, kwargs_repr: dict) -> None:
        rec = {"out_mean": float(out.mean()), "out_q10_50_90": _q(out.flatten())}
        if term not in term_meta:
            term_meta[term] = {"kwargs": kwargs_repr}
        records[term].append(rec)

    def _install(term: str, mod_name: str, fn_name: str) -> None:
        import inspect

        mod = importlib.import_module(mod_name)
        orig = getattr(mod, fn_name)

        if inspect.isclass(orig):
            orig_call = orig.__call__

            def call_shim(self, env, *a, __orig_call=orig_call, __term=term, **kw):
                out = __orig_call(self, env, *a, **kw)
                kwargs_repr = {k: repr(v)[:160] for k, v in vars(self).items() if not hasattr(v, "shape")}
                kwargs_repr["__class__"] = f"{mod_name}.{fn_name}"
                _record(__term, out, kwargs_repr)
                return out

            orig.__call__ = call_shim
            return

        def shim(env, *a, __orig=orig, __term=term, **kw):
            out = __orig(env, *a, **kw)
            kwargs_repr = {k: repr(v)[:160] for k, v in kw.items()}
            kwargs_repr["__func__"] = f"{mod_name}.{fn_name}"
            _record(__term, out, kwargs_repr)
            return out

        # The reward manager inspect.signature()s the term to resolve
        # selector-valued defaults; a bare shim hides them.
        functools.update_wrapper(shim, orig)
        setattr(mod, fn_name, shim)

    for term, (m, f) in _term_map(sim).items():
        _install(term, m, f)
    _stage("shims installed")

    from jaxrlworld.rl.configs.base_config import iter_terms
    from jaxrlworld.rl.configs.presets.g1_29dof.base import G1FlatConfig
    from jaxrlworld.rl.configs.rewards import RewardTermConfig
    from jaxrlworld.rl.configs.scene import SceneEntitySelector
    from jaxrlworld.rl.evals.sim_initializers import get_initializer
    from jaxrlworld.rl.utils.quat_utils import quat_rotate_inverse_wxyz

    cfgs = G1FlatConfig(sim_type=sim, num_envs=num_envs, seed=seed, use_rough_terrain=True).build()

    wiring: dict = {}
    wiring["reward_terms"] = {
        name: {
            "func": f"{t.func.__module__}.{getattr(t.func, '__name__', type(t.func).__name__)}",
            "weight": t.weight,
            "params": {k: repr(v)[:150] for k, v in (t.params or {}).items()},
        }
        for name, t in iter_terms(cfgs.reward, RewardTermConfig).items()
    }
    if sim == "mujoco":
        sim_cfg = cfgs.scene.mjlab_sim_cfg
        if sim_cfg is None:
            wiring["solver"] = "mjlab defaults (mjlab_sim_cfg=None)"
        else:
            mj_cfg = sim_cfg.mujoco
            wiring["solver"] = (
                f"iterations={mj_cfg.iterations} cone={mj_cfg.cone} impratio={mj_cfg.impratio} integrator={mj_cfg.integrator}"
            )
    elif sim == "newton":
        nt = cfgs.scene.solver_cfg
        wiring["solver"] = (
            f"iterations={nt.iterations} impratio={nt.impratio} cone={nt.cone} integrator={nt.integrator} "
            f"use_mujoco_contacts={nt.use_mujoco_contacts} nconmax={nt.nconmax} njmax={nt.njmax}"
        )
    else:
        ro = cfgs.scene.rigid_options
        wiring["solver"] = (
            f"integrator={ro.integrator} constraint_solver={ro.constraint_solver} iterations={ro.iterations} "
            f"constraint_timeconst={ro.constraint_timeconst} friction_cone={ro.friction_cone}"
        )
    tcfg = cfgs.scene.terrain_cfg if hasattr(cfgs.scene, "terrain_cfg") else None
    wiring["terrain_cfg"] = repr(tcfg)[:400]
    _stage("config built")

    env = get_initializer(_SIM_KEY[sim]).init_environment(cfgs)
    env_holder["env"] = env
    wiring["decimation"] = env.decimation
    wiring["physics_dt"] = env.physics_dt
    _stage("env built")

    feet_sel = env.resolve_selector(
        SceneEntitySelector(
            name="robot",
            body_names=("left_ankle_roll_link", "right_ankle_roll_link"),
            preserve_order=True,
        )
    )
    feet_ids = feet_sel.body_ids
    wiring["feet_resolved"] = {
        "names": list(feet_sel.body_names or []),
        "ids": [int(i) for i in feet_ids.tolist()],
    }
    # self_collision group wiring: a different tracked-pair set would
    # explain self_collision_cost divergence by construction.
    wiring["contact_groups"] = {}
    for g in ("feet_ground_contact", "self_collision"):
        names = env.contact_manager.tracked_names(g)
        wiring["contact_groups"][g] = names if len(names) <= 40 else f"{len(names)} tracked: {names[:6]}..."

    env.reset()
    torch.cuda.synchronize()
    _stage("reset done")

    def _obs_capture() -> dict:
        out: dict = {}
        om = env.obs_manager
        if not om._is_term_indices_built:
            om._build_term_indices()
        obs = om.get_observation()
        for gname, tensor in obs.items():
            out[f"__{gname}__nonfinite"] = int((~torch.isfinite(tensor)).sum())
            term_slices = om._group_term_indices.get(gname)
            if not term_slices:
                sl = tensor.float()
                out[f"{gname}/<whole>"] = [round(float(sl.mean()), 5), round(float(sl.std()), 5)]
                continue
            for tname, (a, b) in term_slices.items():
                sl = tensor[:, a:b].float()
                out[f"{gname}/{tname}"] = [
                    round(float(sl.mean()), 5),
                    round(float(sl.std()), 5),
                    round(float(sl.abs().max()), 3),
                ]
        return out

    # Canonical terrain grid (SHARED generator output): sampling it at
    # each foot's world xy gives the expected local ground height, so
    # foot_z - canonical_h is the physically measured clearance. The sim
    # whose settled clearance deviates from the others carries a
    # vertical terrain-anchoring offset of exactly that difference.
    ti = env.scene_manager.terrain
    H = torch.as_tensor(ti.heights_m, device=env.device, dtype=torch.float32)
    hs = float(ti.data.horizontal_scale)
    lx, ly = ti.data.size_xy
    wiring["canonical_heights"] = {
        "shape": list(H.shape),
        "min": round(float(H.min()), 5),
        "max": round(float(H.max()), 5),
        "mean": round(float(H.mean()), 5),
        "corner3x3": [[round(float(v), 4) for v in row] for row in H[:3, :3].tolist()],
        "horizontal_scale": hs,
        "size_xy": [float(lx), float(ly)],
    }

    def canonical_h_at(xy: torch.Tensor) -> torch.Tensor:
        """Nearest-cell canonical height at world (x, y); grid centred on
        the origin. Nearest-cell (not bilinear) is fine: the estimate is
        used through the 4096-env MEDIAN, where iid cell noise cancels."""
        ix = ((xy[..., 0] + lx / 2.0) / hs).round().long().clamp(0, H.shape[0] - 1)
        iy = ((xy[..., 1] + ly / 2.0) / hs).round().long().clamp(0, H.shape[1] - 1)
        return H[ix, iy]

    def state_capture(tag: str) -> dict:
        rd = env.get_robot_data()
        root_p = rd.root_link_pos_w
        fz = rd.body_pos_w_by_ids(feet_ids)[..., 2]
        fv = rd.body_lin_vel_w_by_ids(feet_ids)[..., :2].norm(dim=-1)
        feet_c = env.contact_manager.is_contact("feet_ground_contact")
        feet_f = env.contact_manager.contact_force("feet_ground_contact").norm(dim=-1)
        self_c = env.contact_manager.is_contact("self_collision")
        self_f = env.contact_manager.contact_force("self_collision").norm(dim=-1)
        # flat_orientation input: gravity direction in the base frame.
        g_b = quat_rotate_inverse_wxyz(
            rd.root_link_quat_w,
            torch.tensor([0.0, 0.0, -1.0], device=env.device).expand(num_envs, 3),
        )
        cmd = env.command_manager.get_term("velocity").command
        return {
            "tag": tag,
            # spawn/terrain ground truth
            "root_xyz_first8": [[round(float(x), 4) for x in row] for row in root_p[:8].tolist()],
            "root_z_q10_50_90": _q(root_p[:, 2]),
            "root_xy_absmax": round(float(root_p[:, :2].abs().max()), 3),
            "foot_z_mean": [round(float(v), 5) for v in fz.mean(dim=0).tolist()],
            "foot_z_q10_50_90": _q(fz.flatten()),
            "foot_clearance_vs_canonical_q10_50_90": _q(
                (fz - canonical_h_at(rd.body_pos_w_by_ids(feet_ids)[..., :2])).flatten()
            ),
            "root_clearance_vs_canonical_q10_50_90": _q(root_p[:, 2] - canonical_h_at(root_p[:, :2])),
            "foot_xyspeed_mean": [round(float(v), 5) for v in fv.mean(dim=0).tolist()],
            "feet_contact_frac": [round(float(v), 4) for v in feet_c.float().mean(dim=0).tolist()],
            "feet_force_mean": [round(float(v), 3) for v in feet_f.mean(dim=0).tolist()],
            # self-collision ground truth
            "selfcol_any_frac": round(float(self_c.any(dim=1).float().mean()), 5),
            "selfcol_force_gt10_frac": round(float((self_f > 10.0).any(dim=1).float().mean()), 5),
            "selfcol_force_max": round(float(self_f.max()), 2),
            # flat_orientation input
            "tilt_gxy_norm_q10_50_90": _q(g_b[:, :2].norm(dim=-1)),
            # gates
            "cmd_abs_mean": [round(float(v), 4) for v in cmd.abs().mean(dim=0).tolist()],
            "cmd_first8": [[round(float(x), 3) for x in row] for row in cmd[:8].tolist()],
            "base_angvel_absmean": [round(float(v), 4) for v in rd.root_link_ang_vel_w.abs().mean(dim=0).tolist()],
            "obs_terms": _obs_capture(),
        }

    post_reset = state_capture("post_reset")

    zero = torch.zeros((num_envs, env.num_actions), device=env.device)
    env.step(zero)
    torch.cuda.synchronize()
    first_step = state_capture("after_step1")
    _stage("first step done (first reward call captured)")

    for _k in range(settle_steps - 1):
        env.step(zero)
    settle_end = state_capture(f"after_step{settle_steps}")
    _stage("settle done")

    # Bit-identical random window (early-training proxy), 50-step blocks.
    rand_start = {t: len(rs) for t, rs in records.items()}
    gen = torch.Generator().manual_seed(20260721)
    rand_steps, block = 300, 50
    blocks: dict[str, list] = {
        k: []
        for k in (
            "upright_frac",
            "selfcol_frac",
            "foot_xyspeed",
            "feet_contact_frac",
            "tilt_mean",
            "resets",
        )
    }
    acc = {k: 0.0 for k in blocks}
    for _k in range(rand_steps):
        a = torch.randn((num_envs, env.num_actions), generator=gen, device="cpu").to(env.device)
        _o, _r, term_b, trunc_b, _e = env.step(a)
        rd_now = env.get_robot_data()
        acc["upright_frac"] += float((rd_now.root_link_pos_w[:, 2] > 0.45).float().mean())
        acc["selfcol_frac"] += float(env.contact_manager.is_contact("self_collision").any(dim=1).float().mean())
        acc["foot_xyspeed"] += float(rd_now.body_lin_vel_w_by_ids(feet_ids)[..., :2].norm(dim=-1).mean())
        acc["feet_contact_frac"] += float(env.contact_manager.is_contact("feet_ground_contact").float().mean())
        g_b = quat_rotate_inverse_wxyz(
            rd_now.root_link_quat_w,
            torch.tensor([0.0, 0.0, -1.0], device=env.device).expand(num_envs, 3),
        )
        acc["tilt_mean"] += float(g_b[:, :2].norm(dim=-1).mean())
        acc["resets"] += float((term_b | trunc_b).sum())
        if (_k + 1) % block == 0:
            for k in blocks:
                blocks[k].append(round(acc[k] / (block if k != "resets" else 1), 5))
                acc[k] = 0.0
    after_random = state_capture(f"after_random{rand_steps}")
    _stage("random window done")

    term_summary = {}
    for term, recs in records.items():
        outs = [r["out_mean"] for r in recs]
        n0 = rand_start.get(term, len(outs))
        settle_outs, rand_outs = outs[:n0], outs[n0:]
        term_summary[term] = {
            "n_calls": len(recs),
            "first_out": outs[0] if outs else None,
            "first_q10_50_90": recs[0]["out_q10_50_90"] if recs else None,
            "settle_last10_mean": (sum(settle_outs[-10:]) / max(len(settle_outs[-10:]), 1) if settle_outs else None),
            "rand_block_means": [
                round(sum(rand_outs[i : i + block]) / max(len(rand_outs[i : i + block]), 1), 6)
                for i in range(0, len(rand_outs), block)
            ],
        }

    return {
        "sim": sim,
        "wiring": wiring,
        "term_meta": term_meta,
        "term_summary": term_summary,
        "post_reset": post_reset,
        "first_step": first_step,
        "settle_end": settle_end,
        "after_random": after_random,
        "rand_blocks": blocks,
    }


# ── Parent ──────────────────────────────────────────────────────────


def run_parent(args) -> int:
    out_path = Path(args.out).resolve()
    log_dir = out_path.parent / (out_path.stem + "_logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    for sim in _SIMS:
        log_path = log_dir / f"{sim}.log"
        result_path = log_dir / f"{sim}.json"
        if result_path.exists():
            result_path.unlink()
        print(f"[diag] running {sim} ...", flush=True)
        t0 = time.time()
        with open(log_path, "w") as lf:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    _MODULE,
                    "--cell",
                    sim,
                    "--result-json",
                    str(result_path),
                    "--num-envs",
                    str(args.num_envs),
                    "--settle-steps",
                    str(args.settle_steps),
                    "--seed",
                    str(args.seed),
                ],
                stdout=lf,
                stderr=subprocess.STDOUT,
                env={**os.environ},
            )
        ok = result_path.exists()
        print(f"[diag]   -> {'ok' if ok else 'CRASH (see log)'} ({time.time() - t0:.0f}s)", flush=True)
        if ok:
            results[sim] = json.loads(result_path.read_text())

    sims = [s for s in _SIMS if s in results]
    if not sims:
        print("all cells crashed — see logs")
        return 1
    L: list[str] = []
    L.append("=" * 112)
    L.append("G1 rough-terrain reward parity — first-step divergence hunt")
    L.append("=" * 112)
    L.append(f"num_envs: {args.num_envs}  settle: {args.settle_steps}  seed: {args.seed}  logs: {log_dir}")
    L.append("")

    for s in sims:
        w = results[s]["wiring"]
        L.append(f"── [{s}] wiring ──")
        L.append(f"  solver: {w['solver']}")
        L.append(f"  decimation: {w['decimation']}  physics_dt: {w['physics_dt']}")
        L.append(f"  feet: {w['feet_resolved']}")
        for g, names in w["contact_groups"].items():
            L.append(f"  contact_group {g}: {names}")
        L.append(f"  terrain_cfg: {w['terrain_cfg'][:250]}")
        for name in sorted(w["reward_terms"]):
            if name in _FOCUS or name == "self_collision_cost":
                t = w["reward_terms"][name]
                L.append(f"  term {name}: {t['func']}  w={t['weight']}  params={t['params']}")
        L.append("")

    L.append("── SPAWN / TERRAIN ground truth (post_reset; states must match if terrain+origins match) ──")
    for s in sims:
        st = results[s]["post_reset"]
        L.append(f"  [{s}] root_z q10/50/90: {st['root_z_q10_50_90']}  root_xy_absmax: {st['root_xy_absmax']}")
        L.append(f"  [{s}] root_xyz_first8: {st['root_xyz_first8']}")
    L.append("")
    L.append("── CANONICAL TERRAIN GRID (must be identical across sims) ──")
    for s in sims:
        L.append(f"  [{s}] {results[s]['wiring']['canonical_heights']}")
    L.append("")
    L.append("── SURFACE ANCHOR TEST: foot clearance vs canonical height (settled feet must rest at the")
    L.append("   same clearance = sole offset; a per-sim offset here IS the terrain vertical anchor error) ──")
    for tag in ("post_reset", "first_step", "settle_end"):
        for s in sims:
            st = results[s][tag]
            L.append(
                f"  [{s}][{tag}] foot_clearance q10/50/90: {st['foot_clearance_vs_canonical_q10_50_90']}  "
                f"root_clearance q: {st['root_clearance_vs_canonical_q10_50_90']}"
            )
        L.append("")

    L.append("── FIRST reward call (raw term outputs, weight NOT applied) ──")
    L.append(f"  {'quantity':<40s}" + "".join(f"{s:<26s}" for s in sims))

    def row(label, getter, fmt="{:+.6f}"):
        cells = []
        for s in sims:
            try:
                v = getter(results[s])
                cells.append(fmt.format(v) if isinstance(v, int | float) and v is not None else str(v))
            except Exception as e:  # noqa: BLE001 — show the hole
                cells.append(f"ERR:{type(e).__name__}")
        L.append(f"  {label:<40s}" + "".join(f"{c:<26s}" for c in cells))

    for term in _FOCUS:
        row(f"first {term}", lambda r, t=term: r["term_summary"][t]["first_out"])
        row(f"settle-mean {term}", lambda r, t=term: r["term_summary"][t]["settle_last10_mean"])
    L.append("")
    L.append("  random-window blocks (bit-identical actions):")
    for term in _FOCUS:
        for s in sims:
            bl = results[s]["term_summary"].get(term, {}).get("rand_block_means")
            L.append(f"  [{s}] {term}: {bl}")
        L.append("")

    L.append("── CANONICAL STATE (feet / self-collision / tilt / gates) ──")
    for tag in ("post_reset", "first_step", "settle_end", "after_random300"):
        key = tag if tag != "after_random300" else "after_random"
        for s in sims:
            st = results[s][key]
            L.append(
                f"  [{s}][{tag}] foot_z={st['foot_z_mean']} (q={st['foot_z_q10_50_90']})  "
                f"contact={st['feet_contact_frac']}  force={st['feet_force_mean']}  xyspd={st['foot_xyspeed_mean']}"
            )
        for s in sims:
            st = results[s][key]
            L.append(
                f"  [{s}][{tag}] selfcol any={st['selfcol_any_frac']} force>10N={st['selfcol_force_gt10_frac']} "
                f"maxF={st['selfcol_force_max']}  tilt_q={st['tilt_gxy_norm_q10_50_90']}  "
                f"angvel={st['base_angvel_absmean']}"
            )
        L.append("")

    L.append("── RANDOM-window blocks (state) ──")
    for key in ("upright_frac", "selfcol_frac", "feet_contact_frac", "foot_xyspeed", "tilt_mean", "resets"):
        for s in sims:
            L.append(f"  [{s}] {key}: {results[s]['rand_blocks'][key]}")
        L.append("")

    L.append("── OBSERVATION PARITY (rows that mismatch across sims at synchronized states) ──")
    for tag in ("post_reset", "first_step"):
        ref_terms = sorted(results[sims[0]][tag]["obs_terms"].keys())
        mismatches = 0
        for term in ref_terms:
            vals = [results[s][tag]["obs_terms"].get(term) for s in sims]
            if term.endswith("__nonfinite"):
                if any(v != 0 for v in vals):
                    L.append(f"  [{tag}] !!! NONFINITE {term}: " + " ".join(str(v) for v in vals))
                    mismatches += 1
                continue
            means = [v[0] for v in vals if v is not None]
            if len(means) < len(sims) or (max(means) - min(means)) > max(1e-3, 0.01 * max(abs(m) for m in means)):
                L.append(f"  [{tag}] MISMATCH {term}: " + "  ".join(f"{s}={v}" for s, v in zip(sims, vals)))
                mismatches += 1
        L.append(f"  [{tag}] {len(ref_terms)} obs entries checked, {mismatches} mismatch/nonfinite rows")
    L.append("")

    L.append("── first 8 raw commands (RNG stream alignment) ──")
    for s in sims:
        L.append(f"  [{s}] {results[s]['post_reset']['cmd_first8']}")
    L.append("")

    L.append("── term meta (exact function + kwargs at first call) ──")
    for s in sims:
        for term, m in sorted(results[s]["term_meta"].items()):
            L.append(f"  [{s}] {term}: kwargs={m.get('kwargs')}")
        L.append("")

    report = "\n".join(L)
    out_path.write_text(report + "\n")
    print()
    print(report)
    print(f"Report written to: {out_path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", default=None, choices=(None, *_SIMS))
    ap.add_argument("--result-json", default=None)
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--settle-steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="g1_rough_reward_parity_diag.txt")
    args = ap.parse_args()

    if args.cell is not None:
        result = run_cell(args.cell, args.num_envs, args.settle_steps, args.seed)
        Path(args.result_json).write_text(json.dumps(result, indent=2))
        return 0
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
