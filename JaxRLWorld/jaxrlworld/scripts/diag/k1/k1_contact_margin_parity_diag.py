"""K1 contact-threshold parity: measure the EFFECTIVE contact margin/gap per sim.

Late-training symptoms (K1 G1-recipe: genesis feet_slip and
feet_swing_height ~2x newton/mujoco; K1 joystick: genesis feet_air_time
far off): all three rewards hinge on the contact ON/OFF decision at
touchdown/liftoff. Hypothesis under test: the effective contact
detection threshold (margin / gap / hysteresis) differs across engines.

Source audit (recorded in the wiring section so the report is
self-contained):
- mjwarp (mjlab AND newton, since use_mujoco_contacts): a contact is
  included when dist < margin - gap, per-geom. K1 foot geoms are built
  with margin=0; geom_gap is dumped here (expected 0) → contact only at
  actual penetration.
- genesis plane narrowphase: ``is_col = penetration > 0.0`` — likewise
  only at penetration. The multi-contact merge tolerance
  ``mc_tolerance`` (1e-3 with mujoco compat, else 1.5e-2) and
  ``mc_perturbation`` are dumped as the only genesis-side candidates.

The diag does NOT trust the source reading — it measures behavior:

1. THRESHOLD SWEEP: every env is teleported to the default pose with
   the base z spread over a ±8 mm grid around the height where the sole
   just touches (calibrated in-diag), zero velocity, then ONE zero-
   action step. Per env we record (measured sole gap, found, |force|)
   for the left foot. Binned by gap this IS the effective contact-on
   curve per sim; if the curves overlay, the margin/gap hypothesis is
   dead.

2. GAIT-EDGE STATS: identical open-loop 1.4 Hz gait for probe_steps,
   recording per control step contact + sole gap + foot xy speed. At
   every contact rising/falling edge we log the measured sole gap (the
   height at which contact flips per sim), plus stance/air segment
   length histograms (feet_air_time's raw material) and mean in-stance
   xy speed (feet_slip's integrand). Envs that reset mid-probe have
   their bookkeeping invalidated for that step.

Run (server, from SimForge root):
    python -m jaxrlworld.scripts.diag.k1.k1_contact_margin_parity_diag
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

_MODULE = "jaxrlworld.scripts.diag.k1.k1_contact_margin_parity_diag"
# genesis_elliptic: same genesis backend/preset, but with
# friction_cone=elliptic (impratio auto-resolves to 100 — genesis
# rigid_solver.py:285). Tests whether aligning the friction-cone model
# with the mjlab/newton canonical recipe (elliptic + impratio 100)
# closes the stance-creep gap. The stock genesis cell runs the preset
# as-is: genesis default pyramidal cone + impratio 1.
_SIMS = ("genesis", "genesis_elliptic", "newton", "mujoco")
_SIM_KEY = {"genesis": "Genesis", "newton": "Newton", "mujoco": "MujocoEnv"}
_GAIT_HZ = 1.4
# Foot-link origin to sole: foot collision spheres sit at local z=-0.038
# with radius 0.01 → sole plane is 0.048 below the link origin.
_SOLE_OFFSET = 0.048


def _stage(msg: str) -> None:
    print(f"[STAGE] {msg}", flush=True)


# ── Child: one sim per process ──────────────────────────────────────


def run_cell(sim: str, num_envs: int, probe_steps: int, seed: int, gait_scale: float) -> dict:
    import torch

    torch.manual_seed(seed)
    _stage(f"cell start: {sim} num_envs={num_envs}")

    from jaxrlworld.rl.configs.presets.k1_joystick.g1_recipe import K1G1RecipeConfig
    from jaxrlworld.rl.configs.robots.k1 import K1Config
    from jaxrlworld.rl.configs.scene import SceneEntitySelector
    from jaxrlworld.rl.evals.sim_initializers import get_initializer

    k1 = K1Config()
    backend = "genesis" if sim.startswith("genesis") else sim
    cfgs = K1G1RecipeConfig(sim_type=backend, num_envs=num_envs, seed=seed).build()

    if sim == "genesis_elliptic":
        import genesis as gs

        # Align the friction-cone model with the mjlab/newton canonical
        # recipe. impratio stays None so the solver auto-resolves it to
        # 100 for the elliptic cone (matching mjwarp impratio=100).
        cfgs.scene.rigid_options.friction_cone = gs.friction_cone.elliptic

    env = get_initializer(_SIM_KEY[backend]).init_environment(cfgs)
    _stage("env built")

    # ---- wiring: every margin/gap-like knob each backend has ---------
    wiring: dict = {"decimation": env.decimation, "physics_dt": env.physics_dt}
    if backend == "genesis":
        solver = env.scene_manager.scene.rigid_solver
        wiring["contact_emission_rule"] = "genesis plane narrowphase: is_col = penetration > 0.0 (source-audited)"
        wiring["mc_tolerance"] = float(solver.collider._mc_tolerance)
        wiring["mc_perturbation"] = float(solver.collider._mc_perturbation)
        wiring["enable_mujoco_compatibility"] = bool(solver._enable_mujoco_compatibility)
        ro = cfgs.scene.rigid_options
        wiring["rigid_options"] = (
            f"integrator={ro.integrator} constraint_solver={ro.constraint_solver} "
            f"iterations={ro.iterations} constraint_timeconst={ro.constraint_timeconst}"
        )
        # The solver resolves impratio=None in-place at init
        # (rigid_solver.py: elliptic → 100, pyramidal → 1).
        wiring["friction_cone"] = f"{ro.friction_cone} impratio={ro.impratio}"
    else:
        import mujoco as _mj

        mjm = env.scene_manager.solver.mj_model if sim == "newton" else env.scene_manager.mj_model
        wiring["contact_emission_rule"] = "mjwarp: contact included when dist < margin - gap (per-geom)"
        geom_rows = []
        for gid in range(mjm.ngeom):
            gname = _mj.mj_id2name(mjm, _mj.mjtObj.mjOBJ_GEOM, gid) or f"geom{gid}"
            leaf = gname.rsplit("/", 1)[-1].lower()
            is_ground = int(mjm.geom_type[gid]) == int(_mj.mjtGeom.mjGEOM_PLANE)
            if "foot" in leaf or is_ground:
                geom_rows.append(
                    {
                        "name": gname,
                        "type": int(mjm.geom_type[gid]),
                        "margin": float(mjm.geom_margin[gid]),
                        "gap": float(mjm.geom_gap[gid]),
                        "condim": int(mjm.geom_condim[gid]),
                        "friction": [round(float(v), 4) for v in mjm.geom_friction[gid]],
                        "solref": [round(float(v), 5) for v in mjm.geom_solref[gid]],
                        "solimp": [round(float(v), 5) for v in mjm.geom_solimp[gid]],
                    }
                )
        wiring["foot_and_ground_geoms"] = geom_rows
        wiring["opt.o_margin"] = float(mjm.opt.o_margin)
        if sim == "mujoco":
            mj_cfg = cfgs.scene.mjlab_sim_cfg.mujoco
            wiring["solver"] = f"iterations={mj_cfg.iterations} cone={mj_cfg.cone} impratio={mj_cfg.impratio}"
        else:
            nt = cfgs.scene.solver_cfg
            wiring["solver"] = (
                f"iterations={nt.iterations} cone={nt.cone} impratio={nt.impratio} "
                f"use_mujoco_contacts={nt.use_mujoco_contacts}"
            )

    feet_sel = env.resolve_selector(
        SceneEntitySelector(name="robot", body_names=tuple(k1.foot_names), preserve_order=True)
    )
    feet_ids = feet_sel.body_ids
    wiring["feet_resolved"] = {
        "names": list(feet_sel.body_names or []),
        "ids": [int(i) for i in feet_ids.tolist()],
        "sole_offset_m": _SOLE_OFFSET,
    }

    env.reset()
    torch.cuda.synchronize()
    _stage("reset done")

    writer = env.get_robot_state_writer()
    zero = torch.zeros((num_envs, env.num_actions), device=env.device)
    all_env_ids = torch.arange(num_envs, device=env.device)
    # act_manager.offset is (num_envs, num_actuated): the default joint
    # positions in canonical actuated order.
    default_qpos = env.act_manager.offset.clone()
    # Keep each env's post-reset xy so the teleport makes no assumption
    # about env origins/spacing.
    rest_xy = env.get_robot_data().root_link_pos_w[:, :2].clone()

    def teleport_default_pose(base_z: torch.Tensor) -> None:
        """Default joints, zero velocities, base at per-env z, identity yaw."""
        writer.set_dof_positions(default_qpos, env_ids=all_env_ids)
        writer.set_dof_velocities(torch.zeros_like(default_qpos), env_ids=all_env_ids)
        pos = torch.zeros((num_envs, 3), device=env.device)
        pos[:, :2] = rest_xy
        pos[:, 2] = base_z
        quat = torch.zeros((num_envs, 4), device=env.device)
        quat[:, 0] = 1.0
        writer.set_root_pose(pos, quat, env_ids=all_env_ids)
        writer.set_root_velocity(
            torch.zeros((num_envs, 3), device=env.device),
            torch.zeros((num_envs, 3), device=env.device),
            env_ids=all_env_ids,
        )
        writer.eval_fk(env_ids=all_env_ids)

    def sole_gap() -> torch.Tensor:
        rd = env.get_robot_data()
        return rd.body_pos_w_by_ids(feet_ids)[..., 2] - _SOLE_OFFSET

    def foot_xyspeed() -> torch.Tensor:
        rd = env.get_robot_data()
        return rd.body_lin_vel_w_by_ids(feet_ids)[..., :2].norm(dim=-1)

    def contact_and_force():
        c = env.contact_manager.is_contact("feet_ground_contact")
        f = env.contact_manager.contact_force("feet_ground_contact").norm(dim=-1)
        return c, f

    # ---- 1. threshold sweep ------------------------------------------
    # Calibrate: at the home keyframe height, where does the sole end up
    # after one zero-action step? Shift the grid so it straddles gap=0.
    teleport_default_pose(torch.full((num_envs,), k1.base_init_height, device=env.device))
    env.step(zero)
    gap_ref = float(sole_gap()[:, 0].mean())
    base_for_touch = k1.base_init_height - gap_ref
    _stage(f"calibration: nominal sole gap={gap_ref:.5f} -> base_for_touch={base_for_touch:.5f}")

    grid = torch.linspace(0.008, -0.008, num_envs, device=env.device)
    teleport_default_pose(base_for_touch + grid)
    env.step(zero)
    g1 = sole_gap()
    c1, f1 = contact_and_force()
    bins = []
    edges = torch.arange(-8.0, 8.5, 1.0, device=env.device) * 1e-3
    for i in range(len(edges) - 1):
        m = (g1[:, 0] >= edges[i]) & (g1[:, 0] < edges[i + 1])
        n = int(m.sum())
        if n == 0:
            continue
        bins.append(
            {
                "gap_mm": [round(float(edges[i]) * 1e3, 1), round(float(edges[i + 1]) * 1e3, 1)],
                "n": n,
                "found_frac": round(float(c1[m, 0].float().mean()), 3),
                "force_mean_N": round(float(f1[m, 0].mean()), 2),
            }
        )
    on_gaps = g1[:, 0][c1[:, 0]]
    off_gaps = g1[:, 0][~c1[:, 0]]
    sweep = {
        "gap_ref_at_home_keyframe_m": round(gap_ref, 5),
        "bins": bins,
        "n_contact_of_total": [int(c1[:, 0].sum()), num_envs],
        "max_gap_WITH_contact_mm": round(float(on_gaps.max()) * 1e3, 3) if on_gaps.numel() else None,
        "min_gap_WITHOUT_contact_mm": round(float(off_gaps.min()) * 1e3, 3) if off_gaps.numel() else None,
    }
    _stage("threshold sweep done")

    # ---- 2. standing-creep probe -------------------------------------
    # The pure friction-creep signal, without gait chaos: stand at touch
    # height under zero action and measure the per-step xy displacement
    # of feet that stay in contact across consecutive control steps.
    # Zero action does NOT balance the robot — episodes end within a
    # couple of seconds even when standing (the gait probe showed mean
    # episode ~66 ctrl steps). So don't demand whole-window survival:
    # mask per (env, step) instead — count a displacement sample only
    # when the env is >15 steps past its last reset (teleport included)
    # and the foot is in contact on both consecutive frames.
    teleport_default_pose(torch.full((num_envs,), base_for_touch, device=env.device))
    creep_settle = 15
    since_event = torch.zeros(num_envs, device=env.device)
    prev_xy = env.get_robot_data().body_pos_w_by_ids(feet_ids)[..., :2].clone()
    prev_in = env.contact_manager.is_contact("feet_ground_contact").clone()
    creep_disp_sum = torch.zeros((num_envs, 2), device=env.device)
    creep_step_cnt = torch.zeros((num_envs, 2), device=env.device)
    creep_steps = 125
    for _ in range(creep_steps):
        _o, _r, term, trunc, _e = env.step(zero)
        reset_now = term | trunc
        since_event = torch.where(reset_now, torch.zeros_like(since_event), since_event + 1.0)
        settled = (since_event > creep_settle).unsqueeze(-1)
        rd_now = env.get_robot_data()
        xy = rd_now.body_pos_w_by_ids(feet_ids)[..., :2]
        in_c = env.contact_manager.is_contact("feet_ground_contact")
        sample = in_c & prev_in & settled
        creep_disp_sum += (xy - prev_xy).norm(dim=-1) * sample.float()
        creep_step_cnt += sample.float()
        prev_xy = xy.clone()
        prev_in = in_c.clone()
    # Per (env, foot) mean slip, over feet with enough samples to be
    # meaningful. Crash loudly if nothing qualifies — that would mean
    # the probe design is broken for this sim, not "no data, skip".
    enough = creep_step_cnt >= 20
    if not bool(enough.any()):
        raise RuntimeError(
            f"standing-creep probe collected no (env, foot) with >=20 samples "
            f"(total samples={int(creep_step_cnt.sum())}) — probe design invalid for sim={sim}"
        )
    per_step_mm = (creep_disp_sum / creep_step_cnt.clamp(min=1.0))[enough] * 1e3
    qs3 = torch.tensor([0.1, 0.5, 0.9], device=env.device)
    creep = {
        "steps": creep_steps,
        "n_env_feet_with_20plus_samples": int(enough.sum()),
        "n_samples_total": int(creep_step_cnt.sum()),
        "per_stance_step_slip_mm": {
            "mean": round(float(per_step_mm.mean()), 4),
            "q10_50_90": [round(float(v), 4) for v in torch.quantile(per_step_mm, qs3).tolist()],
        },
    }
    _stage("standing-creep probe done")

    # ---- 3. gait probe with edge stats -------------------------------
    env.reset()
    torch.cuda.synchronize()
    names = [n.rsplit("/", 1)[-1] for n in env.act_manager.actuated_joint_names]
    legs = {
        "L": (names.index("Left_Hip_Pitch"), names.index("Left_Knee_Pitch"), names.index("Left_Ankle_Pitch")),
        "R": (names.index("Right_Hip_Pitch"), names.index("Right_Knee_Pitch"), names.index("Right_Ankle_Pitch")),
    }
    ctrl_dt = env.control_dt

    def gait_action(k: int) -> torch.Tensor:
        a = torch.zeros((num_envs, env.num_actions), device=env.device)
        s = math.sin(2.0 * math.pi * _GAIT_HZ * k * ctrl_dt)
        for leg, (hip, knee, ankle) in legs.items():
            w = (max(s, 0.0) if leg == "L" else max(-s, 0.0)) * gait_scale
            a[:, hip] = -1.0 * w
            a[:, knee] = 1.6 * w
            a[:, ankle] = -0.6 * w
        return a

    class EdgeTracker:
        """Edge/segment bookkeeping for ONE contact definition.

        Run counters update only on valid steps; segment lengths are
        recorded at the edge BEFORE the counters update, so at a falling
        edge ``stance_run`` still holds the length of the stance that
        just ended (and vice versa).
        """

        def __init__(self, c0: torch.Tensor):
            self.prev_c = c0.clone()
            self.stance_run = torch.zeros_like(c0, dtype=torch.float32)
            self.air_run = torch.zeros_like(self.stance_run)
            self.on_edges = 0
            self.off_edges = 0
            self.stance_lens: list = []
            self.air_lens: list = []
            self.off_edge_prev_force: list = []
            self.slip_sum = torch.zeros((), device=c0.device)
            self.slip_cnt = torch.zeros((), device=c0.device)

        def update(self, c, valid, xyspd, prev_force):
            rising = c & ~self.prev_c & valid
            falling = ~c & self.prev_c & valid
            if falling.any():
                self.stance_lens.append(self.stance_run[falling])
                self.off_edge_prev_force.append(prev_force[falling])
                self.off_edges += int(falling.sum())
            if rising.any():
                self.air_lens.append(self.air_run[rising])
                self.on_edges += int(rising.sum())
            self.stance_run = torch.where(c & valid, self.stance_run + 1.0, torch.zeros_like(self.stance_run))
            self.air_run = torch.where(~c & valid, self.air_run + 1.0, torch.zeros_like(self.air_run))
            self.slip_sum += (xyspd * (c & valid).float()).sum()
            self.slip_cnt += (c & valid).float().sum()
            self.prev_c = c.clone()

        def summary(self) -> dict:
            def frac_le1(chunks):
                if not chunks:
                    return None
                t = torch.cat([x.flatten() for x in chunks])
                return round(float((t <= 1.0).float().mean()), 4)

            return {
                "on_edges": self.on_edges,
                "off_edges": self.off_edges,
                "stance_len_ctrl_steps": _stats(self.stance_lens),
                "air_len_ctrl_steps": _stats(self.air_lens),
                "air_frac_len_le1": frac_le1(self.air_lens),
                "off_edge_prev_force_N": _stats(self.off_edge_prev_force),
                "in_stance_xyspeed_mean": round(float(self.slip_sum / self.slip_cnt.clamp(min=1.0)), 5),
            }

    def _stats(chunks: list) -> dict:
        if not chunks:
            return {"n": 0}
        t = torch.cat([x.flatten().float() for x in chunks])
        qs = torch.tensor([0.1, 0.5, 0.9], device=t.device)
        return {
            "n": int(t.numel()),
            "mean": round(float(t.mean()), 5),
            "q10_50_90": [round(float(v), 5) for v in torch.quantile(t, qs).tolist()],
        }

    c0, f0 = contact_and_force()
    # Three contact definitions measured on the SAME rollout:
    #   raw  — backend is_contact (genesis: contact row exists; newton:
    #          |force|>1e-6; mjlab: native found>0)
    #   f1/f5 — |force| above a uniform threshold, the candidate
    #          debounced definition (IsaacLab convention is ~1 N).
    # raw_upright / raw_fallen: same raw definition, but edges counted
    # only while the base is above/below 0.4 m — separates "stepping
    # while standing" cycling from "flailing on the ground" cycling.
    trackers = {
        "raw": EdgeTracker(c0),
        "raw_upright": EdgeTracker(c0),
        "raw_fallen": EdgeTracker(c0),
        "force_gt_1N": EdgeTracker(f0 > 1.0),
        "force_gt_5N": EdgeTracker(f0 > 5.0),
    }
    prev_force = f0.clone()
    # Exclude the first settle_exclude steps after each env reset: the
    # post-reset drop produces edges that have nothing to do with gait.
    settle_exclude = 10
    since_reset = torch.zeros(num_envs, device=env.device)
    n_resets = 0
    for k in range(probe_steps):
        _obs, _rew, terminated, truncated, _extras = env.step(gait_action(k))
        reset_now = terminated | truncated
        since_reset = torch.where(reset_now, torch.zeros_like(since_reset), since_reset + 1.0)
        valid = (since_reset > settle_exclude).unsqueeze(-1).expand(num_envs, 2)
        upright = (env.get_robot_data().root_link_pos_w[:, 2] > 0.4).unsqueeze(-1).expand(num_envs, 2)
        c, f = contact_and_force()
        xyspd = foot_xyspeed()
        trackers["raw"].update(c, valid, xyspd, prev_force)
        trackers["raw_upright"].update(c, valid & upright, xyspd, prev_force)
        trackers["raw_fallen"].update(c, valid & ~upright, xyspd, prev_force)
        trackers["force_gt_1N"].update(f > 1.0, valid, xyspd, prev_force)
        trackers["force_gt_5N"].update(f > 5.0, valid, xyspd, prev_force)
        n_resets += int(reset_now.sum())
        prev_force = f.clone()
    _stage(f"gait probe done ({n_resets} env resets during probe)")

    probe = {
        "n_env_resets_during_probe": n_resets,
        "settle_exclude_steps": settle_exclude,
        "definitions": {name: tr.summary() for name, tr in trackers.items()},
    }

    return {"sim": sim, "wiring": wiring, "sweep": sweep, "creep": creep, "probe": probe}


# ── Parent: orchestrate the three cells + report ────────────────────


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
                    "--probe-steps",
                    str(args.probe_steps),
                    "--gait-scale",
                    str(args.gait_scale),
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
    L: list[str] = []
    L.append("=" * 110)
    L.append("K1 contact margin/gap parity — effective contact-threshold measurement")
    L.append("=" * 110)
    L.append(f"num_envs: {args.num_envs}  probe_steps: {args.probe_steps}  seed: {args.seed}  logs: {log_dir}")
    L.append("")
    for s in sims:
        L.append(f"── [{s}] wiring (every margin/gap-like knob) ──")
        for k, v in results[s]["wiring"].items():
            if k == "foot_and_ground_geoms":
                L.append(f"  {k}:")
                for row in v:
                    L.append(f"    {row}")
            else:
                L.append(f"  {k}: {v}")
        L.append("")

    L.append("── THRESHOLD SWEEP (left-foot sole gap [mm] vs contact found) ──")
    for s in sims:
        sw = results[s]["sweep"]
        L.append(
            f"  [{s}] gap_ref(home)={sw['gap_ref_at_home_keyframe_m']}  "
            f"contact {sw['n_contact_of_total'][0]}/{sw['n_contact_of_total'][1]}  "
            f"max gap WITH contact = {sw['max_gap_WITH_contact_mm']} mm  "
            f"min gap WITHOUT = {sw['min_gap_WITHOUT_contact_mm']} mm"
        )
        for row in sw["bins"]:
            L.append(
                f"      gap {row['gap_mm']} mm  n={row['n']:<4d} "
                f"found={row['found_frac']:<6} force={row['force_mean_N']} N"
            )
        L.append("")

    L.append("── STANDING CREEP (zero action at touch height; friction creep isolated) ──")
    L.append("  The one engine-level physics delta already known: mjwarp regularized")
    L.append("  friction lets stance feet drift; genesis pyramidal cone holds them.")
    L.append("  genesis_elliptic tests whether cone alignment closes the gap.")
    for s in sims:
        cr = results[s]["creep"]
        L.append(
            f"  [{s}] env-feet(>=20 samples)={cr['n_env_feet_with_20plus_samples']} "
            f"samples={cr['n_samples_total']}  "
            f"slip/stance-step [mm]: {cr['per_stance_step_slip_mm']}"
        )
    L.append("")

    L.append("── GAIT PROBE (identical open-loop gait) — same rollout, 3 contact definitions ──")
    L.append("  If the cross-sim spread shrinks under force_gt_1N/5N vs raw, micro-chatter")
    L.append("  in the raw contact signal is the divergence mechanism and a uniform force")
    L.append("  threshold is the candidate unification.")
    for s in sims:
        p = results[s]["probe"]
        L.append(
            f"  [{s}] resets during probe: {p['n_env_resets_during_probe']} "
            f"(first {p['settle_exclude_steps']} steps post-reset excluded)"
        )
        for name, d in p["definitions"].items():
            L.append(
                f"    [{name}] ON/OFF edges: {d['on_edges']}/{d['off_edges']}  air_frac_len<=1: {d['air_frac_len_le1']}"
            )
            L.append(f"      stance len [steps]: {d['stance_len_ctrl_steps']}")
            L.append(f"      air len    [steps]: {d['air_len_ctrl_steps']}")
            L.append(f"      force at OFF edge (prev frame) [N]: {d['off_edge_prev_force_N']}")
            L.append(f"      in-stance xy speed [m/s]: {d['in_stance_xyspeed_mean']}")
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
    ap.add_argument("--num-envs", type=int, default=512)
    ap.add_argument("--probe-steps", type=int, default=300)
    ap.add_argument("--gait-scale", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="k1_contact_margin_parity_diag.txt")
    args = ap.parse_args()

    if args.cell is not None:
        result = run_cell(args.cell, args.num_envs, args.probe_steps, args.seed, args.gait_scale)
        Path(args.result_json).write_text(json.dumps(result, indent=2))
        return 0
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
