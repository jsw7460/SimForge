"""Substep-resolution touchdown profile: WHERE does mjwarp micro-rebound?

Established so far: static contact thresholds, forces, and standing
creep are in 3-sim parity, yet under identical motion the mjwarp sims
(newton, mujoco) break and re-form foot contact ~40% more often than
genesis, and per-sim training deterministically converges to different
gait styles (seed-invariant). The remaining question is the impact
dynamics itself.

This diag drops the robot from a small height (default pose, zero
velocity, zero action = PD hold) and records EVERY PHYSICS SUBSTEP
(dt=0.005) around touchdown:

    sole gap, foot vertical velocity, contact found, |contact force|

per foot per env, for several drop heights. From the traces we
extract, per sim:

- impact velocity at first contact
- max penetration
- REBOUND: fraction of feet that lose contact again after first
  touchdown, number of found on/off transitions in the window,
  max positive (upward) foot velocity after first contact
  ("numerical restitution"), and the gap reached on the rebound
- settle time (substeps from first contact until contact stays on
  for 20 consecutive substeps)

If mjwarp shows a rebound (positive vz / re-separation) where genesis
shows monotone settling, the divergence mechanism is localized to the
contact-impulse response at touchdown, and the source-level audit of
the two constraint solvers has a concrete signal to explain.

Run (server, from SimForge root):
    python -m rlworld.scripts.diag.k1.k1_impact_profile_diag
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

_MODULE = "rlworld.scripts.diag.k1.k1_impact_profile_diag"
_SIMS = ("genesis", "genesis_elliptic", "newton", "mujoco")
_SIM_KEY = {"genesis": "Genesis", "newton": "Newton", "mujoco": "MujocoEnv"}
_SOLE_OFFSET = 0.048
_DROPS_MM = (2.0, 5.0, 10.0)


def _stage(msg: str) -> None:
    print(f"[STAGE] {msg}", flush=True)


# ── Child ───────────────────────────────────────────────────────────


def run_cell(sim: str, num_envs: int, seed: int) -> dict:
    import torch

    torch.manual_seed(seed)
    _stage(f"cell start: {sim} num_envs={num_envs}")

    from rlworld.rl.configs.presets.k1_joystick.g1_recipe import K1G1RecipeConfig
    from rlworld.rl.configs.robots.k1 import K1Config
    from rlworld.rl.configs.scene import SceneEntitySelector
    from rlworld.rl.evals.sim_initializers import get_initializer

    k1 = K1Config()
    backend = "genesis" if sim.startswith("genesis") else sim
    cfgs = K1G1RecipeConfig(sim_type=backend, num_envs=num_envs, seed=seed).build()
    if sim == "genesis_elliptic":
        import genesis as gs

        # Same cone/impratio family as the mjwarp canonical recipe
        # (impratio=None auto-resolves to 100 for elliptic).
        cfgs.scene.rigid_options.friction_cone = gs.friction_cone.elliptic
    env = get_initializer(_SIM_KEY[backend]).init_environment(cfgs)
    _stage("env built")

    feet_sel = env.resolve_selector(
        SceneEntitySelector(name="robot", body_names=tuple(k1.foot_names), preserve_order=True)
    )
    feet_ids = feet_sel.body_ids

    env.reset()
    torch.cuda.synchronize()
    _stage("reset done")

    writer = env.get_robot_state_writer()
    all_env_ids = torch.arange(num_envs, device=env.device)
    default_qpos = env.act_manager.offset.clone()
    rest_xy = env.get_robot_data().root_link_pos_w[:, :2].clone()
    zero = torch.zeros((num_envs, env.num_actions), device=env.device)
    # PD-hold at the default pose for every substep of the experiment.
    env.act_manager.process_actions(zero)

    def teleport(base_z: torch.Tensor, roll_deg: float = 0.0) -> None:
        writer.set_dof_positions(default_qpos, env_ids=all_env_ids)
        writer.set_dof_velocities(torch.zeros_like(default_qpos), env_ids=all_env_ids)
        pos = torch.zeros((num_envs, 3), device=env.device)
        pos[:, :2] = rest_xy
        pos[:, 2] = base_z
        quat = torch.zeros((num_envs, 4), device=env.device)
        half = math.radians(roll_deg) / 2.0
        quat[:, 0] = math.cos(half)
        quat[:, 1] = math.sin(half)
        writer.set_root_pose(pos, quat, env_ids=all_env_ids)
        writer.set_root_velocity(
            torch.zeros((num_envs, 3), device=env.device),
            torch.zeros((num_envs, 3), device=env.device),
            env_ids=all_env_ids,
        )
        writer.eval_fk(env_ids=all_env_ids)

    # One physics SUBSTEP, mirroring each backend's _step_physics body
    # (rlworld/rl/envs/{genesis,newton,mujoco}/*_env.py) without the
    # decimation loop or the manager/reward machinery on top.
    if backend == "genesis":

        def substep() -> None:
            env.act_manager.apply_actions(env.act_manager.processed_actions)
            env.scene_manager.step()
            env._invalidate_cache()
            env.contact_manager.advance(dt=env.physics_dt)

    elif sim == "newton":

        def substep() -> None:
            env.scene_manager.state_0.clear_forces()
            env.act_manager.apply_actions(env.act_manager.processed_actions)
            env.scene_manager.step()
            env._invalidate_cache()
            env.contact_manager.advance(dt=env.physics_dt)

    else:  # mujoco / mjlab

        def substep() -> None:
            env.act_manager.apply_actions(env.act_manager.processed_actions)
            env.scene_manager.write_data_to_sim()
            env.scene_manager.step()
            env.scene_manager.update(dt=env.physics_dt)
            env._invalidate_cache()
            env.contact_manager.advance(dt=env.physics_dt)

    def read_state():
        rd = env.get_robot_data()
        gap = rd.body_pos_w_by_ids(feet_ids)[..., 2] - _SOLE_OFFSET
        vz = rd.body_lin_vel_w_by_ids(feet_ids)[..., 2]
        found = env.contact_manager.is_contact("feet_ground_contact")
        force = env.contact_manager.contact_force("feet_ground_contact").norm(dim=-1)
        return gap, vz, found, force

    # Calibrate the just-touching base height (same approach as the
    # margin-parity diag): teleport at home keyframe, settle briefly,
    # read the mean sole gap.
    base_home = float(k1.base_init_height)
    teleport(torch.full((num_envs,), base_home, device=env.device))
    for _ in range(8):
        substep()
    gap0, _, _, _ = read_state()
    base_touch = base_home - float(gap0.mean())
    _stage(f"calibration: base_touch={base_touch:.5f}")

    settle_need = 20

    def run_trial(n_sub: int, per_substep=None) -> dict:
        """Run n_sub substeps from the CURRENT state and extract impact
        metrics. ``per_substep(t)`` (optional) runs before substep t —
        used by the drag trial to re-inject lateral root velocity."""
        T_gap, T_vz, T_found, T_force = [], [], [], []
        for t in range(n_sub):
            if per_substep is not None:
                per_substep(t)
            substep()
            g, v, c, f = read_state()
            T_gap.append(g.clone())
            T_vz.append(v.clone())
            T_found.append(c.clone())
            T_force.append(f.clone())
        gap = torch.stack(T_gap)  # (T, N, 2)
        vz = torch.stack(T_vz)
        found = torch.stack(T_found)
        force = torch.stack(T_force)
        T = gap.shape[0]

        ever = found.any(dim=0)  # (N,2) feet that touched
        first = torch.where(
            ever,
            found.float().argmax(dim=0),
            torch.full_like(found[0], T, dtype=torch.long),
        )
        t_idx = torch.arange(T, device=env.device).view(T, 1, 1)
        after = t_idx > first.unsqueeze(0)  # strictly after first contact
        at_first = t_idx == first.unsqueeze(0)

        impact_vz = (vz * at_first.float()).sum(dim=0)
        max_pen = (-gap).max(dim=0).values
        lost_after = ((~found) & after).any(dim=0) & ever
        n_transitions = ((found[1:] != found[:-1]) & after[1:]).float().sum(dim=0) + 1.0
        max_up_vz = (vz * after.float()).max(dim=0).values
        max_regap = (gap * ((~found) & after).float()).max(dim=0).values
        run = torch.zeros_like(found[0], dtype=torch.float32)
        settle_t = torch.full_like(found[0], -1, dtype=torch.float32)
        for t in range(T):
            run = torch.where(found[t], run + 1.0, torch.zeros_like(run))
            hit = (run == settle_need) & (settle_t < 0)
            settle_t = torch.where(hit, torch.full_like(settle_t, float(t - settle_need + 1)), settle_t)

        ev = ever.flatten()

        def m(x, mask=None, nd=4):
            x = x.flatten().float()
            msk = ev if mask is None else mask.flatten()
            if int(msk.sum()) == 0:
                return None
            return round(float(x[msk].mean()), nd)

        in_contact = found & (t_idx <= T)  # all contact samples
        force_std = force[in_contact].std() if int(in_contact.sum()) else torch.zeros(())
        return {
            "n_feet_touched": int(ev.sum()),
            "impact_vz_mean": m(impact_vz),
            "max_penetration_mm": m(max_pen * 1e3, nd=3),
            "rebound_frac": round(float(lost_after.flatten()[ev].float().mean()), 4),
            "transitions_mean": m(n_transitions, mask=ev, nd=2),
            "max_upward_vz_after_impact": m(max_up_vz),
            "max_reseparation_gap_mm": m(max_regap * 1e3, mask=lost_after, nd=3),
            "settle_substeps_mean": m(settle_t, mask=(settle_t >= 0) & ever, nd=1),
            "frac_not_settled": round(float(((settle_t < 0) & ever).flatten()[ev].float().mean()), 4),
            "contact_frac": round(float(found.flatten(1)[:, ev].float().mean()), 4),
            "in_contact_force_std_N": round(float(force_std), 2),
            "trace_contact_frac": [round(float(found[t].flatten()[ev].float().mean()), 3) for t in range(min(40, T))],
            "trace_gap_mm": [round(float(gap[t].flatten()[ev].mean()) * 1e3, 3) for t in range(min(40, T))],
            "trace_vz": [round(float(vz[t].flatten()[ev].mean()), 4) for t in range(min(40, T))],
            "trace_force": [round(float(force[t].flatten()[ev].mean()), 1) for t in range(min(40, T))],
        }

    results = {}
    # ---- 1. vertical drops (flat foot, full load) --------------------
    for drop_mm in _DROPS_MM:
        teleport(torch.full((num_envs,), base_touch + drop_mm * 1e-3, device=env.device))
        results[f"drop_{drop_mm:g}mm"] = run_trial(120)
        _stage(f"drop {drop_mm}mm done")

    # ---- 2. shear touchdowns: drop 5mm WITH lateral velocity ---------
    # The gait regime where the engines diverged has tangential motion
    # at contact. Give the whole robot a lateral velocity at release.
    for vx in (0.15, 0.4):
        teleport(torch.full((num_envs,), base_touch + 5e-3, device=env.device))
        lin = torch.zeros((num_envs, 3), device=env.device)
        lin[:, 0] = vx
        writer.set_root_velocity(lin, torch.zeros_like(lin), env_ids=all_env_ids)
        results[f"shear_vx{vx:g}"] = run_trial(120)
        _stage(f"shear vx={vx} done")

    # ---- 2b. tilted touchdown: 12° roll → edge/partial-geom contact --
    # The flailing regime lands on foot edges, not flat soles. With the
    # base rolled, only one side's spheres/box edge engage.
    teleport(torch.full((num_envs,), base_touch + 5e-3, device=env.device), roll_deg=12.0)
    results["tilt12_drop5mm"] = run_trial(120)
    _stage("tilt drop done")

    # ---- 2c. grazing touchdown: shallow drop at high lateral speed ---
    teleport(torch.full((num_envs,), base_touch + 2e-3, device=env.device))
    lin = torch.zeros((num_envs, 3), device=env.device)
    lin[:, 0] = 1.0
    writer.set_root_velocity(lin, torch.zeros_like(lin), env_ids=all_env_ids)
    results["graze_vx1.0"] = run_trial(120)
    _stage("graze done")

    # ---- 3. loaded drag: standing, lateral root kick every 25 substeps
    # Sustained shear on loaded feet (the push/slide regime). The kick
    # keeps the current vertical/angular state and only overwrites vx.
    teleport(torch.full((num_envs,), base_touch, device=env.device))
    for _ in range(20):
        substep()

    def kick(t: int) -> None:
        if t % 25 != 0:
            return
        rd = env.get_robot_data()
        lin = rd.root_link_lin_vel_w.clone()
        ang = rd.root_link_ang_vel_w.clone()
        lin[:, 0] = 0.3
        writer.set_root_velocity(lin, ang, env_ids=all_env_ids)

    results["drag_kick_vx0.3"] = run_trial(100, per_substep=kick)
    _stage("drag trial done")

    return {"sim": sim, "results": results}


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
    L.append("K1 touchdown impact profile — per-substep (5 ms) traces around first contact")
    L.append("=" * 110)
    L.append(f"num_envs: {args.num_envs}  seed: {args.seed}  logs: {log_dir}")
    L.append("")
    trial_keys = list(results[sims[0]]["results"].keys()) if sims else []
    for trial in trial_keys:
        L.append(f"── {trial} ──")
        for s in sims:
            r = results[s]["results"][trial]
            L.append(
                f"  [{s}] impact_vz={r['impact_vz_mean']}  max_pen={r['max_penetration_mm']}mm  "
                f"REBOUND frac={r['rebound_frac']}  transitions={r['transitions_mean']}  "
                f"max_up_vz={r['max_upward_vz_after_impact']}  "
                f"resep_gap={r['max_reseparation_gap_mm']}mm  "
                f"settle={r['settle_substeps_mean']} substeps (unsettled {r['frac_not_settled']})  "
                f"cfrac={r['contact_frac']}  force_std={r['in_contact_force_std_N']}N"
            )
        L.append("")
        L.append("    substep traces (mean over touched feet): contact_frac | gap[mm] | vz | force[N]")
        for s in sims:
            r = results[s]["results"][trial]
            L.append(f"    [{s}]")
            L.append(f"      cfrac: {r['trace_contact_frac']}")
            L.append(f"      gap  : {r['trace_gap_mm']}")
            L.append(f"      vz   : {r['trace_vz']}")
            L.append(f"      force: {r['trace_force']}")
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
    ap.add_argument("--num-envs", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="k1_impact_profile_diag.txt")
    args = ap.parse_args()

    if args.cell is not None:
        result = run_cell(args.cell, args.num_envs, args.seed)
        Path(args.result_json).write_text(json.dumps(result, indent=2))
        return 0
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
