"""K1 foot-ground contact friction (mu): written vs measured, across resets.

Two questions, per backend (Genesis / Newton / mjlab):

1. Does the friction DR land in the solver? ``written`` is the DR'd
   foot-geom friction parameter, ``measured`` is the friction on the
   solver's actual foot-ground contact rows. The two must agree.
2. Does it SURVIVE training-like resets? Friction DR runs in startup
   mode, so after episodes terminate and envs reset the contact rows
   must still carry the startup values. A backend whose reset path
   re-syncs model fields from the host template would silently revert
   friction to the MJCF value — invisible to a settle-only probe but
   dominant over a training run.

Both phases therefore report a per-contact MATCH test: every touching
foot-ground contact's mu must equal one of its env's written foot-geom
values (tol 1e-4). ``n_mu_mismatch`` > 0 means the solver is using a
friction that no DR write produced (e.g. the raw MJCF 0.6).

Phase layout: settle 50 -> measure (post_settle) -> 300 random-action
steps (mass terminations/resets, like early training) -> settle 50 ->
measure (post_resets).

Run (server, from SimForge root):
    python -m jaxrlworld.scripts.diag.k1.k1_contact_mu_parity_diag --num-envs 1024
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

_MODULE = "jaxrlworld.scripts.diag.k1.k1_contact_mu_parity_diag"
_SIMS = ("genesis", "newton", "mujoco")
_SIM_KEY = {"genesis": "Genesis", "newton": "Newton", "mujoco": "MujocoEnv"}
_SETTLE = 50
_RANDOM_STEPS = 300
_MATCH_TOL = 1e-4


def _stage(msg: str) -> None:
    print(f"[STAGE] {msg}", flush=True)


def _stats(vals) -> dict:
    import torch

    t = torch.as_tensor(vals, dtype=torch.float32).flatten()
    if t.numel() == 0:
        return {"n": 0}
    qs = torch.quantile(t, torch.tensor([0.1, 0.5, 0.9]))
    return {
        "n": int(t.numel()),
        "min": round(float(t.min()), 4),
        "q10": round(float(qs[0]), 4),
        "q50": round(float(qs[1]), 4),
        "q90": round(float(qs[2]), 4),
        "max": round(float(t.max()), 4),
        "first8": [round(float(v), 4) for v in t[:8].tolist()],
    }


def _np(x):
    """Host numpy copy of a torch tensor or warp array.

    mjlab allocates its mjwarp buffers via ``wp.from_torch``, so their
    ``.numpy()`` delegates to the wrapped CUDA torch tensor and raises;
    route warp arrays through ``wp.to_torch`` (zero-copy) instead.
    """
    import torch
    import warp as wp

    if isinstance(x, torch.Tensor):
        t = x
    elif isinstance(x, wp.array):
        t = wp.to_torch(x)
    else:
        # Tensor proxies (e.g. mjlab's sim_data.TorchArray) delegate
        # attribute access to their underlying torch tensor.
        t = x.detach()
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"unsupported array type: {type(x)}")
    return t.detach().contiguous().cpu().numpy()


# ── measurement (one call per phase) ─────────────────────────────────


def _measure_mjwarp(wp_model, wp_data, foot_gids: set, plane_gids: set) -> dict:
    import numpy as np

    nacon = int(_np(wp_data.nacon)[0])
    geom = _np(wp_data.contact.geom)[:nacon]
    dist = _np(wp_data.contact.dist)[:nacon]
    fric = _np(wp_data.contact.friction)[:nacon, 0]
    world = _np(wp_data.contact.worldid)[:nacon]

    a_foot = np.isin(geom[:, 0], list(foot_gids))
    b_foot = np.isin(geom[:, 1], list(foot_gids))
    a_gnd = np.isin(geom[:, 0], list(plane_gids))
    b_gnd = np.isin(geom[:, 1], list(plane_gids))
    touching = ((a_foot & b_gnd) | (b_foot & a_gnd)) & (dist < 0.0)

    gf = _np(wp_model.geom_friction)  # (nworld, ngeom, 3)
    written = gf[:, sorted(foot_gids), 0]  # (nworld, n_foot)

    mu = fric[touching]
    w = written[world[touching]]  # (n_c, n_foot)
    mismatch = np.abs(w - mu[:, None]).min(axis=1) > _MATCH_TOL

    return {
        "n_contacts_total": nacon,
        "n_foot_ground_touching": int(touching.sum()),
        "n_worlds_with_contact": int(np.unique(world[touching]).size),
        "written_foot_mu": _stats(written),
        "measured_contact_mu": _stats(mu),
        "n_mu_mismatch": int(mismatch.sum()),
        "mismatch_examples": [round(float(v), 4) for v in mu[mismatch][:8].tolist()],
    }


def _measure_genesis(solver, foot_geoms: list[int], ground_geoms: list[int]) -> dict:
    import torch
    from genesis.utils.misc import qd_to_torch

    base_mu = qd_to_torch(solver.dyn_info.geoms.friction, copy=True)[foot_geoms]
    ratio = solver.get_geoms_friction_ratio(geoms_idx=foot_geoms)  # (B, n_foot)
    written = base_mu.unsqueeze(0).to(ratio.device) * ratio

    cs = solver.collider._collider_state
    n_con = qd_to_torch(cs.n_contacts, copy=True).long()  # (B,)
    fric = qd_to_torch(cs.contact_data.friction, transpose=True, copy=True)  # (B, max_c)
    geom_a = qd_to_torch(cs.contact_data.geom_a, transpose=True, copy=True).long()
    geom_b = qd_to_torch(cs.contact_data.geom_b, transpose=True, copy=True).long()

    max_c = fric.shape[1]
    active = torch.arange(max_c, device=n_con.device).unsqueeze(0) < n_con.unsqueeze(1)
    foot_t = torch.tensor(sorted(foot_geoms), device=geom_a.device)
    gnd_t = torch.tensor(sorted(ground_geoms), device=geom_a.device)
    pair = active & (
        (torch.isin(geom_a, foot_t) & torch.isin(geom_b, gnd_t))
        | (torch.isin(geom_b, foot_t) & torch.isin(geom_a, gnd_t))
    )

    idx = pair.nonzero(as_tuple=False)
    mu = fric[pair]
    w = written.to(mu.device)[idx[:, 0]]  # (n_c, n_foot)
    mismatch = (w - mu.unsqueeze(1)).abs().min(dim=1).values > _MATCH_TOL

    return {
        "n_contacts_total": int(n_con.sum()),
        "n_foot_ground_touching": int(pair.sum()),
        "n_worlds_with_contact": int(pair.any(dim=1).sum()),
        "written_foot_mu": _stats(written),
        "ground_geom_mu": _stats(qd_to_torch(solver.dyn_info.geoms.friction, copy=True)[ground_geoms]),
        "measured_contact_mu": _stats(mu),
        "n_mu_mismatch": int(mismatch.sum()),
        "mismatch_examples": [round(float(v), 4) for v in mu[mismatch][:8].tolist()],
    }


def _host_geom_ids(mj_model, foot_names: list[str]) -> tuple[set, set]:
    """Foot geom ids (by owning-body name match) + plane geom ids."""
    import mujoco

    foot_gids: set = set()
    plane_gids: set = set()
    for gid in range(mj_model.ngeom):
        if mj_model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_PLANE:
            plane_gids.add(gid)
            continue
        body_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, int(mj_model.geom_bodyid[gid])) or ""
        if any(fn in body_name for fn in foot_names):
            # collision geoms only
            if mj_model.geom_contype[gid] or mj_model.geom_conaffinity[gid]:
                foot_gids.add(gid)
    if not foot_gids:
        raise RuntimeError(f"no foot collision geoms matched {foot_names}")
    if not plane_gids:
        raise RuntimeError("no plane geom found on host model")
    return foot_gids, plane_gids


def run_cell(sim: str, num_envs: int, seed: int) -> dict:
    import torch

    torch.manual_seed(seed)
    _stage(f"cell start: {sim} num_envs={num_envs}")

    from jaxrlworld.rl.configs.presets.k1_joystick.base import K1JoystickConfig
    from jaxrlworld.rl.evals.sim_initializers import get_initializer

    preset = K1JoystickConfig(sim_type=sim, num_envs=num_envs, seed=seed)
    foot_names = list(preset.robot.foot_names)
    cfgs = preset.build()
    env = get_initializer(_SIM_KEY[sim]).init_environment(cfgs)
    _stage("env built (startup DR applied)")

    sm = env.scene_manager
    if sim == "genesis":
        solver = sm.scene.sim.rigid_solver
        robot = sm["robot"]
        foot_geoms: list[int] = []
        for link in robot.links:
            if any(fn in link.name for fn in foot_names):
                foot_geoms.extend(range(link.geom_start, link.geom_end))
        if not foot_geoms:
            raise RuntimeError(f"no genesis foot geoms matched {foot_names}")
        ground_geoms: list[int] = []
        for link in sm.terrain.entity.links:
            ground_geoms.extend(range(link.geom_start, link.geom_end))

        def measure() -> dict:
            return _measure_genesis(solver, foot_geoms, ground_geoms)
    elif sim == "newton":
        solver = sm.solver
        foot_gids, plane_gids = _host_geom_ids(solver.mj_model, foot_names)

        def measure() -> dict:
            return _measure_mjwarp(solver.mjw_model, solver.mjw_data, foot_gids, plane_gids)
    else:  # mujoco / mjlab
        foot_gids, plane_gids = _host_geom_ids(sm.mj_model, foot_names)

        def measure() -> dict:
            return _measure_mjwarp(sm.model, sm.data, foot_gids, plane_gids)

    out: dict = {"sim": sim, "foot_names": foot_names}

    zero = torch.zeros((num_envs, env.num_actions), device=env.device)
    for _ in range(_SETTLE):
        env.step(zero)
    _stage(f"settled {_SETTLE} steps")
    out["post_settle"] = measure()

    # Training-like reset churn: random actions topple most envs, so the
    # termination -> reset -> (reset-mode events) path runs many times.
    gen = torch.Generator(device="cpu").manual_seed(seed + 1)
    resets = 0
    for _ in range(_RANDOM_STEPS):
        a = torch.randn((num_envs, env.num_actions), generator=gen, device="cpu").to(env.device)
        _o, _r, term_b, trunc_b, _e = env.step(a)
        resets += int((term_b | trunc_b).sum())
    for _ in range(_SETTLE):
        env.step(zero)
    _stage(f"random window done ({resets} resets)")
    out["resets_during_window"] = resets
    out["post_resets"] = measure()

    _stage("contact read done")
    return out


# ── Parent orchestration ─────────────────────────────────────────────


def _print_phase(label: str, r: dict) -> None:
    print(
        f"  [{label}] contacts={r.get('n_contacts_total')}  foot_ground={r.get('n_foot_ground_touching')}  worlds={r.get('n_worlds_with_contact')}"
    )
    print(f"    written_foot_mu     : {r.get('written_foot_mu')}")
    if "ground_geom_mu" in r:
        print(f"    ground_geom_mu      : {r.get('ground_geom_mu')}")
    print(f"    measured_contact_mu : {r.get('measured_contact_mu')}")
    print(f"    n_mu_mismatch       : {r.get('n_mu_mismatch')}  examples={r.get('mismatch_examples')}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", choices=_SIMS)
    ap.add_argument("--num-envs", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.cell:
        result = run_cell(args.cell, args.num_envs, args.seed)
        print("RESULT_JSON:" + json.dumps(result))
        return 0

    logdir = Path.cwd() / "k1_contact_mu_parity_diag_logs"
    logdir.mkdir(exist_ok=True)
    results: dict = {}
    for sim in _SIMS:
        print(f"[diag] running {sim} ...", flush=True)
        t0 = time.time()
        log_path = logdir / f"{sim}.log"
        with open(log_path, "w") as fh:
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    _MODULE,
                    "--cell",
                    sim,
                    "--num-envs",
                    str(args.num_envs),
                    "--seed",
                    str(args.seed),
                ],
                stdout=fh,
                stderr=subprocess.STDOUT,
                cwd=str(Path.cwd()),
            )
        dt = time.time() - t0
        if proc.returncode != 0:
            print(f"[diag]   -> CRASH (see {log_path}) ({dt:.0f}s)")
            continue
        for line in log_path.read_text().splitlines():
            if line.startswith("RESULT_JSON:"):
                results[sim] = json.loads(line[len("RESULT_JSON:") :])
        print(f"[diag]   -> ok ({dt:.0f}s)")

    print("\n" + "=" * 100)
    print("K1 foot-ground contact mu: written (DR'd foot param) vs measured, before/after resets")
    print("=" * 100)
    for sim in _SIMS:
        if sim not in results:
            print(f"\n[{sim}] MISSING (crashed)")
            continue
        r = results[sim]
        print(f"\n[{sim}]  resets_during_window={r.get('resets_during_window')}")
        _print_phase("post_settle", r["post_settle"])
        _print_phase("post_resets", r["post_resets"])
    print(
        "\nInterpretation: in BOTH phases, every backend must show measured within the DR range"
        "\nwith n_mu_mismatch == 0. A backend whose post_resets phase shows mismatches (e.g. mu"
        "\nback at the raw MJCF value) is losing the startup DR through its reset path."
    )
    return 0 if len(results) == len(_SIMS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
