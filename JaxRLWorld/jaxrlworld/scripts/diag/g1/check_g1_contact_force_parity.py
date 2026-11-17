"""Systematic cross-sim contact-force parity diagnostic for g1_29dof.

Drives Genesis + MuJoCo (optionally also Newton) through one reset + one
step with the **same seed and same num_envs**, dumps every plausibly
relevant quantity that feeds into ``soft_landing`` / ``feet_slip`` (and
the contact pipeline that produces them) to a per-sim JSON, and prints
a side-by-side comparison table.

Observed regression: at first PPO step, ``soft_landing`` reward differs
~3.5×–8× between Genesis and Newton/MuJoCo. The dump separates the
candidate causes:

  * **Static config** — solver name / iterations / impratio,
    constraint_timeconst, contact_pruning_tolerance (Genesis only),
    foot geom sol_params (timeconst, dampratio, dmin/dmax/...).
  * **Initial state** — root z, foot xyz, command vector, command-
    active mask, foot xy speed.
  * **Per-foot contact data** — summed force magnitude, force vector,
    is_contact / first bool, contact count (where the native sim API
    exposes it).
  * **Reward breakdown** — soft_landing and feet_slip recomputed from
    the same RewardTermConfig the preset registers, every intermediate
    tensor logged.

Run modes:

  # Default — driver mode. Spawns Genesis + MuJoCo as subprocesses (CUDA
  # context isolation), reads back their JSONs, prints comparison.
  python -m jaxrlworld.scripts.diag.g1.check_g1_contact_force_parity

  # Add Newton:
  python -m jaxrlworld.scripts.diag.g1.check_g1_contact_force_parity --sims genesis,mujoco,newton

  # Single-combo inline (debugging):
  python -m jaxrlworld.scripts.diag.g1.check_g1_contact_force_parity --sim genesis --no-driver

Output: per-sim JSON saved as ``./g1_contact_<sim>.json``, comparison
table printed to stdout and saved to ``./g1_contact_compare.txt``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from jaxrlworld.rl.configs.presets.g1_29dof.base import G1FlatConfig
from jaxrlworld.rl.configs.scene.entity_selector import SceneEntitySelector
from jaxrlworld.rl.envs.mdp.rewards.common.reward_terms import (
    _command_active,
    _feet_contact_order,
    _foot_pos_vel,
    penalize_feet_slip,
    penalize_soft_landing,
)
from jaxrlworld.rl.runners import BaseRunner

os.environ.setdefault("JAXRLWORLD_ALLOW_MULTI_SIM", "1")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_jsonable(value: Any) -> Any:
    """Convert tensors / numpy / etc. into JSON-serializable primitives.

    Tensors are detached, moved to CPU, then converted to nested lists.
    Floats are kept as-is. ``None`` passes through. Unknown types fall
    back to ``str(value)`` for transparency.
    """
    if isinstance(value, torch.Tensor):
        t = value.detach().cpu()
        if t.dtype == torch.bool:
            return t.tolist()
        return t.float().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating | np.integer):
        return value.item()
    if isinstance(value, list | tuple):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if value is None or isinstance(value, bool | int | float | str):
        return value
    return str(value)


def _try_arr(maybe_arr: Any) -> Any:
    """Convert a torch tensor or a Genesis quadrants tensor (``qd.Tensor``) into a Python list."""
    if maybe_arr is None:
        return None
    if isinstance(maybe_arr, torch.Tensor):
        return maybe_arr.detach().cpu().tolist()
    return maybe_arr.to_numpy().tolist()


def _bridge_arr(x: Any) -> list:
    """Convert an mjlab ``TorchArray`` (a mujoco-warp field behind the torch bridge) into a Python list."""
    return x[...].detach().cpu().tolist()


def _mj_resolve_body(mj, candidates: tuple[str, ...]) -> tuple[str, int]:
    """Return ``(name, body_id)`` of the first candidate that names a body of ``mj``."""
    names = [mj.body(b).name for b in range(mj.nbody)]
    for cand in candidates:
        if cand in names:
            return cand, names.index(cand)
    raise ValueError(f"none of {candidates} names a body of the MjModel")


def _genesis_dump(env, cfg) -> dict:
    """Genesis-specific deep introspection.

    Captures every quantity that could explain higher contact force
    magnitudes:
      - native contact count per env
      - per-contact (position, normal, penetration, force vector) at the
        moment of capture
      - foot link mass + total robot mass + foot collision geom info
      - effective sol_params per foot/ground geom (after sanitization)
      - solver iteration count actually used
    """
    out: dict = {}
    scene = env.scene_manager.scene
    rs = scene.sim.rigid_solver
    robot = env.scene_manager["robot"]

    # ── native contact count per env ──────────────────────────────
    out["n_contacts_per_env"] = _try_arr(rs.collider.collider_state.n_contacts)
    out["n_contacts_attr_path"] = "collider.collider_state.n_contacts"

    # ── solver-internal contact constraint metrics (env 0) ──────────
    # Dump every plausibly-relevant value the constraint solver
    # consumes/produces for each contact, so we can stop guessing
    # which input drives the 4× normal-force gap vs mujoco-warp.
    # Side-by-side with the mujoco-warp deep snapshot they
    # constitute the full closed-form: same imp_aref, same D
    # formula, same pyramidal cone — so anything that differs MUST
    # live in one of these fields.
    cs = rs.collider.collider_state.contact_data
    cs_count = rs.collider.collider_state.n_contacts
    constraint = rs.constraint_solver.constraint_state
    n_env0 = int(_try_arr(cs_count)[0])
    metrics: dict = {
        "n_contacts_env0": n_env0,
        # RigidOptions has no substeps field; the solver integrates
        # ``substeps`` times per scene step (from SimOptions).
        "rigid_solver.substeps": int(rs.substeps),
        "rigid_options.dt": float(rs._options.dt),
        "_substep_dt": float(rs._substep_dt),
        "_sol_default_timeconst": float(rs._sol_default_timeconst),
        "_sol_min_timeconst": float(rs._sol_min_timeconst),
        "integrator": rs._integrator.name,
    }
    # Per-contact native fields — these are exactly the inputs
    # to gu.imp_aref and the diag formula.
    contact_fields = {
        "penetration": cs.penetration,
        "normal": cs.normal,
        "friction": cs.friction,
        "sol_params": cs.sol_params,
        "geom_a": cs.geom_a,
        "geom_b": cs.geom_b,
        "link_a": cs.link_a,
        "link_b": cs.link_b,
        "force": cs.force,
    }
    for field, v in contact_fields.items():
        arr = _try_arr(v)
        # Expected shape (n_contacts_max, n_envs, …) — slice
        # env 0 + truncate to first n_env0 contacts.
        if isinstance(arr, list) and arr and isinstance(arr[0], list):
            col_env0 = [row[0] for row in arr[:n_env0]]
            metrics[f"contact_data.{field}_env0"] = col_env0
        else:
            metrics[f"contact_data.{field}_env0"] = arr[:n_env0] if isinstance(arr, list) else arr
    # Constraint-state fields per contact (4 efc rows per contact
    # for pyramidal): aref, diag, efc_D, efc_force, Jaref.
    # ``ConstraintState`` keeps no per-row position (``efc_pos``).
    constraint_fields = {
        "aref": constraint.aref,
        "diag": constraint.diag,
        "efc_D": constraint.efc_D,
        "efc_force": constraint.efc_force,
        "Jaref": constraint.Jaref,
    }
    for cs_field, v in constraint_fields.items():
        arr = _try_arr(v)
        # Shape (n_efc_max, n_envs) — slice env 0 + first
        # 4 * n_env0 rows (assuming contact constraints
        # come after equality/frictionloss). If preceded
        # by other constraints the row offset is non-
        # zero, so also dump the last 4*n_env0 rows.
        if isinstance(arr, list) and arr and isinstance(arr[0], list):
            col0 = [row[0] for row in arr]
        else:
            col0 = arr
        if isinstance(col0, list):
            metrics[f"constraint.{cs_field}_first_{min(len(col0), 4 * n_env0 + 8)}_env0"] = col0[: 4 * n_env0 + 8]
            metrics[f"constraint.{cs_field}_count_env0"] = len(col0)
    # invweight per foot link + ground link (mass matrix inverse
    # projection — the other half of the D denominator).
    inv = rs.dyn_info.links.invweight
    inv_arr = _try_arr(inv)
    # Map foot links + likely ground link by name.
    links_meta = []
    for link in robot.links:
        lname = link.name
        li = link.idx
        if isinstance(inv_arr, list) and 0 <= li < len(inv_arr):
            links_meta.append({"link_idx": int(li), "link_name": lname, "invweight": inv_arr[int(li)]})
    metrics["robot_links_invweight"] = links_meta
    # Ground entity links
    ground_entity = env.scene_manager.terrain.entity
    if ground_entity is not None:
        g_links_meta = []
        for link in ground_entity.links:
            lname = link.name
            li = link.idx
            if isinstance(inv_arr, list) and 0 <= li < len(inv_arr):
                g_links_meta.append({"link_idx": int(li), "link_name": lname, "invweight": inv_arr[int(li)]})
        metrics["ground_links_invweight"] = g_links_meta
    out["solver_metrics"] = metrics

    # ── per-contact-pair info via proper API: entity.get_contacts ─
    # Genesis exposes a documented dict {geom_a, geom_b, link_a,
    # link_b, position, force_a, force_b, valid_mask} per contact.
    # Filter to foot ↔ ground only and dump env 0's contacts.
    # The ground entity (TerrainImporter-owned).
    ground_entity = env.scene_manager.terrain.entity
    if ground_entity is not None:
        # Genesis ``collider.get_contacts(as_tensor=True)`` pads every
        # env's buffer to ``max(n_contacts)`` across envs (see
        # ``Genesis/genesis/engine/solvers/rigid/collider/collider.py:935-954``).
        # For an env that holds fewer contacts than the cross-env max,
        # slots beyond ``n_contacts[env]`` carry stale data from
        # previous steps (the buffer is overwrite-only, never
        # zero-initialised). ``RigidEntity.get_contacts``'s
        # ``valid_mask`` only checks geom-index ranges — it does not
        # check ``slot_idx < n_contacts[env]`` — so stale slots whose
        # ``geom_a`` / ``geom_b`` happen to lie in the robot ↔ ground
        # geom ranges leak through as "valid" contacts. Slice to env
        # 0's actual contact count before any downstream consumer
        # touches the dict.
        n_env0 = int(out["n_contacts_per_env"][0])
        contacts = robot.get_contacts(with_entity=ground_entity)
        cf: dict = {}
        for k in ("geom_a", "geom_b", "link_a", "link_b", "position", "force_a", "force_b", "valid_mask"):
            v = contacts.get(k) if isinstance(contacts, dict) else None
            if v is None:
                continue
            arr = _try_arr(v)
            if isinstance(arr, list):
                # Truncate to env 0 if multi-env shape.
                if arr and isinstance(arr[0], list) and len(arr) == env.num_envs:
                    cf[k] = arr[0][:n_env0]
                else:
                    cf[k] = arr[:n_env0]
        out["foot_ground_contacts_env0"] = cf

        # ── Resolve geom_a / geom_b indices to names + link
        # names so we can compare against MuJoCo by *name* in the
        # cross-sim diff section below.
        # Build geom-idx → (name, link_name, link_idx, owner_entity) and
        # capsule axis/length info for each robot geom.
        geom_table: dict[int, dict] = {}
        for ent in (robot, ground_entity):
            ent_name = ent.name or "?"
            for link in ent.links:
                lname = link.name
                lidx = link.idx
                for g in link.geoms:
                    # Genesis ``RigidGeom`` carries no name, so the
                    # rows below fall back to ``#<idx>``. The capsule
                    # axis / length lives in the local pose
                    # (``init_pos`` / ``init_quat``, parent-link frame)
                    # and ``data`` (capsule: ``[radius, height, ...]``).
                    geom_table[int(g.idx)] = {
                        "type": type(g).__name__,
                        "gs_type": g.type.name,
                        "link_name": lname,
                        "link_idx": int(lidx),
                        "entity": ent_name,
                        "init_pos": g.init_pos.tolist(),
                        "init_quat": g.init_quat.tolist(),
                        "data": g.data.tolist(),
                    }
        out["geom_table"] = geom_table

        # Resolved per-contact rows (env 0).
        resolved: list[dict] = []
        ga = cf.get("geom_a") or []
        gb = cf.get("geom_b") or []
        pos = cf.get("position") or []
        fa = cf.get("force_a") or []
        fb = cf.get("force_b") or []
        vm = cf.get("valid_mask") or []
        for i in range(min(len(ga), len(gb))):
            if i < len(vm) and not bool(vm[i]):
                break
            a_idx, b_idx = int(ga[i]), int(gb[i])
            a_meta = geom_table.get(a_idx, {"name": f"#{a_idx}", "link_name": "?"})
            b_meta = geom_table.get(b_idx, {"name": f"#{b_idx}", "link_name": "?"})
            force_a_vec = fa[i] if i < len(fa) else None
            fa_mag = float(math.sqrt(sum(c * c for c in force_a_vec))) if force_a_vec else None
            resolved.append(
                {
                    "i": i,
                    "geom_a_idx": a_idx,
                    "geom_a_name": a_meta.get("name") or f"#{a_idx}",
                    "geom_a_link": a_meta.get("link_name"),
                    "geom_b_idx": b_idx,
                    "geom_b_name": b_meta.get("name") or f"#{b_idx}",
                    "geom_b_link": b_meta.get("link_name"),
                    "pos": pos[i] if i < len(pos) else None,
                    "force_a": force_a_vec,
                    "force_a_norm": fa_mag,
                }
            )
        out["foot_ground_contacts_resolved_env0"] = resolved

        # Per-geom contact count: how many contact points
        # were generated for each foot collision capsule?
        per_geom: dict[str, int] = {}
        for r in resolved:
            for key in (r["geom_a_name"], r["geom_b_name"]):
                per_geom[key] = per_geom.get(key, 0) + 1
        # Filter to foot/ground geoms only (drop counterparty
        # double-count: a contact (foot_capsule, ground) shows
        # up under both keys, so each pair adds 1 to foot
        # capsule count AND 1 to ground count).
        out["foot_ground_contacts_per_geom_count_env0"] = per_geom
        # Compute aggregate stats: per-foot force magnitudes, depth
        # via geom-pair grouping. The dict's force_a is force on
        # robot's geom; for foot-ground, force_a should be the
        # foot's reaction (upward).
        fa = contacts.get("force_a")
        if fa is not None:
            fa_t = torch.as_tensor(_try_arr(fa))
            if fa_t.ndim == 3:  # (n_envs, n_contacts, 3)
                # Slice to env 0's real contact count before any
                # reduction — stale buffer slots beyond ``n_env0``
                # carry leftover force vectors that survive the
                # ``> 0`` filter (see padding caveat above the
                # ``cf[k] = arr[0][:n_env0]`` slice).
                magn = fa_t[0, :n_env0].norm(dim=-1)
                out["foot_ground_contact_count_env0"] = int((magn > 0).sum().item())
                out["foot_ground_force_per_contact_env0"] = magn[magn > 0].tolist()
                out["foot_ground_force_total_env0"] = float(magn.sum().item())
            else:
                magn = fa_t.norm(dim=-1)
                out["foot_ground_force_per_contact"] = magn[magn > 0].tolist()[:n_env0]
    else:
        out["foot_ground_contacts_err"] = "ground entity not found"

    # ── foot link mass + collision geometry ───────────────────────
    foot_info: list[dict] = []
    for fname in cfg.robot.foot_names:
        link = robot.get_link(fname)
        fi: dict = {"name": fname}
        # ``get_mass`` is per env when link info is batched; env 0.
        fi["mass"] = float(link.get_mass().flatten()[0])
        # geoms on this link
        geoms = link.geoms
        fi["n_geoms"] = len(geoms)
        if geoms:
            g = geoms[0]
            fi["geom0_type"] = type(g).__name__
            fi["geom0_friction"] = float(g.get_friction())
            fi["geom0_sol_params"] = _try_arr(g.get_sol_params())
            fi["geom0_mesh_n_verts"] = len(g.mesh.verts)
        foot_info.append(fi)
    out["feet"] = foot_info

    # ── total robot mass ─────────────────────────────────────────
    total_mass = 0.0
    for link in robot.links:
        total_mass += float(link.get_mass().flatten()[0])
    out["robot_total_mass"] = total_mass

    # ── effective sol_params per foot/ground geom (post-sanitize) ─
    # The rigid solver stores the sanitized per-geom sol_params in
    # ``dyn_info.geoms.sol_params`` (what ``RigidGeom.get_sol_params``
    # reads).
    sp_path = "dyn_info.geoms.sol_params"
    arr = _try_arr(rs.dyn_info.geoms.sol_params)
    if isinstance(arr, list):
        out["effective_sol_params_first_few"] = arr[:8]
        out["effective_sol_params_path"] = sp_path
        # Per-foot-geom batch sol_params row at the geom's
        # solver index — decides whether foot collision
        # geoms ended up at MIN_TIMECONST (≈2.22e-16) or
        # the sanitized default (0.02). The contact pipeline
        # reads this batch in collider/contact.py.
        for fi, fname in zip(out.get("feet", []), cfg.robot.foot_names):
            link = robot.get_link(fname)
            rows = []
            for g in link.geoms:
                gi = g.idx
                if not (0 <= gi < len(arr)):
                    rows.append({"idx": gi, "row": None})
                else:
                    rows.append({"idx": int(gi), "row": arr[int(gi)]})
            fi["batch_sol_params_per_geom"] = rows
    return out


def _mujoco_dump(env, cfg) -> dict:
    """MuJoCo (mjlab) deep introspection.

    Captures contact count, per-contact details, foot mass/geom info,
    actual solver iterations used, and the foot geom's solref/solimp.
    """
    out: dict = {}
    sm = env.scene_manager
    mj = sm.mj_model
    wd = sm.data  # mujoco-warp Data

    # ── native contact count (total across worlds) ────────────────
    out["ncon_attr"] = "nacon"
    out["ncon_per_world"] = _bridge_arr(wd.nacon)

    # ── per-contact info via mujoco-warp Data.contact ─────────────
    # Fields (from mujoco_warp/_src/types.py Contact dataclass):
    #   dist (naconmax,)       — distance; NEGATIVE means penetration
    #   pos (naconmax, 3)      — world position
    #   frame (naconmax, 3, 3) — frame; normal = frame[0]
    #   friction (naconmax, 5) — friction coefficients
    #   solref (naconmax, 2)   — per-contact solver reference
    #   solimp (naconmax, 5)   — per-contact solver impedance
    #   geom (naconmax, 2)     — geom ids
    #   worldid (naconmax,)    — which world the contact belongs to
    ctx: dict = {}
    # 1) Total contact buffer size + active count.
    ctx["buffer_naconmax"] = wd.naconmax
    # 2) Per-contact fields — convert and filter to foot↔ground.
    con = wd.contact
    raw: dict = {
        "dist": _bridge_arr(con.dist),
        "pos": _bridge_arr(con.pos),
        "frame": _bridge_arr(con.frame),
        "solref": _bridge_arr(con.solref),
        "solimp": _bridge_arr(con.solimp),
        "geom": _bridge_arr(con.geom),
        "worldid": _bridge_arr(con.worldid),
        "friction": _bridge_arr(con.friction),
        "includemargin": _bridge_arr(con.includemargin),
        "dim": _bridge_arr(con.dim),
        "efc_address": _bridge_arr(con.efc_address),
    }
    # Also pull constraint-state fields (efc_*) so we can
    # see the exact aref / D / force that the solver used,
    # mirroring the Genesis solver_metrics block. mujoco-warp
    # stores these on the *efc sub-namespace* (``wd.efc.force``
    # etc., not ``wd.efc_force``), with shape (nworld, nefcmax).
    efc_ns = wd.efc
    for f, v in (("pos", efc_ns.pos), ("aref", efc_ns.aref), ("D", efc_ns.D), ("force", efc_ns.force)):
        raw["efc_" + f] = _bridge_arr(v)
    raw["nefc"] = _bridge_arr(wd.nefc)
    ctx["solver_internals_raw"] = {
        k: (v if not isinstance(v, list) else v[:64])
        for k, v in raw.items()
        if k in ("efc_aref", "efc_D", "efc_force", "efc_pos", "nefc")
    }
    # invweight per body (foot links + world body for ground).
    ctx["body_invweight0"] = mj.body_invweight0.tolist()[: max(20, mj.nbody)]
    # opt fields that influence contact dynamics.
    opt = mj.opt
    ctx["opt_extra"] = {
        "integrator": int(opt.integrator),
        "cone": int(opt.cone),
        "impratio": float(opt.impratio),
        "tolerance": float(opt.tolerance),
        "iterations": int(opt.iterations),
        "ls_iterations": int(opt.ls_iterations),
        "timestep": float(opt.timestep),
        "noslip_iterations": int(opt.noslip_iterations),
    }
    # 3) Build foot-↔-ground filter:
    #    contact has geom pair where one is a foot geom and other is ground.
    foot_geom_ids: set[int] = set()
    ground_geom_ids: set[int] = set()
    for fname in cfg.robot.foot_names:
        _, bid = _mj_resolve_body(mj, (f"robot/{fname}", fname))
        for gid in range(mj.ngeom):
            if mj.geom_bodyid[gid] == bid:
                foot_geom_ids.add(gid)
    # ground geoms: those whose body is the world (body 0) or
    # whose name contains 'ground'/'terrain'.
    for gid in range(mj.ngeom):
        if mj.geom_bodyid[gid] == 0:
            ground_geom_ids.add(gid)
        gname = mj.geom(gid).name or ""
        if "ground" in gname.lower() or "terrain" in gname.lower():
            ground_geom_ids.add(gid)
    ctx["foot_geom_ids_count"] = len(foot_geom_ids)
    ctx["ground_geom_ids_count"] = len(ground_geom_ids)
    # 4) Filter foot↔ground contacts (env 0 only).
    geom = raw.get("geom")
    worldid = raw.get("worldid")
    dist = raw.get("dist")
    if geom and worldid and dist:
        # Pre-build geom-id → (name, body_id, body_name, type, size)
        # so each contact row carries human-readable
        # identification for cross-sim comparison.
        geom_table: dict[int, dict] = {}
        for gid in range(mj.ngeom):
            gname = mj.geom(gid).name or f"#{gid}"
            bid = int(mj.geom_bodyid[gid])
            bname = mj.body(bid).name or f"body#{bid}"
            geom_table[gid] = {
                "name": gname,
                "body_id": bid,
                "body_name": bname,
                "type": int(mj.geom_type[gid]),
                "size": mj.geom_size[gid].tolist(),
                "pos": mj.geom_pos[gid].tolist(),
            }
        ctx["geom_table"] = geom_table

        rows: list[dict] = []
        per_geom: dict[str, int] = {}
        # Pre-fetch efc_force + efc_address for per-contact
        # impulse lookup. The Genesis side dumps per-contact
        # |F| directly; mujoco-warp stores impulses in efc_force
        # and each contact owns up to ``condim`` consecutive
        # rows pointed to by contact.efc_address.
        efc_force_arr = raw.get("efc_force")
        efc_addr_arr = raw.get("efc_address")
        condim_arr = raw.get("dim")
        # Slots ``>= nacon`` hold stale rows from earlier steps.
        nacon = int(sm.wp_data.nacon.numpy()[0])
        for i, (g_pair, w, d) in enumerate(zip(geom[:nacon], worldid[:nacon], dist[:nacon])):
            if w != 0:
                continue  # filter to env 0
            g1_, g2_ = g_pair[0], g_pair[1]
            if (g1_ in foot_geom_ids and g2_ in ground_geom_ids) or (g2_ in foot_geom_ids and g1_ in ground_geom_ids):
                a_meta = geom_table.get(g1_, {"name": f"#{g1_}", "body_name": "?"})
                b_meta = geom_table.get(g2_, {"name": f"#{g2_}", "body_name": "?"})
                entry = {
                    "idx": i,
                    "geom": g_pair,
                    "geom_a_name": a_meta.get("name"),
                    "geom_a_body": a_meta.get("body_name"),
                    "geom_b_name": b_meta.get("name"),
                    "geom_b_body": b_meta.get("body_name"),
                    "dist": d,
                }
                if "pos" in raw:
                    entry["pos"] = raw["pos"][i]
                if "frame" in raw:
                    # frame is 3x3 — normal = frame[0]
                    entry["normal"] = raw["frame"][i][0] if isinstance(raw["frame"][i], list) else None
                if "solref" in raw:
                    entry["solref"] = raw["solref"][i]
                if "solimp" in raw:
                    entry["solimp"] = raw["solimp"][i]
                # Per-contact constraint impulse / normal-force lookup.
                # mujoco-warp shapes:
                #   contact.efc_address: (naconmax, nmaxpyramid) — int, -1 = unused
                #   efc.force:           (nworld, nefcmax)       — float
                # Env 0 lives at efc.force[0][...]. condim is
                # 1 for frictionless / 4 for pyramidal / 6 for
                # elliptic; we cap at 4.
                if efc_force_arr is not None and efc_addr_arr is not None:
                    addr_row = efc_addr_arr[i] if i < len(efc_addr_arr) else None
                    if addr_row is not None:
                        if not isinstance(addr_row, list):
                            addr_row = [addr_row]
                        # efc.force is (nworld, nefcmax) →
                        # env 0 row = efc_force_arr[0].
                        env0_efc = (
                            efc_force_arr[0]
                            if isinstance(efc_force_arr, list) and efc_force_arr and isinstance(efc_force_arr[0], list)
                            else efc_force_arr
                        )
                        cd_local = condim_arr[i] if condim_arr and i < len(condim_arr) else len(addr_row)
                        cd_local = min(int(cd_local), len(addr_row), 4)
                        impulses = []
                        for k in range(cd_local):
                            ai = addr_row[k]
                            if ai is None or ai < 0:
                                continue
                            if isinstance(env0_efc, list) and 0 <= ai < len(env0_efc):
                                impulses.append(env0_efc[ai])
                        entry["efc_force"] = impulses
                        if impulses and impulses[0] is not None:
                            entry["normal_force"] = float(impulses[0])
                            mag = math.sqrt(sum((float(x) ** 2) for x in impulses if x is not None))
                            entry["reaction_norm"] = mag
                rows.append(entry)
                for key in (a_meta.get("name"), b_meta.get("name")):
                    if key:
                        per_geom[key] = per_geom.get(key, 0) + 1
                if len(rows) >= 32:
                    break
        ctx["foot_ground_contacts_env0"] = rows
        ctx["foot_ground_contacts_env0_count"] = len(rows)
        ctx["foot_ground_contacts_per_geom_count_env0"] = per_geom
    out["contact_detail"] = ctx

    # ── foot body mass + geom info ───────────────────────────────
    # mjlab prefixes body names with ``robot/`` (entity name).
    foot_info: list[dict] = []
    for fname in cfg.robot.foot_names:
        # Try several name variants (mjlab prepends entity name).
        tried_name, bid = _mj_resolve_body(mj, (f"robot/{fname}", fname, f"/{fname}"))
        fi: dict = {
            "name": fname,
            "resolved_name": tried_name,
            "body_id": int(bid),
            "mass": float(mj.body_mass[bid]),
            "inertia": mj.body_inertia[bid].tolist(),
        }
        # geoms on this body
        geom_ids = [gid for gid in range(mj.ngeom) if mj.geom_bodyid[gid] == bid]
        fi["n_geoms"] = len(geom_ids)
        if geom_ids:
            gid = geom_ids[0]
            fi["geom0_id"] = int(gid)
            fi["geom0_name"] = mj.geom(gid).name
            fi["geom0_type"] = int(mj.geom_type[gid])
            fi["geom0_size"] = mj.geom_size[gid].tolist()
            fi["geom0_solref"] = mj.geom_solref[gid].tolist()
            fi["geom0_solimp"] = mj.geom_solimp[gid].tolist()
            fi["geom0_friction"] = mj.geom_friction[gid].tolist()
            fi["geom0_margin"] = float(mj.geom_margin[gid])
            # If it's a mesh, get vertex count
            if mj.geom_type[gid] == 7:  # mjGEOM_MESH
                mesh_id = mj.geom_dataid[gid]
                if mesh_id >= 0:
                    fi["geom0_mesh_id"] = int(mesh_id)
                    fi["geom0_mesh_n_verts"] = int(mj.mesh_vertnum[mesh_id])
                    fi["geom0_mesh_n_faces"] = int(mj.mesh_facenum[mesh_id])
        foot_info.append(fi)
    out["feet"] = foot_info

    # ── total robot mass ─────────────────────────────────────────
    out["robot_total_mass"] = float(mj.body_mass[1:].sum())  # skip world body

    # ── solver actual niter (mujoco-warp Data exposes per-world) ──
    out["solver_niter"] = _bridge_arr(wd.solver_niter)

    # ── mj_model opt block ───────────────────────────────────────
    out["opt"] = {
        "solref_default": mj.opt.o_solref.tolist(),
        "solimp_default": mj.opt.o_solimp.tolist(),
        "impratio": float(mj.opt.impratio),
        "solver": int(mj.opt.solver),
        "iterations": int(mj.opt.iterations),
        "ls_iterations": int(mj.opt.ls_iterations),
        "tolerance": float(mj.opt.tolerance),
        "noslip_iterations": int(mj.opt.noslip_iterations),
        "integrator": int(mj.opt.integrator),
        "cone": int(mj.opt.cone),
        "timestep": float(mj.opt.timestep),
        "gravity": mj.opt.gravity.tolist(),
        "wind": mj.opt.wind.tolist(),
        "magnetic": mj.opt.magnetic.tolist(),
        "viscosity": float(mj.opt.viscosity),
        "density": float(mj.opt.density),
    }
    return out


# ---------------------------------------------------------------------------
# Per-sim data capture (runs inside subprocess)
# ---------------------------------------------------------------------------


def _capture_sim(
    sim: str,
    num_envs: int,
    seed: int,
    settle_steps: int = 50,
    action_mode: str = "zero",
) -> dict:
    """Build a g1_29dof env in ``sim``, reset, step until settled, dump.

    The g1 base_init_height puts the robot ~4.5cm above ground; one step
    is not enough for feet to make contact. ``settle_steps`` zero-action
    control steps are applied between reset and capture so gravity lands
    the robot and foot↔ground contacts are present in the dump.

    ``action_mode``:
      - ``zero``: zero actions throughout (static landing)
      - ``random``: small uniform-random actions every step (more
        chaotic, closer to early-training distribution)
    """

    out: dict = {
        "sim": sim,
        "num_envs": num_envs,
        "seed": seed,
        "settle_steps": settle_steps,
        "action_mode": action_mode,
    }

    # ── Build ─────────────────────────────────────────────────────────
    cfg = G1FlatConfig(sim_type=sim, num_envs=num_envs, seed=seed)
    cfgs = cfg.build()
    runner = BaseRunner.create_with_env(cfgs)
    env = runner.env

    out["preset_meta"] = {
        "class": type(cfg).__name__,
        "module": type(cfg).__module__,
        "robot_base_link_name": cfg.robot.base_link_name,
        "robot_foot_names": list(cfg.robot.foot_names),
    }

    # ── Static solver/sim config ──────────────────────────────────────
    sim_cfg: dict = {"physics_dt": env.physics_dt, "decimation": env.decimation}
    if sim == "genesis":
        ro = env.scene_manager.config.rigid_options
        sim_cfg["genesis_rigid_options"] = {
            "dt": ro.dt,
            "constraint_solver": str(ro.constraint_solver),
            "constraint_timeconst": ro.constraint_timeconst,
            "iterations": ro.iterations,
            "ls_iterations": ro.ls_iterations,
            "max_collision_pairs": ro.max_collision_pairs,
            "contact_pruning_tolerance": ro.contact_pruning_tolerance,
            "enable_self_collision": ro.enable_self_collision,
            "enable_collision": ro.enable_collision,
            "batch_dofs_info": ro.batch_dofs_info,
        }
        # Genesis effective sol-params for foot geoms (from rigid solver).
        rs = env.scene_manager.scene.sim.rigid_solver
        sim_cfg["genesis_sol_min_timeconst"] = float(rs._sol_min_timeconst)
        sim_cfg["genesis_sol_default_timeconst"] = float(rs._sol_default_timeconst)
    elif sim == "mujoco":
        mj = env.scene_manager.mj_model
        sim_cfg["mujoco_opt"] = {
            "timestep": float(mj.opt.timestep),
            "solver": int(mj.opt.solver),
            "iterations": int(mj.opt.iterations),
            "ls_iterations": int(mj.opt.ls_iterations),
            "tolerance": float(mj.opt.tolerance),
            "impratio": float(mj.opt.impratio),
            "integrator": int(mj.opt.integrator),
            "cone": int(mj.opt.cone),
            "noslip_iterations": int(mj.opt.noslip_iterations),
            "ccd_iterations": int(mj.opt.ccd_iterations),
        }
        # Foot geom solref/solimp (per-geom). Capture the first foot geom.
        foot_names = list(cfg.robot.foot_names)
        # geom_name → geom_id; in mujoco_menagerie, foot collision geoms
        # may have either the body name or a "<body>_collision" suffix.
        for gid in range(mj.ngeom):
            geom_name = mj.geom(gid).name
            if any(fn in geom_name for fn in foot_names):
                sim_cfg["mujoco_foot_geom_example"] = {
                    "geom_id": gid,
                    "geom_name": geom_name,
                    "solref": mj.geom_solref[gid].tolist(),
                    "solimp": mj.geom_solimp[gid].tolist(),
                    "friction": mj.geom_friction[gid].tolist(),
                    "margin": float(mj.geom_margin[gid]),
                }
                break
    elif sim == "newton":
        # Newton SolverMuJoCoCfg lives on the scene-config side.
        scfg = env.scene_cfg.solver_cfg
        sim_cfg["newton_solver_cfg"] = {
            "solver": scfg.solver,
            "integrator": scfg.integrator,
            "cone": scfg.cone,
            "iterations": scfg.iterations,
            "ls_iterations": scfg.ls_iterations,
            "impratio": scfg.impratio,
            "use_mujoco_contacts": scfg.use_mujoco_contacts,
        }
    out["sim_config"] = sim_cfg

    # ── Reset (same seed across sims) ─────────────────────────────────
    env.reset()

    rd = env.get_robot_data()
    root_pos = rd.root_link_pos_w.detach().cpu()  # (B, 3)
    root_quat = rd.root_link_quat_w.detach().cpu()  # (B, 4)

    # Resolve foot body indices using the same per-sim convention the
    # preset uses (left_ankle_roll_link, right_ankle_roll_link).
    foot_names = list(cfg.robot.foot_names)
    resolved = env.resolve_selector(
        SceneEntitySelector(name="robot", body_names=tuple(foot_names), preserve_order=True)
    )
    foot_ids = resolved.body_ids
    foot_pos = rd.body_pos_w_by_ids(foot_ids).detach().cpu()  # (B, 2, 3)
    foot_lin_vel = rd.body_lin_vel_w_by_ids(foot_ids).detach().cpu()  # (B, 2, 3)

    out["state_after_reset"] = {
        "root_pos_w_env0": root_pos[0].tolist(),
        "root_pos_w_stats": {"mean": root_pos.float().mean().item(), "std": root_pos.float().std().item()},
        "root_quat_w_env0": root_quat[0].tolist(),
        "foot_names": foot_names,
        "foot_pos_w_env0": foot_pos[0].tolist(),
        "foot_lin_vel_w_env0": foot_lin_vel[0].tolist(),
        # Heuristic penetration: foot z minus assumed ground z=0.
        "foot_z_above_ground_env0": foot_pos[0, :, 2].tolist(),
    }

    # ── Settle: step N times, dump per-step time series for impact-phase ─
    # The chronic Genesis-vs-MuJoCo soft_landing gap is at IMPACT (first
    # few steps after the foot lands), not at steady state. We dump
    # forces/fmag/first/cost for every step up to ``timeseries_steps``
    # (default min(20, settle_steps)) plus a "final" snapshot at
    # settle_steps. ``contact_first_hit_step`` records when ``is_contact``
    # first turned True (anywhere across envs).
    n_act = env.num_actions
    zero = torch.zeros(env.num_envs, n_act, device=env.device)

    # Deterministic RNG for random-action mode so the diag is repeatable.
    rng = torch.Generator(device="cpu").manual_seed(seed)

    timeseries_max = min(20, settle_steps)
    timeseries: list[dict] = []

    contact_first_hit_step = None
    # Capture deep introspection snapshot at peak-impact step (heuristic:
    # 3 steps after first contact).
    deep_snapshot_step: int | None = None
    deep_snapshots: list[dict] = []

    for s in range(settle_steps):
        if action_mode == "random":
            # Small zero-mean uniform actions in [-0.3, 0.3]
            act = (torch.rand(env.num_envs, n_act, generator=rng) * 0.6 - 0.3).to(env.device)
        else:
            act = zero
        env.step(act)

        # Per-step contact snapshot (cheap: contact_force + first + is_contact)
        forces_s = env.contact_manager.contact_force("feet_ground_contact").detach().cpu()
        fmag_s = forces_s.norm(dim=-1)
        is_c_s = env.contact_manager.is_contact("feet_ground_contact").detach().cpu()
        first_s = env.contact_manager.compute_first_contact("feet_ground_contact").detach().cpu()

        if contact_first_hit_step is None and bool(is_c_s.any()):
            contact_first_hit_step = s
            # plan deep snapshot for the peak-impact step (heuristic)
            deep_snapshot_step = min(s + 2, settle_steps - 1)

        if s < timeseries_max:
            sl_cost_s = (fmag_s * first_s.float()).sum(dim=1)
            # Per-step trajectory snapshot — root z + foot z position
            # and z-velocity for both feet. Trajectory divergence
            # between sims at the dynamic phase is one of the suspected
            # origins of the residual ~4× peak fmag gap (same
            # closed-form, same inputs, still different output → maybe
            # the *state* at each step is diverging).
            rd_s = env.robot_data
            root_pos_s = rd_s.root_link_pos_w.detach().cpu()
            root_vel_s = rd_s.root_link_lin_vel_w.detach().cpu()
            # Same resolved foot ids as the reset / post-step reads.
            foot_pos_s = rd_s.body_pos_w_by_ids(foot_ids).detach().cpu()
            foot_vel_s = rd_s.body_lin_vel_w_by_ids(foot_ids).detach().cpu()
            entry: dict = {
                "step": s,
                "forces_env0": forces_s[0].tolist(),
                "fmag_env0": fmag_s[0].tolist(),
                "fmag_max_all_envs": float(fmag_s.max().item()),
                "fmag_mean_all_envs": float(fmag_s.mean().item()),
                "z_signed_max_all_envs": float(forces_s[..., 2].max().item()),
                "z_signed_min_all_envs": float(forces_s[..., 2].min().item()),
                "is_contact_env0": is_c_s[0].tolist(),
                "first_env0": first_s[0].tolist(),
                "first_fraction": float(first_s.float().mean().item()),
                "is_contact_fraction": float(is_c_s.float().mean().item()),
                "soft_landing_cost_env0_to_3": sl_cost_s[:4].tolist(),
                "soft_landing_cost_mean": float(sl_cost_s.mean().item()),
                "root_z_env0": float(root_pos_s[0, 2].item()),
                "root_z_mean_all_envs": float(root_pos_s[:, 2].mean().item()),
                "root_vz_env0": float(root_vel_s[0, 2].item()),
                "root_vz_mean_all_envs": float(root_vel_s[:, 2].mean().item()),
                "foot_z_env0": foot_pos_s[0, :, 2].tolist(),
                "foot_z_min_all_envs": float(foot_pos_s[..., 2].min().item()),
                "foot_z_max_all_envs": float(foot_pos_s[..., 2].max().item()),
                "foot_vz_env0": foot_vel_s[0, :, 2].tolist(),
                "foot_vz_min_all_envs": float(foot_vel_s[..., 2].min().item()),
                "foot_vz_max_all_envs": float(foot_vel_s[..., 2].max().item()),
            }
            timeseries.append(entry)

        # Deep introspection at the planned peak-impact step.
        if deep_snapshot_step is not None and s == deep_snapshot_step:
            snap = {"step": s, "tag": "peak_impact"}
            if sim == "genesis":
                snap.update(_genesis_dump(env, cfg))
            elif sim == "mujoco":
                snap.update(_mujoco_dump(env, cfg))
            deep_snapshots.append(snap)

    out["contact_first_hit_step"] = contact_first_hit_step
    out["timeseries"] = timeseries
    out["deep_snapshots"] = deep_snapshots

    # One more deep snapshot at the END (settled state) — for comparison.
    end_snap: dict = {"step": settle_steps - 1, "tag": "settled"}
    if sim == "genesis":
        end_snap.update(_genesis_dump(env, cfg))
    elif sim == "mujoco":
        end_snap.update(_mujoco_dump(env, cfg))
    out["settled_deep"] = end_snap

    # ── Post-step state ───────────────────────────────────────────────
    rd2 = env.get_robot_data()
    root_pos_p = rd2.root_link_pos_w.detach().cpu()
    foot_pos_p = rd2.body_pos_w_by_ids(foot_ids).detach().cpu()
    foot_lin_vel_p = rd2.body_lin_vel_w_by_ids(foot_ids).detach().cpu()

    out["state_after_step"] = {
        "root_pos_w_env0": root_pos_p[0].tolist(),
        "foot_pos_w_env0": foot_pos_p[0].tolist(),
        "foot_lin_vel_w_env0": foot_lin_vel_p[0].tolist(),
        "foot_xy_speed_env0": [float(foot_lin_vel_p[0, i, :2].norm().item()) for i in range(foot_lin_vel_p.shape[1])],
    }

    # ── Command state ────────────────────────────────────────────────
    # The first command term's command (the velocity command for this preset).
    _, cmd_term = next(env.command_manager.iter_terms())
    cmd = cmd_term.command
    out["command"] = {
        "shape": list(cmd.shape),
        "env0": cmd[0].detach().cpu().tolist(),
        "norm_env0": float(cmd[0].norm().item()),
        "norm_stats": {
            "mean": float(cmd.norm(dim=-1).mean().item()),
            "max": float(cmd.norm(dim=-1).max().item()),
            "fraction_above_0.05": float((cmd.norm(dim=-1) > 0.05).float().mean().item()),
        },
    }

    # ── Contact data (post-step, foot ↔ ground via "feet_ground_contact") ──
    forces = env.contact_manager.contact_force("feet_ground_contact").detach().cpu()  # (B, N, 3)
    fmag = forces.norm(dim=-1)  # (B, N)
    is_contact = env.contact_manager.is_contact("feet_ground_contact").detach().cpu()  # (B, N) bool
    first = env.contact_manager.compute_first_contact("feet_ground_contact").detach().cpu()  # (B, N) bool
    out["feet_ground_contact"] = {
        "forces_env0": forces[0].tolist(),
        "fmag_env0": fmag[0].tolist(),
        "fmag_stats": {
            "mean": float(fmag.mean().item()),
            "max": float(fmag.max().item()),
            "nonzero_fraction": float((fmag > 0).float().mean().item()),
        },
        "forces_world_stats": {
            "abs_mean": float(forces.abs().mean().item()),
            "abs_max": float(forces.abs().max().item()),
            "z_mean": float(forces[..., 2].mean().item()),
            "z_signed_max": float(forces[..., 2].max().item()),
            "z_signed_min": float(forces[..., 2].min().item()),
        },
        "is_contact_env0": is_contact[0].tolist(),
        "first_env0": first[0].tolist(),
        "is_contact_fraction": float(is_contact.float().mean().item()),
        "first_fraction": float(first.float().mean().item()),
    }

    # ── Per-sim native contact count (where exposed) ─────────────────
    native_count: dict = {}
    if sim == "genesis":
        rs = env.scene_manager.scene.sim.rigid_solver
        native_count["genesis_n_contacts_per_env"] = rs.collider.collider_state.n_contacts.to_numpy().tolist()
    elif sim == "mujoco":
        # ``Data.nacon`` is a 1-element array: the contact count summed
        # over all worlds (mujoco_warp keeps no per-world counter).
        native_count["mujoco_nacon_all_worlds"] = env.scene_manager.wp_data.nacon.numpy().tolist()
    out["native_contact_count"] = native_count

    # ── Reward recomputation — soft_landing + feet_slip ──────────────
    feet_selector = env.resolve_selector(
        SceneEntitySelector(
            name="robot",
            body_names=("left_foot_frame", "right_foot_frame"),
            preserve_order=True,
        )
    )
    contact_order = [n.replace("_foot_frame", "_ankle_roll_link") for n in feet_selector.body_names]

    sl = penalize_soft_landing(
        env, contact_group="feet_ground_contact", command_threshold=0.05, contact_order=contact_order
    )
    fs = penalize_feet_slip(
        env,
        contact_group="feet_ground_contact",
        command_threshold=0.05,
        contact_order=contact_order,
        asset_cfg=feet_selector,
    )

    # Internals (manual recompute for verbosity)
    sl_forces = env.contact_manager.contact_force("feet_ground_contact", order=contact_order).detach().cpu()
    sl_first = env.contact_manager.compute_first_contact("feet_ground_contact", order=contact_order).detach().cpu()
    sl_fmag = sl_forces.norm(dim=-1)
    sl_cost = (sl_fmag * sl_first.float()).sum(dim=1)
    cmd_active = _command_active(env, 0.05).detach().cpu()

    _, foot_v = _foot_pos_vel(env, feet_selector)
    foot_v = foot_v.detach().cpu()
    fs_vel_xy_sq = (foot_v[..., :2] ** 2).sum(dim=-1)
    fs_is_contact = (
        env.contact_manager.is_contact(
            "feet_ground_contact",
            order=_feet_contact_order(env, "feet_ground_contact", feet_selector, contact_order),
        )
        .detach()
        .cpu()
    )
    fs_cost = (fs_vel_xy_sq * fs_is_contact.float()).sum(dim=1)

    out["reward_soft_landing"] = {
        "contact_order": list(contact_order),
        "forces_env0": sl_forces[0].tolist(),
        "fmag_env0": sl_fmag[0].tolist(),
        "first_env0": sl_first[0].tolist(),
        "cost_first4": sl_cost[:4].tolist(),
        "cost_mean": float(sl_cost.mean().item()),
        "cmd_active_first4": cmd_active[:4].tolist(),
        "cmd_active_fraction": float(cmd_active.float().mean().item()),
        "out_first4": sl[:4].detach().cpu().tolist(),
        "out_mean": float(sl.mean().item()),
    }
    out["reward_feet_slip"] = {
        "contact_order": list(contact_order),
        "foot_vel_env0": foot_v[0].tolist(),
        "vel_xy_norm_sq_env0": fs_vel_xy_sq[0].tolist(),
        "is_contact_env0": fs_is_contact[0].tolist(),
        "cost_first4": fs_cost[:4].tolist(),
        "cost_mean": float(fs_cost.mean().item()),
        "out_first4": fs[:4].detach().cpu().tolist(),
        "out_mean": float(fs.mean().item()),
    }

    return _to_jsonable(out)


# ---------------------------------------------------------------------------
# Subprocess invocation + parent-side comparison
# ---------------------------------------------------------------------------


def _spawn_one(
    sim: str,
    num_envs: int,
    seed: int,
    out_dir: Path,
    settle_steps: int,
    action_mode: str,
) -> Path | None:
    """Run one sim in a fresh subprocess; write JSON to ``out_dir``."""
    out_json = out_dir / f"g1_contact_{sim}.json"
    cmd = [
        sys.executable,
        "-m",
        "jaxrlworld.scripts.diag.g1.check_g1_contact_force_parity",
        "--sim",
        sim,
        "--num-envs",
        str(num_envs),
        "--seed",
        str(seed),
        "--settle-steps",
        str(settle_steps),
        "--action-mode",
        action_mode,
        "--out-json",
        str(out_json),
        "--no-driver",
    ]
    print(f"\n┃ launching subprocess: {' '.join(cmd)}\n")
    res = subprocess.run(cmd, check=False)
    if res.returncode != 0:
        print(f"  ✗ {sim} subprocess returned rc={res.returncode}")
        return None
    if not out_json.exists():
        print(f"  ✗ {sim} produced no JSON at {out_json}")
        return None
    return out_json


def _fmt(v: Any, prec: int = 6) -> str:
    if isinstance(v, float):
        return f"{v:.{prec}g}"
    if isinstance(v, list):
        return "[" + ", ".join(_fmt(x, prec) for x in v) + "]"
    if v is None:
        return "—"
    return str(v)


def _print_compare(per_sim: dict[str, dict], lg) -> None:
    """Print a structured side-by-side comparison."""
    sims = list(per_sim)

    def row(label: str, lookup):
        cells = []
        for s in sims:
            v = lookup(per_sim[s])
            cells.append(_fmt(v))
        col_width = 30
        line = f"  {label:<48s}" + " | ".join(f"{c:<{col_width}s}" for c in cells)
        lg(line)

    def section(title: str):
        lg("")
        lg(f"━━━ {title} " + "━" * max(0, 70 - len(title)))

    # Header
    lg("\n" + "=" * 110)
    lg(f"  g1_29dof contact-force parity   |   sims = {sims}   |   num_envs = {per_sim[sims[0]]['num_envs']}")
    lg("=" * 110)
    col_width = 30
    header = f"  {'metric':<48s}" + " | ".join(f"{s:<{col_width}s}" for s in sims)
    lg(header)
    lg("  " + "-" * (48 + (col_width + 3) * len(sims)))

    section("Preset / robot meta")
    row("preset.class", lambda d: d["preset_meta"]["class"])
    row("robot.foot_names", lambda d: d["preset_meta"]["robot_foot_names"])
    row("robot.base_link_name", lambda d: d["preset_meta"]["robot_base_link_name"])

    section("Static sim config")
    row("physics_dt", lambda d: d["sim_config"]["physics_dt"])
    row("decimation", lambda d: d["sim_config"]["decimation"])
    row(
        "genesis.constraint_timeconst",
        lambda d: d["sim_config"].get("genesis_rigid_options", {}).get("constraint_timeconst"),
    )
    row(
        "genesis.contact_pruning_tolerance",
        lambda d: d["sim_config"].get("genesis_rigid_options", {}).get("contact_pruning_tolerance"),
    )
    row("genesis.iterations", lambda d: d["sim_config"].get("genesis_rigid_options", {}).get("iterations"))
    row(
        "genesis.max_collision_pairs",
        lambda d: d["sim_config"].get("genesis_rigid_options", {}).get("max_collision_pairs"),
    )
    row("genesis._sol_min_timeconst", lambda d: d["sim_config"].get("genesis_sol_min_timeconst"))
    row("genesis._sol_default_timeconst", lambda d: d["sim_config"].get("genesis_sol_default_timeconst"))
    row("mujoco.opt.solref(=solref[0])", lambda d: d["sim_config"].get("mujoco_foot_geom_example", {}).get("solref"))
    row("mujoco.opt.solimp", lambda d: d["sim_config"].get("mujoco_foot_geom_example", {}).get("solimp"))
    row("mujoco.opt.iterations", lambda d: d["sim_config"].get("mujoco_opt", {}).get("iterations"))
    row("mujoco.opt.impratio", lambda d: d["sim_config"].get("mujoco_opt", {}).get("impratio"))
    row("mujoco.foot_geom.friction", lambda d: d["sim_config"].get("mujoco_foot_geom_example", {}).get("friction"))
    row("newton.solver_cfg.iterations", lambda d: d["sim_config"].get("newton_solver_cfg", {}).get("iterations"))
    row("newton.solver_cfg.impratio", lambda d: d["sim_config"].get("newton_solver_cfg", {}).get("impratio"))

    section("State at reset (env 0)")
    row("root_pos_w", lambda d: d["state_after_reset"]["root_pos_w_env0"])
    row("foot_pos_w (left)", lambda d: d["state_after_reset"]["foot_pos_w_env0"][0])
    row("foot_pos_w (right)", lambda d: d["state_after_reset"]["foot_pos_w_env0"][1])
    row("foot_z_above_ground", lambda d: d["state_after_reset"]["foot_z_above_ground_env0"])
    row("foot_lin_vel_w (left)", lambda d: d["state_after_reset"]["foot_lin_vel_w_env0"][0])
    row("foot_lin_vel_w (right)", lambda d: d["state_after_reset"]["foot_lin_vel_w_env0"][1])

    section(f"State after {per_sim[sims[0]].get('settle_steps', '?')} settle steps (env 0)")
    row("contact_first_hit_step (during settle)", lambda d: d.get("contact_first_hit_step"))
    row("root_pos_w", lambda d: d["state_after_step"]["root_pos_w_env0"])
    row("foot_pos_w (left)", lambda d: d["state_after_step"]["foot_pos_w_env0"][0])
    row("foot_pos_w (right)", lambda d: d["state_after_step"]["foot_pos_w_env0"][1])
    row("foot_xy_speed", lambda d: d["state_after_step"]["foot_xy_speed_env0"])

    section("Command")
    row("command env0", lambda d: d["command"]["env0"])
    row("command.norm env0", lambda d: d["command"]["norm_env0"])
    row("command.norm fraction>0.05", lambda d: d["command"]["norm_stats"]["fraction_above_0.05"])

    section("feet_ground_contact (post-step, env 0)")
    row("forces_env0[left]", lambda d: d["feet_ground_contact"]["forces_env0"][0])
    row("forces_env0[right]", lambda d: d["feet_ground_contact"]["forces_env0"][1])
    row("fmag_env0", lambda d: d["feet_ground_contact"]["fmag_env0"])
    row("forces.abs_max (all envs)", lambda d: d["feet_ground_contact"]["forces_world_stats"]["abs_max"])
    row("forces.z_signed_max (all envs)", lambda d: d["feet_ground_contact"]["forces_world_stats"]["z_signed_max"])
    row("forces.z_signed_min (all envs)", lambda d: d["feet_ground_contact"]["forces_world_stats"]["z_signed_min"])
    row("fmag.mean (all envs)", lambda d: d["feet_ground_contact"]["fmag_stats"]["mean"])
    row("fmag.max (all envs)", lambda d: d["feet_ground_contact"]["fmag_stats"]["max"])
    row("is_contact_env0", lambda d: d["feet_ground_contact"]["is_contact_env0"])
    row("first_env0", lambda d: d["feet_ground_contact"]["first_env0"])
    row("is_contact fraction", lambda d: d["feet_ground_contact"]["is_contact_fraction"])
    row("first fraction", lambda d: d["feet_ground_contact"]["first_fraction"])

    section("Native contact count")
    row("genesis n_contacts_per_env", lambda d: d["native_contact_count"].get("genesis_n_contacts_per_env"))
    row("mujoco nacon (all worlds)", lambda d: d["native_contact_count"].get("mujoco_nacon_all_worlds"))

    section("soft_landing breakdown (env 0)")
    row("forces_env0[left]", lambda d: d["reward_soft_landing"]["forces_env0"][0])
    row("forces_env0[right]", lambda d: d["reward_soft_landing"]["forces_env0"][1])
    row("fmag_env0", lambda d: d["reward_soft_landing"]["fmag_env0"])
    row("first_env0", lambda d: d["reward_soft_landing"]["first_env0"])
    row("cost_first4", lambda d: d["reward_soft_landing"]["cost_first4"])
    row("cost.mean", lambda d: d["reward_soft_landing"]["cost_mean"])
    row("cmd_active.fraction", lambda d: d["reward_soft_landing"]["cmd_active_fraction"])
    row("out.mean", lambda d: d["reward_soft_landing"]["out_mean"])

    section("feet_slip breakdown (env 0)")
    row("foot_vel_env0[left]", lambda d: d["reward_feet_slip"]["foot_vel_env0"][0])
    row("foot_vel_env0[right]", lambda d: d["reward_feet_slip"]["foot_vel_env0"][1])
    row("vel_xy_norm_sq_env0", lambda d: d["reward_feet_slip"]["vel_xy_norm_sq_env0"])
    row("is_contact_env0", lambda d: d["reward_feet_slip"]["is_contact_env0"])
    row("cost.mean", lambda d: d["reward_feet_slip"]["cost_mean"])
    row("out.mean", lambda d: d["reward_feet_slip"]["out_mean"])

    # ── Deep introspection at peak-impact step ────────────────────────
    section("Deep snapshot — peak-impact step (per-sim native introspection)")
    for s in sims:
        sn_list = per_sim[s].get("deep_snapshots") or []
        sn = sn_list[-1] if sn_list else {}
        lg(f"  [{s}] step={sn.get('step', '?')}  tag={sn.get('tag', '?')}")
        # Genesis-specific
        for k in (
            "n_contacts_attr_path",
            "n_contacts_per_env",
            "robot_total_mass",
            "effective_sol_params_path",
            "effective_sol_params_first_few",
        ):
            if k in sn:
                lg(f"    {k:<36s} = {sn[k]}")
        # MuJoCo-specific
        for k in ("ncon_per_world", "solver_niter", "solver_iteration", "niter", "robot_total_mass"):
            if k in sn:
                lg(f"    {k:<36s} = {sn[k]}")
        # Common: foot info
        feet = sn.get("feet") or []
        for fi in feet:
            lg(
                f"    foot {fi.get('name', '?')!r}: "
                + ", ".join(f"{k}={v}" for k, v in fi.items() if k != "name" and not isinstance(v, list))
            )
            for k, v in fi.items():
                if isinstance(v, list):
                    lg(f"      {k} = {v}")
        # MuJoCo opt block
        if "opt" in sn:
            lg(f"    opt = {sn['opt']}")
        if "contact_detail" in sn:
            cd = sn["contact_detail"]
            if "opt_extra" in cd:
                lg(f"    opt_extra = {cd['opt_extra']}")
            if "body_invweight0" in cd:
                lg(f"    body_invweight0 (first 20 bodies) = {cd['body_invweight0']}")
            if "solver_internals_raw" in cd:
                sir = cd["solver_internals_raw"]
                for k in ("nefc", "efc_aref", "efc_D", "efc_force", "efc_pos"):
                    if k in sir:
                        v = sir[k]
                        lg(f"    {k} = {v[:32] if isinstance(v, list) and len(v) > 32 else v}")
        # Genesis solver_metrics block — full input/output dump
        if "solver_metrics" in sn:
            sm = sn["solver_metrics"]
            lg("    Genesis solver_metrics:")
            scalar_keys = (
                "n_contacts_env0",
                "rigid_solver.substeps",
                "rigid_options.dt",
                "_substep_dt",
                "_sol_default_timeconst",
                "_sol_min_timeconst",
                "integrator",
            )
            for k in scalar_keys:
                if k in sm:
                    lg(f"      {k:<36s} = {sm[k]}")
            # Per-contact inputs to imp_aref / D
            for k in (
                "contact_data.penetration_env0",
                "contact_data.normal_env0",
                "contact_data.friction_env0",
                "contact_data.sol_params_env0",
                "contact_data.geom_a_env0",
                "contact_data.geom_b_env0",
                "contact_data.link_a_env0",
                "contact_data.link_b_env0",
                "contact_data.force_env0",
            ):
                if k in sm:
                    v = sm[k]
                    lg(f"      {k} = {v if not isinstance(v, list) or len(v) <= 32 else v[:32]}")
            # Constraint-state outputs
            for k in list(sm.keys()):
                if k.startswith("constraint."):
                    v = sm[k]
                    lg(f"      {k} = {v if not isinstance(v, list) or len(v) <= 24 else v[:24]}")
            # invweight per link
            if "robot_links_invweight" in sm:
                lg("      robot_links_invweight (link_idx, name, invweight):")
                for entry in sm["robot_links_invweight"][:32]:
                    lg(f"        {entry}")
            if "ground_links_invweight" in sm:
                lg("      ground_links_invweight:")
                for entry in sm["ground_links_invweight"]:
                    lg(f"        {entry}")
        # Errors
        for k in list(sn.keys()):
            if k.endswith("_err"):
                lg(f"    ! {k}: {sn[k]}")
        lg("")

    # ── Per-contact pair detail at peak impact ──────────────────────
    section("Per-contact-pair detail at peak impact step (foot↔ground only)")
    for s in sims:
        sn_list = per_sim[s].get("deep_snapshots") or []
        sn = sn_list[-1] if sn_list else {}
        lg(f"  [{s}] step={sn.get('step', '?')}")
        if s == "genesis":
            gc = sn.get("foot_ground_contacts_env0") or {}
            if gc:
                lg(f"    geom_a: {gc.get('geom_a')}")
                lg(f"    geom_b: {gc.get('geom_b')}")
                lg(f"    link_a: {gc.get('link_a')}")
                lg(f"    link_b: {gc.get('link_b')}")
                lg(
                    f"    position[:8]: {gc.get('position')[:8] if isinstance(gc.get('position'), list) else gc.get('position')}"
                )
                lg(
                    f"    force_a[:8]:  {gc.get('force_a')[:8] if isinstance(gc.get('force_a'), list) else gc.get('force_a')}"
                )
                lg(f"    valid_mask: {gc.get('valid_mask')}")
            else:
                lg(f"    foot_ground_contacts: {sn.get('foot_ground_contacts_err', 'missing')}")
            lg(f"    foot_ground_contact_count_env0: {sn.get('foot_ground_contact_count_env0')}")
            lg(f"    foot_ground_force_per_contact_env0: {sn.get('foot_ground_force_per_contact_env0')}")
            lg(f"    foot_ground_force_total_env0: {sn.get('foot_ground_force_total_env0')}")
            # Resolved per-contact rows: idx, geom names + link names + force
            resolved = sn.get("foot_ground_contacts_resolved_env0") or []
            if resolved:
                lg("    resolved contacts (env 0):")
                lg(f"      {'i':>3} {'geom_a (link)':<48s} {'geom_b (link)':<28s} {'|F_a|':>9s}")
                for r in resolved:
                    a = f"{r.get('geom_a_name', '?')} ({r.get('geom_a_link', '?')})"
                    b = f"{r.get('geom_b_name', '?')} ({r.get('geom_b_link', '?')})"
                    fn = r.get("force_a_norm")
                    fns = f"{fn:9.3f}" if isinstance(fn, int | float) else "       -"
                    lg(f"      {r.get('i'):>3} {a:<48s} {b:<28s} {fns}")
            pgc = sn.get("foot_ground_contacts_per_geom_count_env0") or {}
            if pgc:
                lg("    per-geom contact count (env 0):")
                for name, n in sorted(pgc.items(), key=lambda kv: (-kv[1], kv[0])):
                    lg(f"      {n:>3d}  {name}")
        elif s == "mujoco":
            cd = sn.get("contact_detail") or {}
            lg(f"    buffer_naconmax: {cd.get('buffer_naconmax')}")
            lg(f"    foot_geom_ids_count: {cd.get('foot_geom_ids_count')}")
            lg(f"    ground_geom_ids_count: {cd.get('ground_geom_ids_count')}")
            lg(f"    foot_ground_contacts_env0_count: {cd.get('foot_ground_contacts_env0_count')}")
            rows = cd.get("foot_ground_contacts_env0") or []
            if rows:
                lg("    resolved contacts (env 0):")
                lg(
                    f"      {'idx':>4} {'geom_a (body)':<48s} {'geom_b (body)':<28s} "
                    f"{'dist':>10s} {'|N|':>9s} {'|F|':>9s}"
                )
                for r in rows:
                    a = f"{r.get('geom_a_name') or '?'} ({r.get('geom_a_body') or '?'})"
                    b = f"{r.get('geom_b_name') or '?'} ({r.get('geom_b_body') or '?'})"
                    d = r.get("dist")
                    nf = r.get("normal_force")
                    rn = r.get("reaction_norm")
                    ds = f"{d:10.6f}" if isinstance(d, int | float) else "         -"
                    ns = f"{abs(nf):9.3f}" if isinstance(nf, int | float) else "        -"
                    rs_ = f"{rn:9.3f}" if isinstance(rn, int | float) else "        -"
                    lg(f"      {r.get('idx'):>4} {a:<48s} {b:<28s} {ds} {ns} {rs_}")
            pgc = cd.get("foot_ground_contacts_per_geom_count_env0") or {}
            if pgc:
                lg("    per-geom contact count (env 0):")
                for name, n in sorted(pgc.items(), key=lambda kv: (-kv[1], kv[0])):
                    lg(f"      {n:>3d}  {name}")
        lg("")

    # ── Cross-sim per-geom contact count diff (foot capsules only) ──
    if len(sims) >= 2 and "genesis" in sims and "mujoco" in sims:
        section("Cross-sim per-geom contact-count diff (foot capsules only, env 0)")
        g_sn = (per_sim["genesis"].get("deep_snapshots") or [{}])[-1]
        m_sn = (per_sim["mujoco"].get("deep_snapshots") or [{}])[-1]
        g_pgc = g_sn.get("foot_ground_contacts_per_geom_count_env0") or {}
        m_pgc = (m_sn.get("contact_detail") or {}).get("foot_ground_contacts_per_geom_count_env0") or {}

        # Filter to foot geom names (drop counterparty / ground entries
        # whose name looks like 'ground_plane', 'plane', or contains
        # 'baselink'/'terrain'). Keep only names containing 'foot' or
        # being a capsule name from our preset (e.g. 'left_foot1_collision').
        def _is_foot(name: str) -> bool:
            n = (name or "").lower()
            return "foot" in n or "ankle" in n

        g_foot = {k: v for k, v in g_pgc.items() if _is_foot(k)}
        m_foot = {k: v for k, v in m_pgc.items() if _is_foot(k)}
        all_names = sorted(set(g_foot) | set(m_foot))
        lg(f"  {'geom_name':<32s} {'genesis':>9s} {'mujoco':>9s} {'gen-muj':>9s}")
        for name in all_names:
            gn = g_foot.get(name, 0)
            mn = m_foot.get(name, 0)
            lg(f"  {name:<32s} {gn:>9d} {mn:>9d} {gn - mn:>+9d}")
        lg(
            f"  {'TOTAL':<32s} {sum(g_foot.values()):>9d} {sum(m_foot.values()):>9d} "
            f"{sum(g_foot.values()) - sum(m_foot.values()):>+9d}"
        )
        lg("")
        # Set-diff: which foot geom NAMES never appear in one sim
        g_only = sorted(set(g_foot) - set(m_foot))
        m_only = sorted(set(m_foot) - set(g_foot))
        if g_only:
            lg(f"  foot geoms with contacts in GENESIS ONLY: {g_only}")
        if m_only:
            lg(f"  foot geoms with contacts in MUJOCO  ONLY: {m_only}")
        if not g_only and not m_only:
            lg("  same set of foot geoms touched in both sims")
        lg("")

    section("Deep snapshot — settled state (per-sim native introspection)")
    for s in sims:
        sn = per_sim[s].get("settled_deep") or {}
        lg(f"  [{s}] step={sn.get('step', '?')}  tag={sn.get('tag', '?')}")
        for k in ("n_contacts_per_env", "ncon_per_world"):
            if k in sn:
                lg(f"    {k:<36s} = {sn[k]}")
        # foot info
        feet = sn.get("feet") or []
        for fi in feet:
            lg(f"    foot {fi.get('name', '?')!r}: mass={fi.get('mass')}, n_geoms={fi.get('n_geoms')}")
        lg("")

    # ── Per-step time series (impact-phase) ───────────────────────────
    section("Impact-phase time series — soft_landing cost per step (all envs)")
    n_steps_to_show = min(
        (len(per_sim[s].get("timeseries", [])) for s in sims if per_sim[s].get("timeseries")), default=0
    )
    if n_steps_to_show > 0:
        lg(f"  {'step':<48s}" + " | ".join(f"{s:<30s}" for s in sims))
        for i in range(n_steps_to_show):
            row(f"step {i}: soft_landing.cost.mean", lambda d, i=i: d["timeseries"][i]["soft_landing_cost_mean"])
        lg("")
        for i in range(n_steps_to_show):
            row(f"step {i}: fmag.max", lambda d, i=i: d["timeseries"][i]["fmag_max_all_envs"])
        lg("")
        for i in range(n_steps_to_show):
            row(f"step {i}: first.fraction", lambda d, i=i: d["timeseries"][i]["first_fraction"])
        lg("")
        for i in range(n_steps_to_show):
            row(f"step {i}: is_contact.fraction", lambda d, i=i: d["timeseries"][i]["is_contact_fraction"])
        lg("")

        # ── Trajectory divergence — root + foot z position/velocity ────
        # If the same actions produce the same root_z / foot_z / foot_vz
        # trace in both sims, the residual fmag gap is *solver-internal*
        # (same state, different reaction). If the traces diverge in the
        # dynamic phase, the gap is *trajectory* (each sim hits the
        # ground with different velocities at the same step index).
        for label, key in (
            ("root_z env0", "root_z_env0"),
            ("root_vz env0", "root_vz_env0"),
            ("foot_z_env0[L,R]", "foot_z_env0"),
            ("foot_vz_env0[L,R]", "foot_vz_env0"),
            ("foot_z.min all_envs", "foot_z_min_all_envs"),
            ("foot_vz.min all_envs", "foot_vz_min_all_envs"),
        ):

            def _has(d, k=key):
                ts = d.get("timeseries", [])
                return ts and k in ts[0]

            if all(_has(per_sim[s]) for s in sims):
                for i in range(n_steps_to_show):
                    row(f"step {i}: {label}", lambda d, i=i, k=key: d["timeseries"][i].get(k))
                lg("")

        # Cumulative soft_landing.cost over first M steps (mimics what
        # the first PPO step's reward-breakdown log averages over).
        section("Cumulative soft_landing cost over first M steps (sum × num_envs)")
        for M in (3, 5, 10, 20):
            if n_steps_to_show >= M:
                row(
                    f"sum(cost.mean) over first {M} steps",
                    lambda d, M=M: sum(d["timeseries"][i]["soft_landing_cost_mean"] for i in range(M)),
                )

    # ── Cross-sim ratio highlight (if 2+ sims) ────────────────────────
    if len(sims) >= 2:
        section("Ratios (relative to last sim)")
        ref = sims[-1]
        for s in sims[:-1]:
            ratio_label = f"  ({s}) / ({ref})"
            a_sl = per_sim[s]["reward_soft_landing"]["cost_mean"]
            b_sl = per_sim[ref]["reward_soft_landing"]["cost_mean"]
            lg(f"  {ratio_label}  soft_landing.cost.mean ratio = {(a_sl / b_sl) if b_sl else float('inf'):.4g}")
            a_fs = per_sim[s]["reward_feet_slip"]["cost_mean"]
            b_fs = per_sim[ref]["reward_feet_slip"]["cost_mean"]
            lg(f"  {ratio_label}  feet_slip.cost.mean ratio   = {(a_fs / b_fs) if b_fs else float('inf'):.4g}")
            a_fmag = per_sim[s]["feet_ground_contact"]["fmag_stats"]["max"]
            b_fmag = per_sim[ref]["feet_ground_contact"]["fmag_stats"]["max"]
            lg(f"  {ratio_label}  fmag.max ratio              = {(a_fmag / b_fmag) if b_fmag else float('inf'):.4g}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--sims",
        type=str,
        default="genesis,mujoco",
        help="Comma-separated subset of {genesis,mujoco,newton}. Default: genesis,mujoco.",
    )
    ap.add_argument(
        "--sim",
        type=str,
        default=None,
        choices=("genesis", "mujoco", "newton"),
        help="Single-sim inline (with --no-driver) — used by driver subprocesses.",
    )
    ap.add_argument(
        "--num-envs",
        type=int,
        default=16,
        help="More envs = better statistical convergence of force/cost ratios. "
        "Default 16 is a compromise; crank to 256-1024 for training-like distributions.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--settle-steps",
        type=int,
        default=50,
        help="Control steps after reset before final capture (lets robot fall and contact ground). "
        "g1 spawns ~4.5cm above ground; default 50 control steps (= 1 sec @ 20ms) is plenty.",
    )
    ap.add_argument(
        "--action-mode",
        type=str,
        default="zero",
        choices=("zero", "random"),
        help="zero=zero actions throughout (static landing). random=small uniform actions "
        "(more chaotic, closer to early-training distribution where impact peaks are larger).",
    )
    ap.add_argument("--out-json", type=str, default=None, help="Inline single-sim mode: where to write the JSON dump.")
    ap.add_argument("--out-dir", type=str, default=".", help="Driver mode: where to put per-sim JSONs + comparison.")
    ap.add_argument("--no-driver", action="store_true")
    args = ap.parse_args()

    # ── Inline single-sim mode (driven by subprocess) ─────────────────
    if args.no_driver:
        if args.sim is None:
            print("inline mode requires --sim", file=sys.stderr)
            return 2
        data = _capture_sim(args.sim, args.num_envs, args.seed, args.settle_steps, args.action_mode)
        target = Path(args.out_json) if args.out_json else Path(f"./g1_contact_{args.sim}.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(data, indent=2))
        print(f"\n✓ wrote {target}")
        return 0

    # ── Driver mode: spawn one subprocess per sim, then compare ──────
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    sims = [s.strip() for s in args.sims.split(",") if s.strip()]
    unknown = [s for s in sims if s not in ("genesis", "mujoco", "newton")]
    if unknown:
        print(f"unknown sim(s): {unknown}", file=sys.stderr)
        return 2

    per_sim: dict[str, dict] = {}
    for sim in sims:
        path = _spawn_one(sim, args.num_envs, args.seed, out_dir, args.settle_steps, args.action_mode)
        if path is None:
            print(f"  ✗ skipping {sim} (subprocess failure)")
            continue
        per_sim[sim] = json.loads(path.read_text())

    if not per_sim:
        print("no sim produced a valid JSON — aborting")
        return 1

    compare_path = out_dir / "g1_contact_compare.txt"
    lines: list[str] = []

    def lg(s: str = "") -> None:
        print(s)
        lines.append(s)

    _print_compare(per_sim, lg)
    compare_path.write_text("\n".join(lines))
    print(f"\n✓ wrote comparison to {compare_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
