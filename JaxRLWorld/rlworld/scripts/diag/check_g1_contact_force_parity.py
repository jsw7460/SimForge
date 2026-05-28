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
  python -m rlworld.scripts.diag.check_g1_contact_force_parity

  # Add Newton:
  python -m rlworld.scripts.diag.check_g1_contact_force_parity --sims genesis,mujoco,newton

  # Single-combo inline (debugging):
  python -m rlworld.scripts.diag.check_g1_contact_force_parity --sim genesis --no-driver

Output: per-sim JSON saved as ``./g1_contact_<sim>.json``, comparison
table printed to stdout and saved to ``./g1_contact_compare.txt``.
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


def _to_jsonable(value: Any) -> Any:
    """Convert tensors / numpy / etc. into JSON-serializable primitives.

    Tensors are detached, moved to CPU, then converted to nested lists.
    Floats are kept as-is. ``None`` passes through. Unknown types fall
    back to ``str(value)`` for transparency.
    """
    try:
        import torch

        if isinstance(value, torch.Tensor):
            t = value.detach().cpu()
            if t.dtype == torch.bool:
                return t.tolist()
            return t.float().tolist()
    except ImportError:
        pass
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.floating | np.integer):
            return value.item()
    except ImportError:
        pass
    if isinstance(value, list | tuple):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if value is None or isinstance(value, bool | int | float | str):
        return value
    return str(value)


def _safe_get(obj: Any, *names: str, default: Any = None) -> Any:
    """Walk attribute chain returning ``default`` on any missing step."""
    cur = obj
    for n in names:
        cur = getattr(cur, n, None)
        if cur is None:
            return default
    return cur


def _try_arr(maybe_arr: Any) -> Any:
    """Convert a possibly-warp/numpy/tensor/quadrants array into a Python list.

    Genesis uses quadrants tensors (``qd.Tensor``) for many arrays. These
    have ``.to_numpy()`` (not ``.numpy()``); we try several converters
    before falling back to ``repr``.
    """
    if maybe_arr is None:
        return None
    # Try a sequence of converters known to produce numpy.
    for converter_name in ("to_numpy", "numpy", "cpu", "detach"):
        if hasattr(maybe_arr, converter_name):
            try:
                arr = getattr(maybe_arr, converter_name)()
                if hasattr(arr, "tolist"):
                    return arr.tolist()
                if hasattr(arr, "numpy"):
                    return arr.numpy().tolist()
            except Exception:
                continue
    # Try warp → torch → numpy.
    try:
        import warp as wp

        arr = wp.to_torch(maybe_arr).detach().cpu().numpy()
        return arr.tolist()
    except Exception:
        pass
    # Try torch directly.
    try:
        import torch

        if isinstance(maybe_arr, torch.Tensor):
            return maybe_arr.detach().cpu().tolist()
    except Exception:
        pass
    # Try numpy directly.
    try:
        import numpy as np

        arr = np.asarray(maybe_arr)
        return arr.tolist()
    except Exception:
        pass
    return repr(maybe_arr)


def _walk_paths(root: Any, paths: list[tuple[str, ...]]) -> tuple[Any, str | None]:
    """Try several attribute paths under ``root``; return (value, successful_path)."""
    for path in paths:
        cur = root
        ok = True
        for name in path:
            if cur is None:
                ok = False
                break
            cur = getattr(cur, name, None)
            if cur is None:
                ok = False
                break
        if ok and cur is not None:
            return cur, ".".join(path)
    return None, None


def _genesis_dump(env, cfg) -> dict:
    """Genesis-specific deep introspection.

    Tries multiple attribute paths so the diag is resilient to Genesis
    refactors. Captures every quantity that could explain higher contact
    force magnitudes:
      - native contact count per env
      - per-contact (position, normal, penetration, force vector) at the
        moment of capture
      - foot link mass + total robot mass + foot collision geom info
      - effective sol_params per foot/ground geom (after sanitization)
      - solver iteration count actually used
    """
    out: dict = {}
    try:
        scene = env.scene_manager.scene
        rs = scene.sim.rigid_solver
        robot = env.scene_manager["robot"]

        # ── native contact count per env ──────────────────────────────
        ncon, ncon_path = _walk_paths(
            rs,
            [
                ("collider", "_collider_state", "n_contacts"),
                ("collider", "contact_data", "n_contacts"),
                ("collider", "n_contacts"),
                ("_collider_state", "n_contacts"),
                ("_contacts", "n_contacts"),
                ("contacts", "n_contacts"),
            ],
        )
        out["n_contacts_per_env"] = _try_arr(ncon)
        out["n_contacts_attr_path"] = ncon_path

        # ── per-contact-pair info via proper API: entity.get_contacts ─
        # Genesis exposes a documented dict {geom_a, geom_b, link_a,
        # link_b, position, force_a, force_b, valid_mask} per contact.
        # Filter to foot ↔ ground only and dump env 0's contacts.
        try:
            ground_entity = None
            # Find the ground entity (TerrainImporter-owned).
            ti = getattr(env.scene_manager, "terrain", None)
            if ti is not None:
                ground_entity = getattr(ti, "entity", None)
            # Fall back: scan scene entities for a plane/terrain morph.
            if ground_entity is None:
                for ent in scene.entities:
                    morph = getattr(ent, "morph", None)
                    if morph is None:
                        continue
                    if "plane" in type(morph).__name__.lower() or "terrain" in type(morph).__name__.lower():
                        ground_entity = ent
                        break
            if ground_entity is not None:
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
                            cf[k] = arr[0][:16]  # first 16 contacts of env 0
                        else:
                            cf[k] = arr[:16]
                out["foot_ground_contacts_env0"] = cf
                # Compute aggregate stats: per-foot force magnitudes, depth
                # via geom-pair grouping. The dict's force_a is force on
                # robot's geom; for foot-ground, force_a should be the
                # foot's reaction (upward).
                try:
                    import torch as _t

                    fa = contacts.get("force_a")
                    if fa is not None:
                        fa_t = _t.as_tensor(_try_arr(fa))
                        if fa_t.ndim == 3:  # (n_envs, n_contacts, 3)
                            magn = fa_t.norm(dim=-1)
                            out["foot_ground_contact_count_env0"] = int((magn[0] > 0).sum().item())
                            out["foot_ground_force_per_contact_env0"] = magn[0][magn[0] > 0].tolist()[:32]
                            out["foot_ground_force_total_env0"] = float(magn[0].sum().item())
                        else:
                            magn = fa_t.norm(dim=-1)
                            out["foot_ground_force_per_contact"] = magn[magn > 0].tolist()[:32]
                except Exception as e:
                    out["foot_ground_force_aggregate_err"] = repr(e)
            else:
                out["foot_ground_contacts_err"] = "ground entity not found"
        except Exception as e:
            out["foot_ground_contacts_err"] = repr(e)

        # ── foot link mass + collision geometry ───────────────────────
        foot_info: list[dict] = []
        for fname in cfg.robot.foot_names:
            try:
                link = robot.get_link(fname)
                fi: dict = {"name": fname}
                fi["mass"] = float(getattr(link, "_inertial_mass", getattr(link, "inertial_mass", 0)))
                # geoms on this link
                geoms = getattr(link, "geoms", None) or []
                fi["n_geoms"] = len(geoms)
                if geoms:
                    g = geoms[0]
                    fi["geom0_type"] = type(g).__name__
                    fi["geom0_friction"] = _try_arr(getattr(g, "friction", None))
                    fi["geom0_sol_params"] = _try_arr(getattr(g, "_sol_params", getattr(g, "sol_params", None)))
                    mesh = getattr(g, "_init_mesh", getattr(g, "init_mesh", None))
                    if mesh is not None:
                        verts = getattr(mesh, "verts", None) or getattr(mesh, "vertices", None)
                        if verts is not None:
                            fi["geom0_mesh_n_verts"] = len(verts) if hasattr(verts, "__len__") else None
                foot_info.append(fi)
            except Exception as e:
                foot_info.append({"name": fname, "err": repr(e)})
        out["feet"] = foot_info

        # ── total robot mass ─────────────────────────────────────────
        try:
            total_mass = 0.0
            for link in getattr(robot, "links", []):
                total_mass += float(getattr(link, "_inertial_mass", getattr(link, "inertial_mass", 0)))
            out["robot_total_mass"] = total_mass
        except Exception as e:
            out["robot_total_mass_err"] = repr(e)

        # ── effective sol_params per foot/ground geom (post-sanitize) ─
        try:
            # rigid solver stores sanitized sol_params after model_post_init
            sp, sp_path = _walk_paths(
                rs,
                [
                    ("geoms_info", "sol_params"),
                    ("_geoms_sol_params",),
                    ("geoms_sol_params",),
                ],
            )
            if sp is not None:
                arr = _try_arr(sp)
                if isinstance(arr, list):
                    out["effective_sol_params_first_few"] = arr[:8]
                    out["effective_sol_params_path"] = sp_path
        except Exception as e:
            out["effective_sol_params_err"] = repr(e)
    except Exception as e:
        out["genesis_introspect_top_err"] = repr(e)
    return out


def _mujoco_dump(env, cfg) -> dict:
    """MuJoCo (mjlab) deep introspection.

    Captures contact count, per-contact details, foot mass/geom info,
    actual solver iterations used, and the foot geom's solref/solimp.
    """
    out: dict = {}
    try:
        sm = env.scene_manager
        mj = sm.mj_model
        wd = sm.data  # mujoco-warp Data

        # ── native contact count per world ────────────────────────────
        # mujoco-warp 's data exposes ``ncon`` as a warp scalar OR a per-
        # world array, depending on version. Try several names.
        ncon = None
        for name in ("ncon", "n_contact", "nacon", "naconmax"):
            v = getattr(wd, name, None)
            if v is not None:
                ncon = v
                out["ncon_attr"] = name
                break
        # Also try the ``contact`` sub-namespace.
        if ncon is None and hasattr(wd, "contact"):
            for name in ("ncon", "n_contact", "size"):
                v = getattr(wd.contact, name, None)
                if v is not None:
                    ncon = v
                    out["ncon_attr"] = f"contact.{name}"
                    break
        out["ncon_per_world"] = _try_arr(ncon)

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
        try:
            if hasattr(wd, "contact"):
                ctx: dict = {}
                # 1) Total contact buffer size + active count.
                ncon_total = getattr(wd, "naconmax", None) or getattr(wd, "nconmax", None)
                ctx["buffer_naconmax"] = _try_arr(ncon_total)
                # 2) Per-contact fields — convert and filter to foot↔ground.
                fields_to_dump = ("dist", "pos", "frame", "solref", "solimp", "geom", "worldid")
                raw: dict = {}
                for f in fields_to_dump:
                    v = getattr(wd.contact, f, None)
                    if v is not None:
                        raw[f] = _try_arr(v)
                # 3) Build foot-↔-ground filter:
                #    contact has geom pair where one is a foot geom and other is ground.
                foot_geom_ids: set[int] = set()
                ground_geom_ids: set[int] = set()
                for fname in cfg.robot.foot_names:
                    for cand in (f"robot/{fname}", fname):
                        try:
                            bid = mj.body(cand).id
                            for gid in range(mj.ngeom):
                                if mj.geom_bodyid[gid] == bid:
                                    foot_geom_ids.add(gid)
                            break
                        except Exception:
                            continue
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
                    rows: list[dict] = []
                    for i, (g_pair, w, d) in enumerate(zip(geom, worldid, dist)):
                        if w != 0:
                            continue  # filter to env 0
                        g1_, g2_ = g_pair[0], g_pair[1]
                        if (g1_ in foot_geom_ids and g2_ in ground_geom_ids) or (
                            g2_ in foot_geom_ids and g1_ in ground_geom_ids
                        ):
                            entry = {"idx": i, "geom": g_pair, "dist": d}
                            if "pos" in raw:
                                entry["pos"] = raw["pos"][i]
                            if "frame" in raw:
                                # frame is 3x3 — normal = frame[0]
                                entry["normal"] = raw["frame"][i][0] if isinstance(raw["frame"][i], list) else None
                            if "solref" in raw:
                                entry["solref"] = raw["solref"][i]
                            if "solimp" in raw:
                                entry["solimp"] = raw["solimp"][i]
                            rows.append(entry)
                            if len(rows) >= 16:
                                break
                    ctx["foot_ground_contacts_env0"] = rows
                    ctx["foot_ground_contacts_env0_count"] = len(rows)
                out["contact_detail"] = ctx
        except Exception as e:
            out["contact_detail_err"] = repr(e)
            traceback.print_exc()

        # ── foot body mass + geom info ───────────────────────────────
        # mjlab prefixes body names with ``robot/`` (entity name).
        foot_info: list[dict] = []
        for fname in cfg.robot.foot_names:
            # Try several name variants (mjlab prepends entity name).
            bid = None
            tried_name = None
            for candidate in (f"robot/{fname}", fname, f"/{fname}"):
                try:
                    bid = mj.body(candidate).id
                    tried_name = candidate
                    break
                except Exception:
                    continue
            if bid is None:
                foot_info.append({"name": fname, "err": f"no body with name in mj_model ({fname!r})"})
                continue
            try:
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
            except Exception as e:
                foot_info.append({"name": fname, "err": repr(e)})
        out["feet"] = foot_info

        # ── total robot mass ─────────────────────────────────────────
        try:
            out["robot_total_mass"] = float(mj.body_mass[1:].sum())  # skip world body
        except Exception as e:
            out["robot_total_mass_err"] = repr(e)

        # ── solver actual niter (mujoco-warp Data exposes per-world) ──
        for fname in ("solver_niter", "solver_iteration", "niter"):
            v = getattr(wd, fname, None)
            if v is not None:
                out[fname] = _try_arr(v)

        # ── mj_model opt block ───────────────────────────────────────
        out["opt"] = {
            "solref_default": mj.opt.o_solref.tolist() if hasattr(mj.opt, "o_solref") else None,
            "solimp_default": mj.opt.o_solimp.tolist() if hasattr(mj.opt, "o_solimp") else None,
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
    except Exception as e:
        out["mujoco_introspect_top_err"] = repr(e)
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
    import torch

    from rlworld.rl.configs.presets.g1_29dof.base import G1FlatConfig
    from rlworld.rl.runners import BaseRunner

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
    sim_cfg: dict = {"physics_dt": getattr(env, "physics_dt", None), "decimation": env.decimation}
    if sim == "genesis":
        ro = env.scene_manager.config.rigid_options
        sim_cfg["genesis_rigid_options"] = {
            "dt": getattr(ro, "dt", None),
            "constraint_solver": str(getattr(ro, "constraint_solver", None)),
            "constraint_timeconst": getattr(ro, "constraint_timeconst", None),
            "iterations": getattr(ro, "iterations", None),
            "ls_iterations": getattr(ro, "ls_iterations", None),
            "max_collision_pairs": getattr(ro, "max_collision_pairs", None),
            "contact_pruning_tolerance": getattr(ro, "contact_pruning_tolerance", None),
            "enable_self_collision": getattr(ro, "enable_self_collision", None),
            "enable_collision": getattr(ro, "enable_collision", None),
            "use_contact_island": getattr(ro, "use_contact_island", None),
            "batch_dofs_info": getattr(ro, "batch_dofs_info", None),
        }
        # Genesis effective sol-params for foot geoms (from rigid solver).
        try:
            rs = env.scene_manager.scene.sim.rigid_solver
            sim_cfg["genesis_sol_min_timeconst"] = float(getattr(rs, "_sol_min_timeconst", 0.0))
            sim_cfg["genesis_sol_default_timeconst"] = float(getattr(rs, "_sol_default_timeconst", 0.0))
        except Exception as e:
            sim_cfg["genesis_sol_introspect_err"] = repr(e)
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
            "ccd_iterations": int(mj.opt.ccd_iterations) if hasattr(mj.opt, "ccd_iterations") else None,
        }
        # Foot geom solref/solimp (per-geom). Capture the first foot geom.
        foot_names = list(cfg.robot.foot_names)
        try:
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
        except Exception as e:
            sim_cfg["mujoco_geom_introspect_err"] = repr(e)
    elif sim == "newton":
        # Newton SolverMuJoCoCfg lives on the scene-config side.
        scfg = env.scene_cfg.solver_cfg
        sim_cfg["newton_solver_cfg"] = {
            "solver": getattr(scfg, "solver", None),
            "integrator": getattr(scfg, "integrator", None),
            "cone": getattr(scfg, "cone", None),
            "iterations": getattr(scfg, "iterations", None),
            "ls_iterations": getattr(scfg, "ls_iterations", None),
            "impratio": getattr(scfg, "impratio", None),
            "use_mujoco_contacts": getattr(scfg, "use_mujoco_contacts", None),
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
    try:
        from rlworld.rl.configs.scene.entity_selector import SceneEntitySelector

        resolved = env.resolve_selector(
            SceneEntitySelector(name="robot", body_names=tuple(foot_names), preserve_order=True)
        )
        foot_ids = resolved.body_ids
        foot_pos = rd.body_pos_w_by_ids(foot_ids).detach().cpu()  # (B, 2, 3)
        foot_lin_vel = rd.body_lin_vel_w_by_ids(foot_ids).detach().cpu()  # (B, 2, 3)
    except Exception as e:
        foot_pos = None
        foot_lin_vel = None
        out["foot_resolve_err"] = repr(e)

    out["state_after_reset"] = {
        "root_pos_w_env0": root_pos[0].tolist(),
        "root_pos_w_stats": {"mean": root_pos.float().mean().item(), "std": root_pos.float().std().item()},
        "root_quat_w_env0": root_quat[0].tolist(),
        "foot_names": foot_names,
        "foot_pos_w_env0": foot_pos[0].tolist() if foot_pos is not None else None,
        "foot_lin_vel_w_env0": foot_lin_vel[0].tolist() if foot_lin_vel is not None else None,
        # Heuristic penetration: foot z minus assumed ground z=0.
        "foot_z_above_ground_env0": foot_pos[0, :, 2].tolist() if foot_pos is not None else None,
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
        try:
            forces_s = env.contact_manager.contact_force("feet_ground_contact").detach().cpu()
            fmag_s = forces_s.norm(dim=-1)
            is_c_s = env.contact_manager.is_contact("feet_ground_contact").detach().cpu()
            first_s = env.contact_manager.compute_first_contact("feet_ground_contact").detach().cpu()
        except Exception:
            forces_s = fmag_s = is_c_s = first_s = None

        if contact_first_hit_step is None and is_c_s is not None and bool(is_c_s.any()):
            contact_first_hit_step = s
            # plan deep snapshot for the peak-impact step (heuristic)
            deep_snapshot_step = min(s + 2, settle_steps - 1)

        if s < timeseries_max and forces_s is not None:
            sl_cost_s = (fmag_s * first_s.float()).sum(dim=1)
            timeseries.append(
                {
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
                }
            )

        # Deep introspection at the planned peak-impact step.
        if deep_snapshot_step is not None and s == deep_snapshot_step:
            try:
                snap = {"step": s, "tag": "peak_impact"}
                if sim == "genesis":
                    snap.update(_genesis_dump(env, cfg))
                elif sim == "mujoco":
                    snap.update(_mujoco_dump(env, cfg))
                deep_snapshots.append(snap)
            except Exception as e:
                deep_snapshots.append({"step": s, "err": repr(e)})

    out["contact_first_hit_step"] = contact_first_hit_step
    out["timeseries"] = timeseries
    out["deep_snapshots"] = deep_snapshots

    # One more deep snapshot at the END (settled state) — for comparison.
    try:
        end_snap: dict = {"step": settle_steps - 1, "tag": "settled"}
        if sim == "genesis":
            end_snap.update(_genesis_dump(env, cfg))
        elif sim == "mujoco":
            end_snap.update(_mujoco_dump(env, cfg))
        out["settled_deep"] = end_snap
    except Exception as e:
        out["settled_deep_err"] = repr(e)

    # ── Post-step state ───────────────────────────────────────────────
    rd2 = env.get_robot_data()
    root_pos_p = rd2.root_link_pos_w.detach().cpu()
    if foot_pos is not None:
        foot_pos_p = rd2.body_pos_w_by_ids(foot_ids).detach().cpu()
        foot_lin_vel_p = rd2.body_lin_vel_w_by_ids(foot_ids).detach().cpu()
    else:
        foot_pos_p = None
        foot_lin_vel_p = None

    out["state_after_step"] = {
        "root_pos_w_env0": root_pos_p[0].tolist(),
        "foot_pos_w_env0": foot_pos_p[0].tolist() if foot_pos_p is not None else None,
        "foot_lin_vel_w_env0": foot_lin_vel_p[0].tolist() if foot_lin_vel_p is not None else None,
        "foot_xy_speed_env0": [float(foot_lin_vel_p[0, i, :2].norm().item()) for i in range(foot_lin_vel_p.shape[1])]
        if foot_lin_vel_p is not None
        else None,
    }

    # ── Command state (try common attribute paths) ───────────────────
    try:
        cm = getattr(env, "command_manager", None)
        cmd = None
        if cm is not None:
            for attr in ("command", "get_commands", "compute"):
                method = getattr(cm, attr, None)
                if method is None:
                    continue
                try:
                    cmd = method() if callable(method) else method
                    if cmd is not None and hasattr(cmd, "shape"):
                        break
                except Exception:
                    continue
            # Some managers expose a dict of named command tensors —
            # concatenate the per-name tensors so the norm reflects the
            # effective command vector.
            if cmd is None and hasattr(cm, "_commands"):
                d = cm._commands
                if isinstance(d, dict) and d:
                    import torch as _t

                    cmd = _t.cat([v.reshape(v.shape[0], -1) for v in d.values() if hasattr(v, "shape")], dim=-1)
        if cmd is not None and hasattr(cmd, "shape"):
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
        else:
            out["command"] = {"note": "no compute()/command attr found on command_manager"}
    except Exception as e:
        out["command_err"] = repr(e)

    # ── Contact data (post-step, foot ↔ ground via "feet_ground_contact") ──
    try:
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
    except Exception as e:
        out["feet_ground_contact_err"] = repr(e)
        traceback.print_exc()

    # ── Per-sim native contact count (where exposed) ─────────────────
    native_count: dict = {}
    if sim == "genesis":
        try:
            rs = env.scene_manager.scene.sim.rigid_solver
            # Genesis stores contacts under collider state. Path may vary
            # across versions, so guard each step.
            n_contacts = _safe_get(rs, "collider", "_collider_state", "n_contacts")
            if n_contacts is None:
                n_contacts = _safe_get(rs, "collider", "contact_data", "n_contacts")
            if n_contacts is not None:
                arr = n_contacts.numpy() if hasattr(n_contacts, "numpy") else n_contacts
                native_count["genesis_n_contacts_per_env"] = arr.tolist() if hasattr(arr, "tolist") else int(arr)
        except Exception as e:
            native_count["genesis_err"] = repr(e)
    elif sim == "mujoco":
        try:
            wd = env.scene_manager.data  # mujoco-warp Data
            ncon = wd.ncon if hasattr(wd, "ncon") else None
            if ncon is not None:
                arr = ncon.numpy() if hasattr(ncon, "numpy") else ncon
                native_count["mujoco_ncon_per_world"] = arr.tolist() if hasattr(arr, "tolist") else int(arr)
        except Exception as e:
            native_count["mujoco_err"] = repr(e)
    out["native_contact_count"] = native_count

    # ── Reward recomputation — soft_landing + feet_slip ──────────────
    try:
        from rlworld.rl.configs.scene.entity_selector import SceneEntitySelector
        from rlworld.rl.envs.mdp.rewards.common.reward_terms import (
            _command_active,
            _feet_contact_order,
            _foot_pos_vel,
            penalize_feet_slip,
            penalize_soft_landing,
        )

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
                "feet_ground_contact", order=_feet_contact_order(feet_selector, contact_order)
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
    except Exception as e:
        out["reward_err"] = repr(e)
        traceback.print_exc()

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
        "rlworld.scripts.diag.check_g1_contact_force_parity",
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
            try:
                v = lookup(per_sim[s])
            except Exception:
                v = "—"
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
    row("mujoco ncon_per_world", lambda d: d["native_contact_count"].get("mujoco_ncon_per_world"))

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
        elif s == "mujoco":
            cd = sn.get("contact_detail") or {}
            lg(f"    buffer_naconmax: {cd.get('buffer_naconmax')}")
            lg(f"    foot_geom_ids_count: {cd.get('foot_geom_ids_count')}")
            lg(f"    ground_geom_ids_count: {cd.get('ground_geom_ids_count')}")
            lg(f"    foot_ground_contacts_env0_count: {cd.get('foot_ground_contacts_env0_count')}")
            rows = cd.get("foot_ground_contacts_env0") or []
            for r in rows[:8]:
                lg(
                    f"      idx={r.get('idx')}  geom={r.get('geom')}  dist={r.get('dist'):.6f}  "
                    f"pos={r.get('pos')}  normal={r.get('normal')}"
                )
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
            try:
                a_sl = per_sim[s]["reward_soft_landing"]["cost_mean"]
                b_sl = per_sim[ref]["reward_soft_landing"]["cost_mean"]
                lg(f"  {ratio_label}  soft_landing.cost.mean ratio = {(a_sl / b_sl) if b_sl else 'inf':.4g}")
            except Exception:
                pass
            try:
                a_fs = per_sim[s]["reward_feet_slip"]["cost_mean"]
                b_fs = per_sim[ref]["reward_feet_slip"]["cost_mean"]
                lg(f"  {ratio_label}  feet_slip.cost.mean ratio   = {(a_fs / b_fs) if b_fs else 'inf':.4g}")
            except Exception:
                pass
            try:
                a_fmag = per_sim[s]["feet_ground_contact"]["fmag_stats"]["max"]
                b_fmag = per_sim[ref]["feet_ground_contact"]["fmag_stats"]["max"]
                lg(f"  {ratio_label}  fmag.max ratio              = {(a_fmag / b_fmag) if b_fmag else 'inf':.4g}")
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
        try:
            data = _capture_sim(args.sim, args.num_envs, args.seed, args.settle_steps, args.action_mode)
        except Exception as e:
            traceback.print_exc()
            print(f"capture failed for {args.sim}: {e!r}", file=sys.stderr)
            return 1
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
