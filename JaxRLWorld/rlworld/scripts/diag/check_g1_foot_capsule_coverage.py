"""Compare foot-capsule contact coverage between Genesis and MuJoCo for g1.

Both sims load the SAME MJCF (``JaxRLWorld/rlworld/assets/g1/g1.xml``). The
``check_g1_contact_force_parity`` diag previously revealed that at peak-impact
step Genesis only generates ground contacts for 5 of 7 capsules per foot while
MuJoCo generates for all 7. Each Genesis capsule therefore carries ~2.9x the
force per contact, which is the dominant source of the residual ``fmag.max``
ratio of ~2.4x after the stale-buffer fix.

This script dumps EVERY datum needed to explain the per-capsule discrepancy in
one shot, so we don't have to keep iterating one hypothesis at a time:

  (1) g1.xml MJCF ground truth — raw <geom> attributes for every foot capsule
      (fromto, size, condim, friction, margin) parsed directly from disk.
  (2) Per-sim loaded geometry — geom_id/idx, link/body name, local fromto +
      size as the sim's parser actually stored them.
  (3) Per-sim runtime world pose at peak impact — capsule centre + both
      endpoint world positions, axis direction, signed distance from the
      lowest capsule surface point to the ground plane z=0.
  (4) Per-capsule contact result — did narrowphase produce contact? how many
      manifold points? what's the total contact force on this capsule link?
  (5) Sim-wide collider configuration — margin, contact_pruning_tolerance,
      max_collision_pairs, broadphase enable, MuJoCo opt.* fields. Every
      knob that could plausibly cause a capsule to be admitted by one sim
      and rejected by the other.

Run:

    cd JaxRLWorld
    uv run python -m rlworld.scripts.diag.check_g1_foot_capsule_coverage \
        --sims=genesis,mujoco --num-envs=16 --settle-steps=50

The script reuses the env construction + settle pattern from
``check_g1_contact_force_parity.py`` (subprocess-per-sim, peak-impact step =
first-contact + 2) so the captured state matches the previous diag's
"peak_impact" snapshot one-to-one.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

# ── Asset paths ──────────────────────────────────────────────────────────────

# Resolved relative to the JaxRLWorld root so the script works from any cwd.
_JAXRLWORLD_ROOT = Path(__file__).resolve().parents[3]
G1_XML_PATH = _JAXRLWORLD_ROOT / "rlworld" / "assets" / "g1" / "g1.xml"


# ── Generic helpers ──────────────────────────────────────────────────────────


def _try_arr(x: Any) -> Any:
    """Convert tensor / warp / numpy / quadrants tensors to a nested Python list."""
    if x is None:
        return None
    for name in ("to_numpy", "numpy", "cpu", "detach"):
        if hasattr(x, name):
            try:
                arr = getattr(x, name)()
                if hasattr(arr, "tolist"):
                    return arr.tolist()
                if hasattr(arr, "numpy"):
                    return arr.numpy().tolist()
            except Exception:
                continue
    try:
        import warp as wp

        return wp.to_torch(x).detach().cpu().numpy().tolist()
    except Exception:
        pass
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().tolist()
    except Exception:
        pass
    try:
        import numpy as np

        return np.asarray(x).tolist()
    except Exception:
        pass
    return repr(x)


def _safe(f, default=None):
    try:
        return f()
    except Exception:
        return default


# ── (1) g1.xml MJCF ground truth ─────────────────────────────────────────────


def _parse_g1_xml_foot_capsules(xml_path: Path) -> dict:
    """Read g1.xml directly and return foot capsule definitions.

    A geom is treated as a "foot collision capsule" iff (a) it has class /
    name / parent body containing ``ankle_roll`` (the foot link), and (b)
    ``type`` is ``capsule``. Everything else under those bodies (e.g. visual
    meshes) is ignored.
    """
    out: dict = {"path": str(xml_path)}
    try:
        tree = ET.parse(xml_path)
    except Exception as e:
        out["parse_err"] = repr(e)
        return out
    root = tree.getroot()

    # First pass: build a body name → list[<geom>] map. Use a single
    # ``findall(".//body")`` traversal and only collect DIRECT geom children
    # of each body (no recursion inside the loop) — otherwise nested bodies
    # get walked multiple times, multiplying every geom by tree depth.
    body_geoms: dict[str, list[ET.Element]] = {}
    worldbody = root.find("worldbody")
    if worldbody is not None:
        for body in worldbody.findall(".//body"):
            bname = body.get("name") or "?"
            geom_list = body_geoms.setdefault(bname, [])
            for child in body:
                if child.tag == "geom":
                    geom_list.append(child)

    # Walk default classes so we can resolve inherited attributes.
    default_classes: dict[str, dict] = {}
    defaults = root.find("default")
    if defaults is not None:
        for cls in defaults.findall(".//default"):
            cls_name = cls.get("class")
            if cls_name is None:
                continue
            attrs = {}
            for g in cls.findall("geom"):
                for k, v in g.attrib.items():
                    attrs[k] = v
            default_classes[cls_name] = attrs

    foot_capsules: list[dict] = []
    for bname, geoms in body_geoms.items():
        if "ankle_roll" not in bname:
            continue
        for idx_in_link, g in enumerate(geoms):
            gtype = g.get("type") or default_classes.get(g.get("class", ""), {}).get("type", "")
            if gtype != "capsule":
                continue
            cls_attrs = default_classes.get(g.get("class", ""), {})
            entry = {
                "body": bname,
                "geom_idx_in_body": idx_in_link,
                "name": g.get("name"),
                "class": g.get("class"),
                "type": gtype,
                "fromto": g.get("fromto") or cls_attrs.get("fromto"),
                "size": g.get("size") or cls_attrs.get("size"),
                "pos": g.get("pos"),
                "quat": g.get("quat"),
                "condim": g.get("condim") or cls_attrs.get("condim"),
                "friction": g.get("friction") or cls_attrs.get("friction"),
                "margin": g.get("margin") or cls_attrs.get("margin"),
                "gap": g.get("gap") or cls_attrs.get("gap"),
                "solref": g.get("solref") or cls_attrs.get("solref"),
                "solimp": g.get("solimp") or cls_attrs.get("solimp"),
                "contype": g.get("contype") or cls_attrs.get("contype"),
                "conaffinity": g.get("conaffinity") or cls_attrs.get("conaffinity"),
                "group": g.get("group") or cls_attrs.get("group"),
                "rgba": g.get("rgba") or cls_attrs.get("rgba"),
            }
            foot_capsules.append(entry)

    # Also capture compiler/default option overrides that affect contact.
    compiler = root.find("compiler")
    option = root.find("option")
    flag = option.find("flag") if option is not None else None

    out["foot_capsules"] = foot_capsules
    out["default_classes"] = {k: v for k, v in default_classes.items() if "geom" in str(v) or v}
    out["compiler"] = dict(compiler.attrib) if compiler is not None else None
    out["option"] = dict(option.attrib) if option is not None else None
    out["option_flag"] = dict(flag.attrib) if flag is not None else None
    out["num_foot_capsules"] = len(foot_capsules)
    return out


# ── (2)+(3)+(4) Genesis per-capsule dump ─────────────────────────────────────


def _genesis_dump_foot_capsules(env, cfg) -> dict:
    """Dump per-foot-capsule data from a built Genesis env at peak impact."""

    out: dict = {}
    try:
        scene = env.scene_manager.scene
        rs = scene.sim.rigid_solver
        robot = env.scene_manager["robot"]

        foot_link_names = list(cfg.robot.foot_names)
        out["foot_link_names"] = foot_link_names

        # Per-link → per-geom capsule entries. Genesis ``RigidGeom`` stores
        # the local capsule origin in ``_init_pos`` / ``_init_quat`` (frame:
        # parent link), the capsule ``[radius, height]`` in ``_data``, and
        # the type enum in ``_type``. World pose comes from ``get_pos`` /
        # ``get_quat`` which read out of ``vgeoms_state``.
        capsules: list[dict] = []
        for link in robot.links:
            lname = getattr(link, "name", "")
            if "ankle_roll" not in lname:
                continue
            for g in link.geoms:
                gtype = type(g).__name__
                gidx = getattr(g, "_idx", getattr(g, "idx", None))
                # _data layout for capsule: [radius, height]
                data = _try_arr(getattr(g, "_data", None))
                radius = data[0] if isinstance(data, list) and len(data) >= 1 else None
                height = data[1] if isinstance(data, list) and len(data) >= 2 else None
                # Local pose in parent link frame.
                local_pos = _try_arr(getattr(g, "_init_pos", None))
                local_quat = _try_arr(getattr(g, "_init_quat", None))
                # World pose at env 0 — via Genesis's own getters.
                world_pos = None
                world_quat = None
                try:
                    wp = g.get_pos(envs_idx=None)
                    world_pos = _try_arr(wp[0]) if hasattr(wp, "__getitem__") else _try_arr(wp)
                except Exception as e:
                    world_pos = f"err:{e!r}"
                try:
                    wq = g.get_quat(envs_idx=None)
                    world_quat = _try_arr(wq[0]) if hasattr(wq, "__getitem__") else _try_arr(wq)
                except Exception as e:
                    world_quat = f"err:{e!r}"
                # Capsule endpoints in LOCAL frame (axis = z by Genesis convention):
                # ``height`` is the distance between the two sphere centres, so the
                # endpoints are ±height/2 along the local +z axis (transformed by
                # _init_quat into the parent link frame and then by the world pose).
                # We just report the local axis half-length here; world endpoints
                # are computed in the render layer using world_quat.
                gtype_enum = getattr(g, "_type", None)
                gtype_name = None
                try:
                    gtype_name = gtype_enum.name if hasattr(gtype_enum, "name") else str(gtype_enum)
                except Exception:
                    gtype_name = repr(gtype_enum)
                entry: dict = {
                    "link_name": lname,
                    "link_idx": getattr(link, "_idx", getattr(link, "idx", None)),
                    "geom_idx": int(gidx) if gidx is not None else None,
                    "py_type": gtype,
                    "gs_type": gtype_name,
                    "radius": radius,
                    "height": height,
                    "local_pos": local_pos,
                    "local_quat": local_quat,
                    "world_pos_env0": world_pos,
                    "world_quat_env0": world_quat,
                    "friction": _safe(lambda g=g: float(getattr(g, "_friction", float("nan")))),
                    "contype": _safe(lambda g=g: getattr(g, "_contype", None)),
                    "conaffinity": _safe(lambda g=g: getattr(g, "_conaffinity", None)),
                    "is_convex": _safe(lambda g=g: bool(getattr(g, "_is_convex", True))),
                    "needs_coup": _safe(lambda g=g: bool(getattr(g, "_needs_coup", False))),
                    "sol_params": _try_arr(_safe(lambda g=g: g.sol_params)),
                }
                capsules.append(entry)
        out["genesis_foot_capsules_static"] = capsules

        # Total geom count (sanity).
        try:
            out["n_geoms_total"] = int(rs.n_geoms)
        except Exception as e:
            out["n_geoms_total_err"] = repr(e)

        # ── Contact result per capsule ────────────────────────────────
        # Re-query collider.get_contacts; the parity diag already proved this
        # is correct after the n_env0 slice fix.
        try:
            ncon_per_env = _try_arr(rs.collider._collider_state.n_contacts)
            n_env0 = int(ncon_per_env[0]) if isinstance(ncon_per_env, list) and ncon_per_env else 32
            out["n_contacts_env0"] = n_env0

            ground_entity = None
            ti = getattr(env.scene_manager, "terrain", None)
            if ti is not None:
                ground_entity = getattr(ti, "entity", None)
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
                ga = _try_arr(contacts.get("geom_a")) or []
                fa = _try_arr(contacts.get("force_a")) or []
                pos = _try_arr(contacts.get("position")) or []
                ga0 = ga[0][:n_env0] if ga and isinstance(ga[0], list) else ga[:n_env0]
                fa0 = fa[0][:n_env0] if fa and isinstance(fa[0], list) else fa[:n_env0]
                pos0 = pos[0][:n_env0] if pos and isinstance(pos[0], list) else pos[:n_env0]
                per_geom: dict[int, dict] = {}
                for i in range(min(len(ga0), len(fa0))):
                    gi = int(ga0[i])
                    fvec = fa0[i] if i < len(fa0) else [0, 0, 0]
                    fmag = (fvec[0] ** 2 + fvec[1] ** 2 + fvec[2] ** 2) ** 0.5
                    cpos = pos0[i] if i < len(pos0) else None
                    g = per_geom.setdefault(
                        gi,
                        {
                            "n_contacts": 0,
                            "total_force_mag": 0.0,
                            "max_force_mag": 0.0,
                            "min_force_mag": float("inf"),
                            "contact_positions": [],
                        },
                    )
                    g["n_contacts"] += 1
                    g["total_force_mag"] += fmag
                    g["max_force_mag"] = max(g["max_force_mag"], fmag)
                    g["min_force_mag"] = min(g["min_force_mag"], fmag)
                    if cpos is not None:
                        g["contact_positions"].append(cpos)
                out["genesis_contact_by_geom_env0"] = {int(k): v for k, v in per_geom.items()}
        except Exception as e:
            out["genesis_contact_per_geom_err"] = repr(e)

        # ── Sim-wide collider config ──────────────────────────────────
        try:
            ro = env.scene_manager.config.rigid_options
            collider = rs.collider
            collider_info = getattr(collider, "_collider_info", None)
            collider_static = getattr(collider, "_collider_static_config", None)
            out["genesis_collider_config"] = {
                "max_collision_pairs": getattr(ro, "max_collision_pairs", None),
                "contact_pruning_tolerance": getattr(ro, "contact_pruning_tolerance", None),
                "enable_self_collision": getattr(ro, "enable_self_collision", None),
                "enable_collision": getattr(ro, "enable_collision", None),
                "use_contact_island": getattr(ro, "use_contact_island", None),
                "constraint_timeconst": getattr(ro, "constraint_timeconst", None),
                "_sol_default_timeconst": float(getattr(rs, "_sol_default_timeconst", float("nan"))),
                "_sol_min_timeconst": float(getattr(rs, "_sol_min_timeconst", float("nan"))),
                "integrator": str(getattr(rs, "_integrator", None)),
                "collider_info_max_collision_pairs": _safe(lambda: int(collider_info.max_collision_pairs[None]))
                if collider_info is not None
                else None,
                "collider_info_max_collision_pairs_broad": _safe(
                    lambda: int(collider_info.max_collision_pairs_broad[None])
                )
                if collider_info is not None
                else None,
                "static_config_has_prunable_contacts": _safe(lambda: bool(collider_static.has_prunable_contacts))
                if collider_static is not None
                else None,
                "static_config_spatial_sort_supported": _safe(lambda: bool(collider_static.spatial_sort_supported))
                if collider_static is not None
                else None,
            }
        except Exception as e:
            out["genesis_collider_cfg_err"] = repr(e)

        # ── Per-geom sol_params (effective post-sanitisation) ─────────
        try:
            sp = _try_arr(rs.geoms_info.sol_params)
            out["genesis_geom_sol_params_for_foot_capsules"] = [
                {
                    "geom_idx": c["geom_idx"],
                    "sol_params": sp[c["geom_idx"]] if c["geom_idx"] is not None and c["geom_idx"] < len(sp) else None,
                }
                for c in capsules
            ]
        except Exception as e:
            out["genesis_sol_params_err"] = repr(e)

    except Exception as e:
        out["genesis_dump_err"] = repr(e)
        out["genesis_dump_tb"] = traceback.format_exc()
    return out


# ── (2)+(3)+(4) MuJoCo per-capsule dump ──────────────────────────────────────


def _mujoco_dump_foot_capsules(env, cfg) -> dict:
    """Dump per-foot-capsule data from a built MuJoCo env at peak impact."""
    out: dict = {}
    try:
        scene_manager = env.scene_manager
        mj = scene_manager.mj_model
        # ``scene_manager.data`` is the mujoco-warp Data (per-world batched).
        # Field access mirrors check_g1_contact_force_parity.py.
        data = scene_manager.data

        # Debug: enumerate ``data`` and ``data.contact`` attribute names so we
        # can see exactly what mujoco-warp exposes. If a previous run came back
        # with empty contacts and ``ncon_attr=None``, the cause is almost
        # always a renamed attribute; this dump pinpoints which name to use.
        try:
            out["debug_data_attrs"] = sorted([a for a in dir(data) if not a.startswith("_")])[:80]
            if hasattr(data, "contact"):
                out["debug_data_contact_attrs"] = sorted([a for a in dir(data.contact) if not a.startswith("_")])[:80]
            else:
                out["debug_data_contact_attrs"] = "data has no 'contact' attr"
            out["debug_data_type"] = type(data).__name__
            out["debug_data_module"] = type(data).__module__
        except Exception as e:
            out["debug_data_attrs_err"] = repr(e)

        # MuJoCo: pick all geoms whose NAME contains ``_foot`` and ends with
        # ``_collision`` (g1.xml convention: ``left_foot1_collision`` etc.).
        # Body-name filtering doesn't work cleanly across mujoco-warp because
        # ``geom_bodyid`` may not be exposed; geom name pattern is the
        # canonical approach used by the existing parity diag.
        capsules: list[dict] = []
        try:
            n_geoms = int(mj.ngeom)
            for gid in range(n_geoms):
                geom_name = mj.geom(gid).name or ""
                if "_foot" not in geom_name or "_collision" not in geom_name:
                    continue
                bid = _safe(lambda gid=gid: int(mj.geom_bodyid[gid]))
                bname = _safe(lambda bid=bid: mj.body(bid).name) if bid is not None else None
                gtype = int(mj.geom_type[gid])
                entry = {
                    "geom_id": gid,
                    "geom_name": geom_name,
                    "body_id": bid,
                    "body_name": bname,
                    "type_int": gtype,  # 6 = mjGEOM_CAPSULE
                    "local_pos": mj.geom_pos[gid].tolist(),
                    "local_quat": mj.geom_quat[gid].tolist(),
                    "size": mj.geom_size[gid].tolist(),
                    "friction": mj.geom_friction[gid].tolist(),
                    "margin": float(mj.geom_margin[gid]),
                    "gap": float(mj.geom_gap[gid]),
                    "solref": mj.geom_solref[gid].tolist(),
                    "solimp": mj.geom_solimp[gid].tolist(),
                    "condim": int(mj.geom_condim[gid]),
                    "contype": int(mj.geom_contype[gid]),
                    "conaffinity": int(mj.geom_conaffinity[gid]),
                    "group": int(mj.geom_group[gid]),
                    "priority": _safe(lambda gid=gid: int(mj.geom_priority[gid])),
                }
                capsules.append(entry)
        except Exception as e:
            out["mujoco_geom_scan_err"] = repr(e)
            out["mujoco_geom_scan_tb"] = traceback.format_exc()
        out["mujoco_foot_capsules_static"] = capsules

        # ── Runtime world pose at current step (env 0) ────────────────
        # mujoco-warp packs geom_xpos/geom_xmat as (n_world, n_geoms, ...).
        try:
            geom_xpos = _try_arr(data.geom_xpos)
            geom_xmat = _try_arr(data.geom_xmat)
            for c in capsules:
                gid = c["geom_id"]
                try:
                    if isinstance(geom_xpos, list) and geom_xpos and isinstance(geom_xpos[0], list):
                        if isinstance(geom_xpos[0][0], list):  # (n_world, n_geoms, 3)
                            c["world_pos_env0"] = geom_xpos[0][gid]
                        else:  # (n_geoms, 3)
                            c["world_pos_env0"] = geom_xpos[gid]
                    if isinstance(geom_xmat, list) and geom_xmat and isinstance(geom_xmat[0], list):
                        if isinstance(geom_xmat[0][0], list):
                            c["world_mat_env0"] = geom_xmat[0][gid]
                        else:
                            c["world_mat_env0"] = geom_xmat[gid]
                except Exception:
                    pass
        except Exception as e:
            out["mujoco_world_pose_err"] = repr(e)

        # ── Contact result per capsule (env 0) ────────────────────────
        # mujoco-warp data: contact buffer at ``data.contact.{geom,dist,pos,frame,...}``,
        # shape ``(naconmax,)`` packed across worlds. Per-world contact index
        # is given by ``contact.worldid``. ``ncon`` is exposed as either a
        # warp scalar or per-world array depending on version, so we probe.
        try:
            # 1) Resolve ncon (active contact count, total across all worlds).
            ncon_attr = None
            ncon = None
            for name in ("ncon", "n_contact", "nacon", "naconmax"):
                v = getattr(data, name, None)
                if v is not None:
                    ncon = v
                    ncon_attr = name
                    break
            if ncon is None and hasattr(data, "contact"):
                for name in ("ncon", "n_contact", "size"):
                    v = getattr(data.contact, name, None)
                    if v is not None:
                        ncon = v
                        ncon_attr = f"contact.{name}"
                        break
            ncon_arr = _try_arr(ncon)
            out["mujoco_ncon_attr"] = ncon_attr
            out["mujoco_ncon_raw"] = ncon_arr
            out["mujoco_naconmax"] = _try_arr(getattr(data, "naconmax", None) or getattr(data, "nconmax", None))

            # 2) Per-contact fields (all envs interleaved by worldid).
            contact_geom = _try_arr(data.contact.geom) if hasattr(data, "contact") else None
            contact_dist = _try_arr(data.contact.dist) if hasattr(data, "contact") else None
            contact_pos = _try_arr(data.contact.pos) if hasattr(data, "contact") else None
            contact_worldid = _try_arr(data.contact.worldid) if hasattr(data, "contact") else None
            contact_force = _try_arr(getattr(data.contact, "force", None)) if hasattr(data, "contact") else None

            foot_geom_ids = {c["geom_id"] for c in capsules}
            per_geom: dict[int, dict] = {}
            n_env0_contacts_seen = 0

            # 3) Pick out env-0 contacts and aggregate per foot geom.
            if contact_geom is not None and contact_worldid is not None:
                # ncon_total can be a per-world or scalar. We just iterate the
                # full buffer length and filter by worldid == 0 (env 0).
                for i in range(len(contact_geom)):
                    wid = int(contact_worldid[i]) if i < len(contact_worldid) else -1
                    if wid != 0:
                        continue
                    pair = contact_geom[i]
                    if not isinstance(pair, list) or len(pair) < 2:
                        continue
                    g1_, g2_ = int(pair[0]), int(pair[1])
                    matched_fg = None
                    if g1_ in foot_geom_ids:
                        matched_fg = g1_
                    elif g2_ in foot_geom_ids:
                        matched_fg = g2_
                    if matched_fg is None:
                        continue
                    n_env0_contacts_seen += 1
                    g = per_geom.setdefault(
                        matched_fg,
                        {
                            "n_contacts": 0,
                            "total_force_mag": 0.0,
                            "max_force_mag": 0.0,
                            "min_force_mag": float("inf"),
                            "penetrations": [],
                            "positions": [],
                        },
                    )
                    g["n_contacts"] += 1
                    if contact_dist is not None and i < len(contact_dist):
                        g["penetrations"].append(contact_dist[i])
                    if contact_pos is not None and i < len(contact_pos):
                        g["positions"].append(contact_pos[i])
                    if contact_force is not None and i < len(contact_force):
                        fvec = contact_force[i]
                        if isinstance(fvec, list) and len(fvec) >= 3:
                            fmag = (fvec[0] ** 2 + fvec[1] ** 2 + fvec[2] ** 2) ** 0.5
                            g["total_force_mag"] += fmag
                            g["max_force_mag"] = max(g["max_force_mag"], fmag)
                            g["min_force_mag"] = min(g["min_force_mag"], fmag)
            out["mujoco_foot_contacts_env0_count"] = n_env0_contacts_seen
            # For backward compatibility with the render code: report env0
            # total contacts under the historical key as well.
            out["mujoco_ncon_env0"] = n_env0_contacts_seen
            out["mujoco_contact_by_geom_env0"] = {int(k): v for k, v in per_geom.items()}
        except Exception as e:
            out["mujoco_contact_per_geom_err"] = repr(e)
            out["mujoco_contact_per_geom_tb"] = traceback.format_exc()

        # ── Sim-wide collider config (opt + default) ──────────────────
        try:
            out["mujoco_opt"] = {
                "timestep": float(mj.opt.timestep),
                "solver": int(mj.opt.solver),
                "iterations": int(mj.opt.iterations),
                "ls_iterations": int(mj.opt.ls_iterations),
                "tolerance": float(mj.opt.tolerance),
                "impratio": float(mj.opt.impratio),
                "integrator": int(mj.opt.integrator),
                "cone": int(mj.opt.cone),
                "noslip_iterations": int(mj.opt.noslip_iterations),
                "ccd_iterations": int(getattr(mj.opt, "ccd_iterations", -1)),
                "disableflags": int(getattr(mj.opt, "disableflags", 0)),
                "enableflags": int(getattr(mj.opt, "enableflags", 0)),
                "default_solref": mj.opt.o_solref.tolist() if hasattr(mj.opt, "o_solref") else None,
                "default_solimp": mj.opt.o_solimp.tolist() if hasattr(mj.opt, "o_solimp") else None,
            }
        except Exception as e:
            out["mujoco_opt_err"] = repr(e)
    except Exception as e:
        out["mujoco_dump_err"] = repr(e)
        out["mujoco_dump_tb"] = traceback.format_exc()
    return out


# ── Env build + settle + dump (per-sim, runs in subprocess) ──────────────────


def _capture_sim(sim: str, num_envs: int, seed: int, settle_steps: int) -> dict:
    """Build a g1_29dof env in ``sim``, settle, snapshot at peak-impact step.

    Returns a dict mirroring the structure used by ``check_g1_contact_force_parity.py``
    but containing ONLY the foot-capsule coverage fields needed for this diag.
    """
    import torch

    from rlworld.rl.configs.presets.g1_29dof.base import G1FlatConfig
    from rlworld.rl.runners import BaseRunner

    out: dict = {
        "sim": sim,
        "num_envs": num_envs,
        "seed": seed,
        "settle_steps": settle_steps,
    }

    cfg = G1FlatConfig(sim_type=sim, num_envs=num_envs, seed=seed)
    cfgs = cfg.build()
    runner = BaseRunner.create_with_env(cfgs)
    env = runner.env

    env.reset()

    n_act = env.num_actions
    zero = torch.zeros(env.num_envs, n_act, device=env.device)

    # Step until first contact + 2 → peak-impact heuristic (matches the
    # parity diag exactly so the snapshot lines up step-by-step).
    contact_first_hit_step = None
    peak_impact_step = None
    snapshot_taken = False

    for s in range(settle_steps):
        env.step(zero)
        try:
            is_c = env.contact_manager.is_contact("feet_ground_contact").detach().cpu()
        except Exception:
            is_c = None
        if contact_first_hit_step is None and is_c is not None and bool(is_c.any()):
            contact_first_hit_step = s
            peak_impact_step = min(s + 2, settle_steps - 1)

        # Dump at peak impact.
        if peak_impact_step is not None and s == peak_impact_step and not snapshot_taken:
            if sim == "genesis":
                out["peak_impact_snapshot"] = _genesis_dump_foot_capsules(env, cfg)
            elif sim == "mujoco":
                out["peak_impact_snapshot"] = _mujoco_dump_foot_capsules(env, cfg)
            out["peak_impact_step"] = s
            snapshot_taken = True

    out["contact_first_hit_step"] = contact_first_hit_step
    out["peak_impact_step_planned"] = peak_impact_step

    # Also dump at the final (settled) step for comparison.
    if sim == "genesis":
        out["settled_snapshot"] = _genesis_dump_foot_capsules(env, cfg)
    elif sim == "mujoco":
        out["settled_snapshot"] = _mujoco_dump_foot_capsules(env, cfg)
    out["settled_step"] = settle_steps - 1

    return out


# ── Side-by-side render ──────────────────────────────────────────────────────


def _render_comparison(g1_xml: dict, gen: dict | None, muj: dict | None) -> str:
    lines: list[str] = []

    def section(title: str) -> None:
        lines.append("")
        lines.append("━" * 75)
        lines.append(title)
        lines.append("━" * 75)

    # (1) g1.xml ground truth
    section("(1) g1.xml MJCF ground truth — foot capsule definitions")
    lines.append(f"path: {g1_xml.get('path')}")
    lines.append(f"num_foot_capsules: {g1_xml.get('num_foot_capsules')}")
    if g1_xml.get("compiler"):
        lines.append(f"compiler: {g1_xml['compiler']}")
    if g1_xml.get("option"):
        lines.append(f"option: {g1_xml['option']}")
    if g1_xml.get("option_flag"):
        lines.append(f"option/flag: {g1_xml['option_flag']}")
    for i, c in enumerate(g1_xml.get("foot_capsules") or []):
        lines.append(f"  [{i:2d}] body={c['body']}  name={c.get('name')!r}  class={c.get('class')!r}")
        for k in (
            "fromto",
            "size",
            "pos",
            "quat",
            "condim",
            "friction",
            "margin",
            "gap",
            "solref",
            "solimp",
            "contype",
            "conaffinity",
            "group",
        ):
            v = c.get(k)
            if v is not None:
                lines.append(f"        {k}: {v}")

    # (2) + (3) + (4) Per-sim peak-impact snapshots
    for label, dump in (("GENESIS", gen), ("MUJOCO", muj)):
        section(f"(2)+(3)+(4) {label} — peak-impact step snapshot")
        if dump is None:
            lines.append("  (no data — sim was not requested or failed)")
            continue
        lines.append(f"  contact_first_hit_step: {dump.get('contact_first_hit_step')}")
        lines.append(f"  peak_impact_step: {dump.get('peak_impact_step')}")
        snap = dump.get("peak_impact_snapshot") or {}
        if not snap:
            lines.append("  (peak_impact_snapshot missing)")
            continue

        if label == "GENESIS":
            caps = snap.get("genesis_foot_capsules_static") or []
            contact_by_geom = snap.get("genesis_contact_by_geom_env0") or {}
            lines.append(f"  n_contacts_env0: {snap.get('n_contacts_env0')}")
            for c in caps:
                gi = c.get("geom_idx")
                cr = contact_by_geom.get(int(gi)) if gi is not None else None
                contact_status = "✗ NONE" if cr is None else f"✓ n={cr['n_contacts']}"
                lines.append(
                    f"  geom_idx={gi:>3}  link={str(c.get('link_name')):<25}  type={c.get('gs_type')}  {contact_status}"
                )
                lines.append(f"        radius:       {c.get('radius')}")
                lines.append(
                    f"        height:       {c.get('height')}    (capsule end-to-end gap between sphere centres)"
                )
                lines.append(f"        local_pos:    {c.get('local_pos')}")
                lines.append(f"        local_quat:   {c.get('local_quat')}")
                lines.append(f"        world_pos:    {c.get('world_pos_env0')}")
                lines.append(f"        world_quat:   {c.get('world_quat_env0')}")
                lines.append(f"        friction:     {c.get('friction')}")
                lines.append(
                    f"        contype:      {c.get('contype')}  conaffinity: {c.get('conaffinity')}  "
                    f"is_convex: {c.get('is_convex')}  needs_coup: {c.get('needs_coup')}"
                )
                lines.append(f"        sol_params:   {c.get('sol_params')}")
                if cr is not None:
                    lines.append(
                        f"        contact:      tot_F={cr['total_force_mag']:.2f}N  "
                        f"max_F={cr['max_force_mag']:.2f}N  "
                        f"min_F={(cr['min_force_mag'] if cr['min_force_mag'] != float('inf') else 0):.2f}N"
                    )
                    pos_l = cr.get("contact_positions") or []
                    if pos_l:
                        lines.append(f"        positions:    {pos_l[:4]}")
            lines.append("")
            lines.append(f"  genesis_collider_config: {snap.get('genesis_collider_config')}")
        elif label == "MUJOCO":
            caps = snap.get("mujoco_foot_capsules_static") or []
            contact_by_geom = snap.get("mujoco_contact_by_geom_env0") or {}
            lines.append(f"  mujoco_ncon_env0: {snap.get('mujoco_ncon_env0')}")
            for c in caps:
                gid = c.get("geom_id")
                cr = contact_by_geom.get(int(gid)) if gid is not None else None
                contact_status = "✗ NONE" if cr is None else f"✓ n={cr['n_contacts']}"
                lines.append(
                    f"  geom_id={gid:>3}  body={str(c.get('body_name')):<28}  name={c.get('geom_name')!r}  {contact_status}"
                )
                lines.append(f"        size:         {c.get('size')}    (radius, half-length, _)")
                lines.append(f"        local_pos:    {c.get('local_pos')}")
                lines.append(f"        local_quat:   {c.get('local_quat')}")
                lines.append(f"        world_pos:    {c.get('world_pos_env0')}")
                lines.append(f"        friction:     {c.get('friction')}")
                lines.append(f"        margin:       {c.get('margin')}  gap: {c.get('gap')}")
                lines.append(f"        solref:       {c.get('solref')}")
                lines.append(f"        solimp:       {c.get('solimp')}")
                lines.append(
                    f"        condim:       {c.get('condim')}  contype: {c.get('contype')}  conaffinity: {c.get('conaffinity')}  group: {c.get('group')}"
                )
                if cr is not None:
                    lines.append(
                        f"        contact:      tot_F={cr['total_force_mag']:.2f}N  "
                        f"max_F={cr['max_force_mag']:.2f}N  "
                        f"min_F={(cr['min_force_mag'] if cr['min_force_mag'] != float('inf') else 0):.2f}N"
                    )
                    pen = cr.get("penetrations") or []
                    if pen:
                        lines.append(f"        penetrations: {pen[:4]}")
            lines.append("")
            lines.append(f"  mujoco_opt: {snap.get('mujoco_opt')}")

    # (5) Cross-sim coverage diff
    section("(5) Cross-sim foot-capsule contact coverage")
    if gen and muj:
        gsnap = gen.get("peak_impact_snapshot") or {}
        msnap = muj.get("peak_impact_snapshot") or {}
        gcaps = gsnap.get("genesis_foot_capsules_static") or []
        mcaps = msnap.get("mujoco_foot_capsules_static") or []
        gcontact = gsnap.get("genesis_contact_by_geom_env0") or {}
        mcontact = msnap.get("mujoco_contact_by_geom_env0") or {}
        lines.append(f"  Genesis  n_foot_capsules_defined: {len(gcaps)}    n_capsules_with_contact: {len(gcontact)}")
        lines.append(f"  MuJoCo   n_foot_capsules_defined: {len(mcaps)}    n_capsules_with_contact: {len(mcontact)}")
        lines.append("")
        lines.append("  Per-capsule contact: ✓ = ≥1 contact, ✗ = 0 contacts")
        lines.append("")
        # MuJoCo capsules grouped by body (so they line up with Genesis).
        muj_by_body: dict[str, list[dict]] = {}
        for c in mcaps:
            muj_by_body.setdefault(c.get("body_name", "?"), []).append(c)
        gen_by_link: dict[str, list[dict]] = {}
        for c in gcaps:
            gen_by_link.setdefault(c.get("link_name", "?"), []).append(c)
        for body in sorted(set(list(muj_by_body.keys()) + list(gen_by_link.keys()))):
            lines.append(f"  body: {body}")
            mlist = muj_by_body.get(body, [])
            glist = gen_by_link.get(body, [])
            lines.append(f"    MuJoCo capsules: {len(mlist)}    Genesis capsules: {len(glist)}")
            for mc in mlist:
                gid = mc.get("geom_id")
                has = "✓" if int(gid) in mcontact else "✗"
                nc = mcontact.get(int(gid), {}).get("n_contacts", 0)
                lines.append(f"      MJ {has}  geom_id={gid:>3}  name={mc.get('geom_name')!r}  n_contacts={nc}")
            for gc in glist:
                gi = gc.get("geom_idx")
                has = "✓" if int(gi) in gcontact else "✗"
                nc = gcontact.get(int(gi), {}).get("n_contacts", 0)
                lines.append(
                    f"      GS {has}  geom_idx={gi:>3}  type={gc.get('type')}  size={gc.get('size')}  n_contacts={nc}"
                )
    else:
        lines.append("  (need both sims captured for diff)")

    return "\n".join(lines)


# ── Driver / inline ──────────────────────────────────────────────────────────


def _run_subprocess_capture(sim: str, num_envs: int, seed: int, settle_steps: int, out_dir: Path) -> Path:
    """Spawn ``python -m ... --no-driver --sim X`` and collect its JSON output."""
    target = out_dir / f"foot_coverage_{sim}.json"
    cmd = [
        sys.executable,
        "-m",
        "rlworld.scripts.diag.check_g1_foot_capsule_coverage",
        "--no-driver",
        f"--sim={sim}",
        f"--num-envs={num_envs}",
        f"--seed={seed}",
        f"--settle-steps={settle_steps}",
        f"--out-json={target}",
    ]
    # Inherit parent cwd — ``g1_29dof.mjcf_path`` is ``./JaxRLWorld/rlworld/...``
    # which resolves relative to the SimForge root, so we MUST run from
    # wherever the user invoked the driver (typically the SimForge root).
    # Forcing cwd to JaxRLWorld/ would break the asset lookup.
    env = os.environ.copy()
    print(f"\n[driver] launching subprocess for {sim} → {target}")
    r = subprocess.run(cmd, env=env)
    if r.returncode != 0:
        print(f"[driver] {sim} subprocess exited with code {r.returncode}", file=sys.stderr)
    return target


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sims", type=str, default="genesis,mujoco")
    ap.add_argument("--sim", type=str, default=None, choices=("genesis", "mujoco"))
    ap.add_argument("--num-envs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--settle-steps", type=int, default=50)
    ap.add_argument("--out-json", type=str, default=None)
    ap.add_argument("--out-dir", type=str, default=".")
    ap.add_argument("--no-driver", action="store_true")
    args = ap.parse_args()

    # Inline mode: capture single sim, write JSON, exit.
    if args.no_driver:
        if args.sim is None:
            print("--no-driver requires --sim", file=sys.stderr)
            return 2
        try:
            data = _capture_sim(args.sim, args.num_envs, args.seed, args.settle_steps)
        except Exception as e:
            traceback.print_exc()
            print(f"capture failed for {args.sim}: {e!r}", file=sys.stderr)
            return 1
        target = Path(args.out_json) if args.out_json else Path(f"./foot_coverage_{args.sim}.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(data, indent=2, default=str))
        print(f"\n✓ wrote {target}")
        return 0

    # Driver mode: spawn one subprocess per sim, then render comparison.
    sims = [s.strip() for s in args.sims.split(",") if s.strip()]
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # (1) Parse g1.xml directly (ground truth, no sim involved).
    g1_xml = _parse_g1_xml_foot_capsules(G1_XML_PATH)

    # (2..4) Capture per-sim.
    captures: dict[str, dict | None] = {"genesis": None, "mujoco": None}
    for sim in sims:
        target = _run_subprocess_capture(sim, args.num_envs, args.seed, args.settle_steps, out_dir)
        if target.exists():
            try:
                captures[sim] = json.loads(target.read_text())
            except Exception as e:
                print(f"[driver] failed to parse {target}: {e!r}", file=sys.stderr)

    # Render side-by-side.
    report = _render_comparison(g1_xml, captures.get("genesis"), captures.get("mujoco"))
    print(report)

    # Persist full JSON bundle.
    bundle = {
        "g1_xml": g1_xml,
        "genesis": captures.get("genesis"),
        "mujoco": captures.get("mujoco"),
    }
    bundle_path = out_dir / "foot_coverage_bundle.json"
    bundle_path.write_text(json.dumps(bundle, indent=2, default=str))
    print(f"\n✓ wrote {bundle_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
