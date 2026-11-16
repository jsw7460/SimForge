"""Compare foot-capsule contact coverage between Genesis and MuJoCo for g1.

Both sims load the SAME MJCF (``JaxRLWorld/jaxrlworld/assets/g1/g1.xml``). The
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
    uv run python -m jaxrlworld.scripts.diag.g1.check_g1_foot_capsule_coverage \
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
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import mujoco_warp as mjwarp
import numpy as np
import torch
import warp as wp

# ── Asset paths ──────────────────────────────────────────────────────────────

# Resolved relative to the JaxRLWorld root so the script works from any cwd.
_JAXRLWORLD_ROOT = Path(__file__).resolve().parents[4]
G1_XML_PATH = _JAXRLWORLD_ROOT / "jaxrlworld" / "assets" / "g1" / "g1.xml"


# ── Generic helpers ──────────────────────────────────────────────────────────


def _try_arr(x: Any) -> Any:
    """Convert a torch tensor / numpy array / Genesis (quadrants) field to a nested Python list."""
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x.to_numpy().tolist()


def _bridge_arr(x: Any) -> list:
    """Convert an mjlab ``TorchArray`` (a mujoco-warp field behind the torch bridge) to a nested list."""
    return x[...].detach().cpu().tolist()


# ── (1) g1.xml MJCF ground truth ─────────────────────────────────────────────


def _parse_g1_xml_foot_capsules(xml_path: Path) -> dict:
    """Read g1.xml directly and return foot capsule definitions.

    A geom is treated as a "foot collision capsule" iff (a) it has class /
    name / parent body containing ``ankle_roll`` (the foot link), and (b)
    ``type`` is ``capsule``. Everything else under those bodies (e.g. visual
    meshes) is ignored.
    """
    out: dict = {"path": str(xml_path)}
    tree = ET.parse(xml_path)
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
    scene = env.scene_manager.scene
    rs = scene.sim.rigid_solver
    robot = env.scene_manager["robot"]

    foot_link_names = list(cfg.robot.foot_names)
    out["foot_link_names"] = foot_link_names

    # Per-link → per-geom capsule entries. Genesis ``RigidGeom`` exposes
    # the local capsule origin as ``init_pos`` / ``init_quat`` (frame:
    # parent link), the capsule ``[radius, height]`` in ``data``, and
    # the type enum as ``type``. World pose comes from ``get_pos`` /
    # ``get_quat``; friction and sol_params from the solver-side getters
    # (the values the simulation is currently using).
    capsules: list[dict] = []
    for link in robot.links:
        lname = link.name
        if "ankle_roll" not in lname:
            continue
        for g in link.geoms:
            # data layout for capsule: [radius, height, ...]
            data = g.data.tolist()
            # Capsule endpoints in LOCAL frame (axis = z by Genesis convention):
            # ``height`` is the distance between the two sphere centres, so the
            # endpoints are ±height/2 along the local +z axis (transformed by
            # init_quat into the parent link frame and then by the world pose).
            # We just report the local axis half-length here; world endpoints
            # are computed in the render layer using world_quat.
            entry: dict = {
                "link_name": lname,
                "link_idx": link.idx,
                "geom_idx": int(g.idx),
                "py_type": type(g).__name__,
                "gs_type": g.type.name,
                "radius": data[0],
                "height": data[1],
                "local_pos": g.init_pos.tolist(),
                "local_quat": g.init_quat.tolist(),
                # World pose at env 0 — via Genesis's own getters.
                "world_pos_env0": _try_arr(g.get_pos()[0]),
                "world_quat_env0": _try_arr(g.get_quat()[0]),
                "friction": float(g.get_friction()),
                "contype": int(g.contype),
                "conaffinity": int(g.conaffinity),
                "is_convex": bool(g.is_convex),
                "needs_coup": bool(g.needs_coup),
                "sol_params": _try_arr(g.get_sol_params()),
            }
            capsules.append(entry)
    out["genesis_foot_capsules_static"] = capsules

    # Total geom count (sanity).
    out["n_geoms_total"] = int(rs.n_geoms)

    # ── Contact result per capsule ────────────────────────────────
    # Re-query collider.get_contacts; the parity diag already proved this
    # is correct after the n_env0 slice fix.
    n_env0 = int(rs.collider.collider_state.n_contacts.to_numpy()[0])
    out["n_contacts_env0"] = n_env0

    ground_entity = env.scene_manager.terrain.entity
    if ground_entity is not None:
        contacts = robot.get_contacts(with_entity=ground_entity)
        ga = _try_arr(contacts["geom_a"])
        fa = _try_arr(contacts["force_a"])
        pos = _try_arr(contacts["position"])
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

    # ── Sim-wide collider config ──────────────────────────────────
    ro = env.scene_manager.config.rigid_options
    collider = rs.collider
    collider_info = collider.collider_info
    collider_static = collider.collider_config
    out["genesis_collider_config"] = {
        "max_collision_pairs": ro.max_collision_pairs,
        "contact_pruning_tolerance": ro.contact_pruning_tolerance,
        "enable_self_collision": ro.enable_self_collision,
        "enable_collision": ro.enable_collision,
        "constraint_timeconst": ro.constraint_timeconst,
        "_sol_default_timeconst": float(rs._sol_default_timeconst),
        "_sol_min_timeconst": float(rs._sol_min_timeconst),
        "integrator": rs._integrator.name,
        "collider_info_max_collision_pairs": int(collider_info.max_collision_pairs[None]),
        "collider_info_max_collision_pairs_broad": int(collider_info.max_collision_pairs_broad[None]),
        "static_config_has_prunable_contacts": bool(collider_static.has_prunable_contacts),
    }

    # ── Per-geom sol_params (effective post-sanitisation) ─────────
    # The per-geom sol_params field ``RigidGeom.get_sol_params`` reads.
    sp = _try_arr(rs.dyn_info.geoms.sol_params)
    out["genesis_geom_sol_params_for_foot_capsules"] = [
        {
            "geom_idx": c["geom_idx"],
            "sol_params": sp[c["geom_idx"]] if c["geom_idx"] is not None and c["geom_idx"] < len(sp) else None,
        }
        for c in capsules
    ]

    return out


# ── (2)+(3)+(4) MuJoCo per-capsule dump ──────────────────────────────────────


def _mujoco_dump_foot_capsules(env, cfg) -> dict:
    """Dump per-foot-capsule data from a built MuJoCo env at peak impact."""
    out: dict = {}
    scene_manager = env.scene_manager
    mj = scene_manager.mj_model
    # ``scene_manager.data`` is the mujoco-warp Data (per-world batched).
    # Field access mirrors check_g1_contact_force_parity.py.
    data = scene_manager.data

    # Debug: enumerate ``data`` and ``data.contact`` attribute names so we
    # can see exactly what mujoco-warp exposes. If a previous run came back
    # with empty contacts and ``ncon_attr=None``, the cause is almost
    # always a renamed attribute; this dump pinpoints which name to use.
    out["debug_data_attrs"] = sorted([a for a in dir(data) if not a.startswith("_")])[:80]
    out["debug_data_contact_attrs"] = sorted([a for a in dir(data.contact) if not a.startswith("_")])[:80]
    out["debug_data_type"] = type(data).__name__
    out["debug_data_module"] = type(data).__module__

    # MuJoCo: pick all geoms whose NAME contains ``_foot`` and ends with
    # ``_collision`` (g1.xml convention: ``left_foot1_collision`` etc.).
    # Body-name filtering doesn't work cleanly across mujoco-warp because
    # ``geom_bodyid`` may not be exposed; geom name pattern is the
    # canonical approach used by the existing parity diag.
    capsules: list[dict] = []
    n_geoms = int(mj.ngeom)
    for gid in range(n_geoms):
        geom_name = mj.geom(gid).name or ""
        if "_foot" not in geom_name or "_collision" not in geom_name:
            continue
        bid = int(mj.geom_bodyid[gid])
        bname = mj.body(bid).name
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
            "priority": int(mj.geom_priority[gid]),
        }
        capsules.append(entry)
    out["mujoco_foot_capsules_static"] = capsules

    # ── Runtime world pose at current step (env 0) ────────────────
    # mujoco-warp packs geom_xpos/geom_xmat as (nworld, ngeom, ...).
    geom_xpos = _bridge_arr(data.geom_xpos)
    geom_xmat = _bridge_arr(data.geom_xmat)
    for c in capsules:
        gid = c["geom_id"]
        c["world_pos_env0"] = geom_xpos[0][gid]
        c["world_mat_env0"] = geom_xmat[0][gid]

    # ── Contact result per capsule (env 0) ────────────────────────
    # mujoco-warp data: contact buffer at ``data.contact.{geom,dist,pos,frame,...}``,
    # shape ``(naconmax,)`` packed across worlds. Per-world contact index
    # is given by ``contact.worldid``. ``nacon`` is the active contact
    # count, total across all worlds: slots ``>= nacon`` hold stale rows
    # from earlier steps and are skipped. ``Contact`` carries no force
    # field; per-contact forces come from ``mujoco_warp.contact_force``,
    # which decodes them from ``efc.force`` (world frame here).
    # 1) Active contact count.
    nacon = int(scene_manager.wp_data.nacon.numpy()[0])
    out["mujoco_ncon_attr"] = "nacon"
    out["mujoco_ncon_raw"] = _bridge_arr(data.nacon)
    out["mujoco_naconmax"] = data.naconmax

    # 2) Per-contact fields (all envs interleaved by worldid).
    contact_geom = _bridge_arr(data.contact.geom)[:nacon]
    contact_dist = _bridge_arr(data.contact.dist)[:nacon]
    contact_pos = _bridge_arr(data.contact.pos)[:nacon]
    contact_worldid = _bridge_arr(data.contact.worldid)[:nacon]
    force_out = wp.zeros(nacon, dtype=wp.spatial_vector, device=scene_manager.sim.wp_device)
    mjwarp.contact_force(
        scene_manager.sim.wp_model,
        scene_manager.wp_data,
        wp.array(np.arange(nacon, dtype=np.int32), dtype=int, device=scene_manager.sim.wp_device),
        True,
        force_out,
    )
    # spatial_vector rows are (force xyz, torque xyz).
    contact_force = force_out.numpy()[:, :3].tolist()

    foot_geom_ids = {c["geom_id"] for c in capsules}
    per_geom: dict[int, dict] = {}
    n_env0_contacts_seen = 0

    # 3) Pick out env-0 contacts and aggregate per foot geom.
    for i in range(nacon):
        if int(contact_worldid[i]) != 0:
            continue
        g1_, g2_ = int(contact_geom[i][0]), int(contact_geom[i][1])
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
        g["penetrations"].append(contact_dist[i])
        g["positions"].append(contact_pos[i])
        fvec = contact_force[i]
        fmag = (fvec[0] ** 2 + fvec[1] ** 2 + fvec[2] ** 2) ** 0.5
        g["total_force_mag"] += fmag
        g["max_force_mag"] = max(g["max_force_mag"], fmag)
        g["min_force_mag"] = min(g["min_force_mag"], fmag)
    out["mujoco_foot_contacts_env0_count"] = n_env0_contacts_seen
    out["mujoco_contact_by_geom_env0"] = {int(k): v for k, v in per_geom.items()}

    # ── Sim-wide collider config (opt + default) ──────────────────
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
        "ccd_iterations": int(mj.opt.ccd_iterations),
        "disableflags": int(mj.opt.disableflags),
        "enableflags": int(mj.opt.enableflags),
        "default_solref": mj.opt.o_solref.tolist(),
        "default_solimp": mj.opt.o_solimp.tolist(),
    }
    return out


# ── Env build + settle + dump (per-sim, runs in subprocess) ──────────────────


def _capture_sim(sim: str, num_envs: int, seed: int, settle_steps: int) -> dict:
    """Build a g1_29dof env in ``sim``, settle, snapshot at peak-impact step.

    Returns a dict mirroring the structure used by ``check_g1_contact_force_parity.py``
    but containing ONLY the foot-capsule coverage fields needed for this diag.
    """
    import torch

    from jaxrlworld.rl.configs.presets.g1_29dof.base import G1FlatConfig
    from jaxrlworld.rl.runners import BaseRunner

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
        is_c = env.contact_manager.is_contact("feet_ground_contact").detach().cpu()
        if contact_first_hit_step is None and bool(is_c.any()):
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
            lines.append(f"  mujoco_foot_contacts_env0_count: {snap.get('mujoco_foot_contacts_env0_count')}")
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
        "jaxrlworld.scripts.diag.g1.check_g1_foot_capsule_coverage",
        "--no-driver",
        f"--sim={sim}",
        f"--num-envs={num_envs}",
        f"--seed={seed}",
        f"--settle-steps={settle_steps}",
        f"--out-json={target}",
    ]
    # Inherit parent cwd — ``g1_29dof.mjcf_path`` is ``./JaxRLWorld/jaxrlworld/...``
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
        data = _capture_sim(args.sim, args.num_envs, args.seed, args.settle_steps)
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
            captures[sim] = json.loads(target.read_text())

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
