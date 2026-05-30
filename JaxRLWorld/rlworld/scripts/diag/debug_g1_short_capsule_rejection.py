"""Pinpoint the exact Genesis collider stage that drops the short toe capsules.

Empirical: ``check_g1_foot_capsule_coverage`` confirmed Genesis loads all 14 g1
foot capsules but produces ground contacts for only the 10 LONG sole capsules.
The 4 SHORT toe capsules (height = 0.05 m, vs sole 0.17-0.19 m) are dropped.

This script builds a Genesis g1_29dof env, settles to the peak-impact step,
then for every foot collision capsule reports — at each Genesis collider stage
— whether the (capsule, plane) pair was admitted, what its geometric inputs
were, and whether a narrowphase contact was actually generated. The combined
picture pinpoints the EXACT stage that drops the short capsules.

Stages probed per capsule:

  S1. ``rs.collider._collider_info.collision_pair_idx[i_capsule, i_plane]``
      — compile-time pair admission (set in
      ``Genesis/genesis/engine/solvers/rigid/collider/collider.py::_compute_collision_pair_idx``).
      Returns ``-1`` if the pair is filtered out (contype/conaffinity, fixed,
      self-collision, weld, neutral-overlap, IPC). A short capsule rejected
      here would NEVER reach broadphase.

  S2. Per-geom static config — ``geoms_info.contype``, ``conaffinity``,
      ``is_collision``, ``link_idx``, ``entity_idx`` — quoted side-by-side so
      LONG vs SHORT capsules can be diffed directly. Any per-geom field that
      differs between LONG and SHORT is the smoking gun.

  S3. ``geoms_info.data[g, :]`` (capsule has ``[radius, height]``) and
      ``geoms_init_AABB[g, :, :]`` (local-frame AABB corners 0 and 7) —
      proves the LOCAL AABB is dimensioned correctly for the capsule extent.

  S4. Runtime world pose at peak impact — ``geom.get_pos()`` /
      ``get_quat()``. Combined with the capsule radius this gives the
      expected lowest world z (capsule_centre_z - radius — only valid when
      the axis is horizontal, which is true here).

  S5. Pure-geometry expected penetration vs the ground plane z=0:
      ``pen_geom = radius - capsule_centre_z`` (positive ⇒ penetrating).
      If ``pen_geom > 0`` but Genesis reports no contact, the issue is
      ENTIRELY downstream of geometry — broadphase or narrowphase rejection.

  S6. Genesis collider contact buffer — walk ``collider_state.contact_data``
      (geom_a, geom_b, pos, penetration), filter to env 0, and count how
      many entries reference this capsule's geom index.

  S7. Runtime collider knobs — ``mc_tolerance``, ``mc_perturbation``,
      ``contact_pruning_tolerance`` — quoted as raw values so we can plug
      them into the narrowphase formulas (file:line citations in the
      printed report).

Run:

    cd $SimForge_ROOT
    python -m rlworld.scripts.diag.debug_g1_short_capsule_rejection \
        --num-envs=16 --settle-steps=50 --out-json=./short_capsule_debug.json
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any


def _try_arr(x: Any) -> Any:
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


def _capsule_lowest_world_z(centre_pos, quat_xyzw_or_wxyz, radius: float, height: float) -> float:
    """Compute the lowest world z of a capsule given its centre, quaternion, radius, height.

    Capsule axis in local frame is +z (Genesis convention). The capsule body
    extends ±(height/2) along the local z. Sphere caps add radius beyond
    each end.

    The lowest point is on whichever end-sphere is lower in world z. Its
    world z = capsule_centre_z + min(axis_z * (h/2), -axis_z * (h/2)) - radius
            = capsule_centre_z - |axis_z * (h/2)| - radius.

    When the axis is perpendicular to world z (axis_z = 0), the capsule
    cylinder body is horizontal and the lowest point is on the cylinder side
    at world z = centre_z - radius.
    """
    import numpy as np

    cx, cy, cz = centre_pos
    # Genesis quaternion convention: check both [w,x,y,z] and [x,y,z,w].
    # ``RigidGeom.get_quat`` returns [w,x,y,z] per Genesis convention; we
    # accept either and detect by the unit-norm component placement.
    q = np.asarray(quat_xyzw_or_wxyz, dtype=float)
    if abs(np.linalg.norm(q) - 1.0) > 1e-3:
        q = q / max(np.linalg.norm(q), 1e-12)
    # Assume wxyz (Genesis default); if w is in last slot, the rotation
    # below still gives the correct axis direction up to a sign which is
    # absolute-valued before use.
    w, x, y, z = q if abs(q[0]) <= 1.0 + 1e-3 else (q[3], q[0], q[1], q[2])

    # Local +z axis transformed to world: [2*(x*z + w*y), 2*(y*z - w*x), 1 - 2*(x*x + y*y)]
    axis_z = 1.0 - 2.0 * (x * x + y * y)
    half_h = 0.5 * float(height)
    return float(cz) - abs(axis_z) * half_h - float(radius)


def _genesis_debug(num_envs: int, seed: int, settle_steps: int) -> dict:
    """Build env, settle to peak impact, dump per-capsule debug picture."""
    import torch

    from rlworld.rl.configs.presets.g1_29dof.base import G1FlatConfig
    from rlworld.rl.runners import BaseRunner

    out: dict = {"sim": "genesis", "num_envs": num_envs, "seed": seed, "settle_steps": settle_steps}

    # ── Provenance: which Genesis is imported, and does the patched sentinel
    # exist in the on-disk narrowphase.py used by this interpreter? Catches
    # the case where the script is run against a pip-installed Genesis or a
    # different working copy than the one we edited locally.
    import genesis as _gs_mod

    out["genesis_module_file"] = str(_gs_mod.__file__)
    out["genesis_module_path"] = str(Path(_gs_mod.__file__).parent)
    narrow_py = Path(_gs_mod.__file__).parent / "engine/solvers/rigid/collider/narrowphase.py"
    out["narrowphase_py_path"] = str(narrow_py)
    if narrow_py.exists():
        narrow_src = narrow_py.read_text()
        out["narrowphase_py_mtime"] = narrow_py.stat().st_mtime
        out["narrowphase_py_size"] = narrow_py.stat().st_size
        out["narrowphase_has_sentinel_042"] = "qd_float(0.42)" in narrow_src
        out["narrowphase_has_force_is_col"] = "DEBUG (short-capsule rejection probe)" in narrow_src
    else:
        out["narrowphase_py_missing"] = True

    cfg = G1FlatConfig(sim_type="genesis", num_envs=num_envs, seed=seed)
    cfgs = cfg.build()
    runner = BaseRunner.create_with_env(cfgs)
    env = runner.env

    # Reset so we match the same initial state ``check_g1_foot_capsule_coverage``
    # used (spawn pose + ``mode="startup"`` DR applied). Without this the
    # robot stays in its default unposed config and the foot trajectory at
    # peak impact differs significantly — capsule world poses and per-capsule
    # narrowphase outcomes diverge from the coverage diag's state.
    env.reset()

    scene = env.scene_manager.scene
    rs = scene.sim.rigid_solver
    robot = env.scene_manager["robot"]
    ground_ent = env.scene_manager.terrain.entity

    # ── Backend & narrowphase routing — which Genesis kernel actually runs?
    out["gs_backend"] = str(getattr(_gs_mod, "backend", None))
    out["gs_cpu_enum"] = str(getattr(_gs_mod, "cpu", None))
    try:
        out["use_split_narrowphase"] = bool(rs.collider._use_split_narrowphase)
    except Exception as e:
        out["use_split_narrowphase_err"] = repr(e)
    try:
        csc = rs.collider._collider_static_config
        out["has_non_box_plane_convex_convex"] = bool(getattr(csc, "has_non_box_plane_convex_convex", None))
        out["has_convex_specialization"] = bool(getattr(csc, "has_convex_specialization", None))
        out["has_terrain"] = bool(getattr(csc, "has_terrain", None)) if hasattr(csc, "has_terrain") else None
        out["has_nonconvex_nonterrain"] = (
            bool(getattr(csc, "has_nonconvex_nonterrain", None)) if hasattr(csc, "has_nonconvex_nonterrain") else None
        )
        out["box_box_detection"] = bool(getattr(rs._static_rigid_sim_config, "box_box_detection", None))
        out["ccd_algorithm"] = str(getattr(csc, "ccd_algorithm", None))
    except Exception as e:
        out["collider_static_config_err"] = repr(e)

    # ── Resolve geom indices for the 14 foot capsules and the ground plane.
    foot_caps: list[dict] = []
    for link in robot.links:
        lname = getattr(link, "name", "")
        if "ankle_roll" not in lname:
            continue
        for g in link.geoms:
            foot_caps.append(
                {
                    "geom_idx": int(getattr(g, "_idx", getattr(g, "idx", None))),
                    "link_name": lname,
                    "link_idx": int(getattr(link, "_idx", getattr(link, "idx", None))),
                    "_geom_obj": g,  # stripped before return
                }
            )

    # Ground plane geom — find the unique geom in the ground entity.
    plane_geom_idx: int | None = None
    plane_geom_obj = None
    for link in ground_ent.links:
        for g in link.geoms:
            plane_geom_idx = int(getattr(g, "_idx", getattr(g, "idx", None)))
            plane_geom_obj = g
            break
        if plane_geom_idx is not None:
            break

    out["plane_geom_idx"] = plane_geom_idx
    out["n_geoms_total"] = int(rs.n_geoms)
    out["n_foot_caps_resolved"] = len(foot_caps)

    # ── S1. compile-time pair admission via ``collision_pair_idx`` ─────────
    try:
        cpi_arr = _try_arr(rs.collider._collider_info.collision_pair_idx)
        out["S1_collision_pair_idx_shape"] = (
            [len(cpi_arr), len(cpi_arr[0]) if cpi_arr else None] if isinstance(cpi_arr, list) and cpi_arr else None
        )
        for c in foot_caps:
            gi = c["geom_idx"]
            pi = plane_geom_idx
            # Genesis stores the upper-triangular pair-idx table, so query
            # both orderings and report whichever is non-negative.
            v_a = cpi_arr[gi][pi] if isinstance(cpi_arr, list) and gi < len(cpi_arr) and pi < len(cpi_arr[gi]) else None
            v_b = cpi_arr[pi][gi] if isinstance(cpi_arr, list) and pi < len(cpi_arr) and gi < len(cpi_arr[pi]) else None
            c["S1_pair_idx_capsule_plane"] = v_a
            c["S1_pair_idx_plane_capsule"] = v_b
            c["S1_pair_admitted"] = (v_a is not None and v_a >= 0) or (v_b is not None and v_b >= 0)
    except Exception as e:
        out["S1_err"] = repr(e)
        out["S1_tb"] = traceback.format_exc()

    # ── S2. per-geom static collider config ────────────────────────────────
    try:
        gi_contype = _try_arr(rs.geoms_info.contype)
        gi_conaffinity = _try_arr(rs.geoms_info.conaffinity)
        gi_link_idx = _try_arr(rs.geoms_info.link_idx)
        gi_entity_idx = _try_arr(rs.geoms_info.entity_idx) if hasattr(rs.geoms_info, "entity_idx") else None
        gi_friction = _try_arr(rs.geoms_info.friction)
        gi_type = _try_arr(rs.geoms_info.type)
        gi_data = _try_arr(rs.geoms_info.data)
        for c in foot_caps:
            gi = c["geom_idx"]
            c["S2_geoms_info"] = {
                "contype": gi_contype[gi] if gi_contype else None,
                "conaffinity": gi_conaffinity[gi] if gi_conaffinity else None,
                "link_idx": gi_link_idx[gi] if gi_link_idx else None,
                "entity_idx": gi_entity_idx[gi] if gi_entity_idx else None,
                "friction": gi_friction[gi] if gi_friction else None,
                "type_enum": gi_type[gi] if gi_type else None,
                # For capsule: data = [radius, height]
                "data": gi_data[gi] if gi_data else None,
            }
        # Plane's own static config.
        if plane_geom_idx is not None:
            out["plane_geoms_info"] = {
                "contype": gi_contype[plane_geom_idx] if gi_contype else None,
                "conaffinity": gi_conaffinity[plane_geom_idx] if gi_conaffinity else None,
                "link_idx": gi_link_idx[plane_geom_idx] if gi_link_idx else None,
                "entity_idx": gi_entity_idx[plane_geom_idx] if gi_entity_idx else None,
                "type_enum": gi_type[plane_geom_idx] if gi_type else None,
                "data": gi_data[plane_geom_idx] if gi_data else None,
            }
    except Exception as e:
        out["S2_err"] = repr(e)

    # ── S3. local-frame AABB (proves capsule extent is loaded correctly) ──
    try:
        init_aabb = _try_arr(rs.geoms_init_AABB)
        out["S3_init_aabb_shape_outer"] = (
            [len(init_aabb), len(init_aabb[0]) if init_aabb else None] if isinstance(init_aabb, list) else None
        )
        for c in foot_caps:
            gi = c["geom_idx"]
            if init_aabb and gi < len(init_aabb):
                # AABB stored as 8 corners (one per octant); corners 0 and 7
                # are the min and max.
                corners = init_aabb[gi]
                if isinstance(corners, list) and len(corners) >= 8:
                    c["S3_local_aabb_min"] = corners[0]
                    c["S3_local_aabb_max"] = corners[7]
                    try:
                        diag = sum((corners[7][k] - corners[0][k]) ** 2 for k in range(3)) ** 0.5
                        c["S3_local_aabb_diag"] = diag
                    except Exception:
                        pass
    except Exception as e:
        out["S3_err"] = repr(e)

    # ── S3b. per-capsule mesh vertices ─────────────────────────────────────
    # The plane-vs-capsule narrowphase path is actually
    # ``func_add_polytope_vertex_contacts_sdf`` (in
    # narrowphase.py), which iterates A's mesh verts and queries B's SDF.
    # PLANE has ``is_convex=False``, so plane-capsule routes here instead of
    # the ``_func_narrowphase_contact0`` PLANE branch.
    # If the SHORT capsule's lowest mesh vertex sits ABOVE the ground plane
    # while the LONG capsule's lowest mesh vertex penetrates the plane,
    # that's the smoking gun.
    try:
        vert_start_arr = _try_arr(rs.geoms_info.vert_start)
        vert_end_arr = _try_arr(rs.geoms_info.vert_end)
        verts_init_pos = _try_arr(rs.verts_info.init_pos)
        for c in foot_caps:
            gi = c["geom_idx"]
            vs = vert_start_arr[gi] if vert_start_arr else None
            ve = vert_end_arr[gi] if vert_end_arr else None
            c["S3b_vert_start"] = vs
            c["S3b_vert_end"] = ve
            c["S3b_n_verts"] = (ve - vs) if (vs is not None and ve is not None) else None
            if verts_init_pos is not None and vs is not None and ve is not None and ve > vs:
                local_verts = verts_init_pos[vs:ve]
                c["S3b_local_verts"] = local_verts
                c["S3b_local_z_min"] = min(v[2] for v in local_verts)
                c["S3b_local_z_max"] = max(v[2] for v in local_verts)
        # Plane mesh verts (PLANE is_convex=False, treated as polytope here).
        if plane_geom_idx is not None and vert_start_arr is not None and vert_end_arr is not None:
            pvs = vert_start_arr[plane_geom_idx]
            pve = vert_end_arr[plane_geom_idx]
            out["plane_n_verts"] = pve - pvs
            out["plane_vert_start"] = pvs
            out["plane_vert_end"] = pve
        # is_convex flag for plane and capsules.
        if hasattr(rs.geoms_info, "is_convex"):
            is_convex_arr = _try_arr(rs.geoms_info.is_convex)
            if is_convex_arr is not None:
                out["plane_is_convex"] = bool(is_convex_arr[plane_geom_idx]) if plane_geom_idx is not None else None
                for c in foot_caps:
                    c["S3b_is_convex"] = bool(is_convex_arr[c["geom_idx"]])
    except Exception as e:
        out["S3b_err"] = repr(e)
        out["S3b_tb"] = traceback.format_exc()

    # ── S4. runtime world pose at peak impact + S5. expected penetration ──
    # Step until peak impact then capture.
    n_act = env.num_actions
    zero = torch.zeros(env.num_envs, n_act, device=env.device)

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

        if peak_impact_step is not None and s == peak_impact_step and not snapshot_taken:
            try:
                for c in foot_caps:
                    g = c["_geom_obj"]
                    wp = g.get_pos(envs_idx=None)
                    wq = g.get_quat(envs_idx=None)
                    wp0 = _try_arr(wp[0]) if hasattr(wp, "__getitem__") else _try_arr(wp)
                    wq0 = _try_arr(wq[0]) if hasattr(wq, "__getitem__") else _try_arr(wq)
                    c["S4_world_pos_env0"] = wp0
                    c["S4_world_quat_env0"] = wq0
                    # S5. expected penetration from pure geometry.
                    data_field = c.get("S2_geoms_info", {}).get("data")
                    radius = data_field[0] if isinstance(data_field, list) and len(data_field) >= 1 else None
                    height = data_field[1] if isinstance(data_field, list) and len(data_field) >= 2 else None
                    if (
                        isinstance(wp0, list)
                        and len(wp0) == 3
                        and isinstance(wq0, list)
                        and len(wq0) == 4
                        and radius is not None
                        and height is not None
                    ):
                        lowest_z = _capsule_lowest_world_z(wp0, wq0, float(radius), float(height))
                        c["S5_lowest_world_z"] = lowest_z
                        c["S5_expected_penetration"] = max(0.0, -lowest_z)
                        c["S5_should_collide"] = lowest_z < 0.0
                    # S5b. lowest MESH VERTEX world z (what
                    # ``func_add_polytope_vertex_contacts_sdf`` actually
                    # uses to detect penetration with the plane SDF).
                    try:
                        import numpy as _np

                        local_verts = c.get("S3b_local_verts")
                        if (
                            isinstance(local_verts, list)
                            and local_verts
                            and isinstance(wp0, list)
                            and isinstance(wq0, list)
                        ):
                            q = _np.asarray(wq0, dtype=float)
                            w, x, y, z = q[0], q[1], q[2], q[3]
                            tx, ty, tz = float(wp0[0]), float(wp0[1]), float(wp0[2])
                            world_zs = []
                            for v in local_verts:
                                vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
                                rz = (
                                    2.0 * (x * z + w * y) * vx
                                    + 2.0 * (y * z - w * x) * vy
                                    + (1.0 - 2.0 * (x * x + y * y)) * vz
                                )
                                world_zs.append(tz + rz)
                            c["S5b_mesh_lowest_world_z"] = min(world_zs)
                            c["S5b_mesh_highest_world_z"] = max(world_zs)
                            c["S5b_mesh_should_collide_pen"] = max(0.0, -min(world_zs))
                    except Exception as ee:
                        c["S5b_err"] = repr(ee)
                # S6. Genesis collider contact buffer (env 0).
                try:
                    cs = rs.collider._collider_state
                    n_env0 = int(_try_arr(cs.n_contacts)[0])
                    ga_arr = _try_arr(cs.contact_data.geom_a)
                    gb_arr = _try_arr(cs.contact_data.geom_b)
                    pen_arr = _try_arr(cs.contact_data.penetration)
                    pos_arr = _try_arr(cs.contact_data.pos)
                    if isinstance(ga_arr, list) and ga_arr:
                        # Layout (after our parity-diag fix): [contact_slot][env]
                        # Extract env 0 column.
                        def _col_env0(lst, n_env0=n_env0):
                            if not lst:
                                return []
                            if isinstance(lst[0], list):
                                return [row[0] if row else None for row in lst[:n_env0]]
                            return lst[:n_env0]

                        ga_env0 = _col_env0(ga_arr)
                        gb_env0 = _col_env0(gb_arr)
                        pen_env0 = _col_env0(pen_arr)
                        pos_env0 = _col_env0(pos_arr)
                        contact_index: dict[int, list[dict]] = {}
                        for k in range(len(ga_env0)):
                            gi_a = ga_env0[k] if ga_env0 else None
                            gi_b = gb_env0[k] if gb_env0 else None
                            entry = {
                                "slot": k,
                                "geom_a": gi_a,
                                "geom_b": gi_b,
                                "penetration": pen_env0[k] if k < len(pen_env0) else None,
                                "pos": pos_env0[k] if k < len(pos_env0) else None,
                            }
                            for gid in (gi_a, gi_b):
                                if gid is None:
                                    continue
                                try:
                                    contact_index.setdefault(int(gid), []).append(entry)
                                except Exception:
                                    pass
                        for c in foot_caps:
                            c["S6_contacts_for_this_geom"] = contact_index.get(int(c["geom_idx"]), [])
                            c["S6_n_contacts"] = len(c["S6_contacts_for_this_geom"])
                        out["S6_n_contacts_env0"] = n_env0
                except Exception as ee:
                    out["S6_err"] = repr(ee)
                    out["S6_tb"] = traceback.format_exc()

                # S7. runtime collider knobs.
                ci = rs.collider._collider_info
                ro = env.scene_manager.config.rigid_options
                out["S7_collider_knobs"] = {
                    "mc_tolerance": _safe(lambda ci=ci: float(_try_arr(ci.mc_tolerance)[0]))
                    if _try_arr(ci.mc_tolerance) is not None
                    else _try_arr(ci.mc_tolerance),
                    "mc_perturbation": _safe(lambda ci=ci: float(_try_arr(ci.mc_perturbation)[0]))
                    if _try_arr(ci.mc_perturbation) is not None
                    else _try_arr(ci.mc_perturbation),
                    "contact_pruning_tolerance_cfg": getattr(ro, "contact_pruning_tolerance", None),
                    "contact_pruning_tolerance_runtime": _safe(
                        lambda ci=ci: float(_try_arr(ci.contact_pruning_tolerance)[0])
                    )
                    if hasattr(ci, "contact_pruning_tolerance")
                    else None,
                    "max_collision_pairs_cfg": getattr(ro, "max_collision_pairs", None),
                    "max_collision_pairs_runtime": _safe(lambda ci=ci: int(_try_arr(ci.max_collision_pairs)[0])),
                    "max_contact_pairs_runtime": _safe(lambda ci=ci: int(_try_arr(ci.max_contact_pairs)[0]))
                    if hasattr(ci, "max_contact_pairs")
                    else None,
                    "integrator_runtime": str(getattr(rs, "_integrator", None)),
                    "_sol_default_timeconst": float(getattr(rs, "_sol_default_timeconst", float("nan"))),
                }
            except Exception as e:
                out["snapshot_err"] = repr(e)
                out["snapshot_tb"] = traceback.format_exc()
            snapshot_taken = True

    # Errno sentinel: the patched ``_func_narrowphase_contact0`` PLANE branch
    # ORs ``0x42000000`` into errno[i_b] every time it runs. If the recorded
    # errno values are all 0 after the run, that branch was never entered
    # (despite our patched source being captured by the qd decorator).
    try:
        errno_arr = _try_arr(rs._errno)
        out["errno_per_env"] = errno_arr
        flat = []
        if isinstance(errno_arr, list):
            stack = [errno_arr]
            while stack:
                v = stack.pop()
                if isinstance(v, list):
                    stack.extend(v)
                else:
                    flat.append(v)
        out["errno_max"] = max(flat) if flat else None
        out["errno_has_sentinel_0x42"] = any((int(v) & 0x42000000) != 0 for v in flat) if flat else False
    except Exception as e:
        out["errno_read_err"] = repr(e)

    out["contact_first_hit_step"] = contact_first_hit_step
    out["peak_impact_step"] = peak_impact_step

    # Strip non-serialisable geom object handles before return.
    for c in foot_caps:
        c.pop("_geom_obj", None)
    out["foot_capsules"] = foot_caps

    return out


def _render(report: dict) -> str:
    L: list[str] = []

    def sec(title: str) -> None:
        L.append("")
        L.append("━" * 78)
        L.append(title)
        L.append("━" * 78)

    sec("Stage probe summary — Genesis foot capsule short-capsule rejection")
    L.append(f"  n_geoms_total: {report.get('n_geoms_total')}")
    L.append(f"  plane_geom_idx: {report.get('plane_geom_idx')}")
    L.append(f"  n_foot_capsules: {len(report.get('foot_capsules') or [])}")
    L.append(f"  contact_first_hit_step: {report.get('contact_first_hit_step')}")
    L.append(f"  peak_impact_step: {report.get('peak_impact_step')}")
    L.append(f"  S6_n_contacts_env0: {report.get('S6_n_contacts_env0')}")
    L.append(f"  S7_collider_knobs: {report.get('S7_collider_knobs')}")
    L.append(f"  Plane geoms_info: {report.get('plane_geoms_info')}")

    sec("Per-capsule stage probe")
    L.append(
        f"{'gidx':>5} {'link':<28} {'R':>6} {'H':>7} "
        f"{'S1_pair':>8} {'S5_pen(mm)':>11} {'S6_n_obs':>9}  S2_static + S3_aabb"
    )
    L.append("-" * 78)
    for c in report.get("foot_capsules") or []:
        gi = c.get("geom_idx")
        link = c.get("link_name", "?")
        s2 = c.get("S2_geoms_info") or {}
        data = s2.get("data") or []
        R = data[0] if len(data) > 0 else None
        H = data[1] if len(data) > 1 else None
        s1 = c.get("S1_pair_admitted")
        s5 = c.get("S5_expected_penetration")
        s6_n = c.get("S6_n_contacts", 0)
        L.append(f"{gi:>5} {link:<28} {R:>6.3f} {H:>7.4f} " f"{str(s1):>8} {((s5 or 0)*1000):>11.4f} {s6_n:>9}")
        L.append(
            f"        contype={s2.get('contype')}  conaffinity={s2.get('conaffinity')}  "
            f"link_idx={s2.get('link_idx')}  entity_idx={s2.get('entity_idx')}"
        )
        L.append(
            f"        local_aabb_min={c.get('S3_local_aabb_min')}  "
            f"local_aabb_max={c.get('S3_local_aabb_max')}  "
            f"diag={c.get('S3_local_aabb_diag')}"
        )
        L.append(f"        world_pos={c.get('S4_world_pos_env0')}  " f"world_quat={c.get('S4_world_quat_env0')}")
        s6 = c.get("S6_contacts_for_this_geom") or []
        if s6:
            for entry in s6[:4]:
                L.append(
                    f"        S6 contact slot={entry['slot']} pair=({entry['geom_a']},{entry['geom_b']}) "
                    f"pen={entry['penetration']} pos={entry['pos']}"
                )
        else:
            L.append("        S6 contact: NONE (no contact buffer entry references this geom)")

    sec("Interpretation cheatsheet")
    L.append(
        "  • S1_pair_admitted=False on any short capsule → compile-time filter "
        "(``_compute_collision_pair_idx``) rejected it. Inspect contype/conaffinity diffs in S2."
    )
    L.append(
        "  • S1_pair_admitted=True AND S5_expected_pen>0 AND S6_n=0 → broadphase or narrowphase "
        "rejected it at runtime despite real geometric penetration. Cause is downstream of S1."
    )
    L.append(
        "  • S5_expected_pen ≤ 0 → no actual penetration; capsule isn't touching the plane. "
        "Trajectory issue, not a collider bug."
    )
    L.append(
        "  • If S6_n>0 with pen<0 on a short capsule → narrowphase ran and got is_col=False from "
        "MPR ``support_driver``. Likely length-dependent support behaviour."
    )
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--num-envs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--settle-steps", type=int, default=50)
    ap.add_argument("--out-json", type=str, default="./short_capsule_debug.json")
    args = ap.parse_args()

    try:
        report = _genesis_debug(args.num_envs, args.seed, args.settle_steps)
    except Exception:
        traceback.print_exc()
        return 1

    print(_render(report))
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(report, indent=2, default=str))
    print(f"\n✓ wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
