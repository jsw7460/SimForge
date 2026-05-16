"""Cross-sim feet_slip trace — find why mujoco's first-rollout cost differs.

Builds g1_29dof on each sim with the same seed, resets, and runs N steps
with identical (per-step seeded) random actions. At every step dumps
*every* quantity that feeds into ``penalize_feet_slip``:

  formula:  cost = Σ_feet (‖foot_xy_vel‖² × is_contact)
            reward = -0.1 × cost × command_active

Per-step rows (per sim):
  - foot_frame world pos (xyz) — sanity (incl. foot height for contact)
  - foot_frame world lin_vel (xy) + |vel_xy|² per foot
  - ankle_roll world lin_vel (xy) + |vel_xy|² per foot   (cross-check)
  - is_contact per foot (bool)
  - |contact_force| per foot (raw force magnitude, for threshold debug)
  - command (vx, vy, ωz) + command_active mask
  - per-foot cost = vel_xy² × is_contact
  - feet_slip reward = -0.1 × Σ cost × command_active
  - base pos z + base quat (fall sanity)

One-time at start (so we KNOW the actual joint frictionloss used at
training time — not just the static CPU mj_model that ``check_action_
indexing_parity`` reads):
  - per-joint frictionloss CPU (compile-time, ``mj_model.dof_frictionloss``)
  - per-joint frictionloss RUNTIME (Genesis: ``entity.get_dofs_friction
    loss``; Newton: ``model.joint_friction``; mujoco: ``_wp_model.
    dof_frictionloss`` — the GPU tensor mjlab DR actually writes to)
  - per-joint armature (sanity)

Writes the trace to ``feet_slip_trace.txt`` in the current working
directory and prints a short summary to stdout.

Usage:
    python -m rlworld.scripts.diag.check_feet_slip_trace
    python -m rlworld.scripts.diag.check_feet_slip_trace --steps 30 --action-scale 0.4 --num-envs 8
"""

from __future__ import annotations

import argparse
import math
import os
import traceback
from pathlib import Path

# Multi-sim run in one process — bypass the single-backend guard.
os.environ.setdefault("JAXRLWORLD_ALLOW_MULTI_SIM", "1")

import numpy as np

_PRESET = ("rlworld.rl.configs.presets.g1_29dof.base", "G1FlatConfig")
_SIMS = ("genesis", "newton", "mujoco")
_FEET = ("left_foot_frame", "right_foot_frame")
_ANKLES = ("left_ankle_roll_link", "right_ankle_roll_link")
_COMMAND_THRESHOLD = 0.05  # matches g1_29dof feet_slip preset
_REWARD_WEIGHT = 0.1  # matches g1_29dof feet_slip RewardTermConfig weight


def _build_env(sim: str, num_envs: int):
    import importlib

    from rlworld.rl.runners import BaseRunner

    mod = importlib.import_module(_PRESET[0])
    cls = getattr(mod, _PRESET[1])
    cfgs = cls(sim_type=sim, num_envs=num_envs).build()
    runner = BaseRunner.create_with_env(cfgs)
    return runner.env


# ---------------------------------------------------------------------------
# per-sim body / friction readers
# ---------------------------------------------------------------------------


def _body_idx(env, sim: str, name: str) -> int | None:
    """Return the body index in each sim's RobotData indexing space.

    Critical: each sim's ``body_pos_w_by_ids`` / ``body_lin_vel_w_by_ids``
    indexes into its OWN body tensor (entity-local), not into the raw
    Genesis link list / Newton flat array / mj_model.body. Always resolve
    via the entity's own ``find_bodies``-equivalent to stay in the same
    index space the RobotData accessor expects.
    """
    if sim == "genesis":
        # Genesis: ``entity.links[i].idx_local`` matches the indexing used by
        # ``GenesisRobotData.body_*_by_ids``. ``links`` are already in entity-
        # local order, so the enumerate index works.
        for i, link in enumerate(env.scene_manager["robot"].links):
            if link.name == name:
                return i
    elif sim == "newton":
        # Newton: ``body_label`` ordered slice for world-0 is the entity-
        # local body list ``NewtonRobotData.body_*_by_ids`` indexes into.
        labels = list(getattr(env.scene_manager.solver.model, "body_label", []))
        num_worlds = env.scene_manager.solver.model.world_count
        per_world = len(labels) // max(num_worlds, 1)
        for i in range(per_world):
            bare = labels[i].rsplit("/", 1)[-1] if "/" in labels[i] else labels[i]
            if bare == name:
                return i
    elif sim == "mujoco":
        # mjlab: ``entity.data.body_link_pos_w`` is shape
        # (num_envs, n_entity_bodies, 3). The entity's ``find_bodies`` gives
        # the entity-local id, NOT the global ``mj_model`` body id.
        try:
            entity = env.scene_manager._scene["robot"]
            body_ids, _ = entity.find_bodies([name], preserve_order=True)
            return int(body_ids[0]) if body_ids else None
        except Exception:
            return None
    return None


def _mujoco_global_body_idx(env, name: str) -> int | None:
    """Global mj_model body id for ``name`` — for cross-check only."""
    try:
        import mujoco

        mj_model = env.scene_manager.mj_model
        for i in range(mj_model.nbody):
            raw = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, i) or ""
            bare = raw.rsplit("/", 1)[-1] if "/" in raw else raw
            if bare == name:
                return i
    except Exception:
        pass
    return None


def _format_body_id_block(envs: dict) -> str:
    """Print the body IDs each sim resolves for the feet — confirms whether
    the latest mujoco fix is loaded (entity-local) vs the previous bug
    (global mj_model id). Also dumps a few alternative pos reads for mujoco
    so we can tell whether mjlab is returning the wrong body or the wrong
    position field of the right body."""
    out = []
    out.append("\n" + "=" * 110)
    out.append("Body ID resolution sanity (verifies _body_idx is up-to-date and reading the intended body)")
    out.append("=" * 110)
    out.append(f"  {'body name':25s} | " + " | ".join(f"{s:^22s}" for s in envs.keys()))
    out.append("  " + "-" * 110)
    for name in (*_FEET, *_ANKLES):
        cells = []
        for sim, env in envs.items():
            try:
                rd_idx = _body_idx(env, sim, name)
                if sim == "mujoco":
                    glob_idx = _mujoco_global_body_idx(env, name)
                    cells.append(f"rd={rd_idx} glob={glob_idx}")
                else:
                    cells.append(f"rd={rd_idx}")
            except Exception as e:
                cells.append(f"<err: {type(e).__name__}>")
        out.append(f"  {name:25s} | " + " | ".join(f"{c:^22s}" for c in cells))

    # For mujoco: dump multiple position fields for left_foot_frame so we can
    # see whether mjlab is reading the wrong body (rd_idx wrong) or the
    # right body but the wrong field.
    out.append("")
    out.append("  mujoco extra: alternative pos reads for left_foot_frame (at reset, env 0)")
    if "mujoco" in envs:
        env = envs["mujoco"]
        try:
            mj_model = env.scene_manager.mj_model
            entity = env.scene_manager._scene["robot"]
            rd = env.get_robot_data()
            rd_idx = _body_idx(env, "mujoco", "left_foot_frame")
            glob_idx = _mujoco_global_body_idx(env, "left_foot_frame")
            import torch

            # rd-indexed read (the value the diag uses)
            try:
                p = rd.body_pos_w_by_ids(torch.tensor([rd_idx], device=env.device))[0, 0].detach().cpu().numpy()
                out.append(f"    RobotData.body_pos_w_by_ids(rd_idx={rd_idx})           = {p.tolist()}")
            except Exception as e:
                out.append(f"    RobotData.body_pos_w_by_ids(rd_idx={rd_idx})           = <err: {e}>")
            # entity.data direct read at rd_idx
            try:
                p = entity.data.body_link_pos_w[0, rd_idx].detach().cpu().numpy()
                out.append(f"    entity.data.body_link_pos_w[0, rd_idx={rd_idx}]        = {p.tolist()}")
            except Exception as e:
                out.append(f"    entity.data.body_link_pos_w[0, rd_idx={rd_idx}]        = <err: {e}>")
            # entity.data at global_idx (cross-check)
            if glob_idx is not None and glob_idx != rd_idx:
                try:
                    p = entity.data.body_link_pos_w[0, glob_idx].detach().cpu().numpy()
                    out.append(f"    entity.data.body_link_pos_w[0, glob_idx={glob_idx}] = {p.tolist()}")
                except Exception as e:
                    out.append(f"    entity.data.body_link_pos_w[0, glob_idx={glob_idx}] = <err: {e}>")
            # entity.data.body_link_pos_w shape
            try:
                shape = tuple(entity.data.body_link_pos_w.shape)
                out.append(f"    entity.data.body_link_pos_w shape                     = {shape}")
            except Exception:
                pass
            # mj_model bodies count (excl. world?)
            try:
                out.append(f"    mj_model.nbody                                        = {mj_model.nbody}")
            except Exception:
                pass
        except Exception as e:
            out.append(f"  <mujoco extra block error: {type(e).__name__}: {e}>")
    return "\n".join(out)


def _read_joint_frictionloss_runtime(env, sim: str) -> dict[str, list[float]]:
    """Read per-canonical-joint frictionloss at runtime — the value the
    physics step actually sees. Returns dict with keys depending on sim."""

    out: dict[str, list[float]] = {"names": list(env.act_manager.actuated_joint_names)}
    indexing = env.act_manager._indexing

    if sim == "genesis":
        entity = env.scene_manager["robot"]
        try:
            fri = entity.get_dofs_frictionloss(dofs_idx_local=indexing.sim_indices)
            arr = fri[0] if fri.ndim == 2 else fri
            out["runtime"] = [float(arr[i]) for i in range(arr.shape[0])]
        except Exception as e:
            out["runtime_error"] = f"{type(e).__name__}: {e}"

    elif sim == "newton":
        import warp as wp

        model = env.scene_manager.solver.model
        try:
            qd_idx = indexing.newton_qd_indices.detach().cpu().numpy()
            fri = wp.to_torch(model.joint_friction).detach().cpu().numpy()
            out["runtime"] = [float(fri[int(q)]) for q in qd_idx]
        except Exception as e:
            out["runtime_error"] = f"{type(e).__name__}: {e}"

    elif sim == "mujoco":
        import mujoco

        mj_model = env.scene_manager.mj_model
        entity = env.scene_manager._scene["robot"]
        entity_joint_names = list(entity.joint_names)
        sim_indices = indexing.sim_indices.detach().cpu().tolist()
        # Build canonical → dof_adr map (entity joint idx → mj joint id → dof_adr).
        dof_adrs: list[int | None] = []
        for ci in range(len(out["names"])):
            entity_jidx = int(sim_indices[ci])
            bare = entity_joint_names[entity_jidx]
            mj_jid = None
            for jid in range(mj_model.njnt):
                raw = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, jid) or ""
                if raw.rsplit("/", 1)[-1] == bare:
                    mj_jid = jid
                    break
            dof_adrs.append(int(mj_model.jnt_dofadr[mj_jid]) if mj_jid is not None else None)
        out["dof_adrs"] = dof_adrs

        # Static compile-time mj_model.dof_frictionloss (CPU, shape (n_dof,)).
        try:
            cpu = mj_model.dof_frictionloss
            out["static_cpu"] = [float(cpu[a]) if a is not None and a < len(cpu) else None for a in dof_adrs]
        except Exception as e:
            out["static_cpu_error"] = f"{type(e).__name__}: {e}"

        # Runtime mjwarp GPU model — mjlab DR writes here.
        for path in (
            ("env.scene_manager._sim.model.dof_frictionloss", lambda: env.scene_manager._sim.model.dof_frictionloss),
            (
                "env.scene_manager._sim._wp_model.dof_frictionloss",
                lambda: env.scene_manager._sim._wp_model.dof_frictionloss,
            ),
        ):
            label, getter = path
            try:
                wp_friction = getter()
            except Exception:
                continue
            try:
                import warp as wp

                arr = wp.to_torch(wp_friction).detach().cpu().numpy()
                arr0 = arr[0] if arr.ndim == 2 else arr
                out["runtime"] = [float(arr0[a]) if a is not None and a < len(arr0) else None for a in dof_adrs]
                out["runtime_source"] = label
                break
            except Exception as e:
                out["runtime_error"] = f"{label}: {type(e).__name__}: {e}"
        if "runtime" not in out:
            out["runtime"] = list(out.get("static_cpu", []))
            out["runtime_source"] = "<falling back to static CPU>"

    return out


def _read_foot_geom_friction(env, sim: str) -> dict:
    """Read the foot collision-geom sliding friction each sim actually
    sees at runtime.

    Returns a dict with sim-specific keys; common keys:
      ``per_link``: {ankle_roll_link_name: [slide_friction_per_geom, ...]}
      ``count``: total number of foot collision geoms found
    """
    out: dict = {"per_link": {}, "count": 0}
    if sim == "genesis":
        # Genesis loses geom names during MJCF import — group foot collision
        # geoms by their parent link (left/right_ankle_roll_link) instead.
        try:
            entity = env.scene_manager["robot"]
            for link in entity.links:
                if link.name in _ANKLES:
                    vals = []
                    for g in link.geoms:
                        try:
                            vals.append(float(g.friction))
                        except Exception:
                            pass
                    if vals:
                        out["per_link"][link.name] = vals
                        out["count"] += len(vals)
        except Exception as e:
            out["error"] = f"{type(e).__name__}: {e}"

    elif sim == "newton":
        # Newton: shape_material_mu indexed by global shape id; group by the
        # ankle_roll body so we can compare to the other sims by link.
        try:
            import warp as wp

            model = env.scene_manager.solver.model
            num_worlds = model.world_count
            body_labels = list(getattr(model, "body_label", []))
            n_bodies = len(body_labels)
            bodies_per_world = n_bodies // max(num_worlds, 1)
            # shape_body[i] = body id that owns shape i (per-world replication
            # means shape ids run over all worlds; restrict to world 0).
            shape_body = (
                wp.to_torch(model.shape_body).detach().cpu().numpy()
                if hasattr(model.shape_body, "numpy")
                else np.asarray(model.shape_body)
            )
            mu = (
                wp.to_torch(model.shape_material_mu).detach().cpu().numpy()
                if hasattr(model.shape_material_mu, "numpy")
                else np.asarray(model.shape_material_mu)
            )
            # Restrict to world 0: shapes whose body is in [0, bodies_per_world).
            for s in range(len(shape_body)):
                b = int(shape_body[s])
                if 0 <= b < bodies_per_world:
                    bare = body_labels[b].rsplit("/", 1)[-1] if "/" in body_labels[b] else body_labels[b]
                    if bare in _ANKLES:
                        out["per_link"].setdefault(bare, []).append(float(mu[s]))
                        out["count"] += 1
        except Exception as e:
            out["error"] = f"{type(e).__name__}: {e}"

    elif sim == "mujoco":
        # MuJoCo: read mj_model.geom_friction by name for each
        # left/right_foot[1-7]_collision geom.
        try:
            import mujoco

            mj_model = env.scene_manager.mj_model
            for side in ("left", "right"):
                key = f"{side}_ankle_roll_link"
                vals = []
                for k in range(1, 8):
                    target = f"{side}_foot{k}_collision"
                    for i in range(mj_model.ngeom):
                        raw = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
                        bare = raw.rsplit("/", 1)[-1] if "/" in raw else raw
                        if bare == target:
                            vals.append(float(mj_model.geom_friction[i, 0]))
                            break
                if vals:
                    out["per_link"][key] = vals
                    out["count"] += len(vals)
        except Exception as e:
            out["error"] = f"{type(e).__name__}: {e}"
    return out


def _format_foot_friction_block(sims, foot_fri_data):
    """Print foot collision-geom sliding friction per sim."""
    out = []
    out.append("\n" + "=" * 110)
    out.append("Foot collision-geom sliding friction (slide axis) — runtime value each sim sees")
    out.append("=" * 110)
    out.append(f"  {'link':30s} | " + " | ".join(f"{s:^28s}" for s in sims))
    out.append("  " + "-" * 110)
    all_links = sorted({k for s in sims for k in foot_fri_data.get(s, {}).get("per_link", {})})
    for link in all_links:
        cells = []
        for s in sims:
            d = foot_fri_data.get(s, {})
            if "error" in d:
                cells.append(f"<err: {d['error']}>"[:28])
                continue
            vals = d.get("per_link", {}).get(link, [])
            if not vals:
                cells.append("<missing>".center(28))
            else:
                mn, mx = min(vals), max(vals)
                if abs(mx - mn) < 1e-6:
                    cells.append(f"{vals[0]:.4f}  (×{len(vals)})".center(28))
                else:
                    cells.append(f"min={mn:.4f} max={mx:.4f} (×{len(vals)})"[:28])
        out.append(f"  {link:30s} | " + " | ".join(c if len(c) >= 28 else c.center(28) for c in cells))
    out.append("  " + "-" * 110)
    counts = " | ".join(f"count={foot_fri_data.get(s, {}).get('count', 0):<24d}" for s in sims)
    out.append(f"  {'(geoms found per sim)':30s} | " + counts)
    out.append("")
    out.append("  Expected after the g1.xml fix: 0.6 across all sims on every ankle_roll_link geom.")
    return "\n".join(out)


def _read_joint_armature_runtime(env, sim: str) -> list[float]:
    indexing = env.act_manager._indexing
    if sim == "genesis":
        try:
            entity = env.scene_manager["robot"]
            arm = entity.get_dofs_armature(dofs_idx_local=indexing.sim_indices)
            arr = arm[0] if arm.ndim == 2 else arm
            return [float(arr[i]) for i in range(arr.shape[0])]
        except Exception:
            return []
    elif sim == "newton":
        try:
            import warp as wp

            qd_idx = indexing.newton_qd_indices.detach().cpu().numpy()
            arr = wp.to_torch(env.scene_manager.solver.model.joint_armature).detach().cpu().numpy()
            return [float(arr[int(q)]) for q in qd_idx]
        except Exception:
            return []
    elif sim == "mujoco":
        try:
            import mujoco

            mj_model = env.scene_manager.mj_model
            entity = env.scene_manager._scene["robot"]
            entity_joint_names = list(entity.joint_names)
            sim_indices = indexing.sim_indices.detach().cpu().tolist()
            names = list(env.act_manager.actuated_joint_names)
            out: list[float] = []
            for ci in range(len(names)):
                entity_jidx = int(sim_indices[ci])
                bare = entity_joint_names[entity_jidx]
                mj_jid = None
                for jid in range(mj_model.njnt):
                    raw = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, jid) or ""
                    if raw.rsplit("/", 1)[-1] == bare:
                        mj_jid = jid
                        break
                if mj_jid is not None:
                    out.append(float(mj_model.dof_armature[int(mj_model.jnt_dofadr[mj_jid])]))
                else:
                    out.append(float("nan"))
            return out
        except Exception:
            return []
    return []


# ---------------------------------------------------------------------------
# per-step capture
# ---------------------------------------------------------------------------


def _capture_step(env, sim: str, action) -> dict:
    """Step env once and capture every quantity that feeds feet_slip."""
    import torch

    env.step(action)
    rd = env.get_robot_data()
    info: dict = {}

    # foot frame / ankle world pos + vel (env 0)
    foot_body_ids = []
    for foot in _FEET:
        i = _body_idx(env, sim, foot)
        foot_body_ids.append(i)
    ankle_body_ids = []
    for ankle in _ANKLES:
        i = _body_idx(env, sim, ankle)
        ankle_body_ids.append(i)

    foot_ids_t = torch.tensor([i for i in foot_body_ids if i is not None], device=env.device, dtype=torch.long)
    ankle_ids_t = torch.tensor([i for i in ankle_body_ids if i is not None], device=env.device, dtype=torch.long)

    try:
        foot_pos = rd.body_pos_w_by_ids(foot_ids_t)[0].detach().cpu().numpy()
        foot_vel = rd.body_lin_vel_w_by_ids(foot_ids_t)[0].detach().cpu().numpy()
        info["foot_pos"] = {_FEET[i]: foot_pos[i].tolist() for i in range(len(_FEET))}
        info["foot_vel"] = {_FEET[i]: foot_vel[i].tolist() for i in range(len(_FEET))}
    except Exception as e:
        info["foot_pos_vel_error"] = f"{type(e).__name__}: {e}"

    try:
        ankle_vel = rd.body_lin_vel_w_by_ids(ankle_ids_t)[0].detach().cpu().numpy()
        info["ankle_vel"] = {_ANKLES[i]: ankle_vel[i].tolist() for i in range(len(_ANKLES))}
    except Exception as e:
        info["ankle_vel_error"] = f"{type(e).__name__}: {e}"

    # is_contact + contact force, ordered to match _FEET (via foot body names → contact_order)
    contact_order = list(_ANKLES)  # matches what g1_29dof preset's feet_contact_order is
    try:
        ic = env.contact_manager.is_contact("feet_ground_contact", order=contact_order)[0].detach().cpu()
        info["is_contact"] = [bool(ic[i]) for i in range(ic.shape[0])]
        info["contact_order"] = contact_order
    except Exception as e:
        info["is_contact_error"] = f"{type(e).__name__}: {e}"
        info["is_contact"] = [False] * len(_FEET)
    try:
        cf = env.contact_manager.contact_force("feet_ground_contact", order=contact_order)[0].detach().cpu().numpy()
        info["contact_force_mag"] = [float(np.linalg.norm(cf[i])) for i in range(cf.shape[0])]
    except Exception as e:
        info["contact_force_error"] = f"{type(e).__name__}: {e}"
        info["contact_force_mag"] = [0.0] * len(_FEET)

    # command
    try:
        cmd_x = float(env.command_manager.lin_vel_x[0])
        cmd_y = float(env.command_manager.lin_vel_y[0])
        cmd_w = float(env.command_manager.ang_vel[0])
        info["command"] = (cmd_x, cmd_y, cmd_w)
        lin_norm = math.sqrt(cmd_x * cmd_x + cmd_y * cmd_y)
        ang_norm = abs(cmd_w)
        info["command_active"] = (lin_norm + ang_norm) > _COMMAND_THRESHOLD
    except Exception as e:
        info["command_error"] = f"{type(e).__name__}: {e}"
        info["command"] = (0.0, 0.0, 0.0)
        info["command_active"] = False

    # base pos / orientation
    try:
        info["base_pos"] = (
            rd.body_pos_w_by_ids(torch.tensor([0], device=env.device, dtype=torch.long))[0, 0]
            .detach()
            .cpu()
            .numpy()
            .tolist()
        )
    except Exception:
        info["base_pos"] = None

    # computed cost
    vel_xy_sq: list[float] = []
    for foot in _FEET:
        v = info.get("foot_vel", {}).get(foot)
        if v:
            vel_xy_sq.append(float(v[0]) ** 2 + float(v[1]) ** 2)
        else:
            vel_xy_sq.append(0.0)
    info["foot_vel_xy_sq"] = vel_xy_sq
    is_c = info.get("is_contact", [False] * len(_FEET))
    cost_per_foot = [vel_xy_sq[i] * (1.0 if is_c[i] else 0.0) for i in range(len(_FEET))]
    info["cost_per_foot"] = cost_per_foot
    info["total_cost"] = sum(cost_per_foot)
    info["feet_slip_reward"] = -_REWARD_WEIGHT * info["total_cost"] * (1.0 if info["command_active"] else 0.0)

    return info


# ---------------------------------------------------------------------------
# formatting
# ---------------------------------------------------------------------------


def _fmt_v(v, w=8):
    if v is None:
        return f"{'--':>{w}}"
    if isinstance(v, bool):
        return f"{'T' if v else 'F':>{w}}"
    if isinstance(v, int | float):
        absv = abs(float(v))
        if not math.isfinite(absv):
            return f"{'nan':>{w}}"
        if absv >= 100:
            return f"{v:{w}.2f}"
        if absv >= 1:
            return f"{v:+{w}.4f}"
        return f"{v:+{w}.5f}"
    return str(v)[:w]


def _row(label, per_sim_values, w=10):
    cells = " | ".join(_fmt_v(v, w=w) for v in per_sim_values)
    return f"  {label:32s} | {cells}"


def _format_friction_block(sims, fri_data):
    out = []
    out.append("\n" + "=" * 110)
    out.append("Joint frictionloss — STATIC (compile-time CPU) vs RUNTIME (what the physics step sees)")
    out.append("=" * 110)
    names = fri_data[sims[0]].get("names", [])
    out.append("")
    out.append(f"  {'joint':35s} | " + " | ".join(f"{s:^28s}" for s in sims))
    out.append("  " + "-" * (35 + 3 + 28 * len(sims) + 3 * (len(sims) - 1)))
    for ji, jn in enumerate(names):
        cells = []
        for s in sims:
            d = fri_data[s]
            rt = d.get("runtime", [])
            sc = d.get("static_cpu", [])
            rt_v = rt[ji] if ji < len(rt) else None
            sc_v = sc[ji] if ji < len(sc) else None
            if s == "mujoco":
                cells.append(f"rt={_fmt_v(rt_v, 8)} cpu={_fmt_v(sc_v, 8)}")
            else:
                cells.append(f"rt={_fmt_v(rt_v, 8)}")
        out.append(f"  {jn:35s} | " + " | ".join(f"{c:28s}" for c in cells))
    # mean / std
    out.append("  " + "-" * (35 + 3 + 28 * len(sims) + 3 * (len(sims) - 1)))
    means = {}
    for s in sims:
        rt = fri_data[s].get("runtime", [])
        rt_vals = [v for v in rt if isinstance(v, int | float) and math.isfinite(v)]
        means[s] = (np.mean(rt_vals) if rt_vals else float("nan"), np.std(rt_vals) if rt_vals else float("nan"))
    out.append(
        "  "
        + f"{'mean ± std (runtime)':35s} | "
        + " | ".join(f"{means[s][0]:.5f} ± {means[s][1]:.5f}".rjust(28) for s in sims)
    )
    src_line_parts = []
    for s in sims:
        src = fri_data[s].get("runtime_source")
        if src:
            src_line_parts.append(f"{s}: {src}")
    if src_line_parts:
        out.append("  (mujoco runtime tensor source: " + "; ".join(src_line_parts) + ")")
    return "\n".join(out)


def _format_step(step_idx, sims, per_sim):
    out = []
    out.append("")
    out.append("─" * 110)
    out.append(f"STEP {step_idx}")
    out.append("─" * 110)
    out.append(f"  {'metric':32s} | " + " | ".join(f"{s:^10s}" for s in sims))
    out.append("  " + "-" * (32 + 3 + 10 * len(sims) + 3 * (len(sims) - 1)))

    def get(s, *path, default=None):
        cur = per_sim[s]
        for p in path:
            if cur is None:
                return default
            if isinstance(p, str):
                cur = cur.get(p, default) if hasattr(cur, "get") else default
            else:
                try:
                    cur = cur[p]
                except (IndexError, TypeError):
                    return default
        return cur

    out.append(_row("base_pos_z", [get(s, "base_pos", 2) for s in sims]))
    for fi, foot in enumerate(_FEET):
        out.append(_row(f"{foot} pos.z", [get(s, "foot_pos", foot, 2) for s in sims]))
        out.append(_row(f"{foot} vel.x", [get(s, "foot_vel", foot, 0) for s in sims]))
        out.append(_row(f"{foot} vel.y", [get(s, "foot_vel", foot, 1) for s in sims]))
        out.append(_row(f"{foot} vel_xy²", [get(s, "foot_vel_xy_sq", fi) for s in sims]))
    for ai, ankle in enumerate(_ANKLES):
        out.append(_row(f"{ankle} vel.x", [get(s, "ankle_vel", ankle, 0) for s in sims]))
        out.append(_row(f"{ankle} vel.y", [get(s, "ankle_vel", ankle, 1) for s in sims]))
    out.append("  " + "-" * (32 + 3 + 10 * len(sims) + 3 * (len(sims) - 1)))
    for fi, foot in enumerate(_FEET):
        out.append(_row(f"is_contact[{foot}]", [get(s, "is_contact", fi) for s in sims]))
        out.append(_row(f"|F_contact|[{foot}]", [get(s, "contact_force_mag", fi) for s in sims]))
    out.append("  " + "-" * (32 + 3 + 10 * len(sims) + 3 * (len(sims) - 1)))
    out.append(_row("command.vx", [get(s, "command", 0) for s in sims]))
    out.append(_row("command.vy", [get(s, "command", 1) for s in sims]))
    out.append(_row("command.ωz", [get(s, "command", 2) for s in sims]))
    out.append(_row("command_active", [get(s, "command_active") for s in sims]))
    out.append("  " + "-" * (32 + 3 + 10 * len(sims) + 3 * (len(sims) - 1)))
    for fi, foot in enumerate(_FEET):
        out.append(_row(f"cost[{foot}]", [get(s, "cost_per_foot", fi) for s in sims]))
    out.append(_row("total_cost", [get(s, "total_cost") for s in sims]))
    out.append(_row("feet_slip_reward", [get(s, "feet_slip_reward") for s in sims]))
    return "\n".join(out)


def _format_summary(sims, all_steps):
    """All_steps: list of dicts {sim: capture}. Compute aggregates per sim."""
    out = []
    out.append("")
    out.append("=" * 110)
    out.append("AGGREGATE SUMMARY (mean across all captured steps)")
    out.append("=" * 110)
    metrics = [
        ("foot_vel_xy_sq[0] (left)", lambda d: d.get("foot_vel_xy_sq", [0, 0])[0]),
        ("foot_vel_xy_sq[1] (right)", lambda d: d.get("foot_vel_xy_sq", [0, 0])[1]),
        ("is_contact mean (left)", lambda d: float(d.get("is_contact", [False, False])[0])),
        ("is_contact mean (right)", lambda d: float(d.get("is_contact", [False, False])[1])),
        ("|F_contact| left", lambda d: d.get("contact_force_mag", [0, 0])[0]),
        ("|F_contact| right", lambda d: d.get("contact_force_mag", [0, 0])[1]),
        ("cost left", lambda d: d.get("cost_per_foot", [0, 0])[0]),
        ("cost right", lambda d: d.get("cost_per_foot", [0, 0])[1]),
        ("total_cost", lambda d: d.get("total_cost", 0.0)),
        ("feet_slip_reward", lambda d: d.get("feet_slip_reward", 0.0)),
        ("command_active mean", lambda d: float(d.get("command_active", False))),
        ("base_pos_z", lambda d: (d.get("base_pos") or [0, 0, 0])[2]),
    ]
    out.append(f"  {'metric':32s} | " + " | ".join(f"{s:^12s}" for s in sims) + " | cross-sim Δ%")
    out.append("  " + "-" * (32 + 3 + 12 * len(sims) + 3 * (len(sims) - 1) + 15))
    for name, fn in metrics:
        means = {}
        for s in sims:
            vals = []
            for step in all_steps:
                d = step.get(s)
                if d:
                    try:
                        vals.append(float(fn(d)))
                    except Exception:
                        pass
            means[s] = float(np.mean(vals)) if vals else float("nan")
        cells = " | ".join(_fmt_v(means[s], 12) for s in sims)
        finite = [v for v in means.values() if math.isfinite(v) and abs(v) > 1e-9]
        if len(finite) >= 2:
            delta_pct = (max(finite) - min(finite)) / max(abs(v) for v in finite) * 100.0
            d_str = f"{delta_pct:.1f}%"
        else:
            d_str = "--"
        out.append(f"  {name:32s} | {cells} | {d_str:>12s}")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    import torch

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num-envs", type=int, default=1)
    ap.add_argument("--settle", type=int, default=0, help="Zero-action settle steps before capture.")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--action-scale", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="feet_slip_trace.txt")
    args = ap.parse_args()

    out_path = Path(args.out).resolve()
    print(f"Writing trace to: {out_path}")

    sections: list[str] = []
    sections.append("=" * 110)
    sections.append(
        f"feet_slip cross-sim trace — preset=g1_29dof, num_envs={args.num_envs}, "
        f"settle={args.settle}, steps={args.steps}, action_scale={args.action_scale}, seed={args.seed}"
    )
    sections.append("=" * 110)

    # Build each sim, capture friction at start, then run identical action sequence.
    envs: dict[str, object] = {}
    friction_data: dict[str, dict] = {}
    action_seq: list[torch.Tensor] = []

    for sim in _SIMS:
        print(f"\n=== building [{sim}] ===")
        try:
            env = _build_env(sim, args.num_envs)
            envs[sim] = env
        except Exception as e:
            print(f"  BUILD ERROR: {type(e).__name__}: {e}")
            sections.append(f"\n[{sim}] BUILD ERROR: {type(e).__name__}: {e}")
            sections.append(traceback.format_exc())
            continue
        # Reset deterministically.
        torch.manual_seed(args.seed)
        env.reset()
        for _ in range(args.settle):
            env.step(torch.zeros(env.num_envs, env.num_actions, device=env.device))
        # Sample friction snapshot (post-reset, post-DR).
        friction_data[sim] = _read_joint_frictionloss_runtime(env, sim)
        friction_data[sim]["armature"] = _read_joint_armature_runtime(env, sim)
        friction_data[sim]["foot_geom"] = _read_foot_geom_friction(env, sim)
        # Make sure action_seq has args.steps entries — use first sim's seeding.
        # Force CPU generation here (Genesis/Newton set torch's default device
        # to CUDA, but the CPU Generator below would mismatch — pass an
        # explicit ``device="cpu"`` to override the default-device context).
        if not action_seq:
            g = torch.Generator(device="cpu").manual_seed(args.seed + 1000)
            for _ in range(args.steps):
                a = (
                    torch.rand(env.num_envs, env.num_actions, generator=g, device="cpu") * 2.0 - 1.0
                ) * args.action_scale
                action_seq.append(a.to(env.device))

    # Body ID resolution sanity FIRST — confirms the latest fix is loaded.
    sections.append(_format_body_id_block(envs))

    # Foot collision-geom friction — verifies the g1.xml fix landed in each sim.
    foot_fri_data = {s: friction_data.get(s, {}).get("foot_geom", {}) for s in _SIMS}
    sections.append(_format_foot_friction_block(list(_SIMS), foot_fri_data))

    # Cross-sim friction block.
    if all(s in friction_data for s in _SIMS):
        sections.append(_format_friction_block(list(_SIMS), friction_data))
    else:
        sections.append("\n(skipping friction block — not all sims built)")

    # Per-step capture.
    all_steps: list[dict] = []
    for step_idx in range(args.steps):
        per_sim: dict[str, dict] = {}
        for sim in _SIMS:
            if sim not in envs:
                continue
            try:
                # Move action to env's device per sim.
                act = action_seq[step_idx].to(envs[sim].device)
                per_sim[sim] = _capture_step(envs[sim], sim, act)
            except Exception as e:
                per_sim[sim] = {"step_error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()}
        all_steps.append(per_sim)
        sections.append(_format_step(step_idx, [s for s in _SIMS if s in envs], per_sim))

    # Final summary.
    sections.append(_format_summary([s for s in _SIMS if s in envs], all_steps))

    out_path.write_text("\n".join(sections), encoding="utf-8")

    # Short stdout summary.
    print(f"\n{'=' * 60}\nDone. Output: {out_path}")
    if all_steps:
        last = all_steps[-1]
        print("  Last-step feet_slip_reward per sim:")
        for s in _SIMS:
            if s in last and "feet_slip_reward" in last[s]:
                print(f"    {s}: {last[s]['feet_slip_reward']:+.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
