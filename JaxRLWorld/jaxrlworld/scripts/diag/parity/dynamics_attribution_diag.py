"""Attribute a cross-sim reward gap: inputs, plant parameters, or engine.

``check_reward_parity`` reports per-term gaps after zero-action steps --
go2_gait ``dof_vel`` 9% lower on Genesis at step 0, K1 ``soft_landing``
twice as large on Genesis. A gap there has three possible sources, and
only the last one is an engine difference:

  A. The inputs differ. ``check_reward_parity`` seeds torch once and leaves
     domain randomization and the random reset on, so the backends start
     from the same state and the same parameters only if every backend's
     DR and reset terms consume the RNG stream identically.
  B. The plant differs. Same state and same action, but a value the engine
     was handed differs (mass, armature, frictionloss, PD gains, the
     integrator) or the actuator path produces a different torque.
  C. Contact differs: the reading (which substep, which quantity) or the
     solve itself.

Stages, one subprocess per preset x stage x backend (single-sim invariant):

  A  The ``check_reward_parity`` setup unchanged: full preset, DR and random
     reset on, ``torch.manual_seed(seed)`` right before ``env.reset()``.
     Dumps per env, read from the live engine: initial root and joint state,
     commands, body masses, collision friction, per-DOF frictionloss /
     damping / armature, PD stiffness / damping / effort limit. Then one
     zero-action step with a per-substep joint trace and the raw reward
     terms.
  B  DR, pushes and terminations removed, the same state written on every
     backend: robot 1 m in the air, fixed joint offsets from home, three
     zero-action control steps traced per substep. Nothing touches the
     ground, so a joint-velocity or torque gap is the actuator path, a DOF
     parameter, the mass matrix or the integrator.
  C  The same stripped env dropped from the reset height and from +0.10 m,
     100 zero-action control steps. Per step: root height and vertical
     velocity, sum of squared joint velocities, every contact group's flags,
     forces, first-contact and substep force history, and every raw reward
     term. Checks: which history slot the reported force is (the reading
     convention), the contact impulse of the touchdown step, and the
     settled ground force against M*g (the force scale).

MuJoCo runs twice in B and C, with ``forward_after_physics`` off (the
training default) and on, since that switch moves both the kinematic and
the contact reading.

Usage (from the SimForge root):

    jaxpy -m jaxrlworld.scripts.diag.parity.dynamics_attribution_diag
    jaxpy -m jaxrlworld.scripts.diag.parity.dynamics_attribution_diag --presets k1_velocity
    python -m jaxrlworld.scripts.diag.parity.dynamics_attribution_diag --compare-only

Dumps land in ``--out`` (default ``runs/dynamics_attribution``), one JSON
per cell; ``--compare-only`` re-prints the report from them.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib
import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from jaxrlworld.rl.configs.base_config import iter_terms
from jaxrlworld.rl.configs.events import EventTermConfig
from jaxrlworld.rl.configs.terminations.termination_term_config import TerminationTermConfig
from jaxrlworld.rl.envs.mdp.events import common as ev
from jaxrlworld.rl.runners.base_runner import BaseRunner
from jaxrlworld.scripts.diag.parity.check_reward_parity import _capture_step

_MODULE = "jaxrlworld.scripts.diag.parity.dynamics_attribution_diag"

# Per preset: the config class (per backend where the preset is split by
# backend), config fields to override before ``build()`` for the stripped
# stages, and commands to hold during the stripped rollouts. K1's
# soft_landing is gated on a moving command, so it gets one; its action
# delay is a random draw per reset, so it is pinned to zero.
PRESETS: dict[str, dict] = {
    "go2_gait": {
        "cls": {
            "genesis": "jaxrlworld.rl.configs.presets.go2.genesis.gait_conditioned:Go2GaitConditionedGenesisConfig",
            "newton": "jaxrlworld.rl.configs.presets.go2.newton.gait_conditioned:Go2GaitConditionedNewtonConfig",
            "mujoco": "jaxrlworld.rl.configs.presets.go2.mujoco.gait_conditioned:Go2GaitConditionedMujocoConfig",
        },
        "stripped_overrides": {},
        "commands": {},
    },
    "k1_velocity": {
        "cls": "jaxrlworld.rl.configs.presets.k1_velocity.base:K1VelocityConfig",
        "stripped_overrides": {"action_delay_min": 0, "action_delay_max": 0},
        "commands": {"velocity": (0.5, 0.0, 0.0)},
    },
}

# Cell label -> (backend, MujocoSceneConfig.forward_after_physics).
CELLS = {
    "genesis": ("genesis", False),
    "newton": ("newton", False),
    "mujoco": ("mujoco", False),
    "mujoco+fwd": ("mujoco", True),
}
# Stage A compares inputs; the forward switch cannot change them.
STAGE_A_CELLS = ("genesis", "newton", "mujoco")
STAGE_BC_CELLS = ("genesis", "newton", "mujoco", "mujoco+fwd")

AIR_LIFT = 1.0
AIR_STEPS = 3
AIR_JOINT_OFFSET = 0.3
DROPS = {"drop_reset_height": 0.0, "drop_plus_10cm": 0.10}
DROP_STEPS = 100
DROP_TRACE_STEPS = 3
SETTLE_WINDOW = 30
STRIPPED_NUM_ENVS = 4

TOL_STATE = 1e-5
TOL_PARAM = 1e-6
# A history slot is attributed only where the force moved by more than this
# across the substeps of one control step; under a constant stance load every
# slot matches and the slot says nothing.
SLOT_SPREAD_N = 1.0

_MJ_INTEGRATOR = {0: "euler", 1: "rk4", 2: "implicit", 3: "implicitfast"}
_MJ_CONE = {0: "pyramidal", 1: "elliptic"}

# What the stripped env's reset writes. Set by the cell before each
# ``env.reset()``; the reset event reads it, so one build serves every
# scenario and every scenario goes through the full reset path (contact
# refresh, reward and action manager resets).
_SCENARIO: dict = {"lift": 0.0, "joint_offset": 0.0}


def script_fingerprint() -> str:
    """Digest of this file, stamped into every dump so cells from two
    versions of the script are never compared against each other."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]


def leaf(name: str) -> str:
    return name.rsplit("/", 1)[-1]


def joint_names(env) -> list[str]:
    return [leaf(n) for n in env.act_manager.actuated_joint_names]


def to_list(t: torch.Tensor) -> list:
    return t.detach().double().cpu().tolist()


def per_env(t: torch.Tensor, num_envs: int) -> torch.Tensor:
    """A model field with its leading world axis expanded to ``num_envs``.

    mjwarp keeps a field shared across worlds as a leading axis of 1 until
    something randomizes it per world; both shapes are legitimate and mean
    the same thing.
    """
    t = t.detach().double().cpu()
    if t.shape[0] == num_envs:
        return t
    if t.shape[0] == 1:
        return t.expand(num_envs, *t.shape[1:])
    raise ValueError(f"leading axis {t.shape[0]} is neither 1 nor num_envs={num_envs}")


def per_env_cols(t: torch.Tensor, num_envs: int) -> torch.Tensor:
    """A Genesis getter's result as ``(num_envs, n)``; unbatched info is ``(n,)``."""
    t = t.detach().double().cpu()
    if t.dim() == 1:
        return t.unsqueeze(0).expand(num_envs, -1)
    if t.shape[0] != num_envs:
        raise ValueError(f"leading axis {t.shape[0]} != num_envs={num_envs}")
    return t


# ── building ─────────────────────────────────────────────────────────


def _write_scenario_state(env, env_ids: torch.Tensor, default_pos, default_quat_wxyz) -> None:
    """Reset event of the stripped env: the preset's default root pose raised
    by ``_SCENARIO["lift"]``, home joints plus a fixed per-joint offset, all
    velocities zero. The offset of a joint depends only on its name's rank,
    so every backend gets the same value whatever its joint order."""
    n = len(env_ids)
    device = env.device
    writer = env.get_robot_state_writer("robot")

    pos = torch.tensor(default_pos, device=device, dtype=torch.float32).unsqueeze(0).repeat(n, 1)
    pos[:, 2] += _SCENARIO["lift"]
    pos = pos + env.scene_manager.env_origins[env_ids]
    quat = torch.tensor(default_quat_wxyz, device=device, dtype=torch.float32).unsqueeze(0).repeat(n, 1)
    zeros3 = torch.zeros(n, 3, device=device)
    writer.set_root_pose(pos, quat, env_ids=env_ids)
    writer.set_root_velocity(zeros3, zeros3, env_ids=env_ids)

    names = joint_names(env)
    rank = {name: i for i, name in enumerate(sorted(names))}
    offsets = torch.tensor(
        [_SCENARIO["joint_offset"] * math.sin(1.7 * rank[name] + 0.5) for name in names], device=device
    )
    home = env._resolve_default_joint_pos("robot")
    q = (home + offsets).unsqueeze(0).repeat(n, 1)
    mid, half = env.act_manager.soft_joint_limits_of("robot")
    lo, hi = mid - half, mid + half
    if lo.dim() == 2:
        lo, hi = lo[env_ids], hi[env_ids]
    q = q.clamp(lo, hi)
    writer.set_dof_state(q, torch.zeros_like(q), env_ids=env_ids)
    writer.eval_fk(env_ids=env_ids)


def _strip(cfgs) -> None:
    """Remove every event and termination, then install the scenario reset.

    The root default pose comes from the preset's own ``reset_root`` term, so
    a backend-specific default (Newton's quaternion hook) is kept.
    """
    events = iter_terms(cfgs.event, EventTermConfig)
    roots = [t for t in events.values() if t.resolved_func is ev.reset_root_state_uniform]
    if len(roots) != 1:
        raise ValueError(f"expected one reset_root_state_uniform term, found {len(roots)} in {sorted(events)}")
    params = roots[0].params
    # ``reset_root_state_uniform``'s own default when the preset does not set one.
    default_quat = params["default_quat_wxyz"] if "default_quat_wxyz" in params else (1.0, 0.0, 0.0, 0.0)
    for name in events:
        setattr(cfgs.event, name, None)
    cfgs.event.diag_scenario_state = EventTermConfig(
        func=_write_scenario_state,
        mode="reset",
        params={"default_pos": tuple(params["default_pos"]), "default_quat_wxyz": tuple(default_quat)},
    )
    for name in iter_terms(cfgs.env.terminations, TerminationTermConfig):
        setattr(cfgs.env.terminations, name, None)


def build_env(preset: str, cell: str, num_envs: int, stripped: bool):
    spec = PRESETS[preset]
    sim, forward = CELLS[cell]
    path = spec["cls"][sim] if isinstance(spec["cls"], dict) else spec["cls"]
    mod_path, cls_name = path.split(":", 1)
    cfg = getattr(importlib.import_module(mod_path), cls_name)(sim_type=sim, num_envs=num_envs)
    if stripped:
        fields = {f.name for f in dataclasses.fields(cfg)}
        for key, value in spec["stripped_overrides"].items():
            if key not in fields:
                raise AttributeError(f"{cls_name} has no field {key!r}")
            setattr(cfg, key, value)
    cfgs = cfg.build()
    if sim == "mujoco":
        cfgs.scene.forward_after_physics = forward
    if stripped:
        _strip(cfgs)
    return BaseRunner._create_env_from_config(cfgs)


# ── live-engine readers ──────────────────────────────────────────────


def read_solver(env, sim: str) -> dict:
    out: dict = {
        "physics_dt": round(env.physics_dt, 9),
        "decimation": int(env.decimation),
        "control_dt": round(env.physics_dt * env.decimation, 9),
    }
    if sim in ("mujoco", "newton"):
        m = env.scene_manager.mj_model if sim == "mujoco" else env.scene_manager.solver.mj_model
        out.update(
            iterations=int(m.opt.iterations),
            ls_iterations=int(m.opt.ls_iterations),
            tolerance=float(m.opt.tolerance),
            impratio=float(m.opt.impratio),
            cone=_MJ_CONE[int(m.opt.cone)],
            integrator=_MJ_INTEGRATOR[int(m.opt.integrator)],
            gravity_z=float(m.opt.gravity[2]),
            disableflags=int(m.opt.disableflags),
        )
    else:
        # Per-backend import: genesis is only loaded in a genesis process.
        import genesis as gs

        rigid = env.scene_manager.scene.sim.rigid_solver
        options = rigid._options
        out.update(
            iterations=int(options.iterations),
            ls_iterations=int(options.ls_iterations),
            # Both default to None in Genesis's options, meaning "resolved
            # internally"; reported as set.
            tolerance=options.tolerance,
            impratio=options.impratio,
            cone=gs.friction_cone(int(options.friction_cone)).name.lower(),
            integrator=gs.integrator(int(options.integrator)).name.lower(),
            gravity_z=float(rigid.get_gravity().reshape(-1, 3)[0, 2]),
            constraint_solver=gs.constraint_solver(int(options.constraint_solver)).name.lower(),
        )
    return out


def _canonical_joint(raw: str, vocab: list[str]) -> str | None:
    hits = [n for n in vocab if raw == n or raw.endswith(("/" + n, "_" + n))]
    return max(hits, key=len) if hits else None


def read_params(env, sim: str) -> dict:
    """Per-env plant parameters as the engine holds them.

    ``mass`` and ``friction`` are keyed by the backend's raw body name (the
    comparison canonicalizes); per-DOF fields by actuated joint name;
    ``friction`` is the largest slide friction among a body's collision
    geoms (the pair takes the max on all three, the ground being the lower).
    """
    n = env.num_envs
    names = joint_names(env)
    out: dict = {"mass": {}, "friction": {}, "ground_friction": [], "dof": {}, "pd": {}}

    if sim in ("mujoco", "newton"):
        if sim == "mujoco":
            mj = env.scene_manager.sim.mj_model
            model = env.scene_manager.sim.model

            def field(attr: str) -> torch.Tensor:
                return per_env(getattr(model, attr)[:], n)

            entity = env.scene_manager.get_entity("robot")
            bodies = {int(b) for b in entity.indexing.body_ids.cpu().tolist()}
        else:
            # Per-backend import: warp is read only for Newton's own model.
            import warp as wp

            solver = env.scene_manager.solver
            mj = solver.mj_model

            def field(attr: str) -> torch.Tensor:
                return per_env(wp.to_torch(getattr(solver.mjw_model, attr)), n)

            bodies = set(range(1, mj.nbody))
        body_mass = field("body_mass")
        geom_friction = field("geom_friction")
        dof_fields = {k: field(f"dof_{k}") for k in ("frictionloss", "damping", "armature")}
        for b in sorted(bodies):
            out["mass"][mj.body(b).name] = to_list(body_mass[:, b])
        per_body: dict[str, list[torch.Tensor]] = {}
        for g in range(mj.ngeom):
            if int(mj.geom_contype[g]) == 0 and int(mj.geom_conaffinity[g]) == 0:
                continue
            b = int(mj.geom_bodyid[g])
            if b in bodies:
                per_body.setdefault(mj.body(b).name, []).append(geom_friction[:, g, 0])
            elif b == 0:
                out["ground_friction"].append(float(geom_friction[0, g, 0]))
        for body, cols in per_body.items():
            out["friction"][body] = to_list(torch.stack(cols, dim=1).max(dim=1).values)
        for j in range(mj.njnt):
            if int(mj.jnt_type[j]) == 0:  # free joint
                continue
            name = _canonical_joint(mj.joint(j).name, names)
            if name is None:
                continue
            dof = int(mj.jnt_dofadr[j])
            out["dof"][name] = {k: to_list(v[:, dof]) for k, v in dof_fields.items()}
    else:
        entity = env.scene_manager["robot"]
        solver = env.scene_manager.scene.sim.rigid_solver
        masses = per_env_cols(entity.get_links_mass(), n)
        for i, link in enumerate(entity.links):
            out["mass"][link.name] = to_list(masses[:, i])
        base = solver.get_geoms_friction().double().cpu()
        ratio = per_env_cols(solver.get_geoms_friction_ratio(), n)
        own = set()
        per_body = {}
        for geom in entity.geoms:
            own.add(geom.idx)
            per_body.setdefault(geom.link.name, []).append(base[geom.idx] * ratio[:, geom.idx])
        for body, cols in per_body.items():
            out["friction"][body] = to_list(torch.stack(cols, dim=1).max(dim=1).values)
        out["ground_friction"] = [float(base[g] * ratio[0, g]) for g in range(base.shape[0]) if g not in own]
        idx = env.act_manager._actuated_joint_indices
        dof_fields = {
            "frictionloss": per_env_cols(entity.get_dofs_frictionloss(dofs_idx_local=idx), n),
            "damping": per_env_cols(entity.get_dofs_damping(dofs_idx_local=idx), n),
            "armature": per_env_cols(entity.get_dofs_armature(dofs_idx_local=idx), n),
        }
        for col, name in enumerate(names):
            out["dof"][name] = {k: to_list(v[:, col]) for k, v in dof_fields.items()}

    for actuator, joint_ids in env.act_manager._actuators:
        for attr in ("stiffness", "damping", "effort_limit"):
            values = getattr(actuator, attr)
            values = per_env_cols(values, n) if values.dim() == 1 else per_env(values, n)
            for col, j in enumerate(torch.as_tensor(joint_ids).cpu().tolist()):
                out["pd"].setdefault(names[j], {})[attr] = to_list(values[:, col])
    return out


def read_state(env) -> dict:
    rd = env.get_entity_data("robot")
    return {
        "root_pos": to_list(rd.root_link_pos_w - env.scene_manager.env_origins),
        "root_quat": to_list(rd.root_link_quat_w),
        "root_lin_vel": to_list(rd.root_link_lin_vel_w),
        "root_ang_vel": to_list(rd.root_link_ang_vel_w),
        "joint_pos": to_list(rd.joint_pos),
        "joint_vel": to_list(rd.joint_vel),
        "commands": to_list(env.command_manager.get_commands_tensor()),
    }


class SubstepTrace:
    """Records joint state and torque after every physics substep.

    Wraps the scene manager's ``step`` (one substep on every backend) on the
    instance. Bumping the env's read-cache generation first only discards
    memoized reads; it changes no simulator state.
    """

    def __init__(self, env):
        self.env = env
        self.rows: list[dict] = []
        self._inner = env.scene_manager.step
        env.scene_manager.step = self._step

    def _step(self) -> None:
        self._inner()
        self.env._invalidate_cache()
        rd = self.env.get_entity_data("robot")
        self.rows.append(
            {
                "joint_pos": rd.joint_pos.detach().double().cpu(),
                "joint_vel": rd.joint_vel.detach().double().cpu(),
                "act_torque": self.env.act_manager.applied_torque.detach().double().cpu(),
                "sim_torque": rd.applied_torque.detach().double().cpu(),
            }
        )

    def take(self, env_index: int | None = None) -> list[dict]:
        rows, self.rows = self.rows, []
        sel = slice(None) if env_index is None else env_index
        return [{k: to_list(v[sel]) for k, v in row.items()} for row in rows]


def read_contacts(env) -> dict:
    cm = env.contact_manager
    out = {}
    for group in cm.group_names():
        entry = {
            "names": list(cm.tracked_names(group)),
            "is_contact": to_list(cm.is_contact(group).float()[0]),
            "first_contact": to_list(cm.compute_first_contact(group).float()[0]),
        }
        if "force" in cm._groups[group].fields:
            entry["force"] = to_list(cm.contact_force(group)[0])
            history = cm.contact_force_history(group)
            entry["history_norm"] = None if history is None else to_list(torch.linalg.norm(history[0], dim=-1))
        out[group] = entry
    return out


# ── cells ────────────────────────────────────────────────────────────


def zero_action(env) -> torch.Tensor:
    return torch.zeros(env.num_envs, env.num_actions, device=env.device)


def cell_inputs(preset: str, cell: str, num_envs: int, seed: int) -> dict:
    sim = CELLS[cell][0]
    env = build_env(preset, cell, num_envs, stripped=False)
    torch.manual_seed(seed)
    env.reset()
    record = {"solver": read_solver(env, sim), "joints": joint_names(env)}
    record["params"] = read_params(env, sim)
    record["state"] = read_state(env)
    trace = SubstepTrace(env)
    record["step0_rewards"] = {k: v.tolist() for k, v in _capture_step(env, zero_action(env)).items()}
    record["step0_substeps"] = trace.take()
    record["params_after_step0"] = read_params(env, sim)
    record["state_after_step0"] = read_state(env)
    return record


def _set_commands(env, preset: str) -> None:
    ids = torch.arange(env.num_envs, device=env.device)
    for term, value in PRESETS[preset]["commands"].items():
        env.command_manager.set_commands(
            ids, **{term: torch.tensor(value, device=env.device).unsqueeze(0).repeat(env.num_envs, 1)}
        )


def _env_spread(env) -> float:
    """Largest deviation of any env's joint velocity from env 0's: the
    stripped envs are identical, so anything above roundoff means they are
    not."""
    qd = env.get_entity_data("robot").joint_vel
    return float((qd - qd[0:1]).abs().max())


def cell_stripped(preset: str, cell: str) -> dict:
    sim = CELLS[cell][0]
    env = build_env(preset, cell, STRIPPED_NUM_ENVS, stripped=True)
    record = {"solver": read_solver(env, sim), "joints": joint_names(env)}
    trace = SubstepTrace(env)

    # B: airborne, joints off home.
    _SCENARIO.update(lift=AIR_LIFT, joint_offset=AIR_JOINT_OFFSET)
    env.reset()
    trace.take()
    _set_commands(env, preset)
    record["params"] = read_params(env, sim)
    air = {"start": read_state(env), "steps": []}
    for _ in range(AIR_STEPS):
        env.step(zero_action(env))
        rd = env.get_entity_data("robot")
        air["steps"].append(
            {
                "substeps": trace.take(env_index=0),
                "root_z": float(rd.root_link_pos_w[0, 2] - env.scene_manager.env_origins[0, 2]),
                "root_vz": float(rd.root_link_lin_vel_w[0, 2]),
                "env_spread": _env_spread(env),
            }
        )
    record["air"] = air

    # C: drops.
    record["drops"] = {}
    for label, lift in DROPS.items():
        _SCENARIO.update(lift=lift, joint_offset=0.0)
        env.reset()
        trace.take()
        _set_commands(env, preset)
        steps = []
        for k in range(DROP_STEPS):
            rewards = _capture_step(env, zero_action(env))
            substeps = trace.take(env_index=0)
            rd = env.get_entity_data("robot")
            steps.append(
                {
                    "root_z": float(rd.root_link_pos_w[0, 2] - env.scene_manager.env_origins[0, 2]),
                    "root_vz": float(rd.root_link_lin_vel_w[0, 2]),
                    "qd_sq": float((rd.joint_vel[0] ** 2).sum()),
                    "rewards": {name: float(v[0]) for name, v in rewards.items()},
                    "contacts": read_contacts(env),
                    "substeps": substeps if k < DROP_TRACE_STEPS else None,
                    "env_spread": _env_spread(env),
                }
            )
        record["drops"][label] = steps
    return record


# ── report ───────────────────────────────────────────────────────────

_PAIRS = (("newton", "mujoco"), ("genesis", "mujoco"), ("genesis", "newton"))


def load(out: Path, preset: str, stage: str, cell: str) -> dict | None:
    path = cell_path(out, preset, stage, cell)
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    if record["fingerprint"] != script_fingerprint():
        raise SystemExit(f"{path} was written by another version of this script; rerun that cell")
    return record


def canon_body(raw: str, vocab: set[str]) -> str:
    """One body name across backends: mjlab prefixes the entity, Newton keeps
    a path-like label, Genesis the bare link name."""
    name = leaf(raw)
    if name in vocab:
        return name
    hits = [v for v in vocab if name.endswith("_" + v)]
    return max(hits, key=len) if hits else name


def body_vocab(records: dict) -> set[str]:
    for cell in ("mujoco", "genesis", "newton"):
        if cell in records:
            return {leaf(k) for k in records[cell]["params"]["mass"]}
    raise ValueError("no record to take body names from")


def by_joint(record: dict, values: list, names: list[str]) -> np.ndarray:
    """``(..., n_joints)`` values reordered from this backend's joint order to ``names``."""
    arr = np.asarray(values, dtype=float)
    index = [record["joints"].index(n) for n in names]
    return arr[..., index]


def flatten_params(record: dict, vocab: set[str], key: str = "params") -> dict[str, np.ndarray]:
    """``group.name -> (num_envs,)`` for every per-env plant parameter."""
    params = record[key]
    out = {}
    for raw, v in params["mass"].items():
        out[f"mass.{canon_body(raw, vocab)}"] = np.asarray(v)
    for raw, v in params["friction"].items():
        out[f"friction.{canon_body(raw, vocab)}"] = np.asarray(v)
    for joint, fields in params["dof"].items():
        for field_name, v in fields.items():
            out[f"dof_{field_name}.{joint}"] = np.asarray(v)
    for joint, fields in params["pd"].items():
        for field_name, v in fields.items():
            out[f"pd_{field_name}.{joint}"] = np.asarray(v)
    return out


def print_param_diffs(records: dict, vocab: set[str], key: str, env_slice: slice) -> None:
    flat = {cell: flatten_params(rec, vocab, key) for cell, rec in records.items()}
    cells = list(records)
    keys = sorted(set().union(*(set(f) for f in flat.values())))
    groups: dict[str, list[str]] = {}
    for k in keys:
        groups.setdefault(k.split(".", 1)[0], []).append(k)
    pairs = [p for p in _PAIRS if p[0] in records and p[1] in records]
    print(f"      {'group':<20}" + "".join(f"{a[:2].upper()}-{b[:2].upper()} max|d|".rjust(16) for a, b in pairs))
    for group, members in groups.items():
        worst = {}
        missing = [m for m in members if any(m not in flat[c] for c in cells)]
        for a, b in pairs:
            diffs = [
                (float(np.max(np.abs(flat[a][m][env_slice] - flat[b][m][env_slice]))), m)
                for m in members
                if m in flat[a] and m in flat[b]
            ]
            worst[(a, b)] = max(diffs) if diffs else (float("nan"), "")
        print(f"      {group:<20}" + "".join(f"{worst[p][0]:>16.3g}" for p in pairs))
        offenders = sorted(
            {m for m in members if all(m in flat[c] for c in cells)},
            key=lambda m: -max(float(np.max(np.abs(flat[a][m][env_slice] - flat[b][m][env_slice]))) for a, b in pairs),
        )
        shown = 0
        for m in offenders:
            gap = max(float(np.max(np.abs(flat[a][m][env_slice] - flat[b][m][env_slice]))) for a, b in pairs)
            if gap <= TOL_PARAM or shown == 6:
                break
            env0 = "  ".join(f"{c}={flat[c][m][0]:.6g}" for c in cells)
            print(f"        {m:<40} env0: {env0}")
            shown += 1
        if missing:
            print(f"        not on every backend: {missing[:6]}")


def report_inputs(out: Path, preset: str, cells: list[str]) -> None:
    records = {c: r for c in STAGE_A_CELLS if c in cells and (r := load(out, preset, "A", c)) is not None}
    if len(records) < 2:
        print("  (stage A: fewer than two backends dumped)")
        return
    print("\n  [A] check_reward_parity inputs: DR + random reset on, seeded once")
    names = sorted(next(iter(records.values()))["joints"])
    pairs = [p for p in _PAIRS if p[0] in records and p[1] in records]
    print(
        f"      {'initial state':<20}" + "".join(f"{a[:2].upper()}-{b[:2].upper()} max|d|".rjust(16) for a, b in pairs)
    )
    for key in ("root_pos", "root_quat", "root_lin_vel", "root_ang_vel", "joint_pos", "joint_vel", "commands"):
        vals = {}
        for cell, rec in records.items():
            v = rec["state"][key]
            vals[cell] = by_joint(rec, v, names) if key.startswith("joint") else np.asarray(v, dtype=float)
        row = []
        for a, b in pairs:
            if vals[a].shape != vals[b].shape:
                row.append(float("inf"))
            elif key == "root_quat":
                row.append(float(np.max(np.minimum(np.abs(vals[a] - vals[b]), np.abs(vals[a] + vals[b])))))
            else:
                row.append(float(np.max(np.abs(vals[a] - vals[b]))))
        flag = "  <-- differs" if any(x > TOL_STATE for x in row) else ""
        print(f"      {key:<20}" + "".join(f"{x:>16.3g}" for x in row) + flag)

    vocab = body_vocab(records)
    print("\n      plant parameters right after reset (per env)")
    print_param_diffs(records, vocab, "params", slice(None))
    print("\n      plant parameters after step 0 (interval DR may fire inside the step)")
    print_param_diffs(records, vocab, "params_after_step0", slice(None))

    print("\n      step 0, mean over envs of sum(qd^2) after each substep")
    for cell, rec in records.items():
        series = [float(np.mean(np.sum(np.asarray(s["joint_vel"]) ** 2, axis=1))) for s in rec["step0_substeps"]]
        print(f"        {cell:<12}" + "".join(f"{x:>10.4f}" for x in series))
    print("\n      step 0 raw reward terms, mean over envs")
    terms = sorted(set().union(*(set(r["step0_rewards"]) for r in records.values())))
    print(f"        {'term':<34}" + "".join(f"{c:>14}" for c in records))
    for term in terms:
        cells_v = [records[c]["step0_rewards"].get(term) for c in records]
        print(
            f"        {term:<34}" + "".join(f"{np.mean(v):>14.5f}" if v is not None else f"{'-':>14}" for v in cells_v)
        )


def feet_group(contacts: dict) -> str:
    named = [g for g in contacts if "feet" in g or "foot" in g]
    ground = [g for g in named if "ground" in g]
    if ground:
        return ground[0]
    if named:
        return named[0]
    return next(iter(contacts))


def report_stripped(out: Path, preset: str, cells: list[str]) -> None:
    records = {c: r for c in STAGE_BC_CELLS if c in cells and (r := load(out, preset, "BC", c)) is not None}
    if len(records) < 2:
        print("  (stages B/C: fewer than two backends dumped)")
        return
    names = sorted(next(iter(records.values()))["joints"])
    ref = "mujoco" if "mujoco" in records else next(iter(records))

    print("\n  [B/C] DR, pushes, terminations off; identical written state")
    print("      solver")
    keys = sorted(set().union(*(set(r["solver"]) for r in records.values())))
    print(f"        {'field':<18}" + "".join(f"{c:>16}" for c in records))
    for key in keys:
        print(f"        {key:<18}" + "".join(f"{str(r['solver'].get(key, '-')):>16}" for r in records.values()))

    vocab = body_vocab(records)
    print("\n      plant parameters, env 0 (no DR: every env equal)")
    print_param_diffs(records, vocab, "params", slice(0, 1))
    mass = {c: sum(v[0] for v in r["params"]["mass"].values()) for c, r in records.items()}
    print("        total mass of the dumped bodies: " + "  ".join(f"{c}={m:.4f}" for c, m in mass.items()))
    print("        ground friction: " + "  ".join(f"{c}={r['params']['ground_friction']}" for c, r in records.items()))

    print(f"\n  [B] airborne {AIR_LIFT} m, joints offset from home, zero action; per substep")
    print(f"      {'step.sub':<10}" + "".join(f"{c + ' sum qd^2':>22}" for c in records) + f"   max|qd - {ref}|")
    for k in range(AIR_STEPS):
        subs = {c: r["air"]["steps"][k]["substeps"] for c, r in records.items()}
        counts = {c: len(v) for c, v in subs.items()}
        if len(set(counts.values())) != 1:
            print(f"      step {k}: substep counts differ {counts}; rows compared up to the shortest")
        for s in range(min(counts.values())):
            qd = {c: by_joint(records[c], subs[c][s]["joint_vel"], names) for c in records}
            dq = "  ".join(f"{c}={np.max(np.abs(qd[c] - qd[ref])):.3g}" for c in records if c != ref)
            print(f"      {k}.{s:<8}" + "".join(f"{np.sum(qd[c] ** 2):>22.5f}" for c in records) + f"   {dq}")
    print("      torque, last substep of step 0 (act = action-manager PD, sim = engine report)")
    for kind in ("act_torque", "sim_torque"):
        t = {c: by_joint(records[c], r["air"]["steps"][0]["substeps"][-1][kind], names) for c, r in records.items()}
        print(
            f"        {kind:<12}"
            + "  ".join(
                f"{c}: |t|={np.linalg.norm(t[c]):.4f} max|d|={np.max(np.abs(t[c] - t[ref])):.3g}" for c in records
            )
        )
    for c, r in records.items():
        g = r["solver"]["gravity_z"]
        dt = r["solver"]["control_dt"]
        vz = [s["root_vz"] for s in r["air"]["steps"]]
        expect = [g * dt * (k + 1) for k in range(AIR_STEPS)]
        spread = max(s["env_spread"] for s in r["air"]["steps"])
        print(
            f"        {c:<12} root vz {np.round(vz, 5).tolist()} (free fall {np.round(expect, 5).tolist()}), env spread {spread:.2g}"
        )

    for label in DROPS:
        report_drop(records, label, names, mass)


def _history_slot(force_norm: float, history: list[float]) -> int | None:
    if max(history) - min(history) <= SLOT_SPREAD_N:
        return None
    return int(np.argmin(np.abs(np.asarray(history) - force_norm)))


def report_drop(records: dict, label: str, names: list[str], mass: dict) -> None:
    print(f"\n  [C] {label}: zero action for {DROP_STEPS} control steps")
    first = next(iter(records.values()))["drops"][label]
    group = feet_group(first[0]["contacts"])
    print(f"      contact group: {group} {first[0]['contacts'][group]['names']}")
    has_soft = "soft_landing" in first[0]["rewards"]
    # The table's last column: the term this diag was written for, per preset.
    col_term = "soft_landing" if has_soft else "dof_vel"

    def feet_force(step: dict) -> float:
        entry = step["contacts"][group]
        return float(np.sum(np.linalg.norm(np.asarray(entry["force"]), axis=-1))) if "force" in entry else float("nan")

    header = f"      {'step':<5}" + "".join(f"{c:>34}" for c in records)
    print(header)
    print(f"      {'':<5}" + "".join(f"{'z  sum qd^2  sum|F|  1st ' + col_term[:6]:>34}" for _ in records))
    for k in range(12):
        row = f"      {k:<5}"
        for r in records.values():
            s = r["drops"][label][k]
            n_first = int(sum(s["contacts"][group]["first_contact"]))
            row += (
                f"{s['root_z']:>7.3f}{s['qd_sq']:>8.2f}{feet_force(s):>8.1f}{n_first:>4}{s['rewards'][col_term]:>7.2f}"
            )
        print(row)

    print("      reported force = which substep-history slot (counted where the history moves)")
    for c, r in records.items():
        slots: Counter = Counter()
        length = None
        for s in r["drops"][label]:
            entry = s["contacts"][group]
            if "force" not in entry or entry["history_norm"] is None:
                continue
            norms = np.linalg.norm(np.asarray(entry["force"]), axis=-1)
            length = len(entry["history_norm"][0])
            for foot, hist in enumerate(entry["history_norm"]):
                slot = _history_slot(float(norms[foot]), hist)
                if slot is not None:
                    slots[slot] += 1
        print(f"        {c:<12} history length {length}, slots {dict(sorted(slots.items()))}")

    print("      touchdown step (first first_contact): reported sum|F|, history impulse over the step, soft_landing")
    for c, r in records.items():
        steps = r["drops"][label]
        td = next((k for k, s in enumerate(steps) if sum(s["contacts"][group]["first_contact"]) > 0), None)
        if td is None:
            print(f"        {c:<12} no touchdown")
            continue
        s = steps[td]
        entry = s["contacts"][group]
        impulse = float("nan")
        if entry.get("history_norm") is not None:
            hist = np.asarray(entry["history_norm"])
            dec = r["solver"]["decimation"]
            if hist.shape[1] == dec:
                impulse = float(hist.sum() * r["solver"]["physics_dt"])
        soft = s["rewards"]["soft_landing"] if has_soft else float("nan")
        print(
            f"        {c:<12} step {td}: sum|F| {feet_force(s):9.2f} N, impulse {impulse:8.4f} N*s, "
            f"soft_landing {soft:9.3f}, first {s['contacts'][group]['first_contact']}"
        )

    print(f"      settled, mean over the last {SETTLE_WINDOW} steps, against M*g")
    for c, r in records.items():
        tail = r["drops"][label][-SETTLE_WINDOW:]
        weight = mass[c] * abs(r["solver"]["gravity_z"])
        rep = float(np.mean([feet_force(s) for s in tail]))
        hist_mean = float("nan")
        if tail[0]["contacts"][group].get("history_norm") is not None:
            hist_mean = float(np.mean([np.sum(np.mean(s["contacts"][group]["history_norm"], axis=1)) for s in tail]))
        others = {
            g: float(np.mean([np.sum(np.linalg.norm(np.asarray(s["contacts"][g]["force"]), axis=-1)) for s in tail]))
            for g in tail[0]["contacts"]
            if g != group and "force" in tail[0]["contacts"][g]
        }
        print(
            f"        {c:<12} reported {rep / weight:6.3f} Mg, history {hist_mean / weight:6.3f} Mg, "
            f"z {tail[-1]['root_z']:.4f}, other groups {{{', '.join(f'{g}: {v:.1f} N' for g, v in others.items())}}}"
        )

    print("      raw reward terms summed over the rollout (and over steps 0-4)")
    terms = sorted(first[0]["rewards"])
    print(f"        {'term':<34}" + "".join(f"{c:>22}" for c in records))
    for term in terms:
        row = f"        {term:<34}"
        for r in records.values():
            series = [s["rewards"][term] for s in r["drops"][label]]
            row += f"{sum(series):>12.4f} ({sum(series[:5]):>7.3f})"
        print(row)
    print("      steps 0-2, sum(qd^2) after each substep")
    for c, r in records.items():
        subs = [sub for s in r["drops"][label][:DROP_TRACE_STEPS] for sub in s["substeps"]]
        print(f"        {c:<12}" + "".join(f"{np.sum(np.asarray(s['joint_vel']) ** 2):>9.3f}" for s in subs))
    spread = {c: max(s["env_spread"] for s in r["drops"][label]) for c, r in records.items()}
    print("      env spread (should be roundoff): " + "  ".join(f"{c}={v:.2g}" for c, v in spread.items()))


def report_preset(out: Path, preset: str, stages: list[str], cells: list[str]) -> None:
    print("\n" + "=" * 110)
    print(f"  {preset}")
    print("=" * 110)
    if "A" in stages:
        report_inputs(out, preset, cells)
    if "BC" in stages:
        report_stripped(out, preset, cells)


# ── driver ───────────────────────────────────────────────────────────


def cell_path(out: Path, preset: str, stage: str, cell: str) -> Path:
    return out / preset / f"{stage}_{cell.replace('+', '_')}.json"


def run_cell(args, preset: str, stage: str, cell: str) -> None:
    path = cell_path(Path(args.out), preset, stage, cell)
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        _MODULE,
        "--_cell",
        f"{preset}@{stage}@{cell}",
        "--out",
        args.out,
        "--num-envs",
        str(args.num_envs),
        "--seed",
        str(args.seed),
    ]
    print(f"=== {preset} stage {stage} [{cell}]", flush=True)
    log = path.with_suffix(".log")
    with open(log, "w") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        print(log.read_text()[-4000:])
        raise SystemExit(f"cell {preset}@{stage}@{cell} failed (exit {proc.returncode}); log {log}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--presets", nargs="+", default=list(PRESETS), choices=list(PRESETS))
    ap.add_argument("--cells", nargs="+", default=list(STAGE_BC_CELLS), choices=list(CELLS))
    ap.add_argument("--stages", nargs="+", default=["A", "BC"], choices=["A", "BC"])
    ap.add_argument("--num-envs", type=int, default=64, help="stage A env count (check_reward_parity used 64)")
    ap.add_argument("--seed", type=int, default=0, help="stage A torch seed (check_reward_parity default 0)")
    ap.add_argument("--out", default="runs/dynamics_attribution")
    ap.add_argument("--compare-only", action="store_true")
    ap.add_argument("--_cell", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args._cell is not None:
        preset, stage, cell = args._cell.split("@")
        if stage == "A":
            record = cell_inputs(preset, cell, args.num_envs, args.seed)
        else:
            record = cell_stripped(preset, cell)
        record["fingerprint"] = script_fingerprint()
        cell_path(Path(args.out), preset, stage, cell).write_text(json.dumps(record))
        return 0

    if not args.compare_only:
        for preset in args.presets:
            for stage in args.stages:
                cells = STAGE_A_CELLS if stage == "A" else STAGE_BC_CELLS
                for cell in cells:
                    if cell in args.cells:
                        run_cell(args, preset, stage, cell)

    for preset in args.presets:
        report_preset(Path(args.out), preset, args.stages, args.cells)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
