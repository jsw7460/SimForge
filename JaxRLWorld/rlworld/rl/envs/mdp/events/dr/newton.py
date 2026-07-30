"""Newton-specific event terms — fixed-value friction setters.

Cross-sim domain randomization (friction, body mass, COM offset, PD
gains, armature, joint friction) lives in :mod:`.unified`.  What
remains here are the *non-randomised* counterparts used to pin the
Newton env to a single configured friction value (optionally with a
narrow DR band): :func:`set_joint_friction` and :func:`set_foot_friction`.

Both follow the standard event-term signature
``func(env: NewtonEnv, env_ids: torch.Tensor, **params) -> None`` and
are installed by ``go2/_newton_builders.build_dr_terms`` only when
the matching ``Go2Config.*_override`` field is set on the preset.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import torch
import warp as wp
from newton import ModelFlags

from ._utils import sample
from .unified import _newton_notify

if TYPE_CHECKING:
    from rlworld.rl.envs import NewtonEnv


# ------------------------------------------------------------------ #
#  Fixed-value setters (with optional narrow DR band)                  #
# ------------------------------------------------------------------ #
# ``mode="reset_dr"`` matches the unified randomise_* terms so the
# value is re-applied on every reset (equivalent to writing to the
# model once at build time, since the value never changes — but riding
# the existing reset hook avoids adding a new event mode).


def set_joint_friction(
    env: NewtonEnv,
    env_ids: torch.Tensor,
    value: float,
    dr_scale: tuple[float, float] | None = None,
) -> None:
    """Set joint Coulomb friction across every actuated DOF, optionally
    with narrow domain randomization centered on the configured value.

    Deterministic counterpart to ``unified.randomize_joint_friction``:
    when ``cfg.robot.joint_frictionloss_override`` is set on the preset,
    this term is installed in the event config so every reset writes the
    configured value to ``model.joint_friction``, plus an optional
    ``dr_scale`` for a narrow band around it.

    Args:
        value: Joint Coulomb friction to pin (the DR center).
        dr_scale: Optional ``(lo, hi)`` multiplicative band. When
            ``None`` the value is written exactly each reset. When set,
            every reset writes ``value * uniform(lo, hi)`` per env —
            narrow DR centered on ``value``, e.g. ``(0.9, 1.1)`` gives
            ±10 % margin.
    """
    if len(env_ids) == 0:
        return

    view = env.scene_manager.robot_view
    model = env.scene_manager.model

    friction = wp.to_torch(view.get_attribute("joint_friction", model))
    if dr_scale is None:
        friction[env_ids] = float(value)
    else:
        scale = sample(
            friction[env_ids].shape,
            *dr_scale,
            env.device,
            "uniform",
        )
        friction[env_ids] = float(value) * scale
    view.set_attribute("joint_friction", model, friction)
    _newton_notify(env, ModelFlags.JOINT_DOF_PROPERTIES)


def set_foot_friction(
    env: NewtonEnv,
    env_ids: torch.Tensor,
    value: float,
    foot_label_pattern: str = r".*foot_collision$",
    dr_scale: tuple[float, float] | None = None,
) -> None:
    """Pin the foot collision shapes' contact mu to a configured value,
    optionally with a narrow multiplicative DR band.

    Foot collision shapes are matched by name against the raw
    ``model.shape_label`` and written directly into
    ``model.shape_material_mu``. This deliberately avoids
    ``model.body_shapes[foot]`` (empty — ``collapse_fixed_joints``
    reparents the foot collision geom onto its parent calf body) and the
    ``ArticulationView`` / ``resolve_selector`` shape indices (offset by
    one shape from this array, so a write lands on the neighbouring
    shape). The ground geom is not touched: Go2's foot collision geoms
    have priority 1 vs the ground plane's 0, so the foot value wins the
    foot↔ground pair.

    Args:
        value: Foot-ground friction coefficient to pin (the DR center).
        foot_label_pattern: Regex matched against ``model.shape_label``
            to select the foot collision shapes.
        dr_scale: Optional ``(lo, hi)`` multiplicative band. ``None`` →
            write ``value`` exactly each reset.
    """
    if len(env_ids) == 0:
        return

    model = env.scene_manager.model

    # Per-env mu — scalar broadcast when dr_scale is None, otherwise a
    # ``(len(env_ids),)`` tensor sampled once per reset.
    if dr_scale is None:
        mu_val = torch.tensor(float(value), dtype=torch.float32, device=env.device)
    else:
        scale = sample((len(env_ids),), *dr_scale, env.device, "uniform")
        mu_val = float(value) * scale

    shapes_per_env = model.shape_count // env.num_envs
    n_robot_shapes = env.num_envs * shapes_per_env
    flat_mu = wp.to_torch(model.shape_material_mu)
    shape_mu = flat_mu[:n_robot_shapes].reshape(env.num_envs, shapes_per_env)

    rx = re.compile(foot_label_pattern)
    target_shapes = [si for si in range(shapes_per_env) if rx.search(model.shape_label[si])]
    if not target_shapes:
        raise ValueError(f"set_foot_friction matched no shapes for foot_label_pattern={foot_label_pattern!r}.")
    for si in target_shapes:
        shape_mu[env_ids, si] = mu_val

    wp.copy(model.shape_material_mu, wp.from_torch(flat_mu, dtype=wp.float32))
    _newton_notify(env, ModelFlags.SHAPE_PROPERTIES)


# ------------------------------------------------------------------ #
#  Per-dim fixed-value setters (heterogeneous friction)               #
# ------------------------------------------------------------------ #
# Per-DOF joint friction and per-foot contact friction counterparts of
# the scalar setters above. Used when the "real world" ships distinct
# friction per joint / per foot (e.g. a 50D per-dim ground truth). The DOF
# / shape matching mirrors the per-dim apply helpers exactly (same
# ``joint_dof_names`` regex order, same ``shape_label`` reshape order), so
# the column order of ``values`` is identical between the collect env that
# bakes the ground truth here and the identification env that recovers it.


def _resolve_joint_dof_indices(view, joint_patterns: tuple[str, ...]) -> list[int]:
    """Articulation-local DOF indices whose joint name matches any pattern.

    Matches against ``view.joint_dof_names`` with the ``":N"`` multi-DOF
    suffix stripped, so one regex covers every sub-DOF of a multi-DOF
    joint. Returns indices in ``joint_dof_names`` order.
    """
    compiled = [re.compile(p) for p in joint_patterns]
    matched = [
        dof_idx
        for dof_idx, dof_name in enumerate(view.joint_dof_names)
        if any(rx.search(dof_name.rsplit(":", 1)[0]) for rx in compiled)
    ]
    if not matched:
        raise ValueError(
            f"set_joint_friction_per_dim: no joint DOFs match "
            f"joint_patterns={joint_patterns!r}. Available: {view.joint_names}"
        )
    return matched


def set_joint_friction_per_dim(
    env: NewtonEnv,
    env_ids: torch.Tensor,
    values: list[float],
    joint_patterns: tuple[str, ...] = (r".*_joint$",),
    dr_scale: tuple[float, float] | None = None,
) -> None:
    """Set heterogeneous Coulomb joint friction per matched DOF.

    Deterministic per-DOF counterpart to :func:`set_joint_friction`: when
    ``cfg.robot.joint_friction_per_joint_override`` is set, this term is
    installed so every reset writes the configured vector to the matched
    DOFs of ``model.joint_friction``.

    Args:
        values: One friction value per matched DOF, in ``joint_dof_names``
            order (length must equal the number of matched DOFs).
        joint_patterns: Regexes selecting the actuated DOFs. Default
            ``(.*_joint$,)`` picks the 12 leg joints, excluding the
            floating base.
        dr_scale: Optional ``(lo, hi)`` multiplicative band per DOF. When
            ``None`` the vector is written exactly each reset.
    """
    if len(env_ids) == 0:
        return

    view = env.scene_manager.robot_view
    model = env.scene_manager.model

    dof_indices = _resolve_joint_dof_indices(view, joint_patterns)
    if len(values) != len(dof_indices):
        raise ValueError(f"set_joint_friction_per_dim: got {len(values)} values for {len(dof_indices)} matched DOFs.")
    dof_idx_t = torch.as_tensor(dof_indices, dtype=torch.long, device=env.device)
    vals_t = torch.as_tensor(values, dtype=torch.float32, device=env.device)

    attrib = wp.to_torch(view.get_attribute("joint_friction", model))
    # (num_envs, count_per_world, joint_dof_count) — one articulation per
    # world in these setups, so squeeze the middle dim as a live view.
    attrib_flat = attrib[:, 0, :]
    if dr_scale is None:
        attrib_flat[env_ids.unsqueeze(1), dof_idx_t] = vals_t
    else:
        scale = sample(
            (len(env_ids), len(dof_indices)),
            *dr_scale,
            env.device,
            "uniform",
        )
        attrib_flat[env_ids.unsqueeze(1), dof_idx_t] = vals_t * scale

    view.set_attribute("joint_friction", model, attrib)
    _newton_notify(env, ModelFlags.JOINT_DOF_PROPERTIES)


def set_foot_friction_per_foot(
    env: NewtonEnv,
    env_ids: torch.Tensor,
    values: list[float],
    foot_label_pattern: str = r".*foot_collision$",
    dr_scale: tuple[float, float] | None = None,
) -> None:
    """Pin per-foot contact mu on the foot collision shapes.

    Deterministic per-foot counterpart to :func:`set_foot_friction`: when
    ``cfg.robot.foot_friction_per_foot_override`` is set, this term writes
    one mu per foot every reset. Foot collision shapes are matched against
    the raw ``model.shape_label`` (see :func:`set_foot_friction` for why
    the ArticulationView shape path is avoided); ground geoms are left
    untouched (foot priority wins the foot↔ground pair).

    Args:
        values: One mu per matched foot shape, in ``shape_label`` match
            order (length must equal the number of matched foot shapes).
        foot_label_pattern: Regex matched against ``model.shape_label``.
        dr_scale: Optional ``(lo, hi)`` multiplicative band per foot.
    """
    if len(env_ids) == 0:
        return

    model = env.scene_manager.model

    shapes_per_env = model.shape_count // env.num_envs
    n_robot_shapes = env.num_envs * shapes_per_env
    flat_mu = wp.to_torch(model.shape_material_mu)
    shape_mu = flat_mu[:n_robot_shapes].reshape(env.num_envs, shapes_per_env)

    rx = re.compile(foot_label_pattern)
    target_shapes = [si for si in range(shapes_per_env) if rx.search(model.shape_label[si])]
    if not target_shapes:
        raise ValueError(f"set_foot_friction_per_foot matched no shapes for foot_label_pattern={foot_label_pattern!r}.")
    if len(values) != len(target_shapes):
        raise ValueError(
            f"set_foot_friction_per_foot: got {len(values)} values for {len(target_shapes)} matched foot shapes."
        )

    for i, si in enumerate(target_shapes):
        if dr_scale is None:
            mu_val = torch.tensor(
                float(values[i]),
                dtype=torch.float32,
                device=env.device,
            )
        else:
            scale = sample((len(env_ids),), *dr_scale, env.device, "uniform")
            mu_val = float(values[i]) * scale
        shape_mu[env_ids, si] = mu_val

    wp.copy(model.shape_material_mu, wp.from_torch(flat_mu, dtype=wp.float32))
    _newton_notify(env, ModelFlags.SHAPE_PROPERTIES)
