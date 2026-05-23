"""Newton-specific event terms — fixed-value friction setters.

Cross-sim domain randomization (friction, body mass, COM offset, PD
gains, armature, joint friction) lives in :mod:`.unified`.  What
remains here are the *non-randomised* counterparts used to pin the
Newton env to a single configured friction value (optionally with a
narrow DR band): :func:`set_joint_friction` and :func:`set_foot_friction`.

Both follow the standard event-term signature
``func(env: NewtonEnv, env_ids: torch.Tensor, **params) -> None`` and
are installed by ``go2_flat/_newton_builders.build_dr_terms`` only when
the matching ``Go2Config.*_override`` field is set on the preset.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp
from newton.solvers import SolverNotifyFlags

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
    _newton_notify(env, SolverNotifyFlags.JOINT_DOF_PROPERTIES)


def set_foot_friction(
    env: NewtonEnv,
    env_ids: torch.Tensor,
    value: float,
    foot_pattern: str = ".*foot$",
    dr_scale: tuple[float, float] | None = None,
) -> None:
    """Set foot-shape contact mu *and* ground geom mu around a configured
    value, optionally with narrow domain randomization.

    Deterministic counterpart to ``unified.randomize_friction``, but:
    (a) targets only the foot-pattern bodies, and (b) also patches the
    ground/terrain ``geom_friction`` directly on the MuJoCo solver so
    the max-of-pair friction rule resolves to the configured foot value
    rather than the ground default.

    Args:
        value: Foot-ground friction coefficient to pin (the DR center).
        foot_pattern: Regex for foot bodies (default ``".*foot$"``).
        dr_scale: Optional ``(lo, hi)`` multiplicative band. ``None`` →
            write ``value`` exactly each reset. ``(0.9, 1.1)`` → write
            ``value * uniform(0.9, 1.1)`` per env (the same scalar is
            broadcast across all foot shapes + ground geoms within one
            env, so per-env contact mu stays consistent across pairs).
    """
    if len(env_ids) == 0:
        return

    from rlworld.rl.envs.utils.newton.body_cache import get_cache

    cache = get_cache(env)
    model = env.scene_manager.model
    body_indices = cache.get_body_indices(foot_pattern)
    if hasattr(body_indices, "tolist"):
        body_indices = body_indices.tolist()

    # Per-env mu — scalar broadcast when dr_scale is None, otherwise a
    # ``(len(env_ids),)`` tensor sampled once per reset so foot-foot,
    # foot-ground, etc. all see the *same* friction within an env.
    if dr_scale is None:
        mu_val = torch.tensor(
            float(value),
            dtype=torch.float32,
            device=env.device,
        )
    else:
        scale = sample(
            (len(env_ids),),
            *dr_scale,
            env.device,
            "uniform",
        )
        mu_val = float(value) * scale

    # ── Per-shape mu on the foot bodies ─────────────────────────────
    shapes_per_env = model.shape_count // env.num_envs
    n_robot_shapes = env.num_envs * shapes_per_env
    flat_mu = wp.to_torch(model.shape_material_mu)
    shape_mu = flat_mu[:n_robot_shapes].reshape(env.num_envs, shapes_per_env)

    for body_idx in body_indices:
        for si in model.body_shapes[int(body_idx)]:
            shape_mu[env_ids, int(si)] = mu_val

    wp.copy(model.shape_material_mu, wp.from_torch(flat_mu, dtype=wp.float32))
    _newton_notify(env, SolverNotifyFlags.SHAPE_PROPERTIES)

    # ── Ground geom mu (MuJoCo solver pair max() rule) ──────────────
    solver = env.scene_manager.solver
    mj_model = solver.mj_model
    mjw_friction = wp.to_torch(solver.mjw_model.geom_friction)  # [nworld, ngeom, 3]

    if not hasattr(env, "_ground_geom_indices"):
        ground_indices = []
        for i in range(mj_model.ngeom):
            name = mj_model.geom(i).name.lower()
            if "terrain" in name or "ground" in name or "plane" in name:
                ground_indices.append(i)
        env._ground_geom_indices = ground_indices

    for gi in env._ground_geom_indices:
        mjw_friction[env_ids, gi, 0] = mu_val
