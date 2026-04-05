"""Newton domain randomization terms.

All functions follow the standard event term signature::

    func(env: NewtonEnv, env_ids: torch.Tensor, **params) -> None

They are meant to be used with :class:`EventTermConfig` in preset configs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp
from newton.solvers import SolverNotifyFlags

from ._utils import DefaultCache, apply_operation, sample

if TYPE_CHECKING:
    from rlworld.rl.envs import NewtonEnv

_defaults = DefaultCache()


# ------------------------------------------------------------------ #
#  Friction                                                           #
# ------------------------------------------------------------------ #

def randomize_friction(
    env: "NewtonEnv",
    env_ids: torch.Tensor,
    friction_range: tuple[float, float] = (0.3, 1.2),
    operation: str = "abs",
    distribution: str = "uniform",
) -> None:
    """Randomize friction for the robot's shapes via ArticulationView.

    Only touches robot shapes — the ground plane is left untouched.

    Args:
        env: Newton environment instance.
        env_ids: Environment indices to randomize.
        friction_range: Value range.  Interpretation depends on *operation*:
            ``"abs"`` — absolute friction coefficient,
            ``"scale"`` — multiplier on default value,
            ``"add"`` — additive offset to default value.
        operation: ``"abs"`` | ``"scale"`` | ``"add"``.
        distribution: ``"uniform"`` | ``"log_uniform"`` | ``"gaussian"``.
    """
    if len(env_ids) == 0:
        return

    view = env.scene_manager.robot_view
    model = env.scene_manager.model

    mu = wp.to_torch(view.get_attribute("shape_material_mu", model))

    shape = (len(env_ids), mu.shape[1], mu.shape[-1])
    sampled = sample(shape, *friction_range, env.device, distribution)

    if operation == "abs":
        mu[env_ids] = sampled
    else:
        defaults = _defaults.get_or_cache("friction", mu.clone())
        mu[env_ids] = apply_operation(defaults[env_ids], sampled, operation)

    view.set_attribute("shape_material_mu", model, mu)
    env.scene_manager.solver.notify_model_changed(SolverNotifyFlags.SHAPE_PROPERTIES)


# ------------------------------------------------------------------ #
#  Body mass                                                          #
# ------------------------------------------------------------------ #

def randomize_body_mass(
    env: "NewtonEnv",
    env_ids: torch.Tensor,
    mass_range: tuple[float, float] = (0.85, 1.15),
    operation: str = "scale",
    distribution: str = "uniform",
    body_patterns: str | list[str] = (".*",),
) -> None:
    """Randomize body masses for matched bodies.

    Args:
        env: Newton environment instance.
        env_ids: Environment indices to randomize.
        mass_range: Value range (interpreted per *operation*).
        operation: ``"abs"`` | ``"scale"`` | ``"add"``.
        distribution: ``"uniform"`` | ``"log_uniform"`` | ``"gaussian"``.
        body_patterns: Regex patterns matched against body labels.
    """
    if len(env_ids) == 0:
        return

    from rlworld.rl.envs.utils.newton.body_cache import get_cache

    cache = get_cache(env)
    model = env.scene_manager.model
    body_indices = cache.get_body_indices(body_patterns)

    mass = wp.to_torch(model.body_mass).reshape(env.num_envs, cache.bodies_per_env)

    n_envs = len(env_ids)
    n_bodies = len(body_indices)
    sampled = sample((n_envs, n_bodies), *mass_range, env.device, distribution)

    defaults = _defaults.get_or_cache(
        "body_mass", mass.clone(),
    )
    mass[env_ids.unsqueeze(1), body_indices] = apply_operation(
        defaults[env_ids.unsqueeze(1), body_indices], sampled, operation,
    )

    wp.copy(
        model.body_mass,
        wp.from_torch(mass.reshape(-1).contiguous(), dtype=wp.float32),
    )
    env.scene_manager.solver.notify_model_changed(
        SolverNotifyFlags.BODY_INERTIAL_PROPERTIES,
    )


# ------------------------------------------------------------------ #
#  Body COM offset                                                    #
# ------------------------------------------------------------------ #

def randomize_body_com_offset(
    env: "NewtonEnv",
    env_ids: torch.Tensor,
    ranges: dict[int, tuple[float, float]],
    body_patterns: str | list[str] = ("torso_link",),
) -> None:
    """Randomize body COM offset for specified bodies.

    Always additive on the original COM (cached on first call).

    Args:
        env: Newton environment instance.
        env_ids: Environment indices to randomize.
        ranges: Per-axis ``{axis_index: (min, max)}`` offset in meters.
        body_patterns: Regex patterns matched against body labels.
    """
    if len(env_ids) == 0:
        return

    from rlworld.rl.envs.utils.newton.body_cache import get_cache

    cache = get_cache(env)
    model = env.scene_manager.model
    body_indices = cache.get_body_indices(body_patterns)

    body_com = wp.to_torch(model.body_com).reshape(
        env.num_envs, cache.bodies_per_env, 3,
    )

    defaults = _defaults.get_or_cache("body_com", body_com.clone())

    n_envs = len(env_ids)
    n_bodies = len(body_indices)
    original = defaults[:, body_indices, :][env_ids]

    offsets = torch.zeros(n_envs, n_bodies, 3, device=env.device)
    for axis, (lo, hi) in ranges.items():
        offsets[:, :, axis] = torch.empty(
            n_envs, n_bodies, device=env.device,
        ).uniform_(lo, hi)

    body_com[env_ids.unsqueeze(1), body_indices] = original + offsets

    wp.copy(
        model.body_com,
        wp.from_torch(body_com.reshape(-1, 3).contiguous(), dtype=wp.vec3),
    )
    env.scene_manager.solver.notify_model_changed(
        SolverNotifyFlags.BODY_INERTIAL_PROPERTIES,
    )


# ------------------------------------------------------------------ #
#  Joint PD gains / armature                                          #
# ------------------------------------------------------------------ #

def randomize_pd_gains(
    env: "NewtonEnv",
    env_ids: torch.Tensor,
    kp_range: tuple[float, float] | None = None,
    kd_range: tuple[float, float] | None = None,
    operation: str = "scale",
    distribution: str = "uniform",
) -> None:
    """Randomize PD controller gains (joint_target_ke / joint_target_kd).

    Args:
        env: Newton environment instance.
        env_ids: Environment indices to randomize.
        kp_range: Range for proportional gain (stiffness). ``None`` to skip.
        kd_range: Range for derivative gain (damping). ``None`` to skip.
        operation: ``"abs"`` | ``"scale"`` | ``"add"``.
        distribution: ``"uniform"`` | ``"log_uniform"`` | ``"gaussian"``.
    """
    if len(env_ids) == 0:
        return

    view = env.scene_manager.robot_view
    model = env.scene_manager.model
    notify = False

    if kp_range is not None:
        ke = wp.to_torch(view.get_attribute("joint_target_ke", model))
        shape = (len(env_ids),) + ke.shape[1:]
        sampled = sample(shape, *kp_range, env.device, distribution)
        if operation == "abs":
            ke[env_ids] = sampled
        else:
            defaults = _defaults.get_or_cache("joint_target_ke", ke.clone())
            ke[env_ids] = apply_operation(defaults[env_ids], sampled, operation)
        view.set_attribute("joint_target_ke", model, ke)
        notify = True

    if kd_range is not None:
        kd = wp.to_torch(view.get_attribute("joint_target_kd", model))
        shape = (len(env_ids),) + kd.shape[1:]
        sampled = sample(shape, *kd_range, env.device, distribution)
        if operation == "abs":
            kd[env_ids] = sampled
        else:
            defaults = _defaults.get_or_cache("joint_target_kd", kd.clone())
            kd[env_ids] = apply_operation(defaults[env_ids], sampled, operation)
        view.set_attribute("joint_target_kd", model, kd)
        notify = True

    if notify:
        env.scene_manager.solver.notify_model_changed(
            SolverNotifyFlags.JOINT_DOF_PROPERTIES,
        )


def randomize_joint_armature(
    env: "NewtonEnv",
    env_ids: torch.Tensor,
    armature_range: tuple[float, float] = (0.9, 1.1),
    operation: str = "scale",
    distribution: str = "uniform",
) -> None:
    """Randomize joint armature (reflected rotor inertia).

    Args:
        env: Newton environment instance.
        env_ids: Environment indices to randomize.
        armature_range: Value range (interpreted per *operation*).
        operation: ``"abs"`` | ``"scale"`` | ``"add"``.
        distribution: ``"uniform"`` | ``"log_uniform"`` | ``"gaussian"``.
    """
    if len(env_ids) == 0:
        return

    view = env.scene_manager.robot_view
    model = env.scene_manager.model

    armature = wp.to_torch(view.get_attribute("joint_armature", model))
    shape = (len(env_ids),) + armature.shape[1:]
    sampled = sample(shape, *armature_range, env.device, distribution)

    if operation == "abs":
        armature[env_ids] = sampled
    else:
        defaults = _defaults.get_or_cache("joint_armature", armature.clone())
        armature[env_ids] = apply_operation(defaults[env_ids], sampled, operation)

    view.set_attribute("joint_armature", model, armature)
    env.scene_manager.solver.notify_model_changed(
        SolverNotifyFlags.JOINT_DOF_PROPERTIES,
    )


def randomize_joint_friction(
    env: "NewtonEnv",
    env_ids: torch.Tensor,
    friction_range: tuple[float, float] = (0.0, 0.05),
    operation: str = "abs",
    distribution: str = "uniform",
) -> None:
    """Randomize joint friction (load-independent).

    Args:
        env: Newton environment instance.
        env_ids: Environment indices to randomize.
        friction_range: Value range (interpreted per *operation*).
        operation: ``"abs"`` | ``"scale"`` | ``"add"``.
        distribution: ``"uniform"`` | ``"log_uniform"`` | ``"gaussian"``.
    """
    if len(env_ids) == 0:
        return

    view = env.scene_manager.robot_view
    model = env.scene_manager.model

    friction = wp.to_torch(view.get_attribute("joint_friction", model))
    shape = (len(env_ids),) + friction.shape[1:]
    sampled = sample(shape, *friction_range, env.device, distribution)

    if operation == "abs":
        friction[env_ids] = sampled
    else:
        defaults = _defaults.get_or_cache("joint_friction", friction.clone())
        friction[env_ids] = apply_operation(defaults[env_ids], sampled, operation)

    view.set_attribute("joint_friction", model, friction)
    env.scene_manager.solver.notify_model_changed(
        SolverNotifyFlags.JOINT_DOF_PROPERTIES,
    )
