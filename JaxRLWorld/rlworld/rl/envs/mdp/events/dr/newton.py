"""Newton domain randomization terms.

All functions follow the standard event term signature::

    func(env: NewtonEnv, env_ids: torch.Tensor, **params) -> None

They are meant to be used with :class:`EventTermConfig` in preset configs.
"""

from __future__ import annotations

from fnmatch import fnmatch
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
    env: NewtonEnv,
    env_ids: torch.Tensor,
    friction_range: tuple[float, float] = (0.3, 1.2),
    operation: str = "abs",
    distribution: str = "uniform",
    shape_patterns: tuple[str, ...] | None = None,
    shared_random: bool = False,
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
        shape_patterns: Optional fnmatch globs against ``model.shape_label``
            (full XPath labels, e.g. ``"go2/worldbody/.../FR_foot_collision"``).
            When ``None``, every robot shape is randomized (legacy behavior).
            When set, only matched shapes are touched — used to mirror
            mjlab's foot-only friction DR via patterns like
            ``("*/FR_foot_collision", "*/FL_foot_collision", ...)``.
        shared_random: When ``True``, draw a single sample per env and
            broadcast it across all matched shapes. Mirrors mjlab's
            ``shared_random=True`` so the four foot geoms receive the
            same friction value within an env (rather than independent
            samples). Default ``False`` preserves legacy per-shape sampling.
    """
    if len(env_ids) == 0:
        return

    view = env.scene_manager.robot_view
    model = env.scene_manager.model

    mu = wp.to_torch(view.get_attribute("shape_material_mu", model))
    # Newton stores ``shape_material_mu`` as
    # ``(num_worlds, n_axes, n_shapes_per_world)`` — the axis dim sits
    # in the middle (slide-only here, torsional / rolling live on
    # separate attributes). Earlier the reverse layout
    # ``(num_worlds, n_shapes, n_axes)`` was assumed and produced
    # device-side asserts on shape mismatch.
    n_axes = mu.shape[1]
    n_shapes_per_env = mu.shape[-1]

    # Resolve shape indices. Newton stores ``shape_label`` as a flat
    # ``world_count * n_shapes_per_env`` list (the same per-shape labels
    # repeated for each world); slicing the first world is enough since
    # the per-shape ordering is identical across worlds.
    if shape_patterns is None:
        shape_indices = None
        n_target = n_shapes_per_env
    else:
        all_labels = list(model.shape_label)
        world_count = model.world_count
        labels_per_env = len(all_labels) // world_count
        first_env_labels = all_labels[:labels_per_env]
        matched = [i for i, l in enumerate(first_env_labels) if any(fnmatch(l, p) for p in shape_patterns)]
        if not matched:
            raise ValueError(
                f"shape_patterns={shape_patterns} matched 0 shapes. " f"Sample labels: {first_env_labels[:6]}"
            )
        shape_indices = torch.tensor(matched, device=env.device, dtype=torch.long)
        n_target = len(matched)

    # Sample. Layout matches mu: (num_envs, n_axes, n_target).
    if shared_random:
        sampled = (
            sample((len(env_ids), n_axes, 1), *friction_range, env.device, distribution)
            .expand(len(env_ids), n_axes, n_target)
            .contiguous()
        )
    else:
        sampled = sample(
            (len(env_ids), n_axes, n_target),
            *friction_range,
            env.device,
            distribution,
        )

    # Apply.
    if shape_indices is None:
        if operation == "abs":
            mu[env_ids] = sampled
        else:
            defaults = _defaults.get_or_cache("friction", mu.clone())
            mu[env_ids] = apply_operation(defaults[env_ids], sampled, operation)
    else:
        env_grid, shape_grid = torch.meshgrid(env_ids, shape_indices, indexing="ij")
        # mu is (num_envs, n_axes, n_shapes_per_env): iterate over the
        # axis dim so the (env, shape) advanced indexing stays clean.
        if operation == "abs":
            for axis_idx in range(n_axes):
                mu[env_grid, axis_idx, shape_grid] = sampled[:, axis_idx, :]
        else:
            defaults = _defaults.get_or_cache("friction", mu.clone())
            for axis_idx in range(n_axes):
                base = defaults[env_grid, axis_idx, shape_grid]
                mu[env_grid, axis_idx, shape_grid] = apply_operation(base, sampled[:, axis_idx, :], operation)
    view.set_attribute("shape_material_mu", model, mu)
    env.scene_manager.solver.notify_model_changed(SolverNotifyFlags.SHAPE_PROPERTIES)


# Axis-index → Newton attribute name. Mirrors MuJoCo's ``geom_friction``
# vec3 layout ``(slide, torsional, rolling)`` — Newton stores the three
# axes as separate per-shape float arrays and its MuJoCo solver bridge
# packs them back into ``geom_friction[world, geom]`` on notify.
_FRICTION_AXIS_ATTR = {
    0: "shape_material_mu",
    1: "shape_material_mu_torsional",
    2: "shape_material_mu_rolling",
}


def randomize_geom_friction_axis(
    env: NewtonEnv,
    env_ids: torch.Tensor,
    ranges: tuple[float, float],
    axes: list[int] = (0,),
    operation: str = "abs",
    distribution: str = "uniform",
    body_patterns: str | list[str] | None = None,
) -> None:
    """Randomize per-axis geom friction (slide / torsional / rolling).

    Mirrors mjlab's ``dr.geom_friction`` with its ``axes`` parameter —
    used by mjlab_playground's getup task to independently randomize
    each of the three MuJoCo friction components:

        axis 0 = slide       (uniform 0.3..1.5     over all collision geoms)
        axis 1 = torsional   (log_uniform 1e-4..2e-2 over foot geoms)
        axis 2 = rolling     (log_uniform 1e-5..5e-3 over foot geoms)

    Newton exposes the three axes as separate float arrays on the
    model (``shape_material_mu``/``_torsional``/``_rolling``), and its
    MuJoCo solver bridge syncs all three on ``notify_model_changed``
    (``SHAPE_PROPERTIES``). This function therefore produces behaviour
    bit-compatible with mjlab when the Newton env is using the MuJoCo
    backend solver.

    **Filtering**: Newton's URDF loader drops geom names (all shapes
    come out as ``shape_0``, ``shape_1``, ...), so mjlab's geom-name
    regex filter is unusable here. Instead we filter by *body name*
    — the body index → shape indices mapping lives on
    ``model.body_shapes`` (a dict) and body labels survive the URDF
    load. Pass e.g. ``body_patterns=r"T1/(left|right)_foot_link"`` to
    target only the foot shapes. This matches the pattern SysID uses
    in ``sysid/param_terms/newton.py::apply_contact_friction``.

    Args:
        env: Newton environment instance.
        env_ids: Environment indices to randomize.
        ranges: ``(lo, hi)`` value range. Interpretation follows
            ``operation`` ("abs" replaces, "scale" multiplies cached
            default, "add" offsets cached default).
        axes: Which friction axes to randomize. Any subset of
            ``[0, 1, 2]``. Default ``[0]`` (slide only) matches mjlab.
        operation: ``"abs"`` | ``"scale"`` | ``"add"``.
        distribution: ``"uniform"`` | ``"log_uniform"`` | ``"gaussian"``.
        body_patterns: Optional regex pattern(s) matched against body
            labels via :class:`NewtonBodyCache`. ``None`` randomizes
            every shape on the robot view (the legacy
            ``randomize_friction`` behaviour). When given, only the
            shapes attached to the matched bodies are touched.
    """
    if len(env_ids) == 0:
        return

    view = env.scene_manager.robot_view
    model = env.scene_manager.model

    # Resolve body_patterns → shape indices (per-env-local) once for
    # the whole call. We use ``model.body_shapes`` — same path that
    # ``sysid.param_terms.newton.apply_contact_friction`` uses and
    # that SysID has validated against Newton's URDF-loaded T1.
    shape_idx: list[int] | None = None
    if body_patterns is not None:
        from rlworld.rl.envs.utils.newton.body_cache import get_cache

        cache = get_cache(env)
        body_indices = cache.get_body_indices(body_patterns)
        if hasattr(body_indices, "tolist"):
            body_indices = body_indices.tolist()
        collected: list[int] = []
        for bi in body_indices:
            for si in model.body_shapes[int(bi)]:
                collected.append(int(si))
        if not collected:
            raise ValueError(
                f"body_patterns={body_patterns!r} matched bodies "
                f"{body_indices} but none of them have collision "
                f"shapes on model.body_shapes."
            )
        shape_idx = collected

    for axis in axes:
        if axis not in _FRICTION_AXIS_ATTR:
            raise ValueError(f"Unknown friction axis {axis}. Valid: {sorted(_FRICTION_AXIS_ATTR)}.")
        attr = _FRICTION_AXIS_ATTR[axis]
        values = wp.to_torch(view.get_attribute(attr, model))

        if shape_idx is None:
            full_shape = (len(env_ids), values.shape[1], values.shape[-1])
            sampled = sample(full_shape, *ranges, env.device, distribution)
            if operation == "abs":
                values[env_ids] = sampled
            else:
                defaults = _defaults.get_or_cache(f"geom_friction_axis_{axis}", values.clone())
                values[env_ids] = apply_operation(defaults[env_ids], sampled, operation)
        else:
            shape_idx_t = torch.as_tensor(shape_idx, device=env.device, dtype=torch.long)
            sub = values[env_ids][:, :, shape_idx_t]
            sampled = sample(sub.shape, *ranges, env.device, distribution)
            if operation == "abs":
                sub_new = sampled
            else:
                defaults = _defaults.get_or_cache(f"geom_friction_axis_{axis}", values.clone())
                default_sub = defaults[env_ids][:, :, shape_idx_t]
                sub_new = apply_operation(default_sub, sampled, operation)
            values[
                env_ids[:, None, None],
                torch.arange(values.shape[1], device=env.device)[None, :, None],
                shape_idx_t[None, None, :],
            ] = sub_new

        view.set_attribute(attr, model, values)

    env.scene_manager.solver.notify_model_changed(SolverNotifyFlags.SHAPE_PROPERTIES)


# ------------------------------------------------------------------ #
#  Body mass                                                          #
# ------------------------------------------------------------------ #


def randomize_body_mass(
    env: NewtonEnv,
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

    defaults = _defaults.get_or_cache("body_mass", mass.clone())
    mass[env_ids.unsqueeze(1), body_indices] = apply_operation(
        defaults[env_ids.unsqueeze(1), body_indices],
        sampled,
        operation,
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
    env: NewtonEnv,
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
        env.num_envs,
        cache.bodies_per_env,
        3,
    )

    defaults = _defaults.get_or_cache("body_com", body_com.clone())

    n_envs = len(env_ids)
    n_bodies = len(body_indices)
    original = defaults[:, body_indices, :][env_ids]

    offsets = torch.zeros(n_envs, n_bodies, 3, device=env.device)
    for axis, (lo, hi) in ranges.items():
        offsets[:, :, axis] = torch.empty(
            n_envs,
            n_bodies,
            device=env.device,
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
    env: NewtonEnv,
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
    env: NewtonEnv,
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
    env: NewtonEnv,
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


# ------------------------------------------------------------------ #
#  Deterministic SysID-aligned setters                                #
# ------------------------------------------------------------------ #
# These are the *non-randomised* counterparts of ``randomize_friction``
# and ``randomize_joint_friction``, used to drive the env to a single
# fixed value identified by the SysID pipeline. Installed by
# ``go2_flat/_newton_builders.build_dr_terms`` whenever the matching
# ``Go2Config.*_override`` field is set on the preset. ``mode="reset_dr"``
# matches the randomise_* terms so the value is re-applied on every
# reset (equivalent to writing to the model once at build time, since
# the value never changes — but riding the existing reset hook avoids
# adding a new event mode).


def set_joint_friction(
    env: NewtonEnv,
    env_ids: torch.Tensor,
    value: float,
    dr_scale: tuple[float, float] | None = None,
) -> None:
    """Set joint Coulomb friction across every actuated DOF, optionally
    with narrow domain randomization centered on the identified value.

    Counterpart to :func:`randomize_joint_friction` for SysID-aligned
    training: when ``cfg.robot.joint_frictionloss_override`` is set on
    the preset, this term is installed in the event config so every
    reset writes the identified value to ``model.joint_friction``. Args
    mirror the SysID-side ``apply_joint_friction_scalar``, plus an
    optional ``dr_scale``.

    Args:
        value: Identified joint Coulomb friction (the SysID center).
        dr_scale: Optional ``(lo, hi)`` multiplicative band. When
            ``None`` the value is written exactly each reset (pure
            identified). When set, every reset writes
            ``value * uniform(lo, hi)`` per env — narrow DR centered on
            the identified value, e.g. ``(0.9, 1.1)`` gives ±10 % margin.
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
    env.scene_manager.solver.notify_model_changed(
        SolverNotifyFlags.JOINT_DOF_PROPERTIES,
    )


def set_foot_friction(
    env: NewtonEnv,
    env_ids: torch.Tensor,
    value: float,
    foot_pattern: str = ".*foot$",
    dr_scale: tuple[float, float] | None = None,
) -> None:
    """Set foot-shape contact mu *and* ground geom mu around a SysID-
    identified value, optionally with narrow domain randomization.

    Counterpart to :func:`randomize_friction` for SysID-aligned training,
    but: (a) targets only the foot-pattern bodies (matching the SysID
    identification scope on ``foot_friction``), and (b) also patches the
    ground/terrain ``geom_friction`` directly on the MuJoCo solver so
    the max-of-pair friction rule resolves to the identified foot value
    rather than the ground default. Mirrors
    ``consysid.sysid.param_terms.newton.apply_contact_friction`` so
    identification ↔ training contact behaviour is identical, plus an
    optional ``dr_scale`` band for narrow DR centered on ``value``.

    Args:
        value: Identified foot-ground friction coefficient (SysID center).
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
    env.scene_manager.solver.notify_model_changed(
        SolverNotifyFlags.SHAPE_PROPERTIES,
    )

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
