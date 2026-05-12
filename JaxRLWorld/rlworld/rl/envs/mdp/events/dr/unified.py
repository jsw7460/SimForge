"""Sim-agnostic domain-randomization terms.

Each public function in this module accepts a
:class:`~rlworld.rl.configs.scene.entity_selector.SceneEntitySelector`
instead of sim-specific name/pattern parameters, then dispatches to a
private backend based on ``env.sim_type``.

Currently exposed:

* :func:`randomize_friction`         — geom-level friction (slide/spin/roll)
* :func:`randomize_body_mass`        — per-body mass randomization
* :func:`randomize_body_com_offset`  — per-body center-of-mass offset
* :func:`randomize_pd_gains`         — joint PD controller gains (kp/kd)
* :func:`randomize_joint_armature`   — reflected rotor inertia
* :func:`randomize_joint_friction`   — joint Coulomb friction
* :func:`randomize_encoder_bias`     — joint encoder bias (MuJoCo only)

These are the sole DR entry points presets should target.  Newton
keeps a couple of *non-randomised* SysID-aligned setters in
``events/dr/newton.py`` (``set_joint_friction`` / ``set_foot_friction``)
that have no cross-sim counterpart.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch

from rlworld.rl.configs.scene.entity_selector import ResolvedEntity, SceneEntitySelector
from rlworld.rl.envs.mdp.events.mujoco import _MujocoEnvAdapter

from ._utils import DefaultCache, apply_operation, sample

if TYPE_CHECKING:
    from rlworld.rl.envs.world import World


# NOTE on imports: ``warp``/``newton``/``mjlab`` are imported inside
# the per-backend helpers below.  pyproject.toml lists each simulator
# as an *optional* extra (``genesis``, ``newton``), so a user running
# with only one backend installed must still be able to import this
# module — top-level imports of the others would crash at import time.
# This mirrors the existing pattern in ``events/mujoco.py``.

_defaults_newton_friction = DefaultCache()


# Newton stores the three friction axes (slide / torsional / rolling)
# as separate per-shape float arrays.  The MuJoCo solver bridge inside
# Newton repacks them into ``geom_friction[world, geom]`` whenever
# ``notify_model_changed(SHAPE_PROPERTIES)`` fires, so writing the
# axis-specific attributes individually produces behaviour that matches
# mjlab's vec3 ``geom_friction`` field.  Mirrors the constant of the
# same name in ``events/dr/newton.py``.
_NEWTON_FRICTION_AXIS_ATTR = {
    0: "shape_material_mu",
    1: "shape_material_mu_torsional",
    2: "shape_material_mu_rolling",
}


def randomize_friction(
    env: World,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntitySelector = SceneEntitySelector(name="robot"),
    friction_range: tuple[float, float] = (0.3, 1.2),
    operation: Literal["abs", "scale", "add"] = "scale",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    shared_random: bool = False,
    axes: list[int] | None = None,
) -> None:
    """Randomize friction for the bodies / geoms selected by *asset_cfg*.

    Args:
        env: The simulator-agnostic World instance.
        env_ids: Environment indices to randomize.
        asset_cfg: Selector identifying the entity (and optional
            body/geom subset) to touch.
        friction_range: ``(lo, hi)`` value range; interpretation
            depends on *operation*.
        operation: ``"abs"`` (replace), ``"scale"`` (multiply default),
            or ``"add"`` (offset default).  Genesis only supports
            ``"scale"``.
        distribution: ``"uniform"`` | ``"log_uniform"`` | ``"gaussian"``.
        shared_random: If True draw a single sample per env and
            broadcast across all matched shapes (so e.g. all four feet
            share one friction value within an env).
        axes: Which friction axes to randomize.  Any subset of
            ``[0, 1, 2]`` (slide / torsional / rolling).  ``None``
            defaults to ``[0]`` (slide only) for parity with mjlab's
            ``geom_friction`` and the legacy per-sim ``randomize_friction``.

    Backends:

    * **Genesis** — uses ``RigidEntity.set_friction_ratio`` (link-level
      multiplier).  Only ``operation="scale"`` and ``axes=[0]`` are
      supported — Genesis exposes no separate torsional/rolling axes
      and no absolute/additive write path.
      ``asset_cfg.geom_names`` is rejected (Genesis geoms have no
      names) — pass ``body_names`` instead.
    * **Newton** — writes the three ``shape_material_mu*`` attributes
      directly via ``ArticulationView.set_attribute``.  When
      ``asset_cfg.body_names`` is given the matched links are expanded
      to their collision shapes; when ``asset_cfg.geom_names`` is given
      the regex matches against per-shape names directly.
    * **MuJoCo** — delegates to ``mjlab.envs.mdp.dr.geom_friction``
      with a transient mjlab ``SceneEntityCfg`` constructed inside the
      backend (mjlab is never exposed to the caller).
    """
    if len(env_ids) == 0:
        return

    resolved_axes = [0] if axes is None else list(axes)

    if env.sim_type == "genesis":
        _genesis_friction_backend(
            env,
            env_ids,
            env.resolve_selector(asset_cfg),
            friction_range,
            operation,
            distribution,
            shared_random,
            resolved_axes,
        )
    elif env.sim_type == "newton":
        _newton_friction_backend(
            env,
            env_ids,
            env.resolve_selector(asset_cfg),
            friction_range,
            operation,
            distribution,
            shared_random,
            resolved_axes,
        )
    elif env.sim_type == "mujoco":
        _mujoco_friction_backend(
            env,
            env_ids,
            asset_cfg,
            friction_range,
            operation,
            distribution,
            shared_random,
            resolved_axes,
        )
    else:
        raise NotImplementedError(f"randomize_friction has no backend for sim_type={env.sim_type!r}")


# ──────────────────────────────────────────────────────────────────────
# Genesis backend
# ──────────────────────────────────────────────────────────────────────


def _genesis_friction_backend(
    env: World,
    env_ids: torch.Tensor,
    resolved: ResolvedEntity,
    friction_range: tuple[float, float],
    operation: str,
    distribution: str,
    shared_random: bool,
    axes: list[int],
) -> None:
    if operation != "scale":
        raise NotImplementedError(
            f"Genesis friction DR only supports operation='scale' "
            f"(got {operation!r}); the underlying set_friction_ratio API is "
            f"a multiplier on the URDF default friction."
        )
    if axes != [0]:
        raise NotImplementedError(
            f"Genesis friction DR only supports axes=[0] (slide); got "
            f"{axes}.  Genesis does not expose torsional/rolling friction "
            f"as separate writable attributes."
        )

    entity = resolved.backend_handle
    if resolved.body_ids is None:
        links_idx = list(range(entity.n_links))
    else:
        links_idx = resolved.body_ids.tolist()
    n_target = len(links_idx)

    if shared_random:
        ratios = (
            sample((len(env_ids), 1), *friction_range, env.device, distribution)
            .expand(len(env_ids), n_target)
            .contiguous()
        )
    else:
        ratios = sample((len(env_ids), n_target), *friction_range, env.device, distribution)
    entity.set_friction_ratio(friction_ratio=ratios, links_idx_local=links_idx, envs_idx=env_ids)


# ──────────────────────────────────────────────────────────────────────
# Newton backend
# ──────────────────────────────────────────────────────────────────────


def _newton_friction_backend(
    env: World,
    env_ids: torch.Tensor,
    resolved: ResolvedEntity,
    friction_range: tuple[float, float],
    operation: str,
    distribution: str,
    shared_random: bool,
    axes: list[int],
) -> None:
    import warp as wp
    from newton.solvers import SolverNotifyFlags

    view = resolved.backend_handle
    model = env.scene_manager.model

    # Resolve target shape indices once for all axes.
    if resolved.geom_ids is not None:
        shape_indices: torch.Tensor | None = resolved.geom_ids
    elif resolved.body_ids is not None:
        # Expand each body to its collision shapes via the view's
        # ``link_shapes`` mapping (per-link list of shape indices into
        # the ArticulationView's per-shape attribute arrays).
        expanded: list[int] = []
        for body_idx in resolved.body_ids.tolist():
            expanded.extend(view.link_shapes[body_idx])
        if not expanded:
            raise ValueError(
                f"SceneEntitySelector body_names matched bodies with no "
                f"collision shapes (body_ids={resolved.body_ids.tolist()})."
            )
        shape_indices = torch.tensor(expanded, device=env.device, dtype=torch.long)
    else:
        shape_indices = None

    for axis in axes:
        if axis not in _NEWTON_FRICTION_AXIS_ATTR:
            raise ValueError(f"Unknown friction axis {axis}; valid axes are " f"{sorted(_NEWTON_FRICTION_AXIS_ATTR)}.")
        attr = _NEWTON_FRICTION_AXIS_ATTR[axis]
        values = wp.to_torch(view.get_attribute(attr, model))
        # Layout: (num_worlds, n_axes_inner, n_shapes_per_world).  The
        # inner ``n_axes`` is always 1 for Newton's per-axis friction
        # attributes; we keep the dim for indexing parity with the
        # ``view.get_attribute`` shape.
        n_axes_inner = values.shape[1]
        n_shapes_per_env = values.shape[-1]
        n_target = n_shapes_per_env if shape_indices is None else len(shape_indices)

        if shared_random:
            sampled = (
                sample(
                    (len(env_ids), n_axes_inner, 1),
                    *friction_range,
                    env.device,
                    distribution,
                )
                .expand(len(env_ids), n_axes_inner, n_target)
                .contiguous()
            )
        else:
            sampled = sample(
                (len(env_ids), n_axes_inner, n_target),
                *friction_range,
                env.device,
                distribution,
            )

        cache_key = f"friction_axis_{axis}"
        if shape_indices is None:
            if operation == "abs":
                values[env_ids] = sampled
            else:
                defaults = _defaults_newton_friction.get_or_cache(cache_key, values.clone())
                values[env_ids] = apply_operation(defaults[env_ids], sampled, operation)
        else:
            env_grid, shape_grid = torch.meshgrid(env_ids, shape_indices, indexing="ij")
            if operation == "abs":
                for inner in range(n_axes_inner):
                    values[env_grid, inner, shape_grid] = sampled[:, inner, :]
            else:
                defaults = _defaults_newton_friction.get_or_cache(cache_key, values.clone())
                for inner in range(n_axes_inner):
                    base = defaults[env_grid, inner, shape_grid]
                    values[env_grid, inner, shape_grid] = apply_operation(base, sampled[:, inner, :], operation)

        view.set_attribute(attr, model, values)

    env.scene_manager.solver.notify_model_changed(SolverNotifyFlags.SHAPE_PROPERTIES)


# ──────────────────────────────────────────────────────────────────────
# MuJoCo backend (delegates to mjlab DR)
# ──────────────────────────────────────────────────────────────────────


def _mujoco_friction_backend(
    env: World,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntitySelector,
    friction_range: tuple[float, float],
    operation: str,
    distribution: str,
    shared_random: bool,
    axes: list[int],
) -> None:
    # mjlab's geom_friction expects an mjlab SceneEntityCfg.  Build it
    # locally and resolve against the mjlab Scene; this is the *only*
    # place in the unified DR module where SceneEntitySelector → mjlab
    # cfg conversion happens for friction.
    from mjlab.envs.mdp.dr import geom_friction as _mjlab_geom_friction
    from mjlab.managers.scene_entity_config import SceneEntityCfg as _MjlabSceneEntityCfg

    mjlab_cfg = _MjlabSceneEntityCfg(
        name=asset_cfg.name,
        joint_names=tuple(asset_cfg.joint_names) if asset_cfg.joint_names else None,
        body_names=tuple(asset_cfg.body_names) if asset_cfg.body_names else None,
        geom_names=tuple(asset_cfg.geom_names) if asset_cfg.geom_names else None,
        site_names=tuple(asset_cfg.site_names) if asset_cfg.site_names else None,
    )
    adapter = _MujocoEnvAdapter(env)
    mjlab_cfg.resolve(adapter.scene)

    _mjlab_geom_friction(
        env=adapter,
        env_ids=env_ids,
        ranges=friction_range,
        asset_cfg=mjlab_cfg,
        operation=operation,
        axes=axes,
        shared_random=shared_random,
        distribution=distribution,
    )


# ──────────────────────────────────────────────────────────────────────
# Helpers shared by the per-function backends below
# ──────────────────────────────────────────────────────────────────────


def _selector_to_mjlab_cfg(asset_cfg: SceneEntitySelector, scene):
    """Build a resolved mjlab ``SceneEntityCfg`` from our selector.

    Used by every MuJoCo backend; mjlab import is local because the
    module-level rule says simulator deps are optional extras.
    """
    from mjlab.managers.scene_entity_config import SceneEntityCfg as _MjlabSceneEntityCfg

    cfg = _MjlabSceneEntityCfg(
        name=asset_cfg.name,
        joint_names=tuple(asset_cfg.joint_names) if asset_cfg.joint_names else None,
        body_names=tuple(asset_cfg.body_names) if asset_cfg.body_names else None,
        geom_names=tuple(asset_cfg.geom_names) if asset_cfg.geom_names else None,
        site_names=tuple(asset_cfg.site_names) if asset_cfg.site_names else None,
        actuator_names=(tuple(asset_cfg.actuator_names) if asset_cfg.actuator_names else None),
        preserve_order=asset_cfg.preserve_order,
    )
    cfg.resolve(scene)
    return cfg


# ══════════════════════════════════════════════════════════════════════
# randomize_body_mass
# ══════════════════════════════════════════════════════════════════════


def randomize_body_mass(
    env: World,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntitySelector = SceneEntitySelector(name="robot"),
    mass_range: tuple[float, float] = (0.85, 1.15),
    operation: Literal["abs", "scale", "add"] = "scale",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    shared_random: bool = False,
) -> None:
    """Randomize per-body mass for the bodies selected by *asset_cfg*.

    Genesis enforces ``operation="scale"`` (its ``set_mass_shift`` is a
    multiplier on URDF defaults); Newton/MuJoCo accept all operations.
    """
    if len(env_ids) == 0:
        return

    if env.sim_type == "genesis":
        _genesis_body_mass_backend(env, env_ids, env.resolve_selector(asset_cfg), mass_range, operation, distribution)
    elif env.sim_type == "newton":
        _newton_body_mass_backend(env, env_ids, env.resolve_selector(asset_cfg), mass_range, operation, distribution)
    elif env.sim_type == "mujoco":
        _mujoco_body_mass_backend(env, env_ids, asset_cfg, mass_range, operation, distribution, shared_random)
    else:
        raise NotImplementedError(f"randomize_body_mass has no backend for sim_type={env.sim_type!r}")


def _genesis_body_mass_backend(env, env_ids, resolved, mass_range, operation, distribution):
    if operation != "scale":
        raise NotImplementedError(
            f"Genesis body mass DR only supports operation='scale' "
            f"(got {operation!r}); set_mass_shift is a multiplier."
        )
    entity = resolved.backend_handle
    if resolved.body_ids is None:
        raise ValueError(
            "Genesis randomize_body_mass requires asset_cfg.body_names; " "got selector with no body subset."
        )
    links_idx = resolved.body_ids.tolist()
    n_envs, n_links = len(env_ids), len(links_idx)
    ratios = sample((n_envs, n_links), *mass_range, env.device, distribution)
    mass_shift = torch.zeros(n_envs, n_links, device=env.device)
    for i, idx in enumerate(links_idx):
        original_mass = entity.links[idx].get_mass()
        mass_shift[:, i] = original_mass * (ratios[:, i] - 1.0)
    entity.set_mass_shift(mass_shift=mass_shift, links_idx_local=links_idx, envs_idx=env_ids)


_defaults_newton_body_mass = DefaultCache()


def _newton_body_mass_backend(env, env_ids, resolved, mass_range, operation, distribution):
    import warp as wp
    from newton.solvers import SolverNotifyFlags

    from rlworld.rl.envs.utils.newton.body_cache import get_cache

    cache = get_cache(env)
    model = env.scene_manager.model
    if resolved.body_ids is None:
        body_indices = torch.arange(cache.bodies_per_env, device=env.device)
    else:
        # Map view-local link indices to model-global body indices via cache.
        # ``view.link_names`` is a subset (the robot articulation), so resolve
        # the names back to bodies via NewtonBodyCache for consistency with
        # legacy ``body_patterns`` path.
        view = resolved.backend_handle
        names = [view.link_names[i] for i in resolved.body_ids.tolist()]
        body_indices = cache.get_body_indices(names)
        if hasattr(body_indices, "tolist"):
            body_indices = torch.tensor(body_indices, device=env.device, dtype=torch.long)

    mass = wp.to_torch(model.body_mass).reshape(env.num_envs, cache.bodies_per_env)
    n_bodies = len(body_indices)
    sampled = sample((len(env_ids), n_bodies), *mass_range, env.device, distribution)
    defaults = _defaults_newton_body_mass.get_or_cache("body_mass", mass.clone())
    mass[env_ids.unsqueeze(1), body_indices] = apply_operation(
        defaults[env_ids.unsqueeze(1), body_indices], sampled, operation
    )
    wp.copy(model.body_mass, wp.from_torch(mass.reshape(-1).contiguous(), dtype=wp.float32))
    env.scene_manager.solver.notify_model_changed(SolverNotifyFlags.BODY_INERTIAL_PROPERTIES)


def _mujoco_body_mass_backend(env, env_ids, asset_cfg, mass_range, operation, distribution, shared_random):
    from mjlab.envs.mdp.dr import body_mass as _mjlab_body_mass

    adapter = _MujocoEnvAdapter(env)
    mjlab_cfg = _selector_to_mjlab_cfg(asset_cfg, adapter.scene)
    _mjlab_body_mass(
        env=adapter,
        env_ids=env_ids,
        ranges=mass_range,
        asset_cfg=mjlab_cfg,
        operation=operation,
        shared_random=shared_random,
    )


# ══════════════════════════════════════════════════════════════════════
# randomize_body_com_offset
# ══════════════════════════════════════════════════════════════════════


def randomize_body_com_offset(
    env: World,
    env_ids: torch.Tensor,
    ranges: dict[int, tuple[float, float]],
    asset_cfg: SceneEntitySelector = SceneEntitySelector(name="robot"),
    operation: Literal["abs", "scale", "add"] = "add",
    axes: list[int] | None = None,
    shared_random: bool = False,
) -> None:
    """Randomize per-body center-of-mass offset.

    ``ranges`` is a dict ``{axis_index: (lo, hi)}`` — e.g.
    ``{0: (-0.025, 0.025), 2: (-0.03, 0.03)}``.  Genesis and Newton are
    purely additive on the URDF default COM; MuJoCo accepts all
    ``operation`` values via mjlab's body_com_offset.
    """
    if len(env_ids) == 0:
        return

    if env.sim_type == "genesis":
        _genesis_body_com_offset_backend(env, env_ids, env.resolve_selector(asset_cfg), ranges, operation)
    elif env.sim_type == "newton":
        _newton_body_com_offset_backend(env, env_ids, env.resolve_selector(asset_cfg), ranges, operation)
    elif env.sim_type == "mujoco":
        _mujoco_body_com_offset_backend(env, env_ids, asset_cfg, ranges, operation, axes, shared_random)
    else:
        raise NotImplementedError(f"randomize_body_com_offset has no backend for sim_type={env.sim_type!r}")


def _genesis_body_com_offset_backend(env, env_ids, resolved, ranges, operation):
    if operation != "add":
        raise NotImplementedError(
            f"Genesis body COM offset DR only supports operation='add' "
            f"(got {operation!r}); set_COM_shift is purely additive."
        )
    entity = resolved.backend_handle
    if resolved.body_ids is None:
        raise ValueError("Genesis randomize_body_com_offset requires asset_cfg.body_names.")
    links_idx = resolved.body_ids.tolist()
    n_envs, n_links = len(env_ids), len(links_idx)
    com_shift = torch.zeros(n_envs, n_links, 3, device=env.device)
    for axis, (lo, hi) in ranges.items():
        com_shift[:, :, axis] = torch.empty(n_envs, n_links, device=env.device).uniform_(lo, hi)
    entity.set_COM_shift(com_shift=com_shift, links_idx_local=links_idx, envs_idx=env_ids)


_defaults_newton_body_com = DefaultCache()


def _newton_body_com_offset_backend(env, env_ids, resolved, ranges, operation):
    import warp as wp
    from newton.solvers import SolverNotifyFlags

    from rlworld.rl.envs.utils.newton.body_cache import get_cache

    if operation != "add":
        raise NotImplementedError(
            f"Newton body COM offset DR only supports operation='add' "
            f"(got {operation!r}); cached defaults are added to."
        )
    cache = get_cache(env)
    model = env.scene_manager.model
    if resolved.body_ids is None:
        body_indices = torch.arange(cache.bodies_per_env, device=env.device)
    else:
        view = resolved.backend_handle
        names = [view.link_names[i] for i in resolved.body_ids.tolist()]
        body_indices = cache.get_body_indices(names)
        if hasattr(body_indices, "tolist"):
            body_indices = torch.tensor(body_indices, device=env.device, dtype=torch.long)

    body_com = wp.to_torch(model.body_com).reshape(env.num_envs, cache.bodies_per_env, 3)
    defaults = _defaults_newton_body_com.get_or_cache("body_com", body_com.clone())
    n_envs, n_bodies = len(env_ids), len(body_indices)
    original = defaults[:, body_indices, :][env_ids]
    offsets = torch.zeros(n_envs, n_bodies, 3, device=env.device)
    for axis, (lo, hi) in ranges.items():
        offsets[:, :, axis] = torch.empty(n_envs, n_bodies, device=env.device).uniform_(lo, hi)
    body_com[env_ids.unsqueeze(1), body_indices] = original + offsets
    wp.copy(model.body_com, wp.from_torch(body_com.reshape(-1, 3).contiguous(), dtype=wp.vec3))
    env.scene_manager.solver.notify_model_changed(SolverNotifyFlags.BODY_INERTIAL_PROPERTIES)


def _mujoco_body_com_offset_backend(env, env_ids, asset_cfg, ranges, operation, axes, shared_random):
    from mjlab.envs.mdp.dr import body_com_offset as _mjlab_body_com_offset

    adapter = _MujocoEnvAdapter(env)
    mjlab_cfg = _selector_to_mjlab_cfg(asset_cfg, adapter.scene)
    _mjlab_body_com_offset(
        env=adapter,
        env_ids=env_ids,
        ranges=ranges,
        asset_cfg=mjlab_cfg,
        operation=operation,
        axes=axes,
        shared_random=shared_random,
    )


# ══════════════════════════════════════════════════════════════════════
# randomize_pd_gains
# ══════════════════════════════════════════════════════════════════════


_defaults_newton_pd = DefaultCache()


def randomize_pd_gains(
    env: World,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntitySelector = SceneEntitySelector(name="robot"),
    kp_range: tuple[float, float] | None = None,
    kd_range: tuple[float, float] | None = None,
    operation: Literal["abs", "scale", "add"] = "scale",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
) -> None:
    """Randomize PD controller gains (kp / kd) for actuated DOFs.

    Genesis only supports ``operation="scale"`` (no built-in ratio API
    on ``set_dofs_kp/kv`` so we sample a ratio and multiply).
    """
    if len(env_ids) == 0:
        return
    if kp_range is None and kd_range is None:
        return

    if env.sim_type == "genesis":
        _genesis_pd_gains_backend(
            env, env_ids, env.resolve_selector(asset_cfg), kp_range, kd_range, operation, distribution
        )
    elif env.sim_type == "newton":
        _newton_pd_gains_backend(env, env_ids, kp_range, kd_range, operation, distribution)
    elif env.sim_type == "mujoco":
        _mujoco_pd_gains_backend(env, env_ids, asset_cfg, kp_range, kd_range, operation, distribution)
    else:
        raise NotImplementedError(f"randomize_pd_gains has no backend for sim_type={env.sim_type!r}")


def _genesis_pd_gains_backend(env, env_ids, resolved, kp_range, kd_range, operation, distribution):
    if operation != "scale":
        raise NotImplementedError(
            f"Genesis pd_gains DR only supports operation='scale' "
            f"(got {operation!r}); set_dofs_kp/kv take absolute values."
        )
    entity = resolved.backend_handle
    n_dofs = entity.n_dofs
    if kp_range is not None:
        current_kp = entity.get_dofs_kp()
        ratios = sample((len(env_ids), n_dofs), *kp_range, env.device, distribution)
        kp_new = (current_kp * ratios) if current_kp.dim() == 1 else (current_kp[env_ids] * ratios)
        entity.set_dofs_kp(kp=kp_new.cpu().numpy(), envs_idx=env_ids)
    if kd_range is not None:
        current_kv = entity.get_dofs_kv()
        ratios = sample((len(env_ids), n_dofs), *kd_range, env.device, distribution)
        kv_new = (current_kv * ratios) if current_kv.dim() == 1 else (current_kv[env_ids] * ratios)
        entity.set_dofs_kv(kv=kv_new.cpu().numpy(), envs_idx=env_ids)


def _newton_pd_gains_backend(env, env_ids, kp_range, kd_range, operation, distribution):
    import warp as wp
    from newton.solvers import SolverNotifyFlags

    view = env.scene_manager.robot_view
    model = env.scene_manager.model
    notify = False
    for attr_name, value_range in (("joint_target_ke", kp_range), ("joint_target_kd", kd_range)):
        if value_range is None:
            continue
        values = wp.to_torch(view.get_attribute(attr_name, model))
        shape = (len(env_ids),) + values.shape[1:]
        sampled = sample(shape, *value_range, env.device, distribution)
        if operation == "abs":
            values[env_ids] = sampled
        else:
            defaults = _defaults_newton_pd.get_or_cache(attr_name, values.clone())
            values[env_ids] = apply_operation(defaults[env_ids], sampled, operation)
        view.set_attribute(attr_name, model, values)
        notify = True
    if notify:
        env.scene_manager.solver.notify_model_changed(SolverNotifyFlags.JOINT_DOF_PROPERTIES)


def _mujoco_pd_gains_backend(env, env_ids, asset_cfg, kp_range, kd_range, operation, distribution):
    from mjlab.envs.mdp.dr import pd_gains as _mjlab_pd_gains

    if kp_range is None or kd_range is None:
        raise NotImplementedError(
            "MuJoCo pd_gains requires both kp_range and kd_range " "(mjlab's dr.pd_gains has no None-skip option)."
        )
    adapter = _MujocoEnvAdapter(env)
    mjlab_cfg = _selector_to_mjlab_cfg(asset_cfg, adapter.scene)
    _mjlab_pd_gains(
        env=adapter,
        env_ids=env_ids,
        kp_range=kp_range,
        kd_range=kd_range,
        asset_cfg=mjlab_cfg,
        distribution=distribution,
        operation=operation,
    )


# ══════════════════════════════════════════════════════════════════════
# randomize_joint_armature
# ══════════════════════════════════════════════════════════════════════


_defaults_newton_armature = DefaultCache()


def randomize_joint_armature(
    env: World,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntitySelector = SceneEntitySelector(name="robot"),
    armature_range: tuple[float, float] = (0.9, 1.1),
    operation: Literal["abs", "scale", "add"] = "scale",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    shared_random: bool = False,
) -> None:
    """Randomize joint armature (reflected rotor inertia).

    Genesis enforces ``operation="scale"``.  All sims operate on the
    full actuated DOF set; per-joint subset selection via
    ``asset_cfg.joint_names`` is honored on MuJoCo (mjlab supports it)
    and ignored on Genesis/Newton (whose APIs touch all DOFs at once).
    """
    if len(env_ids) == 0:
        return

    if env.sim_type == "genesis":
        _genesis_armature_backend(
            env, env_ids, env.resolve_selector(asset_cfg), armature_range, operation, distribution
        )
    elif env.sim_type == "newton":
        _newton_armature_backend(env, env_ids, armature_range, operation, distribution)
    elif env.sim_type == "mujoco":
        _mujoco_armature_backend(env, env_ids, asset_cfg, armature_range, operation, distribution, shared_random)
    else:
        raise NotImplementedError(f"randomize_joint_armature has no backend for sim_type={env.sim_type!r}")


def _genesis_armature_backend(env, env_ids, resolved, armature_range, operation, distribution):
    if operation != "scale":
        raise NotImplementedError(f"Genesis joint_armature DR only supports operation='scale' (got {operation!r}).")
    entity = resolved.backend_handle
    n_dofs = entity.n_dofs
    current = entity.get_dofs_armature()
    ratios = sample((len(env_ids), n_dofs), *armature_range, env.device, distribution)
    arm_new = (current * ratios) if current.dim() == 1 else (current[env_ids] * ratios)
    entity.set_dofs_armature(armature=arm_new.cpu().numpy(), envs_idx=env_ids)


def _newton_armature_backend(env, env_ids, armature_range, operation, distribution):
    import warp as wp
    from newton.solvers import SolverNotifyFlags

    view = env.scene_manager.robot_view
    model = env.scene_manager.model
    armature = wp.to_torch(view.get_attribute("joint_armature", model))
    shape = (len(env_ids),) + armature.shape[1:]
    sampled = sample(shape, *armature_range, env.device, distribution)
    if operation == "abs":
        armature[env_ids] = sampled
    else:
        defaults = _defaults_newton_armature.get_or_cache("joint_armature", armature.clone())
        armature[env_ids] = apply_operation(defaults[env_ids], sampled, operation)
    view.set_attribute("joint_armature", model, armature)
    env.scene_manager.solver.notify_model_changed(SolverNotifyFlags.JOINT_DOF_PROPERTIES)


def _mujoco_armature_backend(env, env_ids, asset_cfg, armature_range, operation, distribution, shared_random):
    from mjlab.envs.mdp.dr import joint_armature as _mjlab_joint_armature

    adapter = _MujocoEnvAdapter(env)
    mjlab_cfg = _selector_to_mjlab_cfg(asset_cfg, adapter.scene)
    _mjlab_joint_armature(
        env=adapter,
        env_ids=env_ids,
        ranges=armature_range,
        asset_cfg=mjlab_cfg,
        operation=operation,
        shared_random=shared_random,
    )


# ══════════════════════════════════════════════════════════════════════
# randomize_joint_friction
# ══════════════════════════════════════════════════════════════════════


_defaults_newton_joint_friction = DefaultCache()


def randomize_joint_friction(
    env: World,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntitySelector = SceneEntitySelector(name="robot"),
    friction_range: tuple[float, float] = (0.0, 0.05),
    operation: Literal["abs", "scale", "add"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    shared_random: bool = False,
) -> None:
    """Randomize joint Coulomb friction (load-independent).

    Genesis enforces ``operation="abs"`` (its set_dofs_frictionloss
    takes absolute values).  All sims default to the full actuated DOF
    set.
    """
    if len(env_ids) == 0:
        return

    if env.sim_type == "genesis":
        _genesis_joint_friction_backend(
            env, env_ids, env.resolve_selector(asset_cfg), friction_range, operation, distribution
        )
    elif env.sim_type == "newton":
        _newton_joint_friction_backend(env, env_ids, friction_range, operation, distribution)
    elif env.sim_type == "mujoco":
        _mujoco_joint_friction_backend(env, env_ids, asset_cfg, friction_range, operation, distribution, shared_random)
    else:
        raise NotImplementedError(f"randomize_joint_friction has no backend for sim_type={env.sim_type!r}")


def _genesis_joint_friction_backend(env, env_ids, resolved, friction_range, operation, distribution):
    if operation != "abs":
        raise NotImplementedError(
            f"Genesis joint_friction DR only supports operation='abs' "
            f"(got {operation!r}); set_dofs_frictionloss takes absolute values."
        )
    entity = resolved.backend_handle
    n_dofs = entity.n_dofs
    values = sample((len(env_ids), n_dofs), *friction_range, env.device, distribution)
    entity.set_dofs_frictionloss(frictionloss=values.cpu().numpy(), envs_idx=env_ids)


def _newton_joint_friction_backend(env, env_ids, friction_range, operation, distribution):
    import warp as wp
    from newton.solvers import SolverNotifyFlags

    view = env.scene_manager.robot_view
    model = env.scene_manager.model
    friction = wp.to_torch(view.get_attribute("joint_friction", model))
    shape = (len(env_ids),) + friction.shape[1:]
    sampled = sample(shape, *friction_range, env.device, distribution)
    if operation == "abs":
        friction[env_ids] = sampled
    else:
        defaults = _defaults_newton_joint_friction.get_or_cache("joint_friction", friction.clone())
        friction[env_ids] = apply_operation(defaults[env_ids], sampled, operation)
    view.set_attribute("joint_friction", model, friction)
    env.scene_manager.solver.notify_model_changed(SolverNotifyFlags.JOINT_DOF_PROPERTIES)


def _mujoco_joint_friction_backend(env, env_ids, asset_cfg, friction_range, operation, distribution, shared_random):
    from mjlab.envs.mdp.dr import joint_friction as _mjlab_joint_friction

    adapter = _MujocoEnvAdapter(env)
    mjlab_cfg = _selector_to_mjlab_cfg(asset_cfg, adapter.scene)
    _mjlab_joint_friction(
        env=adapter,
        env_ids=env_ids,
        ranges=friction_range,
        asset_cfg=mjlab_cfg,
        operation=operation,
        shared_random=shared_random,
    )


# ══════════════════════════════════════════════════════════════════════
# randomize_encoder_bias  (MuJoCo only)
# ══════════════════════════════════════════════════════════════════════


def randomize_encoder_bias(
    env: World,
    env_ids: torch.Tensor,
    bias_range: tuple[float, float],
    asset_cfg: SceneEntitySelector = SceneEntitySelector(name="robot"),
) -> None:
    """Randomize joint encoder bias.

    MuJoCo-only — Genesis/Newton do not expose an encoder bias hook
    on their PD controllers and raise ``NotImplementedError``.
    """
    if len(env_ids) == 0:
        return

    if env.sim_type == "mujoco":
        from mjlab.envs.mdp.dr import encoder_bias as _mjlab_encoder_bias

        adapter = _MujocoEnvAdapter(env)
        mjlab_cfg = _selector_to_mjlab_cfg(asset_cfg, adapter.scene)
        _mjlab_encoder_bias(
            env=adapter,
            env_ids=env_ids,
            bias_range=bias_range,
            asset_cfg=mjlab_cfg,
        )
    else:
        raise NotImplementedError(
            f"randomize_encoder_bias is MuJoCo-only; sim_type={env.sim_type!r} "
            f"does not implement an encoder bias hook."
        )
