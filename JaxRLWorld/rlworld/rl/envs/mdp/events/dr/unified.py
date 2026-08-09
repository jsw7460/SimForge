"""Sim-agnostic domain-randomization terms.

Each public function takes an ``asset_cfg`` selector — a preset writes a
:class:`~rlworld.rl.configs.scene.entity_selector.SceneEntitySelector`,
which the owning EventManager resolves once at setup into a
:class:`~rlworld.rl.configs.scene.entity_selector.ResolvedEntity` (the
``_DEFAULT_SELECTOR`` default is resolved the same way).  The function
then dispatches to a private per-sim backend based on ``env.sim_type``;
backends re-fetch the actual entity via ``env.scene_manager[...]``.

Currently exposed:

* :func:`randomize_friction`         — geom-level friction (slide/spin/roll)
* :func:`randomize_body_mass`        — per-body mass randomization
* :func:`randomize_body_com_offset`  — per-body center-of-mass offset
* :func:`randomize_pd_gains`         — joint PD controller gains (kp/kd)
* :func:`randomize_joint_armature`   — reflected rotor inertia
* :func:`randomize_joint_friction`   — joint Coulomb friction
* :func:`randomize_encoder_bias`     — joint encoder bias (sim-agnostic, obs-side)

These are the sole DR entry points presets should target.  Newton
keeps a couple of *non-randomised* fixed-value setters in
``events/dr/newton.py`` (``set_joint_friction`` / ``set_foot_friction``)
that have no cross-sim counterpart.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch

from rlworld.rl.configs.scene.entity_selector import ResolvedEntity, SceneEntitySelector
from rlworld.rl.envs.mdp.events.mujoco import _MujocoEnvAdapter

from ._model_fields import RecomputeLevel, requires_model_fields
from ._utils import apply_operation, sample

if TYPE_CHECKING:
    from rlworld.rl.envs.world import World


# NOTE on imports: ``warp``/``newton``/``mjlab`` are imported inside
# the per-backend helpers below.  pyproject.toml lists each simulator
# as an *optional* extra (``genesis``, ``newton``), so a user running
# with only one backend installed must still be able to import this
# module — top-level imports of the others would crash at import time.
# This mirrors the existing pattern in ``events/mujoco.py``.


def _newton_notify(env: World, flag) -> None:
    """Notify the Newton solver of a model-property change.

    Each Newton ``solver.notify_model_changed`` triggers a non-trivial GPU
    state refresh; when multiple DR terms fire in the same reset (e.g. body
    inertia + joint friction + foot friction), three sequential notifies
    add up. The event manager wraps the ``reset_dr`` loop with a deferred-
    flag attribute (``env._dr_pending_notify_flags``); when present, we OR
    flags into it instead of notifying, and the manager flushes a single
    combined notify after the loop. Outside that context the call is
    unchanged (immediate notify). Newton-only — no behavior change for
    Genesis / MuJoCo backends, which never reach this helper.
    """
    pending = getattr(env, "_dr_pending_notify_flags", None)
    flag_int = int(flag)
    if pending is None:
        env.scene_manager.solver.notify_model_changed(flag_int)
    else:
        env._dr_pending_notify_flags = pending | flag_int


# Sentinel: distinguishes "not inside a reset_dr batch" (immediate recompute)
# from "inside a batch, no recompute requested yet" (pending is None).
_MUJOCO_NO_BATCH = object()


def _mujoco_recompute(env: World, sim, level) -> None:
    """Recompute derived MuJoCo constants after a DR write (MuJoCo-only).

    ``sim.recompute_constants`` is model-GLOBAL and its most expensive level
    (``set_const``, needed for body_mass) dominates the reset path when several
    DR terms fire per reset — K1 has 2-3 body_mass terms + body_com, each of
    which would otherwise trigger its own full set_const. During a ``reset_dr``
    batch the event manager sets ``env._dr_pending_recompute_level``; we then
    keep only the MAX level (``RecomputeLevel`` is an ordered IntEnum whose
    higher levels recompute a superset of the lower), and the manager flushes a
    SINGLE recompute after all terms run. Outside that context the call is
    immediate, so nothing changes for non-batched callers. Mirrors the deferred
    pattern in :func:`_newton_notify`.
    """
    pending = getattr(env, "_dr_pending_recompute_level", _MUJOCO_NO_BATCH)
    if pending is _MUJOCO_NO_BATCH:
        sim.recompute_constants(level)
    elif pending is None:
        env._dr_pending_recompute_level = level
    else:
        env._dr_pending_recompute_level = max(pending, level)


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


@requires_model_fields("geom_friction")
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
            asset_cfg,
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
            asset_cfg,
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

    entity = env.scene_manager[resolved.name]
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
    from newton import ModelFlags

    view = env.scene_manager.articulation_views[resolved.name]
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
            raise ValueError(f"Unknown friction axis {axis}; valid axes are {sorted(_NEWTON_FRICTION_AXIS_ATTR)}.")
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

        if shape_indices is None:
            if operation == "abs":
                values[env_ids] = sampled
            else:
                defaults = getattr(env._dr_baselines, attr)
                values[env_ids] = apply_operation(defaults[env_ids], sampled, operation)
        else:
            env_grid, shape_grid = torch.meshgrid(env_ids, shape_indices, indexing="ij")
            if operation == "abs":
                for inner in range(n_axes_inner):
                    values[env_grid, inner, shape_grid] = sampled[:, inner, :]
            else:
                defaults = getattr(env._dr_baselines, attr)
                for inner in range(n_axes_inner):
                    base = defaults[env_grid, inner, shape_grid]
                    values[env_grid, inner, shape_grid] = apply_operation(base, sampled[:, inner, :], operation)

        view.set_attribute(attr, model, values)

    _newton_notify(env, ModelFlags.SHAPE_PROPERTIES)


# ──────────────────────────────────────────────────────────────────────
# MuJoCo backend (delegates to mjlab DR)
# ──────────────────────────────────────────────────────────────────────


def _mujoco_friction_backend(
    env: World,
    env_ids: torch.Tensor,
    asset_cfg: ResolvedEntity,
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


def _selector_to_mjlab_cfg(asset_cfg: SceneEntitySelector | ResolvedEntity, scene):
    """Build a resolved mjlab ``SceneEntityCfg`` from our selector.

    Accepts either a raw :class:`SceneEntitySelector` or a
    :class:`ResolvedEntity` (in which case the original selector is read
    from ``asset_cfg.source_selector``).  Used by every MuJoCo backend;
    mjlab import is local because the module-level rule says simulator
    deps are optional extras.
    """
    from mjlab.managers.scene_entity_config import SceneEntityCfg as _MjlabSceneEntityCfg

    if isinstance(asset_cfg, ResolvedEntity):
        asset_cfg = asset_cfg.source_selector

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


@requires_model_fields("body_mass", recompute=RecomputeLevel.set_const)
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
        _genesis_body_mass_backend(env, env_ids, asset_cfg, mass_range, operation, distribution)
    elif env.sim_type == "newton":
        _newton_body_mass_backend(env, env_ids, asset_cfg, mass_range, operation, distribution)
    elif env.sim_type == "mujoco":
        _mujoco_body_mass_backend(env, env_ids, asset_cfg, mass_range, operation, distribution, shared_random)
    else:
        raise NotImplementedError(f"randomize_body_mass has no backend for sim_type={env.sim_type!r}")


def _genesis_body_mass_backend(env, env_ids, resolved, mass_range, operation, distribution):
    if operation != "scale":
        raise NotImplementedError(
            f"Genesis body mass DR only supports operation='scale' (got {operation!r}); set_mass_shift is a multiplier."
        )
    entity = env.scene_manager[resolved.name]
    if resolved.body_ids is None:
        raise ValueError("Genesis randomize_body_mass requires asset_cfg.body_names; got selector with no body subset.")
    links_idx = resolved.body_ids.tolist()
    n_envs, n_links = len(env_ids), len(links_idx)
    ratios = sample((n_envs, n_links), *mass_range, env.device, distribution)
    # mass_shift = torch.zeros(n_envs, n_links, device=env.device)
    # for i, idx in enumerate(links_idx):
    #     original_mass = entity.links[idx].get_mass()
    #     mass_shift[:, i] = original_mass * (ratios[:, i] - 1.0)

    original_masses = entity.get_links_inertial_mass(links_idx_local=links_idx)
    mass_shift = original_masses * (ratios - 1.0)
    entity.set_mass_shift(mass_shift=mass_shift, links_idx_local=links_idx, envs_idx=env_ids)


def _newton_body_mass_backend(env, env_ids, resolved, mass_range, operation, distribution):
    import warp as wp
    from newton import ModelFlags

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
        view = env.scene_manager.articulation_views[resolved.name]
        names = [view.link_names[i] for i in resolved.body_ids.tolist()]
        body_indices = cache.get_body_indices(names)
        if hasattr(body_indices, "tolist"):
            body_indices = torch.tensor(body_indices, device=env.device, dtype=torch.long)

    mass = wp.to_torch(model.body_mass).reshape(env.num_envs, cache.bodies_per_env)
    n_bodies = len(body_indices)
    sampled = sample((len(env_ids), n_bodies), *mass_range, env.device, distribution)
    defaults = env._dr_baselines.body_mass
    mass[env_ids.unsqueeze(1), body_indices] = apply_operation(
        defaults[env_ids.unsqueeze(1), body_indices], sampled, operation
    )
    wp.copy(model.body_mass, wp.from_torch(mass.reshape(-1).contiguous(), dtype=wp.float32))
    _newton_notify(env, ModelFlags.BODY_INERTIAL_PROPERTIES)


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
    # mjlab's EventManager runs this after firing mass DR (derived inertial
    # constants like invweights go stale otherwise); we call the term
    # directly, so recompute here (deferred to one flush per reset_dr batch).
    _mujoco_recompute(env, adapter.sim, randomize_body_mass.recompute)


# ══════════════════════════════════════════════════════════════════════
# randomize_body_com_offset
# ══════════════════════════════════════════════════════════════════════


@requires_model_fields("body_ipos", recompute=RecomputeLevel.set_const)
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
        _genesis_body_com_offset_backend(env, env_ids, asset_cfg, ranges, operation)
    elif env.sim_type == "newton":
        _newton_body_com_offset_backend(env, env_ids, asset_cfg, ranges, operation)
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
    entity = env.scene_manager[resolved.name]
    if resolved.body_ids is None:
        raise ValueError("Genesis randomize_body_com_offset requires asset_cfg.body_names.")
    links_idx = resolved.body_ids.tolist()
    n_envs, n_links = len(env_ids), len(links_idx)
    # Read current i_pos_shift so non-target axes preserve prior EventTerm writes within the same
    # reset.  Genesis's set_COM_shift kernel writes the full 3-vec; without this read-modify-write
    # a follow-up EventTerm targeting another axis would clobber the prior one.  Entity has no
    # get_COM_shift wrapper, so go through solver.get_links_COM_shift with global link indices.
    links_global_idx = torch.as_tensor([entity._link_start + i for i in links_idx], device=env.device, dtype=torch.long)
    com_shift = entity._solver.get_links_COM_shift(links_idx=links_global_idx, envs_idx=env_ids)
    for axis, (lo, hi) in ranges.items():
        com_shift[:, :, axis] = torch.empty(n_envs, n_links, device=env.device).uniform_(lo, hi)
    entity.set_COM_shift(com_shift=com_shift, links_idx_local=links_idx, envs_idx=env_ids)


def _newton_body_com_offset_backend(env, env_ids, resolved, ranges, operation):
    import warp as wp
    from newton import ModelFlags

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
        view = env.scene_manager.articulation_views[resolved.name]
        names = [view.link_names[i] for i in resolved.body_ids.tolist()]
        body_indices = cache.get_body_indices(names)
        if hasattr(body_indices, "tolist"):
            body_indices = torch.tensor(body_indices, device=env.device, dtype=torch.long)

    body_com = wp.to_torch(model.body_com).reshape(env.num_envs, cache.bodies_per_env, 3)
    defaults = env._dr_baselines.body_com
    n_envs, n_bodies = len(env_ids), len(body_indices)
    original = defaults[:, body_indices, :][env_ids]
    # Write target axes only.  Writing the full 3-vec (original + zero-filled offsets) would set
    # non-target axes back to ``original`` and clobber a prior EventTerm that randomized a
    # different axis of the same field within the same reset.
    env_grid = env_ids.unsqueeze(1)
    for axis, (lo, hi) in ranges.items():
        sampled = torch.empty(n_envs, n_bodies, device=env.device).uniform_(lo, hi)
        body_com[env_grid, body_indices, axis] = original[..., axis] + sampled
    wp.copy(model.body_com, wp.from_torch(body_com.reshape(-1, 3).contiguous(), dtype=wp.vec3))
    _newton_notify(env, ModelFlags.BODY_INERTIAL_PROPERTIES)


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
    _mujoco_recompute(env, adapter.sim, randomize_body_com_offset.recompute)


# ══════════════════════════════════════════════════════════════════════
# randomize_pd_gains
# ══════════════════════════════════════════════════════════════════════


@requires_model_fields("actuator_gainprm", "actuator_biasprm")
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

    Two gain stores exist, and this term dispatches to whichever one the
    robot actually uses (mirrors IsaacLab's ``randomize_actuator_gains``):

    * **Explicit actuators** (IdealPD / DelayedPD / DCMotor / ActuatorNet):
      torque is computed in Python from the actuator's own
      ``stiffness``/``damping`` tensors on EVERY backend, and the sim-side
      PD store is zeroed/unused (Genesis force mode, Newton
      ``joint_target_mode=NONE``, mjlab ``BuiltinMotorActuatorCfg``).
      Randomizing the sim store would be a silent no-op, so the actuator
      tensors are randomized instead — sim-agnostic.
    * **Implicit actuators** (sim-internal PD): the sim's gain store is
      the live one, randomized via the per-backend path below. Genesis
      only supports ``operation="scale"`` there (no built-in ratio API on
      ``set_dofs_kp/kv`` so we sample a ratio and multiply).

    ``asset_cfg.joint_names`` selects a canonical actuated-joint SUBSET to
    randomize (e.g. arm joints only on a mobile manipulator); ``None``
    randomizes every actuated DOF. Honored on the explicit path (every
    backend) and the Genesis/Newton implicit paths; the mujoco IMPLICIT path
    raises instead (mjlab's dr.pd_gains randomizes whole actuator groups, so
    joint-level subsets cannot be expressed there).
    """
    if len(env_ids) == 0:
        return
    if kp_range is None and kd_range is None:
        return

    if env.act_manager.has_explicit_actuators:
        _explicit_pd_gains_backend(env, env_ids, asset_cfg, kp_range, kd_range, operation, distribution)
        return

    if env.sim_type == "genesis":
        _genesis_pd_gains_backend(env, env_ids, asset_cfg, kp_range, kd_range, operation, distribution)
    elif env.sim_type == "newton":
        _newton_pd_gains_backend(env, env_ids, asset_cfg, kp_range, kd_range, operation, distribution)
    elif env.sim_type == "mujoco":
        _mujoco_pd_gains_backend(env, env_ids, asset_cfg, kp_range, kd_range, operation, distribution)
    else:
        raise NotImplementedError(f"randomize_pd_gains has no backend for sim_type={env.sim_type!r}")


def _selected_joint_ids(env, asset_cfg) -> torch.Tensor | None:
    """Canonical actuated-joint indices selected by ``asset_cfg.joint_names``,
    or ``None`` when no subset was requested (= randomize all actuated DOFs).

    Subset detection keys on the ORIGINAL selector's ``joint_names``: the
    event manager resolves selectors before the term runs, and backends
    populate ``ResolvedEntity.joint_ids`` with the full actuated set even
    when no ``joint_names`` were given — so ``joint_ids`` alone cannot
    distinguish "all joints (default)" from "subset requested". Direct calls
    (diags, scripts) may pass a raw :class:`SceneEntitySelector`, which is
    resolved here on demand.
    """
    if isinstance(asset_cfg, SceneEntitySelector):
        if asset_cfg.joint_names is None:
            return None
        asset_cfg = env.resolve_selector(asset_cfg)
    if asset_cfg.source_selector.joint_names is None:
        return None
    return asset_cfg.joint_ids


def _slice_dofs(values: torch.Tensor, sel: torch.Tensor | None) -> torch.Tensor:
    """Select DOF columns from a per-DOF tensor of shape ``(n,)`` or ``(B, n)``."""
    if sel is None:
        return values
    return values[sel] if values.dim() == 1 else values[:, sel]


def _newton_dof_view(values: torch.Tensor) -> torch.Tensor:
    """``(num_envs, dof_count)`` zero-copy view of a Newton view attribute.

    ``ArticulationView.get_attribute`` returns joint-DOF attributes shaped
    ``(world_count, articulations_per_world, dof_count)``; our robot views hold
    exactly one articulation per world, so squeeze that axis. In-place writes
    through the returned view hit the original tensor (basic indexing), which
    is then written back via ``set_attribute``.
    """
    if values.dim() == 2:
        return values
    if values.dim() == 3 and values.shape[1] == 1:
        return values[:, 0]
    raise NotImplementedError(f"Unexpected Newton view attribute shape {tuple(values.shape)}")


def _genesis_dr_baseline(env, entity_name: str, param: str, getter) -> torch.Tensor:
    """Pre-DR value of a Genesis ``dofs_info`` parameter, captured on the
    term's first call (before any DR has touched it).

    Genesis ``set_dofs_kp/kv/armature`` are ABSOLUTE-set APIs writing the
    persistent ``dofs_info`` store, ``get_dofs_*`` reads back the last-set
    value, and nothing restores ``dofs_info`` on reset (episode resets only
    touch ``dofs_state``). Scaling the CURRENT value therefore multiplies
    ratios across resets into a log random-walk; always scale this captured
    baseline instead (counterpart of Newton's ``_dr_baselines`` snapshot).
    """
    key = (entity_name, param)
    cache = env._genesis_dr_baselines
    if key not in cache:
        cache[key] = torch.as_tensor(getter()).clone()
    return cache[key]


def _genesis_pd_gains_backend(env, env_ids, resolved, kp_range, kd_range, operation, distribution):
    if operation != "scale":
        raise NotImplementedError(
            f"Genesis pd_gains DR only supports operation='scale' "
            f"(got {operation!r}); set_dofs_kp/kv take absolute values."
        )
    entity = env.scene_manager[resolved.name]
    # Canonical joint subset -> Genesis local dof indices (None = all dofs).
    selected = _selected_joint_ids(env, resolved)
    sel_dofs = None if selected is None else env.act_manager.indexing.sim_indices[selected]
    sel_np = None if sel_dofs is None else sel_dofs.cpu().numpy()
    n_sel = entity.n_dofs if sel_dofs is None else len(sel_dofs)
    if kp_range is not None:
        base_kp = _slice_dofs(_genesis_dr_baseline(env, resolved.name, "kp", entity.get_dofs_kp), sel_dofs)
        ratios = sample((len(env_ids), n_sel), *kp_range, env.device, distribution)
        kp_new = (base_kp * ratios) if base_kp.dim() == 1 else (base_kp[env_ids] * ratios)
        entity.set_dofs_kp(kp=kp_new.cpu().numpy(), dofs_idx_local=sel_np, envs_idx=env_ids)
    if kd_range is not None:
        base_kv = _slice_dofs(_genesis_dr_baseline(env, resolved.name, "kv", entity.get_dofs_kv), sel_dofs)
        ratios = sample((len(env_ids), n_sel), *kd_range, env.device, distribution)
        kv_new = (base_kv * ratios) if base_kv.dim() == 1 else (base_kv[env_ids] * ratios)
        entity.set_dofs_kv(kv=kv_new.cpu().numpy(), dofs_idx_local=sel_np, envs_idx=env_ids)


def _newton_pd_gains_backend(env, env_ids, asset_cfg, kp_range, kd_range, operation, distribution):
    # Implicit actuators only (the explicit case is handled sim-agnostically in
    # randomize_pd_gains via _explicit_pd_gains_backend): Newton's internal PD
    # consumes joint_target_ke/kd, so randomize the solver store.
    import warp as wp
    from newton import ModelFlags

    view = env.scene_manager.robot_view
    model = env.scene_manager.model
    # Canonical joint subset -> per-env dof columns of the view attributes
    # (same mapping RobotData uses; None = all dofs).
    selected = _selected_joint_ids(env, asset_cfg)
    cols = None if selected is None else env.act_manager.indexing.newton_qd_indices[selected]
    notify = False
    for attr_name, value_range in (("joint_target_ke", kp_range), ("joint_target_kd", kd_range)):
        if value_range is None:
            continue
        values = wp.to_torch(view.get_attribute(attr_name, model))
        defaults = getattr(env._dr_baselines, attr_name)
        if cols is None:
            sampled = sample((len(env_ids),) + values.shape[1:], *value_range, env.device, distribution)
            if operation == "abs":
                values[env_ids] = sampled
            else:
                values[env_ids] = apply_operation(defaults[env_ids], sampled, operation)
        else:
            vals2d = _newton_dof_view(values)
            defs2d = _newton_dof_view(defaults)
            sampled = sample((len(env_ids), len(cols)), *value_range, env.device, distribution)
            if operation == "abs":
                vals2d[env_ids[:, None], cols[None, :]] = sampled
            else:
                vals2d[env_ids[:, None], cols[None, :]] = apply_operation(defs2d[env_ids][:, cols], sampled, operation)
        view.set_attribute(attr_name, model, values)
        notify = True
    if notify:
        _newton_notify(env, ModelFlags.JOINT_DOF_PROPERTIES)


def _explicit_pd_gains_backend(env, env_ids, asset_cfg, kp_range, kd_range, operation, distribution):
    """Randomize the per-env stiffness/damping of the EXPLICIT actuators — the
    gains that actually produce torque in ``actuator_pd.compute`` (the sim-side
    PD store is zeroed/unused in direct-torque mode). Sim-agnostic: the gains
    are plain torch tensors on the Python actuator objects, shared by all three
    backends (Genesis / Newton / mjlab route the computed torque through their
    own force API, but the gains live here). Each explicit actuator owns
    ``(num_envs, n_actuated_joints)`` stiffness/damping tensors; we edit them in
    place so the next step's torque reflects the new gains (no sim notify
    needed). The pre-DR gains are captured once per actuator so 'scale'/'add'
    do not compound across resets.
    """
    selected = _selected_joint_ids(env, asset_cfg)
    for actuator, joint_idx in env.act_manager.actuators:
        # Columns of this actuator group covered by the subset: group column j
        # drives canonical joint joint_idx[j].
        cols = None
        if selected is not None:
            cols = torch.nonzero(torch.isin(joint_idx, selected), as_tuple=True)[0]
            if cols.numel() == 0:
                continue
        if not hasattr(actuator, "_dr_base_stiffness"):
            actuator._dr_base_stiffness = actuator.stiffness.clone()
            actuator._dr_base_damping = actuator.damping.clone()
        for attr, base_attr, value_range in (
            ("stiffness", "_dr_base_stiffness", kp_range),
            ("damping", "_dr_base_damping", kd_range),
        ):
            if value_range is None:
                continue
            gains = getattr(actuator, attr)  # [num_envs, n_actuated_joints]
            base = getattr(actuator, base_attr)
            n_cols = gains.shape[1] if cols is None else cols.numel()
            sampled = sample((len(env_ids), n_cols), *value_range, env.device, distribution)
            if cols is None:
                gains[env_ids] = sampled if operation == "abs" else apply_operation(base[env_ids], sampled, operation)
            else:
                new = sampled if operation == "abs" else apply_operation(base[env_ids][:, cols], sampled, operation)
                gains[env_ids[:, None], cols[None, :]] = new


def _mujoco_pd_gains_backend(env, env_ids, asset_cfg, kp_range, kd_range, operation, distribution):
    from mjlab.envs.mdp.dr import pd_gains as _mjlab_pd_gains

    if kp_range is None or kd_range is None:
        raise NotImplementedError(
            "MuJoCo pd_gains requires both kp_range and kd_range (mjlab's dr.pd_gains has no None-skip option)."
        )
    if _selected_joint_ids(env, asset_cfg) is not None:
        # mjlab's dr.pd_gains selects whole actuator OBJECTS
        # (asset_cfg.actuator_ids), and one Builtin actuator config can span
        # several joints — a per-joint subset cannot be expressed through it.
        # Fail loudly rather than silently randomizing all joints. (Explicit
        # actuators support joint subsets on every backend.)
        raise NotImplementedError(
            "Joint-level subsets are not supported for pd-gain DR on the mujoco "
            "IMPLICIT path (mjlab randomizes whole actuator groups). Use explicit "
            "actuators or drop joint_names for full-set randomization."
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


@requires_model_fields("dof_armature", recompute=RecomputeLevel.set_const_0)
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

    Genesis enforces ``operation="scale"``.  ``asset_cfg.joint_names``
    selects a canonical actuated-joint subset (``None`` = all actuated
    DOFs); honored on every backend.
    """
    if len(env_ids) == 0:
        return

    if env.sim_type == "genesis":
        _genesis_armature_backend(env, env_ids, asset_cfg, armature_range, operation, distribution)
    elif env.sim_type == "newton":
        _newton_armature_backend(env, env_ids, asset_cfg, armature_range, operation, distribution)
    elif env.sim_type == "mujoco":
        _mujoco_armature_backend(env, env_ids, asset_cfg, armature_range, operation, distribution, shared_random)
    else:
        raise NotImplementedError(f"randomize_joint_armature has no backend for sim_type={env.sim_type!r}")


def _genesis_armature_backend(env, env_ids, resolved, armature_range, operation, distribution):
    if operation != "scale":
        raise NotImplementedError(f"Genesis joint_armature DR only supports operation='scale' (got {operation!r}).")
    entity = env.scene_manager[resolved.name]
    selected = _selected_joint_ids(env, resolved)
    sel_dofs = None if selected is None else env.act_manager.indexing.sim_indices[selected]
    n_sel = entity.n_dofs if sel_dofs is None else len(sel_dofs)
    base = _slice_dofs(_genesis_dr_baseline(env, resolved.name, "armature", entity.get_dofs_armature), sel_dofs)
    ratios = sample((len(env_ids), n_sel), *armature_range, env.device, distribution)
    arm_new = (base * ratios) if base.dim() == 1 else (base[env_ids] * ratios)
    entity.set_dofs_armature(
        armature=arm_new.cpu().numpy(),
        dofs_idx_local=None if sel_dofs is None else sel_dofs.cpu().numpy(),
        envs_idx=env_ids,
    )


def _newton_armature_backend(env, env_ids, asset_cfg, armature_range, operation, distribution):
    import warp as wp
    from newton import ModelFlags

    view = env.scene_manager.robot_view
    model = env.scene_manager.model
    selected = _selected_joint_ids(env, asset_cfg)
    cols = None if selected is None else env.act_manager.indexing.newton_qd_indices[selected]
    armature = wp.to_torch(view.get_attribute("joint_armature", model))
    defaults = env._dr_baselines.joint_armature
    if cols is None:
        sampled = sample((len(env_ids),) + armature.shape[1:], *armature_range, env.device, distribution)
        if operation == "abs":
            armature[env_ids] = sampled
        else:
            armature[env_ids] = apply_operation(defaults[env_ids], sampled, operation)
    else:
        vals2d = _newton_dof_view(armature)
        defs2d = _newton_dof_view(defaults)
        sampled = sample((len(env_ids), len(cols)), *armature_range, env.device, distribution)
        if operation == "abs":
            vals2d[env_ids[:, None], cols[None, :]] = sampled
        else:
            vals2d[env_ids[:, None], cols[None, :]] = apply_operation(defs2d[env_ids][:, cols], sampled, operation)
    view.set_attribute("joint_armature", model, armature)
    _newton_notify(env, ModelFlags.JOINT_DOF_PROPERTIES)


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
    _mujoco_recompute(env, adapter.sim, randomize_joint_armature.recompute)


# ══════════════════════════════════════════════════════════════════════
# randomize_joint_friction
# ══════════════════════════════════════════════════════════════════════


@requires_model_fields("dof_frictionloss")
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
    takes absolute values).  ``asset_cfg.joint_names`` selects a
    canonical actuated-joint subset (``None`` = all actuated DOFs);
    honored on every backend.
    """
    if len(env_ids) == 0:
        return

    if env.sim_type == "genesis":
        _genesis_joint_friction_backend(env, env_ids, asset_cfg, friction_range, operation, distribution)
    elif env.sim_type == "newton":
        _newton_joint_friction_backend(env, env_ids, asset_cfg, friction_range, operation, distribution)
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
    entity = env.scene_manager[resolved.name]
    selected = _selected_joint_ids(env, resolved)
    sel_dofs = None if selected is None else env.act_manager.indexing.sim_indices[selected]
    n_sel = entity.n_dofs if sel_dofs is None else len(sel_dofs)
    values = sample((len(env_ids), n_sel), *friction_range, env.device, distribution)
    entity.set_dofs_frictionloss(
        frictionloss=values.cpu().numpy(),
        dofs_idx_local=None if sel_dofs is None else sel_dofs.cpu().numpy(),
        envs_idx=env_ids,
    )


def _newton_joint_friction_backend(env, env_ids, asset_cfg, friction_range, operation, distribution):
    import warp as wp
    from newton import ModelFlags

    view = env.scene_manager.robot_view
    model = env.scene_manager.model
    selected = _selected_joint_ids(env, asset_cfg)
    cols = None if selected is None else env.act_manager.indexing.newton_qd_indices[selected]
    friction = wp.to_torch(view.get_attribute("joint_friction", model))
    defaults = env._dr_baselines.joint_friction
    if cols is None:
        sampled = sample((len(env_ids),) + friction.shape[1:], *friction_range, env.device, distribution)
        if operation == "abs":
            friction[env_ids] = sampled
        else:
            friction[env_ids] = apply_operation(defaults[env_ids], sampled, operation)
    else:
        vals2d = _newton_dof_view(friction)
        defs2d = _newton_dof_view(defaults)
        sampled = sample((len(env_ids), len(cols)), *friction_range, env.device, distribution)
        if operation == "abs":
            vals2d[env_ids[:, None], cols[None, :]] = sampled
        else:
            vals2d[env_ids[:, None], cols[None, :]] = apply_operation(defs2d[env_ids][:, cols], sampled, operation)
    view.set_attribute("joint_friction", model, friction)
    _newton_notify(env, ModelFlags.JOINT_DOF_PROPERTIES)


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
# randomize_joint_damping
# ══════════════════════════════════════════════════════════════════════


@requires_model_fields("dof_damping")
def randomize_joint_damping(
    env: World,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntitySelector = SceneEntitySelector(name="robot"),
    damping_range: tuple[float, float] = (0.0, 1.0),
    operation: Literal["abs", "scale", "add"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    shared_random: bool = False,
) -> None:
    """Randomize passive joint (viscous) damping — the model's ``dof_damping``.

    This is the PLANT's own velocity damping (the training MJCF's default-class
    ``damping`` of 3 legs / 2 arms), NOT the controller PD kd (that lives on the
    actuator and is randomized by :func:`randomize_pd_gains`). The two are
    separate velocity-opposing layers.

    With ``operation="abs"`` (the default) the sampled value REPLACES the model
    value, so ``damping_range=(0.0, 1.0)`` overrides the XML 3/2 with a per-env
    draw in [0, 1] — hedging the unknown real-robot joint damping. Genesis
    enforces ``operation="abs"`` (``set_dofs_damping`` takes absolute values).
    ``asset_cfg.joint_names`` selects a canonical actuated-joint subset
    (``None`` = all actuated DOFs); honored on every backend.
    """
    if len(env_ids) == 0:
        return

    if env.sim_type == "genesis":
        _genesis_joint_damping_backend(env, env_ids, asset_cfg, damping_range, operation, distribution)
    elif env.sim_type == "newton":
        _newton_joint_damping_backend(env, env_ids, asset_cfg, damping_range, operation, distribution)
    elif env.sim_type == "mujoco":
        _mujoco_joint_damping_backend(env, env_ids, asset_cfg, damping_range, operation, distribution, shared_random)
    else:
        raise NotImplementedError(f"randomize_joint_damping has no backend for sim_type={env.sim_type!r}")


def _genesis_joint_damping_backend(env, env_ids, resolved, damping_range, operation, distribution):
    if operation != "abs":
        raise NotImplementedError(
            f"Genesis joint_damping DR only supports operation='abs' "
            f"(got {operation!r}); set_dofs_damping takes absolute values."
        )
    entity = env.scene_manager[resolved.name]
    selected = _selected_joint_ids(env, resolved)
    sel_dofs = None if selected is None else env.act_manager.indexing.sim_indices[selected]
    n_sel = entity.n_dofs if sel_dofs is None else len(sel_dofs)
    values = sample((len(env_ids), n_sel), *damping_range, env.device, distribution)
    entity.set_dofs_damping(
        damping=values.cpu().numpy(),
        dofs_idx_local=None if sel_dofs is None else sel_dofs.cpu().numpy(),
        envs_idx=env_ids,
    )


def _newton_joint_damping_backend(env, env_ids, asset_cfg, damping_range, operation, distribution):
    import warp as wp
    from newton import ModelFlags

    view = env.scene_manager.robot_view
    model = env.scene_manager.model
    selected = _selected_joint_ids(env, asset_cfg)
    cols = None if selected is None else env.act_manager.indexing.newton_qd_indices[selected]
    damping = wp.to_torch(view.get_attribute("joint_damping", model))
    # abs replaces the value outright, so no baseline is needed (newton's DR
    # baseline snapshot does not capture joint_damping); only the scale/add
    # paths dereference it.
    if cols is None:
        sampled = sample((len(env_ids),) + damping.shape[1:], *damping_range, env.device, distribution)
        if operation == "abs":
            damping[env_ids] = sampled
        else:
            defaults = env._dr_baselines.joint_damping
            damping[env_ids] = apply_operation(defaults[env_ids], sampled, operation)
    else:
        vals2d = _newton_dof_view(damping)
        sampled = sample((len(env_ids), len(cols)), *damping_range, env.device, distribution)
        if operation == "abs":
            vals2d[env_ids[:, None], cols[None, :]] = sampled
        else:
            defs2d = _newton_dof_view(env._dr_baselines.joint_damping)
            vals2d[env_ids[:, None], cols[None, :]] = apply_operation(defs2d[env_ids][:, cols], sampled, operation)
    view.set_attribute("joint_damping", model, damping)
    _newton_notify(env, ModelFlags.JOINT_DOF_PROPERTIES)


def _mujoco_joint_damping_backend(env, env_ids, asset_cfg, damping_range, operation, distribution, shared_random):
    from mjlab.envs.mdp.dr import joint_damping as _mjlab_joint_damping

    adapter = _MujocoEnvAdapter(env)
    mjlab_cfg = _selector_to_mjlab_cfg(asset_cfg, adapter.scene)
    _mjlab_joint_damping(
        env=adapter,
        env_ids=env_ids,
        ranges=damping_range,
        asset_cfg=mjlab_cfg,
        operation=operation,
        shared_random=shared_random,
    )


# ══════════════════════════════════════════════════════════════════════
# randomize_tau_scale  (tanh actuator-saturation scale kappa)
# ══════════════════════════════════════════════════════════════════════


@requires_model_fields()
def randomize_tau_scale(
    env: World,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntitySelector = SceneEntitySelector(name="robot"),
    tau_scale_range: tuple[float, float] = (0.5, 1.0),
    operation: Literal["abs", "scale", "add"] = "scale",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
) -> None:
    """Randomize the tanh actuator-saturation scale kappa (``tau_scale``) per env.

    kappa is the asymptote of ``tau = kappa*tanh(tau_PD/kappa)`` — the effective
    max deliverable torque. It lives on the explicit Python PD actuator (like
    kp/kd), NOT in any sim model store, so this is sim-agnostic and needs no
    per-backend path or model-field expansion. ``operation="scale"`` multiplies
    the configured kappa (default = effort limit), so ``(0.5, 1.0)`` spans from
    strong torque decay (half the effort limit) to none (full limit), hedging
    the unknown real value. reset_dr-safe: the base kappa is captured once per
    actuator so scale/add do not compound across resets.
    """
    if len(env_ids) == 0:
        return
    if not env.act_manager.has_explicit_actuators:
        raise NotImplementedError(
            "randomize_tau_scale requires explicit actuators (kappa lives on the actuator, not a sim PD store)."
        )
    selected = _selected_joint_ids(env, asset_cfg)
    for actuator, joint_idx in env.act_manager.actuators:
        if not getattr(actuator, "_use_tau_scale", False):
            continue  # actuator has no tanh saturation configured → nothing to randomize
        cols = None
        if selected is not None:
            cols = torch.nonzero(torch.isin(joint_idx, selected), as_tuple=True)[0]
            if cols.numel() == 0:
                continue
        if not hasattr(actuator, "_dr_base_tau_scale"):
            actuator._dr_base_tau_scale = actuator.tau_scale.clone()
        kap = actuator.tau_scale  # [num_envs, n_actuated_joints]
        base = actuator._dr_base_tau_scale
        n_cols = kap.shape[1] if cols is None else cols.numel()
        sampled = sample((len(env_ids), n_cols), *tau_scale_range, env.device, distribution)
        if cols is None:
            kap[env_ids] = sampled if operation == "abs" else apply_operation(base[env_ids], sampled, operation)
        else:
            new = sampled if operation == "abs" else apply_operation(base[env_ids][:, cols], sampled, operation)
            kap[env_ids[:, None], cols[None, :]] = new


# ══════════════════════════════════════════════════════════════════════
# randomize_encoder_bias  (sim-agnostic, obs-side)
# ══════════════════════════════════════════════════════════════════════


@requires_model_fields()
def randomize_encoder_bias(
    env: World,
    env_ids: torch.Tensor,
    bias_range: tuple[float, float],
    asset_cfg: SceneEntitySelector = SceneEntitySelector(name="robot"),
) -> None:
    """Randomize per-joint encoder bias (sim-agnostic, obs-side).

    Samples ``bias ~ U(bias_range)`` per (env, joint) and writes to
    ``env.act_manager._encoder_bias`` via ``set_encoder_bias``. The biased
    value is consumed by the ``dof_pos_biased`` observation
    (``q + encoder_bias``); the underlying physics state is unchanged.

    To activate, the actor observation group must register
    ``dof_pos_biased`` (not plain ``dof_pos``); otherwise the bias has no
    effect. Critic typically keeps the unbiased ``dof_pos``.

    NOTE: This path bypasses mjlab's own ``entity.data.encoder_bias``
    (which mjlab subtracts from PD targets in
    ``Mjlab/src/mjlab/envs/mdp/actions/actions.py``). The two mechanisms
    are duals (obs-bias vs action-bias) but not identical; using the
    sim-agnostic obs-bias for all three sims gives consistent behavior
    across Genesis / Newton / mjlab.
    """
    if len(env_ids) == 0:
        return

    n_envs = len(env_ids)
    n_joints = env.act_manager._encoder_bias.shape[1]
    lo, hi = bias_range
    bias_samples = torch.empty(n_envs, n_joints, device=env.device).uniform_(lo, hi)
    env.act_manager.set_encoder_bias(bias_samples, env_ids=env_ids)
