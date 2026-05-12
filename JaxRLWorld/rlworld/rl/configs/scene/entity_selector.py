"""Sim-agnostic selector pointing at an already-spawned entity (or its
joints / bodies / geoms / sites).

This is **not** the same as :class:`EntityCfg` in
:mod:`unified_entity_config` — that one is a *spawn spec* (recipe used
at scene-build time), whereas :class:`SceneEntitySelector` is a
*runtime pointer* used by event/reward/observation/termination terms to
address parts of an entity that already lives in the scene.

Pattern syntax: ``joint_names`` / ``body_names`` / ``geom_names`` /
``site_names`` / ``actuator_names`` accept regular expressions (the same
convention used by mjlab and IsaacLab — :func:`re.fullmatch` against
each candidate name).  ``"FR_foot_collision"`` matches exactly;
``".*foot.*"`` matches any name containing ``foot``;
``".*/FR_foot_collision"`` matches Newton's ``shape_label`` paths.

Lifecycle
---------
Presets put a :class:`SceneEntitySelector` (or rely on a term function's
``_DEFAULT_SELECTOR`` default) in a term's ``params``.  When the owning
manager is constructed it calls ``World.resolve_selector(selector)``
once and replaces the selector in ``params`` with the returned
:class:`ResolvedEntity` — so every term invocation receives a fully
resolved struct with no per-step resolution cost.  ``ResolvedEntity``
holds only ``name`` / index tensors / matched-name lists, so it stays
deep-copyable (the config is cloned for the eval env) and re-fetching the
backend entity is a cheap ``env.scene_manager[name]`` /
``env.get_robot_data(name)`` away.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class SceneEntitySelector:
    """Sim-agnostic pointer to an entity (and optional component subset)
    that has already been spawned in the scene.

    All ``*_names`` fields accept regex patterns matched with
    ``re.fullmatch`` — ``None`` means "all components of this kind".

    Distinct from :class:`EntityCfg` in :mod:`unified_entity_config`
    (spawn spec); this one is the runtime selector.
    """

    name: str = "robot"
    """Name of the entity in the scene (key into ``scene_manager.entities``)."""

    joint_names: tuple[str, ...] | None = None
    """Regex patterns matched against ``act_manager.actuated_joint_names``."""

    body_names: tuple[str, ...] | None = None
    """Regex patterns matched against the entity's body/link names."""

    geom_names: tuple[str, ...] | None = None
    """Regex patterns matched against geom/shape names. Genesis raises
    ``NotImplementedError`` because its geoms are unnamed; use
    :attr:`body_names` for Genesis."""

    site_names: tuple[str, ...] | None = None
    """Regex patterns matched against site names. Site is a MuJoCo
    concept; backends without sites raise ``NotImplementedError`` if
    this is set."""

    actuator_names: tuple[str, ...] | None = None
    """Regex patterns matched against actuator names. On Genesis and
    Newton actuators are 1:1 with :attr:`act_manager.actuated_joint_names`,
    so the resolver matches against the canonical joint list.  On
    MuJoCo this dispatches to mjlab's ``find_actuators``."""

    preserve_order: bool = False
    """When True the resolver preserves the order of the regex patterns
    in the output (``resolve_matching_names(preserve_order=True)``).
    Default False follows the entity's internal ordering — same default
    as mjlab/IsaacLab.  Some downstream code (e.g. gait-aligned site
    indices) requires a specific ordering and must opt in."""


@dataclass(eq=False)
class ResolvedEntity:
    """Sim-agnostic resolved view of a :class:`SceneEntitySelector`.

    Returned by ``World.resolve_selector``.  Reward / observation /
    termination / event functions consume this directly; sim-specific
    backends re-fetch the actual entity via ``env.scene_manager[name]``
    or ``env.get_robot_data(name)`` when they need to write to it.

    ``eq=False`` keeps the default identity-based ``__eq__`` / ``__hash__``
    — important because instances carry ``torch.Tensor`` fields (which are
    unhashable) yet must be hashable for ``@EnvStepCache``-decorated
    observation functions.  A selector is resolved once at setup and the
    same instance is reused every step, so the identity hash is stable.
    """

    name: str
    """Echo of the selector's entity name (key into ``scene_manager.entities``)."""

    joint_ids: torch.Tensor | None
    """Joint indices in **canonical** (``act_manager.actuated_joint_names``)
    order — use to slice ``RobotData.joint_pos`` / ``joint_vel`` /
    ``applied_torque``.  ``None`` when joints were not requested."""

    joint_ids_native: torch.Tensor | None
    """Joint indices in the **sim-native** ordering used by raw backend
    APIs (e.g. mjlab's ``robot.data.joint_pos`` columns).  ``None`` when
    not requested or not populated by the backend."""

    body_ids: torch.Tensor | None
    """Body / link indices in **sim-native** order.  ``None`` when bodies
    were not requested."""

    geom_ids: torch.Tensor | None
    """Geom / shape indices in **sim-native** order.  ``None`` when geoms
    were not requested or the backend has no named geoms (Genesis)."""

    site_ids: torch.Tensor | None
    """Site indices in **sim-native** order.  ``None`` when sites were not
    requested or the backend has no sites (Genesis, Newton)."""

    actuator_ids: torch.Tensor | None
    """Actuator indices.  On Genesis/Newton this equals :attr:`joint_ids`
    (canonical actuator==joint mapping); on MuJoCo it is mjlab's actuator
    id space.  ``None`` when actuators were not requested."""

    # ── Resolved names (matched against the entity's name list) ──────
    # Populated alongside the corresponding ``*_ids`` field so terms that
    # need name strings (e.g. mjlab-style accessors taking a name list)
    # read them directly without re-resolving.

    joint_names: list[str] | None = None
    """Joint names matched (canonical actuated-joint order)."""

    body_names: list[str] | None = None
    """Body / link names matched."""

    geom_names: list[str] | None = None
    """Geom / shape names matched."""

    site_names: list[str] | None = None
    """Site names matched (MuJoCo only)."""

    actuator_names: list[str] | None = None
    """Actuator names matched."""

    source_selector: SceneEntitySelector | None = None
    """The :class:`SceneEntitySelector` this was resolved from.  Backends
    that must re-derive a sim-native selector spec (e.g. the MuJoCo DR
    backends, which rebuild an mjlab ``SceneEntityCfg`` to delegate to
    ``mjlab.envs.mdp.dr.*``) read the original regex patterns from here."""

    extras: dict[str, Any] = field(default_factory=dict)
    """Per-backend escape hatch for the rare cases that need more than the
    fields above."""

    def __repr__(self) -> str:
        # The default dataclass repr would dump every tensor; keep it to a
        # one-liner so it doesn't wreck the env-config console panel that
        # prints reward/event ``params``.
        resolved = [
            f for f in ("joint_ids", "body_ids", "geom_ids", "site_ids", "actuator_ids") if getattr(self, f) is not None
        ]
        return f"ResolvedEntity(name={self.name!r}, resolved={resolved})"
