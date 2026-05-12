"""Sim-agnostic selector pointing at an already-spawned entity (or its
joints / bodies / geoms / sites).

This is **not** the same as :class:`EntityCfg` in
:mod:`unified_entity_config` — that one is a *spawn spec* (recipe used
at scene-build time), whereas :class:`SceneEntitySelector` is a
*runtime pointer* used by event/reward terms to address parts of an
entity that already lives in the scene.

Pattern syntax: ``joint_names`` / ``body_names`` / ``geom_names`` /
``site_names`` accept regular expressions (the same convention used by
mjlab and IsaacLab — :func:`re.fullmatch` against each candidate name).
``"FR_foot_collision"`` matches exactly; ``".*foot.*"`` matches any
name containing ``foot``; ``".*/FR_foot_collision"`` matches Newton's
``shape_label`` paths.

Resolution is performed by ``World.resolve_selector(selector)`` which
returns a :class:`ResolvedEntity` containing both canonical-order joint
indices (aligned with ``RobotData.joint_pos``) and any sim-native
indices needed by backend-level event terms.
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

    Returned by ``World.resolve_selector``.  Reward and event functions
    consume this struct directly, so they no longer need to know which
    backend they are running against.

    ``eq=False`` keeps the default identity-based ``__eq__`` / ``__hash__``
    — important because instances carry ``torch.Tensor`` fields (which are
    unhashable) yet need to be hashable for ``@EnvStepCache``-decorated
    observation functions.  Managers resolve a selector **once** at setup
    and stash the resulting instance in the term's ``params``, so the same
    object is reused every step and its identity hash is stable.
    """

    name: str
    """Echo of the selector's entity name."""

    backend_handle: Any
    """Sim-native entity handle (Genesis ``RigidEntity``, mjlab
    ``Entity``, Newton ``ArticulationView``).  Use only in backend-level
    code that already knows the sim type — common code should ignore."""

    joint_ids: torch.Tensor | None
    """Joint indices in **canonical** (``act_manager.actuated_joint_names``)
    order.  Use this to slice ``RobotData.joint_pos`` / ``joint_vel`` /
    ``applied_torque``.  ``None`` when the selector did not request
    joints (``selector.joint_names is None`` and the field is unused)."""

    joint_ids_native: torch.Tensor | None
    """Joint indices in the **sim-native** ordering (whatever the
    backend uses for raw API calls).  ``None`` when the backend has not
    populated this yet — friction / mass / geom-level DR does not need
    joints, so most backends can leave this unfilled in the PoC."""

    body_ids: torch.Tensor | None
    """Body/link indices in **sim-native** order.  ``None`` when the
    selector did not request bodies."""

    geom_ids: torch.Tensor | None
    """Geom indices in **sim-native** order.  ``None`` when the
    selector did not request geoms or the backend does not expose
    geoms (e.g. Genesis)."""

    site_ids: torch.Tensor | None
    """Site indices in **sim-native** order.  ``None`` when the
    selector did not request sites or the backend does not expose
    sites (Genesis, Newton)."""

    actuator_ids: torch.Tensor | None
    """Actuator indices.  On Genesis/Newton this equals
    :attr:`joint_ids` (canonical actuator==joint mapping).  On MuJoCo
    this is mjlab's actuator id space.  ``None`` when the selector
    did not request actuators."""

    # ── Resolved names (matched against the entity's name list) ──────
    # Populated alongside the corresponding ``*_ids`` field so reward /
    # event terms that need name strings (e.g. mjlab-style accessors
    # that take a list of names) can read them directly without
    # re-resolving.

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

    extras: dict[str, Any] = field(default_factory=dict)
    """Per-backend escape hatch.  Newton stores ``shape_ids`` here
    (the resolved per-shape indices that ``shape_material_mu`` and
    similar arrays are indexed by — distinct from collision geoms in
    mjlab/Genesis terminology)."""

    def __repr__(self) -> str:
        # The default dataclass repr would dump ``backend_handle``'s full
        # repr — for a Genesis ``RigidEntity`` that's a huge multi-line
        # dump (n_qs, n_dofs, n_geoms, ...) which wrecks any table that
        # prints reward/event ``params`` (e.g. the env-config console
        # panel). Keep it to a one-liner: name, backend type, and which
        # id fields were resolved.
        bh = type(self.backend_handle).__name__ if self.backend_handle is not None else "None"
        resolved = [
            f for f in ("joint_ids", "body_ids", "geom_ids", "site_ids", "actuator_ids") if getattr(self, f) is not None
        ]
        return f"ResolvedEntity(name={self.name!r}, backend={bh}, resolved={resolved})"
