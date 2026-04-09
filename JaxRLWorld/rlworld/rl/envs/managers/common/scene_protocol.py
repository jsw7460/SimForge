"""SceneManagerProtocol — minimum cross-sim contract for scene managers.

This Protocol documents the **stable subset** of the scene manager
interface that is guaranteed by every backend (Newton / Genesis /
mjlab). Code that wants to stay sim-agnostic should rely only on the
attributes and methods listed here.

Anything *not* in this Protocol is intentionally sim-specific:

- ``model`` (Newton, mjlab) — physics model object, completely
  different shape on each backend.
- ``state`` / ``state_0`` / ``state_1`` (Newton) — warp State objects
  Newton uses for explicit state management. Genesis hides state
  inside ``gs.Scene`` and mjlab keeps it inside ``Simulation``.
- ``solver`` (Newton) — Newton-only solver handle.
- ``robot_view`` / ``robot_state`` / ``robot_state_writer`` (Newton) —
  Newton's ArticulationView abstraction. The closest cross-sim
  equivalent is ``RobotData`` / ``RobotStateWriter``, accessed via
  ``env.get_robot_data()`` / ``env.get_robot_state_writer()`` rather
  than the scene manager.
- ``scene`` (Genesis, mjlab) — backend-native scene wrapper.
- ``sim`` (mjlab) — mjlab Simulation handle.
- ``physics_dt`` (mjlab) — mjlab-only attribute; the active control
  timestep should come from ``env.control_dt`` instead.
- ``get_entity`` (mjlab) — mjlab-only convenience; Newton/Genesis
  callers index ``entities[name]`` directly because their entity
  storage shape differs.
- ``get_sensor`` (Newton, mjlab) — Genesis stores sensors in a
  3-level dict keyed by entity → link → sensor class, so a flat
  ``get_sensor(name)`` does not generalize.

Use this Protocol via ``isinstance(scene_manager, SceneManagerProtocol)``
to runtime-check that an object satisfies the cross-sim contract,
or as a type annotation in cross-sim helpers/event terms that should
not depend on a specific backend.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    import torch

    from rlworld.rl.configs.robots.kinematic_tree import KinematicTree


@runtime_checkable
class SceneManagerProtocol(Protocol):
    """Minimum cross-sim interface guaranteed by every scene manager.

    Concrete classes (NewtonSceneManager, Genesis SceneManager,
    MujocoSceneManager) satisfy this structurally — they do not need
    to inherit from it. Membership is checked by attribute presence,
    not nominal subtyping.
    """

    # ── Attributes ────────────────────────────────────────────────────

    entities: "dict[str, Any]"
    """Mapping ``entity_name → backend-native entity object`` (or wrapper).

    The value type is sim-specific; cross-sim code should treat the
    object as opaque or downcast to a known sim type.
    """

    sensors: "Any"
    """Sensor collection. Shape is sim-specific:

    - Newton: ``dict[str, Sensor]`` (flat by sensor name)
    - mjlab:  ``dict[str, Sensor]`` (flat by sensor name)
    - Genesis: ``dict[str, dict[str, dict[str, Sensor]]]`` keyed by
      ``entity_name → link_name → sensor_class_name``

    Cross-sim helpers should usually go through the contact manager
    (``env.contact_manager``) instead of poking this dict directly.
    """

    trees: "dict[str, KinematicTree]"
    """KinematicTree per entity, built by
    ``managers.common.scene_helpers.build_kinematic_trees``."""

    # ── Methods ───────────────────────────────────────────────────────

    def step(self) -> None:
        """Advance the physics by one control step.

        The internal step loop is sim-specific (Newton uses substeps +
        explicit state swap, Genesis calls ``scene.step()``, mjlab
        does write→step→update). Cross-sim code only needs to know
        that one call moves the simulation forward by ``env.control_dt``.
        """
        ...

    def reset(self, env_ids: "torch.Tensor | None" = None) -> None:
        """Reset the given environments to their initial state.

        Pass ``None`` to reset all environments. Genesis currently
        no-ops because its scene state lives inside ``gs.Scene`` and
        is reset by event terms instead — the method is kept for
        API symmetry.
        """
        ...

    def find_body_names(
        self, body_names: "list[str]", entity_name: str = "robot"
    ) -> "list[str]":
        """Resolve a list of regex patterns to concrete body names.

        The resolution preserves query order. Each backend reads its
        body name list from a different source (Newton: model.body_label,
        Genesis: entity link list, mjlab: entity.find_bodies) but the
        return shape is identical.
        """
        ...
