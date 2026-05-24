"""Simulator-agnostic contact sensor configuration.

Mirrors mjlab's ``ContactSensorCfg`` / ``ContactMatch`` so the *same*
config object expresses the *same* intent across mjlab, Genesis, and
Newton. Per-simulator backend adapters
(``managers/genesis/contact_sensor.py``, the Newton equivalent, the
mjlab equivalent) translate this declarative config into their native
sensor / contact API.

This module deliberately does not import anything from mjlab — the
config classes here are standalone copies, kept structurally compatible
so a config written against this module reads the same as one written
against mjlab's ``mjlab.sensor.ContactSensorCfg``.

Backend support matrix
----------------------
========================  ======  =========  ========
field / value             mjlab   Genesis    Newton
========================  ======  =========  ========
primary mode="body"        yes     yes        yes
primary mode="geom"        yes     no         yes
primary mode="subtree"     yes     no         no
primary pattern/exclude    yes     yes        yes
secondary (entity-scoped)  yes     yes        yes
fields={"found","force"}   yes     yes        yes
fields w/ torque/dist/...  yes     no         no
reduce="netforce"          yes     yes        yes
reduce="none/mindist/..."  yes     no         no
num_slots > 1              yes     no         no
global_frame               yes     yes        yes (already world)
history_length > 0         yes     yes        yes
========================  ======  =========  ========

Air-time / contact-time accumulators are maintained by
``BaseContactManager`` for every group on every backend, so they are
always available (there is no opt-in flag).

The Genesis / Newton backends raise ``NotImplementedError`` for the
combinations they do not support (Genesis: ``primary.mode`` of
``"geom"`` / ``"subtree"``; both: ``reduce`` other than ``netforce``,
``num_slots > 1``, fields beyond ``{"found", "force"}``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class ContactMatch:
    """Specifies what to match on one side of a contact pair.

    Args:
        mode: Element type to match against — ``"geom"`` or ``"body"``.
            (``"subtree"`` is mjlab-only; the Genesis / Newton backends
            reject it.)
        pattern: A regex (or tuple of regexes) matched against element
            names within ``entity``. If ``entity`` is ``None`` / ``""``
            the pattern is taken as a literal element name (no regex
            expansion).
        entity: Entity name to scope the pattern to. ``None`` / ``""``
            means the pattern is a literal name. For ``secondary`` this
            is how you say "the counterpart must belong to this entity"
            (e.g. ``entity="terrain"`` → only contacts with the ground
            count). Two sentinels are accepted: ``"self"`` means "the
            counterpart must be another link of the same entity"
            (self-collision); ``"terrain"`` resolves to the
            ``TerrainImporter``-owned ground (which is not in the
            ``entities`` dict).
        exclude: Names to filter out of the match (blacklist). Each
            entry is treated as a regex if it contains regex
            metacharacters, otherwise as an exact name.
    """

    mode: Literal["geom", "body", "subtree"]
    pattern: str | tuple[str, ...]
    entity: str | None = None
    exclude: tuple[str, ...] = ()


@dataclass
class ContactSensorCfg:
    """Simulator-agnostic contact sensor configuration.

    A contact sensor watches contacts between a ``primary`` set of
    elements (e.g. the robot's feet) and an optional ``secondary``
    element (e.g. the terrain). ``primary`` patterns typically expand
    to multiple elements; each becomes one column on the per-contact
    axis of the produced tensors. The sensor is registered under
    ``name`` as a contact group in the env's ``ContactManager``, so
    reward / observation terms read it via
    ``env.contact_manager.is_contact(name)`` etc.

    Args:
        name: Group name. Used as the ``ContactManager`` group key.
        primary: Elements to measure (e.g. the robot's feet). Usually a
            regex resolving to several elements.
        secondary: Filter on what the primary may contact. ``None``
            means "any contact with a primary counts". When
            ``secondary.entity`` names an entity, only contacts with
            that entity count (whitelist). The backends translate this
            into their native filter — for Genesis this means inverting
            it into the native blacklist ``filter_link_idx`` (= all
            links not belonging to the secondary entity).
        fields: Contact quantities to extract. The Genesis / Newton
            backends support ``{"found", "force"}`` only; mjlab also
            supports ``torque`` / ``dist`` / ``pos`` / ``normal`` /
            ``tangent``.
        reduce: How to collapse simultaneous contacts on one primary
            into ``num_slots`` representatives. ``"netforce"`` (sum all
            contacts into a single net wrench) is the only mode the
            Genesis / Newton backends support; mjlab also offers
            ``"none"`` / ``"mindist"`` / ``"maxforce"``.
        num_slots: Contacts retained per primary after reduction. Must
            be ``1`` on the Genesis / Newton backends (and is the
            sensible default everywhere — pattern expansion already
            gives many primaries).
        global_frame: Report ``force`` in the global (world) frame
            rather than the contact / link-local frame. Newton already
            reports world-frame contact force; Genesis rotates the
            link-local force by the link orientation.
        history_length: If ``> 0``, keep a rolling buffer of the last
            N substeps of ``force``. Set this to your decimation value
            so the buffer covers exactly one policy step — lets
            ``penalize_contact_force_count`` (collision penalty) catch
            brief contacts that resolve mid-substep. ``0`` disables it.
    """

    name: str
    primary: ContactMatch
    secondary: ContactMatch | None = None
    fields: tuple[str, ...] = ("found", "force")
    reduce: Literal["none", "mindist", "maxforce", "netforce"] = "netforce"
    num_slots: int = 1
    global_frame: bool = False
    history_length: int = 0

    def __post_init__(self) -> None:
        if not self.fields:
            raise ValueError(f"ContactSensorCfg {self.name!r}: fields must be non-empty")
        if self.num_slots < 1:
            raise ValueError(f"ContactSensorCfg {self.name!r}: num_slots must be >= 1")
