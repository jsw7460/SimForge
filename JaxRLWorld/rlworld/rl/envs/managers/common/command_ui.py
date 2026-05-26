"""UI declaration dataclasses for interactive CommandTerm visualization.

A ``CommandTerm`` returns a :class:`CommandTermUISpec` from
``get_ui_spec()`` to declare which knobs it wants exposed in an
interactive viewer. The viewer (e.g. viser) reads the spec and builds
generic widgets — it does not need to know about the concrete term
subclass.

This module is intentionally backend-agnostic:
  * No torch / numpy / viser imports.
  * Pure dataclasses + literal column references.

This keeps the term ↔ viewer contract narrow and lets future
front-ends (web UI, TUI, mjviewer) reuse the same declarations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union


@dataclass(frozen=True)
class SliderControl:
    """A scalar slider bound to one command column.

    The viewer writes the slider's current value into the term's
    command column identified by ``column``. The viewer is also
    responsible for calling ``set_command`` / ``release_command`` with
    the appropriate ``columns=`` selector when the user toggles the
    slider on/off.
    """

    column: str
    """Must match an entry in ``term.column_names``."""

    label: str
    """Human-readable label shown next to the slider."""

    low: float
    high: float
    step: float = 0.05
    initial: float = 0.0
    unit: str = ""
    """Optional unit string appended to the label (e.g. ``"m/s"``)."""

    pad_factor: float = 0.5
    """Fraction of ``(high - low)`` to pad beyond [low, high] on each
    side so the user can explore values outside the training
    distribution. ``0.0`` disables padding."""


@dataclass(frozen=True)
class PresetButton:
    """One-click button that snaps a set of columns to fixed values.

    Pressing the button updates the corresponding slider widgets (and
    therefore the locked command columns on the next tick). Useful for
    discrete modes — e.g. snapping a gait command to "trot" / "pace".
    """

    label: str
    values: dict[str, float]
    """Mapping ``column_name -> value``. Each column must already be
    declared by a :class:`SliderControl` in the same spec."""


@dataclass(frozen=True)
class GroupControl:
    """Folder grouping for visual organization in the viewer."""

    label: str
    children: tuple[Control, ...] = field(default_factory=tuple)


Control = Union[SliderControl, PresetButton, GroupControl]


@dataclass(frozen=True)
class CommandTermUISpec:
    """Top-level declaration for one CommandTerm's interactive UI.

    A term returning ``None`` from ``get_ui_spec()`` means "no
    interactive knobs"; the viewer may still display the current
    command vector read-only.
    """

    section_label: str
    """Header shown above the panel (e.g. ``"Velocity Command"``)."""

    controls: tuple[Control, ...]
    """Flat list of top-level controls. Use :class:`GroupControl` to
    nest sliders into named folders."""

    enable_manual_toggle: bool = True
    """If True, render a top-level "Manual override" checkbox that
    gates whether the sliders actually drive the term."""

    zero_button: bool = True
    """If True, render a "Zero" button that resets every slider in
    this spec to ``0.0``."""


def iter_sliders(controls: tuple[Control, ...]):
    """Yield every :class:`SliderControl` nested under ``controls``.

    Order matches the declared tree (depth-first), which is also the
    order the viewer should render widgets.
    """
    for c in controls:
        if isinstance(c, SliderControl):
            yield c
        elif isinstance(c, GroupControl):
            yield from iter_sliders(c.children)
        # PresetButton: not a slider, skipped here.
