"""Generic viser panel for any :class:`CommandTerm` exposing a UI spec.

The panel is fully driven by the term's ``get_ui_spec()`` declaration —
this module has no per-term knowledge. The viewer instantiates one
panel per term, the panel builds folder/slider/preset widgets from the
spec, and on every sim tick it pins the followed env's command columns
to the slider values via the column-wise ``set_command`` API.

Override scope: only the env currently followed by the camera is
overridden. All other envs keep auto-resampling normally, so the
training-distribution agents on screen are undisturbed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import viser

from rlworld.rl.envs.managers.common.command_ui import (
    CommandTermUISpec,
    GroupControl,
    PresetButton,
    SliderControl,
    iter_sliders,
)

if TYPE_CHECKING:
    from rlworld.rl.envs.managers.common.command_term import CommandTerm


def _padded(low: float, high: float, pad_factor: float) -> tuple[float, float]:
    """Pad ``[low, high]`` outward by ``pad_factor * (high - low)`` per side.

    Lets the user explore beyond the training distribution. Degenerate
    ranges (low == high) get a symmetric ±0.5 pad so the slider has a
    usable extent.
    """
    if high <= low:
        return low - 0.5, high + 0.5
    pad = pad_factor * (high - low)
    return low - pad, high + pad


class ViserCommandPanel:
    """One interactive panel bound to a single :class:`CommandTerm`.

    Lifecycle:
        ``__init__`` → builds widgets under the supplied GUI parent.
        ``apply(env_idx)`` → call every sim tick from the viewer.
        ``on_env_switch(old_idx, new_idx)`` → call when the camera
            switches the followed env.
        ``cleanup()`` → release any externally-controlled columns on
            shutdown.
    """

    def __init__(
        self,
        server: viser.ViserServer,
        term_name: str,
        term: CommandTerm,
        spec: CommandTermUISpec,
    ):
        """Build the panel under the current viser GUI context.

        The caller must already be inside an ``add_tab`` / ``add_folder``
        ``with`` block — newly created GUI elements auto-nest under the
        active viser context, so we don't pass an explicit parent.
        """
        self._server = server
        self._term_name = term_name
        self._term = term
        self._spec = spec
        # Track every column the spec exposes — used as the ``columns=``
        # selector when locking/releasing the followed env. We never lock
        # columns that the spec doesn't expose.
        self._exposed_columns: tuple[str, ...] = tuple(c.column for c in iter_sliders(spec.controls))
        # Validate up-front; misconfigured specs are loud bugs, not
        # silent no-ops at apply() time.
        for col in self._exposed_columns:
            if col not in term.column_names:
                raise ValueError(
                    f"CommandTerm {term_name!r} ({type(term).__name__}) declares slider for "
                    f"column {col!r}, which is not in column_names={term.column_names!r}."
                )

        self._slider_handles: dict[str, Any] = {}
        self._manual_cb = None
        # The env we currently hold a lock on (or None if no lock held).
        # Updated by :meth:`apply` and :meth:`on_env_switch`.
        self._locked_env: int | None = None
        self._build()

    # ── Widget construction ────────────────────────────────────────

    def _build(self) -> None:
        with self._server.gui.add_folder(self._spec.section_label):
            if self._spec.enable_manual_toggle:
                self._manual_cb = self._server.gui.add_checkbox(
                    "Manual override (followed env only)",
                    initial_value=False,
                    hint=(
                        "When ON, pin this term's command for the env "
                        "currently followed by the camera. Other envs "
                        "keep auto-resampling — the training-time "
                        "distribution on screen is preserved."
                    ),
                )
            self._build_controls(self._spec.controls)
            if self._spec.zero_button:
                zero_btn = self._server.gui.add_button("Zero all", icon=viser.Icon.PLAYER_STOP)

                @zero_btn.on_click
                def _(_):
                    for s in self._slider_handles.values():
                        s.value = 0.0

    def _build_controls(self, controls: tuple) -> None:
        for c in controls:
            if isinstance(c, GroupControl):
                with self._server.gui.add_folder(c.label):
                    self._build_controls(c.children)
            elif isinstance(c, SliderControl):
                low, high = _padded(c.low, c.high, c.pad_factor)
                # Initial value clamped to padded range so very narrow
                # cfg ranges (e.g. duration_range=(0.5, 0.5)) still
                # surface a usable initial slider position.
                initial = float(min(max(c.initial, low), high))
                label = f"{c.label} [{c.unit}]" if c.unit else c.label
                handle = self._server.gui.add_slider(
                    label,
                    min=low,
                    max=high,
                    step=c.step,
                    initial_value=initial,
                )
                self._slider_handles[c.column] = handle
            elif isinstance(c, PresetButton):
                btn = self._server.gui.add_button(c.label)

                @btn.on_click
                def _(_, values=c.values):
                    for col, val in values.items():
                        if col in self._slider_handles:
                            self._slider_handles[col].value = float(val)
            else:
                raise TypeError(f"Unknown control type: {type(c).__name__}")

    # ── Per-tick hook (called from play_viewer) ────────────────────

    def apply(self, env_idx: int) -> None:
        """Pin (or release) the followed env's columns based on UI state.

        Called once per sim tick from the viewer. Cheap when manual
        override is off — exits before touching any tensor.
        """
        manual_on = self._manual_cb is not None and self._manual_cb.value

        if not manual_on:
            if self._locked_env is not None:
                self._release(self._locked_env)
                self._locked_env = None
            return

        # Manual mode is active. If we previously locked a DIFFERENT
        # env (e.g. the user switched envs without going through
        # ``on_env_switch``), drop the stale lock first.
        if self._locked_env is not None and self._locked_env != env_idx:
            self._release(self._locked_env)
            self._locked_env = None

        self._lock_followed(env_idx)
        self._locked_env = env_idx

    def on_env_switch(self, old_idx: int, new_idx: int) -> None:
        """Re-bind the lock to the new env (called by play_viewer).

        Idempotent: safe to call even when ``new_idx == old_idx`` or
        manual mode is off.
        """
        if self._locked_env is None:
            return
        if self._locked_env == old_idx:
            self._release(old_idx)
        # Re-locking on ``new_idx`` happens on the next ``apply`` tick.
        self._locked_env = None

    def cleanup(self) -> None:
        """Release any held lock — call on viewer shutdown."""
        if self._locked_env is not None:
            self._release(self._locked_env)
            self._locked_env = None

    # ── Helpers ────────────────────────────────────────────────────

    def _lock_followed(self, env_idx: int) -> None:
        env_ids = torch.tensor([env_idx], dtype=torch.long, device=self._term.device)
        values = torch.tensor(
            [[float(self._slider_handles[col].value) for col in self._exposed_columns]],
            dtype=torch.float32,
            device=self._term.device,
        )
        self._term.set_command(env_ids, values, columns=self._exposed_columns)

    def _release(self, env_idx: int) -> None:
        env_ids = torch.tensor([env_idx], dtype=torch.long, device=self._term.device)
        self._term.release_command(env_ids, columns=self._exposed_columns)
