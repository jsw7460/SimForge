"""Newton body/joint label canonicalization — leaf-name extraction.

Newton's MJCF loader stores labels as XPath hierarchies
(``g1_29dof/worldbody/pelvis/left_hip_pitch_link/left_hip_pitch_joint``);
URDF loader stores ``{entity}/{name}`` (``g1_29dof/left_hip_pitch_joint``).
We canonicalize both down to the bare leaf name
(``left_hip_pitch_joint``) so every downstream consumer — builder-time
site / PD-gain application, model-time ArticulationIndexing, body
cache, DR / reset / contact lookups — sees the same name format that
Newton's ``ArticulationView.link_names`` / ``joint_dof_names``
already expose. This matches IsaacLab's Newton integration, which
feeds bare names from ArticulationView directly into user-facing
config.

Prefix-based multi-robot namespacing is replaced by per-entity
``ArticulationView`` instances that filter by ``body_label_prefix``
at view-construction time, so joint / body name collisions between
robots are resolved by which view you query, not by mangling the
names themselves.
"""

from __future__ import annotations

import re


def leaf_name(label: str) -> str:
    """Return the leaf segment of a slash-delimited Newton label.

    Mirrors Newton ``ArticulationView.get_name_from_label`` (selection.py:374).

    Examples:
        ``g1_29dof/worldbody/pelvis/.../left_hip_pitch_joint`` → ``left_hip_pitch_joint``
        ``g1_29dof/left_hip_pitch_joint`` → ``left_hip_pitch_joint``
        ``floating_base`` → ``floating_base``
    """
    return label.rsplit("/", maxsplit=1)[-1]


# Bare-leaf-name characters — anything outside this set (``*?[]`` for glob,
# ``.+()|`` for regex, ``/`` for path segments) signals an intentional pattern
# that the caller built themselves, and we must leave it alone.
_BARE_LEAF_NAME_RE = re.compile(r"^[A-Za-z0-9_\-]+$")


def as_leaf_glob(pattern: str) -> str:
    """Widen a bare leaf-name body pattern to ``*/<name>``.

    Newton's sensor / selection APIs match patterns against the full
    hierarchical ``model.body_label`` via ``fnmatch``. URDF loading
    produces flat labels (``<entity>/<name>``), MJCF loading produces
    XPath-style labels (``<entity>/<ancestor>/.../<name>``). A bare
    leaf-name pattern like ``"left_ankle_roll_link"`` matches the URDF
    layout but silently misses the MJCF layout. Wrapping it as
    ``"*/left_ankle_roll_link"`` matches both.

    This is a one-way widening: patterns that already contain a path
    separator or any glob (``*?[]``) / regex (``.+()|``) metacharacter
    pass through unchanged, so it never narrows a pattern the caller
    deliberately scoped.

    Examples:
        ``"left_ankle_roll_link"`` → ``"*/left_ankle_roll_link"``
        ``"g1/left_ankle_roll_link"`` → ``"g1/left_ankle_roll_link"`` (scoped)
        ``"*"`` → ``"*"`` (already a glob)
        ``"(left|right)_foot.*"`` → unchanged (regex metacharacters)
    """
    if not pattern:
        return pattern
    if "/" in pattern:
        return pattern
    if not _BARE_LEAF_NAME_RE.fullmatch(pattern):
        return pattern
    return f"*/{pattern}"


def as_leaf_globs(patterns):
    """Apply :func:`as_leaf_glob` element-wise. Accepts ``str`` / ``list`` / ``None``."""
    if patterns is None:
        return None
    if isinstance(patterns, str):
        return as_leaf_glob(patterns)
    return [as_leaf_glob(p) for p in patterns]
