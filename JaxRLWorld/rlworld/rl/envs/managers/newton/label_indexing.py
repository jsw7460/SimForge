"""Per-entity Newton label resolver.

This is the **Newton-only** counterpart to the cross-sim
:class:`~rlworld.rl.envs.indexing.ArticulationIndexing` (which maps
*joint* order canonical ↔ sim). Here we deal with **body / shape
labels**: Newton stores them in one flat, world-major list per
``model.body_label`` / ``model.shape_label``, and downstream consumers
(``newton.sensors.SensorContact``) want a ``list[int]`` of *model-global*
indices ready to pass through.

Why a separate class
~~~~~~~~~~~~~~~~~~~~
Genesis and mjlab each expose per-entity body/shape views at the
framework level (``entity.link_start:link_end`` / ``entity.body_ids``),
so the (entity, pattern) → indices resolution is a one-liner there.
Newton's flat-list layout has no such per-entity view at the level
:class:`newton.sensors.SensorContact` wants, so we keep a thin
JaxRLWorld helper that scans the flat label arrays **once at
scene-build time** and serves all subsequent (entity, pattern) queries
from the cached slice.

Two construction modes
~~~~~~~~~~~~~~~~~~~~~~
* ``from_articulation`` — the entity is replicated world-major across
  envs (every robot). The label slice is selected via
  ``body_label_prefix`` (auto-extracted by
  :class:`~rlworld.rl.envs.managers.newton.scene.NewtonSceneManager`).
* ``from_singleton`` — the entity is a one-off, world=-1 attachment
  (the :class:`~rlworld.rl.terrains.TerrainImporter`-owned
  ``"ground_plane"`` shape). The label slice is selected via a
  user-supplied predicate.

Query API
~~~~~~~~~
:meth:`find_bodies` / :meth:`find_shapes` take ``patterns`` (regex,
fullmatch against the leaf segment) plus ``exclude`` (regex, search
against the leaf) and return the matched **model-global** indices,
world-major. The result is suitable for
:class:`newton.sensors.SensorContact`'s ``sensing_obj_*`` /
``counterpart_*`` kwargs.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rlworld.rl.envs.utils.newton.label import leaf_name
from rlworld.rl.utils import string as string_utils

if TYPE_CHECKING:
    import newton


@dataclass
class _LabelPool:
    """One entity's slice of ``model.{body,shape}_label``.

    Already entity-scoped (by prefix or predicate) and world-major
    sorted. For an articulated entity with ``num_envs`` worlds and
    ``n_per_world`` labels per world, the layout is
    ``[(w0, slot0), (w0, slot1), ..., (w1, slot0), ...]``. For a
    singleton (world=-1) entity, the whole pool IS world 0.
    """

    ids: list[int]
    """Global ``model.{body,shape}_label`` indices, world-major sorted."""

    leaves: list[str]
    """Leaf names parallel to :attr:`ids` (last ``/``-separated token)."""

    n_per_world: int
    """Per-world slot count. Equals ``len(ids)`` for singletons (no replication)."""


@dataclass
class NewtonLabelIndexing:
    """Resolve ``(patterns, exclude)`` to model-global body/shape indices.

    See module docstring for context. Construct via
    :meth:`from_articulation` or :meth:`from_singleton`; query via
    :meth:`find_bodies` / :meth:`find_shapes`.
    """

    name: str
    """Entity name this index belongs to (for error messages)."""

    bodies: _LabelPool
    """Body-label pool. May be empty (``ids == []``) when the entity
    contributes no bodies — e.g. a heightfield terrain whose contact
    surface is purely a shape on the world body."""

    shapes: _LabelPool
    """Shape-label pool. May be empty when the entity contributes no
    shapes (rare)."""

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    @classmethod
    def from_articulation(
        cls,
        *,
        name: str,
        model: newton.Model,
        prefix: str | None,
        num_envs: int,
    ) -> NewtonLabelIndexing:
        """Build an index for a per-env-replicated entity.

        Scans ``model.body_label`` / ``model.shape_label`` once; keeps
        the entries whose label equals ``prefix`` or starts with
        ``prefix + "/"``. Verifies the scoped count divides evenly by
        ``num_envs`` (Newton's world-major replication invariant).
        """
        bodies = _scan_articulation_pool(
            labels=list(model.body_label),
            prefix=prefix,
            num_envs=num_envs,
            kind="body",
            entity_name=name,
        )
        shapes = _scan_articulation_pool(
            labels=list(model.shape_label),
            prefix=prefix,
            num_envs=num_envs,
            kind="shape",
            entity_name=name,
        )
        return cls(name=name, bodies=bodies, shapes=shapes)

    @classmethod
    def from_singleton(
        cls,
        *,
        name: str,
        model: newton.Model,
        body_label_predicate: Callable[[str], bool] | None = None,
        shape_label_predicate: Callable[[str], bool] | None = None,
    ) -> NewtonLabelIndexing:
        """Build an index for a world=-1 singleton entity (e.g. terrain).

        ``body_label_predicate`` / ``shape_label_predicate`` decide which
        flat-list entries belong to this singleton. ``None`` means the
        respective pool is empty.
        """
        bodies = _scan_singleton_pool(list(model.body_label), body_label_predicate)
        shapes = _scan_singleton_pool(list(model.shape_label), shape_label_predicate)
        return cls(name=name, bodies=bodies, shapes=shapes)

    # ------------------------------------------------------------------
    # query
    # ------------------------------------------------------------------

    def find_bodies(self, patterns: Sequence[str], exclude: Sequence[str] = ()) -> list[int]:
        """Return ``model.body_label`` global indices matching ``patterns``."""
        return self._find(pool=self.bodies, kind="body", patterns=patterns, exclude=exclude)

    def find_shapes(self, patterns: Sequence[str], exclude: Sequence[str] = ()) -> list[int]:
        """Return ``model.shape_label`` global indices matching ``patterns``."""
        return self._find(pool=self.shapes, kind="shape", patterns=patterns, exclude=exclude)

    def _find(
        self,
        *,
        pool: _LabelPool,
        kind: str,
        patterns: Sequence[str],
        exclude: Sequence[str],
    ) -> list[int]:
        if not pool.ids:
            raise ValueError(
                f"NewtonLabelIndexing[{self.name!r}]: entity contributes no {kind} labels — "
                f"cannot resolve pattern {list(patterns)!r}."
            )

        # Validate patterns against world-0 leaves (the canonical per-env
        # name set). resolve_matching_names raises if any pattern matches
        # nothing — caller behaviour matches the legacy resolver.
        world0_leaves = pool.leaves[: pool.n_per_world]
        _, matched_leaves = string_utils.resolve_matching_names(list(patterns), world0_leaves, preserve_order=True)
        if exclude:
            exclude_tuple = tuple(exclude)
            matched_leaves = [n for n in matched_leaves if not _matches_any_search(n, exclude_tuple)]
        if not matched_leaves:
            raise ValueError(
                f"NewtonLabelIndexing[{self.name!r}]: pattern(s) {list(patterns)!r} (exclude "
                f"{tuple(exclude)!r}) matched no {kind} leaves."
            )
        matched = set(matched_leaves)

        # Walk pool in its world-major order so result is stable.
        indices = [gid for gid, leaf in zip(pool.ids, pool.leaves, strict=True) if leaf in matched]
        if not indices:
            raise ValueError(
                f"NewtonLabelIndexing[{self.name!r}]: resolved leaves {sorted(matched)!r} to " f"zero {kind} indices."
            )
        return indices


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------


def _scan_articulation_pool(
    *,
    labels: list[str],
    prefix: str | None,
    num_envs: int,
    kind: str,
    entity_name: str,
) -> _LabelPool:
    """Select labels by prefix, verify world-major divisibility, cache."""
    if prefix:
        scoped: list[tuple[int, str]] = [
            (i, lbl) for i, lbl in enumerate(labels) if lbl == prefix or lbl.startswith(prefix + "/")
        ]
    else:
        # No prefix → unscoped (rare for articulations; matches legacy fallback).
        scoped = list(enumerate(labels))
    if not scoped:
        return _LabelPool(ids=[], leaves=[], n_per_world=0)
    n_total = len(scoped)
    if n_total % num_envs != 0:
        raise ValueError(
            f"NewtonLabelIndexing[{entity_name!r}]: {kind} label count {n_total} is not divisible "
            f"by num_envs={num_envs}. Sample labels: {[lbl for _, lbl in scoped[:8]]}."
        )
    n_per_world = n_total // num_envs
    return _LabelPool(
        ids=[i for i, _ in scoped],
        leaves=[leaf_name(lbl) for _, lbl in scoped],
        n_per_world=n_per_world,
    )


def _scan_singleton_pool(
    labels: list[str],
    predicate: Callable[[str], bool] | None,
) -> _LabelPool:
    """Select labels matching ``predicate``; the whole match-set IS world 0."""
    if predicate is None:
        return _LabelPool(ids=[], leaves=[], n_per_world=0)
    ids: list[int] = []
    leaves: list[str] = []
    for i, lbl in enumerate(labels):
        if predicate(lbl):
            ids.append(i)
            leaves.append(leaf_name(lbl))
    return _LabelPool(ids=ids, leaves=leaves, n_per_world=len(ids))


def _matches_any_search(name: str, patterns: tuple[str, ...]) -> bool:
    """Whether ``name`` matches any of ``patterns`` (regex search).

    Mirrors the legacy resolver and the Genesis contact-sensor backend.
    """
    import re

    return any(re.search(p, name) for p in patterns)
