"""Newton contact sensor — simulator-agnostic ``ContactSensorCfg`` backend.

This wraps Newton's native :class:`newton.sensors.SensorContact` so the
*same* ``ContactSensorCfg`` that drives the Genesis / mjlab backends also
works on Newton. Two pieces of functionality are layered on top of the
native sensor:

1. **Declarative resolution.** ``ContactSensorCfg.primary`` /
   ``secondary`` (regex patterns, entity scoping) are resolved here
   against the Newton model's ``body_label`` / ``shape_label``, then the
   matched *full labels* are handed to ``SensorContact``. Patterns are
   matched with ``re.fullmatch`` against the **leaf** segment of each
   label (IsaacLab convention) — so a bare ``"FR_foot"`` or a regex
   ``".*foot"`` resolves identically on URDF-flat labels
   (``go2/FR_foot``) and MJCF-XPath labels
   (``go2/worldbody/.../FR_foot``). This is *deliberately not* fnmatch —
   Newton's ``SensorContact`` itself fnmatch-es, but we never let the
   user's regex reach it; we resolve to concrete labels first.

2. **Substep history.** Newton's ``SensorContact`` has no ring buffer,
   so when ``cfg.history_length > 0`` this wrapper keeps one of shape
   ``(num_envs, N, history_length, 3)``. The Newton scene manager calls
   :meth:`update` once per physics step (= ``decimation`` times per
   control step); each call refreshes the native sensor and pushes the
   current per-primary net contact force into the ring buffer. The ring
   is **newest-last** (the slot at ``cursor-1`` mod ``H`` is the most
   recent write) — but the only consumer, ``penalize_contact_force_count``,
   reduces over the history axis with ``.any(dim=2)``, so the ordering
   is informational only.

Secondary → counterpart mapping (NO inversion — Newton's
``counterpart_*`` is a positive whitelist, unlike Genesis's blacklist):

* ``secondary is None`` → no counterpart args; ``total_force`` reports
  the force on the primary from *all* contacts.
* ``secondary.mode == "geom"`` → ``counterpart_shapes`` = the resolved
  full shape labels of ``secondary`` (e.g. ``["ground_plane"]`` for the
  Newton ground plane, which is a single global shape with no parent
  body — so it is reached via shapes, not bodies).
* ``secondary.mode == "body"`` → ``counterpart_bodies`` = the resolved
  full body labels of ``secondary``. ``secondary.entity == "self"`` is
  just an alias for ``cfg.primary.entity`` (self-collision: counterpart
  is another body of the same robot) — Newton's whitelist handles it
  with no special path.

Force semantics when a counterpart filter IS present: Newton's
``accumulate_contact_forces_kernel`` accumulates ``total_force`` for
*every* contact touching a sensing object, regardless of whether the
counterpart matches — only ``force_matrix`` is counterpart-filtered. So
this wrapper sums ``force_matrix`` over its columns to get the *filtered*
net force, and only uses ``total_force`` when no counterpart was given.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import torch
import warp as wp
from newton.sensors import SensorContact

from rlworld.rl.configs.sensors import ContactSensorCfg
from rlworld.rl.envs.utils.newton.label import leaf_name
from rlworld.rl.utils import string as string_utils

if TYPE_CHECKING:
    from rlworld.rl.envs.managers.newton.scene import NewtonSceneManager

# ``found`` threshold (N) on the filtered net force magnitude. Matches the
# legacy ``NewtonContactManager._compute_group_is_contact`` (``> 1.0``) so
# air-time tracking / contact rewards behave consistently after migration.
_CONTACT_FORCE_EPS = 1.0


def _matches_any_search(name: str, patterns: tuple[str, ...]) -> bool:
    """Whether ``name`` matches any of ``patterns`` (regex search). Mirrors Genesis."""
    return any(re.search(p, name) for p in patterns)


class NewtonContactSensor:
    """Runtime contact sensor backing one ``ContactManager`` group on Newton.

    Constructed (post scene-build) by ``NewtonSceneManager.build_scene``
    and adopted by ``NewtonContactManager.register_sensors``. Holds the
    native ``SensorContact`` and (optionally) a substep ring buffer.
    """

    def __init__(self, scene_manager: NewtonSceneManager, cfg: ContactSensorCfg):
        if not isinstance(cfg, ContactSensorCfg):
            raise TypeError(f"NewtonContactSensor expects a ContactSensorCfg, got {type(cfg).__name__}")

        self.scene_manager = scene_manager
        self.cfg = cfg
        model = scene_manager.model
        self._model = model
        self.device = scene_manager.env.device
        self.num_envs = model.world_count

        # ---- backend support matrix ---------------------------------
        if cfg.primary.mode == "subtree":
            raise NotImplementedError(
                f"Newton backend: ContactSensorCfg {cfg.name!r} primary.mode='subtree' "
                "is not supported (mjlab-only)."
            )
        if cfg.primary.mode not in ("body", "geom"):
            raise NotImplementedError(
                f"Newton backend: ContactSensorCfg {cfg.name!r} primary.mode={cfg.primary.mode!r}; "
                "only 'body' and 'geom' are supported."
            )
        if cfg.reduce != "netforce":
            raise NotImplementedError(
                f"Newton backend: ContactSensorCfg {cfg.name!r} reduce={cfg.reduce!r}; only "
                "'netforce' (sum of all contacts into one net wrench) is supported."
            )
        if cfg.num_slots != 1:
            raise NotImplementedError(
                f"Newton backend: ContactSensorCfg {cfg.name!r} num_slots={cfg.num_slots}; only "
                "num_slots=1 is supported."
            )
        unsupported_fields = set(cfg.fields) - {"found", "force"}
        if unsupported_fields:
            raise NotImplementedError(
                f"Newton backend: ContactSensorCfg {cfg.name!r} fields={cfg.fields}; only "
                f"{{'found', 'force'}} are supported (got extra {sorted(unsupported_fields)})."
            )

        # ---- resolve primary ----------------------------------------
        primary_entity = cfg.primary.entity or "robot"
        primary_patterns = (
            (cfg.primary.pattern,) if isinstance(cfg.primary.pattern, str) else tuple(cfg.primary.pattern)
        )
        primary_labels = self._resolve_full_labels(
            entity_name=primary_entity,
            mode=cfg.primary.mode,
            patterns=primary_patterns,
            exclude=cfg.primary.exclude,
            what=f"ContactSensorCfg {cfg.name!r} primary",
        )

        sensing_kwargs: dict[str, list[str]] = {}
        if cfg.primary.mode == "body":
            sensing_kwargs["sensing_obj_bodies"] = primary_labels
        else:
            sensing_kwargs["sensing_obj_shapes"] = primary_labels

        # ---- resolve secondary → counterpart whitelist (NO inversion) ----
        self._has_counterpart = cfg.secondary is not None
        counterpart_kwargs: dict[str, list[str]] = {}
        sec = cfg.secondary
        if sec is not None:
            if sec.mode == "subtree":
                raise NotImplementedError(
                    f"Newton backend: ContactSensorCfg {cfg.name!r} secondary.mode='subtree' " "is not supported."
                )
            if sec.mode not in ("body", "geom"):
                raise NotImplementedError(
                    f"Newton backend: ContactSensorCfg {cfg.name!r} secondary.mode={sec.mode!r}; "
                    "only 'body' and 'geom' are supported."
                )
            if not sec.entity:
                raise NotImplementedError(
                    f"Newton backend: ContactSensorCfg {cfg.name!r} secondary with a literal pattern "
                    "(no entity scope) is not supported; use secondary.entity=<name> or "
                    "secondary.entity='self'."
                )
            sec_entity = primary_entity if sec.entity == "self" else sec.entity
            sec_patterns = (sec.pattern,) if isinstance(sec.pattern, str) else tuple(sec.pattern)
            sec_labels = self._resolve_full_labels(
                entity_name=sec_entity,
                mode=sec.mode,
                patterns=sec_patterns,
                exclude=sec.exclude,
                what=f"ContactSensorCfg {cfg.name!r} secondary",
            )
            if sec.mode == "body":
                counterpart_kwargs["counterpart_bodies"] = sec_labels
            else:
                counterpart_kwargs["counterpart_shapes"] = sec_labels

        # ---- build native sensor ------------------------------------
        # ``measure_total`` is only needed when there is no counterpart
        # filter (then ``total_force`` is the answer). With a counterpart
        # filter we read ``force_matrix`` instead, so skip the extra
        # allocation. ``SensorContact`` refuses ``measure_total=False``
        # with no counterparts — which never happens here.
        self._native = SensorContact(
            model,
            measure_total=not self._has_counterpart,
            **sensing_kwargs,
            **counterpart_kwargs,
        )

        # ---- derive tracked names (world-0 order, from the native sensor) ----
        obj_type = self._native.sensing_obj_type  # "body" | "shape"
        label_list = model.body_label if obj_type == "body" else model.shape_label
        n_total = len(self._native.sensing_obj_idx)
        if n_total % self.num_envs != 0:
            raise RuntimeError(
                f"Newton backend: ContactSensorCfg {cfg.name!r} resolved {n_total} sensing objects "
                f"which is not divisible by world_count={self.num_envs}; cannot derive per-env tracking."
            )
        self._n_per_env = n_total // self.num_envs
        first_env_indices = self._native.sensing_obj_idx[: self._n_per_env]
        self._tracked_names: list[str] = [leaf_name(label_list[idx]) for idx in first_env_indices]

        # ---- substep history ring buffer ----------------------------
        self._history_length = int(cfg.history_length)
        if self._history_length > 0:
            self._history = torch.zeros(self.num_envs, self._n_per_env, self._history_length, 3, device=self.device)
            self._cursor = 0
        else:
            self._history = None
            self._cursor = 0

    # ------------------------------------------------------------------
    # label resolution
    # ------------------------------------------------------------------

    def _resolve_full_labels(
        self,
        *,
        entity_name: str,
        mode: str,
        patterns: tuple[str, ...],
        exclude: tuple[str, ...],
        what: str,
    ) -> list[str]:
        """Resolve ``patterns`` to a list of full Newton labels (across all worlds).

        Patterns are matched via ``re.fullmatch`` against the *leaf* of
        each candidate label; ``exclude`` entries are dropped via
        ``re.search`` on the leaf (Genesis-compatible). The candidate
        pool is the entity's bodies / shapes — scoped by
        ``body_label_prefix`` when the entity config has one (articulation
        entities); unscoped (whole model) for prefix-less entities such as
        ``GroundPlaneCfg``, which contributes a single global
        ``ground_plane`` shape.
        """
        model = self._model
        all_labels = list(model.body_label if mode == "body" else model.shape_label)
        if not all_labels:
            raise ValueError(f"Newton backend: {what}: model has no {'body' if mode == 'body' else 'shape'} labels.")

        # Entity scoping by label prefix (best-effort; harmless no-op for
        # prefix-less entities like GroundPlaneCfg).
        entity_info = self.scene_manager.entities.get(entity_name)
        if entity_info is None:
            raise ValueError(
                f"Newton backend: {what}: entity {entity_name!r} not found in scene "
                f"(known entities: {list(self.scene_manager.entities)})."
            )
        prefix = getattr(entity_info["config"], "body_label_prefix", None)
        if prefix:
            scoped = [(i, lbl) for i, lbl in enumerate(all_labels) if lbl == prefix or lbl.startswith(prefix + "/")]
        else:
            scoped = list(enumerate(all_labels))
        if not scoped:
            raise ValueError(
                f"Newton backend: {what}: entity {entity_name!r} (prefix {prefix!r}) matched no "
                f"{'body' if mode == 'body' else 'shape'} labels. Sample labels: {all_labels[:8]}"
            )

        # Validate user patterns against the world-0 leaf-name pool (the
        # canonical per-env name set). ``resolve_matching_names`` raises if
        # a pattern matches nothing.
        n_total_scoped = len(scoped)
        # world_count partitioning of the *scoped* labels — Newton replicates
        # entities world-major, so the first 1/world_count slice is world 0.
        if n_total_scoped % self.num_envs == 0:
            n_per_env = n_total_scoped // self.num_envs
        else:
            # Single global entity (e.g. ground plane shape, world=-1): the
            # whole scoped set IS "world 0".
            n_per_env = n_total_scoped
        world0 = scoped[:n_per_env]
        world0_leaves = [leaf_name(lbl) for _, lbl in world0]
        _, matched_leaves = string_utils.resolve_matching_names(list(patterns), world0_leaves, preserve_order=True)
        if exclude:
            matched_leaves = [n for n in matched_leaves if not _matches_any_search(n, tuple(exclude))]
        if not matched_leaves:
            raise ValueError(
                f"Newton backend: {what}: pattern(s) {list(patterns)} (entity {entity_name!r}, "
                f"after exclude {tuple(exclude)}) matched no {'bodies' if mode == 'body' else 'shapes'}."
            )
        matched_set = set(matched_leaves)

        # Expand to full labels across every world (SensorContact then
        # fnmatch-es each full label against itself → exact match). Iterate
        # ``scoped`` in index order so the resulting list — and hence the
        # sensor's ``sensing_obj_idx`` — stays world-major with a stable
        # within-world order.
        full_labels = [lbl for _, lbl in scoped if leaf_name(lbl) in matched_set]
        if not full_labels:
            raise ValueError(f"Newton backend: {what}: resolved leaf names {sorted(matched_set)} to zero labels.")
        return full_labels

    # ------------------------------------------------------------------
    # properties
    # ------------------------------------------------------------------

    @property
    def tracked_names(self) -> list[str]:
        return self._tracked_names

    @property
    def native_sensor(self) -> SensorContact:
        return self._native

    # ------------------------------------------------------------------
    # readings
    # ------------------------------------------------------------------

    def compute_force(self) -> torch.Tensor:
        """Filtered net contact force per primary. Shape ``(num_envs, N, 3)``."""
        if self._has_counterpart:
            # ``total_force`` is NOT counterpart-filtered (see module
            # docstring) — sum the per-counterpart columns instead.
            fm = wp.to_torch(self._native.force_matrix)  # (n_obj, max_cp, 3)
            net = fm.sum(dim=1)  # (n_obj, 3)
        else:
            net = wp.to_torch(self._native.total_force)  # (n_obj, 3)
        return net.reshape(self.num_envs, self._n_per_env, 3)

    def compute_found(self) -> torch.Tensor:
        """Contact present per primary (``‖filtered net force‖ > eps``). Shape ``(num_envs, N)`` bool."""
        return torch.norm(self.compute_force(), dim=-1) > _CONTACT_FORCE_EPS

    # ------------------------------------------------------------------
    # substep history (only when history_length > 0)
    # ------------------------------------------------------------------

    def update(self, state, contacts) -> None:
        """Refresh the native sensor and (if enabled) push one history frame.

        Called by ``NewtonSceneManager._update_sensors`` once per physics
        step. ``state`` / ``contacts`` are the scene manager's current
        ``state_0`` / ``sensor_contacts``.
        """
        self._native.update(state, contacts)
        if self._history is None:
            return
        # Push the current per-primary net force into the ring at ``cursor``.
        self._history[:, :, self._cursor, :] = self.compute_force()
        self._cursor = (self._cursor + 1) % self._history_length

    def compute_history(self) -> torch.Tensor | None:
        """Substep contact-force history ``(num_envs, N, H, 3)``, or ``None`` if disabled.

        Ring order is **newest-last** (slot ``cursor-1`` mod ``H`` is the
        most recent). The only consumer reduces over ``H`` with
        ``.any(dim=2)``, so order does not matter to it.
        """
        return self._history

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Zero the history ring for the given envs (no-op if history disabled)."""
        if self._history is None or env_ids is None or len(env_ids) == 0:
            return
        self._history[env_ids] = 0.0
