from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from rlworld.rl.configs.sensors import ContactSensorCfg
from rlworld.rl.envs.managers.common.contact import BaseContactManager, ContactGroup
from rlworld.rl.envs.managers.genesis.contact_sensor import GenesisContactSensor

if TYPE_CHECKING:
    from rlworld.rl.envs import GenesisEnv


class ContactManager(BaseContactManager):
    """Named-group contact manager for Genesis environments.

    Each :class:`rlworld.rl.configs.sensors.ContactSensorCfg` becomes a
    named group, backed by :class:`GenesisContactSensor` (which prefers
    Genesis's native link sensors). Those native sensors must be created
    *before* ``scene.build()``, so they are pre-built by
    ``GenesisEnv._build_scene`` and merely *adopted* here (see
    ``env._genesis_contact_sensors``).
    """

    def __init__(self, env: GenesisEnv):
        super().__init__(env=env)
        self._sensors: dict[str, GenesisContactSensor] = {}
        # Substep counter for the batched advance() — see the override below.
        self._substeps_seen = 0

    def register_sensor(self, cfg: ContactSensorCfg) -> None:
        """Register a contact sensor config as a named group.

        The sensor (and, for the native path, its ``gs.sensors.*``
        objects added to the still-unbuilt scene) is pre-built by
        ``GenesisEnv._build_scene``; here we just adopt it.
        """
        pre_built = getattr(self.env, "_genesis_contact_sensors", {})
        sensor = pre_built.get(cfg.name)
        if sensor is None:
            raise RuntimeError(
                f"ContactSensorCfg {cfg.name!r} was not pre-registered before "
                "scene.build(); ensure GenesisEnv._build_scene() iterates "
                "scene_cfg.contact_sensors and calls create_native_sensors()."
            )
        if cfg.history_length < self.env.decimation:
            raise ValueError(
                f"Genesis ContactSensorCfg {cfg.name!r}: history_length="
                f"{cfg.history_length} < decimation={self.env.decimation}. The batched "
                "contact-timing path replays the native per-substep history once per "
                "control step, so the sensor ring must retain every substep frame; "
                "set history_length=decimation (as every genesis preset does)."
            )
        self._sensors[cfg.name] = sensor
        self._register_group(cfg.name, sensor.tracked_names)

    # -- batched timing accumulation --

    def advance(self, dt: float) -> None:
        """Batched contact-timing accumulation: one read per control step.

        The base implementation reads each group's CURRENT contact boolean
        on every substep — on Genesis that is one GPU sync per primary
        link per substep (~128 syncs per control step for G1's
        feet + self-collision groups; measured ~14 ms/step at 4096 envs).
        The native sensors already record a per-substep history ring
        (``history_length >= decimation``, enforced in
        :meth:`register_sensor`), so this override skips the intermediate
        substeps and, on the LAST substep of the control step, reads the
        ring ONCE and replays its frames oldest-first through the
        identical arithmetic (:meth:`_apply_contact_frame`).  Same
        booleans, same order, same dt — bit-identical timing buffers at
        1/decimation of the sync cost.  Verified per-robot by
        ``scripts/diag/genesis_contact_batching_diag.py``.
        """
        self._substeps_seen += 1
        if self._substeps_seen < self.env.decimation:
            return
        self._substeps_seen = 0
        for group in self._groups.values():
            hist = self._sensors[group.name].read_found_history()  # (n, H, N) oldest-first
            frames = hist[:, -self.env.decimation :, :]
            for k in range(frames.shape[1]):
                self._apply_contact_frame(group, frames[:, k, :], dt)

    # -- abstract impl --

    def _compute_group_contact_force(self, group: ContactGroup) -> torch.Tensor | None:
        # Only the force channel — calling ``compute()`` here would
        # also read every contact (found) sensor and pay the per-link
        # GPU sync for data this caller does not use.
        return self._sensors[group.name].read_force()

    def _compute_group_contact_force_history(self, group: ContactGroup) -> torch.Tensor | None:
        sensor = self._sensors[group.name]
        compute_history = sensor.compute_history
        return compute_history()

    def _compute_group_is_contact(self, group: ContactGroup) -> torch.Tensor:
        """Contact bool from Genesis's first-class ``gs.sensors.Contact``
        (pair-count > 0), not the base force-magnitude threshold. The
        ``Contact`` sensor returns a solver-level binary signal — see
        ``Genesis/examples/sensors/contact_force_go2.py`` for the canonical
        pattern — which is invariant to solver-iteration force jitter, so
        a settled standing foot keeps ``is_contact=True`` even when the
        net force momentarily dips below the 1 N noise floor used by
        the base classifier. Aligns with mjlab's ``data.found > 0``
        semantics in MujocoContactManager._compute_group_is_contact.

        Uses the ``read_found``-only fast path so the per-substep
        ``ContactManager.advance`` does not also pay a force-channel
        sync per primary link.
        """
        return self._sensors[group.name].read_found()

    # -- pretty print --

    def __str__(self) -> str:
        from rlworld.rl.utils.pretty import create_manager_table, table_to_string

        if not self._groups:
            return ""

        rows = []
        for gname, group in self._groups.items():
            sensor = self._sensors[gname]
            cfg = sensor.cfg
            if cfg.secondary is None:
                sec = "any"
            elif cfg.secondary.entity:
                sec = cfg.secondary.entity
            else:
                sec = str(cfg.secondary.pattern)
            for idx, name in enumerate(group.tracked_names):
                rows.append([gname, idx, name, sec])

        table = create_manager_table(
            title="Contact Tracking (Genesis)",
            columns=["Group", "Idx", "Link", "Secondary"],
            rows=rows,
            footer=f"{len(self._groups)} groups, {sum(g.num_tracked for g in self._groups.values())} tracked",
        )
        return table_to_string(table)
