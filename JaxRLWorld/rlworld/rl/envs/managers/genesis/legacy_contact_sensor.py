"""Legacy Genesis contact sensor (``GenesisContactSensorCfg`` backend).

This is the original ``entity.get_contacts()``-based wrapper, kept
verbatim for the g1 / t1 presets which still use the legacy
``GenesisContactSensorCfg`` config. New presets use
``rlworld.rl.configs.sensors.ContactSensorCfg`` →
:class:`rlworld.rl.envs.managers.genesis.contact_sensor.GenesisContactSensor`.

Wraps ``entity.get_contacts()`` with primary/secondary link filtering
and per-primary-link force aggregation, producing fixed-shape tensors.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import torch

from rlworld.rl.envs.managers.genesis.contact_sensor import ContactSensorData
from rlworld.rl.utils import entity_utils as eu

if TYPE_CHECKING:
    from rlworld.rl.configs.genesis_config_classes import GenesisContactSensorCfg
    from rlworld.rl.envs import GenesisEnv


class LegacyGenesisContactSensor:
    """Runtime contact sensor that filters ``get_contacts()`` per step.

    Produces fixed-shape ``(num_envs, N)`` / ``(num_envs, N, 3)`` tensors
    from Genesis's variable-length contact data.
    """

    def __init__(self, env: GenesisEnv, cfg: GenesisContactSensorCfg):
        self.env = env
        self.cfg = cfg
        self.device = env.device
        self.num_envs = env.num_envs

        entity = env.scene_manager[cfg.entity_name]
        self._entity = entity

        # Resolve primary link names (regex supported) → global link indices
        link_ids, link_names = eu.find_links(entity, cfg.primary_links, global_ids=True, preserve_order=True)

        # Apply exclude filter
        if cfg.exclude_links:
            exclude_patterns = [re.compile(p) for p in cfg.exclude_links]
            filtered = [
                (lid, lname)
                for lid, lname in zip(link_ids, link_names)
                if not any(rx.search(lname) for rx in exclude_patterns)
            ]
            if filtered:
                link_ids, link_names = zip(*filtered)
                link_ids, link_names = list(link_ids), list(link_names)
            else:
                link_ids, link_names = [], []

        self._primary_link_ids = torch.tensor(link_ids, dtype=torch.int32, device=self.device)
        self._tracked_names = link_names
        self._num_primary = len(link_names)

        # Resolve secondary entity for get_contacts filtering
        self._with_entity = None
        self._exclude_self = cfg.exclude_self_contact

        if cfg.secondary_entity == "self":
            # Self-collision only
            self._with_entity = entity
            self._exclude_self = False
        elif cfg.secondary_entity is not None:
            # Specific entity (e.g. "ground")
            self._with_entity = env.scene_manager[cfg.secondary_entity]

        # If secondary is a specific entity, collect its link ids for filtering
        self._secondary_link_ids: torch.Tensor | None = None
        if cfg.secondary_entity is not None and cfg.secondary_entity != "self":
            sec_entity = env.scene_manager[cfg.secondary_entity]
            sec_ids = list(range(sec_entity.link_start, sec_entity.link_end))
            self._secondary_link_ids = torch.tensor(sec_ids, dtype=torch.int32, device=self.device)

        # Pre-allocate output buffers
        self._found_buf = torch.zeros(self.num_envs, self._num_primary, dtype=torch.bool, device=self.device)
        self._force_buf = torch.zeros(self.num_envs, self._num_primary, 3, device=self.device)

    @property
    def tracked_names(self) -> list[str]:
        return self._tracked_names

    def compute_history(self) -> torch.Tensor | None:
        """Legacy sensor has no substep history."""
        return None

    def compute(self) -> ContactSensorData:
        """Query contacts, filter, aggregate. Returns fixed-shape tensors."""
        self._found_buf.zero_()
        self._force_buf.zero_()

        raw = self._entity.get_contacts(
            with_entity=self._with_entity,
            exclude_self_contact=self._exclude_self,
        )

        if raw is None:
            return ContactSensorData(
                found=self._found_buf,
                force=self._force_buf,
                tracked_names=self._tracked_names,
            )

        link_a = raw["link_a"]  # (num_envs, max_contacts) or (max_contacts,)
        link_b = raw["link_b"]
        force_a = raw["force_a"]  # (num_envs, max_contacts, 3)
        force_b = raw["force_b"]
        valid_mask = raw.get("valid_mask")  # (num_envs, max_contacts) or None

        # Handle empty contacts
        if link_a.numel() == 0:
            return ContactSensorData(
                found=self._found_buf,
                force=self._force_buf,
                tracked_names=self._tracked_names,
            )

        # Ensure batch dimension
        if link_a.dim() == 1:
            link_a = link_a.unsqueeze(0)
            link_b = link_b.unsqueeze(0)
            force_a = force_a.unsqueeze(0)
            force_b = force_b.unsqueeze(0)
            if valid_mask is not None:
                valid_mask = valid_mask.unsqueeze(0)

        # (num_envs, max_contacts, num_primary)
        primary_ids = self._primary_link_ids  # (N,)
        match_a = link_a.unsqueeze(-1) == primary_ids  # primary is link_a side
        match_b = link_b.unsqueeze(-1) == primary_ids  # primary is link_b side

        # Apply valid_mask
        if valid_mask is not None:
            vm = valid_mask.unsqueeze(-1)  # (num_envs, max_contacts, 1)
            match_a = match_a & vm
            match_b = match_b & vm

        # Secondary filter: counterpart must be in secondary entity links
        if self._secondary_link_ids is not None:
            sec_ids = self._secondary_link_ids
            # If primary is link_a, counterpart is link_b → must be in secondary
            counterpart_b_ok = torch.isin(link_b, sec_ids).unsqueeze(-1)
            # If primary is link_b, counterpart is link_a → must be in secondary
            counterpart_a_ok = torch.isin(link_a, sec_ids).unsqueeze(-1)
            match_a = match_a & counterpart_b_ok
            match_b = match_b & counterpart_a_ok

        # Found: any contact exists for each primary link
        # (num_envs, max_contacts, N) → any over contacts → (num_envs, N)
        self._found_buf = (match_a | match_b).any(dim=1)

        # Force aggregation: sum forces on each primary link
        # When primary is link_a: force on primary = force_a
        # When primary is link_b: force on primary = force_b
        # (num_envs, max_contacts, 3) * (num_envs, max_contacts, N) → sum → (num_envs, N, 3)
        force_from_a = torch.einsum("bci,bcn->bni", force_a, match_a.float())
        force_from_b = torch.einsum("bci,bcn->bni", force_b, match_b.float())
        self._force_buf = force_from_a + force_from_b

        return ContactSensorData(
            found=self._found_buf.clone(),
            force=self._force_buf.clone(),
            tracked_names=self._tracked_names,
        )
