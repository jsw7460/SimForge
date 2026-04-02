"""Named-group contact manager base class.

Each simulator backend registers one or more **contact groups** (e.g.
"feet_ground_contact", "body_ground_contact").  Every group tracks an
independent set of bodies/links with its own timing buffers.

Public API is method-based with a ``group_name`` parameter::

    env.contact_manager.is_contact("feet_ground_contact")       # (B, N) bool
    env.contact_manager.contact_force("feet_ground_contact")    # (B, N, 3)
    env.contact_manager.current_air_time("feet_ground_contact") # (B, N)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from rlworld.rl.envs.managers.base import BaseManager
from rlworld.rl.utils import string as string_utils

if TYPE_CHECKING:
    from rlworld.rl.envs import World


@dataclass
class ContactGroup:
    """One named group of tracked contact bodies with independent timing state."""

    name: str
    tracked_names: list[str]
    num_tracked: int

    # Timing buffers — shape (num_envs, num_tracked)
    current_air_time: torch.Tensor = field(repr=False)
    current_contact_time: torch.Tensor = field(repr=False)
    last_air_time: torch.Tensor = field(repr=False)
    last_contact_time: torch.Tensor = field(repr=False)
    _prev_is_contact: torch.Tensor = field(repr=False)


class BaseContactManager(BaseManager, ABC):
    """Base class for contact managers across all simulator backends.

    Subclasses must implement:
        - ``_compute_group_is_contact(group)``  → ``(num_envs, N)`` bool
        - ``_compute_group_contact_force(group)`` → ``(num_envs, N, 3)`` or ``None``
    """

    def __init__(self, env: "World"):
        super().__init__(env=env)
        self.num_envs = env.num_envs
        self.dt = env.control_dt
        self._groups: dict[str, ContactGroup] = {}

    # ------------------------------------------------------------------
    # Group registration (called by subclasses)
    # ------------------------------------------------------------------

    def _register_group(self, name: str, tracked_names: list[str]) -> None:
        """Create a named contact group with allocated buffers."""
        n = len(tracked_names)
        shape = (self.num_envs, n)
        group = ContactGroup(
            name=name,
            tracked_names=tracked_names,
            num_tracked=n,
            current_air_time=torch.zeros(shape, device=self.device),
            current_contact_time=torch.zeros(shape, device=self.device),
            last_air_time=torch.zeros(shape, device=self.device),
            last_contact_time=torch.zeros(shape, device=self.device),
            _prev_is_contact=torch.zeros(shape, dtype=torch.bool, device=self.device),
        )
        self._groups[name] = group

    # ------------------------------------------------------------------
    # Abstract — subclass must implement per-group
    # ------------------------------------------------------------------

    @abstractmethod
    def _compute_group_is_contact(self, group: ContactGroup) -> torch.Tensor:
        """Return contact state ``(num_envs, group.num_tracked)`` as bool tensor."""
        ...

    @abstractmethod
    def _compute_group_contact_force(self, group: ContactGroup) -> torch.Tensor | None:
        """Return contact forces ``(num_envs, group.num_tracked, 3)`` or ``None``."""
        ...

    def _compute_group_contact_force_history(self, group: ContactGroup) -> torch.Tensor | None:
        """Return contact force history ``(num_envs, group.num_tracked, H, 3)`` or ``None``.

        Override in backends that support substep history (e.g. MuJoCo).
        Returns ``None`` by default (no history available).
        """
        return None

    # ------------------------------------------------------------------
    # Public API — named group access
    # ------------------------------------------------------------------

    def _get_group(self, name: str) -> ContactGroup:
        try:
            return self._groups[name]
        except KeyError:
            available = list(self._groups.keys())
            raise KeyError(
                f"Contact group '{name}' not found. Available: {available}"
            ) from None

    # -- reindex cache for order parameter --

    _reindex_cache: dict[tuple[str, tuple[str, ...]], torch.Tensor] = {}

    def _get_reindex(self, group_name: str, order: list[str]) -> torch.Tensor:
        """Get (or compute+cache) reindex tensor for requested order."""
        key = (group_name, tuple(order))
        if key not in self._reindex_cache:
            group = self._get_group(group_name)
            tracked = group.tracked_names
            if len(order) != group.num_tracked:
                raise ValueError(
                    f"order has {len(order)} elements but group '{group_name}' "
                    f"tracks {group.num_tracked}: {tracked}"
                )
            missing = set(order) - set(tracked)
            if missing:
                raise ValueError(
                    f"order contains names not in group '{group_name}': {missing}. "
                    f"Available: {tracked}"
                )
            self._reindex_cache[key] = torch.tensor(
                [tracked.index(name) for name in order],
                dtype=torch.long, device=self.device,
            )
        return self._reindex_cache[key]

    def _apply_order(
        self, tensor: torch.Tensor, group_name: str, order: list[str] | None
    ) -> torch.Tensor:
        """Reorder dim=1 of tensor if order is specified."""
        if order is None:
            return tensor
        reindex = self._get_reindex(group_name, order)
        return tensor[:, reindex]

    def _apply_order_3d(
        self, tensor: torch.Tensor, group_name: str, order: list[str] | None
    ) -> torch.Tensor:
        """Reorder dim=1 of (num_envs, N, 3) tensor if order is specified."""
        if order is None:
            return tensor
        reindex = self._get_reindex(group_name, order)
        return tensor[:, reindex, :]

    # -- public API --

    def is_contact(
        self, group_name: str, order: list[str] | None = None
    ) -> torch.Tensor:
        """Bool contact state. Shape: ``(num_envs, N)``."""
        group = self._get_group(group_name)
        result = self._compute_group_is_contact(group)
        return self._apply_order(result, group_name, order)

    def prev_is_contact(
        self, group_name: str, order: list[str] | None = None
    ) -> torch.Tensor:
        """Previous step contact state. Shape: ``(num_envs, N)`` bool."""
        result = self._get_group(group_name)._prev_is_contact
        return self._apply_order(result, group_name, order)

    def contact_force(
        self, group_name: str, order: list[str] | None = None
    ) -> torch.Tensor:
        """Contact force vectors. Shape: ``(num_envs, N, 3)``."""
        group = self._get_group(group_name)
        force = self._compute_group_contact_force(group)
        if force is None:
            force = torch.zeros(
                self.num_envs, group.num_tracked, 3, device=self.device
            )
        return self._apply_order_3d(force, group_name, order)

    def contact_force_history(
        self, group_name: str, order: list[str] | None = None
    ) -> torch.Tensor | None:
        """Contact force history across substeps. Shape: ``(num_envs, N, H, 3)``.

        Returns ``None`` if the backend does not support substep history
        (Genesis, Newton). MuJoCo returns history when ``history_length > 0``
        in the ContactSensorCfg.
        """
        group = self._get_group(group_name)
        history = self._compute_group_contact_force_history(group)
        if history is None:
            return None
        if order is not None:
            reindex = self._get_reindex(group_name, order)
            history = history[:, reindex, :, :]
        return history

    def current_air_time(
        self, group_name: str, order: list[str] | None = None
    ) -> torch.Tensor:
        result = self._get_group(group_name).current_air_time
        return self._apply_order(result, group_name, order)

    def last_air_time(
        self, group_name: str, order: list[str] | None = None
    ) -> torch.Tensor:
        result = self._get_group(group_name).last_air_time
        return self._apply_order(result, group_name, order)

    def current_contact_time(
        self, group_name: str, order: list[str] | None = None
    ) -> torch.Tensor:
        result = self._get_group(group_name).current_contact_time
        return self._apply_order(result, group_name, order)

    def last_contact_time(
        self, group_name: str, order: list[str] | None = None
    ) -> torch.Tensor:
        result = self._get_group(group_name).last_contact_time
        return self._apply_order(result, group_name, order)

    def compute_first_contact(
        self, group_name: str, abs_tol: float = 1e-6, order: list[str] | None = None
    ) -> torch.Tensor:
        """``(num_envs, N)`` bool: True for contacts established this step."""
        g = self._get_group(group_name)
        is_in = g.current_contact_time > 0
        just_landed = g.current_contact_time < (self.dt + abs_tol)
        result = is_in & just_landed
        return self._apply_order(result, group_name, order)

    def compute_first_air(
        self, group_name: str, abs_tol: float = 1e-6, order: list[str] | None = None
    ) -> torch.Tensor:
        """``(num_envs, N)`` bool: True for contacts broken this step."""
        g = self._get_group(group_name)
        is_in = g.current_air_time > 0
        just_lifted = g.current_air_time < (self.dt + abs_tol)
        result = is_in & just_lifted
        return self._apply_order(result, group_name, order)

    def tracked_names(self, group_name: str) -> list[str]:
        return self._get_group(group_name).tracked_names

    def get_indices(
        self,
        group_name: str,
        patterns: str | list[str],
        preserve_order: bool = False,
    ) -> list[int]:
        """Get indices within a group matching name patterns (regex supported)."""
        names = self.tracked_names(group_name)
        _, matched = string_utils.resolve_matching_names(
            patterns, names, preserve_order=preserve_order
        )
        return [names.index(n) for n in matched]

    def group_names(self) -> list[str]:
        return list(self._groups.keys())

    def has_group(self, name: str) -> bool:
        return name in self._groups

    # ------------------------------------------------------------------
    # Timing logic (operates on all groups)
    # ------------------------------------------------------------------

    def advance(self) -> None:
        for group in self._groups.values():
            self._advance_group(group)

    def _advance_group(self, g: ContactGroup) -> None:
        is_contact = self._compute_group_is_contact(g)

        is_landing = ~g._prev_is_contact & is_contact
        is_liftoff = g._prev_is_contact & ~is_contact

        g.last_air_time = torch.where(
            is_landing, g.current_air_time, g.last_air_time
        )
        g.last_contact_time = torch.where(
            is_liftoff, g.current_contact_time, g.last_contact_time
        )

        g.current_contact_time = torch.where(
            is_contact,
            g.current_contact_time + self.dt,
            torch.zeros_like(g.current_contact_time),
        )
        g.current_air_time = torch.where(
            ~is_contact,
            g.current_air_time + self.dt,
            torch.zeros_like(g.current_air_time),
        )

        g._prev_is_contact = is_contact

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None or len(env_ids) == 0:
            return
        for g in self._groups.values():
            g.current_air_time[env_ids] = 0.0
            g.current_contact_time[env_ids] = 0.0
            g.last_air_time[env_ids] = 0.0
            g.last_contact_time[env_ids] = 0.0
            g._prev_is_contact[env_ids] = False
