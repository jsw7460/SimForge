from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch

from rlworld.rl.envs.managers.base import BaseManager
from rlworld.rl.utils import string as string_utils

if TYPE_CHECKING:
    from rlworld.rl.envs import World


class BaseContactManager(BaseManager, ABC):
    """Base class for contact managers across all simulator backends.

    Handles timing logic (air_time / contact_time tracking) and provides
    a common interface for reward/observation code. Subclasses only need
    to implement sensor discovery and ``_compute_is_contact()``.

    All timing tensors have shape ``(num_envs, num_tracked)`` where axis 1
    matches the ordering returned by ``tracked_names``.
    """

    def __init__(self, env: "World"):
        super().__init__(env=env)
        self.num_envs = env.num_envs
        self.dt = env.control_dt
        self._num_tracked: int = 0

        # Timing buffers — initialized lazily via _init_buffers()
        self.current_air_time: torch.Tensor | None = None
        self.current_contact_time: torch.Tensor | None = None
        self.last_air_time: torch.Tensor | None = None
        self.last_contact_time: torch.Tensor | None = None
        self._prev_is_contact: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Buffer initialization (call from subclass after num_tracked is set)
    # ------------------------------------------------------------------

    def _init_buffers(self) -> None:
        """Allocate timing buffers. Call after ``_num_tracked`` is set."""
        if self._num_tracked == 0:
            return
        shape = (self.num_envs, self._num_tracked)
        self.current_air_time = torch.zeros(shape, device=self.device)
        self.current_contact_time = torch.zeros(shape, device=self.device)
        self.last_air_time = torch.zeros(shape, device=self.device)
        self.last_contact_time = torch.zeros(shape, device=self.device)
        self._prev_is_contact = torch.zeros(
            shape, dtype=torch.bool, device=self.device
        )

    # ------------------------------------------------------------------
    # Abstract — subclass must implement
    # ------------------------------------------------------------------

    @abstractmethod
    def _compute_is_contact(self) -> torch.Tensor:
        """Return contact state ``(num_envs, num_tracked)`` as bool tensor."""
        ...

    @property
    @abstractmethod
    def tracked_names(self) -> list[str]:
        """Canonical list of tracked body/shape/link names."""
        ...

    # ------------------------------------------------------------------
    # Common interface
    # ------------------------------------------------------------------

    @property
    def is_contact(self) -> torch.Tensor:
        return self._compute_is_contact()

    def get_link_indices(
        self,
        links: str | list[str],
        entity_name: str = "robot",
        preserve_order: bool = False,
    ) -> list[int]:
        """Get indices of tracked names matching *links* (regex supported).

        ``entity_name`` is accepted for API compatibility but ignored by
        backends that do not distinguish entities (Newton, MuJoCo).
        """
        _, matched = string_utils.resolve_matching_names(
            links, self.tracked_names, preserve_order=preserve_order
        )
        return [self.tracked_names.index(n) for n in matched]

    def get_link_names(self, indices: list[int]) -> list[str]:
        return [self.tracked_names[i] for i in indices]

    # ------------------------------------------------------------------
    # Timing logic
    # ------------------------------------------------------------------

    def advance(self) -> None:
        if self._num_tracked == 0:
            return

        is_contact = self._compute_is_contact()

        is_landing = ~self._prev_is_contact & is_contact
        is_liftoff = self._prev_is_contact & ~is_contact

        self.last_air_time = torch.where(
            is_landing, self.current_air_time, self.last_air_time
        )
        self.last_contact_time = torch.where(
            is_liftoff, self.current_contact_time, self.last_contact_time
        )

        self.current_contact_time = torch.where(
            is_contact,
            self.current_contact_time + self.dt,
            torch.zeros_like(self.current_contact_time),
        )
        self.current_air_time = torch.where(
            ~is_contact,
            self.current_air_time + self.dt,
            torch.zeros_like(self.current_air_time),
        )

        self._prev_is_contact = is_contact

    def compute_first_contact(self, abs_tol: float = 1e-6) -> torch.Tensor:
        if self._num_tracked == 0:
            return torch.zeros(
                self.num_envs, 0, dtype=torch.bool, device=self.device
            )
        is_contact = self.current_contact_time > 0
        just_landed = self.current_contact_time < (self.dt + abs_tol)
        return is_contact & just_landed

    def compute_first_air(self, abs_tol: float = 1e-6) -> torch.Tensor:
        if self._num_tracked == 0:
            return torch.zeros(
                self.num_envs, 0, dtype=torch.bool, device=self.device
            )
        is_air = self.current_air_time > 0
        just_lifted = self.current_air_time < (self.dt + abs_tol)
        return is_air & just_lifted

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if self._num_tracked == 0 or env_ids is None or len(env_ids) == 0:
            return
        self.current_air_time[env_ids] = 0.0
        self.current_contact_time[env_ids] = 0.0
        self.last_air_time[env_ids] = 0.0
        self.last_contact_time[env_ids] = 0.0
        self._prev_is_contact[env_ids] = False
