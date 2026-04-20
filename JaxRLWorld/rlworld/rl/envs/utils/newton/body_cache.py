from __future__ import annotations

import re
from typing import TYPE_CHECKING

import warp as wp

from rlworld.rl.envs.utils.newton.label import flatten_xpath_label

if TYPE_CHECKING:
    from rlworld.rl.envs import NewtonEnv


class NewtonBodyCache:
    """Cache for body index lookups."""

    def __init__(self, env: "NewtonEnv"):
        self.env = env
        self._body_cache: dict[tuple[str, ...], list[int]] = {}
        self._contact_cache: dict[tuple[str, ...], tuple[list[int], list[int]]] = {}

        model = env.scene_manager.model
        self.bodies_per_env = len(model.body_label) // env.num_envs
        # Canonicalize MJCF XPath labels to flat ``{prefix}/{leaf}``
        # form so downstream pattern matching is URDF/MJCF-agnostic.
        self.body_names = [
            flatten_xpath_label(n) for n in model.body_label[:self.bodies_per_env]
        ]

        # Cache original values for domain randomization
        body_mass = wp.to_torch(model.body_mass).reshape(env.num_envs, self.bodies_per_env)
        self.original_body_mass = body_mass[0].clone()  # (bodies_per_env,)

    def get_body_indices(self, body_patterns: str | list[str]) -> list[int]:
        """Get body_q indices for patterns, preserving query pattern order.

        Each pattern is matched against body_names in order. Results are
        appended per-pattern so that the output follows the query order,
        not the internal body_names order.
        """
        if isinstance(body_patterns, str):
            body_patterns = [body_patterns]

        key = tuple(body_patterns)
        if key not in self._body_cache:
            seen = set()
            body_indices = []
            for pattern in body_patterns:
                regex = re.compile(pattern)
                pattern_matches = [
                    idx for idx, name in enumerate(self.body_names)
                    if regex.match(name) and idx not in seen
                ]
                for idx in pattern_matches:
                    seen.add(idx)
                body_indices.extend(pattern_matches)

            if not body_indices:
                raise ValueError(
                    f"No bodies matching '{body_patterns}'. "
                    f"Available: {self.body_names}"
                )
            self._body_cache[key] = body_indices

        return self._body_cache[key]

    def get_body_indices_with_contact(
        self, body_patterns: str | list[str]
    ) -> tuple[list[int], list[int]]:
        """Get (body_q_indices, contact_indices) for patterns tracked by contact_manager."""
        if isinstance(body_patterns, str):
            body_patterns = [body_patterns]

        key = tuple(body_patterns)
        if key not in self._contact_cache:
            contact_indices = self.env.contact_manager.get_indices(
                "foot_contact", list(body_patterns), preserve_order=True
            )

            if not contact_indices:
                tracked = self.env.contact_manager.tracked_names("foot_contact")
                raise ValueError(
                    f"No bodies matching '{body_patterns}' in contact group 'contact'. "
                    f"Available: {tracked}"
                )

            tracked_names = self.env.contact_manager.tracked_names("foot_contact")
            body_names = [tracked_names[i] for i in contact_indices]
            body_indices = [self.body_names.index(name) for name in body_names]
            self._contact_cache[key] = (body_indices, contact_indices)

        return self._contact_cache[key]


_caches: dict[int, NewtonBodyCache] = {}


def get_cache(env: "NewtonEnv") -> NewtonBodyCache:
    env_id = id(env)
    if env_id not in _caches:
        _caches[env_id] = NewtonBodyCache(env)
    return _caches[env_id]


def clear_cache(env: "NewtonEnv" = None) -> None:
    if env is None:
        _caches.clear()
    else:
        _caches.pop(id(env), None)
