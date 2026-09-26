"""Observation terms for the K1 velocity task.

The actor/critic groups are otherwise assembled from the common
proprioception terms; this covers the K1-specific piece.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from jaxrlworld.rl.envs.world import World


def velocity_command(env: World, command_term: str = "velocity") -> torch.Tensor:
    """The 3-D velocity command alone.

    ``all_commands`` would concatenate every command term, so the velocity
    term is read explicitly.
    """
    return env.command_manager.get_term(command_term).command
