from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rlworld.rl.envs import GenesisEnv, NewtonEnv

from rlworld.rl.configs.terminations import TerminationResult


def max_episode_exceed(env: GenesisEnv | NewtonEnv) -> TerminationResult:
    """Check if episodes have reached maximum length.

    Args:
        env: The locomotion environment.

    Returns:
        Boolean tensor of shape (num_envs,) indicating which environments
        have reached max episode length.
    """
    return TerminationResult(
        env.termination_manager.episode_length_buf >= env.termination_manager.max_episode_length,
        is_timeout=True
    )
