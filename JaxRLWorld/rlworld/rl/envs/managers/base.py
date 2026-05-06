from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rlworld.rl.envs import World


class BaseManager:
    """Base class for all managers."""

    def __init__(self, env: "World"):
        self.env = env
        self.device = env.device

    @property
    def env_step_calls(self) -> int:
        """Number of step() calls on the parent environment."""
        return self.env._env_step_counter
