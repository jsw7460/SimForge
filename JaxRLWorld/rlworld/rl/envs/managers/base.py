from typing import TYPE_CHECKING

from rlworld.rl.envs.utils import NumStepCallsObserver, LearningIterationObserver

if TYPE_CHECKING:
    from rlworld.rl.envs import World


class BaseManager(NumStepCallsObserver, LearningIterationObserver):
    """Base class for all managers."""

    def __init__(self, env: "World"):
        super().__init__()
        self.env = env
        self.device = env.device