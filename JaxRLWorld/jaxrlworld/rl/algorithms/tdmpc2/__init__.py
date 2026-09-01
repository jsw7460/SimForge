"""TD-MPC2: Scalable, Robust World Models for Continuous Control."""

from jaxrlworld.rl.algorithms.tdmpc2.metrics import TDMPC2Metrics
from jaxrlworld.rl.algorithms.tdmpc2.tdmpc2 import TDMPC2

__all__ = ["TDMPC2", "TDMPC2Metrics"]
