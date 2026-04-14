"""Sim-agnostic curriculum functions (reward/termination stage apply).

Ported from ``mjlab/envs/mdp/curriculums.py`` so existing mjlab-style
curriculum configs (list of stage dicts with ``step`` + ``weight`` /
``params`` keys) work unchanged in JaxRLWorld.
"""

from .step_stages import (
    RewardCurriculumStage,
    TerminationCurriculumStage,
    reward_curriculum,
    termination_curriculum,
)

__all__ = [
    "RewardCurriculumStage",
    "TerminationCurriculumStage",
    "reward_curriculum",
    "termination_curriculum",
]
