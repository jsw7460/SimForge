from collections import deque
from itertools import chain
from typing import Dict, Any, Set, Deque, Union, TYPE_CHECKING

import torch

from rlworld.rl.configs import CurriculumConfig

if TYPE_CHECKING:
    from rlworld.rl.runners import (
        OnPolicyRunner,
    )


class CurriculumManager:
    """Abstract base class for curriculum learning management.

    This class handles the progression of difficulty levels based on specified criteria
    and applies corresponding environment settings.
    """

    def __init__(
        self,
        runner: Union[
            "OnPolicyRunner",
        ],
        curriculum_cfg: CurriculumConfig
    ):
        """Initialize curriculum manager."""
        self.runner = runner
        self.env = self.runner.env
        self.cfg = curriculum_cfg
        self._current_level = self.cfg.initial_level
        self._steps_in_level = 0

        # Initialize histories and criteria
        self._criterion_histories: Dict[str, Deque[bool]] = {}
        self._history_idx = torch.zeros(self.env.num_envs, device=self.env.device, dtype=torch.long)
        self._registered_criteria: Set[str] = set()

        self._initialize_criteria()
        self._verify_implementation()

    def get_curriculum_info(self) -> Dict[str, Any]:
        return {
            "current_level": self.current_level,
            "steps_in_level": self.steps_in_level,
        }

    @property
    def current_level(self) -> int:
        """Current difficulty level."""
        return self._current_level

    @property
    def steps_in_level(self) -> int:
        """Number of steps spent in current level."""
        return self._steps_in_level

    def _initialize_criteria(self) -> None:
        """Initialize all criteria specified in config."""
        for name in self.cfg.criterion:
            self.register_criterion(name)

    def register_criterion(self, name: str) -> None:
        """Register a new criterion for tracking.

        Args:
            name: Criterion name (should match with _check_{name} method)
        """
        self._registered_criteria.add(name)
        self._criterion_histories[name] = deque(maxlen=self.cfg.eval_window_size)

    def _verify_implementation(self) -> None:
        """Verify all required methods are implemented."""
        self._verify_criteria_methods()
        self._verify_component_methods()

    def _verify_criteria_methods(self) -> None:
        """Verify criterion check methods exist."""
        missing_methods = [
            criterion for criterion in self._registered_criteria
            if not hasattr(self, f'_check_{criterion}')
        ]
        if missing_methods:
            raise NotImplementedError(
                f"Missing check methods for criteria: {missing_methods}"
            )

    def _verify_component_methods(self) -> None:
        """Verify component application methods exist."""
        missing_methods = [
            component for component in self.cfg.curriculum_components
            if not hasattr(self, f'_apply_{component}')
        ]
        if missing_methods:
            raise NotImplementedError(
                f"Missing apply methods for components: {missing_methods}"
            )

    def update_difficulty(self, episode_metrics: Dict[str, Any]) -> bool:
        """Update difficulty level based on episode metrics.

        Args:
            episode_metrics: Dictionary containing episode performance metrics

        Returns:
            bool: Whether difficulty level was updated
        """
        if not self._should_update_difficulty():
            return False

        self._steps_in_level += 1
        self._update_metrics(episode_metrics)

        if self._check_level_completion():
            self._advance_level()
            return True

        return False

    def _should_update_difficulty(self) -> bool:
        """Check if difficulty update should be considered."""
        return (self.cfg.enable and
                self._current_level < self.cfg.max_level)

    def _update_metrics(self, episode_metrics: Dict[str, Any]) -> None:
        """Update tracking metrics with latest episode results."""
        for criterion in self._registered_criteria:
            check_method = getattr(self, f'_check_{criterion}')
            is_met = check_method(episode_metrics)
            self._criterion_histories[criterion].append(is_met)

    def _check_level_completion(self) -> bool:
        """Check if all criteria are met for level advancement."""
        history_deques = self._criterion_histories.values()
        ret = (all(chain.from_iterable(history_deques)) and
               self._all_histories_complete())
        return ret

    def _all_histories_complete(self) -> bool:
        """Check if all criterion histories have enough data."""
        ret = all(
            len(deque) == self.cfg.eval_window_size
            for deque in self._criterion_histories.values()
        )

        return ret

    def _advance_level(self) -> None:
        """Advance to next difficulty level."""
        self._increment_level()
        self._apply_level_settings()
        self._reset_tracking()

    def _increment_level(self) -> None:
        """Increment current level within bounds."""
        self._current_level = min(self._current_level + 1, self.cfg.max_level)

    def _reset_tracking(self) -> None:
        """Reset tracking metrics for new level."""
        self._steps_in_level = 0
        for history in self._criterion_histories.values():
            history.clear()

    def _apply_level_settings(self) -> None:
        """Apply settings for current difficulty level."""
        for component in self.cfg.curriculum_components:
            apply_method = getattr(self, f'_apply_{component}')
            apply_method()


class Go2Curriculum(CurriculumManager):
    """Curriculum manager for Go2 environment."""

    def _check_tracking_lin_vel_xy(self, episode_metrics: Dict[str, Any]) -> bool:
        """Check linear velocity tracking performance.

        Args:
            episode_metrics: Episode performance metrics

        Returns:
            bool indicating if criterion is met
        """
        tracking_performance = episode_metrics["reward_stats"]["tracking_lin_vel"]["mean"]
        threshold = self.cfg.criterion["tracking_lin_vel_xy"]

        return tracking_performance > threshold

    def _check_mean_return(self, episode_metrics: Dict[str, Any]) -> bool:
        """Check mean episode return.

        Args:
            episode_metrics: Episode performance metrics

        Returns:
            bool indicating if criterion is met
        """
        mean_return = episode_metrics["reward_stats"]["total_reward"]["mean"]
        threshold = self.cfg.criterion["mean_return"]

        return mean_return > threshold

    def _check_iteration(self, episode_metrics: Dict[str, Any]) -> bool:
        return self.runner.current_learning_iteration > self.cfg.criterion["iteration"]

    def _apply_command_ranges(self) -> None:
        """Apply command range settings for current level."""
        commands = self.cfg.curriculum_components["command_ranges"][self._current_level]

        self.env.command_cfg.lin_vel_x_range = commands["lin_vel_x"]
        self.env.command_cfg.lin_vel_y_range = commands["lin_vel_y"]
        self.env.command_cfg.ang_vel_range = commands["ang_vel"]

    def _apply_magnetic_forces(self) -> None:
        """Apply magnetic force settings for current level."""
        pass

    def _apply_reward_scales(self) -> None:
        """Apply reward scale settings for current level."""
        self.env.reward_calculator.reward_scales.update(
            **self.cfg.curriculum_components["reward_scales"][self._current_level]
        )
        self.env.reward_calculator._register_rewards()

    def _apply_stability_thresholds(self) -> None:
        """Apply stability threshold settings for current level."""
        pass


class MarvelCurriculum(CurriculumManager):
    def _check_to_update(self) -> torch.Tensor:
        """..."""
