"""K1 fast-locomotion variant of the G1 recipe (target: 2 m/s).

Identical task/rewards/DR/terminations to :class:`K1G1RecipeConfig`;
only the velocity command pipeline changes:

* The ``velocity`` command term is swapped for
  :class:`AdaptiveVelocityCommandTermCfg`: the active ``lin_vel_x``
  range starts at the recipe's (-1, 1) and is widened bin-by-bin toward
  ``lin_vel_x_final_range`` (default (-1, 2)) by a performance gate,
  while sampling is frontier-weighted so hard (fast) bins are drawn
  more often than uniform would.
* A ``command_velocity_range`` curriculum term drives the widening and
  logs ``Curriculum/vx_range/{vx_max,vx_min,frontier_*_success}`` to
  wandb — a stalling ``vx_max`` is the direct readout of where the
  plant (T-N torque derating) or the reward tuning caps the speed.

Deliberately NOT changed in v1 (revisit if vx_max stalls below 2):
tracking ``std`` (0.25 — at 2 m/s a 0.5 m/s error already zeroes the
reward, which is the desired pressure), resampling time, push/DR
ranges.

Train:
    jaxpy -m jaxrlworld.scripts.k1.newton.joystick_fast
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields

from jaxrlworld.rl.configs.curriculums import (
    CurriculumManagerConfig,
    CurriculumTermConfig,
)
from jaxrlworld.rl.envs.managers.common.command_term import VelocityCommandTermCfg
from jaxrlworld.rl.envs.mdp.commands.adaptive_velocity import AdaptiveVelocityCommandTermCfg
from jaxrlworld.rl.envs.mdp.curriculums import command_velocity_range

from .g1_recipe import K1G1RecipeConfig


@dataclass
class K1FastConfig(K1G1RecipeConfig):
    """G1 recipe + adaptive velocity-command curriculum toward 2 m/s."""

    # Final command envelope; ``lin_vel_x_range`` (inherited, (-1, 1))
    # stays the INITIAL active range.
    lin_vel_x_final_range: tuple = (-1.0, 2.0)
    num_vx_bins: int = 12
    vx_adaptive_fraction: float = 0.5
    # Tour success = mean |vx_cmd - vx_meas| below this [m/s].
    vx_success_error_threshold: float = 0.3
    # Expansion gate: frontier-bin success EMA / minimum tour count.
    vx_promote_threshold: float = 0.8
    vx_min_tours: float = 300.0

    _RUN_NAMES = {
        "newton": "K1_Newton_Fast",
        "mujoco": "K1_Mujoco_Fast",
        "genesis": "K1_Genesis_Fast",
    }

    def _build_command_config(self):
        cfg = super()._build_command_config()
        base_v = cfg.terms["velocity"]
        # Carry every plain velocity field over verbatim so this stays
        # in sync with the base builder (heading control, standing
        # fraction, resample timing, ...).
        cfg.terms["velocity"] = AdaptiveVelocityCommandTermCfg(
            **{f.name: getattr(base_v, f.name) for f in fields(VelocityCommandTermCfg)},
            lin_vel_x_final_range=self.lin_vel_x_final_range,
            num_vx_bins=self.num_vx_bins,
            adaptive_fraction=self.vx_adaptive_fraction,
            success_error_threshold=self.vx_success_error_threshold,
        )
        return cfg

    def _build_curriculum_config(self) -> CurriculumManagerConfig:
        promote_threshold = self.vx_promote_threshold
        min_tours = self.vx_min_tours

        @dataclass
        class _CurriculumCfg(CurriculumManagerConfig):
            vx_range: CurriculumTermConfig = field(
                default_factory=lambda: CurriculumTermConfig(
                    func=command_velocity_range,
                    params={
                        "command_name": "velocity",
                        "promote_threshold": promote_threshold,
                        "min_tours": min_tours,
                    },
                )
            )

        return _CurriculumCfg()

    def build(self):
        cfgs = super().build()
        cfgs.curriculum = self._build_curriculum_config()
        return cfgs
