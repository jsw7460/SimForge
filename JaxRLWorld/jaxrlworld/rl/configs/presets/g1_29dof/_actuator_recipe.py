"""Actuator model shared by the three G1 scene builders.

One decision for all backends. The MuJoCo and Genesis builders used to
hard-code ``Implicit`` (flat) / ``DelayedPD`` (rough) while only the Newton
builder read ``use_ideal_pd_actuator``, so a build that did not go through
``mlp.get_config`` gave Newton an explicit PD with no command delay and the
other two backends the engine's position actuator or the delayed PD.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from jaxrlworld.rl.actuators import (
    ActuatorBaseCfg,
    DelayedPDActuatorCfg,
    IdealPDActuatorCfg,
    ImplicitActuatorCfg,
)

if TYPE_CHECKING:
    from .base import G1FlatConfig


def actuator_recipe(cfg: G1FlatConfig) -> tuple[type[ActuatorBaseCfg], dict[str, int]]:
    """Actuator config class and delay kwargs for ``cfg``.

    ``use_ideal_pd_actuator`` selects the explicit PD with no delay (the
    explicit-PD collection arm, so kp/kd map onto a clean torque path).
    Otherwise rough terrain keeps the DelayedPD sim2real modeling (command
    delay 0..2 substeps) and flat follows the Mjlab-Velocity-Flat-Unitree-G1
    reference: implicit (engine-side) position actuators, which also avoid
    the per-substep torch PD that cost ~0.4 ms/step at 16384 envs.
    """
    if cfg.use_ideal_pd_actuator:
        return IdealPDActuatorCfg, {}
    if cfg.use_rough_terrain:
        return DelayedPDActuatorCfg, {"min_delay": 0, "max_delay": 2}
    return ImplicitActuatorCfg, {}
