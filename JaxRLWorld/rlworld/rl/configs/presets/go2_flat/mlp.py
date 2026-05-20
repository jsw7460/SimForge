"""Go2 flat-terrain locomotion with MLP actor.

Single entry point for all three simulators (Newton, Genesis, MuJoCo).
The simulator is selected via the ``sim`` argument to ``get_config``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

from .base import Go2FlatConfig

if TYPE_CHECKING:
    from rlworld.rl.configs.genesis_config_classes import GenesisConfigsForRun
    from rlworld.rl.configs.mujoco_config_classes import MujocoConfigsForRun
    from rlworld.rl.configs.newton_config_classes import NewtonConfigsForRun


@overload
def get_config(sim: Literal["newton"] = ...) -> NewtonConfigsForRun: ...
@overload
def get_config(sim: Literal["mujoco"]) -> MujocoConfigsForRun: ...
@overload
def get_config(sim: Literal["genesis"]) -> GenesisConfigsForRun: ...
def get_config(sim: str = "newton"):
    """Build the Go2 flat MLP config for the specified simulator.

    Args:
        sim: Simulator backend, one of ``"newton"``, ``"genesis"``, or
            ``"mujoco"``.

    Returns:
        A built ``ConfigsForRun`` of the appropriate sim-specific type
        (``NewtonConfigsForRun``, ``GenesisConfigsForRun``, or
        ``MujocoConfigsForRun``), ready for ``with_cli_overrides()``.
    """
    sim = sim.lower()
    run_name_suffix = {"newton": "Newton", "genesis": "Genesis", "mujoco": "Mujoco"}[sim]
    cfg = Go2FlatConfig(
        sim_type=sim,
        run_name=f"Go2_{run_name_suffix}_MLP",
    )
    return cfg.build()
