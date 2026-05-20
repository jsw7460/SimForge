"""G1 29-DOF flat-terrain locomotion with MLP actor.

Single entry point for all three simulators (Newton, Genesis, MuJoCo).
The simulator is selected via the ``sim`` argument to ``get_config``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

from .base import G1FlatConfig

if TYPE_CHECKING:
    from rlworld.rl.configs.genesis_config_classes import GenesisConfigsForRun
    from rlworld.rl.configs.mujoco_config_classes import MujocoConfigsForRun
    from rlworld.rl.configs.newton_config_classes import NewtonConfigsForRun

# Per-sim default run name preserves the existing strings so wandb runs
# stay grouped under the same names as before the unification.
_DEFAULT_RUN_NAMES = {
    "newton": "G1_29Dof_Newton_MLP",
    "genesis": "G1_29Dof_Genesis_MLP",
    "mujoco": "G1_29Dof_Mujoco_MLP",
}


@overload
def get_config(sim: Literal["newton"] = ...) -> NewtonConfigsForRun: ...
@overload
def get_config(sim: Literal["mujoco"]) -> MujocoConfigsForRun: ...
@overload
def get_config(sim: Literal["genesis"]) -> GenesisConfigsForRun: ...
def get_config(sim: str = "newton"):
    """Build the G1 29-DOF flat MLP config for the specified simulator.

    Args:
        sim: Simulator backend, one of ``"newton"``, ``"genesis"``, or
            ``"mujoco"``.

    Returns:
        A built ``ConfigsForRun`` of the appropriate sim-specific type.
    """
    sim = sim.lower()
    cfg = G1FlatConfig(
        sim_type=sim,
        run_name=_DEFAULT_RUN_NAMES[sim],
    )
    return cfg.build()
