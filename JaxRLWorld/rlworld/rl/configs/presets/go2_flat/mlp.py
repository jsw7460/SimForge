"""Go2 flat-terrain locomotion with MLP actor.

Single entry point for all three simulators (Newton, Genesis, MuJoCo).
The simulator is selected via the ``sim`` argument to ``get_config``.
"""

from __future__ import annotations

from .base import Go2FlatConfig


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
        actor_class_name="MLPActor",
        run_name=f"Go2_{run_name_suffix}_MLP",
    )
    return cfg.build()
