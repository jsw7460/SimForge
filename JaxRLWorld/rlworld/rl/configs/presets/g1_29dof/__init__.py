"""
G1 29-DOF flat terrain locomotion configs.

Single unified config across simulator backends. Choose the simulator
via ``sim_type`` (or pass ``sim=`` to ``mlp.get_config``):

    from rlworld.rl.configs.presets.g1_29dof.mlp import get_config
    cfgs = get_config(sim="newton")  # or "genesis" / "mujoco"

Or directly:

    from rlworld.rl.configs.presets.g1_29dof.base import G1FlatConfig
    cfgs = G1FlatConfig(sim_type="newton").build()
"""

__all__ = ["base", "mlp"]
