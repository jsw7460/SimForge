"""
Go2 flat terrain locomotion configs.

Single unified config across simulator backends. Choose the simulator
via ``sim_type`` (or pass ``sim=`` to ``mlp.get_config``):

    from rlworld.rl.configs.presets.go2_flat.mlp import get_config
    cfgs = get_config(sim="newton")  # or "genesis" / "mujoco"

Or directly:

    from rlworld.rl.configs.presets.go2_flat.base import Go2FlatConfig
    cfgs = Go2FlatConfig(sim_type="newton").build()
"""

__all__ = ["base", "mlp"]
