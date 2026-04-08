"""Compat shim: delegates to the unified ``presets/g1_29dof/mlp.py``.

Phase B migration: kept so existing scripts that import from
``presets.g1_29dof.mujoco.mlp`` continue to work. A later phase will
update those callers and remove this shim.
"""

from rlworld.rl.configs.presets.g1_29dof.base import G1FlatConfig


def get_config():
    cfg = G1FlatConfig(
        sim_type="mujoco",
        actor_class_name="MLPActor",
        run_name="G1_29Dof_Mujoco_MLP",
    )
    return cfg.build()
