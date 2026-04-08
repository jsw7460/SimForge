"""Compat shim: delegates to the unified ``presets/g1_29dof/mlp.py``.

Phase B migration: kept so existing scripts that import from
``presets.g1_29dof.newton.mlp`` continue to work. A later phase will
update those callers and remove this shim.

NOTE: The original run_name was "G1_29Dof_NT_MLP" (different from the
unified default "G1_29Dof_Newton_MLP"). Preserving the legacy string
here so wandb runs land in the same logical group.
"""

from rlworld.rl.configs.presets.g1_29dof.base import G1FlatConfig


def get_config():
    cfg = G1FlatConfig(
        sim_type="newton",
        actor_class_name="MLPActor",
        run_name="G1_29Dof_NT_MLP",
    )
    return cfg.build()
