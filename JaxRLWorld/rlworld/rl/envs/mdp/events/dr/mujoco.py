"""SysID-aligned setters for mjlab/MuJoCo.

Counterparts to ``dr/newton.py``'s ``set_joint_friction`` /
``set_foot_friction``. Each writes a fixed identified value (optionally
with a narrow DR band) into the live mjwarp model arrays at every
reset, so a Newton-trained-then-MuJoCo-deployed policy sees the same
identified physics as the Stage 1b SysID inferred.

The mjwarp model exposes ``geom_friction`` / ``dof_frictionloss`` as
torch tensors of shape ``(num_envs, ...)``; we index into ``env_ids``
the same way ``unified`` DR terms do. The foot-friction setter also
mirrors the identified value onto every ground geom by name so
MuJoCo's max() contact-friction combine rule yields the foot's value
(not the ground's higher default) at every robot-foot contact pair.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ._utils import sample

if TYPE_CHECKING:
    from rlworld.rl.envs.mujoco.mjlab_env import MjlabEnv


def set_joint_friction(
    env: MjlabEnv,
    env_ids: torch.Tensor,
    value: float,
    dr_scale: tuple[float, float] | None = None,
) -> None:
    """Write identified joint Coulomb friction to every actuated DOF.

    SysID-aligned counterpart to ``newton.set_joint_friction``. The
    mjwarp ``model.dof_frictionloss`` is a per-env torch tensor of
    shape ``(num_envs, ndof)``; we overwrite the rows in ``env_ids``
    each reset.

    Args:
        value: Identified joint Coulomb friction (the SysID center).
        dr_scale: Optional ``(lo, hi)`` multiplicative band. ``None``
            writes the value exactly; otherwise every reset writes
            ``value * uniform(lo, hi)`` per env (e.g. ``(0.9, 1.1)``
            for ±10 % margin around the identified value).
    """
    if len(env_ids) == 0:
        return

    model = env.scene_manager.model
    friction = model.dof_frictionloss  # (num_envs, ndof) torch tensor
    if dr_scale is None:
        friction[env_ids] = float(value)
    else:
        scale = sample(
            friction[env_ids].shape,
            *dr_scale,
            env.device,
            "uniform",
        )
        friction[env_ids] = float(value) * scale


def set_foot_friction(
    env: MjlabEnv,
    env_ids: torch.Tensor,
    value: float,
    foot_geom_names: tuple[str, ...] = (
        "FR_foot_collision",
        "FL_foot_collision",
        "RR_foot_collision",
        "RL_foot_collision",
    ),
    ground_name_tokens: tuple[str, ...] = ("terrain", "ground", "plane"),
    dr_scale: tuple[float, float] | None = None,
) -> None:
    """Write identified foot-slide friction onto foot geoms AND every
    ground geom (max() combine-rule workaround).

    SysID-aligned counterpart to ``newton.set_foot_friction``. Lifted
    almost verbatim from the collect-side
    ``Go2GaitConditionedSysIDCollectConfig.runtime_setup`` so the
    training-time and collect-time foot mu are bit-identical.

    Args:
        value: Identified foot slide friction (mu) — the SysID center.
        foot_geom_names: Robot foot collision geom names to override.
        ground_name_tokens: Substrings used to detect ground/terrain
            geoms by name (case-insensitive). The identified value is
            mirrored onto all matching geoms so the MuJoCo max() rule
            doesn't let the ground's default mu dominate the contact.
        dr_scale: Optional ``(lo, hi)`` multiplicative band, applied
            uniformly across all foot+ground geoms touched. ``None``
            writes the exact identified value.
    """
    if len(env_ids) == 0:
        return

    robot = env.scene_manager.scene["robot"]
    local_geom_ids, _ = robot.find_geoms(foot_geom_names)
    foot_geom_ids = robot.indexing.geom_ids[local_geom_ids]

    mj_model = env.scene_manager.mj_model
    ground_indices = [
        i for i in range(mj_model.ngeom) if any(tok in mj_model.geom(i).name.lower() for tok in ground_name_tokens)
    ]
    all_geom_ids = torch.cat(
        [
            torch.as_tensor(foot_geom_ids, device=env.device, dtype=torch.long),
            torch.as_tensor(ground_indices, device=env.device, dtype=torch.long),
        ]
    )

    friction = env.scene_manager.model.geom_friction  # (num_envs, ngeom, 3)
    if dr_scale is None:
        friction[env_ids[:, None], all_geom_ids[None, :], 0] = float(value)
    else:
        scale = sample(
            (len(env_ids), len(all_geom_ids)),
            *dr_scale,
            env.device,
            "uniform",
        )
        friction[env_ids[:, None], all_geom_ids[None, :], 0] = float(value) * scale
