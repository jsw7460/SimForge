"""Observation terms for the K1 joystick task.

The actor/critic groups are otherwise assembled from the public common
proprioception terms; these cover the K1-specific pieces.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from rlworld.rl.configs.scene.entity_selector import ResolvedEntity, SceneEntitySelector
from rlworld.rl.envs.utils import EnvStepCache

if TYPE_CHECKING:
    from rlworld.rl.envs.world import World

_DEFAULT_SELECTOR = SceneEntitySelector(name="robot")


def velocity_command(env: World, command_term: str = "velocity") -> torch.Tensor:
    """The 3-D velocity command alone.

    ``all_commands`` would concatenate every command term (including the
    gait-phase state), so the velocity term is read explicitly.
    """
    return env.command_manager.get_term(command_term).command


@EnvStepCache()
def base_ang_vel_w(env: World, asset_cfg: ResolvedEntity = _DEFAULT_SELECTOR) -> torch.Tensor:
    """Root angular velocity in the WORLD frame (upstream ``global_angvel``)."""
    return env.get_robot_data(asset_cfg.name).root_link_ang_vel_w


def gait_phase_encoding(env: World, command_term: str = "gait_phase") -> torch.Tensor:
    """``[cos(phi_L), cos(phi_R), sin(phi_L), sin(phi_R)]``.

    Reads :attr:`K1GaitPhaseCommandTerm.phase_obs` — the pre-advance
    snapshot equal to the phase this step's rewards used, matching the
    upstream within-step pairing (see the command term's docstring).
    """
    phase = env.command_manager.get_term(command_term).phase_obs
    return torch.cat([torch.cos(phase), torch.sin(phase)], dim=1)


@EnvStepCache()
def feet_lin_vel_w(env: World, asset_cfg: ResolvedEntity = _DEFAULT_SELECTOR) -> torch.Tensor:
    """World-frame linear velocity of the feet bodies, flattened.

    ``asset_cfg.body_names`` selects the feet (left, right) → (N, 6).
    """
    vel = env.get_robot_data(asset_cfg.name).body_lin_vel_w_by_ids(asset_cfg.body_ids)
    return vel.reshape(env.num_envs, -1)


@EnvStepCache()
def feet_contact(env: World, contact_group: str) -> torch.Tensor:
    """Per-foot boolean ground contact as float — upstream's privileged
    ``contact`` entry (any collision-sphere of the foot touching)."""
    return env.contact_manager.is_contact(contact_group).float()


@EnvStepCache()
def feet_air_time(env: World, contact_group: str) -> torch.Tensor:
    """Per-foot current air time from the contact manager.

    Semantics note: the contact manager integrates air time at PHYSICS
    rate without upstream's control-rate OR-filtered bookkeeping, so
    values can differ from upstream by sub-control-step amounts. This
    is a privileged (critic-only) observation; the reward path uses the
    exact stateful bookkeeping in ``rewards.k1_locomotion.K1FeetAirTime``.
    """
    return env.contact_manager.current_air_time(contact_group)
