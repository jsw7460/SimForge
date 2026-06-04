"""Frozen baselines for Newton domain randomization.

Newton's DR backends in :mod:`rlworld.rl.envs.mdp.events.dr.unified`
need access to the URDF / MJCF *default* values of each per-env model
field (body mass, joint armature, friction axes, ...) so that
``operation="scale"`` and ``operation="add"`` perturbations are always
applied to the baseline rather than to the previously-perturbed value.

The earlier implementation kept these baselines in *module-level*
``DefaultCache`` instances populated lazily on the first DR call.  That
broke when two envs sharing the same process (e.g. ``train`` + ``eval``)
had different ``num_envs``: the first env's snapshot was silently
reused for the second env's index space, which corrupted the wrong
rows under ``operation="scale"`` and raised ``device-side assert`` once
``N_eval > N_train``.

This module captures the baseline once per ``NewtonEnv`` right after
the scene is built (in ``NewtonEnv._post_setup``) and stores the typed
snapshot at ``env._dr_baselines``.  Each Newton DR backend reads the
field it needs directly off ``env._dr_baselines``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import warp as wp

from rlworld.rl.envs.utils.newton.body_cache import get_cache

if TYPE_CHECKING:
    from rlworld.rl.envs.newton.newton_env import NewtonEnv


@dataclass(frozen=True)
class NewtonDRBaselines:
    """Frozen copy of Newton model fields that DR perturbations
    multiply / add to.

    Shapes (per env layout the matching DR backend sees):

    * ``body_mass``                       — ``(num_envs, bodies_per_env)``
    * ``body_com``                        — ``(num_envs, bodies_per_env, 3)``
    * ``joint_armature``                  — ``(num_envs, n_dof_per_env)``
    * ``joint_friction``                  — ``(num_envs, n_dof_per_env)``
    * ``joint_target_ke``                 — ``(num_envs, n_dof_per_env)``
    * ``joint_target_kd``                 — ``(num_envs, n_dof_per_env)``
    * ``shape_material_mu`` (axis 0)      — ``(num_envs, 1, n_shapes_per_env)``
    * ``shape_material_mu_torsional`` (axis 1) — same as above
    * ``shape_material_mu_rolling``   (axis 2) — same as above

    Cloned tensors own their own memory and are unaffected by
    subsequent warp-side writes; treat them as read-only.
    """

    body_mass: torch.Tensor
    body_com: torch.Tensor
    joint_armature: torch.Tensor
    joint_friction: torch.Tensor
    joint_target_ke: torch.Tensor
    joint_target_kd: torch.Tensor
    shape_material_mu: torch.Tensor
    shape_material_mu_torsional: torch.Tensor
    shape_material_mu_rolling: torch.Tensor


def snapshot_newton_dr_baselines(env: NewtonEnv) -> NewtonDRBaselines:
    """Capture the frozen DR baseline snapshot for *env*.

    Must be called exactly once per env, after ``scene_manager.build_scene``
    runs and before any DR term mutates a model field.  ``NewtonEnv._post_setup``
    is the canonical call site; the snapshot must precede
    ``scene_manager.capture()`` so the cached CUDA graph never replays
    a warp write into the baseline tensors.
    """
    model = env.scene_manager.model
    view = env.scene_manager.robot_view
    cache = get_cache(env)
    n_envs = env.num_envs
    bodies_per_env = cache.bodies_per_env

    return NewtonDRBaselines(
        body_mass=wp.to_torch(model.body_mass).reshape(n_envs, bodies_per_env).clone(),
        body_com=wp.to_torch(model.body_com).reshape(n_envs, bodies_per_env, 3).clone(),
        joint_armature=wp.to_torch(view.get_attribute("joint_armature", model)).clone(),
        joint_friction=wp.to_torch(view.get_attribute("joint_friction", model)).clone(),
        joint_target_ke=wp.to_torch(view.get_attribute("joint_target_ke", model)).clone(),
        joint_target_kd=wp.to_torch(view.get_attribute("joint_target_kd", model)).clone(),
        shape_material_mu=wp.to_torch(view.get_attribute("shape_material_mu", model)).clone(),
        shape_material_mu_torsional=wp.to_torch(view.get_attribute("shape_material_mu_torsional", model)).clone(),
        shape_material_mu_rolling=wp.to_torch(view.get_attribute("shape_material_mu_rolling", model)).clone(),
    )
