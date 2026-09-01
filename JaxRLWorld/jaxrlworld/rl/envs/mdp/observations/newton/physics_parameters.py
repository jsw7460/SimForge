"""Newton physics-parameter observations.

Exposes the *current* per-env physical parameters of the robot (body mass,
COM offset, joint armature / friction, PD target gains, contact friction
axes) as a flat observation vector. Intended as a privileged observation —
e.g. feeding the ground-truth dynamics of each (domain-randomized) env to
an asymmetric critic, an RMA-style context encoder, or a system-id head.

Values are read live from the simulator on every call, so they reflect any
domain randomization applied at reset. They are returned raw (no
normalization); scale them with the standard observation pipeline or a
downstream running normalizer.

The supported field names mirror :class:`NewtonDRBaselines` so that a
caller can expose exactly the fields it randomizes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

from jaxrlworld.rl.envs.utils.newton.body_cache import get_cache

if TYPE_CHECKING:
    from jaxrlworld.rl.envs import NewtonEnv

# Field names that map onto a current Newton model field, mirroring
# ``NewtonDRBaselines``. ``body_mass`` / ``body_com`` live directly on the
# model; the rest are per-dof / per-shape fields read through the view.
_MODEL_FIELDS = ("body_mass", "body_com")
_VIEW_FIELDS = (
    "joint_armature",
    "joint_friction",
    "joint_target_ke",
    "joint_target_kd",
    "shape_material_mu",
    "shape_material_mu_torsional",
    "shape_material_mu_rolling",
)
_SUPPORTED = _MODEL_FIELDS + _VIEW_FIELDS


def _read_attribute(env: NewtonEnv, name: str) -> torch.Tensor:
    """Read the current value of one model field as ``(num_envs, ...)``."""
    model = env.scene_manager.model
    n_envs = env.num_envs
    if name == "body_mass":
        bodies_per_env = get_cache(env).bodies_per_env
        return wp.to_torch(model.body_mass).reshape(n_envs, bodies_per_env)
    if name == "body_com":
        bodies_per_env = get_cache(env).bodies_per_env
        return wp.to_torch(model.body_com).reshape(n_envs, bodies_per_env, 3)
    # PD gains: with EXPLICIT actuators (IdealPD/...) the joint runs in direct-
    # torque mode and the solver's joint_target_ke/kd are zeroed/unused — the gains
    # that actually act are the actuators' own stiffness/damping (also what DR
    # randomizes). Report THOSE so the privileged obs reflects the real per-env
    # gains. Implicit actuators (internal PD) genuinely use joint_target_ke/kd, so
    # fall through to the solver field for them.
    if name in ("joint_target_ke", "joint_target_kd") and getattr(env.act_manager, "has_explicit_actuators", False):
        attr = "stiffness" if name == "joint_target_ke" else "damping"
        cols = [getattr(actuator, attr) for actuator, _ in env.act_manager.actuators]
        return torch.cat(cols, dim=-1)  # [n_envs, total_actuated_joints]
    view = env.scene_manager.robot_view
    return wp.to_torch(view.get_attribute(name, model))


def physics_parameters(env: NewtonEnv, attributes: list[str]) -> torch.Tensor:
    """Current per-env physical parameters, flattened and concatenated.

    Args:
        env: Newton environment.
        attributes: ordered list of model field names to expose. Each must be
            one of :data:`_SUPPORTED` (the fields mirrored by
            ``NewtonDRBaselines``).

    Returns:
        Tensor of shape ``(num_envs, total_dim)`` holding the requested fields,
        read live from the sim and concatenated in the given order. Raw values.
    """
    cols: list[torch.Tensor] = []
    for name in attributes:
        if name not in _SUPPORTED:
            raise ValueError(f"physics_parameters: unsupported attribute {name!r}; expected one of {_SUPPORTED}")
        cols.append(_read_attribute(env, name).reshape(env.num_envs, -1).float())
    return torch.cat(cols, dim=-1)
