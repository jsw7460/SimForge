"""Unified termination conditions using the RobotData interface.

All functions accept any ``World`` subclass and read state exclusively
through ``env.get_robot_data(asset_cfg.name)``, making them simulator-agnostic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from rlworld.rl.configs.scene.entity_selector import ResolvedEntity, SceneEntitySelector
from rlworld.rl.configs.terminations import TerminationResult
from rlworld.rl.utils.quat_utils import quat_to_euler_wxyz

if TYPE_CHECKING:
    from rlworld.rl.envs.world import World


_DEFAULT_SELECTOR = SceneEntitySelector(name="robot")


def energy_termination(
    env: World,
    threshold: float = float("inf"),
    skip_steps: int = 0,
    asset_cfg: ResolvedEntity = _DEFAULT_SELECTOR,
) -> TerminationResult:
    """Terminate when instantaneous mechanical power exceeds a threshold.

    Mirrors mjlab_playground getup ``energy_termination``:

        power = sum(|applied_torque * joint_vel|)
        terminate = (power > threshold) & (episode_length_buf >= skip_steps)

    ``applied_torque`` is read from :attr:`RobotData.applied_torque` —
    the simulator's per-DOF ``qfrc_actuator`` after PD-law evaluation
    and effort-limit clipping. Works uniformly for implicit (simulator
    internal PD) and explicit (Python-computed) actuator modes. This
    diverges from an older implementation that read
    ``act_manager.applied_torque`` (which is zero in pure-implicit
    mode and silently disabled the termination).

    The ``skip_steps`` argument suppresses the check during the initial
    settle / landing phase where large impact torques are expected and
    would fire the termination spuriously. Set it to the same value as
    ``act_manager.settle_steps`` (or higher) when used together with
    the settle hook.

    Args:
        env: Any environment with ``get_robot_data``.
        threshold: Maximum allowed mechanical power in watts. Set to
            ``float("inf")`` to disable the check (e.g. as the initial
            value of a curriculum schedule).
        skip_steps: Number of post-reset control steps during which the
            termination is suppressed.
        entity_name: Entity to query for joint velocity / torque.

    Returns:
        TerminationResult indicating which envs exceeded the threshold.
    """
    rd = env.get_robot_data(asset_cfg.name)
    power = torch.sum(torch.abs(rd.applied_torque * rd.joint_vel), dim=-1)
    exceeded = power > threshold
    if skip_steps > 0:
        exceeded = exceeded & (env.episode_length_buf >= skip_steps)
    return TerminationResult(exceeded)


def roll_pitch_violation(
    env: World,
    roll_threshold_degree: float = 15.0,
    pitch_threshold_degree: float = 15.0,
    asset_cfg: ResolvedEntity = _DEFAULT_SELECTOR,
) -> TerminationResult:
    """Terminate if robot's roll or pitch exceeds safe thresholds.

    Args:
        env: Any environment with ``get_robot_data``.
        roll_threshold_degree: Maximum allowed roll angle in degrees.
        pitch_threshold_degree: Maximum allowed pitch angle in degrees.
        entity_name: Name of the entity to check.

    Returns:
        TerminationResult indicating which envs should reset.
    """
    quat_wxyz = env.get_robot_data(asset_cfg.name).root_link_quat_w
    euler = quat_to_euler_wxyz(quat_wxyz)  # (num_envs, 3) radians

    roll_deg = torch.abs(euler[:, 0]) * (180.0 / torch.pi)
    pitch_deg = torch.abs(euler[:, 1]) * (180.0 / torch.pi)

    violated = (roll_deg > roll_threshold_degree) | (pitch_deg > pitch_threshold_degree)
    return TerminationResult(violated)


def terrain_out_of_bounds(
    env: World,
    margin: float = 0.5,
    asset_cfg: ResolvedEntity = _DEFAULT_SELECTOR,
) -> TerminationResult:
    """Terminate when the robot nears the edge of a generated terrain.

    A finite terrain mesh has a cliff at its boundary; without this the
    robot walks off and falls into the void, polluting training. Mirrors
    IsaacLab's ``terrain_out_of_bounds`` and mjlab's
    ``out_of_terrain_bounds``: terminate when ``|root_xy|`` exceeds the
    terrain half-extent minus ``margin``, so the episode resets (to the
    terrain centre) *before* the robot reaches the edge.

    No-op when there is no generated terrain (flat plane / no terrain
    data on the scene manager), so it is safe to register unconditionally.

    Args:
        env: Any environment with ``get_robot_data`` and a scene manager.
        margin: Safety distance (m) inside the terrain edge at which to
            terminate.
        asset_cfg: Entity whose root position is checked.
    """
    # ``TerrainImporter`` exposes the generated terrain via ``terrain.data``;
    # for flat-plane scenes ``data`` is ``None`` and we short-circuit.
    terrain = env.scene_manager.terrain
    if terrain.data is None:
        return TerminationResult(torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))

    # Terrain is centred on the origin, so the half-extent bounds the field.
    half_x, half_y = terrain.half_extent
    limit_x = max(0.0, half_x - margin)
    limit_y = max(0.0, half_y - margin)

    root_xy = env.get_robot_data(asset_cfg.name).root_link_pos_w[:, :2]
    out = (root_xy[:, 0].abs() > limit_x) | (root_xy[:, 1].abs() > limit_y)
    return TerminationResult(out)


def illegal_contact(
    env: World,
    contact_group: str,
    force_threshold: float = 1.0,
) -> TerminationResult:
    """Terminate when any body tracked by ``contact_group`` registers a
    ground-contact force above ``force_threshold``.

    Reads ``env.contact_manager`` (a sim-agnostic interface), so it works
    on any backend. Point it at a contact group whose primary is the set
    of bodies that must NOT touch the ground — e.g. the go2 preset's
    ``"body_ground_contact"`` group (trunk + hips + thighs, feet and calf
    excluded). The episode then terminates the moment a non-foot body hits
    the ground, the standard "illegal contact" fall condition (mirrors
    IsaacLab's ``illegal_contact`` and mjlab's contact terminations).

    Uses substep force history when the backend provides it (MuJoCo with
    ``history_length > 0`` — a hit on any substep counts); otherwise falls
    back to the instantaneous per-body contact force (Newton / Genesis).

    Args:
        env: Any ``World`` subclass with a ``contact_manager``.
        contact_group: Registered contact group name whose tracked bodies
            should trigger termination on ground contact.
        force_threshold: Contact-force magnitude (N) above which a body
            counts as touching. ``1.0`` cleanly separates "no contact"
            (~0 N) from a real contact (hundreds of N) — verified for the
            go2 ``body_ground_contact`` group.

    Returns:
        ``TerminationResult`` of shape ``(num_envs,)`` — True where any
        tracked body exceeds the force threshold.
    """
    history = env.contact_manager.contact_force_history(contact_group)
    if history is not None:
        # (B, N, H, 3) -> (B, N, H) -> (B, N): hit if any substep crosses.
        force_mag = torch.norm(history, dim=-1)
        hit = (force_mag > force_threshold).any(dim=2)
    else:
        forces = env.contact_manager.contact_force(contact_group)  # (B, N, 3)
        hit = torch.norm(forces, dim=-1) > force_threshold  # (B, N)
    return TerminationResult(hit.any(dim=-1))
