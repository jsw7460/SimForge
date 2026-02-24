from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from genesis.engine.entities import RigidEntity

if TYPE_CHECKING:
    from rlworld.rl.envs import RLEnv


def initialize_dof_pos(
    env: RLEnv,
    env_ids: torch.Tensor,
    entity_name: str = "robot",
    zero_velocity: bool = True
):
    if len(env_ids) == 0:
        return

    dof_pos = env.act_manager.offset
    robot: RigidEntity = env.scene_manager[entity_name]
    robot.set_dofs_position(
        position=dof_pos[env_ids],
        dofs_idx_local=env.act_manager.actuated_dof_ids,
        zero_velocity=zero_velocity,
        envs_idx=env_ids
    )
    if zero_velocity:
        robot.zero_all_dofs_velocity(env_ids)


def initialize_pos_quat(
    env: RLEnv,
    env_ids: torch.Tensor,
    base_init_pos: list[float],
    base_init_quat: list[float],
    entity_name: str = "robot",
):
    """Initialize robot base position and quaternion.

    Args:
        env: Environment instance
        env_ids: Environment indices to reset
        base_init_pos: [x, y, z] position
        base_init_quat: [w, x, y, z] quaternion
        entity_name: Name of robot entity
    """
    if len(env_ids) == 0:
        return

    robot: RigidEntity = env.scene_manager[entity_name]

    base_pos = torch.tensor(base_init_pos, device=env.device)
    base_quat = torch.tensor(base_init_quat, device=env.device)

    robot.set_pos(base_pos, envs_idx=env_ids)
    robot.set_quat(base_quat, envs_idx=env_ids)


def initialize_pos_quat_on_terrain(
    env: RLEnv,
    env_ids: torch.Tensor,
    base_init_quat: list[float],
    spawn_height_offset: float,
    entity_name: str = "robot",
    terrain_entity_name: str = "base_entity",
    spawn_margin: float = 2.0,
    randomize_yaw: bool = True,
):
    """Initialize robot position randomly on terrain.

    Args:
        env: Environment instance
        env_ids: Indices of environments to initialize
        base_init_quat: [w, x, y, z] default quaternion (used when randomize_yaw=False)
        spawn_height_offset: Height offset above terrain (meters)
        entity_name: Name of robot entity
        terrain_entity_name: Name of terrain entity
        spawn_margin: Margin from terrain edges (meters)
        randomize_yaw: Whether to randomize yaw orientation
    """
    if len(env_ids) == 0:
        return

    robot: RigidEntity = env.scene_manager[entity_name]
    terrain: RigidEntity = env.scene_manager[terrain_entity_name]

    terrain_geom = terrain.geoms[0]
    height_field = terrain_geom.metadata["height_field"]
    terrain_morph = terrain.morph

    if hasattr(terrain_morph, 'n_subterrains') and hasattr(terrain_morph, 'subterrain_size'):
        terrain_size_x = terrain_morph.n_subterrains[0] * terrain_morph.subterrain_size[0]
        terrain_size_y = terrain_morph.n_subterrains[1] * terrain_morph.subterrain_size[1]
    else:
        terrain_size_x = height_field.shape[0] * terrain_morph.horizontal_scale
        terrain_size_y = height_field.shape[1] * terrain_morph.horizontal_scale

    horizontal_scale = terrain_morph.horizontal_scale
    vertical_scale = terrain_morph.vertical_scale

    if not isinstance(height_field, torch.Tensor):
        height_field = torch.tensor(height_field, dtype=torch.float32, device=env.device)

    n_envs = len(env_ids)

    random_x = torch.rand(n_envs, device=env.device) * (terrain_size_x - 2 * spawn_margin) + spawn_margin
    random_y = torch.rand(n_envs, device=env.device) * (terrain_size_y - 2 * spawn_margin) + spawn_margin

    indices_x = (random_x / horizontal_scale).long().clamp(0, height_field.shape[0] - 1)
    indices_y = (random_y / horizontal_scale).long().clamp(0, height_field.shape[1] - 1)

    terrain_heights = height_field[indices_x, indices_y] * vertical_scale

    base_pos = torch.stack([
        random_x,
        random_y,
        terrain_heights + spawn_height_offset
    ], dim=-1)

    if randomize_yaw:
        random_yaw = torch.rand(n_envs, device=env.device) * 2 * torch.pi
        cy = torch.cos(random_yaw * 0.5)
        sy = torch.sin(random_yaw * 0.5)
        base_quat = torch.stack([cy, torch.zeros_like(cy), torch.zeros_like(cy), sy], dim=-1)
    else:
        base_quat = torch.tensor(base_init_quat, device=env.device).unsqueeze(0).expand(n_envs, -1)

    robot.set_pos(base_pos, envs_idx=env_ids)
    robot.set_quat(base_quat, envs_idx=env_ids)


def randomize_base_mass(
    env: RLEnv,
    env_ids: torch.Tensor,
    mass_ratio_range: tuple[float, float] = (1.3, 1.3),
    entity_name: str = "robot",
    base_name: str = "base"
):
    """
    Randomize the mass of the robot's base link using a ratio of the original mass.

    Args:
        env: The locomotion environment instance.
        env_ids: Tensor of environment indices to apply mass randomization.
                 Shape: (n_envs,)
        mass_ratio_range: Tuple of (min_ratio, max_ratio) for mass scaling.
                         Defaults to (1.0, 1.5).
                         - 1.0 = original mass (no change)
                         - 1.5 = 150% of original mass (50% increase)
                         Example: (1.0, 2.0) means 100%-200% of original mass
        entity_name: Name of the robot entity in the scene. Defaults to "robot".
        base_name: Name of the base link. Defaults to "base".

    Example:
        >>> # Test with 0% to 50% mass increase
        >>> env_ids = torch.arange(100, device='cuda')
        >>> randomize_base_mass(env, env_ids, mass_ratio_range=(1.0, 1.5))

        >>> # Test with 50% to 100% mass increase
        >>> randomize_base_mass(env, env_ids, mass_ratio_range=(1.5, 2.0))
    """
    entity = env.scene_manager[entity_name]
    base_link = entity.get_link(base_name)
    base_idx = base_link.idx_local

    # Get original mass of the base link
    original_mass = base_link.get_mass()

    # Sample random mass ratios uniformly from the specified range
    n_envs = len(env_ids)
    mass_ratios = (
        torch.rand(n_envs, device=env_ids.device)
        * (mass_ratio_range[1] - mass_ratio_range[0])
        + mass_ratio_range[0]
    )

    # Calculate mass shifts
    # actual_mass = original_mass + mass_shift
    # target_mass = original_mass * ratio
    # Therefore: mass_shift = original_mass * (ratio - 1)
    mass_shifts = original_mass * (mass_ratios - 1.0)

    # Reshape to (n_envs, 1) for set_mass_shift API
    mass_shifts = mass_shifts.unsqueeze(1)  # Shape: (n_envs, 1)

    entity.set_mass_shift(
        mass_shift=mass_shifts,
        links_idx_local=[base_idx],
        envs_idx=env_ids
    )


def randomize_link_mass(
    env: RLEnv,
    env_ids: torch.Tensor,
    mass_ratio_range: tuple[float, float] = (1.75, 1.75),
    entity_name: str = "robot",
):
    """
    Randomize the mass of all links in the robot.

    Args:
        env: The locomotion environment instance.
        env_ids: Tensor of environment indices to apply mass randomization.
        mass_ratio_range: Tuple of (min_ratio, max_ratio) for mass scaling.
        entity_name: Name of the robot entity in the scene.
    """
    entity = env.scene_manager[entity_name]
    n_envs = len(env_ids)
    n_links = len(entity.links)

    # Collect link indices and original masses
    links_idx_local = []
    original_masses = []
    for link in entity.links:
        links_idx_local.append(link.idx_local)
        original_masses.append(link.get_mass())

    original_masses = torch.tensor(original_masses, device=env_ids.device)  # (n_links,)

    # Sample random mass ratios: (n_envs, n_links)
    mass_ratios = (
        torch.rand(n_envs, n_links, device=env_ids.device)
        * (mass_ratio_range[1] - mass_ratio_range[0])
        + mass_ratio_range[0]
    )

    # mass_shift = original_mass * (ratio - 1)
    mass_shifts = original_masses.unsqueeze(0) * (mass_ratios - 1.0)  # (n_envs, n_links)

    entity.set_mass_shift(
        mass_shift=mass_shifts,
        links_idx_local=links_idx_local,
        envs_idx=env_ids
    )


def randomize_p_gain(
    env: RLEnv,
    env_ids: torch.Tensor,
    p_gain_range: dict[str, tuple[float, float]],
    entity_name: str = "robot",
):
    """Randomize P gains for specified environments.

    Args:
        env: The locomotion environment
        env_ids: Environment IDs to randomize
        p_gain_range: Dict mapping joint patterns to (min, max) gain ranges
                      e.g., {".*": (16.0, 24.0)} or {".*_hip": (10, 20), ".*_knee": (15, 25)}
        entity_name: Name of the entity to randomize
    """
    from rlworld.rl.utils import entity_utils as eu
    from rlworld.rl.utils import string as string_utils

    entity = env.scene_manager[entity_name]

    # Find matching joints
    dof_ids, joint_names = eu.find_dofs(
        entity=entity,
        name_keys=list(p_gain_range.keys())
    )

    # Get ranges for each joint
    _, _, gain_ranges = string_utils.resolve_matching_names_values(
        p_gain_range,
        joint_names
    )

    num_joints = len(dof_ids)

    # Set different random gains for each environment
    for env_idx in env_ids:
        # Sample random gains for this environment: shape (num_joints,)
        random_gains = torch.zeros(num_joints, device=env.device)

        for j, (min_gain, max_gain) in enumerate(gain_ranges):
            random_gains[j] = torch.rand(1, device=env.device).item() * (max_gain - min_gain) + min_gain

        # Set gains for this specific environment
        entity.set_dofs_kp(kp=random_gains, dofs_idx_local=dof_ids, envs_idx=[env_idx])


def randomize_d_gain(
    env: RLEnv,
    env_ids: torch.Tensor,
    d_gain_range: dict[str, tuple[float, float]],
    entity_name: str = "robot",
):
    """Randomize D gains for specified environments.

    Args:
        env: The locomotion environment
        env_ids: Environment IDs to randomize
        d_gain_range: Dict mapping joint patterns to (min, max) gain ranges
        entity_name: Name of the entity to randomize
    """
    from rlworld.rl.utils import entity_utils as eu
    from rlworld.rl.utils import string as string_utils

    entity = env.scene_manager[entity_name]

    # Find matching joints
    dof_ids, joint_names = eu.find_dofs(
        entity=entity,
        name_keys=list(d_gain_range.keys())
    )

    # Get ranges for each joint
    _, _, gain_ranges = string_utils.resolve_matching_names_values(
        d_gain_range,
        joint_names
    )

    num_joints = len(dof_ids)

    # Set different random gains for each environment
    for env_idx in env_ids:
        # Sample random gains for this environment: shape (num_joints,)
        random_gains = torch.zeros(num_joints, device=env.device)

        for j, (min_gain, max_gain) in enumerate(gain_ranges):
            random_gains[j] = torch.rand(1, device=env.device).item() * (max_gain - min_gain) + min_gain

        # Set gains for this specific environment
        entity.set_dofs_kv(kv=random_gains, dofs_idx_local=dof_ids, envs_idx=[env_idx])


def initialize_dof_pos_with_noise(
    env: RLEnv,
    env_ids: torch.Tensor,
    position_noise_range: tuple[float, float] = (0.0, 0.0),
    velocity_noise_range: tuple[float, float] = (0.0, 0.0),
    entity_name: str = "robot",
):
    """Initialize DOF positions with optional noise.

    Args:
        env: The environment instance.
        env_ids: Environment indices to initialize.
        position_noise_range: (min, max) uniform noise added to default positions.
        velocity_noise_range: (min, max) uniform noise for initial velocities.
        entity_name: Name of the robot entity.
    """
    if len(env_ids) == 0:
        return

    robot: RigidEntity = env.scene_manager[entity_name]
    dof_pos = env.act_manager.offset[env_ids].clone()  # (n_envs, n_dofs)

    # Add position noise
    if position_noise_range != (0.0, 0.0):
        noise = torch.empty_like(dof_pos).uniform_(
            position_noise_range[0],
            position_noise_range[1]
        )
        dof_pos = dof_pos + noise

    robot.set_dofs_position(
        position=dof_pos,
        dofs_idx_local=env.act_manager.actuated_dof_ids,
        zero_velocity=velocity_noise_range == (0.0, 0.0),
        envs_idx=env_ids
    )

    # Set velocity noise if specified
    if velocity_noise_range != (0.0, 0.0):
        dof_vel = torch.empty(
            len(env_ids),
            len(env.act_manager.actuated_dof_ids),
            device=env.device
        ).uniform_(velocity_noise_range[0], velocity_noise_range[1])

        robot.set_dofs_velocity(
            velocity=dof_vel,
            dofs_idx_local=env.act_manager.actuated_dof_ids,
            envs_idx=env_ids
        )


def randomize_friction(
    env: RLEnv,
    env_ids: torch.Tensor,
    friction_ratio_range: tuple[float, float] = (0.6, 1.4),
    entity_name: str = "robot",
):
    """
    Randomize friction ratio for robot links.

    Args:
        env: The environment instance.
        env_ids: Environment indices to randomize.
        friction_ratio_range: (min, max) ratio multiplier for friction.
        entity_name: Name of the robot entity.
    """
    if len(env_ids) == 0:
        return

    entity = env.scene_manager[entity_name]
    n_envs = len(env_ids)
    n_links = entity.n_links

    # Sample random friction ratios: (n_envs, n_links)
    friction_ratios = (
        torch.rand(n_envs, n_links, device=env.device)
        * (friction_ratio_range[1] - friction_ratio_range[0])
        + friction_ratio_range[0]
    )

    entity.set_friction_ratio(
        friction_ratio=friction_ratios,
        links_idx_local=list(range(n_links)),
        envs_idx=env_ids,
    )