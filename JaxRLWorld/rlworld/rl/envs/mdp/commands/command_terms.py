import torch
from rlworld.rl.utils import gs_rand_float


def lin_vel_x(
    env,
    env_ids: torch.Tensor,
    range: tuple[float, float] = (-1.0, 1.5)
) -> torch.Tensor:
    """
    Generate linear velocity X commands.

    Args:
        env: Environment instance
        env_ids: Environment indices to generate commands for
        range: (min, max) range for forward/backward velocity

    Returns:
        Tensor [len(env_ids)]
    """
    return gs_rand_float(range[0], range[1], (len(env_ids),), env.device)


def lin_vel_y(
    env,
    env_ids: torch.Tensor,
    range: tuple[float, float] = (-0.5, 0.5)
) -> torch.Tensor:
    """
    Generate linear velocity Y commands.

    Args:
        env: Environment instance
        env_ids: Environment indices
        range: (min, max) range for lateral velocity

    Returns:
        Tensor [len(env_ids)]
    """
    return gs_rand_float(range[0], range[1], (len(env_ids),), env.device)


def ang_vel(
    env,
    env_ids: torch.Tensor,
    range: tuple[float, float] = (-1.0, 1.0)
) -> torch.Tensor:
    """
    Generate angular velocity (yaw rate) commands.

    Args:
        env: Environment instance
        env_ids: Environment indices
        range: (min, max) range for yaw rate (rad/s)

    Returns:
        Tensor [len(env_ids)]
    """
    return gs_rand_float(range[0], range[1], (len(env_ids),), env.device)


def base_height(
    env,
    env_ids: torch.Tensor,
    range: tuple[float, float] = (0.6, 0.8)
) -> torch.Tensor:
    """
    Generate base height commands.

    Args:
        env: Environment instance
        env_ids: Environment indices
        range: (min, max) range for base height (m)

    Returns:
        Tensor [len(env_ids)]
    """
    return gs_rand_float(range[0], range[1], (len(env_ids),), env.device)


def foot_height(
    env,
    env_ids: torch.Tensor,
    range: tuple[float, float] = (0.1, 0.3)
) -> torch.Tensor:
    """
    Generate foot height commands for swing phase.

    Args:
        env: Environment instance
        env_ids: Environment indices
        range: (min, max) range for foot clearance height (m)

    Returns:
        Tensor [len(env_ids)]
    """
    return gs_rand_float(range[0], range[1], (len(env_ids),), env.device)


def gait_frequency(
    env,
    env_ids: torch.Tensor,
    range: tuple[float, float] = (0.8, 1.5)
) -> torch.Tensor:
    """
    Generate gait frequency commands.

    Args:
        env: Environment instance
        env_ids: Environment indices
        range: (min, max) range for gait frequency (Hz)

    Returns:
        Tensor [len(env_ids)]
    """
    return gs_rand_float(range[0], range[1], (len(env_ids),), env.device)


def constant_height(
    env,
    env_ids: torch.Tensor,
    value: float = 0.75
) -> torch.Tensor:
    """
    Generate constant height command.

    Args:
        env: Environment instance
        env_ids: Environment indices
        value: Constant height value (m)

    Returns:
        Tensor [len(env_ids)]
    """
    return torch.full((len(env_ids),), value, device=env.device)


def zero_command(
    env,
    env_ids: torch.Tensor
) -> torch.Tensor:
    """
    Generate zero commands (standing still).

    Args:
        env: Environment instance
        env_ids: Environment indices

    Returns:
        Tensor [len(env_ids)] of zeros
    """
    return torch.zeros(len(env_ids), device=env.device)