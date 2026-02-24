"""
Reward components for rlworld configs.

Usage (import only what you need):
    from rlworld.rl.configs.components.rewards.genesis import TrackingRewards
    # or
    from rlworld.rl.configs.components.rewards.newton import TrackingRewards
    # or
    from rlworld.rl.configs.components.rewards.mujoco import TrackingRewards
"""

# No automatic imports - use explicit imports to avoid loading both backends
__all__ = ["genesis", "newton", "mujoco"]
