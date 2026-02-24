"""
Reusable configuration components for rlworld.

Components are building blocks that can be composed to create complete environment configs.
Each component provides a `to_terms()` or `to_dict()` method for config generation.

Usage (import only what you need):
    # Genesis
    from rlworld.rl.configs.components.observations.genesis import LocomotionObservations
    from rlworld.rl.configs.components.rewards.genesis import TrackingRewards

    # Newton
    from rlworld.rl.configs.components.observations.newton import LocomotionObservations
    from rlworld.rl.configs.components.rewards.newton import TrackingRewards
"""

# No automatic imports - use explicit imports to avoid loading both backends
__all__ = ["observations", "rewards"]
