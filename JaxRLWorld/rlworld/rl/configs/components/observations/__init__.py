"""
Observation components for rlworld configs.

Usage (import only what you need):
    from rlworld.rl.configs.components.observations.genesis import LocomotionObservations
    # or
    from rlworld.rl.configs.components.observations.newton import LocomotionObservations
    # or
    from rlworld.rl.configs.components.observations.mujoco import LocomotionObservations
"""

# No automatic imports - use explicit imports to avoid loading both backends
__all__ = ["genesis", "newton", "mujoco"]
