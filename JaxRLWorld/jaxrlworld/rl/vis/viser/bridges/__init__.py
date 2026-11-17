"""Simulator bridges for the viser viewer, one module per backend.

Import the bridge from its module (``bridges.newton_bridge``,
``bridges.genesis_bridge``, ``bridges.mujoco_bridge``): the Newton bridge
loads ``newton`` at import, so a package-level re-export would load it on
every backend.
"""
