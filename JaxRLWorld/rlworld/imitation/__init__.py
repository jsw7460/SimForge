"""Imitation learning methods.

Sibling namespace to ``rlworld.rl``. Imitation algorithms here may
*read* environments and checkpoint utilities from ``rlworld.rl`` but
should not be imported from inside ``rlworld.rl`` — the dependency is
strictly one-directional so that pure-RL setups stay independent of
the imitation stack.
"""
