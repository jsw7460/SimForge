"""RLWorld core package.

Intentionally side-effect free: importing ``rlworld.rl`` (or any of its
runner / config submodules) must not import a simulator package.  In
particular the Genesis runtime (``genesis.init``) is *not* called here;
the Genesis-backed environments call it lazily from their ``__init__``
(``GenesisEnv`` / ``GymnasiumEnv`` / ``ManiSkillEnv``), so a Newton- or
MuJoCo-only process never imports or initialises Genesis.
"""
