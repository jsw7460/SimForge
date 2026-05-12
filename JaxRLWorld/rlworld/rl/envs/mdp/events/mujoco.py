"""MuJoCo-specific event helpers.

Cross-sim domain-randomization terms live in
:mod:`rlworld.rl.envs.mdp.events.dr.unified`; general-purpose
reset / push functions live in ``common.py``.  What remains here is
:class:`_MujocoEnvAdapter` — a thin shim that exposes the minimal
interface mjlab's own DR functions expect (``num_envs`` / ``device`` /
``scene`` / ``sim``).  The unified DR backends construct one of these
when delegating to ``mjlab.envs.mdp.dr.*``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rlworld.rl.envs.mujoco import MujocoEnv


class _MujocoEnvAdapter:
    """Adapter that exposes the interface mjlab DR functions expect.

    mjlab's ``dr.*`` functions read ``env.num_envs`` / ``env.device`` /
    ``env.scene`` / ``env.sim`` (the ``ManagerBasedRlEnv`` surface); our
    :class:`MujocoEnv` keeps the equivalents on ``scene_manager``.  This
    wrapper bridges the gap so the unified DR backends can delegate to
    mjlab without exposing mjlab's env type to JaxRLWorld code.
    """

    def __init__(self, rlworld_env: MujocoEnv):
        self._env = rlworld_env

    @property
    def num_envs(self):
        return self._env.num_envs

    @property
    def device(self):
        return self._env.device

    @property
    def scene(self):
        return self._env.scene_manager.scene

    @property
    def sim(self):
        return self._env.scene_manager.sim
