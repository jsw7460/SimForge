"""Lifting command: where the object is, and where it should go.

Ported from mjlab's ``tasks/manipulation/mdp/commands.py``. One term owns
both halves of the task's episode setup, because they are one decision:
resampling the goal without moving the object would leave the policy
solving the same layout with a new label.

The object is placed by writing its root pose, which is how every other
reset in this repo places an entity, so the same term works on all three
backends without knowing which one it is running on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from rlworld.rl.configs.scene.entity_selector import ResolvedEntity, SceneEntitySelector
from rlworld.rl.envs.managers.common.command_term import CommandTerm, CommandTermCfg
from rlworld.rl.envs.mdp.entity_points import entity_point_w
from rlworld.rl.utils.quat_utils import quat_from_euler_xyz_wxyz, quat_mul_wxyz

if TYPE_CHECKING:
    from rlworld.rl.envs.world import World


def _sample(lower: torch.Tensor, upper: torch.Tensor, shape: tuple[int, ...], device) -> torch.Tensor:
    return torch.rand(*shape, device=device) * (upper - lower) + lower


@dataclass
class LiftingCommandCfg(CommandTermCfg):
    """Configuration for :class:`LiftingCommand`.

    Attributes:
        entity_name: Scene name of the object to lift.
        success_threshold: Distance from the goal at which the episode
            counts as solved [m].
        difficulty: ``"fixed"`` puts the goal at :attr:`fixed_target`
            every episode; ``"dynamic"`` samples it from
            :attr:`target_position_range`. Fixed first is worth having —
            a policy that cannot solve one goal will not solve a
            distribution of them, and the failure is easier to read.
        fixed_target: The goal used in ``"fixed"`` mode, in the
            environment's own frame.
        target_position_range: Goal sampling box, environment frame.
        object_pose_range: Where the object is put at the start of each
            episode. ``None`` leaves it wherever it was, which is only
            sensible if some other event term places it.
    """

    entity_name: str = "cube"
    entity_site: str | None = None
    """Site on the object that IS the object, for the purpose of arriving.

    None means its frame origin, which is right for a cube and wrong for
    anything shaped. The reward terms already aim at a site when one is
    named -- and if this command keeps measuring the origin while they
    aim at the site, the two disagree by however far apart the points
    are. On the spring tong that is 52.6 mm against a success threshold
    of 50, so bringing the grip point exactly onto the goal left the
    origin outside it and the episode could never be solved.
    """
    success_threshold: float = 0.05
    difficulty: str = "dynamic"

    fixed_target: tuple[float, float, float] = (0.40, 0.0, 0.70)

    target_x: tuple[float, float] = (0.30, 0.50)
    target_y: tuple[float, float] = (-0.20, 0.20)
    target_z: tuple[float, float] = (0.60, 0.80)

    object_x: tuple[float, float] = (0.30, 0.35)
    object_y: tuple[float, float] = (-0.10, 0.10)
    object_z: tuple[float, float] = (0.42, 0.45)
    object_yaw: tuple[float, float] = (-3.14159265, 3.14159265)
    object_base_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    """The orientation the object rests in, before the yaw above is
    applied on top of it (w, x, y, z).

    Identity suits a cube, which looks the same whichever face is down.
    An object with only one resting pose does not: a spring tong lies on
    its side, and placing it with a yaw alone would stand it back up
    every episode, in a pose it does not stay in and cannot be picked up
    from. Stated here rather than left to a separate event term, so the
    one place that decides where an episode's object goes also decides
    which way up it goes."""

    place_object: bool = True
    place_object_on_resample: bool = True
    """Whether the timer's resample also moves the object, or only the
    episode reset does.

    True is the original behaviour: a resample poses the task afresh,
    which doubles the attempts an episode is worth. That works when the
    policy is TOLD where the object is — the observation changes and it
    goes after it. A policy that has to see the object cannot detect the
    move at all when its camera is pointed elsewhere, and one that was
    holding the object cannot tell that its gripper is now empty, so the
    teleport reads as "nothing happened" and the rest of the episode is
    spent holding nothing. Set False there.
    """

    resampling_time_range: tuple[float, float] = (8.0, 12.0)
    """mjlab's own range: the goal and the object are re-drawn twice or
    so within an episode. That is a deliberate part of the task rather
    than an accident — it gives several attempts per episode, so a policy
    that fumbles the first one still sees what a fresh layout looks
    like."""

    def build(self, env: World) -> LiftingCommand:
        return LiftingCommand(env, self)


class LiftingCommand(CommandTerm):
    """Goal position for an object, plus that object's episode start pose."""

    column_names = ("target_x", "target_y", "target_z")

    cfg: LiftingCommandCfg

    def __init__(self, env: World, cfg: LiftingCommandCfg) -> None:
        super().__init__(env, cfg)
        self.cfg = cfg
        self.target_pos = torch.zeros(self.num_envs, 3, device=self.device)
        # Latched, not instantaneous: an episode counts as solved if the
        # object was ever brought to the goal, so a policy that reaches it
        # and then drifts is still credited with having got there.
        self.episode_success = torch.zeros(self.num_envs, device=self.device)
        # Set only while an episode reset is resampling, so the object
        # placement can be told apart from the timer's resample without
        # changing _resample_command's signature for every command term.
        self._placing_on_reset = False
        # Resolved once. Deferred rather than done here because a
        # selector naming a site needs the scene's index tables, and a
        # command is built while those are still being assembled.
        self._object: ResolvedEntity | None = None

    @property
    def command(self) -> torch.Tensor:
        return self.target_pos

    # ── Task state ───────────────────────────────────────────────────

    @property
    def object_point(self) -> ResolvedEntity:
        """The object, and the point on it this command measures."""
        if self._object is None:
            sites = (self.cfg.entity_site,) if self.cfg.entity_site else None
            self._object = self._env.resolve_selector(SceneEntitySelector(name=self.cfg.entity_name, site_names=sites))
        return self._object

    @property
    def object_pos_w(self) -> torch.Tensor:
        return entity_point_w(self._env, self.object_point)

    @property
    def position_error(self) -> torch.Tensor:
        """Distance from the object to its goal, per env [m]."""
        return torch.norm(self.target_pos - self.object_pos_w, dim=-1)

    @property
    def at_goal(self) -> torch.Tensor:
        return self.position_error < self.cfg.success_threshold

    def _update_command(self) -> None:
        self.episode_success = torch.maximum(self.episode_success, self.at_goal.float())

    # ── Resampling ───────────────────────────────────────────────────

    def reset(self, env_ids: torch.Tensor) -> None:
        self._placing_on_reset = True
        try:
            super().reset(env_ids)
        finally:
            self._placing_on_reset = False

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        if n == 0:
            return
        origins = self._env.scene_manager.env_origins[env_ids]
        self.episode_success[env_ids] = 0.0

        if self.cfg.difficulty == "fixed":
            target = torch.tensor(self.cfg.fixed_target, device=self.device).expand(n, 3)
        elif self.cfg.difficulty == "dynamic":
            lower = torch.tensor([self.cfg.target_x[0], self.cfg.target_y[0], self.cfg.target_z[0]], device=self.device)
            upper = torch.tensor([self.cfg.target_x[1], self.cfg.target_y[1], self.cfg.target_z[1]], device=self.device)
            target = _sample(lower, upper, (n, 3), self.device)
        else:
            raise ValueError(f"difficulty must be 'fixed' or 'dynamic', got {self.cfg.difficulty!r}")
        self.target_pos[env_ids] = target + origins

        if not self.cfg.place_object:
            return
        if not self._placing_on_reset and not self.cfg.place_object_on_resample:
            return

        lower = torch.tensor([self.cfg.object_x[0], self.cfg.object_y[0], self.cfg.object_z[0]], device=self.device)
        upper = torch.tensor([self.cfg.object_x[1], self.cfg.object_y[1], self.cfg.object_z[1]], device=self.device)
        pos = _sample(lower, upper, (n, 3), self.device) + origins
        # Yaw only. A cube tipped onto an edge is a different grasp
        # problem, and not the one this task is posing.
        yaw = _sample(
            torch.tensor(self.cfg.object_yaw[0], device=self.device),
            torch.tensor(self.cfg.object_yaw[1], device=self.device),
            (n,),
            self.device,
        )
        zero = torch.zeros(n, device=self.device)
        quat = quat_from_euler_xyz_wxyz(zero, zero, yaw)
        base = torch.tensor(self.cfg.object_base_quat, device=self.device, dtype=quat.dtype)
        # Yaw about the world's vertical, applied to the resting pose —
        # not the other way round, which would turn the object about an
        # axis that moved with it and tip a tong onto its side twice.
        quat = quat_mul_wxyz(quat, base.expand_as(quat))

        # Polymorphic: the same call places a passive prop or an
        # articulation, so the object need not be one or the other.
        writer = self._env.get_root_state_writer(self.cfg.entity_name)
        writer.set_root_pose(pos, quat, env_ids=env_ids)
        writer.set_root_velocity(
            torch.zeros(n, 3, device=self.device),
            torch.zeros(n, 3, device=self.device),
            env_ids=env_ids,
        )
