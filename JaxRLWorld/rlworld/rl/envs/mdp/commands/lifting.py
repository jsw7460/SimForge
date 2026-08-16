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

from rlworld.rl.envs.managers.common.command_term import CommandTerm, CommandTermCfg
from rlworld.rl.utils.quat_utils import quat_from_euler_xyz_wxyz

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
    place_object: bool = True

    resampling_time_range: tuple[float, float] = (1e9, 1e9)
    """Never resample mid-episode by default. Moving the goal AND the
    object under a policy that is halfway through a reach is a different
    task from the one it is being scored on; mjlab's own lift config
    likewise resamples only on reset."""

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

    @property
    def command(self) -> torch.Tensor:
        return self.target_pos

    # ── Task state ───────────────────────────────────────────────────

    @property
    def object_pos_w(self) -> torch.Tensor:
        return self._env.get_entity_data(self.cfg.entity_name).root_link_pos_w

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

        # Polymorphic: the same call places a passive prop or an
        # articulation, so the object need not be one or the other.
        writer = self._env.get_root_state_writer(self.cfg.entity_name)
        writer.set_root_pose(pos, quat, env_ids=env_ids)
        writer.set_root_velocity(
            torch.zeros(n, 3, device=self.device),
            torch.zeros(n, 3, device=self.device),
            env_ids=env_ids,
        )
