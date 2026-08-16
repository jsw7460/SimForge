from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import warp as wp

from rlworld.rl.envs.managers.common.action import (
    ActionManagerBase,
    ActionManagerBaseConfig,
)

if TYPE_CHECKING:
    from rlworld.rl.envs import World


def _reject_out_of_range(indices, width: int, entity_name: str, indexing, array_name: str) -> None:
    """Raise before writing an index past one world's share of an array.

    The destination's own per-world width is the only bound that counts
    here: Newton sizes ``joint_target_q`` to whichever layout is active,
    which need not match the model's coordinate count. An index past it
    is written anyway on the GPU and surfaces later as a device-side
    assert whose stack points at whatever kernel ran next — several
    manager calls away, with nothing naming the entity that caused it.
    """
    values = indices.tolist()
    bad = [i for i, v in enumerate(values) if not 0 <= v < width]
    if not bad:
        return
    names = list(indexing.joint_names)
    detail = ", ".join(f"{names[i]}->{values[i]}" for i in bad)
    raise IndexError(
        f"Entity {entity_name!r} writes {array_name} at {detail}, outside this world's {width} slots. "
        f"Its {len(names)} joints resolved to indices {values}."
    )


@dataclass
class NewtonActionManagerConfig(ActionManagerBaseConfig):
    """Newton-specific action manager configuration."""

    pass


class NewtonActionManager(ActionManagerBase):
    """Newton action manager.

    Uses ArticulationIndexing for all index mappings. Position targets are
    written into ``joint_target_q``; under the legacy DOF layout this matches
    the ``joint_qd`` stride so ``newton_qd_indices`` is the right indexer,
    but under the coord layout (Newton PR #2965) the same array follows the
    ``joint_q`` stride and ``newton_q_indices`` must be used instead. Forces
    are always ``joint_qd``-strided, so ``_apply_force`` keeps
    ``newton_qd_indices`` unconditionally.
    """

    def __init__(self, env: World, config: NewtonActionManagerConfig):
        super().__init__(env, config)

    def _apply_position(self, targets: torch.Tensor, entity_name: str) -> None:
        """Apply position targets via Newton/Warp.

        Written **in place**, into a view of ``control.joint_target_q``
        itself, so the per-world stride comes from the array rather than
        from a count that can disagree with it. Newton sizes that array
        more generously than ``joint_dof_count`` for a model that keeps its
        FIXED joints (``collapse_fixed_joints=False``): a bench-mounted arm
        with a linkage gripper measured 13 slots per world against a dof
        count of 8. Building a separate ``(worlds, dof_per_world)`` buffer
        and copying it over therefore packed four worlds' targets into the
        space of two and a half — world 0 landed correctly, every later
        world read someone else's joint angles, and nothing raised, because
        a short ``wp.copy`` into a longer array is legal.

        Writing in place also leaves the FREE joint's quaternion slot as
        the simulator set it, rather than reconstructing it from zeros
        (``newton.Control.clear`` documents that zeroing this array
        corrupts that slot).
        """
        control = self.env.scene_manager.control
        num_worlds = self.env.scene_manager.model.world_count

        # Under the coord layout introduced in Newton PR #2965,
        # ``joint_target_q`` follows the ``joint_q`` stride (7 slots for a
        # FREE joint) and is indexed by ``newton_q_indices``; under the
        # legacy DOF layout it follows ``joint_qd`` (6 slots for a FREE
        # joint) and is indexed by ``newton_qd_indices``.
        indexing = self.env.entity_indexing(entity_name)
        # The MODEL's snapshot, not the module-level global. Newton takes
        # the flag once at ``finalize()`` and states that every layout
        # decision for that model consults the snapshot, so reading the
        # global would pick the wrong index array for any model finalized
        # under a different setting — and the two differ by exactly one
        # slot per FREE joint, which lands a write just past the end.
        model = self.env.scene_manager.model
        write_indices = indexing.newton_q_indices if model.use_coord_layout_targets else indexing.newton_qd_indices

        dest = wp.to_torch(control.joint_target_q)
        if dest.numel() % num_worlds != 0:
            raise RuntimeError(
                f"Newton joint_target_q has {dest.numel()} slots, which is not divisible by "
                f"world_count={num_worlds}; the per-world layout cannot be derived."
            )
        rows = dest.view(num_worlds, -1)
        _reject_out_of_range(write_indices, rows.shape[1], entity_name, indexing, "joint_target_q")
        rows[:, write_indices] = targets

    def _apply_force(self, torques: torch.Tensor, entity_name: str) -> None:
        """Apply torques directly via Newton/Warp.

        In place, for the same reason as :meth:`_apply_position`: the
        destination's own length is the only trustworthy source of the
        per-world stride, and every non-actuated slot keeps whatever the
        simulator had there instead of being zeroed by a wholesale copy.
        """
        control = self.env.scene_manager.control
        num_worlds = self.env.scene_manager.model.world_count

        dest = wp.to_torch(control.joint_f)
        if dest.numel() % num_worlds != 0:
            raise RuntimeError(
                f"Newton joint_f has {dest.numel()} slots, which is not divisible by "
                f"world_count={num_worlds}; the per-world layout cannot be derived."
            )
        rows = dest.view(num_worlds, -1)
        # Write only this entity's slots. Zeroing the whole buffer first
        # would erase the torque another entity's term wrote earlier in
        # the same step. Nothing goes stale: the caller hands over a
        # full-width buffer covering every one of this entity's actuated
        # joints, zeros included, so each of them is rewritten every step.
        write_indices = self.env.entity_indexing(entity_name).newton_qd_indices
        _reject_out_of_range(
            write_indices,
            rows.shape[1],
            entity_name,
            self.env.entity_indexing(entity_name),
            "joint_f",
        )
        rows[:, write_indices] = torques

    # -- Backward compat properties ------------------------------------------

    @property
    def actuated_q_indices(self) -> torch.Tensor:
        return self._indexing.newton_q_indices

    @property
    def actuated_qd_indices(self) -> torch.Tensor:
        return self._indexing.newton_qd_indices
