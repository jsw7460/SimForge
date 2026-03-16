from collections.abc import Iterable
from itertools import chain
from typing import Sequence

from genesis.engine.entities import RigidEntity
from rlworld.rl.utils import string as string_utils


def find_joints(
    entity: RigidEntity,
    name_keys: str | Sequence[str],
    joint_subset: list[str] | None = None,
    preserve_order: bool = True
) -> tuple[list[int], list[str]]:
    """Find joints in the articulation based on the name keys."""
    joint_names = [joint.name for joint in chain.from_iterable(entity._joints)]

    if joint_subset is None:
        joint_subset = joint_names

    _, _joint_names = string_utils.resolve_matching_names(name_keys, joint_subset, preserve_order)

    _joint_ids_local = []
    for name in _joint_names:
        _joint_ids_local.append(entity.get_joint(name).idx_local)

    return _joint_ids_local, _joint_names


def find_dofs(
    entity: RigidEntity,
    name_keys: str | Sequence[str],
    joint_subset: list[str] | None = None,
    preserve_order: bool = True,
) -> tuple[list[int], list[str]]:
    """Find DOFs based on joint name keys."""
    _joint_ids_local, _joint_names = find_joints(entity, name_keys, joint_subset, preserve_order)

    _dof_ids_local = []
    for name in _joint_names:
        ids = entity.get_joint(name).dofs_idx_local
        if isinstance(ids, Iterable):
            _dof_ids_local.extend(ids)
        else:
            _dof_ids_local.append(ids)

    return _dof_ids_local, _joint_names


def find_links(
    entity: RigidEntity,
    name_keys: str | Sequence[str],
    link_subset: list[str] | None = None,
    global_ids: bool = False,
    preserve_order: bool = False,
) -> tuple[list[int], list[str]]:
    """Find links based on name keys."""
    links_names = [link.name for link in entity._links]

    if link_subset is None:
        link_subset = links_names

    _, _link_names = string_utils.resolve_matching_names(name_keys, link_subset, preserve_order)

    _link_ids_local = []
    for name in _link_names:
        if global_ids:
            idx = entity.get_link(name).idx
        else:
            idx = entity.get_link(name).idx_local
        _link_ids_local.append(idx)

    return _link_ids_local, _link_names
