"""Marker for observation functions whose columns follow the simulator's joint order.

Cross-simulator evaluation (:mod:`jaxrlworld.rl.envs.joint_permutation`) has
to reorder every observation column that is indexed by actuated joint, and
only those, because Genesis orders a robot's joints by the URDF while
Newton and mjlab follow the pattern order. Which terms are joint-indexed is
a property of the observation function, so the function declares it here;
a name list kept elsewhere silently went stale as terms were added.

Mark a function with ``@joint_indexed`` when its output is
``(num_envs, k * num_actuated_joints)`` with the joints in the simulator's
order (``k`` > 1 for stacked history). Mark it ``@not_joint_indexed`` when
its width happens to be a multiple of the joint count but the columns are
not joints (the permutation code refuses to guess and raises on an unmarked
term of that width).
"""

from __future__ import annotations

from typing import Callable, TypeVar

F = TypeVar("F", bound=Callable)

_ATTR = "joint_indexed"


def joint_indexed(fn: F) -> F:
    """Declare ``fn``'s output columns to be the actuated joints in simulator order."""
    setattr(fn, _ATTR, True)
    return fn


def not_joint_indexed(fn: F) -> F:
    """Declare ``fn``'s output columns to be unrelated to joint order."""
    setattr(fn, _ATTR, False)
    return fn


def joint_indexed_flag(fn: Callable) -> bool | None:
    """``True`` / ``False`` when ``fn`` is marked, ``None`` when it is not.

    ``fn`` is an arbitrary callable (plain function, ``EnvStepCache`` wrapper,
    functools.partial, class instance), so the marker is looked up by name.
    """
    return getattr(fn, _ATTR, None)
