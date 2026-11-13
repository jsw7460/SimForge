"""Raise on a mujoco-warp capacity overflow instead of simulating past it."""

from __future__ import annotations

import functools
import operator
from collections.abc import Callable
from enum import IntFlag

import mujoco_warp
import numpy as np
import torch
import warp as wp

# The bits of ``OverflowType`` that say a buffer was too small and work was
# dropped. The two iteration bits say a solve hit its iteration budget, which
# the presets choose deliberately (20/50) and hit on most steps.
_CAPACITY_BITS = int(mujoco_warp.OverflowType.ALL) & ~int(
    mujoco_warp.OverflowType.ITERATIONS | mujoco_warp.OverflowType.LS_ITERATIONS
)
# The option that sizes the buffer each bit stands for.
_OPTION_OF_FLAG = {
    "NEFC": "njmax",
    "NJMAX_NNZ": "njmax_nnz",
    "BROADPHASE": "nconmax",
    "NARROWPHASE": "nconmax",
    "SOLVER_NCONMAX": "nconmax",
    "CONTACT_PIPELINE": "rigid_contact_max",
}


class MjwarpOverflowGuard:
    """Raise once mujoco-warp reports that a world overflowed a buffer.

    mujoco-warp stores only as many contacts and constraint rows as its
    buffers hold; the rest are dropped without an error, the bodies they
    belonged to interpenetrate, and the state blows up a few steps later
    with nothing naming the cause. What it does publish is a per-world
    ``OverflowType`` bitmask (``Data.overflow``; Newton collects it into
    its overflow observable with its own bits on top). The mask is read
    back without waiting for the device: each check enqueues a copy into
    page-locked host memory, on warp's own stream so it follows the step's
    kernels, and inspects the copy the previous check enqueued. An overflow
    therefore raises one control step after it happened. On a CPU device
    the mask is read directly.

    ``overflow`` is a callable returning the live bitmask array: a backend
    may replace its data after this guard is built (mjlab's model
    expansion), so the array is not held. ``extra_flags`` names the bits a
    backend adds above mujoco-warp's own (Newton's ``OverflowBits``).
    """

    def __init__(self, overflow: Callable[[], wp.array], extra_flags: type[IntFlag] | None = None) -> None:
        self._overflow = overflow
        self._flags: list[tuple[int, str]] = [
            (int(flag), flag.name)
            for flag in mujoco_warp.OverflowType
            if flag.name not in ("NONE", "ALL") and int(flag) & _CAPACITY_BITS
        ]
        self._raise_bits = _CAPACITY_BITS
        if extra_flags is not None:
            self._flags.extend((int(flag), flag.name) for flag in extra_flags)
            self._raise_bits |= int(functools.reduce(operator.or_, extra_flags))
        live = overflow()
        self._on_cuda = not live.device.is_cpu
        if self._on_cuda:
            self._stream = torch.cuda.ExternalStream(wp.get_stream(live.device).cuda_stream)
            self._host = torch.zeros(live.shape[0], dtype=torch.int32, pin_memory=True)
            self._event = torch.cuda.Event()

    def check(self) -> None:
        """Raise if the previous check's mask shows an overflow; enqueue this one's."""
        overflow = wp.to_torch(self._overflow())
        if not self._on_cuda:
            self._raise_if_set(overflow.numpy())
            return
        self._event.synchronize()
        self._raise_if_set(self._host.numpy())
        with torch.cuda.stream(self._stream):
            self._host.copy_(overflow, non_blocking=True)
            self._event.record(self._stream)

    def _raise_if_set(self, per_world: np.ndarray) -> None:
        union = int(np.bitwise_or.reduce(per_world)) & self._raise_bits
        if not union:
            return
        worlds = int(np.count_nonzero(per_world & union))
        names = []
        for bit, name in self._flags:
            if union & bit:
                option = _OPTION_OF_FLAG.get(name)
                names.append(name if option is None else f"{name} (increase {option})")
        raise RuntimeError(
            f"mujoco-warp overflowed a buffer in {worlds} of {per_world.shape[0]} worlds and dropped work: "
            + ", ".join(names)
        )
