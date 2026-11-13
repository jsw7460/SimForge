"""Raise on a mujoco-warp contact-slot overflow instead of simulating past it."""

from __future__ import annotations

from collections.abc import Callable

import torch
import warp as wp


class ContactOverflowGuard:
    """Raise once mujoco-warp dropped contacts for want of ``nconmax`` slots.

    mujoco-warp counts every contact it finds in ``Data.nacon`` but stores
    only the first ``naconmax``; the rest are skipped without an error, the
    bodies they belonged to interpenetrate, and the state blows up a few
    steps later with nothing naming the cause. The count is read back
    without waiting for the device: each check enqueues a copy into
    page-locked host memory, on warp's own stream so it follows the
    collision kernels, and inspects the copy the previous check enqueued.
    An overflow therefore raises one control step after it happened. On a
    CPU device the count is read directly.

    ``data`` is a callable returning the live mujoco-warp ``Data``: a
    backend may replace it after this guard is built (mjlab's model
    expansion), so it is not held.
    """

    def __init__(self, data: Callable[[], object], option_name: str) -> None:
        self._data = data
        self._option_name = option_name
        live = data()
        self._naconmax = int(live.naconmax)
        warp_device = live.nacon.device
        self._on_cuda = not warp_device.is_cpu
        if self._on_cuda:
            self._stream = torch.cuda.ExternalStream(wp.get_stream(warp_device).cuda_stream)
            self._host = torch.zeros(1, dtype=torch.int32, pin_memory=True)
            self._event = torch.cuda.Event()

    def check(self) -> None:
        """Raise if the previous check's count overflowed; enqueue this one's."""
        nacon = wp.to_torch(self._data().nacon)
        if not self._on_cuda:
            self._raise_if_over(int(nacon[0]))
            return
        self._event.synchronize()
        self._raise_if_over(int(self._host[0]))
        with torch.cuda.stream(self._stream):
            self._host.copy_(nacon, non_blocking=True)
            self._event.record(self._stream)

    def _raise_if_over(self, nacon: int) -> None:
        if nacon > self._naconmax:
            raise RuntimeError(
                f"mujoco-warp found {nacon} contacts for {self._naconmax} slots and dropped the rest. "
                f"Increase the scene option {self._option_name!r}."
            )
