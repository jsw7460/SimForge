from .circular_buffer import CircularBuffer
from .device_replay_buffer import DeviceReplayBuffer
from .replay_buffer import (
    ReplayBatch,
    ReplayBuffer,
)
from .rollout_storage import (
    RolloutBatch,
    RolloutStorage,
)


def make_replay_buffer(device: str, **kwargs):
    """Pick the replay buffer whose storage sits where the data does.

    ``"host"`` is right for transitions born on the host and is what
    every existing preset was measured on; ``"device"`` avoids a round
    trip per transition when they are born on the accelerator, at the
    cost of device memory. See ``DeviceReplayBuffer``'s module docstring
    for the trade and ``check_replay_buffer_parity`` for the gate that
    keeps the two interchangeable.
    """
    if device == "host":
        return ReplayBuffer(**kwargs)
    if device == "device":
        return DeviceReplayBuffer(**kwargs)
    raise ValueError(f"replay_buffer_device must be 'host' or 'device', got {device!r}")
