import torch


class CircularBuffer:
    """Circular buffer for storing observation history.

    Maintains a fixed-length history of observations with automatic
    oldest-data removal when new data is appended.

    Args:
        max_len: Maximum number of timesteps to store.
        batch_size: Number of parallel environments.
        device: Torch device.
    """

    def __init__(self, max_len: int, batch_size: int, device: torch.device):
        self.max_length = max_len
        self.batch_size = batch_size
        self.device = device
        self._buffer = None  # Initialized on first append
        self._current_idx = 0

    @property
    def buffer(self) -> torch.Tensor:
        """Get the full buffer in chronological order (oldest to newest).

        Returns:
            Tensor of shape (batch_size, max_length, obs_dim)
        """
        if self._buffer is None:
            raise RuntimeError("Buffer not initialized. Call append() first.")

        # Reorder buffer to be chronological (oldest to newest)
        indices = torch.arange(self._current_idx, self._current_idx + self.max_length,
                               device=self.device) % self.max_length
        return self._buffer[:, indices]

    def append(self, data: torch.Tensor) -> None:
        """Append new observation to the buffer.

        Args:
            data: New observation tensor of shape (batch_size, obs_dim)
        """
        # Initialize buffer on first call
        if self._buffer is None:
            obs_dim = data.shape[1:]
            self._buffer = torch.zeros(
                (self.batch_size, self.max_length, *obs_dim),
                dtype=data.dtype,
                device=self.device
            )

        # Add new data at current index (overwrites oldest)
        self._buffer[:, self._current_idx] = data

        # Move to next index (circular)
        self._current_idx = (self._current_idx + 1) % self.max_length

    def reset(self, batch_ids: torch.Tensor | None = None) -> None:
        """Reset buffer for specified environments.

        Args:
            batch_ids: Indices of environments to reset. If None, reset all.
        """
        if self._buffer is None:
            return

        if batch_ids is None:
            self._buffer.zero_()
        else:
            self._buffer[batch_ids] = 0.0
