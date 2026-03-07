from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from rlworld.rl.configs.observations import ObservationTermConfig
from rlworld.rl.envs.managers.base import BaseManager
from rlworld.rl.storages import CircularBuffer
from rlworld.rl.configs.observations.noise import apply_noise

if TYPE_CHECKING:
    from rlworld.rl.envs import World


@dataclass
class ObsManagerConfig:
    """Configuration for observation manager.

    This is the common config used by both Genesis and Newton environments.
    """
    num_envs: int
    obs_group: dict[str, list[ObservationTermConfig]]
    enable_noise: bool = True


class ObservationManager(BaseManager):
    """Manages observation generation and processing for RL environments.

    This is a simulator-agnostic observation manager that can be used by both
    Genesis and Newton environments. The observation processing is configured
    through ObservationTermConfig objects that specify:
    - Which observation function to call
    - Scaling factors to apply
    - Additional parameters for the observation function
    - History management (history_length, flatten_history_dim)

    The observation functions themselves are simulator-specific and should be
    implemented in the respective mdp/observations/ modules.

    History Management:
    - Each observation term can maintain a history buffer if history_length > 0
    - History is stored in a CircularBuffer (oldest to newest ordering)
    - History can be flattened (shape: [num_envs, history_length * obs_dim])
      or kept separate (shape: [num_envs, history_length, obs_dim])
    """

    def __init__(
        self,
        env: "World",
        config: ObsManagerConfig
    ):
        BaseManager.__init__(self, env=env)

        self.config = config

        # Observation buffers (populated during runtime)
        self.obs_dict = {}  # Dictionary of observation tensors grouped by name
        self.extras = {}  # Additional information (e.g., robot state for logging)

        # History buffers for each observation term with history enabled
        # Structure: {group_name: {term_name: CircularBuffer}}
        self._group_obs_term_history_buffer: dict[str, dict[str, CircularBuffer]] = {}

        # Initialize history buffers for each group
        self._initialize_history_buffers()

        # Term index mapping for extraction
        self._group_term_indices: dict[str, dict[str, tuple[int, int]]] = {}
        self._is_term_indices_built = False

        # Initialize vision system if enabled
        # if self.config.use_vision:
        #     raise NotImplementedError("Vision observations not yet implemented in common manager")

    # ========== Initialization ==========

    def _initialize_history_buffers(self) -> None:
        """Initialize circular buffers for observation terms with history enabled.

        Creates a CircularBuffer for each observation term that has history_length > 0.
        The buffers are organized by group and term name for easy access during
        observation processing.
        """
        for group_name, terms in self.config.obs_group.items():
            # Initialize history buffer dictionary for this group
            self._group_obs_term_history_buffer[group_name] = {}

            for term_idx, obs_term in enumerate(terms):
                # Check if this term has history enabled
                if obs_term.history_length > 0:
                    # Create a unique term name (using index if name not available)
                    term_name = getattr(obs_term, 'name', f"term_{term_idx}")

                    # Create circular buffer for this term
                    # Use self.env.device instead of gs.device for simulator independence
                    self._group_obs_term_history_buffer[group_name][term_name] = CircularBuffer(
                        max_len=obs_term.history_length,
                        batch_size=self.config.num_envs,
                        device=self.env.device
                    )

    def _build_term_indices(self) -> None:
        """Build mapping from term function to its position in concatenated obs."""
        for group_name, terms in self.config.obs_group.items():
            self._group_term_indices[group_name] = {}

            current_idx = 0
            for term_idx, obs_term in enumerate(terms):
                # Get term name
                term_name = getattr(obs_term.func, '__name__', f"term_{term_idx}")

                # Compute this term's dimension
                dummy_value = obs_term.func(self.env, **obs_term.params)
                base_dim = dummy_value.shape[-1]

                # Account for history
                history_length = getattr(obs_term, 'history_length', 0)
                flatten_history = getattr(obs_term, 'flatten_history_dim', True)

                if history_length > 0 and flatten_history:
                    term_dim = base_dim * history_length
                else:
                    term_dim = base_dim

                # Store indices
                self._group_term_indices[group_name][term_name] = (current_idx, current_idx + term_dim)
                current_idx += term_dim

    # ========== Public API ==========

    def calculate_obs_dim(self) -> dict[str, int]:
        """Calculate the dimensionality of each observation group.

        Processes observations once to determine the size of each group's
        observation vector. Useful for initializing neural network input layers.

        Note: This accounts for history by processing observations with
        update_history=False to avoid polluting history buffers.

        Returns:
            Dictionary mapping observation group names to their dimensions.
            Example: {"actor": 48, "critic": 52}
        """
        if not self._is_term_indices_built:
            self._build_term_indices()
        self.process_observations(update_history=False)
        return defaultdict(int, {group: tensor.shape[-1] for group, tensor in self.obs_dict.items()})

    def get_observation(self) -> dict[str, torch.Tensor]:
        """Get processed observations for actor and critic networks.

        Returns:
            Dictionary of observation tensors keyed by group name.
        """
        return self.obs_dict

    def get_robot_state(self) -> torch.Tensor | None:
        """Get current robot state information.

        Returns robot state stored in extras during observation processing.
        Useful for logging, visualization, or privileged information.

        Returns:
            RobotState object containing current robot state information.
        """
        return self.extras.get("robot_state", None)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset observation history for specified environments.

        Clears the history buffers for all observation terms with history enabled.

        Args:
            env_ids: Indices of environments to reset. If None, all environments are reset.
        """
        if env_ids is None:
            env_ids = torch.arange(self.config.num_envs, device=self.env.device)

        # Reset history buffers for each group
        for group_name, history_buffers in self._group_obs_term_history_buffer.items():
            for term_name, buffer in history_buffers.items():
                buffer.reset(batch_ids=env_ids)

    def extract_term(
        self,
        group_name: str,
        term_name: str,
        observations: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Extract a specific observation term from concatenated observations.

        Args:
            group_name: Observation group name (e.g., "actor")
            term_name: Function name of the term (e.g., "dof_vel")
            observations: Optional pre-computed observations. If None, uses self.obs_dict

        Returns:
            Extracted term tensor
        """

        if observations is None:
            observations = self.obs_dict[group_name]

        with torch.no_grad():
            start_idx, end_idx = self._group_term_indices[group_name][term_name]
            result = observations[:, start_idx:end_idx]

        return result

    def get_raw_term(self, term_func: callable, **params) -> torch.Tensor:
        """
        Get raw (current, no history) value of an observation term.

        Args:
            term_func: Observation function
            **params: Parameters for the function

        Returns:
            Raw observation tensor (batch, term_dim)
        """
        return term_func(self.env, **params)

    # ========== Core Processing ==========

    def process_observations(self, update_history: bool = True) -> None:
        """Process and update all observation components.

        This is the main observation processing pipeline:
        1. Iterates through each observation group (actor, critic, etc.)
        2. For each group, computes all observation terms defined in config
        3. Applies scaling factors to each term
        4. Updates history buffers if the term has history enabled
        5. Retrieves observations from history buffers (flattened or not)
        6. Concatenates all terms within a group into a single tensor

        The processed observations are stored in self.obs_dict.

        Args:
            update_history: Whether to append new observations to history buffers.
                Default is True. Set to False when you want to peek at observations
                without modifying history (e.g., during initialization).

        Example:
            config.obs_group = {
                "actor": [
                    ObservationTermConfig(
                        func=get_base_lin_vel,
                        scale=1.0,
                        history_length=3,  # Keep 3 timesteps of history
                        flatten_history_dim=True  # Flatten to single vector
                    ),
                    ObservationTermConfig(
                        func=get_joint_pos,
                        scale=0.5,
                        history_length=0  # No history
                    ),
                ],
            }

            After processing:
            - base_lin_vel has shape (num_envs, 3 * 3) = (num_envs, 9) after flattening
            - joint_pos has shape (num_envs, 12)
            - Final actor obs: (num_envs, 9 + 12) = (num_envs, 21)
        """
        self.obs_dict = {}

        for group_name, terms in self.config.obs_group.items():
            obs_list = []

            # Process each observation term in this group
            for term_idx, obs_term in enumerate(terms):
                func = obs_term.func  # Observation function to call
                scale = obs_term.scale  # Scaling factor to apply

                # Call observation function and apply scaling
                obs_value = func(self.env, **obs_term.params)

                if self.config.enable_noise and obs_term.noise is not None:
                    obs_value = apply_noise(obs_value, obs_term.noise)

                if obs_term.clip is not None:
                    obs_value = obs_value.clip_(min=obs_term.clip[0], max=obs_term.clip[1])

                obs_value = obs_value * scale

                # Handle history if enabled for this term
                term_name = getattr(obs_term, 'name', f"term_{term_idx}")
                history_length = getattr(obs_term, 'history_length', 0)

                if history_length > 0:
                    # Get the circular buffer for this term
                    circular_buffer = self._group_obs_term_history_buffer[group_name][term_name]

                    # Update history if requested
                    if update_history:
                        circular_buffer.append(obs_value)
                    elif circular_buffer._buffer is None:
                        # Initialize buffer on first call (even if not updating)
                        circular_buffer.append(obs_value)

                    # Retrieve observation from history buffer
                    flatten_history = getattr(obs_term, 'flatten_history_dim', True)
                    if flatten_history:
                        # Flatten history: (num_envs, history_length, obs_dim) -> (num_envs, history_length * obs_dim)
                        obs_with_history = circular_buffer.buffer.reshape(self.config.num_envs, -1)
                    else:
                        # Keep history dimension: (num_envs, history_length, obs_dim)
                        obs_with_history = circular_buffer.buffer

                    obs_list.append(obs_with_history)
                else:
                    # No history, use current observation directly
                    obs_list.append(obs_value)

            # Concatenate all terms in this group
            self.obs_dict[group_name] = torch.concat(obs_list, dim=-1)

    def advance(self) -> None:
        """Update observations at each timestep.

        Called by the environment after physics simulation and before
        computing rewards. Ensures observations reflect the current state
        and history buffers are updated.
        """
        self.process_observations(update_history=True)

    def __str__(self) -> str:
        """Pretty print observation manager configuration."""
        from rlworld.rl.utils.pretty import create_manager_table, table_to_string, format_shape

        output_parts = []

        for group_name, terms in self.config.obs_group.items():
            rows = []
            total_dim = 0

            for idx, obs_term in enumerate(terms):
                func_name = getattr(obs_term.func, '__name__', f"term_{idx}")

                # Calculate actual dimension
                try:
                    dummy = obs_term.func(self.env, **obs_term.params)
                    base_dim = dummy.shape[-1]
                except Exception:
                    base_dim = "?"

                # Account for history
                history_str = "-"
                display_dim = base_dim
                if obs_term.history_length > 0:
                    mode = "flatten" if obs_term.flatten_history_dim else "stack"
                    history_str = f"{obs_term.history_length} ({mode})"
                    if obs_term.flatten_history_dim and isinstance(base_dim, int):
                        display_dim = base_dim * obs_term.history_length

                if isinstance(display_dim, int):
                    total_dim += display_dim

                # Format scale
                scale_str = f"{obs_term.scale}" if obs_term.scale != 1.0 else "1.0"

                # Format noise
                noise_str = "-"
                if obs_term.noise is not None:
                    if self.config.enable_noise:
                        noise_str = type(obs_term.noise).__name__
                    else:
                        noise_str = f"{type(obs_term.noise).__name__} (off)"

                rows.append([idx, func_name, format_shape(base_dim), scale_str, history_str, noise_str])

            table = create_manager_table(
                title=f"Observation Space ({group_name})",
                columns=["Idx", "Name", "Shape", "Scale", "History", "Noise"],
                rows=rows,
                footer=f"Total: {total_dim} dims" if isinstance(total_dim, int) else None
            )
            output_parts.append(table_to_string(table))

        return "\n".join(output_parts)