"""Neural network actuator models (MLP and LSTM)."""

from __future__ import annotations

import io
from collections.abc import Sequence

import torch

from .actuator_base import ActuatorBase
from .actuator_cfg import ActuatorNetLSTMCfg, ActuatorNetMLPCfg


class ActuatorNetMLP(ActuatorBase):
    """Actuator model based on a multi-layer perceptron with joint history.

    A pretrained TorchScript network maps a history of
    ``(position_error, velocity)`` to output torque per joint, following
    the approach of Hwangbo et al. (2019).

    The history buffer stores the last ``max(input_idx) + 1`` steps.
    At each call to :meth:`compute`, the buffer is rolled and the
    selected history indices are concatenated as network input.
    """

    cfg: ActuatorNetMLPCfg

    def __init__(
        self,
        cfg: ActuatorNetMLPCfg,
        num_envs: int,
        num_joints: int,
        device: str,
        joint_names: list[str] | None = None,
    ) -> None:
        super().__init__(cfg, num_envs, num_joints, device, joint_names)

        if not cfg.network_file:
            raise ValueError("network_file must be specified for ActuatorNetMLP")

        self.network = torch.jit.load(cfg.network_file, map_location=device).eval()

        history_length = max(cfg.input_idx) + 1
        self._pos_error_history = torch.zeros(
            num_envs, history_length, num_joints, device=device
        )
        self._vel_history = torch.zeros(
            num_envs, history_length, num_joints, device=device
        )

    def reset(self, env_ids: Sequence[int]) -> None:
        self._pos_error_history[env_ids] = 0.0
        self._vel_history[env_ids] = 0.0

    def compute(
        self,
        target_pos: torch.Tensor,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> torch.Tensor:
        # Roll history and update current step
        self._pos_error_history = self._pos_error_history.roll(1, 1)
        self._pos_error_history[:, 0] = target_pos - joint_pos

        self._vel_history = self._vel_history.roll(1, 1)
        self._vel_history[:, 0] = joint_vel

        # Gather selected history indices → (num_envs * num_joints, len(input_idx))
        pos_input = torch.cat(
            [self._pos_error_history[:, i].unsqueeze(2) for i in self.cfg.input_idx],
            dim=2,
        )
        pos_input = pos_input.view(self._num_envs * self._num_joints, -1)

        vel_input = torch.cat(
            [self._vel_history[:, i].unsqueeze(2) for i in self.cfg.input_idx],
            dim=2,
        )
        vel_input = vel_input.view(self._num_envs * self._num_joints, -1)

        # Scale and concatenate
        if self.cfg.input_order == "pos_vel":
            network_input = torch.cat(
                [pos_input * self.cfg.pos_scale, vel_input * self.cfg.vel_scale],
                dim=1,
            )
        else:
            network_input = torch.cat(
                [vel_input * self.cfg.vel_scale, pos_input * self.cfg.pos_scale],
                dim=1,
            )

        # Forward pass
        with torch.inference_mode():
            torques = self.network(network_input)

        self.computed_effort = (
            torques.view(self._num_envs, self._num_joints) * self.cfg.torque_scale
        )
        self.applied_effort = self._clip_effort(self.computed_effort)
        return self.applied_effort


class ActuatorNetLSTM(ActuatorBase):
    """Actuator model based on a recurrent neural network (LSTM).

    The LSTM hidden state implicitly captures temporal dynamics,
    removing the need for an explicit history buffer.  Based on
    Rudin et al. (2022).

    The network receives ``(position_error, velocity)`` at each step
    and outputs per-joint torques.
    """

    cfg: ActuatorNetLSTMCfg

    def __init__(
        self,
        cfg: ActuatorNetLSTMCfg,
        num_envs: int,
        num_joints: int,
        device: str,
        joint_names: list[str] | None = None,
    ) -> None:
        super().__init__(cfg, num_envs, num_joints, device, joint_names)

        if not cfg.network_file:
            raise ValueError("network_file must be specified for ActuatorNetLSTM")

        self.network = torch.jit.load(cfg.network_file, map_location=device).eval()

        # Infer LSTM dimensions from loaded weights
        num_layers = len(self.network.lstm.state_dict()) // 4
        hidden_dim = self.network.lstm.state_dict()["weight_hh_l0"].shape[1]

        flat_size = num_envs * num_joints
        # Network input: (flat_size, 1, 2)  — sequence_len=1, features=(pos_err, vel)
        self.sea_input = torch.zeros(flat_size, 1, 2, device=device)
        self.sea_hidden_state = torch.zeros(
            num_layers, flat_size, hidden_dim, device=device
        )
        self.sea_cell_state = torch.zeros(
            num_layers, flat_size, hidden_dim, device=device
        )

        # Views for per-env reset
        layer_shape = (num_layers, num_envs, num_joints, hidden_dim)
        self._hidden_per_env = self.sea_hidden_state.view(layer_shape)
        self._cell_per_env = self.sea_cell_state.view(layer_shape)

    def reset(self, env_ids: Sequence[int]) -> None:
        with torch.no_grad():
            self._hidden_per_env[:, env_ids] = 0.0
            self._cell_per_env[:, env_ids] = 0.0

    def compute(
        self,
        target_pos: torch.Tensor,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> torch.Tensor:
        self.sea_input[:, 0, 0] = (target_pos - joint_pos).flatten()
        self.sea_input[:, 0, 1] = joint_vel.flatten()

        with torch.inference_mode():
            torques, (self.sea_hidden_state[:], self.sea_cell_state[:]) = (
                self.network(
                    self.sea_input,
                    (self.sea_hidden_state, self.sea_cell_state),
                )
            )

        self.computed_effort = torques.reshape(self._num_envs, self._num_joints)
        self.applied_effort = self._clip_effort(self.computed_effort)
        return self.applied_effort
