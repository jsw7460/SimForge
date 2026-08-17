from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco_warp
import torch
import warp as wp

from rlworld.rl.envs.managers.common.contact import BaseContactManager, ContactGroup
from rlworld.rl.envs.managers.newton.contact_sensor import NewtonContactSensor

if TYPE_CHECKING:
    from rlworld.rl.envs import World


class NewtonContactManager(BaseContactManager):
    """Named-group contact manager for Newton environments.

    Each :class:`rlworld.rl.configs.sensors.ContactSensorCfg` becomes a
    named group, backed by :class:`NewtonContactSensor` (which wraps a
    native ``newton.sensors.SensorContact`` and adds substep history).
    Those wrappers are built post scene-build by ``NewtonSceneManager``
    and stored in ``scene_manager.sensors`` keyed by ``cfg.name``; here
    we just adopt them.
    """

    def __init__(self, env: World):
        super().__init__(env)
        # group name -> NewtonContactSensor
        self._group_sensors: dict[str, NewtonContactSensor] = {}
        # Lazily captured CUDA graph for the post-reset forward (see
        # refresh_after_reset); None until first capture or when the
        # scene manager runs without graphs.
        self._refresh_graph = None

    # -- sensor registration --

    def register_sensors(self) -> None:
        """Discover all contact sensors in the scene and register each as a group."""
        for sensor_name, sensor in self.env.scene_manager.sensors.items():
            if not isinstance(sensor, NewtonContactSensor):
                continue
            # ContactSensorCfg path — tracking names already resolved
            # (world-0 leaf names) by the wrapper.
            self._group_sensors[sensor_name] = sensor
            self._register_group(sensor_name, list(sensor.tracked_names), tuple(sensor.cfg.fields))

    # -- abstract impl --

    def _compute_group_contact_force(self, group: ContactGroup) -> torch.Tensor | None:
        return self._group_sensors[group.name].compute_force()

    def _compute_group_contact_force_history(self, group: ContactGroup) -> torch.Tensor | None:
        return self._group_sensors[group.name].compute_history()  # (num_envs, N, H, 3) or None

    # -- per-env reset --

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        super().reset(env_ids)
        for sensor in self._group_sensors.values():
            sensor.reset(env_ids)

    def refresh_after_reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Recompute contacts for the freshly reset poses (MuJoCo semantics).

        Syncs the reset Newton state into the mjwarp buffers and runs a
        forward pass — the full MuJoCo pipeline up to constraint forces
        with no integration — so post-reset sensor reads carry the NEW
        pose's contacts and forces, exactly like mjlab's reset path.
        """
        if not self._group_sensors:
            return
        sm = self.env.scene_manager
        solver = sm.solver
        if not sm._use_mujoco_contacts:
            raise NotImplementedError(
                "refresh_after_reset only supports the use_mujoco_contacts=True "
                "path; the Newton-native contact pipeline injects externally "
                "collided contacts that a lone mjwarp forward would not reproduce."
            )
        # The eager sync + forward launches hundreds of kernels from the
        # host every reset (~15 ms/step under training reset churn), so
        # capture them into a CUDA graph once and replay it — the mjwarp
        # buffers and the state_0 reference are stable across steps, and
        # the scene manager captures its step graph the same way.
        with wp.ScopedDevice(sm.model.device):
            if sm.use_cuda_graph and self._refresh_graph is None:
                with wp.ScopedCapture() as capture:
                    solver._update_mjc_data(solver.mjw_data, sm.model, sm.state_0)
                    mujoco_warp.forward(solver.mjw_model, solver.mjw_data)
                self._refresh_graph = capture.graph
            if self._refresh_graph is not None:
                wp.capture_launch(self._refresh_graph)
            else:
                solver._update_mjc_data(solver.mjw_data, sm.model, sm.state_0)
                mujoco_warp.forward(solver.mjw_model, solver.mjw_data)
        sm._update_sensors()

    # -- pretty print --

    def __str__(self) -> str:
        from rlworld.rl.utils.pretty import create_manager_table, table_to_string

        if not self._groups:
            return ""

        rows = []
        for gname, group in self._groups.items():
            for idx, name in enumerate(group.tracked_names):
                rows.append([gname, idx, name])

        table = create_manager_table(
            title="Contact Tracking (Newton)",
            columns=["Group", "Idx", "Name"],
            rows=rows,
            footer=f"{len(self._groups)} groups, {sum(g.num_tracked for g in self._groups.values())} tracked",
        )
        return table_to_string(table)
