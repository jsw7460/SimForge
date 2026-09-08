"""The env-shaped view reward terms see when the reward chain is compiled.

A reward term is a function of ``env``: it reads state out of the engine
(``env.get_entity_data("robot").joint_vel``, ``env.contact_manager
.is_contact(...)``, ``env.command_manager.lin_vel_x``, ...) and does a
few lines of tensor math on it. The math is what ``torch.compile`` can
fuse; the reads are what it cannot trace — they go through mjlab's lazy
warp proxies, Newton's per-generation memos and Genesis's taichi
getters, the library boundaries that kill a whole-step compile.

So the two are separated in time. Every read a term makes is done
eagerly BEFORE the compiled call, into plain tensors, and the term is
then handed a :class:`RewardEnvView`: an object shaped like ``env`` whose
robot data, scene entities, sensors and contact reads come from those
tensors, and which forwards everything else (command and action
managers, ``control_dt``, ``episode_length_buf``) to the real env, all
of which already is plain tensor state.

Which reads to snapshot is not declared anywhere: it is recorded. The
reward manager's first call runs the terms eagerly through the
recording variants below, which note every robot-data attribute,
entity-data attribute, sensor and contact group the terms touch. From
then on each step refreshes exactly those. A term that reaches for
something the snapshot does not model raises at the first compiled
call rather than tracing into the engine.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from jaxrlworld.rl.envs import World
    from jaxrlworld.rl.envs.managers.common.contact import BaseContactManager

# RobotData methods that gather from a full per-body / per-site array;
# the snapshot holds the array and does the gather in-graph.
_BY_IDS = {
    "body_pos_w_by_ids": "body_pos_w_all",
    "body_lin_vel_w_by_ids": "body_lin_vel_w_all",
    "site_pos_w_by_ids": "site_pos_w_all",
    "site_lin_vel_w_by_ids": "site_lin_vel_w_all",
}


def _plain(value: Any) -> Any:
    """A tensor for anything tensor-like the engine hands back.

    mjlab's ``TorchArray`` is a warp-backed proxy whose ``__getitem__``
    returns the real tensor; a tuple (joint limits) is mapped
    element-wise; tensors and Python scalars pass through.
    """
    if isinstance(value, torch.Tensor) or value is None or isinstance(value, int | float | bool | str):
        return value
    if isinstance(value, tuple):
        return tuple(_plain(v) for v in value)
    if hasattr(value, "__getitem__"):
        return value[...]
    raise TypeError(f"The reward snapshot cannot hold a {type(value).__name__}: {value!r}")


class RewardReadRecord:
    """What the terms read, filled by one eager pass through the recorders."""

    def __init__(self) -> None:
        self.robot_attrs: set[str] = set()
        self.angmom_sensors: set[str | None] = set()
        self.entity_attrs: dict[str, set[str]] = {}
        self.sensors: set[str] = set()
        self.contact_is_contact: set[str] = set()
        self.contact_force: set[str] = set()
        # Command / action manager reads and the env's own tensor
        # attributes are plain tensors already, but the objects that
        # serve them are not traceable (CommandManager resolves column
        # names in ``__getattr__`` through ``object.__getattribute__``),
        # so they are snapshotted rather than delegated.
        self.command_attrs: set[str] = set()
        self.command_terms: dict[str, set[str]] = {}
        self.action_attrs: set[str] = set()
        self.env_tensor_attrs: set[str] = set()


# ---------------------------------------------------------------------
# recording wrappers (the eager first call)
# ---------------------------------------------------------------------


class _RecordingRobotData:
    def __init__(self, real, record: RewardReadRecord):
        self._real = real
        self._record = record

    def __getattr__(self, name: str):
        value = getattr(self._real, name)
        if name in _BY_IDS:
            self._record.robot_attrs.add(_BY_IDS[name])
            return value
        if name == "angular_momentum_w":

            def angular_momentum_w(sensor_name: str | None = None):
                self._record.angmom_sensors.add(sensor_name)
                return value(sensor_name=sensor_name)

            return angular_momentum_w
        if callable(value) and not isinstance(value, torch.Tensor):
            raise NotImplementedError(
                f"A reward term calls RobotData.{name}(...), which the compiled reward chain's snapshot does "
                "not model; keep compile_terms off for this preset or extend reward_view.py."
            )
        self._record.robot_attrs.add(name)
        return value


class _RecordingAttrs:
    """Records every attribute name read off ``real`` and serves the real value."""

    def __init__(self, real, attrs: set[str]):
        self._real = real
        self._attrs = attrs

    def __getattr__(self, name: str):
        self._attrs.add(name)
        return getattr(self._real, name)


class _RecordingCommandManager:
    def __init__(self, real, record: RewardReadRecord):
        self._real = real
        self._record = record

    def get_command(self, name: str):
        self._record.command_terms.setdefault(name, set()).add("command")
        return self._real.get_command(name)

    def get_term(self, name: str):
        return _RecordingAttrs(self._real.get_term(name), self._record.command_terms.setdefault(name, set()))

    def __getattr__(self, name: str):
        self._record.command_attrs.add(name)
        return getattr(self._real, name)


class _RecordingEntity:
    def __init__(self, real, attrs: set[str]):
        self._real = real
        self._attrs = attrs

    @property
    def data(self):
        return _RecordingAttrs(self._real.data, self._attrs)

    def __getattr__(self, name: str):
        return getattr(self._real, name)


class _RecordingSensor:
    def __init__(self, real, name: str, record: RewardReadRecord):
        self._real = real
        self._name = name
        self._record = record

    @property
    def data(self):
        self._record.sensors.add(self._name)
        return self._real.data

    def __getattr__(self, name: str):
        return getattr(self._real, name)


class _RecordingScene:
    def __init__(self, real, record: RewardReadRecord):
        self._real = real
        self._record = record

    def get_entity(self, name: str):
        return _RecordingEntity(self._real.get_entity(name), self._record.entity_attrs.setdefault(name, set()))

    def get_sensor(self, name: str):
        return _RecordingSensor(self._real.get_sensor(name), name, self._record)

    def __getattr__(self, name: str):
        return getattr(self._real, name)


class _RecordingContactManager:
    def __init__(self, real: BaseContactManager, record: RewardReadRecord):
        self._real = real
        self._record = record

    def is_contact(self, group_name: str, order=None):
        self._record.contact_is_contact.add(group_name)
        return self._real.is_contact(group_name, order)

    def contact_force(self, group_name: str, order=None):
        self._record.contact_force.add(group_name)
        return self._real.contact_force(group_name, order)

    def __getattr__(self, name: str):
        return getattr(self._real, name)


class RecordingEnvView:
    """``env`` with recorders in front of the engine-facing readers."""

    def __init__(self, env: World, record: RewardReadRecord):
        self._env = env
        self._record = record
        self.scene_manager = _RecordingScene(env.scene_manager, record)
        self.contact_manager = _RecordingContactManager(env.contact_manager, record)
        self.command_manager = _RecordingCommandManager(env.command_manager, record)
        self.act_manager = _RecordingAttrs(env.act_manager, record.action_attrs)

    def get_robot_data(self, entity_name: str | None = None):
        name = entity_name or self._env.robot_entity_name
        if name != self._env.robot_entity_name:
            raise NotImplementedError(f"The compiled reward chain snapshots the driven robot only, not {name!r}.")
        return _RecordingRobotData(self._env.get_robot_data(name), self._record)

    def get_entity_data(self, entity_name: str | None = None):
        return self.get_robot_data(entity_name)

    @property
    def robot_data(self):
        return self.get_robot_data()

    def __getattr__(self, name: str):
        value = getattr(self._env, name)
        if isinstance(value, torch.Tensor):
            self._record.env_tensor_attrs.add(name)
        return value


# ---------------------------------------------------------------------
# the snapshot (refreshed every step) and the view the compiled call sees
# ---------------------------------------------------------------------


class _RobotDataSnapshot:
    """Recorded RobotData attributes as plain tensors, plus the gathers."""

    def __init__(self) -> None:
        self._angmom: dict[str | None, torch.Tensor] = {}

    def body_pos_w_by_ids(self, body_ids):
        return self.body_pos_w_all[:, body_ids, :]

    def body_lin_vel_w_by_ids(self, body_ids):
        return self.body_lin_vel_w_all[:, body_ids, :]

    def site_pos_w_by_ids(self, site_ids):
        return self.site_pos_w_all[:, site_ids, :]

    def site_lin_vel_w_by_ids(self, site_ids):
        return self.site_lin_vel_w_all[:, site_ids, :]

    def angular_momentum_w(self, sensor_name: str | None = None):
        return self._angmom[sensor_name]

    def __getattr__(self, name: str):
        raise AttributeError(
            f"RobotData.{name} was not read during the recording pass, so the reward snapshot does not hold it."
        )


class _Holder:
    """A bag of tensors under attribute names (entity data, sensor data)."""

    def __getattr__(self, name: str):
        raise AttributeError(
            f"{name!r} was not read during the recording pass, so the reward snapshot does not hold it."
        )


class _EntitySnapshot:
    def __init__(self) -> None:
        self.data = _Holder()


class _CommandSnapshot(_Holder):
    """Recorded command columns / term attributes as tensors."""

    def __init__(self) -> None:
        self.__dict__["_terms"] = {}

    def get_command(self, name: str):
        return self._terms[name].command

    def get_term(self, name: str):
        return self._terms[name]


class _SceneSnapshot:
    def __init__(self) -> None:
        self._entities: dict[str, _EntitySnapshot] = {}
        self._sensors: dict[str, _Holder] = {}

    def get_entity(self, name: str):
        return self._entities[name]

    def get_sensor(self, name: str):
        return self._sensors[name]


class _ContactManagerView:
    """The contact manager with its two engine reads served from tensors.

    ``is_contact`` and ``contact_force`` are per-group snapshots; the
    timing helpers (``compute_first_contact``, ``current_air_time``,
    ...) and the order re-indexing run on the real manager, whose
    buffers are plain tensors it updates itself every substep.
    """

    def __init__(self, real: BaseContactManager):
        self._real = real
        self._is_contact: dict[str, torch.Tensor] = {}
        self._force: dict[str, torch.Tensor] = {}

    def is_contact(self, group_name: str, order=None):
        return self._real._apply_order(self._is_contact[group_name], group_name, order)

    def contact_force(self, group_name: str, order=None):
        self._real._require_force(self._real._get_group(group_name))
        return self._real._apply_order_3d(self._force[group_name], group_name, order)

    def contact_force_history(self, group_name: str, order=None):
        raise NotImplementedError("contact_force_history is not part of the compiled reward chain's snapshot.")

    def __getattr__(self, name: str):
        return getattr(self._real, name)


class RewardEnvView:
    """``env`` for the compiled reward chain: engine reads served from a
    snapshot that :meth:`refresh` fills eagerly each step."""

    def __init__(self, env: World, record: RewardReadRecord):
        self._env = env
        self._record = record
        self._robot = _RobotDataSnapshot()
        self.scene_manager = _SceneSnapshot()
        for name in record.entity_attrs:
            self.scene_manager._entities[name] = _EntitySnapshot()
        for name in record.sensors:
            self.scene_manager._sensors[name] = _Holder()
        self.contact_manager = _ContactManagerView(env.contact_manager)
        self.command_manager = _CommandSnapshot()
        for name in record.command_terms:
            self.command_manager._terms[name] = _Holder()
        self.act_manager = _Holder()
        self._env_tensors: dict[str, torch.Tensor] = {}

    def refresh(self) -> None:
        """Read everything the terms need out of the engine, as tensors."""
        env = self._env
        for name in self._record.env_tensor_attrs:
            self._env_tensors[name] = getattr(env, name)
        cmd = env.command_manager
        command = self.command_manager.__dict__
        for name in self._record.command_attrs:
            command[name] = getattr(cmd, name)
        for term_name, attrs in self._record.command_terms.items():
            term = cmd.get_term(term_name)
            holder = self.command_manager._terms[term_name].__dict__
            for name in attrs:
                holder[name] = getattr(term, name)
        act = self.act_manager.__dict__
        for name in self._record.action_attrs:
            act[name] = getattr(env.act_manager, name)
        rd = env.get_robot_data(env.robot_entity_name)
        robot = self._robot.__dict__
        for name in self._record.robot_attrs:
            robot[name] = _plain(getattr(rd, name))
        for sensor_name in self._record.angmom_sensors:
            self._robot._angmom[sensor_name] = _plain(rd.angular_momentum_w(sensor_name=sensor_name))
        for ent_name, attrs in self._record.entity_attrs.items():
            data = env.scene_manager.get_entity(ent_name).data
            holder = self.scene_manager._entities[ent_name].data.__dict__
            for name in attrs:
                holder[name] = _plain(getattr(data, name))
        for name in self._record.sensors:
            self.scene_manager._sensors[name].__dict__["data"] = _plain(env.scene_manager.get_sensor(name).data)
        cm = env.contact_manager
        for name in self._record.contact_is_contact:
            self.contact_manager._is_contact[name] = cm._compute_group_is_contact(cm._get_group(name))
        for name in self._record.contact_force:
            group = cm._get_group(name)
            cm._require_force(group)
            force = cm._compute_group_contact_force(group)
            if force is None:
                force = torch.zeros(cm.num_envs, group.num_tracked, 3, device=cm.device)
            self.contact_manager._force[name] = force

    def get_robot_data(self, entity_name: str | None = None):
        name = entity_name or self._env.robot_entity_name
        if name != self._env.robot_entity_name:
            raise NotImplementedError(f"The compiled reward chain snapshots the driven robot only, not {name!r}.")
        return self._robot

    def get_entity_data(self, entity_name: str | None = None):
        return self.get_robot_data(entity_name)

    @property
    def robot_data(self):
        return self._robot

    def __getattr__(self, name: str):
        tensors = self.__dict__["_env_tensors"]
        if name in tensors:
            return tensors[name]
        return getattr(self.__dict__["_env"], name)
