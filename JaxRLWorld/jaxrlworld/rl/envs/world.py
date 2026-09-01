from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, Tuple

import numpy as np
import torch
from gymnasium import spaces

from jaxrlworld.rl.envs.lifecycle import LifecycleEvent, LifecycleManager

if TYPE_CHECKING:
    from jaxrlworld.rl.configs.scene.entity_selector import ResolvedEntity, SceneEntitySelector
    from jaxrlworld.rl.envs.managers.common.robot_state_writer_protocol import RobotStateWriterProtocol
    from jaxrlworld.rl.envs.robot_data import RigidObjectData, RobotData


def _empty_articulation_indexing(device):
    """An :class:`ArticulationIndexing` describing something with no joints."""
    from jaxrlworld.rl.envs.indexing import ArticulationIndexing

    empty_long = torch.zeros(0, device=device, dtype=torch.long)
    empty_float = torch.zeros(0, device=device, dtype=torch.float32)
    return ArticulationIndexing(
        joint_names=(),
        sim_indices=empty_long,
        sim_to_canonical=empty_long.clone(),
        joint_limits_lower=empty_float,
        joint_limits_upper=empty_float.clone(),
    )


class World(ABC):
    """Abstract base class for all RL environments."""

    sim_name: str = "World"
    sim_type: str = "world"  # lowercase key for ManagerRegistry ("genesis", "newton", "mujoco")

    # Required attributes (set in subclass __init__)
    num_envs: int
    device: torch.device
    seed: int

    # ── Timing ──────────────────────────────────────────────────────
    #
    # Three numbers define the simulation/control timing:
    #
    #   physics_dt   – timestep of one physics substep (seconds).
    #                  e.g. 0.005 s = 200 Hz physics.
    #
    #   decimation   – how many times action is repeated (physics steps
    #                  per control step).  e.g. decimation=4 means the
    #                  same action is applied for 4 physics steps before
    #                  the policy is queried again.
    #
    #   control_dt   – wall-clock time per policy step = physics_dt × decimation.
    #                  e.g. 0.005 × 4 = 0.02 s = 50 Hz control.
    #
    # For MuJoCo there is an additional `substeps` factor inside the
    # scene manager: each physics_dt is subdivided into `substeps`
    # MuJoCo mj_step calls.  The MuJoCo solver timestep is then
    # physics_dt / substeps (e.g. 0.005 / 2 = 0.0025 s).
    # Newton and Genesis handle substeps internally.
    physics_dt: float
    decimation: int
    control_dt: float

    # Managers (set in subclass _setup_environment)
    scene_manager: Any
    obs_manager: Any
    act_manager: Any
    reward_manager: Any
    termination_manager: Any
    command_manager: Any
    event_manager: Any
    contact_manager: Any

    def __init__(self):
        super().__init__()

        # ── EnvStepCache generation counter ─────────────────────────
        #
        # Observation/reward functions decorated with @EnvStepCache()
        # cache their return value and re-use it as long as
        # _cache_generation hasn't changed.  This avoids redundant
        # RobotData reads when the same quantity (e.g. dof_pos) is
        # needed by both the observation builder and a reward function
        # within the same step.
        #
        # _invalidate_cache() increments this counter, which makes
        # all cached values stale on the next access.  It is called
        # twice per step():
        #   1. After _step_physics()  – physics state changed
        #   2. After _reset_idx()     – reset envs have new state
        self._cache_generation = 0

        self._env_step_counter = 0
        self.lifecycle = LifecycleManager()

        # Which observation groups the terminal-observation pass computes.
        # ``None`` means every group. A runner that consumes only part of
        # ``final_observation`` narrows this (the on-policy runner sets
        # ``("critic",)`` — PPO's truncation bootstrap reads nothing else),
        # so terminal steps skip re-computing groups nobody reads.
        self.terminal_obs_groups: tuple[str, ...] | None = None

        # Per-entity RobotData cache for articulations. Populated by each
        # backend's env at build time; read via :meth:`get_robot_data`.
        # Initialized here so a read before the backend fills it fails with a
        # KeyError naming the entity rather than an AttributeError.
        self._robot_data_cache: dict = {}

        # Per-entity RigidObjectData cache for passive (non-articulated) scene
        # entities declared in ``scene.rigid_objects`` — a table, a graspable
        # object. Empty unless a backend's env populates it at build time;
        # read via :meth:`get_rigid_object_data`. Kept separate from the
        # articulation ``_robot_data_cache`` (mirrors IsaacLab's RigidObject vs
        # Articulation split).
        self._rigid_object_data_cache: dict = {}

        # Root-state writers for passive rigid objects (companion to
        # ``_rigid_object_data_cache``). Populated by a backend's env at build
        # time; resolved together with the articulation writers in
        # :meth:`get_root_state_writer`.
        self._rigid_object_state_writer_cache: dict = {}

        # Memoized ``_resolve_default_joint_pos`` results per entity —
        # static config, but the reset event path asks on every reset.
        self._default_joint_pos_resolved: dict = {}

        # One ArticulationIndexing per articulation entity, built on first
        # use by :meth:`entity_indexing`. The driven entity is not stored
        # here — it defers to the action manager's own indexing.
        self._entity_indexing_cache: dict = {}

    def _init_buffers(self) -> None:
        """Initialize common buffers. Call after setting num_envs and device."""
        self.rew_buf = torch.zeros(self.num_envs, device=self.device)
        self.rew_buf_per_type = defaultdict(lambda: torch.zeros(self.num_envs, device=self.device, dtype=torch.float32))
        self.extras = {}

    def _invalidate_cache(self) -> None:
        """Bump the cache generation so all @EnvStepCache values are recomputed."""
        self._cache_generation += 1

    # ========== Interactive external wrench (viewer drag) ==========
    # A single per-link world-frame force on one env, applied every
    # physics substep while set. Default ``None`` so the training step
    # path costs one identity check per substep. The viewer's force-drag
    # tool is the only writer; the backend ``_step_physics`` reads it via
    # ``_write_external_wrench``.

    _external_wrench: tuple[str, torch.Tensor, int] | None = None

    def set_external_wrench(self, link_name: str, force_w: torch.Tensor, env_idx: int) -> None:
        """Set a persistent world-frame force on ``link_name`` for one env.

        ``force_w`` is a ``(3,)`` tensor on ``self.device``. Overwrites any
        previous wrench; :meth:`clear_external_wrench` removes it.
        """
        self._external_wrench = (link_name, force_w, env_idx)

    def clear_external_wrench(self) -> None:
        """Remove the wrench and zero any residual left in the sim buffer.

        Backends whose external-force buffer persists across steps (mjlab
        ``xfrc_applied``, Genesis's accumulator) must zero it on release;
        :meth:`_flush_external_wrench` does that per backend before the
        wrench is dropped.
        """
        if self._external_wrench is not None:
            self._flush_external_wrench()
        self._external_wrench = None

    def _write_external_wrench(self) -> None:
        """Push ``self._external_wrench`` into the backend force buffer.

        Called from ``_step_physics`` once per substep, only when a wrench
        is set. Base class is a no-op; each simulator env overrides it
        with its native per-link external-force API.
        """
        return None

    def _flush_external_wrench(self) -> None:
        """Zero the backend force buffer for the current wrench's target.

        Base class is a no-op (correct for Newton, whose ``body_f`` is
        cleared every substep). Overridden by backends with a persistent
        buffer.
        """
        return None

    def _update_num_step_calls(self) -> None:
        self._env_step_counter += 1

    @property
    def env_step_counter(self) -> int:
        """Number of step() calls on this environment instance."""
        return self._env_step_counter

    # ========== Properties ==========

    @property
    def task_name(self):
        return self.env_cfg.task_name

    @property
    def action_low(self):
        return self.act_manager._clip_low

    @property
    def action_high(self):
        return self.act_manager._clip_high

    @property
    def num_actions(self) -> int:
        return self.act_manager.total_action_dim

    @property
    def max_episode_length(self) -> int:
        return self.termination_manager.max_episode_length

    @property
    def reset_buf(self) -> torch.Tensor:
        return self.termination_manager.reset_buf

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.termination_manager.episode_length_buf

    @property
    @abstractmethod
    def robot(self) -> Any:
        """Get the main robot entity/model."""
        pass

    @abstractmethod
    def get_robot_data(self, entity_name: str = "robot") -> RobotData:
        """Get the RobotData interface for a named entity.

        Args:
            entity_name: Name of the entity in the scene (default: "robot").

        Returns:
            An object satisfying the ``RobotData`` protocol.
        """
        pass

    def get_rigid_object_data(self, entity_name: str = "object") -> RigidObjectData:
        """Get the RigidObjectData (root + body reads, no joints) for a passive
        rigid entity declared in the scene's ``rigid_objects`` registry.

        Companion to :meth:`get_robot_data` for non-articulated entities such
        as a table or a graspable object. The per-backend env populates
        ``_rigid_object_data_cache`` at build time; backends without rigid
        objects leave it empty.

        Args:
            entity_name: Name of the rigid object in the scene's
                ``rigid_objects`` dict (default: "object").

        Returns:
            An object satisfying the ``RigidObjectData`` protocol.
        """
        return self._rigid_object_data_cache[entity_name]

    def get_entity_data(self, entity_name: str = "robot") -> RigidObjectData | RobotData:
        """State reader for any scene entity — articulation or rigid object.

        The read-side counterpart of :meth:`get_root_state_writer`, and the
        accessor MDP terms should use: an observation like "how high is X" is
        meaningful for a robot and for a graspable object alike, so the term
        cannot know in advance which registry ``asset_cfg.name`` lives in.
        Articulations resolve to their :class:`RobotData`
        (``_robot_data_cache``); passive bodies resolve to their
        :class:`RigidObjectData` (``_rigid_object_data_cache``).

        Mirrors IsaacLab's ``InteractiveScene.__getitem__``, which likewise
        keeps one dict per asset family and searches them in turn.

        The returned object's capability follows the entity: a term that reads
        ``joint_pos`` off a rigid object raises ``AttributeError`` naming the
        missing attribute, which is the protocol split doing its job — a cube
        has no joints to report.

        Args:
            entity_name: Name of the entity in either scene registry.

        Returns:
            :class:`RobotData` for an articulation, :class:`RigidObjectData`
            for a passive rigid object.

        Raises:
            KeyError: If the name is in neither registry.
        """
        if entity_name in self._robot_data_cache:
            return self._robot_data_cache[entity_name]
        return self._rigid_object_data_cache[entity_name]

    def get_root_state_writer(self, entity_name: str = "robot"):
        """Root-state writer for any scene entity — articulation or rigid object.

        Reset events that only move an entity's root (e.g.
        :func:`~jaxrlworld.rl.envs.mdp.events.common.reset_root_state_uniform`)
        use this so a single generic event works for both a robot and a passive
        rigid object — mirroring IsaacLab's polymorphic ``scene[name]`` lookup
        that returns either an Articulation or a RigidObject. Articulations
        resolve to their full joint+root writer (``_robot_state_writer_cache``);
        passive bodies resolve to the root-only writer
        (``_rigid_object_state_writer_cache``). Raises ``KeyError`` if the name
        is in neither registry.
        """
        if entity_name in self._robot_state_writer_cache:
            return self._robot_state_writer_cache[entity_name]
        return self._rigid_object_state_writer_cache[entity_name]

    def resolve_selector(self, selector: SceneEntitySelector) -> ResolvedEntity:
        """Resolve a sim-agnostic selector against this world's scene.

        Joint indices in the returned :class:`ResolvedEntity` are in
        canonical (``act_manager.actuated_joint_names``) order so they
        align with ``RobotData.joint_pos`` and friends.  Body / geom /
        site indices follow each backend's native ordering — those have
        no canonical concept.

        Physics-backend Worlds (Genesis / Newton / MuJoCo) override
        this; gymnasium / ManiSkill wrappers leave the base
        implementation in place because they have no scene-entity
        concept.  Backends also raise ``NotImplementedError`` for
        component kinds they cannot resolve (e.g. Genesis has no geom
        names; Newton/Genesis have no sites).
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement resolve_selector "
            f"(no scene-entity concept on this World subclass)."
        )

    @property
    def robot_entity_name(self) -> str:
        """Name of the entity the action manager drives.

        A single name while one action manager owns one articulation.
        Everything that needs to name *an* entity should take the name as
        an argument instead of reaching for this.
        """
        return getattr(self.scene_manager.config, "robot_entity_name", "robot")

    def entity_indexing(self, entity_name: str):
        """The :class:`ArticulationIndexing` for one articulation entity.

        Every entity gets its own. Sharing one entity's indexing across
        all of them — which is what happened before — means a second
        robot's joint reads, writes and selector lookups silently use the
        first robot's joint order and simulator indices. Nothing raises;
        the numbers are simply wrong.

        The driven entity returns the action manager's own indexing
        rather than an equal copy, because ``RobotData`` column order has
        to *be* the action manager's order, not merely match it.
        """
        if entity_name == self.robot_entity_name:
            act_manager = getattr(self, "act_manager", None)
            if act_manager is not None:
                return act_manager.indexing
            # Called from inside the action manager's own constructor —
            # its terms ask for this before ``env.act_manager`` is bound.
            # Build fresh and do NOT cache: a cached copy would outlive
            # construction and be handed out in place of the manager's,
            # breaking the identity that RobotData column order relies on.
            return self.scene_manager.build_articulation_indexing(
                actuated_dof_names=self._actuated_dof_names_for(entity_name),
                entity_name=entity_name,
            )

        cached = self._entity_indexing_cache.get(entity_name)
        if cached is None:
            if entity_name in self.scene_manager.config.entities:
                cached = self.scene_manager.build_articulation_indexing(
                    actuated_dof_names=self._actuated_dof_names_for(entity_name),
                    entity_name=entity_name,
                )
            else:
                # A passive rigid object. It has no joints, so it has no
                # joint indexing — answered here rather than per backend,
                # both because the answer is the same everywhere and
                # because the backends disagree about which registry a
                # prop lives in.
                cached = _empty_articulation_indexing(self.device)
            self._entity_indexing_cache[entity_name] = cached
        return cached

    def _actuated_dof_names_for(self, entity_name: str) -> list[str]:
        """Which of an entity's joints the framework should index.

        The driven entity's set is the one the action config declares.
        Any other entity takes its set from its own declared actuators,
        which is the same thing said in the scene rather than in the
        action config — and it has to be narrowed the same way, because
        a joint with no actuator has no drive slot to write a target
        into. An entity with no actuators at all is indexed whole, so a
        passive mechanism's joints can still be read and named by a
        selector.
        """
        if entity_name == self.robot_entity_name:
            return list(self.act_cfg.actuated_dof_names)
        actuators = self.scene_manager.config.entities[entity_name].articulation.actuators
        patterns = [p for actuator in actuators for p in actuator.target_names_expr]
        return patterns or [".*"]

    def _resolve_canonical_joint_ids(
        self,
        joint_name_patterns: tuple[str, ...] | None,
        preserve_order: bool = False,
        entity_name: str | None = None,
    ) -> tuple[torch.Tensor, list[str]]:
        """Resolve joint regex patterns against one entity's joint list.

        Used by every backend's :meth:`resolve_selector` to fill the
        canonical-order ``joint_ids`` field.  When ``joint_name_patterns``
        is ``None`` returns an arange covering that entity's joints.

        ``entity_name`` must be the entity the selector points at.
        Resolving every selector against the driven robot's joint list —
        the previous behaviour — hands back indices into the wrong
        articulation for any other entity.

        ``preserve_order`` mirrors mjlab's ``find_*`` semantics — when
        True the result follows the order of the regex patterns; when
        False (default) the result follows the canonical joint order.
        """
        from jaxrlworld.rl.utils import string as _su

        name = entity_name if entity_name is not None else self.robot_entity_name
        all_names = list(self.entity_indexing(name).joint_names)
        if joint_name_patterns is None:
            return (
                torch.arange(len(all_names), device=self.device, dtype=torch.long),
                all_names,
            )
        idx, names = _su.resolve_matching_names(list(joint_name_patterns), all_names, preserve_order=preserve_order)
        return torch.tensor(idx, device=self.device, dtype=torch.long), names

    def _resolve_default_joint_pos(self, entity_name: str | None = None) -> torch.Tensor:
        """Resolve one entity's ``init_state.joint_pos`` into a per-joint tensor.

        In that entity's own canonical joint order, so it can be stored
        directly on its ``RobotData.default_joint_pos``.

        The entity is named, not searched for: the previous version took
        the first entity in the scene that declared any ``joint_pos`` and
        applied it to every robot, so a second robot inherited the
        first's home pose.

        Memoized per entity: the pose is static config, but the reset
        event term calls this on every reset, and the uncached resolve
        pays a regex match plus one scalar H2D kernel per joint each
        time. Callers treat the returned tensor as read-only.
        """
        from jaxrlworld.rl.utils import string as _su

        name = entity_name if entity_name is not None else self.robot_entity_name
        cached = self._default_joint_pos_resolved.get(name)
        if cached is not None:
            return cached
        all_names = list(self.entity_indexing(name).joint_names)
        base = torch.zeros(len(all_names), device=self.device, dtype=torch.float32)

        cfg = self.scene_manager.config.entities.get(name)
        joint_pos = getattr(getattr(cfg, "init_state", None), "joint_pos", None)
        if joint_pos and isinstance(joint_pos, dict):
            matched_idx, _, matched_vals = _su.resolve_matching_names_values(joint_pos, all_names)
            for idx, val in zip(matched_idx, matched_vals):
                base[idx] = val

        self._default_joint_pos_resolved[name] = base
        return base

    @abstractmethod
    def get_robot_state_writer(self, entity_name: str = "robot") -> RobotStateWriterProtocol:
        """Get the RobotStateWriter interface for a named entity.

        Args:
            entity_name: Name of the entity in the scene (default: "robot").

        Returns:
            An object satisfying the ``RobotStateWriterProtocol``.
        """
        pass

    @property
    def robot_data(self) -> RobotData:
        """Shortcut for ``get_robot_data("robot")``."""
        return self.get_robot_data("robot")

    @property
    def heading_w(self) -> torch.Tensor:
        """Get the heading (yaw angle) of the robot in world frame.

        Returns:
            Tensor of shape [num_envs] in radians.
        """
        return self.robot_data.heading_w

    def calculate_obs_dim(self) -> dict[str, int]:
        return self.obs_manager.calculate_obs_dim()

    @property
    def action_space(self) -> spaces.Box:
        """Get the action space (gymnasium-style).

        Returns:
            spaces.Box: Continuous action space with shape (num_actions,)
        """
        num_actions = self.act_manager.total_action_dim
        # Get clip range from action manager if available
        if hasattr(self.act_manager, "clip") and self.act_manager.clip is not None:
            low, high = self.act_manager.clip
        elif hasattr(self.act_manager, "clip_actions") and self.act_manager.clip_actions is not None:
            low, high = self.act_manager.clip_actions
        else:
            low, high = -np.inf, np.inf

        return spaces.Box(
            low=np.float32(low),
            high=np.float32(high),
            shape=(num_actions,),
            dtype=np.float32,
        )

    @property
    def observation_space(self) -> Dict[str, spaces.Box]:
        """Get the observation space (gymnasium-style).

        Returns:
            Dict[str, spaces.Box]: Dictionary of observation spaces for each group
        """
        obs_shapes = self.obs_manager.calculate_obs_shapes()
        obs_spaces = {}

        for group_name, shape in obs_shapes.items():
            if isinstance(shape, list):
                raise ValueError(
                    f"Group {group_name!r} keeps its terms separate (concatenate_terms=False), "
                    f"so it has no single Box shape. Term shapes: {shape}."
                )
            obs_spaces[group_name] = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=shape,
                dtype=np.float32,
            )

        return obs_spaces

    # ========== Environment Setup (phased lifecycle) ==========

    def _setup_environment(self) -> None:
        """Initialize all managers in a structured, phased sequence.

        Subclasses implement the abstract hooks (_build_scene,
        _build_sim_managers) and optionally override _post_setup for
        customisation.  The ManagerRegistry resolves backend-specific
        classes (including reward manager) automatically via sim_type.
        Lifecycle events fire at well-defined points so external code
        can hook in.
        """
        # Register backend-specific managers in the registry
        from jaxrlworld.rl.envs.managers._registrations import register_all_for

        register_all_for(self.sim_type)

        # Phase 1 — Build physics scene (simulator-specific)
        self._build_scene()
        self.lifecycle.dispatch(LifecycleEvent.SCENE_BUILT)

        # Phase 1.5 — Simulator-specific work that must happen after the
        # physics scene exists but BEFORE any manager is constructed
        # (e.g. mjlab per-env model-field expansion, which replaces GPU
        # arrays and must precede anything that could hold references
        # to them).
        self._pre_manager_setup()

        # Phase 2 — Create managers
        self._build_sim_managers()
        self._build_common_managers()
        self.lifecycle.dispatch(LifecycleEvent.MANAGERS_READY)

        # Phase 3 — Simulator-specific finalization
        self._post_setup()

        # Phase 4 — Startup events
        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")

        self.lifecycle.dispatch(LifecycleEvent.ENV_READY)

        # Pretty print environment summary
        from jaxrlworld.rl.utils.pretty import print_env_summary

        print_env_summary(self)

        # Print joint mapping for debugging cross-simulator consistency
        # if hasattr(self, "act_manager") and hasattr(self.act_manager, "print_joint_mapping"):
        #     self.act_manager.print_joint_mapping()

    @abstractmethod
    def _build_scene(self) -> None:
        """Create the scene manager and build the physics world.

        This is the first setup phase.  After this method returns the
        scene manager must be assigned to ``self.scene_manager`` and the
        scene must be fully built (entities registered, simulation ready).
        """

    @abstractmethod
    def _build_sim_managers(self) -> None:
        """Create simulator-specific managers.

        At minimum this must set ``self.act_manager``,
        ``self.obs_manager``, and ``self.contact_manager``.
        Visualization managers are also created here.
        """

    def _build_common_managers(self) -> None:
        """Create simulator-agnostic managers (command, reward, termination, event).

        The reward manager class is resolved via ManagerRegistry so that
        backends like MuJoCo automatically get MujocoRewardManager without
        needing a subclass hook.
        """
        from jaxrlworld.rl.envs.managers import (
            CommandManager,
            CommandManagerConfig,
            EventManager,
            TerminationManager,
        )
        from jaxrlworld.rl.envs.managers.registry import ManagerRegistry

        self.command_manager = CommandManager(
            env=self,
            config=CommandManagerConfig(terms=self.command_cfg.terms),
        )

        reward_cls = ManagerRegistry.get_class(self.sim_type, "reward")
        self.reward_manager = reward_cls(
            env=self,
            config=self.reward_cfg,
        )

        self.termination_manager = TerminationManager(
            env=self,
            config=self.env_cfg.terminations,
            episode_length_s=self.env_cfg.episode_length_s,
        )

        self.event_manager = EventManager(
            env=self,
            config=self.event_cfg,
        )

        # Curriculum manager (must come after reward+termination managers
        # because curriculum terms resolve references into them at init).
        # ``self.curriculum_cfg`` is always a valid CurriculumManagerConfig
        # instance — the three ConfigsForRun dataclasses default it to an
        # empty config via ``_default_curriculum_cfg``, so presets that
        # don't register any curriculum terms still get a no-op manager.
        from jaxrlworld.rl.envs.managers.common.curriculum import CurriculumManager

        self.curriculum_manager = CurriculumManager(
            env=self,
            config=self.curriculum_cfg,
        )

    def _pre_manager_setup(self) -> None:
        """Simulator-specific work between scene build and manager creation.

        mjlab expands model fields for per-env domain randomization here:
        ``expand_model_fields`` replaces the GPU arrays behind the fields and
        re-captures the CUDA graphs, so it must run before any manager or
        entity can hold a reference to the old arrays (mirrors mjlab's own
        ``ManagerBasedRlEnv`` ordering: Sim init -> expand -> managers).
        """

    def _post_setup(self) -> None:
        """Simulator-specific finalization after all managers are created.

        Example: Newton captures CUDA graphs.  Called before startup events.
        """

    @abstractmethod
    def _step_physics(self) -> None:
        """Execute physics step(s). Implement in subclass."""
        pass

    # ========== Common Implementation ==========

    def get_observation(self):
        return self.obs_manager.get_observation()

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Execute one environment step."""

        # Process and apply actions
        self.act_manager.process_actions(actions)

        # Step physics (simulator-specific). Backends accumulate contact
        # timing inside their substep loop (``contact_manager.advance(physics_dt)``
        # per substep), matching IsaacLab/mjlab. No policy-step-end
        # advance call is needed here.
        self._step_physics()
        self._invalidate_cache()

        # Increment the episode-step counter AFTER physics and BEFORE
        # termination/reward, matching IsaacLab / mjlab which do
        # ``episode_length_buf += 1`` right after the decimation loop and
        # before termination_manager.compute. Doing it here — rather than at
        # the end of step() — makes the timeout check and every
        # episode_length_buf reader see a consistent count from the first
        # episode onward, instead of relying on just-reset envs picking up a
        # trailing end-of-step increment (which the initial reset() and eval
        # resets never get, leaving their first episode one step too long).
        self.termination_manager.advance()

        # Pre-termination hook
        self._pre_termination_hook()

        # Check termination BEFORE computing rewards so reward terms can read
        # the current step's termination state (e.g. is_terminated / is_alive),
        # matching IsaacLab / mjlab ordering (termination_manager is computed
        # before reward_manager). Termination terms depend only on physics
        # state — not on rewards or the gait phase advanced in
        # _pre_reward_hook — so this ordering preserves behaviour for all
        # existing terms.
        terminated, truncated = self.termination_manager.check_termination()
        reset_buf = terminated | truncated
        reset_env_ids = reset_buf.nonzero(as_tuple=False).flatten()

        # Pre-reward hook (e.g., gait advance that rewards depend on)
        self._pre_reward_hook()

        # Compute rewards
        self.rew_buf[:] = 0.0
        self.reward_manager.set_rewards(reward_buffer=self.rew_buf, reward_buffer_per_type=self.rew_buf_per_type)

        # Handle terminal observations.
        #
        # ``process_observations(update_history=True)`` appends the terminal
        # frame into every term's circular buffer so that history-based obs
        # in ``final_observation`` include it (matters for PPO truncated
        # bootstrap V(s_T) on history-based tasks). We immediately
        # ``rollback_last_history_append`` to rewind the write head so the
        # later per-step ``_advance_managers`` is the only append that
        # actually counts — otherwise the same step would advance history
        # by two frames. ``_reset_idx`` clears history for terminated envs
        # downstream, so the rollback only matters for the non-terminated
        # ones.
        final_observation = None
        if len(reset_env_ids) > 0:
            # The terminal observation describes the state that ENDED the
            # episode, so its rendered sensors have to be of that state —
            # taken here, before the reset writes the next episode's pose
            # over it. Only on steps that actually terminate something,
            # since this is a second render.
            groups = self.terminal_obs_groups
            self._render_sensors()
            self.obs_manager.process_observations(update_history=True, groups=groups)
            group_names = self.obs_manager.obs_dict.keys() if groups is None else groups
            final_observation = {key: self.obs_manager.obs_dict[key].clone() for key in group_names}
            self.obs_manager.rollback_last_history_append(groups=groups)

        # Reset terminated environments
        self._reset_idx(reset_env_ids)
        self._invalidate_cache()

        # Post-reset forward pass — refresh derived quantities (xpos,
        # xquat, site positions, sensor data, ...) so the upcoming
        # observation sees fresh kinematics. Override in backends that
        # need an explicit FK pass (mjlab: sim.forward()).
        self._post_reset_forward()

        # Advance commands (timer-based resampling + per-step post-processing)
        self.command_manager.compute(self.control_dt)

        # Apply interval events AFTER reset/command and BEFORE the observation
        # is built in _advance_managers, matching IsaacLab and mjlab (both apply
        # mode="interval" after the reward/termination/reset block, right before
        # observation_manager.compute). Interval disturbances (e.g.
        # push_by_setting_velocity) must NOT contaminate this step's reward or
        # termination — those reflect the policy's own action — but MUST be
        # visible to the returned observation so the policy can react next step.
        # The cache is invalidated afterwards so the obs build sees the new
        # state written by the events.
        if hasattr(self, "event_manager") and self.event_manager is not None:
            if "interval" in self.event_manager.available_modes:
                self.event_manager.apply(mode="interval", dt=self.control_dt)
                self._invalidate_cache()
            # interval_dr: global-period domain randomization (mass/friction/gains
            # re-sampled for all envs every interval_dr_period_s, one recompute).
            if "interval_dr" in self.event_manager.available_modes:
                self.event_manager.apply(mode="interval_dr", dt=self.control_dt)
                self._invalidate_cache()

        # Rendered sensors last, after the interval events above: mjlab's
        # own step ends forward -> command -> sense -> observation, and an
        # image taken before a push describes a robot that has since moved.
        self._render_sensors()

        # Advance managers
        self._advance_managers()

        self._update_num_step_calls()

        # Build extras
        self.extras = {
            "final_observation": final_observation,
            "terminal_env_ids": reset_env_ids if len(reset_env_ids) > 0 else None,
            "rewards_per_type": self.rew_buf_per_type,
            # Per-env mask of terminals whose value should be bootstrapped
            # (truncations + non-absorbing terminations). Consumed by the
            # on-policy bootstrap; equals ``truncated & ~terminated`` unless
            # a termination term sets ``bootstrap_value=True``.
            "bootstrap_mask": self.termination_manager.bootstrap_buf,
            **self.obs_manager.extras,
            **self.termination_manager.extras,
            **self.command_manager.extras,
        }
        result = (self.obs_manager.get_observation(), self.rew_buf, terminated, truncated, self.extras)

        return result

    def _pre_reward_hook(self) -> None:
        """Override in subclass for logic that must run before reward computation.

        Example: advancing gait manager so desired_contact_states are available
        for gait-tracking rewards.
        """
        pass

    def _pre_termination_hook(self) -> None:
        """Override in subclass for pre-termination logic."""
        pass

    def _reset_scene(self, env_ids: torch.Tensor) -> None:
        """Simulator-side reset for ``env_ids``, before any reset event runs.

        Called from :meth:`_reset_idx` immediately after
        ``curriculum_manager.compute`` and before the ``reset`` /
        ``reset_dr`` event modes.  That position is load-bearing:

        * **After** curriculum compute, so curriculum terms still observe the
          ending episode's terminal state.  Implementations must therefore not
          disturb what a curriculum term reads (joint / body state) *before*
          this point — the two backends that implement this hook either write
          only solver-internal buffers (Newton) or are themselves the reset
          the curriculum is deliberately ordered ahead of (MuJoCo).
        * **Before** the reset events, so the new pose is written on top of a
          clean simulator state and the post-reset
          ``contact_manager.refresh_after_reset`` pass sees no stale
          externally-applied force.

        Mirrors the ``sim.reset(env_ids)`` slot in mjlab's and IsaacLab's
        ``_reset_idx``.  Base implementation is a no-op — Genesis manages its
        state internally and has nothing to clear here.
        """
        return None

    def _render_sensors(self) -> None:
        """Run the backend's sensor rendering, just before an observation.

        Separate from :meth:`_post_reset_forward` on purpose. That hook
        refreshes derived kinematics; this one produces sensor readings
        that are themselves rendered — cameras, raycasts — and the two
        belong at different points in the step. mjlab runs its whole
        sense pipeline (BVH refit, camera rendering, raycasting) as the
        LAST thing before the observation, after the interval events
        that may have moved the robot; folding it into the FK hook put
        it before both, so a rendered image would describe a state the
        observation no longer reports.

        Each backend renders differently — mjlab's ``sim.sense()``,
        Newton's tiled-camera sensor after a BVH refit, Genesis's batch
        renderer — so the shared step only says WHEN, never how.

        A no-op by default: a backend with no rendered sensors has
        nothing to do here.
        """

    def _post_reset_forward(self) -> None:
        """Refresh derived quantities after resets, before observations.

        MuJoCo (mjlab) overrides this to call ``sim.forward()`` which
        recomputes all derived quantities (xpos, xquat, site positions,
        sensor data) from the current qpos/qvel. This ensures the
        observation computed in ``_advance_managers`` sees fresh
        kinematics for ALL environments — not just the ones that were
        reset.

        Newton and Genesis do not need this hook because their FK is
        either evaluated explicitly after writes (Newton ``eval_fk``)
        or handled internally by ``scene.step()`` (Genesis).
        """
        pass

    def _advance_managers(self) -> None:
        """Advance all managers. Override to add custom managers.

        Note: the episode-step counter (``termination_manager.advance``) is
        intentionally NOT advanced here — it is incremented at the start of
        :meth:`step` (after physics, before termination), matching IsaacLab.
        """
        self.obs_manager.advance()
        self.reward_manager.advance()
        self.act_manager.advance()

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return

        # Curriculum compute FIRST — before any reset write — so curriculum
        # terms can read the ending episode's terminal state (e.g. distance
        # walked from root_pos_w, episode_length_buf) and the episode's command
        # before the reset events / manager resets below overwrite them. Matches
        # IsaacLab and mjlab, which both call curriculum_manager.compute as the
        # first line of _reset_idx. The current step-stage curricula only read
        # env.env_step_counter (reset-independent), but this early placement is
        # what unlocks terminal-state-based curricula such as terrain levels.
        # curriculum.reset (stateful-term reset) is forwarded with the other
        # manager resets below.
        self.curriculum_manager.compute(env_ids=env_ids)

        # Simulator-side reset (backend hook). Runs after the curriculum has
        # read the terminal state and before any reset event writes the new
        # pose — see :meth:`_reset_scene` for why that ordering matters.
        self._reset_scene(env_ids)

        # State initialization via event manager
        self.event_manager.reset(env_ids)
        if "reset" in self.event_manager.available_modes:
            self.event_manager.apply(mode="reset", env_ids=env_ids)
        if "reset_dr" in self.event_manager.available_modes:
            self.event_manager.apply(mode="reset_dr", env_ids=env_ids)

        self.termination_manager.reset(env_ids)
        self.command_manager.reset(env_ids)
        self.act_manager.reset(env_ids)
        self.obs_manager.reset(env_ids)
        self.contact_manager.reset(env_ids)
        # All reset events have written the new states by now; recompute
        # contact state so post-reset reads see the NEW pose's contacts
        # (reference frameworks run sim.forward() at this point).
        self.contact_manager.refresh_after_reset(env_ids)
        self.reward_manager.reset(env_ids)
        # Forward reset to stateful curriculum terms (compute already ran at
        # the top of _reset_idx). Curriculum state is intentionally NOT
        # injected into ``rew_buf_per_type`` — its values (e.g.
        # ``energy_threshold`` in Watts) live on a different scale and
        # polluted the wandb "Rewards" breakdown; the runner logs the latest
        # state under a ``Curriculum/`` namespace in
        # :meth:`BaseRunner.log_training_data`.
        self.curriculum_manager.reset(env_ids)

    def reset(self, *, seed=None, options=None) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """Reset all environments."""
        all_env_ids = torch.arange(self.num_envs, device=self.device)
        self._reset_idx(all_env_ids)
        self._post_reset_forward()
        self.command_manager.compute(dt=0.0)
        self._invalidate_cache()
        self._render_sensors()
        self.obs_manager.advance()

        self.extras = {
            "time_outs": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            "terminal_observations": None,
            "terminal_env_ids": None,
            "rewards_per_type": self.rew_buf_per_type,
        }

        return self.obs_manager.get_observation(), self.extras

    def get_robot_state(self) -> torch.Tensor | None:
        """Get current robot state."""
        return self.obs_manager.get_robot_state()

    def get_observation_dims(self) -> dict[str, int]:
        """Get observation dimensions."""
        return self.obs_manager.calculate_obs_dim()

    def __str__(self) -> str:
        """Pretty print environment summary with all manager information."""
        from io import StringIO

        from rich.console import Console

        from jaxrlworld.rl.utils.pretty import create_env_panel

        output = StringIO()
        console = Console(file=output, force_terminal=True, width=100)

        # Environment header panel
        env_rows = [
            ("Simulator", self.sim_name),
            ("Seed", str(self.seed)),
            ("Num Envs", str(self.num_envs)),
            ("Device", str(self.device)),
            ("Physics dt", f"{self.physics_dt:.4f}s"),
            ("Control dt", f"{self.control_dt:.4f}s"),
        ]

        if hasattr(self, "decimation"):
            env_rows.append(("Decimation", str(self.decimation)))

        if hasattr(self, "task_name") and self.task_name:
            env_rows.append(("Task", str(self.task_name)))

        panel = create_env_panel(
            title=f"{self.sim_name} Environment",
            rows=env_rows,
            border_style="blue",
        )
        console.print(panel)
        console.print()

        # Print each manager
        managers = [
            ("obs_manager", "Observation Manager"),
            ("act_manager", "Action Manager"),
            ("reward_manager", "Reward Manager"),
            ("termination_manager", "Termination Manager"),
            ("contact_manager", "Contact Manager"),
            ("command_manager", "Command Manager"),
            ("event_manager", "Event Manager"),
        ]

        for attr_name, _ in managers:
            if hasattr(self, attr_name):
                manager = getattr(self, attr_name)
                if manager is not None:
                    try:
                        manager_str = str(manager)
                        if manager_str.strip():
                            console.print(manager_str)
                    except Exception:
                        pass

        return output.getvalue()
