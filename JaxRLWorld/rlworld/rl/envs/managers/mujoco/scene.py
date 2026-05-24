from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import torch

from rlworld.rl.configs.robots.kinematic_tree import KinematicTree
from rlworld.rl.configs.scene.terrain_config import TerrainCfg
from rlworld.rl.configs.scene.unified_entity_config import (
    EntityCfg,
    MujocoEntityCfg,
)
from rlworld.rl.configs.sensors import ContactSensorCfg
from rlworld.rl.envs.managers.base import BaseManager
from rlworld.rl.envs.managers.common.canonical_joint_order import filter_canonical_to_actuated
from rlworld.rl.envs.managers.common.scene_helpers import build_kinematic_trees
from rlworld.rl.envs.managers.common.visual_mesh import extract_visual_meshes_from_mj_model
from rlworld.rl.envs.managers.registry import ManagerRegistry

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.scene import Scene
    from mjlab.sim import Simulation

    from rlworld.rl.envs import World


def _canonical_joint_order_mujoco(mj_model) -> list[str]:
    """Canonical joint name list — DFS walk of the MjModel body tree with
    siblings sorted alphabetically by bare body name at each node, collecting
    each body's joints (sorted alphabetically when a body owns multiple) when
    visited.

    Returns bare joint names (entity prefix stripped); the world body (id 0)
    is skipped.
    """
    import mujoco

    n = mj_model.nbody
    raw_body_names = [mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, i) or "" for i in range(n)]

    def _bare(name: str) -> str:
        return name.rsplit("/", 1)[-1] if "/" in name else name

    bare_body_names = [_bare(b) for b in raw_body_names]
    children: dict[int, list[int]] = {}
    for i in range(n):
        p = int(mj_model.body_parentid[i])
        if p == i:  # world body: parent is itself (id 0).
            continue
        children.setdefault(p, []).append(i)
    for k in children:
        children[k].sort(key=lambda i: bare_body_names[i])
    roots = list(children.get(0, []))  # already sorted by the loop above.

    out: list[str] = []
    stack = list(reversed(roots))
    while stack:
        i = stack.pop()
        adr = int(mj_model.body_jntadr[i])
        num = int(mj_model.body_jntnum[i])
        if num > 0:
            joint_pairs = []
            for j in range(adr, adr + num):
                jn = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
                joint_pairs.append((j, _bare(jn)))
            joint_pairs.sort(key=lambda pair: pair[1])
            for _j_id, jn in joint_pairs:
                out.append(jn)
        kids = children.get(i, [])
        for kid in reversed(kids):
            stack.append(kid)
    return out


def _to_mjlab_sensor_cfg(cfg: Any) -> Any:
    """Convert a sim-agnostic ContactSensorCfg to mjlab's ContactSensorCfg.

    Other sensor-config types are returned unchanged (the mjlab scene
    only needs ContactSensorCfg conversion today). The ``mjlab`` import
    stays function-local: ``mujoco/scene.py`` imports mjlab lazily
    everywhere so the module loads without mjlab installed.
    """
    if not isinstance(cfg, ContactSensorCfg):
        return cfg
    from mjlab.sensor import ContactMatch as MjContactMatch, ContactSensorCfg as MjContactSensorCfg

    def _match(m):
        return None if m is None else MjContactMatch(mode=m.mode, pattern=m.pattern, entity=m.entity, exclude=m.exclude)

    return MjContactSensorCfg(
        name=cfg.name,
        primary=_match(cfg.primary),
        secondary=_match(cfg.secondary),
        fields=cfg.fields,
        reduce=cfg.reduce,
        num_slots=cfg.num_slots,
        global_frame=cfg.global_frame,
        history_length=cfg.history_length,
    )


@dataclass
class MujocoSceneManagerConfig:
    """Internal config consumed by MujocoSceneManager.

    Populated from MujocoSceneConfig by mjlab_env at build time.
    Users should not construct this directly — use MujocoSceneConfig
    in presets instead.
    """

    device: str = "cuda:0"
    robot_entity_name: str = "robot"
    num_envs: int = 4096
    env_spacing: float = 2.0
    physics_dt: float = 0.002
    substeps: int = 1

    # Entities — unified EntityCfg dict (auto-converted to mjlab)
    entities: dict[str, EntityCfg] | None = None

    # Sensors — sim-agnostic rlworld.rl.configs.sensors.ContactSensorCfg
    # objects, converted to mjlab sensor configs in
    # MujocoSceneManager.build_scene.
    sensors: tuple[ContactSensorCfg, ...] = ()

    # Terrain (flat plane by default; generator → injected heightfield).
    terrain_cfg: TerrainCfg = field(default_factory=lambda: TerrainCfg(terrain_type="plane"))

    # Solver
    solver_iterations: int = 10
    solver_ls_iterations: int = 20
    ccd_iterations: int = 50
    nconmax: int | None = 35
    njmax: int | None = 1500
    impratio: float = 1.0
    cone: Literal["pyramidal", "elliptic"] = "pyramidal"
    contact_sensor_maxmatch: int = 64

    # Legacy — set by mjlab_env for backward compat
    mjlab_scene_cfg: Any = None
    mjlab_sim_cfg: Any = None
    unified_entities: Any = None


class MujocoSceneManager(BaseManager):
    """Manages mjlab Scene and Simulation lifecycle.

    This manager wraps mjlab's Scene and Simulation classes to provide
    a rlworld-compatible interface for MuJoCo-based environments.

    The scene is built in the following order:
    1. Create Scene from SceneCfg (builds MjSpec)
    2. Compile Scene to get MjModel
    3. Create Simulation with MjModel
    4. Initialize Scene with Simulation data

    Example:
        scene_manager = MujocoSceneManager(env, config)
        scene_manager.build_scene()

        # In simulation loop:
        scene_manager.step()
    """

    def __init__(self, env: World, config: MujocoSceneManagerConfig):
        super().__init__(env)
        self.config = config

        # mjlab objects (initialized in build_scene)
        self._scene: Scene = None
        self._sim: Simulation = None

        # Kinematic trees for each entity
        self.trees: dict[str, KinematicTree] = {}

        # Timing
        self._physics_dt: float | None = None  # updated from sim config

        # Terrain importer (owns terrain data + per-env origins / curriculum).
        self.terrain = ManagerRegistry.create(
            "mujoco",
            "terrain",
            cfg=self.config.terrain_cfg,
            num_envs=self.config.num_envs,
            device=self.config.device,
        )

    @property
    def scene(self) -> Scene:
        """The mjlab Scene instance."""
        return self._scene

    @property
    def sim(self) -> Simulation:
        """The mjlab Simulation instance."""
        return self._sim

    @property
    def model(self):
        """The mujoco-warp Model."""
        if self._sim is None:
            return None
        return self._sim.model

    @property
    def data(self):
        """The mujoco-warp Data."""
        if self._sim is None:
            return None
        return self._sim.data

    @property
    def mj_model(self):
        """The MuJoCo MjModel."""
        if self._sim is None:
            return None
        return self._sim.mj_model

    def get_visual_meshes(self, body_names: tuple[str, ...]):
        """Per-body visual ``trimesh.Trimesh`` in body-local frame for the
        viser ghost overlay (and any other consumer needing static visual
        geometry). mjlab path: ``Scene.compile()`` returns a fresh single-
        robot ``MjModel`` reflecting any per-env ``spec_fn`` rewrites; we
        hand it to the shared MjModel-based extractor."""
        return extract_visual_meshes_from_mj_model(self._scene.compile(), body_names)

    @property
    def physics_dt(self) -> float:
        """Physics timestep."""
        return self._physics_dt

    @property
    def robot(self) -> Entity:
        """Get the main robot entity."""
        return self._scene[self.config.robot_entity_name]

    @property
    def entities(self) -> dict[str, Entity]:
        """Get all entities."""
        return self._scene.entities

    @property
    def sensors(self) -> dict[str, Any]:
        """Get all sensors."""
        return self._scene.sensors

    @property
    def env_origins(self) -> torch.Tensor:
        """Per-env world-frame spawn offsets ``(num_envs, 3)``.

        Generator terrain: comes from the ``TerrainImporter`` sub-terrain
        grid. Plane terrain: comes from mjlab's ``Scene.env_origins``
        (its ``env_spacing`` grid), so we keep mjlab's native multi-env
        layout for flat scenes.
        """
        if self.terrain.data is not None:
            return self.terrain.env_origins
        return self._scene.env_origins

    def build_scene(self) -> None:
        """Build the scene and simulation from config.

        Constructs mjlab SceneCfg and SimulationCfg internally from the
        rlworld config fields.  No mjlab imports needed at config level.
        """
        from mjlab.scene import Scene, SceneCfg
        from mjlab.sim import MujocoCfg, Simulation, SimulationCfg
        from mjlab.terrains import TerrainEntityCfg

        # Terrain importer (constructed in __init__) decides between
        # mjlab's built-in plane and a spec_fn that injects our generated
        # heightfield in a "terrain" body. ``self.terrain.data`` is then
        # available for the out-of-bounds termination + viser bridge.
        terrain_spec_fn = self.terrain.build_spec_fn()

        # Build mjlab SceneCfg
        if self.config.mjlab_scene_cfg is not None:
            # Legacy path — user provided mjlab SceneCfg directly
            scene_cfg = self.config.mjlab_scene_cfg
        else:
            scene_cfg = SceneCfg(
                num_envs=self.config.num_envs,
                env_spacing=self.config.env_spacing,
                terrain=None if terrain_spec_fn is not None else TerrainEntityCfg(terrain_type="plane"),
                entities={},
                sensors=tuple(_to_mjlab_sensor_cfg(s) for s in self.config.sensors),
                spec_fn=terrain_spec_fn,
            )
            self.config.mjlab_scene_cfg = scene_cfg

        # Convert unified entities → mjlab entities and merge
        if self.config.entities is not None:
            self._build_mjlab_entities()

        # Create scene
        self._scene = Scene(scene_cfg, device=self.config.device)

        # Compile to get MjModel
        mj_model = self._scene.compile()

        # Build SimulationCfg
        if self.config.mjlab_sim_cfg is not None:
            sim_cfg = self.config.mjlab_sim_cfg
        else:
            substep_dt = self.config.physics_dt / self.config.substeps
            sim_cfg = SimulationCfg(
                nconmax=self.config.nconmax,
                njmax=self.config.njmax,
                mujoco=MujocoCfg(
                    timestep=substep_dt,
                    iterations=self.config.solver_iterations,
                    ls_iterations=self.config.solver_ls_iterations,
                    ccd_iterations=self.config.ccd_iterations,
                    impratio=self.config.impratio,
                    cone=self.config.cone,
                    disableflags=("nativeccd",),
                ),
                contact_sensor_maxmatch=self.config.contact_sensor_maxmatch,
            )

        # Update physics dt (the per-substep dt used by MuJoCo internally)
        self._physics_dt = sim_cfg.mujoco.timestep

        # Create simulation
        self._sim = Simulation(
            num_envs=self._scene.num_envs,
            cfg=sim_cfg,
            model=mj_model,
            device=self.config.device,
        )

        # Initialize scene with simulation data
        self._scene.initialize(
            mj_model=self._sim.mj_model,
            model=self._sim.model,
            data=self._sim.data,
        )

        # Build kinematic trees for each entity
        self._set_kinematic_tree()

    def _build_mjlab_entities(self) -> None:
        """Convert unified entities to mjlab EntityCfg and merge into SceneCfg.

        Actuator type mapping (automatic):
          ImplicitActuatorCfg → BuiltinPositionActuatorCfg (simulator PD)
          Any other type      → BuiltinMotorActuatorCfg (direct torque)
        """
        from mjlab.actuator.builtin_actuator import (
            BuiltinMotorActuatorCfg,
            BuiltinPositionActuatorCfg,
        )
        from mjlab.entity import EntityArticulationInfoCfg, EntityCfg as MjlabEntityCfg

        from rlworld.rl.actuators.actuator_cfg import ImplicitActuatorCfg

        for entity_name, cfg in self.config.entities.items():
            if not isinstance(cfg, MujocoEntityCfg):
                raise ValueError(f"MuJoCo entity '{entity_name}' must be MujocoEntityCfg, got {type(cfg).__name__}")
            if cfg.spec_fn is None:
                raise ValueError(f"MuJoCo entity '{entity_name}' requires 'spec_fn'")

            # Convert actuator configs → mjlab actuator configs.
            # When stiffness/damping/armature/effort_limit are dicts, we expand
            # into one mjlab actuator per regex key so each gets the correct
            # per-pattern value. mjlab's Builtin*ActuatorCfg only accepts scalar
            # effort_limit, so dict expansion here is mandatory.
            def _pick(value, pattern, fallback):
                """Resolve scalar-or-dict to a single scalar for one pattern."""
                if isinstance(value, dict):
                    return value.get(pattern, fallback)
                if isinstance(value, int | float):
                    return value
                return fallback

            mjlab_actuators = []
            for act_cfg in cfg.articulation.actuators:
                if isinstance(act_cfg, ImplicitActuatorCfg):
                    if isinstance(act_cfg.stiffness, dict):
                        # Expand: one BuiltinPositionActuator per gain key
                        for pattern in act_cfg.stiffness.keys():
                            mjlab_actuators.append(
                                BuiltinPositionActuatorCfg(
                                    target_names_expr=(pattern,),
                                    stiffness=_pick(act_cfg.stiffness, pattern, 0.0),
                                    damping=_pick(act_cfg.damping, pattern, 0.0),
                                    effort_limit=_pick(act_cfg.effort_limit, pattern, None),
                                    armature=_pick(act_cfg.armature, pattern, 0.0),
                                    frictionloss=act_cfg.frictionloss,
                                )
                            )
                    else:
                        stiffness = act_cfg.stiffness if isinstance(act_cfg.stiffness, int | float) else 0.0
                        damping = act_cfg.damping if isinstance(act_cfg.damping, int | float) else 0.0
                        armature = act_cfg.armature if isinstance(act_cfg.armature, int | float) else 0.0
                        effort_limit = act_cfg.effort_limit if isinstance(act_cfg.effort_limit, int | float) else None
                        mjlab_actuators.append(
                            BuiltinPositionActuatorCfg(
                                target_names_expr=act_cfg.target_names_expr,
                                stiffness=stiffness,
                                damping=damping,
                                effort_limit=effort_limit,
                                armature=armature,
                                frictionloss=act_cfg.frictionloss,
                            )
                        )
                else:
                    # Explicit actuator (IdealPD, LSTM, etc.) → motor mode
                    if isinstance(act_cfg.armature, dict):
                        # Expand: one BuiltinMotorActuator per armature key
                        for pattern, arm in act_cfg.armature.items():
                            mjlab_actuators.append(
                                BuiltinMotorActuatorCfg(
                                    target_names_expr=(pattern,),
                                    effort_limit=_pick(act_cfg.effort_limit, pattern, 1000.0),
                                    armature=arm,
                                    frictionloss=act_cfg.frictionloss,
                                )
                            )
                    else:
                        armature = act_cfg.armature if isinstance(act_cfg.armature, int | float) else 0.0
                        effort_limit = act_cfg.effort_limit if isinstance(act_cfg.effort_limit, int | float) else 1000.0
                        mjlab_actuators.append(
                            BuiltinMotorActuatorCfg(
                                target_names_expr=act_cfg.target_names_expr,
                                effort_limit=effort_limit,
                                armature=armature,
                                frictionloss=act_cfg.frictionloss,
                            )
                        )

            articulation_info = (
                EntityArticulationInfoCfg(
                    actuators=tuple(mjlab_actuators),
                    soft_joint_pos_limit_factor=cfg.articulation.soft_joint_pos_limit_factor,
                )
                if mjlab_actuators
                else None
            )

            # Build mjlab EntityCfg
            init_state = MjlabEntityCfg.InitialStateCfg(
                pos=cfg.init_state.pos,
                rot=cfg.init_state.rot,
                joint_pos=cfg.init_state.joint_pos or None,
                joint_vel=cfg.init_state.joint_vel,
            )
            spec_fn = cfg.spec_fn
            if isinstance(spec_fn, str):
                from rlworld.rl.utils.resolve import resolve_callable

                spec_fn = resolve_callable(spec_fn)
            mjlab_cfg = MjlabEntityCfg(
                init_state=init_state,
                spec_fn=spec_fn,
                articulation=articulation_info,
                collisions=cfg.collisions,
                sort_actuators=True,
            )

            self.config.mjlab_scene_cfg.entities[entity_name] = mjlab_cfg

    def build_articulation_indexing(
        self,
        actuated_dof_names: list[str],
        entity_name: str = "robot",
    ):
        """Build ArticulationIndexing for the given entity in canonical joint order.

        Joint order is computed by a DFS walk of the kinematic body tree
        (siblings sorted alphabetically by bare body name at each level),
        emitting each body's inbound joint(s) when visited. This pins the
        order to the kinematic structure + names so it agrees with Genesis
        and Newton for the same robot, regardless of how each backend's
        parser flattened the tree internally. ``actuated_dof_names`` then
        filters that canonical list in canonical (not query) order.

        Args:
            actuated_dof_names: Regex patterns for actuated joints.
            entity_name: Which entity to index.

        Returns:
            ArticulationIndexing with canonical ↔ simulator mappings.
        """
        from rlworld.rl.envs.indexing import ArticulationIndexing

        entity = self._scene[entity_name]
        all_entity_joint_names = list(entity.joint_names)
        canonical_full = _canonical_joint_order_mujoco(self._sim.mj_model)
        # Restrict canonical list to joints the entity actually exposes.
        entity_joint_set = set(all_entity_joint_names)
        canonical_actuatable = [n for n in canonical_full if n in entity_joint_set]

        if actuated_dof_names:
            matched_names, _ = filter_canonical_to_actuated(canonical_actuatable, actuated_dof_names)
        else:
            matched_names = list(canonical_actuatable)

        # An entity with zero actuated joints (e.g. a free-flying drone)
        # is a legitimate case — return an empty ArticulationIndexing
        # so the action manager can still operate via term-based actions
        # (PropellerThrustAction, etc.) that bypass the joint-PD path.
        if not matched_names:
            empty_long = torch.zeros(0, device=self.config.device, dtype=torch.long)
            empty_float = torch.zeros(0, device=self.config.device, dtype=torch.float32)
            return ArticulationIndexing(
                joint_names=(),
                sim_indices=empty_long,
                sim_to_canonical=empty_long.clone(),
                joint_limits_lower=empty_float,
                joint_limits_upper=empty_float.clone(),
            )

        # Look up each matched name's index within the entity's joint list.
        indices = [all_entity_joint_names.index(n) for n in matched_names]

        sim_indices = torch.tensor(indices, device=self.config.device, dtype=torch.long)
        num_joints = len(indices)
        sim_to_canonical = torch.zeros(num_joints, device=self.config.device, dtype=torch.long)
        for canonical_i, _sim_i in enumerate(sim_indices):
            sim_to_canonical[canonical_i] = canonical_i  # identity: RobotData indexes via sim_indices.

        soft_limits = entity.data.soft_joint_pos_limits
        lower = soft_limits[0, sim_indices, 0]
        upper = soft_limits[0, sim_indices, 1]

        return ArticulationIndexing(
            joint_names=tuple(matched_names),
            sim_indices=sim_indices,
            sim_to_canonical=sim_to_canonical,
            joint_limits_lower=lower,
            joint_limits_upper=upper,
        )

    def step(self) -> None:
        """Advance physics by one control sub-interval.

        When ``substeps > 1``, executes multiple ``sim.step()`` calls
        at ``physics_dt / substeps`` each — matching Newton's internal
        substep loop. The outer ``_step_physics`` decimation loop
        calls this method ``decimation`` times per action, so the total
        physics time per action is ``physics_dt × decimation``.
        """
        for _ in range(self.config.substeps):
            self._sim.step()

    def forward(self) -> None:
        """Compute forward kinematics."""
        self._sim.forward()

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset environments to initial state."""
        self._sim.reset(env_ids)
        self._scene.reset(env_ids)

    def update(self, dt: float) -> None:
        """Update scene (entities and sensors) after physics step."""
        self._scene.update(dt)

    def write_data_to_sim(self) -> None:
        """Write entity data to simulation."""
        self._scene.write_data_to_sim()

    def get_entity(self, entity_name: str) -> Entity:
        """Get an entity by name."""
        return self._scene[entity_name]

    def get_sensor(self, sensor_name: str) -> Any:
        """Get a sensor by name."""
        return self._scene.sensors.get(sensor_name)

    def find_body_names(self, body_names: list[str], entity_name: str = "robot") -> list[str]:
        """Find body names matching patterns (for GaitManager compatibility).

        Args:
            body_names: List of body name patterns to match.
            entity_name: Name of the entity to search in.

        Returns:
            List of matched body names.
        """
        entity = self.get_entity(entity_name)
        # Use mjlab's find_bodies API
        _, matched_names = entity.find_bodies(body_names, preserve_order=True)
        return matched_names

    def _set_kinematic_tree(self) -> None:
        """Build kinematic trees for all entities from their MjSpec XML.

        Once the scene has been compiled, each entity's ``entity.spec``
        is attached by reference to the parent world spec, so calling
        ``to_xml()`` directly raises "cannot compile child spec if
        attached by reference to a parent spec". Falling back to
        ``spec.copy().to_xml()`` works around the compile guard but
        returns an empty ``<worldbody/>`` (the body hierarchy lives in
        the parent spec, not the copy).

        Instead we re-invoke the original ``spec_fn`` callable that
        was used to build the entity in the first place — for the
        standard ``EntityCfg`` path this is e.g.
        ``mujoco.MjSpec.from_file(...)``, which always returns a fresh
        standalone spec with the full body tree intact. Entities that
        don't carry a ``spec_fn`` (e.g. terrain primitives built
        in-place by mjlab) are skipped.
        """

        def _resolve(name: str):
            entity = self._scene.entities[name]
            spec_fn = getattr(getattr(entity, "cfg", None), "spec_fn", None)
            if spec_fn is None:
                return None
            try:
                fresh_spec = spec_fn()
                xml_content = fresh_spec.to_xml()
            except Exception:
                return None
            return ("mjcf_xml", xml_content)

        self.trees = build_kinematic_trees(self._scene.entities.keys(), _resolve)
