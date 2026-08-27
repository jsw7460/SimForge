from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import genesis as gs
import mujoco
import torch
import trimesh

from rlworld.rl.actuators.actuator_cfg import ImplicitActuatorCfg
from rlworld.rl.configs.scene.terrain_config import TerrainCfg
from rlworld.rl.configs.scene.unified_entity_config import (
    EntityCfg,
    GenesisEntityCfg,
)
from rlworld.rl.configs.sensors import SensorConfig
from rlworld.rl.envs.indexing import ArticulationIndexing
from rlworld.rl.envs.managers.base import BaseManager
from rlworld.rl.envs.managers.common.canonical_joint_order import filter_canonical_to_actuated
from rlworld.rl.envs.managers.common.scene_helpers import build_kinematic_trees
from rlworld.rl.envs.managers.common.visual_mesh_transform import apply_local_transform
from rlworld.rl.envs.managers.registry import ManagerRegistry
from rlworld.rl.utils import entity_utils, string as string_utils

if TYPE_CHECKING:
    # ``genesis.engine.entities`` / ``genesis.engine.sensors`` evaluate
    # ``gs.qd_float`` at module load (i.e. need ``genesis.init()``), so they
    # are imported for type hints only — not on the runtime import path.
    from genesis.engine.entities import RigidEntity
    from genesis.engine.sensors.base_sensor import Sensor

    from rlworld.rl.envs import World


def _canonical_joint_order_genesis(entity: RigidEntity) -> list[str]:
    """Canonical joint name list — DFS walk of ``entity.links`` with
    siblings sorted alphabetically by bare link name at each node,
    collecting each link's inbound joint(s) when visited.

    Genesis enumerates bodies in BFS-by-depth internally, which gave a
    different action order than Newton/mjlab for humanoids that mix arm
    and leg chains at the same depth. This DFS-with-sorted-siblings walk
    pins the canonical order solely to the kinematic structure + names,
    independent of any sim's parser order.
    """
    links = list(entity.links)
    if not links:
        return []
    by_idx = {link.idx: link for link in links}
    roots: list = []
    children: dict[int, list] = {}
    for link in links:
        p = link.parent_idx
        if p == -1 or p not in by_idx:
            roots.append(link)
        else:
            children.setdefault(p, []).append(link)
    # Sort siblings alphabetically at every level (roots included).
    roots.sort(key=lambda lk: lk.name)
    for k in children:
        children[k].sort(key=lambda lk: lk.name)

    out: list[str] = []
    stack = list(reversed(roots))
    while stack:
        link = stack.pop()
        # ``link.joints`` are the joints connecting this link to its parent.
        # Sort by name for determinism when a body has multiple inbound joints
        # (rare — typically zero or one).
        for joint in sorted(link.joints, key=lambda j: j.name):
            # A free base is NOT one of an articulation's joints: its state
            # is the root pose, read through ``root_link_pos_w`` /
            # ``root_link_quat_w``. mjlab already leaves it out of an
            # entity's joint list, and leaving it in here makes the same
            # floating entity report a different joint count and a
            # different ``joint_pos`` width on this backend than on that
            # one. It stayed hidden while every floating robot named its
            # actuated joints explicitly, which filtered the base out
            # downstream; it appears the moment an entity is indexed
            # whole, which is what a passive mechanism with no actuators
            # is.
            if joint.n_dofs > 0 and joint.type != gs.JOINT_TYPE.FREE:
                out.append(joint.name)
        kids = children.get(link.idx, [])
        for kid in reversed(kids):
            stack.append(kid)
    return out


@dataclass
class SceneManagerConfig:
    """Configuration for scene creation"""

    sim_options: gs.options.SimOptions
    viewer_options: gs.options.ViewerOptions
    vis_options: gs.options.VisOptions
    rigid_options: gs.options.RigidOptions
    entities: dict[str, EntityCfg]
    sensors: list[SensorConfig] | None
    env_spacing: tuple
    show_viewer: bool
    num_envs: int = 1
    device: str = "cpu"
    # Simulator-agnostic cameras (CameraSensorCfg); each becomes a
    # ``GenesisCameraSensor`` drawn by Madrona's batch renderer.
    cameras: tuple = ()
    # Passive rigid objects (no actuated joints) — graspable objects, props,
    # static fixtures. Loaded into the separate ``self.rigid_objects`` registry.
    rigid_objects: dict = field(default_factory=dict)
    # Terrain (flat plane by default; generator → heightfield) — fed to a
    # GenesisTerrainImporter constructed via ManagerRegistry.
    terrain_cfg: TerrainCfg = field(default_factory=lambda: TerrainCfg(terrain_type="plane"))


class SceneManager(BaseManager):
    """Manages scene creation and configuration"""

    def __init__(self, env: World, config: SceneManagerConfig):
        BaseManager.__init__(self, env=env)
        self.config = config
        self.scene = None
        self.entities: dict[str, RigidEntity] = defaultdict()
        self.sensors: dict[str, dict[str, dict[str, Sensor]]] = defaultdict(lambda: defaultdict(dict))
        # CameraSensorCfg-backed cameras, flat by name. Kept apart from
        # ``sensors`` above, which Genesis nests by entity and link.
        self._cameras: dict[str, object] = {}

        self.trees: dict = {}

        # Terrain importer (owns terrain data + per-env origins / curriculum).
        self.terrain = ManagerRegistry.create(
            "genesis",
            "terrain",
            cfg=self.config.terrain_cfg,
            num_envs=self.config.num_envs,
            device=self.config.device,
        )

    def __getattr__(self, item) -> RigidEntity:
        return self.entities[item]

    def __getitem__(self, item) -> RigidEntity:
        # Articulations and passive rigid objects share the same Genesis entity
        # type; resolve across both registries so selectors (e.g. a reset event
        # targeting a graspable object) can name a rigid object too.
        if item in self.entities:
            return self.entities[item]
        return self.rigid_objects[item]

    def find_body_names(self, body_names: list[str], entity_name: str = "robot"):
        _, names = entity_utils.find_links(self.entities[entity_name], body_names, preserve_order=True)
        return names

    @property
    def env_origins(self) -> torch.Tensor:
        """Per-env world-frame spawn offsets ``(num_envs, 3)``.

        Sourced from the ``TerrainImporter`` sub-terrain grid; all-zeros
        when terrain is a flat plane.
        """
        return self.terrain.env_origins

    def get_visual_meshes(self, body_names: tuple[str, ...]) -> dict[str, trimesh.Trimesh | None]:
        """Per-body visual ``trimesh.Trimesh`` in body-local frame for the
        viser ghost overlay. Genesis path: harvest ``RigidVisGeom``s
        directly off each link — no mujoco round-trip, so the training
        model and the ghost mesh agree by construction. A body with
        zero vgeoms returns ``None``; an unresolvable name crashes via
        Genesis's own ``get_link``."""
        robot = self.entities["robot"]
        out: dict[str, trimesh.Trimesh | None] = {}
        for bname in body_names:
            link = robot.get_link(name=bname)
            parts: list[trimesh.Trimesh] = []
            for vg in link.vgeoms:
                mesh = trimesh.Trimesh(
                    vertices=vg.init_vverts.copy(),
                    faces=vg.init_vfaces.copy(),
                    process=False,
                )
                apply_local_transform(mesh, vg.init_pos, vg.init_quat)
                parts.append(mesh)
            if not parts:
                out[bname] = None
            elif len(parts) == 1:
                out[bname] = parts[0]
            else:
                out[bname] = trimesh.util.concatenate(parts)
        return out

    def register_entities(self) -> None:
        """Build complete scene with all components"""
        self._create_scene()
        self._add_entities()
        self._add_sensors()
        self._set_kinematic_tree()
        self._add_cameras()
        self.env.vis_manager._setup_visualization_cameras()

    def _add_entities(self):
        """Add articulated entities (``config.entities``) and passive rigid
        objects (``config.rigid_objects``).

        Terrain is added separately via the TerrainImporter first. Both kinds
        load through the same Genesis morph / ``add_entity`` path (a rigid
        object is just a zero-actuator entity), but are kept in separate
        registries — ``self.entities`` (articulations) vs ``self.rigid_objects``
        — mirroring IsaacLab's scene.articulations / scene.rigid_objects.
        """
        # Terrain (flat plane or generated heightfield) — added once,
        # before any other entities. Stored as ``self.terrain`` (importer).
        self.terrain.add_to_scene(self.scene)

        self.rigid_objects = {}
        for entity_name, cfg in self.config.entities.items():
            self.entities[entity_name] = self._load_entity(entity_name, cfg)
        for object_name, cfg in self.config.rigid_objects.items():
            self.rigid_objects[object_name] = self._load_entity(object_name, cfg)

    def _load_entity(self, name: str, cfg):
        """Load one Genesis ``RigidEntity`` from a unified EntityCfg / RigidObjectCfg.

        Shared by articulations and rigid objects; the only difference is which
        registry the caller stores the result in. ``fixed=not cfg.floating``
        gives a static body (e.g. a table) or a free body (a robot or a
        graspable object).
        """
        if name in self.entities or name in self.rigid_objects:
            raise ValueError(f"Entity '{name}' is already registered")

        # ``convexify`` comes from the shared base, because it decides
        # whether a link's collision geoms stay separate or get merged into
        # one -- a question a prop has as much as a robot. Forcing it False
        # for anything that was not a GenesisEntityCfg meant a two-geom prop
        # would have been merged with no way to say otherwise.
        convexify = cfg.convexify

        # These two are Genesis's alone: a render surface and a contact
        # overlay have no meaning to the other backends.
        if isinstance(cfg, GenesisEntityCfg):
            surface = cfg.surface
            visualize = cfg.visualize_contact
        else:
            surface = None
            visualize = False

        mjcf_path = getattr(cfg, "mjcf_path", None)
        if mjcf_path:
            mjcf_kwargs = {
                "file": mjcf_path,
                "convexify": convexify,
                "batch_fixed_verts": True,
                "requires_jac_and_IK": False,
                # Genesis defaults this to 0.1 kg*m^2 and injects it into
                # every joint whose armature the file does not state
                # (``genesis/utils/mjcf.py:182``). MuJoCo and Newton use the
                # MJCF default of zero, so leaving it on makes Genesis alone
                # simulate a different robot — silently, since the file is
                # unchanged and the number never appears anywhere. On a light
                # joint it is not a correction but a takeover: a spring-loaded
                # tong hinge carries 1.44e-4, so 0.1 is roughly seven hundred
                # times its real inertia. None means "believe the file", which
                # is what the other two backends do. Joints that genuinely
                # need armature get it from the robot config's ``armature``
                # dict in ``_configure_robot_dynamics`` below, which runs
                # after this and is the single place it is stated.
                "default_armature": None,
            }
            # Keep morph offset at origin so the new relative=True default of
            # get_pos/set_pos/get_quat/set_quat (Genesis #2934) collapses to the
            # absolute frame.  init_state.pos is applied by the reset events instead.
            morph = gs.morphs.MJCF(**mjcf_kwargs)
        else:
            urdf_kwargs = {
                "file": cfg.urdf_path,
                "fixed": not cfg.floating,
                "convexify": convexify,
                # Required before a fixed link carrying geometry can hold a
                # different pose per environment: without it Genesis keeps one
                # shared vertex buffer for such links and refuses the write
                # ("Specifying env-specific pos for fixed links with at least
                # one geometry requires setting morph option
                # 'batch_fixed_verts=True'"). That is exactly what a reset
                # event does to a table or tank, and the MJCF branch above
                # already sets it.
                "batch_fixed_verts": True,
                # Same silent-armature injection as the MJCF branch above.
                "default_armature": None,
            }
            if cfg.links_to_keep:
                urdf_kwargs["links_to_keep"] = cfg.links_to_keep
            morph = gs.morphs.URDF(**urdf_kwargs)

        return self.scene.add_entity(
            morph=morph,
            surface=surface,
            visualize_contact=visualize,
        )

    def _add_sensors(self):
        sensor_configs = self.config.sensors

        if not sensor_configs:
            return

        for sensor_config in sensor_configs:
            entity_name = sensor_config.entity_name
            link_name = sensor_config.link_name
            if entity_name not in self.entities:
                print(f"Entity {entity_name} not found for sensor. Skipping.")
                continue

            entity = self.entities[entity_name]
            sensor = sensor_config.create_sensor(scene=self.scene, entity=entity)

            sensor_class_name = sensor.__class__.__name__
            self.sensors[entity_name][link_name][sensor_class_name] = sensor

    def _add_cameras(self) -> None:
        """Attach the sim-agnostic cameras, before the scene is built."""
        from rlworld.rl.envs.managers.genesis.camera_sensor import GenesisCameraSensor

        for camera_cfg in self.config.cameras:
            if camera_cfg.name in self._cameras:
                raise ValueError(f"Camera '{camera_cfg.name}' already exists")
            entity_cfg = self.config.entities[camera_cfg.entity_name]
            link_name, offset, optics = camera_cfg.resolve(entity_cfg.mjcf_path)
            self._cameras[camera_cfg.name] = GenesisCameraSensor(
                env=self.env,
                cfg=camera_cfg,
                link_name=link_name,
                offset=offset,
                optics=optics,
            )

    def render_cameras(self) -> None:
        """Draw every camera against the current state."""
        for camera in self._cameras.values():
            camera.render()

    @property
    def camera_sensors(self) -> dict:
        """The sim-agnostic cameras, by name."""
        return self._cameras

    LOCAL_COLLISION_MASK_FLAG = "_is_local_collision_mask"
    """The private field on a Genesis entity that gates cross-entity masks.

    Private, so it is checked before it is written: assigning an attribute
    Python does not already know simply creates a new one, and a rename
    upstream would leave this silently doing nothing at all -- which is the
    failure mode this repo keeps paying for.
    """

    def _share_one_collision_mask_namespace(self) -> None:
        """Let contype / conaffinity mean the same thing across entities.

        Genesis marks an entity loaded from MJCF or USD as carrying LOCAL
        collision masks (``rigid_entity.py``) and then declines to compare
        those bits against another entity's::

            con_skip = (same_entity | ~has_local_mask) & (con_match == 0)

        That is right in general -- bit 4 in one file need not mean bit 4 in
        another -- and wrong for how these scenes are built. mjlab attaches
        the robot's spec INTO the scene's and compiles ONE model, so one
        vocabulary covers robot, props and ground alike, and every mask an
        asset authors is honoured across all of them. Genesis loads each as
        its own entity, so the same masks stop at the entity boundary.

        Measured on K1: its foot carries a shell box masked 4/4 against a
        1/1 ground -- forbidden, and mjlab forbids it -- while Genesis let
        the box carry the robot for 30 of the foot's contacts. The foot is
        four spheres on one backend and a flat sole on the other.

        Cleared on every entity, not just the robot, because that is what
        matching mjlab means: one namespace for the scene. Clearing only the
        robot would fix robot-against-ground and leave robot-against-prop
        still diverging -- and a gripper closing on a tool is exactly that
        pair.

        THE INVARIANT THIS ASSUMES, for whoever adds the next asset: within
        one scene, a non-default contype / conaffinity BIT means the same
        thing in every asset. K1 spends bit 4 on "foot shell"; a second
        asset spending bit 4 on something else would make the two interact
        by accident.

        That exposure is not new and not ours to avoid: mjlab compiles the
        whole scene into one model, so it already reads every asset's bits
        in one vocabulary and would mis-collide the same pair. Setting this
        False makes Genesis exactly as exposed as the backend we are
        matching, no more. What guards it is
        ``scripts/diag/contact_pair_parity_diag.py`` -- a bit clash makes
        the three backends disagree about which geoms touch, and that is
        the one thing it checks.

        The universal alternative -- renumbering each entity's bits into
        globally unique positions on import -- would make namespaces truly
        composable, and would also spend a 32-bit budget a few entities at
        a time and put us somewhere mjlab is not. The reference is mjlab.
        """
        for name, entity in self.entities.items():
            if not hasattr(entity, self.LOCAL_COLLISION_MASK_FLAG):
                raise AttributeError(
                    f"Genesis entity {name!r} has no {self.LOCAL_COLLISION_MASK_FLAG!r}. "
                    "Genesis renamed or removed the flag that gates cross-entity "
                    "collision masks; find its replacement rather than letting the "
                    "assignment below quietly create a field nothing reads."
                )
            source = self._entity_mjcf_path(name)
            if source is None or self._authored_masks_survive_import(entity, source):
                setattr(entity, self.LOCAL_COLLISION_MASK_FLAG, False)

    @staticmethod
    def _authored_masks_survive_import(entity, mjcf_path: str) -> bool:
        """Are this asset's contype / conaffinity still the ones its author wrote?

        Only then may they be compared against another entity's.

        MuJoCo forbids a pair two ways -- the masks, and a separate
        ``<contact><exclude>`` list it keeps as ``exclude_signature``.
        Genesis's collider has only the masks, so when an MJCF carries
        exclusions it folds both into one field: it collects every
        forbidden pair and hands them to a z3 solve for a fresh, minimal
        bit assignment, overwriting what the author wrote
        (``genesis/utils/mjcf.py``, guarded by ``if mj.nexclude:``, and
        ``genesis/utils/collision.py:solve_contype_conaffinity``).

        That assignment is exactly right for the pairs it was given, and
        those are this entity's geoms alone. Nothing in the solve mentions
        the ground or a prop, so the bits it hands out have no relationship
        to theirs. T1 measured: its trunk came back 2/0 against a 1/1
        ground, `(2 & 1) | (1 & 0) == 0`, and the trunk stopped touching
        the floor -- while the feet drew 3/3 and were fine. Not a rule,
        a coincidence per geom.

        So the predicate here is the one Genesis itself branches on. An
        asset with no exclusions keeps the masks its author wrote and can
        be read across entities; an asset with exclusions cannot, and
        keeps Genesis's own guard.

        Unless Genesis has been taught to keep exclusions in a list of
        their own, in which case it never rewrites and every asset
        qualifies. That is what the attribute probe below asks.

        What that costs: an asset that uses BOTH exclusions AND masks
        meant to reach another entity gets the guard, and the second
        intent is silently dropped. No asset here does -- K1 is the only
        one spending non-default bits and it declares no exclusions -- and
        ``scripts/diag/contact_pair_parity_diag.py`` is what would catch a
        new one, since the three backends would stop agreeing about which
        geoms touch. The real fix is upstream: give Genesis a pair-exclude
        list of its own, as MuJoCo and Newton both have, and the masks
        never need to carry two meanings at once.
        """
        # A Genesis that carries exclusions in their own list never rewrote
        # the masks, whatever the file declares. Probed rather than assumed
        # so this holds against a stock Genesis too: there the attribute is
        # absent and the exclusion count decides, as it must.
        if hasattr(entity, "_excluded_link_names"):
            return True
        return not mujoco.MjSpec.from_file(mjcf_path).excludes

    def _entity_mjcf_path(self, name: str) -> str | None:
        """The MJCF this entity came from, or None if it came from a URDF.

        A URDF has no exclusion list to fold in, so Genesis leaves its
        masks alone and they may be compared across entities.
        """
        cfg = self.config.entities.get(name)
        return None if cfg is None else cfg.mjcf_path

    def build_scene(self):
        # Before build: the collider reads the flag when it assembles its
        # pair list, which happens inside scene.build().
        self._share_one_collision_mask_namespace()
        self.scene.build(n_envs=self.env.num_envs, env_spacing=self.config.env_spacing, center_envs_at_origin=False)
        self._place_fixed_entities()
        self._configure_robot_dynamics()
        self.env.vis_manager.inject_custom_context()

    def _place_fixed_entities(self) -> None:
        """Put every welded entity at its declared ``init_state`` pose.

        Free-floating entities are placed by reset events, which is why every
        morph above stays at the origin. A welded entity has no root joint, so
        no reset event can ever move it — its pose has to be written once,
        here, or it stays at the origin no matter what the config declared.
        Newton bakes the same pose into ``add_urdf(xform=...)`` and mjlab
        writes ``root_body.pos`` on the spec, so this is what makes
        ``init_state.pos`` mean the same thing on all three backends.

        The write goes through ``relative=False`` (world frame) rather than a
        morph offset: a morph offset would also shift the frame that the
        ``relative=True`` default of ``get_pos`` / ``get_quat`` reports in, and
        every read would come back relative to the object instead of the world.

        ``base_link.is_fixed`` is the predicate rather than ``cfg.floating``
        because the MJCF path never passes ``floating`` to the morph at all —
        Genesis infers the base joint from the file, so only the built entity
        knows the answer.
        """
        for cfgs, entities in ((self.config.entities, self.entities), (self.config.rigid_objects, self.rigid_objects)):
            for name, entity in entities.items():
                if not entity.base_link.is_fixed:
                    continue
                init_state = cfgs[name].init_state
                pos = torch.tensor(init_state.pos, dtype=torch.float32, device=self.env.device)
                quat = torch.tensor(init_state.rot, dtype=torch.float32, device=self.env.device)
                entity.set_pos(pos.expand(self.env.num_envs, 3).contiguous(), relative=False, zero_velocity=False)
                entity.set_quat(quat.expand(self.env.num_envs, 4).contiguous(), relative=False, zero_velocity=False)

    def _set_kinematic_tree(self):
        def _resolve(name: str):
            cfg = self.config.entities.get(name)
            if cfg is None:
                return None
            mjcf_path = getattr(cfg, "mjcf_path", None)
            if mjcf_path:
                return ("mjcf_path", mjcf_path)
            urdf_path = getattr(cfg, "urdf_path", None)
            if urdf_path:
                return ("urdf", urdf_path)
            return None

        self.trees = build_kinematic_trees(self.entities.keys(), _resolve)

    def _create_scene(self) -> None:
        """Initialize scene with basic settings"""
        # A camera needs the batch renderer, and Genesis only accepts one
        # when the scene is constructed — adding a camera to a scene
        # without it draws one environment at a time.
        renderer = gs.renderers.BatchRenderer() if self.config.cameras else None
        self.scene = gs.Scene(
            sim_options=self.config.sim_options,
            viewer_options=self.config.viewer_options,
            vis_options=self.config.vis_options,
            rigid_options=self.config.rigid_options,
            show_viewer=self.config.show_viewer,
            renderer=renderer,
        )

    def _configure_robot_dynamics(self) -> None:
        """Apply gains/armature from ArticulationCfg actuators.

        For **implicit** actuators, we set the simulator's PD gains (Kp/Kd)
        so the simulator drives the joints internally.

        For **explicit** actuators (IdealPD, DelayedPD, LSTM, etc.), the
        simulator's PD gains are still set here but are effectively unused:
        Genesis switches a joint to force mode when ``control_dofs_force()``
        is called (the last-called control mode wins), so the Kp/Kd values
        have no effect once force mode is active.

        This differs from Newton, where PD forces are *always* summed with
        ``joint_f`` regardless of calling order, requiring explicit ke=0/kd=0
        to disable the internal PD.  (See ``_load_urdf_entity`` in
        ``newton/scene.py`` for that handling.)
        """
        for entity_name, entity in self.entities.items():
            cfg = self.config.entities.get(entity_name)
            if cfg is None:
                continue

            for act_cfg in cfg.articulation.actuators:
                name_keys = list(act_cfg.target_names_expr)
                dof_ids, joint_names = entity_utils.find_dofs(entity=entity, name_keys=name_keys)
                if not dof_ids:
                    continue

                num_dofs = len(dof_ids)

                # Only set Kp/Kd for implicit actuators (simulator PD)
                if isinstance(act_cfg, ImplicitActuatorCfg):
                    # Stiffness — float or dict[regex, float]
                    if isinstance(act_cfg.stiffness, dict):
                        sub_ids, sub_names = entity_utils.find_dofs(
                            entity=entity, name_keys=list(act_cfg.stiffness.keys())
                        )
                        if sub_ids:
                            _, _, vals = string_utils.resolve_matching_names_values(act_cfg.stiffness, sub_names)
                            entity.set_dofs_kp(vals, sub_ids)
                    elif act_cfg.stiffness is not None and act_cfg.stiffness > 0:
                        entity.set_dofs_kp([act_cfg.stiffness] * num_dofs, dof_ids)

                    # Damping — float or dict[regex, float]
                    if isinstance(act_cfg.damping, dict):
                        sub_ids, sub_names = entity_utils.find_dofs(
                            entity=entity, name_keys=list(act_cfg.damping.keys())
                        )
                        if sub_ids:
                            _, _, vals = string_utils.resolve_matching_names_values(act_cfg.damping, sub_names)
                            entity.set_dofs_kv(vals, sub_ids)
                    elif act_cfg.damping is not None and act_cfg.damping > 0:
                        entity.set_dofs_kv([act_cfg.damping] * num_dofs, dof_ids)

                # Armature — float or dict[regex, float]
                if isinstance(act_cfg.armature, dict):
                    sub_ids, sub_names = entity_utils.find_dofs(entity=entity, name_keys=list(act_cfg.armature.keys()))
                    if sub_ids:
                        _, _, vals = string_utils.resolve_matching_names_values(act_cfg.armature, sub_names)
                        entity.set_dofs_armature(vals, sub_ids)
                elif isinstance(act_cfg.armature, int | float) and act_cfg.armature > 0:
                    entity.set_dofs_armature([act_cfg.armature] * num_dofs, dof_ids)

                # Effort limit — symmetric force range [-limit, +limit]. Applied
                # to both implicit and explicit actuators so Genesis enforces the
                # same motor saturation as Newton/Mjlab. For explicit actuators
                # this is redundant with the Python-side _clip_effort in
                # IdealPDActuator but keeps cross-sim behavior identical when
                # URDF-declared limits differ from cfg.
                if isinstance(act_cfg.effort_limit, dict):
                    sub_ids, sub_names = entity_utils.find_dofs(
                        entity=entity, name_keys=list(act_cfg.effort_limit.keys())
                    )
                    if sub_ids:
                        _, _, vals = string_utils.resolve_matching_names_values(act_cfg.effort_limit, sub_names)
                        neg = [-float(v) for v in vals]
                        pos = [float(v) for v in vals]
                        entity.set_dofs_force_range(neg, pos, sub_ids)
                elif act_cfg.effort_limit is not None and act_cfg.effort_limit > 0:
                    limit = float(act_cfg.effort_limit)
                    entity.set_dofs_force_range(
                        [-limit] * num_dofs,
                        [limit] * num_dofs,
                        dof_ids,
                    )

                # Friction loss — static joint friction [N*m]. Scalar only.
                # ``None`` keeps whatever the asset declares; a number forces
                # it on every matched joint, zero included.
                if act_cfg.frictionloss is not None:
                    entity.set_dofs_frictionloss(
                        [float(act_cfg.frictionloss)] * num_dofs,
                        dof_ids,
                    )

    def build_articulation_indexing(
        self,
        actuated_dof_names: list[str],
        entity_name: str = "robot",
    ):
        """Build ArticulationIndexing for the given entity in canonical joint order.

        Joint order is computed by a DFS walk of the kinematic body tree
        (siblings sorted alphabetically by bare body name at each level),
        emitting each body's inbound joint when visited. This order depends
        only on the kinematic structure + joint/body names, so it is identical
        across simulators when the same robot is loaded — regardless of how
        Genesis / Newton / mjlab happen to enumerate bodies internally
        (Genesis uses BFS by depth, Newton/mjlab follow MJCF declaration
        order, etc.). The user's ``actuated_dof_names`` regexes filter that
        canonical list while preserving canonical order.

        Args:
            actuated_dof_names: Regex patterns for actuated joints.
            entity_name: Which entity to index.

        Returns:
            ArticulationIndexing with canonical ↔ simulator mappings.
        """
        entity = self.entities[entity_name]
        canonical_names = _canonical_joint_order_genesis(entity)
        matched_names, _ = filter_canonical_to_actuated(canonical_names, actuated_dof_names)
        # An entity with zero actuated joints (e.g. a free-flying drone)
        # is a legitimate case — return an empty ArticulationIndexing
        # so the action manager can still operate via term-based actions.
        if not matched_names:
            empty_long = torch.zeros(0, device=self.env.device, dtype=torch.long)
            empty_float = torch.zeros(0, device=self.env.device, dtype=torch.float32)
            return ArticulationIndexing(
                joint_names=(),
                sim_indices=empty_long,
                sim_to_canonical=empty_long.clone(),
                joint_limits_lower=empty_float,
                joint_limits_upper=empty_float.clone(),
            )

        # Resolve each matched joint name to its Genesis-local DOF id(s).
        dof_ids: list[int] = []
        for name in matched_names:
            joint = entity.get_joint(name)
            ids = joint.dofs_idx_local
            if hasattr(ids, "__iter__"):
                dof_ids.extend(int(i) for i in ids)
            else:
                dof_ids.append(int(ids))
        sim_indices = torch.tensor(dof_ids, device=self.env.device)

        # sim_to_canonical: identity since RobotData indexes by sim_indices.
        sim_to_canonical = torch.arange(len(dof_ids), device=self.env.device)

        # Joint limits in canonical order.
        dof_lower, dof_upper = entity.get_dofs_limit(dofs_idx_local=sim_indices)

        return ArticulationIndexing(
            joint_names=tuple(matched_names),
            sim_indices=sim_indices,
            sim_to_canonical=sim_to_canonical,
            joint_limits_lower=dof_lower[0],
            joint_limits_upper=dof_upper[0],
        )

    def step(self):
        # Headless training with no cameras has nothing to draw: skip the
        # visualizer/recorder bookkeeping Genesis otherwise runs on every
        # substep. Any visual consumer (native viewer, camera sensors)
        # keeps the exact historical behavior.
        update_vis = bool(self.config.show_viewer or self.config.cameras)
        self.scene.step(update_visualizer=update_vis, refresh_visualizer=update_vis)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        # Genesis manages scene state internally; nothing to do here
        # for partial reset. Kept for cross-sim API symmetry.
        pass
