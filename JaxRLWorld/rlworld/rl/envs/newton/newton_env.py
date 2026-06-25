import torch
import warp as wp

from rlworld.rl.configs import CommandConfig, CurriculumManagerConfig, EventConfig, RewardConfig
from rlworld.rl.configs.newton_config_classes import (
    NewtonActionConfig,
    NewtonEnvConfig,
    NewtonObservationConfig,
    NewtonSceneConfig,
    VisualizationConfig,
)
from rlworld.rl.configs.scene.entity_selector import ResolvedEntity, SceneEntitySelector
from rlworld.rl.envs.managers.newton import (
    NewtonVisualizationManager,
    NewtonVisualizationManagerConfig,
)
from rlworld.rl.envs.managers.registry import ManagerRegistry
from rlworld.rl.envs.utils.newton.dr_baselines import snapshot_newton_dr_baselines
from rlworld.rl.envs.world import World
from rlworld.rl.utils import set_seed, string as _su


class NewtonEnv(World):
    sim_name: str = "Newton"
    sim_type: str = "newton"

    def __init__(
        self,
        num_envs: int,
        env_cfg: NewtonEnvConfig,
        scene_cfg: NewtonSceneConfig,
        visualization_cfg: VisualizationConfig,
        obs_cfg: NewtonObservationConfig,
        act_cfg: NewtonActionConfig,
        reward_cfg: RewardConfig,
        command_cfg: CommandConfig,
        event_cfg: EventConfig,
        curriculum_cfg: CurriculumManagerConfig,
    ):
        set_seed(env_cfg.seed)
        # Seed warp's kernel RNG too. Newton's solvers don't sample
        # randoms themselves, but anything else running through warp
        # (e.g. user-side terrain / control perturbation) does, and we
        # want a single ``--seed`` to pin both.
        wp.rand_init(wp.int32(env_cfg.seed))
        super().__init__()

        self.seed = env_cfg.seed
        self.num_envs = num_envs
        self.device = wp.device_to_torch(wp.get_device())

        # Store high-level configs
        self.env_cfg = env_cfg
        self.scene_cfg = scene_cfg
        self.visualization_cfg = visualization_cfg
        self.obs_cfg = obs_cfg
        self.act_cfg = act_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg
        self.event_cfg = event_cfg
        self.curriculum_cfg = curriculum_cfg

        # Timing
        self.physics_dt = scene_cfg.dt
        self.decimation = getattr(env_cfg, "decimation", 1)
        self.control_dt = self.physics_dt * self.decimation

        # Initialize buffers
        self._init_buffers()

        # Setup
        self._setup_environment()

    @property
    def robot(self):
        return self.scene_manager.model

    @property
    def robot_data(self):
        return self.get_robot_data("robot")

    def get_robot_data(self, entity_name: str = "robot"):
        return self._robot_data_cache[entity_name]

    def get_robot_state_writer(self, entity_name: str = "robot"):
        """Return the write-API companion to ``get_robot_data``.

        Used by event terms / reset functions to mutate joint and root
        state via ``NewtonRobotStateWriter`` (see
        ``rlworld/rl/envs/newton/robot_state_writer.py``).
        """
        return self._robot_state_writer_cache[entity_name]

    def resolve_selector(self, selector: SceneEntitySelector) -> ResolvedEntity:
        view = self.scene_manager.articulation_views[selector.name]

        joint_ids, joint_names_resolved = self._resolve_canonical_joint_ids(
            selector.joint_names, preserve_order=selector.preserve_order
        )

        body_ids: torch.Tensor | None = None
        body_names_resolved: list[str] | None = None
        if selector.body_names is not None:
            link_idx, link_names_matched = _su.resolve_matching_names(
                list(selector.body_names),
                list(view.link_names),
                preserve_order=selector.preserve_order,
            )
            body_ids = torch.tensor(link_idx, device=self.device, dtype=torch.long)
            body_names_resolved = list(link_names_matched)

        # In Newton, "geom" maps to a per-shape entry under
        # ``ArticulationView.shape_names`` — per-shape bare names, the
        # same canonical name list the ContactMatch resolver uses via
        # ``NewtonLabelIndexing.find_shapes``.  Patterns are matched
        # against these bare leaves with ``re.fullmatch`` — preset
        # ``geom_names`` must be bare-style (``"FR_foot_collision"`` or
        # ``".*_foot_collision"``); XPath-style ``".*/<name>"`` would
        # demand a literal ``/`` the bare list does not contain.
        geom_ids: torch.Tensor | None = None
        geom_names_resolved: list[str] | None = None
        if selector.geom_names is not None:
            shape_idx, shape_names_matched = _su.resolve_matching_names(
                list(selector.geom_names),
                list(view.shape_names),
                preserve_order=selector.preserve_order,
            )
            geom_ids = torch.tensor(shape_idx, device=self.device, dtype=torch.long)
            geom_names_resolved = list(shape_names_matched)

        # Newton actuators are 1:1 with act_manager.actuated_joint_names.
        actuator_ids: torch.Tensor | None = None
        actuator_names_resolved: list[str] | None = None
        if selector.actuator_names is not None:
            actuator_ids, actuator_names_resolved = self._resolve_canonical_joint_ids(
                selector.actuator_names, preserve_order=selector.preserve_order
            )

        if selector.site_names is not None:
            raise NotImplementedError(
                "Newton does not expose sites as a first-class concept; "
                "use SceneEntitySelector.body_names or geom_names instead."
            )

        return ResolvedEntity(
            source_selector=selector,
            name=selector.name,
            joint_ids=joint_ids,
            joint_ids_native=None,
            body_ids=body_ids,
            geom_ids=geom_ids,
            site_ids=None,
            actuator_ids=actuator_ids,
            joint_names=joint_names_resolved if selector.joint_names is not None else None,
            body_names=body_names_resolved,
            geom_names=geom_names_resolved,
            actuator_names=actuator_names_resolved,
        )

    def _build_scene(self) -> None:
        """Create Newton scene and visualization manager."""
        SceneCls = ManagerRegistry.get_class(self.sim_type, "scene")
        SceneCfgCls = ManagerRegistry.get_config_class(self.sim_type, "scene")

        self.scene_manager = SceneCls(
            env=self,
            config=SceneCfgCls(
                num_worlds=self.num_envs,
                entities=self.scene_cfg.entities,
                sensors=self.scene_cfg.sensors,
                contact_sensors=getattr(self.scene_cfg, "contact_sensors", None),
                terrain_cfg=self.scene_cfg.terrain_cfg,
                dt=self.scene_cfg.dt,
                substeps=self.scene_cfg.substeps,
                gravity=self.scene_cfg.gravity,
                solver_type=self.scene_cfg.solver_type,
                solver_cfg=self.scene_cfg.solver_cfg,
                env_spacing=self.scene_cfg.env_spacing,
                collision_max_triangle_pairs=self.scene_cfg.collision_max_triangle_pairs,
            ),
        )
        self.scene_manager.register_entities()
        self.scene_manager.build_scene()

        # Visualization Manager (after scene build)
        # viewer_type="viser" → always use ViserVisualizationManager (matches Genesis pattern)
        if self.visualization_cfg.viewer_type == "viser":
            from rlworld.rl.vis.viser import ViserVisualizationManager
            from rlworld.rl.vis.viser.bridges import NewtonBridge
            from rlworld.rl.vis.viser.viewer import ViserViewerConfig

            bridge = NewtonBridge(self.scene_manager)
            viser_cfg = ViserViewerConfig(
                port=self.visualization_cfg.viser_port,
                share=self.visualization_cfg.viser_share,
                enable_reward_plots=self.visualization_cfg.viser_enable_reward_plots,
                enable_debug_viz=self.visualization_cfg.viser_enable_debug_viz,
                scene=self.visualization_cfg.viser_scene,
            )
            self.vis_manager = ViserVisualizationManager(env=self, bridge=bridge, config=viser_cfg)
        elif self.visualization_cfg.show_viewer or self.visualization_cfg.record_video:
            # Non-viser viewer (GL, rerun, usd, file) — only when actively viewing or recording
            vis_manager_cfg = NewtonVisualizationManagerConfig(
                show_viewer=self.visualization_cfg.show_viewer,
                record_video=self.visualization_cfg.record_video,
                video_dir=self.visualization_cfg.video_dir,
                video_fps=self.visualization_cfg.video_fps or 60,
                viewer_type=self.visualization_cfg.viewer_type,
                viser_port=self.visualization_cfg.viser_port,
                viser_share=self.visualization_cfg.viser_share,
                rerun_web_port=self.visualization_cfg.rerun_web_port,
            )
            self.vis_manager = NewtonVisualizationManager(env=self, config=vis_manager_cfg)
            self.vis_manager.setup()
        else:
            # Headless — no visualization overhead
            self.vis_manager = None

    def _build_sim_managers(self) -> None:
        """Create Newton-specific managers via ManagerRegistry."""
        ActCls = ManagerRegistry.get_class(self.sim_type, "action")
        ActCfgCls = ManagerRegistry.get_config_class(self.sim_type, "action")
        self.act_manager = ActCls(
            env=self,
            config=ActCfgCls(
                actuated_dof_names=self.act_cfg.actuated_dof_names,
                scale=self.act_cfg.action_scale,
                clip=self.act_cfg.clip_actions,
                offset=self.act_cfg.offset,
                settle_steps=self.act_cfg.settle_steps,
                joint_limit_soft_factor=self.act_cfg.joint_limit_soft_factor,
                action_terms=self.act_cfg.action_terms,
            ),
        )

        ObsCls = ManagerRegistry.get_class(self.sim_type, "observation")
        self.obs_manager = ObsCls(
            env=self,
            config=self.obs_cfg,
        )

        ContactCls = ManagerRegistry.get_class(self.sim_type, "contact")
        self.contact_manager = ContactCls(env=self)
        self.contact_manager.register_sensors()

        from rlworld.rl.envs.newton.robot_data import NewtonRobotData
        from rlworld.rl.envs.newton.robot_state_writer import NewtonRobotStateWriter

        # Per-entity RobotData / state-writer caches (mirrors GenesisEnv). The
        # "robot" entry is built from the same articulation view as before
        # (scene_manager.robot_view IS articulation_views["robot"]), so the
        # single-robot path is unchanged; additional entities (rigid objects,
        # extra robots) get their own entry keyed by name.
        self._robot_data_cache: dict[str, NewtonRobotData] = {}
        self._robot_state_writer_cache: dict[str, NewtonRobotStateWriter] = {}
        _default_jp = self._resolve_default_joint_pos()
        for name in self.scene_manager.entities:
            view = self.scene_manager.articulation_views[name]
            self._robot_data_cache[name] = NewtonRobotData(self, view, default_joint_pos=_default_jp)
            self._robot_state_writer_cache[name] = NewtonRobotStateWriter(self, view)
        self._robot_data = self._robot_data_cache["robot"]
        self._robot_state_writer = self._robot_state_writer_cache["robot"]

    def _post_setup(self) -> None:
        """Snapshot DR baselines, then capture CUDA graph.

        ``_dr_baselines`` must be filled BEFORE ``capture()`` so the
        cached CUDA graph never replays a warp write into the cloned
        baseline tensors, and BEFORE any DR term fires so the snapshot
        reflects the URDF/MJCF defaults.
        """
        self._dr_baselines = snapshot_newton_dr_baselines(self)
        self.scene_manager.capture()

    def _step_physics(self) -> None:
        """Newton physics step with decimation.

        Each decimation iteration:
          1. Clears Newton's external-wrench buffer (``state_0.body_f``
             and ``particle_f``). Newton's solver does NOT clear these
             itself; the per-substep clear that used to live inside
             ``NewtonSceneManager.step`` was moved out so that callers
             writing external wrenches can do so AFTER the clear but
             BEFORE the solver runs (matches the convention used in
             every ``newton/examples/*/example_*.py``).
          2. Calls ``act_manager.apply_actions``, which recomputes
             actuator joint torques (``control.joint_f``) AND lets
             non-joint action terms (e.g. ``PropellerThrustAction``)
             write to ``state_0.body_f``.
          3. Runs ``scene_manager.step()``, which now ONLY runs the
             solver substep loop — no clear inside.
        """
        for _ in range(self.decimation):
            self.scene_manager.state_0.clear_forces()
            self.act_manager.apply_actions(self.act_manager.processed_actions)
            self.scene_manager.step()
            self.contact_manager.advance(dt=self.physics_dt)

        # Update visualization
        if self.vis_manager is not None:
            self.vis_manager.advance()
