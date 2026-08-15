import os

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
from rlworld.rl.envs.utils.warp_logging import configure_warp_logging
from rlworld.rl.envs.world import World
from rlworld.rl.utils import set_seed, string as _su

# Opt-in first-NaN forensics (JAXRLWORLD_NAN_TRIPWIRE=1): stop the run at
# the first non-finite physics state and dump contact/constraint counter
# utilization plus per-env kinematics — mid-training Newton NaNs have
# historically been rare-tail events (1e8 steps in) that offline
# random-action rollouts cannot reproduce.
_NAN_TRIPWIRE = os.environ.get("JAXRLWORLD_NAN_TRIPWIRE", "0") == "1"


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
        configure_warp_logging()
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

        # Per-world reset mask consumed by ``_reset_scene`` -> ``solver.reset``.
        # Newton's ``SolverBase.reset`` expects shape ``(world_count + 1,)``:
        # entries before the last select local worlds by index, the final entry
        # selects global entities living in world -1 (we never reset those, so
        # it stays False). Allocated once and refilled per reset — the warp
        # view aliases the torch buffer, so the data pointer stays stable.
        self._reset_world_mask = torch.zeros(self.num_envs + 1, dtype=torch.bool, device=self.device)
        self._reset_world_mask_wp = wp.from_torch(self._reset_world_mask)

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
            selector.joint_names,
            preserve_order=selector.preserve_order,
            entity_name=selector.name,
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
                selector.actuator_names,
                preserve_order=selector.preserve_order,
                entity_name=selector.name,
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
                rigid_objects=self.scene_cfg.rigid_objects,
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

        from rlworld.rl.envs.newton.robot_data import NewtonRigidObjectData, NewtonRobotData
        from rlworld.rl.envs.newton.robot_state_writer import (
            NewtonRigidObjectStateWriter,
            NewtonRobotStateWriter,
        )

        # Per-entity RobotData / state-writer caches (mirrors GenesisEnv). The
        # "robot" entry is built from the same articulation view as before
        # (scene_manager.robot_view IS articulation_views["robot"]), so the
        # single-robot path is unchanged; additional entities (rigid objects,
        # extra robots) get their own entry keyed by name.
        self._robot_data_cache: dict[str, NewtonRobotData] = {}
        self._robot_state_writer_cache: dict[str, NewtonRobotStateWriter] = {}
        # Home pose per entity: taking the first entity that declared any
        # joint_pos and applying it everywhere gave a second robot the
        # first one's pose.
        for name in self.scene_manager.entities:
            view = self.scene_manager.articulation_views[name]
            self._robot_data_cache[name] = NewtonRobotData(
                self,
                view,
                default_joint_pos=self._resolve_default_joint_pos(name),
                soft_joint_pos_limit_factor=self.scene_cfg.entities[name].articulation.soft_joint_pos_limit_factor,
            )
            self._robot_state_writer_cache[name] = NewtonRobotStateWriter(self, view)
        self._robot_data = self._robot_data_cache["robot"]
        self._robot_state_writer = self._robot_state_writer_cache["robot"]

        # Passive rigid objects — RigidObjectData (root + body, no joints) from
        # each object's own ArticulationView (free-joint single body).
        for name in self.scene_manager.rigid_objects:
            view = self.scene_manager.articulation_views[name]
            data = NewtonRigidObjectData(self, view, default_joint_pos=None)
            self._rigid_object_data_cache[name] = data
            # Immovability is read off the model once (welded, or kinematic)
            # and handed to the writer so both sides agree on one source.
            self._rigid_object_state_writer_cache[name] = NewtonRigidObjectStateWriter(
                self, view, immovable=data.is_fixed_base
            )

    def _post_setup(self) -> None:
        """Snapshot DR baselines, then capture CUDA graph.

        ``_dr_baselines`` must be filled BEFORE ``capture()`` so the
        cached CUDA graph never replays a warp write into the cloned
        baseline tensors, and BEFORE any DR term fires so the snapshot
        reflects the URDF/MJCF defaults.
        """
        self._dr_baselines = snapshot_newton_dr_baselines(self)
        self.scene_manager.capture()

    def _reset_scene(self, env_ids: torch.Tensor) -> None:
        """Clear solver-internal state for the environments being reset.

        MuJoCo carries buffers across ``step`` calls that belong to the *old*
        episode: ``qacc_warmstart`` (the constraint solver's initial guess),
        ``qfrc_applied`` / ``xfrc_applied`` (externally applied forces), ``act``
        (actuator activation) and ``ctrl``. Writing a fresh joint pose does not
        touch any of them, so without this call a reset environment warm-starts
        its first constraint solve from a completely different configuration —
        and, worse, a single NaN produced in one solve survives every
        subsequent reset because the solver warm-starts from the NaN, leaving
        that world permanently dead (newton-physics/newton#1266).

        ``flags=0`` deliberately skips Newton's own joint-state reset: the reset
        events own ``joint_q`` / ``joint_qd`` and have not written the new pose
        yet at this point in :meth:`~rlworld.rl.envs.world.World._reset_idx`.
        Only the solver-owned buffers above are zeroed, for the masked worlds
        only, which is why this is safe to run before the curriculum-visible
        state has been overwritten.

        Mirrors ``isaaclab_newton``'s ``NewtonMJWarpManager._reset_solver_internals``.
        """
        scene_manager = self.scene_manager
        # SolverXPBD keeps no cross-step solver buffers and inherits the
        # ``SolverBase.reset`` no-op, so there is nothing to clear there.
        if scene_manager.config.solver_type != "mujoco":
            return
        solver = scene_manager.solver
        # The CPU MuJoCo path owns a single global ``MjData`` whose reset is not
        # mask-aware: it would clear every world, including the ones still
        # mid-episode. Skip rather than corrupt the non-reset environments.
        if solver.use_mujoco_cpu:
            return
        self._reset_world_mask.zero_()
        self._reset_world_mask[env_ids] = True
        solver.reset(scene_manager.state_0, world_mask=self._reset_world_mask_wp, flags=0)

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

        The single write in step 1-2 is valid for *every* substep because
        ``NewtonSceneManager._substep_loop`` runs SolverMuJoCo in single-state
        mode: ``state_0`` is both step input and output, so the buffer holding
        ``body_f`` never stops being the solver's input. (Under the previous
        double-buffered swap the write only reached the first substep, which
        scaled any external wrench down by roughly ``1/substeps``.) The xpbd
        path still swaps and would need a per-substep re-write if it ever grows
        an external-wrench consumer.
        """
        for _ in range(self.decimation):
            self.scene_manager.state_0.clear_forces()
            self.act_manager.apply_actions(self.act_manager.processed_actions)
            # After clear_forces / actuator body_f writes, before the solver
            # (mirrors the propeller action convention).
            if self._external_wrench is not None:
                self._write_external_wrench()
            self.scene_manager.step()
            self.contact_manager.advance(dt=self.physics_dt)

        if _NAN_TRIPWIRE:
            self._nan_tripwire_check()

    def _write_external_wrench(self) -> None:
        """Add the viewer force into ``state_0.body_f`` for one body/env.

        ``State.body_f`` is a world-frame spatial wrench ``[force, torque]``
        referenced at the body COM. Only the target (env, body) slot is
        touched; other slots keep whatever the actuator path wrote this
        substep.
        """
        link_name, force_w, env_idx = self._external_wrench
        model = self.scene_manager.model
        num_worlds = model.world_count
        num_bodies_per_world = model.body_count // num_worlds
        labels = list(getattr(model, "body_label", []) or getattr(model, "body_key", []))
        bare = [bl.rsplit("/", 1)[-1] if "/" in bl else bl for bl in labels[:num_bodies_per_world]]
        key = link_name.rsplit("/", 1)[-1] if "/" in link_name else link_name
        body_idx = bare.index(key)

        body_f = wp.to_torch(self.scene_manager.state_0.body_f).view(num_worlds, num_bodies_per_world, 6)
        body_f[int(env_idx), body_idx, 0:3] += force_w
        wp.copy(
            self.scene_manager.state_0.body_f,
            wp.from_torch(body_f.view(-1, 6), dtype=wp.spatial_vector, requires_grad=False),
        )

        # Update visualization
        if self.vis_manager is not None:
            self.vis_manager.advance()

    def _nan_tripwire_check(self) -> None:
        """Dump forensics and crash on the FIRST non-finite physics state.

        Reports, for every poisoned env: episode step (recent reset?),
        its mjwarp contact and constraint-row counts (buffer-overflow
        attribution), root kinematics and the blown-up joint. Raises so
        the run stops at the first poisoned transition instead of
        feeding NaNs to the learner for days.
        """
        rd = self.get_robot_data()
        checks = (
            ("joint_pos", rd.joint_pos),
            ("joint_vel", rd.joint_vel),
            ("root_pos", rd.root_link_pos_w),
            ("root_lin_vel", rd.root_link_lin_vel_w),
            ("root_ang_vel", rd.root_link_ang_vel_w),
        )
        bad_envs = None
        bad_names = []
        for name, t in checks:
            finite = torch.isfinite(t).reshape(self.num_envs, -1).all(dim=1)
            if not bool(finite.all()):
                bad_names.append(name)
                bad_envs = ~finite if bad_envs is None else (bad_envs | ~finite)
        if bad_envs is None:
            return

        ids = torch.nonzero(bad_envs).flatten()
        mjw_data = self.scene_manager.solver.mjw_data
        naconmax = int(mjw_data.naconmax)
        nacon = int(wp.to_torch(mjw_data.nacon)[0].item())
        n_written = min(nacon, naconmax)
        ncon_world = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        if n_written > 0:
            worldid = wp.to_torch(mjw_data.contact.worldid)[:n_written].to(self.device).long()
            ncon_world = torch.bincount(worldid, minlength=self.num_envs)
        nefc = wp.to_torch(mjw_data.nefc).to(self.device)
        contacts = self.scene_manager.contacts
        rigid_count = int(wp.to_torch(contacts.rigid_contact_count)[0].item())
        rigid_cap = int(contacts.rigid_contact_point_id.shape[0])
        joint_names = [n.rsplit("/", 1)[-1] for n in self.act_manager.actuated_joint_names]

        lines = [
            f"[NAN-TRIPWIRE] non-finite physics state in: {bad_names}",
            f"[NAN-TRIPWIRE] bad envs: {int(bad_envs.sum())}/{self.num_envs}  ids(first16)={ids[:16].tolist()}",
            f"[NAN-TRIPWIRE] contact pools: nacon={nacon}/naconmax={naconmax} (overflow={nacon > naconmax})  "
            f"newton rigid={rigid_count}/{rigid_cap} (sat={rigid_count >= rigid_cap})  njmax={int(mjw_data.njmax)}",
        ]
        for e in ids[:8].tolist():
            jv_f = torch.nan_to_num(rd.joint_vel[e].abs(), nan=float("inf"))
            j = int(jv_f.argmax())
            lines.append(
                f"[NAN-TRIPWIRE] env {e}: ep_len={int(self.episode_length_buf[e])}  "
                f"ncon={int(ncon_world[e])}  nefc={int(nefc[e])}  "
                f"root_z={float(rd.root_link_pos_w[e, 2]):.4f}  "
                f"max|joint_vel|={float(jv_f[j]):.3e} @ {joint_names[j]}  "
                f"root_lin_vel={[round(float(v), 3) for v in rd.root_link_lin_vel_w[e].tolist()]}"
            )
        print("\n".join(lines), flush=True)
        raise RuntimeError("JAXRLWORLD_NAN_TRIPWIRE: first non-finite physics state (details above)")
