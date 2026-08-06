"""MuJoCo (mjlab) builders for the K1 joystick task.

The vendored MJCF is the upstream model verbatim (modulo the foot-
sphere contact triplet).

Solver recipe: implicitfast + pyramidal cone + impratio 1 + 1 substep
(``_SIM_TIMINGS``), matching mjlab's velocity task on the three knobs
that dominate the mjwarp step cost (substeps, cone, impratio) for ~2-3x
faster physics. iterations stay at 100/50 (they early-converge under
pyramidal/impratio 1, so they add little cost).

CONTACT TRADE-OFF: an earlier recipe used elliptic cone + impratio 100
+ 2 substeps to fully quiet stance-foot micro-bounce (per-step contact-
force STD ~10 N -> ~1 N, contact toggling 9%/step -> 2%, genesis-level
in the k1_feet_contact_parity diag). The current recipe trades some of
that contact quiet for speed — re-check feet_air_time/feet_phase and
gait quality if it regresses.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

import mujoco
from mjlab.sim.sim import MujocoCfg, SimulationCfg

from rlworld.rl.actuators import DelayedPDActuatorCfg
from rlworld.rl.configs import TerminationTermConfig
from rlworld.rl.configs.common_config_classes import TerminationsConfig
from rlworld.rl.configs.mujoco_config_classes import (
    MujocoActionConfig,
    MujocoConfigsForRun,
    MujocoEnvConfig,
    MujocoObservationConfig,
    MujocoSceneConfig,
    VisualizationConfig,
)
from rlworld.rl.configs.scene.unified_entity_config import (
    ArticulationCfg,
    InitialStateCfg,
    MujocoEntityCfg,
)
from rlworld.rl.configs.sensors import ContactMatch, ContactSensorCfg
from rlworld.rl.envs.mdp.terminations.common.terminations import nan_detection
from rlworld.rl.envs.mdp.terminations.k1_locomotion import gravity_z_positive
from rlworld.rl.envs.mdp.terminations.mujoco import terminations as tf

if TYPE_CHECKING:
    from .base import K1JoystickConfig

CONFIGS_FOR_RUN_CLS = MujocoConfigsForRun
OBSERVATION_CFG_CLS = MujocoObservationConfig


@dataclass
class K1SpecFn:
    """Picklable ``spec_fn``: load the vendored MJCF and strip its
    ``<actuator>`` block.

    The XML ships position actuators (upstream drives them directly);
    here the explicit IdealPD actuators own the joints, and mjlab would
    otherwise add same-named motors → "repeated name ... in actuator".
    """

    mjcf_path: str

    def __call__(self):
        spec = mujoco.MjSpec.from_file(str(Path(self.mjcf_path).resolve()))
        for act in list(spec.actuators):
            spec.delete(act)
        # Whole-robot angular-momentum sensor (attached name:
        # "robot/root_angmom"). Read by the G1-recipe variant's
        # angular_momentum_penalty on this backend; unused (harmless)
        # under the pal recipe.
        sensor = spec.add_sensor()
        sensor.name = "root_angmom"
        sensor.type = mujoco.mjtSensor.mjSENS_SUBTREEANGMOM
        sensor.objtype = mujoco.mjtObj.mjOBJ_BODY
        sensor.objname = "Trunk"
        return spec


def build_visualization(cfg: K1JoystickConfig) -> VisualizationConfig:
    return VisualizationConfig(show_viewer=False, record_video=False)


def build_env(cfg: K1JoystickConfig, timing: Dict[str, Any]) -> MujocoEnvConfig:
    @dataclass
    class _TerminationsCfg(TerminationsConfig):
        fall = TerminationTermConfig(gravity_z_positive)
        nan = TerminationTermConfig(nan_detection)
        time_out = TerminationTermConfig(tf.time_out)

    return MujocoEnvConfig(
        num_envs=cfg.num_envs,
        env_name="MujocoEnv",
        task_name="K1_Joystick",
        seed=cfg.seed,
        episode_length_s=cfg.episode_length_s,
        decimation=timing["decimation"],
        terminations=_TerminationsCfg(),
    )


def build_scene(cfg: K1JoystickConfig, timing: Dict[str, Any]) -> MujocoSceneConfig:
    r = cfg.robot

    robot_entity = MujocoEntityCfg(
        mjcf_path=r.mjcf_path,
        init_state=InitialStateCfg(
            pos=(0.0, 0.0, r.base_init_height),
            joint_pos=r.default_joint_angles,
        ),
        floating=True,
        articulation=ArticulationCfg(
            actuators=(
                DelayedPDActuatorCfg(
                    target_names_expr=(".*",),
                    stiffness=r.p_gains,
                    damping=r.d_gains,
                    armature=r.armature,
                    effort_limit=r.effort_limits,
                    tau_scale=r.tau_scale,
                    velocity_limit=r.velocity_limit,
                    knee_point_velocity=r.knee_point_velocity,
                    tau_lpf_time_constant=r.tau_lpf_time_constant,
                    physics_dt=timing["dt"],
                    dyn_gain=r.dyn_gain,
                    dyn_gain_velocity=r.dyn_gain_velocity,
                    frictionloss=0.1,
                    min_delay=cfg.action_delay_min,
                    max_delay=cfg.action_delay_max,
                ),
            ),
            soft_joint_pos_limit_factor=0.95,
        ),
        # mjlab entities are built from an MjSpec (actuators stripped —
        # see K1SpecFn).
        spec_fn=K1SpecFn(mjcf_path=r.mjcf_path),
        collisions=(),
    )

    # Feet↔ground: one slot per foot, each aggregating that foot's four
    # collision spheres. Feet↔feet: the box pair (collision cost).
    # Body-mode primaries give one slot per FOOT (geom-mode expands to
    # one slot per sphere -> 8 columns and a critic-dim mismatch vs the
    # other backends). The box geom cannot touch the floor (contype), so
    # the body aggregate equals the sphere aggregate.
    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(mode="body", pattern=tuple(r.foot_names), entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
    )
    feet_pair_cfg = ContactSensorCfg(
        name="feet_pair_contact",
        primary=ContactMatch(mode="geom", pattern=("left_foot",), entity="robot"),
        # Secondary must be a SINGLE exact name (mjlab restriction).
        secondary=ContactMatch(mode="geom", pattern="right_foot", entity="robot"),
        fields=("found",),
        reduce="netforce",
        num_slots=1,
    )

    substep_dt = timing["dt"] / timing.get("substeps", 1)
    return MujocoSceneConfig(
        physics_dt=timing["dt"],
        substeps=timing.get("substeps", 1),
        num_envs=cfg.num_envs,
        env_spacing=2.0,
        robot_entity_name="robot",
        entities={"robot": robot_entity},
        sensors=(feet_ground_cfg, feet_pair_cfg),
        # Full SimulationCfg because the wrapper's field set doesn't
        # expose the integrator. Canonical mjwarp humanoid solver recipe
        # (see module docstring for why this departs from upstream).
        mjlab_sim_cfg=SimulationCfg(
            nconmax=None,
            njmax=None,
            mujoco=MujocoCfg(
                timestep=substep_dt,
                integrator="implicitfast",
                iterations=100,
                ls_iterations=50,
                # mjlab-parity on the two per-iteration-cost knobs (pyramidal
                # cone + impratio 1). With substeps=1 these are the recipe's
                # dominant speedups; see the module docstring for the contact
                # trade-off. iterations stay at 100/50 (they early-converge
                # under pyramidal/impratio 1, so they cost little).
                impratio=1.0,
                cone="pyramidal",
                ccd_iterations=50,
            ),
        ),
        preset_class_name=type(cfg).__name__,
        preset_module_path=type(cfg).__module__,
    )


def build_action(cfg: K1JoystickConfig) -> MujocoActionConfig:
    r = cfg.robot
    return MujocoActionConfig(
        entity_name="robot",
        actuated_dof_names=r.actuated_dof_patterns,
        # Physical actuator mode supplies a per-joint action scale (0.25·effort/kp);
        # it overrides the recipe's scale. legacy mode ⇒ None ⇒ recipe scale.
        action_scale=(r.physical_action_scale if r.physical_action_scale is not None else cfg.action_scale),
        clip_actions=cfg.action_clip,
        offset=r.get_action_offset(),
    )
