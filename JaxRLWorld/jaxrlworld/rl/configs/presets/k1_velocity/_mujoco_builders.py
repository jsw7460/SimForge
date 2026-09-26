"""MuJoCo (mjlab) builders for the Booster K1 velocity port.

This is the backend the source recipe runs on, so it is the reference cell:
where the three disagree, this one is presumed right until a diagnostic says
otherwise.

Two deliberate departures from the source, both forced:

- **Explicit PD instead of the simulator's built-in position actuators.** The
  source drives joints through MuJoCo's own position actuators and clamps their
  force range each step to follow the motor's torque-speed curve. The same
  curve is available here only on the explicit-PD actuator, which also carries
  the command delay the source applies. The control law is the same; its
  integration is not (built-in PD is implicit in the solve, explicit PD is a
  force applied before it).
- **Command delay resamples per reset.** The source holds a drawn lag with
  probability 0.3 from step to step, giving temporally correlated latency. The
  actuator here draws one lag per environment per reset and holds it for the
  episode.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

import mujoco

from jaxrlworld.rl.actuators import DelayedPDActuatorCfg
from jaxrlworld.rl.configs import TerminationTermConfig
from jaxrlworld.rl.configs.common_config_classes import TerminationsConfig
from jaxrlworld.rl.configs.mujoco_config_classes import (
    MujocoActionConfig,
    MujocoConfigsForRun,
    MujocoEnvConfig,
    MujocoObservationConfig,
    MujocoSceneConfig,
    VisualizationConfig,
)
from jaxrlworld.rl.configs.scene.unified_entity_config import (
    ArticulationCfg,
    InitialStateCfg,
    MujocoEntityCfg,
)
from jaxrlworld.rl.configs.sensors import ContactMatch, ContactSensorCfg
from jaxrlworld.rl.envs.mdp.terminations import k1_velocity as booster_tf
from jaxrlworld.rl.envs.mdp.terminations.common import terminations as common_tf
from jaxrlworld.rl.envs.mdp.terminations.mujoco import terminations as tf

if TYPE_CHECKING:
    from .base import K1VelocityConfig

CONFIGS_FOR_RUN_CLS = MujocoConfigsForRun
OBSERVATION_CFG_CLS = MujocoObservationConfig


@dataclass
class K1SpecFn:
    """Picklable ``spec_fn``: load the asset and hand mjlab the spec.

    Nothing is edited. The asset ships no ``<actuator>`` block for mjlab's
    own actuators to collide with, and the whole-robot angular-momentum
    sensor the reward reads is declared in the file. Both facts are asserted
    rather than assumed, so a change to the asset fails here instead of
    somewhere downstream.
    """

    mjcf_path: str

    def __call__(self) -> mujoco.MjSpec:
        spec = mujoco.MjSpec.from_file(str(Path(self.mjcf_path).resolve()))
        actuators = list(spec.actuators)
        if actuators:
            raise ValueError(
                f"{self.mjcf_path} declares {len(actuators)} actuators; this preset's "
                "explicit PD actuators own the joints and mjlab would add same-named motors"
            )
        sensor_names = {s.name for s in spec.sensors}
        if "root_angmom" not in sensor_names:
            raise ValueError(
                f"{self.mjcf_path} is missing the 'root_angmom' sensor the "
                f"angular-momentum reward reads; found {sorted(sensor_names)}"
            )
        return spec


def build_visualization(cfg: K1VelocityConfig) -> VisualizationConfig:
    return VisualizationConfig(show_viewer=False, record_video=False)


def build_env(cfg: K1VelocityConfig, timing: Dict[str, Any]) -> MujocoEnvConfig:
    limit_angle = math.radians(cfg.fall_limit_angle_deg)
    probability = cfg.fall_probability

    @dataclass
    class _TerminationsCfg(TerminationsConfig):
        fell_over = TerminationTermConfig(
            booster_tf.stochastic_bad_orientation,
            {"limit_angle": limit_angle, "probability": probability},
        )
        illegal_contact = TerminationTermConfig(
            common_tf.illegal_contact,
            {"contact_group": "non_foot_ground_contact"},
        )
        time_out = TerminationTermConfig(tf.time_out)

    return MujocoEnvConfig(
        num_envs=cfg.num_envs,
        env_name="MujocoEnv",
        task_name="K1 Booster Velocity",
        seed=cfg.seed,
        episode_length_s=cfg.episode_length_s,
        decimation=timing["decimation"],
        terminations=_TerminationsCfg(),
    )


def build_scene(cfg: K1VelocityConfig, timing: Dict[str, Any]) -> MujocoSceneConfig:
    r = cfg.robot
    history = timing["decimation"]
    foot_pattern = r"^(left_foot_link|right_foot_link)$"

    feet_ground = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(mode="subtree", pattern=foot_pattern, entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        history_length=history,
        num_slots=1,
    )
    # Drives the fall-on-illegal-contact termination: anything but a foot
    # touching the ground ends the episode.
    non_foot_ground = ContactSensorCfg(
        name="non_foot_ground_contact",
        primary=ContactMatch(
            mode="body",
            pattern=".*",
            entity="robot",
            exclude=tuple(r.foot_names),
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        history_length=history,
        num_slots=1,
    )
    # One subtree-against-itself pair, which is the shape the reference
    # framework's own humanoid tasks use. The reward reads it as a binary
    # any-self-contact signal, which the other two backends reproduce from
    # their per-link groups.
    self_collision = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern=r.trunk_body_name, entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern=r.trunk_body_name, entity="robot"),
        reduce="none",
        history_length=history,
        num_slots=1,
    )

    robot_entity = MujocoEntityCfg(
        mjcf_path=r.mjcf_path,
        init_state=InitialStateCfg(
            pos=(0.0, 0.0, r.base_init_height),
            joint_pos=r.default_joint_angles,
        ),
        floating=True,
        enable_self_collisions=True,
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
                    frictionloss=r.joint_frictionloss,
                    min_delay=cfg.action_delay_min,
                    max_delay=cfg.action_delay_max,
                ),
            ),
            soft_joint_pos_limit_factor=r.soft_joint_pos_limit_factor,
        ),
        spec_fn=K1SpecFn(mjcf_path=r.mjcf_path),
        # Contact parameters live in the asset's own default classes, so there
        # is nothing to patch onto the compiled spec here -- and nothing that
        # would be invisible to the other two backends.
        collisions=(),
    )

    return MujocoSceneConfig(
        physics_dt=timing["dt"],
        substeps=timing["substeps"],
        num_envs=cfg.num_envs,
        env_spacing=2.0,
        robot_entity_name="robot",
        cone="pyramidal",
        entities={"robot": robot_entity},
        sensors=(feet_ground, non_foot_ground, self_collision),
        # The source's solver budget for this task.
        solver_iterations=10,
        solver_ls_iterations=20,
        ccd_iterations=50,
        # Contact budgets are the source's flat-terrain values. A full-body
        # collision model with self-collision enabled generates far more
        # contact rows than a feet-only one, and mjwarp SKIPS rows past the
        # budget silently -- feet lose contact, penetrate, and the run NaNs.
        # Re-measure after any mujoco-warp bump.
        nconmax=50,
        njmax=300,
        contact_sensor_maxmatch=64,
        preset_class_name=type(cfg).__name__,
        preset_module_path=type(cfg).__module__,
    )


def build_action(cfg: K1VelocityConfig) -> MujocoActionConfig:
    r = cfg.robot
    return MujocoActionConfig(
        entity_name="robot",
        actuated_dof_names=r.actuated_dof_patterns,
        action_scale=r.physical_action_scale,
        clip_actions=cfg.action_clip,
        offset=r.get_action_offset(),
    )
