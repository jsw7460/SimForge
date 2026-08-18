"""Lift the cube from a wrist camera instead of the cube's coordinates.

The task, the rewards and the arm are the vector preset's. What changes
is what the actor is told: the two terms that hand it the cube's
position outright are gone, and a depth image from the D405 on the
wrist takes their place. The critic keeps the coordinates — it is not
deployed, and a value function that has to infer the cube from pixels
learns far more slowly than the policy it is supposed to be teaching.

Mirrors mjlab's ``yam_lift_cube_vision_env_cfg``: 32x32 depth, a 0.5 m
far plane, and a spatial-softmax CNN in front of the MLP.

Usage::

    from rlworld.rl.configs.presets.yam_lift.vision import YamLiftVisionConfig
    cfgs = YamLiftVisionConfig(sim_type="mujoco").build()
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rlworld.rl.configs.common_config_classes import (
    Activation,
    CNNEncoderCfg,
    DistributionType,
    MLPActorCfg,
    MLPCriticCfg,
    NNConfig,
    ObservationGroupConfig,
    OrthoInit,
    PPOPolicyConfig,
    StdType,
    VisionActorCfg,
    VisionCriticCfg,
)
from rlworld.rl.configs.observations import ObservationTermConfig
from rlworld.rl.configs.presets.yam_lift.base import YamLiftConfig
from rlworld.rl.configs.sensors.camera_sensor_config import CameraSensorCfg
from rlworld.rl.envs.mdp.observations.common import perception

CAMERA_GROUP = "camera"
"""Observation group holding the image. Its own group because it is
``(C, H, W)`` and the state vector is ``(D,)``."""

CAMERA_SENSOR = "camera_d405"
"""Sensor name, and the bare name of the camera in the arm's MJCF. The
asset ships the real D405's mounting transform, so wrapping the model's
own camera beats placing one by hand."""

PRIVILEGED_ACTOR_TERMS = ("ee_to_cube", "cube_to_goal", "cube_height")
"""Terms dropped from the actor: each is the cube's state handed over
directly, which is exactly what the camera is there to replace. They
stay on the critic."""


@dataclass
class YamLiftVisionConfig(YamLiftConfig):
    """The lift task with a wrist depth camera in place of cube coordinates."""

    camera_width: int = 32
    camera_height: int = 32

    near_clip: float = 0.07
    """Closest distance the camera reports, metres; nearer surfaces read
    as "hit nothing". A real D405 cannot measure closer than about this.

    It is also the only thing the two backends genuinely disagree about.
    Below a few centimetres the camera is not looking at a surface, it is
    BURIED in one — the cross-sim diag traced every remaining mismatch to
    poses where the wrist had been driven inside the table — and what a
    camera inside a solid sees is undefined. mjwarp culls the backface
    and reports nothing; Newton reports the inner wall. Neither is wrong,
    and neither is a state the physics would ever produce.

    Applied to the observation rather than to a backend, so mjlab, Newton
    and the robot all obey one rule.
    """

    visible_geometry: str = "collision"
    """Which of the arm's two descriptions the camera sees: the coarse
    colliders or the detailed visual meshes. mjlab's own vision task
    shows the colliders. Both backends are made to honour this, which
    they otherwise would not — mjlab draws geom groups, Newton draws
    whatever carries a visibility flag, and after an MJCF import that is
    both descriptions at once."""

    cutoff_distance: float = 0.5
    """Far plane, metres. Everything past it saturates at 1.0, so this is
    the depth range the policy actually resolves — set it to the
    workspace, not the room."""

    cnn: CNNEncoderCfg = field(default_factory=CNNEncoderCfg)

    def build(self):
        cfgs = super().build()
        # One config, both backends. The placement and the field of view
        # come from the arm's own MJCF camera, so mjlab and Newton read
        # the same numbers out of the same file instead of keeping two
        # hand-copied sets in step.
        cfgs.scene.cameras = tuple(cfgs.scene.cameras) + (
            CameraSensorCfg(
                name=CAMERA_SENSOR,
                entity_name="robot",
                camera_name=CAMERA_SENSOR,
                width=self.camera_width,
                height=self.camera_height,
                data_types=("depth",),
                visible_geometry=self.visible_geometry,
            ),
        )
        return cfgs

    def _build_command_config(self):
        cfg = super()._build_command_config()
        # The timer's resample teleports the object, which a policy that
        # can only see it cannot detect: with the camera pointed
        # elsewhere nothing in the observation changes, and a gripper
        # that was holding the object has no way to tell it is now
        # empty. Measured on a trained policy: it recovered from a reset
        # 3 times out of 3, and from a resample 0 times out of 8.
        cfg.terms["lift"].place_object_on_resample = False
        return cfg

    def _build_observation_config(self):
        cfg = super()._build_observation_config()

        for term in PRIVILEGED_ACTOR_TERMS:
            setattr(cfg.actor, term, None)

        cutoff = self.cutoff_distance
        near_clip = self.near_clip

        @dataclass
        class _CameraObsCfg(ObservationGroupConfig):
            enable_corruption: bool = False
            # Channel axis, so a second camera (or a mask) stacks onto
            # this one rather than being laid alongside it.
            concatenate_dim: int = 0
            wrist_depth = ObservationTermConfig(
                func=perception.camera_depth,
                scale=1.0,
                params={"sensor_name": CAMERA_SENSOR, "cutoff_distance": cutoff, "near_clip": near_clip},
            )

        setattr(cfg, CAMERA_GROUP, _CameraObsCfg())
        return cfg

    def _build_nn_config(self) -> NNConfig:
        return NNConfig(
            policy=PPOPolicyConfig(
                actor=VisionActorCfg(
                    trunk=MLPActorCfg(
                        activation=Activation.ELU,
                        init=OrthoInit(output_gain=1.0),
                        hidden_dims=[256, 256, 128],
                    ),
                    cnn=self.cnn,
                    image_groups=(CAMERA_GROUP,),
                ),
                critic=VisionCriticCfg(
                    trunk=MLPCriticCfg(
                        activation=Activation.ELU,
                        init=OrthoInit(output_gain=1.0),
                        hidden_dims=[256, 256, 128],
                    ),
                    cnn=self.cnn,
                    image_groups=(CAMERA_GROUP,),
                ),
                init_noise_std=1.0,
                distribution_type=DistributionType.GAUSSIAN,
                # A state-dependent std would have to read either the
                # image or the vector alone; mjlab's vision task uses a
                # learnable scalar and so do we.
                std_type=StdType.SCALAR,
            ),
        )
