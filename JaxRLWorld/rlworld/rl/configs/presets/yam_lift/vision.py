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

from mjlab.sensor import CameraSensorCfg

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
    camera_fovy: float = 58.0
    """The D405's vertical field of view, degrees."""

    cutoff_distance: float = 0.5
    """Far plane, metres. Everything past it saturates at 1.0, so this is
    the depth range the policy actually resolves — set it to the
    workspace, not the room."""

    cnn: CNNEncoderCfg = field(default_factory=CNNEncoderCfg)

    def build(self):
        if self.sim_type != "mujoco":
            raise NotImplementedError(
                f"The wrist camera is wired for mjlab only; {self.sim_type!r} has no camera sensor yet."
            )
        cfgs = super().build()
        cfgs.scene.sensors = tuple(cfgs.scene.sensors) + (
            CameraSensorCfg(
                name=CAMERA_SENSOR,
                camera_name=f"robot/{CAMERA_SENSOR}",
                width=self.camera_width,
                height=self.camera_height,
                fovy=self.camera_fovy,
                data_types=("depth",),
                enabled_geom_groups=(0, 3),
                use_shadows=False,
            ),
        )
        return cfgs

    def _build_observation_config(self):
        cfg = super()._build_observation_config()

        for term in PRIVILEGED_ACTOR_TERMS:
            setattr(cfg.actor, term, None)

        cutoff = self.cutoff_distance

        @dataclass
        class _CameraObsCfg(ObservationGroupConfig):
            enable_corruption: bool = False
            # Channel axis, so a second camera (or a mask) stacks onto
            # this one rather than being laid alongside it.
            concatenate_dim: int = 0
            wrist_depth = ObservationTermConfig(
                func=perception.camera_depth,
                scale=1.0,
                params={"sensor_name": CAMERA_SENSOR, "cutoff_distance": cutoff},
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
