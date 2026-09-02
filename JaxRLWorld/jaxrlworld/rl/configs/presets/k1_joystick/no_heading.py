"""K1 G1-recipe variant with the absolute-heading command turned off.

Identical to :class:`K1G1RecipeConfig` except that the yaw command is
sampled from ``ang_vel_range`` instead of being generated from the heading
error. Exists to measure what heading control is actually buying: it was
added to stop a yaw drift of ~2.5 deg/s under wz=0 (k1_yaw_drift_diag),
and this preset is the control that shows what the policy does without it.

Train:
    jaxpy -m jaxrlworld.scripts.k1.newton.joystick_no_heading
"""

from dataclasses import dataclass

from .g1_recipe import K1G1RecipeConfig


@dataclass
class K1NoHeadingConfig(K1G1RecipeConfig):
    """G1 recipe with rate-only yaw commands."""

    heading_command: bool = False

    _RUN_NAMES = {
        "newton": "K1_Newton_G1Recipe_NoHeading",
        "mujoco": "K1_Mujoco_G1Recipe_NoHeading",
        "genesis": "K1_Genesis_G1Recipe_NoHeading",
    }
