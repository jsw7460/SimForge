"""I2RT YAM — 6-DOF tabletop arm with a single-DOF parallel gripper.

The first fixed-base robot in this repo: the MJCF's root body carries no
free joint, so the arm is welded to the world and every preset built on
it must leave out the locomotion machinery that assumes a moving base
(root-velocity pushes, orientation terminations, base-velocity
observations — all constants here).

Gains are derived from each joint's effective inertia and a target
closed-loop natural frequency, the same construction the other robot
configs in this package use, with the motor specs (DM 4340 / DM 4310)
taken from the vendor driver.

**Gripper.** The MJCF drives ``left_finger`` only; ``right_finger``
mirrors it through an MJCF ``<equality>`` joint constraint
(``polycoef="0 -1 0 0 0"``). Declaring both would fight the constraint,
so ``right_finger`` is deliberately absent from the actuated set. The
finger joint is linear while the motor behind it is rotary, so its
armature / effort / velocity limits are the motor's specs reflected
through the crank transmission.
"""

from dataclasses import dataclass, field
from typing import Dict, List

from .base import RobotConfig

# ── Motor specs ──────────────────────────────────────────────────────────
# Reflected inertia at the motor output [kg·m²], and the vendor limits.
ARMATURE_DM_4340 = 0.032
ARMATURE_DM_4310 = 0.0018

EFFORT_DM_4340 = 28.0  # N·m
EFFORT_DM_4310 = 10.0  # N·m
VELOCITY_DM_4340 = 10.0  # rad/s
VELOCITY_DM_4310 = 30.0  # rad/s

# ── Per-joint effective inertia (seen by each actuator through the arm) ──
EFFECTIVE_INERTIAS: Dict[str, float] = {
    "joint1": 0.123153,
    "joint2": 0.277411,
    "joint3": 0.232763,
    "joint4": 0.030154,
    "joint5": 0.009126,
    "joint6": 0.002868,
    "left_finger": 2.781624,
}

_ARM_MOTOR = {
    "joint1": (ARMATURE_DM_4340, EFFORT_DM_4340, VELOCITY_DM_4340),
    "joint2": (ARMATURE_DM_4340, EFFORT_DM_4340, VELOCITY_DM_4340),
    "joint3": (ARMATURE_DM_4340, EFFORT_DM_4340, VELOCITY_DM_4340),
    "joint4": (ARMATURE_DM_4310, EFFORT_DM_4310, VELOCITY_DM_4310),
    "joint5": (ARMATURE_DM_4310, EFFORT_DM_4310, VELOCITY_DM_4310),
    "joint6": (ARMATURE_DM_4310, EFFORT_DM_4310, VELOCITY_DM_4310),
}

# ── PD gains from effective inertia ──────────────────────────────────────
NATURAL_FREQ = 2 * 2.0 * 3.1415926535  # 2 Hz
DAMPING_RATIO = 2.0

ARM_STIFFNESS = {n: EFFECTIVE_INERTIAS[n] * NATURAL_FREQ**2 for n in _ARM_MOTOR}
ARM_DAMPING = {n: 2.0 * DAMPING_RATIO * EFFECTIVE_INERTIAS[n] * NATURAL_FREQ for n in _ARM_MOTOR}

# ── Gripper transmission ─────────────────────────────────────────────────
# A DM 4310 turns a crank that pushes the finger. The mechanism spans 8°
# to 170° for 71 mm of travel; the operating range is 10° to 165°
# (2.7 rad), which is the stroke the ratio below is taken over. The true
# ratio varies with crank angle as r(θ) = r_crank·sin(θ) — this is its
# average, which is what a position-controlled joint needs.
GRIPPER_MOTOR_STROKE = 2.7  # rad
GRIPPER_LINEAR_STROKE = 0.071  # m
GRIPPER_TRANSMISSION_RATIO = GRIPPER_LINEAR_STROKE / GRIPPER_MOTOR_STROKE  # m/rad

# Rotary motor specs reflected onto a linear joint: mass m = I/r²,
# velocity v = r·ω, force F = τ/r.
ARMATURE_GRIPPER = ARMATURE_DM_4310 / GRIPPER_TRANSMISSION_RATIO**2
VELOCITY_GRIPPER = VELOCITY_DM_4310 * GRIPPER_TRANSMISSION_RATIO
# The full reflected force is far more than the mechanism should deliver;
# the hardware limits it too, and leaving it unclamped makes the contact
# solve stiff enough to be unstable.
EFFORT_GRIPPER = (EFFORT_DM_4310 / GRIPPER_TRANSMISSION_RATIO) * 0.1

NATURAL_FREQ_GRIPPER = 1.0 * 2.0 * 3.1415926535  # 1 Hz
STIFFNESS_GRIPPER = EFFECTIVE_INERTIAS["left_finger"] * NATURAL_FREQ_GRIPPER**2
DAMPING_GRIPPER = 2.0 * DAMPING_RATIO * EFFECTIVE_INERTIAS["left_finger"] * NATURAL_FREQ_GRIPPER

# ── Assembled per-joint tables ───────────────────────────────────────────
YAM_ARM_JOINTS: List[str] = list(_ARM_MOTOR)
YAM_GRIPPER_JOINT = "left_finger"
YAM_MIRRORED_GRIPPER_JOINT = "right_finger"  # driven by the MJCF equality

YAM_STIFFNESS: Dict[str, float] = {**ARM_STIFFNESS, YAM_GRIPPER_JOINT: STIFFNESS_GRIPPER}
YAM_DAMPING: Dict[str, float] = {**ARM_DAMPING, YAM_GRIPPER_JOINT: DAMPING_GRIPPER}
YAM_ARMATURE: Dict[str, float] = {
    **{n: m[0] for n, m in _ARM_MOTOR.items()},
    YAM_GRIPPER_JOINT: ARMATURE_GRIPPER,
}
YAM_EFFORT_LIMIT: Dict[str, float] = {
    **{n: m[1] for n, m in _ARM_MOTOR.items()},
    YAM_GRIPPER_JOINT: EFFORT_GRIPPER,
}
YAM_VELOCITY_LIMIT: Dict[str, float] = {
    **{n: m[2] for n, m in _ARM_MOTOR.items()},
    YAM_GRIPPER_JOINT: VELOCITY_GRIPPER,
}

# Action scale: a quarter of the joint's saturation deflection, so a
# unit action commands a displacement the actuator can actually hold.
YAM_ACTION_SCALE: Dict[str, float] = {n: 0.25 * YAM_EFFORT_LIMIT[n] / YAM_STIFFNESS[n] for n in YAM_STIFFNESS}

# Collision geoms are named ``*_collision`` in the MJCF. The finger pads
# (the six spheres per side) are the surfaces that actually grasp, so
# they get their own friction/condim treatment.
YAM_COLLISION_GEOMS = ".*_collision"
YAM_FINGER_PAD_GEOMS = "[lr]f_down(6|7|8|9|10|11)_collision"
YAM_GRIPPER_COLLISION_GEOMS = "(link6|[lr]f)_.*_collision"


@dataclass
class YamConfig(RobotConfig):
    """Configuration for the I2RT YAM 6-DOF arm."""

    name: str = "yam"
    # MJCF on every backend: Genesis only carries MuJoCo ``<equality>``
    # constraints through its MJCF path, and the gripper's finger coupling
    # is one. Loading the URDF would silently decouple the fingers.
    urdf_path: str | None = None
    mjcf_path: str | None = "./JaxRLWorld/rlworld/assets/i2rt_yam/xmls/yam.xml"

    base_link_name: str = "arm"
    base_init_height: float = 0.0

    # Welded to the world: the MJCF root body carries no free joint.
    floating: bool = False

    # Actuated joints only. The framework resolves a scene entity's
    # ``init_state.joint_pos`` against the ACTUATED joint list and rejects
    # any key that does not match, so the mirrored finger cannot be named
    # here — and it should not be: its value is not independent state but
    # whatever the equality constraint makes it.
    default_joint_angles: Dict[str, float] = field(
        default_factory=lambda: {
            "joint1": 0.0,
            "joint2": 1.047,
            "joint3": 1.05,
            "joint4": -0.9,
            "joint5": 0.0,
            "joint6": 0.0,
            "left_finger": 0.0375 / 2,
        }
    )

    # ``right_finger`` is absent on purpose — see the module docstring.
    actuated_dof_patterns: List[str] = field(default_factory=lambda: [*YAM_ARM_JOINTS, YAM_GRIPPER_JOINT])

    p_gains: Dict[str, float] = field(default_factory=lambda: dict(YAM_STIFFNESS))
    d_gains: Dict[str, float] = field(default_factory=lambda: dict(YAM_DAMPING))
    armature: Dict[str, float] = field(default_factory=lambda: dict(YAM_ARMATURE))

    def get_action_offset(self) -> Dict[str, float]:
        """Home pose for the actuated joints."""
        return {n: self.default_joint_angles[n] for n in self.actuated_dof_patterns}

    def mirrored_home_joint_pos(self) -> float:
        """Where the equality constraint puts the mirrored finger at home.

        Not settable through the config (see ``default_joint_angles``);
        stated here so a diag can check the constraint actually holds
        rather than assuming it.
        """
        return -self.default_joint_angles[YAM_GRIPPER_JOINT]
