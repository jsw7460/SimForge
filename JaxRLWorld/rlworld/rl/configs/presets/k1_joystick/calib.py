"""K1 G1-recipe (mirror) variant with log-replay calibrated actuator
parameters.

Same task/rewards/DR/mirror loss as :class:`K1G1RecipeMirrorConfig`;
only the plant differs. The values come from replaying real K1 walking logs (the d6
session, collected with a policy from this recipe) in Newton and
fitting the plant so the replayed states match the hardware states —
judged purely on hardware-observable kinematics (encoders + IMU), no
torque-sensor channel.

Calibrated (replay fit improves 9.9% over the reference plant, and the
two terms below carry all of it):

* ``armature`` — the legs run 10-30% above the Booster reference
  values and the arm reflected inertia is ~20x the reference 0.001
  (0.019-0.022 kg·m², a typical geared-motor range; the reference was
  an order-of-magnitude understatement, and the free-swinging arms are
  the cleanest axis in the fit). Left/right averaged — the per-side
  splits (5-15%) cost only ~1% replay fit and the motors are
  identical per pair.
* ``tau_lpf_time_constant`` — first-order torque lag per actuator
  group. Small everywhere (sub-physics-step for most groups);
  shoulder ~17 ms and hip_roll ~11 ms are the only visible lags.

Explicitly NOT changed (the replay fit found no signal): ``dyn_gain``
(velocity-gated efficiency came out 0.95-1.0 — the walking-time torque
deficit the firmware sensor reports is a measurement artifact, not a
plant property), joint friction (stays at the DR default), kp/kd (the
commanded physical gains, statically verified on hardware).

The existing armature DR (x U(1.0, 1.05) at reset) keeps applying on
top of these centers.

Train:
    jaxpy -m rlworld.scripts.k1.newton.joystick_calib
"""

from dataclasses import dataclass, field

from rlworld.rl.configs.robots.k1 import K1Config, _pattern_dict

from .g1_recipe_mirror import K1G1RecipeMirrorConfig

# L/R-averaged calibrated armature [kg·m²] (head joints are unpaired
# and keep their own fitted values).
_CALIB_ARMATURE = {
    r".*AAHead_yaw": 0.001104,
    r".*Head_pitch": 0.003769,
    r".*_Shoulder_Pitch": 0.018872,
    r".*_Shoulder_Roll": 0.022372,
    r".*_Elbow_Pitch": 0.000656,
    r".*_Elbow_Yaw": 0.002851,
    r".*_Hip_Pitch": 0.052666,
    r".*_Hip_Roll": 0.042265,
    r".*_Hip_Yaw": 0.038228,
    r".*_Knee_Pitch": 0.111964,
    r".*_Ankle_Pitch": 0.069102,
    r".*_Ankle_Roll": 0.062764,
}

# Calibrated torque-lag time constants [s] per actuator group.
_CALIB_TAU_LPF_S = _pattern_dict(
    {
        "head": 0.006145,
        "shoulder": 0.017061,
        "elbow": 0.000923,
        "hip_pitch": 0.000946,
        "hip_roll": 0.010645,
        "hip_yaw": 0.001800,
        "knee": 0.005541,
        "ankle": 0.000774,
    }
)


def _calib_robot() -> K1Config:
    r = K1Config()
    r.armature = dict(_CALIB_ARMATURE)
    r.tau_lpf_time_constant = dict(_CALIB_TAU_LPF_S)
    return r


@dataclass
class K1CalibConfig(K1G1RecipeMirrorConfig):
    """G1 recipe + mirror loss on the log-replay calibrated plant.

    Extends the MIRROR recipe (not the plain one): the policy that
    collected the calibration logs was a mirror-loss run, so the
    calibrated retrain keeps the same training lineage.
    """

    robot: K1Config = field(default_factory=_calib_robot)
    _RUN_NAMES = {
        "newton": "K1_Newton_Calib_Mirror",
        "mujoco": "K1_Mujoco_Calib_Mirror",
        "genesis": "K1_Genesis_Calib_Mirror",
    }
