"""K1 G1-recipe (mirror) variant with log-replay calibrated actuator
parameters.

Same task/rewards/DR/mirror loss as :class:`K1G1RecipeConfig`;
only the plant differs. The values come from replaying real K1 logs in
Newton and fitting the plant so the replayed states match the hardware
states — judged purely on hardware-observable kinematics (encoders +
IMU), no torque-sensor channel. The current table is the fit from a
10 s optimized excitation-schedule collection: it cross-predicts the
held-out 102 s walking session BETTER (-14% vs nominal) than the
walking session's own fit does in-sample (-11%), so it is the best
plant estimate available.

Calibrated (the two terms below carry the whole fit):

* ``armature`` — the legs run 30-45% above the Booster reference
  values (knee 0.127 vs 0.096) and the arm reflected inertia is far
  above the reference 0.001 (shoulder-pitch 0.055 — free-swinging arms
  are the cleanest axis in the fit; the reference was an order-of-
  magnitude understatement). Left/right averaged — symmetrization was
  A/B-verified to cost ~0.5%p of replay fit.
* ``tau_lpf_time_constant`` — first-order torque lag per actuator
  group. Sub-physics-step for the sagittal leg groups; head ~11 ms,
  shoulder ~16 ms, hip_roll ~9 ms are the visible lags.

Explicitly NOT changed (the replay fit found no signal): ``dyn_gain``
(velocity-gated efficiency came out 0.95-1.0 — the walking-time torque
deficit the firmware sensor reports is a measurement artifact, not a
plant property), joint friction (stays at the DR default), kp/kd (the
commanded physical gains, statically verified on hardware).

The existing armature DR (x U(1.0, 1.05) at reset) keeps applying on
top of these centers.

Train:
    jaxpy -m jaxrlworld.scripts.k1.newton.joystick_calib
"""

from dataclasses import dataclass, field

from jaxrlworld.rl.configs.robots.k1 import K1Config, _pattern_dict

from .g1_recipe import K1G1RecipeConfig

# L/R-averaged calibrated armature [kg·m²] (head joints are unpaired
# and keep their own fitted values). Source: the excitation-schedule
# replay fit that cross-predicts every held-out walking log best; the
# symmetrized+lag form was A/B-verified within ~0.5%p of the full fit.
_CALIB_ARMATURE = {
    r".*AAHead_yaw": 0.000600,
    r".*Head_pitch": 0.005750,
    r".*_Shoulder_Pitch": 0.055078,
    r".*_Shoulder_Roll": 0.011444,
    r".*_Elbow_Pitch": 0.002902,
    r".*_Elbow_Yaw": 0.012227,
    r".*_Hip_Pitch": 0.070082,
    r".*_Hip_Roll": 0.046692,
    r".*_Hip_Yaw": 0.041349,
    r".*_Knee_Pitch": 0.126687,
    r".*_Ankle_Pitch": 0.074305,
    r".*_Ankle_Roll": 0.070259,
}

# Calibrated torque-lag time constants [s] per actuator group.
_CALIB_TAU_LPF_S = _pattern_dict(
    {
        "head": 0.010712,
        "shoulder": 0.015748,
        "elbow": 0.001611,
        "hip_pitch": 0.000592,
        "hip_roll": 0.008802,
        "hip_yaw": 0.003612,
        "knee": 0.004461,
        "ankle": 0.000081,
    }
)


def _calib_robot() -> K1Config:
    r = K1Config()
    r.armature = dict(_CALIB_ARMATURE)
    r.tau_lpf_time_constant = dict(_CALIB_TAU_LPF_S)
    return r


@dataclass
class K1CalibConfig(K1G1RecipeConfig):
    """G1 recipe + mirror loss on the log-replay calibrated plant.

    Extends the g1-recipe (which now enables the mirror loss by default):
    the policy that collected the calibration logs was a mirror-loss run,
    so the calibrated retrain keeps the same training lineage.
    """

    robot: K1Config = field(default_factory=_calib_robot)
    _RUN_NAMES = {
        "newton": "K1_Newton_Calib_Mirror",
        "mujoco": "K1_Mujoco_Calib_Mirror",
        "genesis": "K1_Genesis_Calib_Mirror",
    }
