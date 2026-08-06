"""K1 G1-recipe variant with the measured actuator dynamic losses.

Same task/rewards/DR as :class:`K1G1RecipeConfig`; only the actuator
model differs — the two dynamic-delivery deficits measured from the
real-robot walking logs are baked into the robot config:

* ``tau_lpf_time_constant``: first-order torque lag. The hip-pitch
  motors track their PD torque with a ~20-40 ms bandwidth (regressing
  logged torque against the commanded PD law improves markedly under a
  low-pass with that time constant); other groups showed no lag.
* ``dyn_gain``: velocity-gated transmission efficiency. At standstill
  the joints deliver the full commanded torque, but during motion the
  logged torque is only a fraction of the PD law — most severely at the
  knees (~0.4) and sagittal/ankle joints. Arms/head showed no evidence
  and stay lossless.

Values are per actuator group (left/right tied). They are measured
point estimates from log regressions, good enough to train a first
policy that behaves reasonably on hardware; replay-based
identification can refine them later.

Train:
    python -m rlworld.scripts.k1.mujoco.joystick_dyn_loss
"""

from dataclasses import dataclass, field

from rlworld.rl.configs.robots.k1 import K1Config, _pattern_dict

from .g1_recipe import K1G1RecipeConfig

# Measured dynamic-loss table (see module docstring). Group keys follow
# the robot config's ``_GROUPS`` partition.
_TAU_LPF_S = _pattern_dict(
    {
        "head": 0.0,
        "shoulder": 0.0,
        "elbow": 0.0,
        "hip_pitch": 0.03,
        "hip_roll": 0.0,
        "hip_yaw": 0.0,
        "knee": 0.0,
        "ankle": 0.0,
    }
)
_DYN_GAIN = _pattern_dict(
    {
        "head": 1.0,
        "shoulder": 1.0,
        "elbow": 1.0,
        "hip_pitch": 0.45,
        "hip_roll": 0.7,
        "hip_yaw": 0.75,
        "knee": 0.4,
        "ankle": 0.5,
    }
)


def _dyn_loss_robot() -> K1Config:
    r = K1Config()
    r.tau_lpf_time_constant = dict(_TAU_LPF_S)
    r.dyn_gain = dict(_DYN_GAIN)
    r.dyn_gain_velocity = 0.5
    return r


@dataclass
class K1DynLossConfig(K1G1RecipeConfig):
    """G1 recipe on the measured dynamic-loss actuator world."""

    robot: K1Config = field(default_factory=_dyn_loss_robot)
    _RUN_NAMES = {
        "newton": "K1_Newton_DynLoss",
        "mujoco": "K1_Mujoco_DynLoss",
        "genesis": "K1_Genesis_DynLoss",
    }
