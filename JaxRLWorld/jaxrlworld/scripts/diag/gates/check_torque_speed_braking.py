"""Does the torque-speed curve bound driving torque only, never braking torque?

The explicit PD actuator's T-N clip (``IdealPDActuator._clip_effort_tn``)
limits the torque a motor can deliver at speed. The limit is on torque that
drives the joint on in its direction of motion; torque against the motion is
bounded by the motor's rating alone. A clip that cuts both directions leaves
a joint past its velocity limit with no torque at all, which is how the K1
head yaw came to bounce between its position limits indefinitely.

Every expectation below is computed from the curve's definition on random
PD states, independently of the actuator, and compared exactly:

  1. braking (PD torque against the velocity) at any speed, including past
     ``velocity_limit``: applied == clip(raw, +-effort_limit), and never 0
     when raw is not;
  2. driving past ``velocity_limit``: applied == 0;
  3. |vel| <= knee_point: applied == clip(raw, +-effort_limit) either way;
  4. driving between the knee and the limit: |applied| <= the linear ramp;
  5. the change against the source's symmetric clip is confined to braking
     above the knee: everywhere else the two agree exactly.

    python -m jaxrlworld.scripts.diag.gates.check_torque_speed_braking
"""

from __future__ import annotations

import torch

from jaxrlworld.rl.actuators.actuator_cfg import IdealPDActuatorCfg
from jaxrlworld.rl.actuators.actuator_pd import IdealPDActuator

_JOINTS = ["head", "arm", "knee"]
_EFFORT = {"head": 6.0, "arm": 14.0, "knee": 112.0}
_VMAX = {"head": 7.85, "arm": 33.51, "knee": 12.57}
_KNEE = {"head": 7.85, "arm": 5.24, "knee": 2.09}


def _curve(vel: torch.Tensor, effort: torch.Tensor, vmax: torch.Tensor, knee: torch.Tensor) -> torch.Tensor:
    ramp = effort * (vmax - vel.abs()) / (vmax - knee).clamp(min=1e-6)
    return torch.minimum(ramp.clamp(min=0.0), effort)


def main() -> None:
    num_envs = 4096
    torch.manual_seed(0)
    cfg = IdealPDActuatorCfg(
        target_names_expr=(".*",),
        stiffness={"head": 4.0, "arm": 10.0, "knee": 80.0},
        damping={"head": 0.25, "arm": 0.45, "knee": 4.0},
        effort_limit=dict(_EFFORT),
        velocity_limit=dict(_VMAX),
        knee_point_velocity=dict(_KNEE),
    )
    act = IdealPDActuator(cfg, num_envs=num_envs, num_joints=len(_JOINTS), device="cpu", joint_names=_JOINTS)
    effort = torch.tensor([_EFFORT[j] for j in _JOINTS]).expand(num_envs, -1)
    vmax = torch.tensor([_VMAX[j] for j in _JOINTS]).expand(num_envs, -1)
    knee = torch.tensor([_KNEE[j] for j in _JOINTS]).expand(num_envs, -1)

    # Velocities spanning well past every limit, targets far enough from the
    # position for the PD torque to saturate in either direction.
    vel = (torch.rand(num_envs, len(_JOINTS)) * 2 - 1) * 2.0 * vmax
    pos = (torch.rand(num_envs, len(_JOINTS)) * 2 - 1) * 1.0
    target = (torch.rand(num_envs, len(_JOINTS)) * 2 - 1) * 3.0
    applied = act.compute(target, pos, vel)
    raw = act.computed_effort
    assert torch.equal(raw, act.stiffness * (target - pos) - act.damping * vel)

    braking = raw * vel < 0
    driving = ~braking
    past = vel.abs() >= vmax
    below_knee = vel.abs() <= knee
    between = ~below_knee & ~past
    box = torch.clip(raw, -effort, effort)
    ramp = _curve(vel, effort, vmax, knee)

    # 1. braking: the box clip alone, at any speed.
    assert torch.equal(applied[braking], box[braking]), "braking torque is not bounded by the rating alone"
    nonzero_raw = braking & past & (raw != 0)
    assert nonzero_raw.sum() > 100
    assert (applied[nonzero_raw] != 0).all(), "braking torque past the velocity limit was cut to zero"
    # 2. driving past the limit: nothing.
    assert (driving & past).sum() > 100
    assert (applied[driving & past] == 0).all(), "driving torque past the velocity limit is not zero"
    # 3. below the knee: the box clip, both ways.
    assert below_knee.sum() > 100
    assert torch.equal(applied[below_knee], box[below_knee])
    # 4. driving on the ramp: within the ramp.
    assert (driving & between).sum() > 100
    assert (applied[driving & between].abs() <= ramp[driving & between] + 1e-6).all()
    # 5. the source's symmetric clip agrees everywhere except braking above the knee.
    symmetric = torch.clip(raw, -ramp, ramp)
    changed = braking & ~below_knee
    assert torch.equal(applied[~changed], symmetric[~changed]), "the clip changed outside braking above the knee"
    assert (applied[changed].abs() >= symmetric[changed].abs()).all()
    assert (applied[changed] != symmetric[changed]).any()

    counts = {
        "braking past limit": int((braking & past).sum()),
        "driving past limit": int((driving & past).sum()),
        "below knee": int(below_knee.sum()),
        "driving on ramp": int((driving & between).sum()),
        "changed vs source": int(changed.sum()),
    }
    print("  " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    print("PASS")


if __name__ == "__main__":
    main()
