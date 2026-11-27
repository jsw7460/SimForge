"""Gate: ``set_joint_state_by_name`` poses a joint outside the action set.

The robot state writers' DOF writes cover the actuated joints the action
manager indexes. The YAM's gripper has a second finger that an MJCF
equality constraint drives from the actuated one and that is deliberately
not actuated, so a reset leaves it wherever the solver had it; the
constraint then takes ~100 ms to drag the pair to agreement, and the
actuated finger is dragged half-way to meet it. ``set_joint_state_by_name``
writes such a joint directly at reset.

Observable through the finger BODIES, which every backend reports: with
the actuated finger closed to ``Q_CLOSED`` at reset, the distance between
the two finger bodies says where the mirrored finger is. Written to
``-Q_CLOSED`` it must equal, before any step, the distance the pair
settles to once the constraint has had time (``SETTLE_STEPS`` of holding
the actuated finger's target); left to the solver it must differ by
millimetres at the first step, which the gate also requires, so it
notices if the constraint is ever made instantaneous and names it.

Run on a GPU box::

    jaxpy -m jaxrlworld.scripts.diag.gates.check_joint_write_by_name --sim mujoco newton genesis
"""

from __future__ import annotations

import argparse

import torch

from jaxrlworld.rl.configs.presets.yam_lift.base import YamLiftConfig
from jaxrlworld.rl.configs.robots.yam import YAM_ACTION_SCALE, YAM_GRIPPER_JOINT, YAM_MIRRORED_GRIPPER_JOINT
from jaxrlworld.rl.configs.scene.entity_selector import SceneEntitySelector
from jaxrlworld.rl.runners import BaseRunner

Q_CLOSED = 0.005
MATCH_TOL = 0.0003
LAG_MIN = 0.002
SETTLE_STEPS = 25
FINGER_BODIES = ["lf_down", "rf_down"]


def run(sim: str, num_envs: int) -> bool:
    cfg = YamLiftConfig(sim_type=sim, num_envs=num_envs)
    env = BaseRunner._create_env_from_config(cfg.build())
    env.reset()
    env._invalidate_cache()
    writer = env.get_robot_state_writer("robot")
    col = int(env.resolve_selector(SceneEntitySelector(name="robot", joint_names=(YAM_GRIPPER_JOINT,))).joint_ids[0])
    home = cfg.robot.default_joint_angles[YAM_GRIPPER_JOINT]
    ids = torch.arange(num_envs, device=env.device)
    zeros = torch.zeros(num_envs, 1, device=env.device)

    def closed_hold_action() -> torch.Tensor:
        act = torch.zeros(num_envs, env.act_manager.num_actions, device=env.device)
        act[:, col] = (Q_CLOSED - home) / YAM_ACTION_SCALE[YAM_GRIPPER_JOINT]
        return act

    def finger_body_distance() -> torch.Tensor:
        env._invalidate_cache()
        pads = env.get_entity_data("robot").body_pos_w(FINGER_BODIES)
        return torch.norm(pads[:, 0] - pads[:, 1], dim=-1)

    def pose_actuated() -> None:
        q = env.get_entity_data("robot").joint_pos.clone()
        q[:, col] = Q_CLOSED
        writer.set_dof_state(q, torch.zeros_like(q), env_ids=ids)

    # The pair's distance once the constraint has settled with the actuated finger held closed.
    env.reset()
    pose_actuated()
    writer.eval_fk(env_ids=ids)
    for _ in range(SETTLE_STEPS):
        env.step(closed_hold_action())
    settled = finger_body_distance()

    # Left to the solver: the mirrored finger is wherever the reset left it.
    env.reset()
    pose_actuated()
    writer.eval_fk(env_ids=ids)
    lag = (finger_body_distance() - settled).abs().max()

    # Written: the pair agrees before any step.
    env.reset()
    pose_actuated()
    writer.set_joint_state_by_name(
        (YAM_MIRRORED_GRIPPER_JOINT,), torch.full((num_envs, 1), -Q_CLOSED, device=env.device), zeros, env_ids=ids
    )
    writer.eval_fk(env_ids=ids)
    written = (finger_body_distance() - settled).abs().max()

    ok = float(written) < MATCH_TOL and float(lag) > LAG_MIN
    print(
        f"[{sim}] finger bodies' distance against the settled pair: mirrored finger written {float(written) * 1e3:.2f} mm "
        f"(< {MATCH_TOL * 1e3:.1f}), left to the solver {float(lag) * 1e3:.2f} mm (> {LAG_MIN * 1e3:.1f}) "
        f"-> {'PASS' if ok else 'FAIL'}"
    )
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", nargs="+", default=["mujoco", "newton", "genesis"])
    ap.add_argument("--num-envs", type=int, default=4)
    args = ap.parse_args()
    results = {sim: run(sim, args.num_envs) for sim in args.sim}
    print("SUMMARY: " + ", ".join(f"{s} {'PASS' if v else 'FAIL'}" for s, v in results.items()))
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
