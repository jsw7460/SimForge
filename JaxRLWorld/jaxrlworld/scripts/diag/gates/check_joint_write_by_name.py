"""Gate: ``set_joint_state_by_name`` poses a joint outside the action set.

The robot state writers' DOF writes cover the actuated joints the action
manager indexes. The YAM's gripper has a second finger that an MJCF
equality constraint drives from the actuated one and that is deliberately
not actuated, so a reset leaves it wherever the solver had it; the
constraint then takes ~100 ms to drag the pair to agreement, and the
actuated finger is dragged half-way to meet it. ``set_joint_state_by_name``
writes such a joint directly at reset.

Observable through the finger BODIES, which every backend reports: with
the actuated finger closed to ``Q_CLOSED``, the distance between the two
finger bodies says where the mirrored finger is. The gate first lets the
pair settle at ``Q_CLOSED`` (the reference) and then at ``Q_FAR``, so the
mirrored finger is parked far from the reference; a reset does not touch
a joint outside the action set on Newton, Genesis or PhysX, and mjlab
puts it at the model's zero, so either way the next episode starts with
it wrong. Posing the actuated finger alone must then leave the pair
millimetres off the reference after the first step; writing the mirrored
finger too must put it on the reference. Both are required, so the gate
also notices if the constraint is ever made instantaneous and names it.

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
Q_FAR = 0.030
MATCH_TOL = 0.0005
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

    def hold_action(q: float) -> torch.Tensor:
        act = torch.zeros(num_envs, env.act_manager.num_actions, device=env.device)
        act[:, col] = (q - home) / YAM_ACTION_SCALE[YAM_GRIPPER_JOINT]
        return act

    def finger_body_distance() -> torch.Tensor:
        """After one control step: a write marks kinematics stale and a backend
        may refresh body poses only when it steps, so the read follows a step.
        The constraint moves a lagging mirrored finger ~5 mm in that step,
        against the ~27 mm it has to make up, so the lag stays legible."""
        env.step(hold_action(Q_CLOSED))
        env._invalidate_cache()
        pads = env.get_entity_data("robot").body_pos_w(FINGER_BODIES)
        return torch.norm(pads[:, 0] - pads[:, 1], dim=-1)

    def pose_actuated(q_finger: float) -> None:
        q = env.get_entity_data("robot").joint_pos.clone()
        q[:, col] = q_finger
        writer.set_dof_state(q, torch.zeros_like(q), env_ids=ids)

    def settle(q_finger: float) -> None:
        env.reset()
        pose_actuated(q_finger)
        writer.eval_fk(env_ids=ids)
        for _ in range(SETTLE_STEPS):
            env.step(hold_action(q_finger))

    # The pair's distance once the constraint has settled with the actuated finger held closed.
    settle(Q_CLOSED)
    settled = finger_body_distance()

    # Park the mirrored finger far away, then start an episode closing the actuated one alone.
    settle(Q_FAR)
    env.reset()
    pose_actuated(Q_CLOSED)
    writer.eval_fk(env_ids=ids)
    lag = (finger_body_distance() - settled).abs().max()

    # Written: the pair agrees from the first step.
    settle(Q_FAR)
    env.reset()
    pose_actuated(Q_CLOSED)
    writer.set_joint_state_by_name(
        (YAM_MIRRORED_GRIPPER_JOINT,), torch.full((num_envs, 1), -Q_CLOSED, device=env.device), zeros, env_ids=ids
    )
    writer.eval_fk(env_ids=ids)
    written = (finger_body_distance() - settled).abs().max()

    ok = float(written) < MATCH_TOL and float(lag) > LAG_MIN
    print(
        f"[{sim}] finger bodies' distance against the settled pair ({float(settled.mean()) * 1e3:.2f} mm): "
        f"mirrored finger written {float(written) * 1e3:.2f} mm off (< {MATCH_TOL * 1e3:.1f}), "
        f"left to the solver {float(lag) * 1e3:.2f} mm off (> {LAG_MIN * 1e3:.1f}) -> {'PASS' if ok else 'FAIL'}"
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
