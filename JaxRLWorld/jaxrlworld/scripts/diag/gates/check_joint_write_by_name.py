"""Gate: ``set_joint_state_by_name`` poses a joint outside the action set.

The robot state writers' DOF writes cover the actuated joints the action
manager indexes. The YAM's gripper has a second finger that an MJCF
equality constraint drives from the actuated one and that is deliberately
not actuated, so a reset leaves it wherever the solver had it; the
constraint then takes ~100 ms to drag the pair to agreement, and the
actuated finger is dragged half-way to meet it. ``set_joint_state_by_name``
writes such a joint directly at reset.

Observable without reading the passive joint: close the actuated finger
to ``Q_CLOSED`` at reset and hold its target there through the position
action. With the mirrored finger ALSO written to ``-Q_CLOSED`` the pair
agrees from the first step and the actuated finger stays put; left where
the solver had it, the constraint pulls the actuated finger several
millimetres toward the mean within two control steps. The gate requires
the written case to move the actuated finger by less than ``HOLD_TOL``
and the unwritten case by more than ``LAG_MIN``, so it also fails if the
constraint is ever made instantaneous (then the writer would be
unobservable and this check meaningless) and names it.

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
HOLD_TOL = 0.0005
LAG_MIN = 0.002
STEPS = 2


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

    def finger() -> torch.Tensor:
        env._invalidate_cache()
        return env.get_entity_data("robot").joint_pos[:, col]

    def pose_actuated() -> None:
        q = env.get_entity_data("robot").joint_pos.clone()
        q[:, col] = Q_CLOSED
        writer.set_dof_state(q, torch.zeros_like(q), env_ids=ids)

    moved = {}
    for written in (False, True):
        env.reset()
        env._invalidate_cache()
        pose_actuated()
        if written:
            writer.set_joint_state_by_name(
                (YAM_MIRRORED_GRIPPER_JOINT,),
                torch.full((num_envs, 1), -Q_CLOSED, device=env.device),
                zeros,
                env_ids=ids,
            )
        writer.eval_fk(env_ids=ids)
        q0 = finger()
        for _ in range(STEPS):
            env.step(closed_hold_action())
        moved[written] = float((finger() - q0).abs().max())
    ok = moved[True] < HOLD_TOL and moved[False] > LAG_MIN
    print(
        f"[{sim}] actuated finger moved over {STEPS} steps: mirrored finger written {moved[True] * 1e3:.2f} mm "
        f"(< {HOLD_TOL * 1e3:.1f}), left to the constraint {moved[False] * 1e3:.2f} mm (> {LAG_MIN * 1e3:.1f}) "
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
