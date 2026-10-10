"""Does a state write made after the physics step reach the same step's observation?

Two writers run between the physics step and the observation: interval
events (``push_by_setting_velocity`` writes a root velocity) and command
terms that teleport the robot (motion rollover, lifting resample). On a
backend whose derived quantities (``cvel``, ``xpos``) are only refreshed by
an explicit forward pass, the observation built right after such a write
still describes the pre-write state. This diag performs each write on every
environment, reads the quantities the observation terms read, and then runs
the backend's forward hook to show what the observation should have seen.

Usage (any backend; the tracking check needs a preset with a ``motion`` command):

    jaxpy -m jaxrlworld.scripts.diag.parity.check_state_write_visibility --preset go2_flat --sim mujoco
    jaxpy -m jaxrlworld.scripts.diag.parity.check_state_write_visibility --preset go2_flat --sim newton
    jaxpy -m jaxrlworld.scripts.diag.parity.check_state_write_visibility --preset g1_tracking --sim mujoco

A row's ``after write`` column equal to ``after forward`` means the write was
visible without a forward; a gap means the observation of that step is stale.
"""

from __future__ import annotations

import argparse

import torch

from jaxrlworld.rl.configs.scene.entity_selector import SceneEntitySelector
from jaxrlworld.rl.envs.mdp.events import common as common_ef
from jaxrlworld.rl.runners import BaseRunner
from jaxrlworld.scripts.diag.gates.check_all_presets import _PUBLIC, _load

_PUSH_X = 1.0  # m/s added to every env's root x velocity


def _fmt(t: torch.Tensor) -> str:
    return f"{float(t.abs().max()):.3e}"


def check_push(env, resolved) -> None:
    rd = env.get_entity_data("robot")
    env_ids = torch.arange(env.num_envs, device=env.device)
    before = rd.root_link_lin_vel_w.clone()
    common_ef.push_by_setting_velocity(env, env_ids, {"x": (_PUSH_X, _PUSH_X)}, asset_cfg=resolved)
    env._invalidate_cache()
    after_write = rd.root_link_lin_vel_w.clone()
    # What World.step does after a writer ran: forward only if the writer
    # marked the kinematics stale (mjlab), nothing on the eager backends.
    env._forward_if_kinematics_stale()
    env._invalidate_cache()
    after_hook = rd.root_link_lin_vel_w.clone()
    env._post_reset_forward()
    env._invalidate_cache()
    after_forward = rd.root_link_lin_vel_w.clone()
    expected = before.clone()
    expected[:, 0] += _PUSH_X
    print("push_by_setting_velocity (+1.0 m/s on root x, every env)")
    print(f"  |root_lin_vel_w after write   - expected| max : {_fmt(after_write - expected)}")
    print(
        f"  |root_lin_vel_w after hook    - expected| max : {_fmt(after_hook - expected)}  (World's stale-kinematics hook)"
    )
    print(f"  |root_lin_vel_w after forward - expected| max : {_fmt(after_forward - expected)}")
    print(f"  |after write - after forward|            max : {_fmt(after_write - after_forward)}")


def check_motion_teleport(env) -> None:
    if "motion" not in dict(env.command_manager.iter_terms()):
        print("motion teleport: preset has no 'motion' command term, skipped")
        return
    term = env.command_manager.get_term("motion")
    env_ids = torch.arange(env.num_envs, device=env.device)
    term._resample_command(env_ids, keep_motion_id=True)
    env._invalidate_cache()
    robot_after_write = term.robot_anchor_pos_w.clone()
    err_after_write = (term.anchor_pos_w - term.robot_anchor_pos_w).norm(dim=-1)
    env._forward_if_kinematics_stale()
    env._invalidate_cache()
    robot_after_hook = term.robot_anchor_pos_w.clone()
    env._post_reset_forward()
    env._invalidate_cache()
    robot_after_forward = term.robot_anchor_pos_w.clone()
    err_after_forward = (term.anchor_pos_w - term.robot_anchor_pos_w).norm(dim=-1)
    print("motion command teleport (_resample_command on every env)")
    print(f"  |robot_anchor_pos_w after write - after forward| max : {_fmt(robot_after_write - robot_after_forward)}")
    print(f"  |robot_anchor_pos_w after hook  - after forward| max : {_fmt(robot_after_hook - robot_after_forward)}")
    print(
        f"  anchor error |motion - robot|  mean after write {float(err_after_write.mean()):.3e}"
        f"  mean after forward {float(err_after_forward.mean()):.3e}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--preset", required=True, help="label from check_all_presets' table")
    ap.add_argument("--sim", required=True, choices=("mujoco", "newton", "genesis"))
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--settle-steps", type=int, default=5, help="random-action steps before the writes")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    loader = next(loader for label, loader, sims in _PUBLIC if label == args.preset and args.sim in sims)
    cfgs = _load(loader, args.sim, args.num_envs)
    env = BaseRunner.create_with_env(cfgs, use_wandb=False).env
    env.reset()
    gen = torch.Generator().manual_seed(args.seed)
    for _ in range(args.settle_steps):
        actions = (torch.randn((env.num_envs, env.num_actions), generator=gen) * 0.5).to(env.device)
        env.step(actions)

    print(f"{args.preset}:{args.sim}  num_envs={env.num_envs}")
    resolved = env.resolve_selector(SceneEntitySelector(name="robot"))
    check_push(env, resolved)
    check_motion_teleport(env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
