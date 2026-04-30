"""Diagnostic for T1 motion-tracking eval mismatch.

Symptom: training shows ``time_out`` ≈ 1.0, but eval (mujoco→mujoco)
ALWAYS terminates with ``bad_anchor_pos`` or ``bad_ee_pos``. Other
non-tracking trainings eval correctly, so the bug is specific to the
T1 tracking pipeline (config, MotionCommand, transformer plumbing,
or normaliser/weight roundtrip).

This script loads the eval pipeline exactly like ``eval_mujoco.py``
(via ``PolicyEvaluator``) and dumps:

  1. Actor / critic / std-module weight L2-norm summaries
     (zero-norm or init-noise indicates weights were not loaded).
  2. Observation normalizer state (count, mean[:8], std[:8]).
     A small ``count`` (~1e-4) means the normaliser was NOT loaded,
     so eval obs go through identity-normalisation while training saw
     normalised obs — guaranteed train-eval mismatch.
  3. First-reset state: motion anchor pos vs. robot anchor pos,
     motion joint_pos[0] vs. robot joint_pos[0]. After reset the two
     should match within ~1 cm / few mrad. A large gap means
     ``MotionCommand._write_reference_state_to_sim`` did not actually
     place the robot at the motion's first frame.
  4. Step-by-step rollout: per-step anchor-error and per-body
     ee-error magnitudes, ``time_steps`` cursor, and termination
     reasons. The exact step where ``bad_anchor_pos`` /
     ``bad_motion_body_pos_z`` fires is logged.

Usage::

    uv run python -m rlworld.scripts.diag.t1_tracking_eval_diag \
        --policy-path outputs/models/2026-04-29/14-45-08/checkpoint_latest \
        --max-steps 200
"""
from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch

from rlworld.rl.evals import PolicyEvaluator


def _l2(x: jax.Array) -> float:
    return float(jnp.sqrt(jnp.sum(x.astype(jnp.float32) ** 2)))


def _summarise_module(name: str, module) -> None:
    """Print L2 norm and shape of every JAX-array leaf in ``module``."""
    leaves, _ = jax.tree_util.tree_flatten(module)
    arr_leaves = [l for l in leaves if isinstance(l, jax.Array)]
    print(f"\n[{name}]  ({len(arr_leaves)} array leaves)")
    total = 0.0
    n_params = 0
    for i, leaf in enumerate(arr_leaves):
        total += float(jnp.sum(leaf.astype(jnp.float32) ** 2))
        n_params += int(leaf.size)
        if i < 6:
            print(f"  leaf[{i}] shape={tuple(leaf.shape)} "
                  f"l2={_l2(leaf):.4f} mean={float(leaf.mean()):+.4f} "
                  f"std={float(leaf.std()):.4f}")
    if len(arr_leaves) > 6:
        print(f"  ... ({len(arr_leaves) - 6} more leaves)")
    print(f"  TOTAL params={n_params:,}  global_l2={float(np.sqrt(total)):.4f}")


def _dump_obs_normalizer(label: str, norm) -> None:
    if norm is None:
        print(f"\n[{label}] obs_normalizer is None (no normalisation in this run)")
        return
    count = float(np.array(norm.count))
    mean = np.array(norm.mean)
    var = np.array(norm.var)
    std = np.sqrt(np.maximum(var, 0.0))
    print(f"\n[{label}] obs_normalizer:")
    print(f"  count = {count:.6g}    "
          f"({'LOADED — valid stats' if count > 1.0 else 'NOT LOADED — fresh init (count≈1e-4)'})")
    print(f"  mean[:8] = {np.array2string(mean[:8], precision=4, suppress_small=True)}")
    print(f"  std[:8]  = {np.array2string(std[:8],  precision=4, suppress_small=True)}")
    print(f"  ||mean|| = {float(np.linalg.norm(mean)):.4f}, "
          f"||std||  = {float(np.linalg.norm(std)):.4f}")


def _get_motion_cmd(env):
    return env.command_manager.get_term("motion")


def _format_vec(t: torch.Tensor, prec: int = 3) -> str:
    return np.array2string(
        t.detach().cpu().numpy(), precision=prec, suppress_small=True,
    )


def _dump_first_reset(env) -> None:
    cmd = _get_motion_cmd(env)
    rd = env.get_robot_data("robot")

    print("\n══════════════ FIRST-RESET STATE ══════════════")
    print(f"  motion files       = {cmd.cfg.motion_files}")
    print(f"  rollover_teleport  = {cmd.cfg.rollover_teleport}")
    print(f"  sampling_mode      = {cmd.cfg.sampling_mode}")
    print(f"  num envs           = {env.num_envs}")
    print(f"  time_steps[0]      = {int(cmd.time_steps[0].item())}  "
          f"(motion length = {int(cmd._motion_lengths[0].item())})")

    # Motion's reference at current cursor
    motion_anchor_pos = cmd.body_pos_w[:, cmd.motion_anchor_body_index][0]
    motion_anchor_quat = cmd.body_quat_w[:, cmd.motion_anchor_body_index][0]
    motion_joint_pos = cmd.joint_pos[0]
    motion_joint_vel = cmd.joint_vel[0]

    # Robot's actual state right after the very first reset
    robot_anchor_pos = rd.body_pos_w_all[:, cmd.robot_anchor_body_index][0]
    robot_anchor_quat = rd.body_quat_w_all[:, cmd.robot_anchor_body_index][0]
    robot_joint_pos = rd.joint_pos[0]
    robot_joint_vel = rd.joint_vel[0]

    print(f"\n  motion anchor pos  = {_format_vec(motion_anchor_pos)}")
    print(f"  robot  anchor pos  = {_format_vec(robot_anchor_pos)}")
    delta_pos = (robot_anchor_pos - motion_anchor_pos).norm().item()
    print(f"  Δ‖anchor pos‖      = {delta_pos:.5f} m  "
          f"({'OK (< 1 cm)' if delta_pos < 0.01 else 'MISMATCH — reference state was NOT written into sim'})")

    print(f"\n  motion anchor quat = {_format_vec(motion_anchor_quat, 4)}")
    print(f"  robot  anchor quat = {_format_vec(robot_anchor_quat, 4)}")

    j_diff = (robot_joint_pos - motion_joint_pos).abs()
    print(f"\n  joint_pos|motion vs robot|  max-abs-diff = {j_diff.max().item():.5f}")
    print(f"  joint_vel|motion vs robot|  "
          f"max-abs-diff = {(robot_joint_vel - motion_joint_vel).abs().max().item():.5f}")
    print(f"  ‖robot_joint_vel‖           = {robot_joint_vel.norm().item():.5f}")

    # Per-body errors at first reset
    body_pos_motion = cmd.body_pos_w[0]
    body_pos_robot = rd.body_pos_w_all[0, cmd.body_indexes]
    pb_err = (body_pos_robot - body_pos_motion).norm(dim=-1)
    print(f"\n  per-body pos error after reset (m):")
    for i, name in enumerate(cmd.cfg.body_names):
        print(f"    [{i}] {name:<22s}  err = {pb_err[i].item():.5f}")


def _dump_step_metrics(env, step_idx: int, action: torch.Tensor) -> None:
    cmd = _get_motion_cmd(env)
    rd = env.get_robot_data("robot")

    motion_anchor_pos = cmd.body_pos_w[0, cmd.motion_anchor_body_index]
    robot_anchor_pos = rd.body_pos_w_all[0, cmd.robot_anchor_body_index]
    anchor_err = (robot_anchor_pos - motion_anchor_pos).norm().item()
    z_err = float((robot_anchor_pos[2] - motion_anchor_pos[2]).abs().item())

    body_pos_motion = cmd.body_pos_w[0]
    body_pos_robot = rd.body_pos_w_all[0, cmd.body_indexes]
    body_z_err = (body_pos_robot[:, 2] - body_pos_motion[:, 2]).abs().max().item()

    a = action[0]
    print(f"  step {step_idx:>4d}  ts={int(cmd.time_steps[0].item()):>4d}  "
          f"|a|={a.norm().item():.3f}  amax={a.abs().max().item():.3f}  "
          f"anchor_err={anchor_err:.4f}  anchor_z_err={z_err:.4f}  "
          f"body_z_err_max={body_z_err:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy-path",
        type=str,
        default="outputs/models/2026-04-29/14-45-08/checkpoint_latest/",
        help="Path to checkpoint directory (default matches eval_mujoco.py)",
    )
    parser.add_argument("--wandb", type=str, default=None,
                        help="Optional wandb run path (e.g. user/proj/run_id) — overrides --policy-path")
    parser.add_argument("--max-steps", type=int, default=200,
                        help="Max rollout steps to print")
    parser.add_argument("--print-every", type=int, default=1,
                        help="Print metrics every N steps")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("══════════════ T1 TRACKING EVAL DIAGNOSTIC ══════════════")
    print(f"  policy_path = {args.policy_path}")
    if args.wandb:
        print(f"  wandb       = {args.wandb}")

    overrides = {"env": {"num_envs": 1}}

    if args.wandb is not None:
        evaluator = PolicyEvaluator(
            wandb_run_path=args.wandb,
            num_evals=1,
            seed=args.seed,
            record_video=False,
            save_data=False,
            extra_overrides=overrides,
        )
    else:
        evaluator = PolicyEvaluator(
            policy_path=args.policy_path,
            num_evals=1,
            seed=args.seed,
            record_video=False,
            save_data=False,
            extra_overrides=overrides,
        )

    env = evaluator.env
    runner = evaluator.runner
    policy = evaluator.policy

    # ── 1. Weight summary ────────────────────────────────────────────
    model = runner.alg.train_state.model
    print("\n══════════════ MODEL WEIGHTS ══════════════")
    if hasattr(model, "actor"):
        _summarise_module("actor", model.actor)
    if hasattr(model, "critic"):
        _summarise_module("critic", model.critic)
    if hasattr(model, "std_module"):
        _summarise_module("std_module", model.std_module)

    # ── 2. Obs normalizer state ─────────────────────────────────────
    print("\n══════════════ OBSERVATION NORMALIZER ══════════════")
    _dump_obs_normalizer("actor", getattr(model, "actor_obs_normalizer", None))
    _dump_obs_normalizer("critic", getattr(model, "critic_obs_normalizer", None))

    # ── 3. First reset diagnostics ───────────────────────────────────
    # PolicyEvaluator already called env.reset() in __init__. Run a
    # fresh reset so we read state right after MotionCommand wrote
    # the reference frame.
    env.reset()
    _dump_first_reset(env)

    # ── 4. Step-by-step rollout ──────────────────────────────────────
    print("\n══════════════ ROLLOUT ══════════════")
    obs = env.obs_manager.get_observation()
    robot_states = env.get_robot_state()

    for step_idx in range(args.max_steps):
        action = policy.get_action(obs, robot_states)

        if step_idx % args.print_every == 0:
            _dump_step_metrics(env, step_idx, action)

        obs, rewards, terminated, truncated, infos = env.step(action)
        robot_states = env.get_robot_state()

        if bool(terminated[0].item()) or bool(truncated[0].item()):
            print(f"\n  TERMINATED at step {step_idx} "
                  f"(terminated={bool(terminated[0].item())}, "
                  f"truncated={bool(truncated[0].item())})")
            tm = getattr(env, "termination_manager", None)
            if tm is not None and hasattr(tm, "term_dones"):
                fired = [
                    name for name, mask in tm.term_dones.items()
                    if bool(mask[0].item())
                ]
                print(f"  term_dones fired this step: {fired}")
            break
    else:
        print(f"\n  Survived all {args.max_steps} steps without termination.")

    print("\n══════════════ DIAGNOSTIC COMPLETE ══════════════")


if __name__ == "__main__":
    main()
