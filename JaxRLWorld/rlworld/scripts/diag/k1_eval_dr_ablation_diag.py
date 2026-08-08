"""K1 eval reset_dr-removal ablation — is dof_damping default the fall cause?

Hypothesis under test
---------------------
The eval-defaults path strips every ``interval`` / ``reset_dr`` event
(evaluator.py:204-212). During training the ONLY writer of the plant's passive
``dof_damping`` is the ``reset_dr`` term ``dr_joint_damping`` (base.py:573-581,
``operation="abs"``, U(0, 1)). When eval removes it, ``dof_damping`` reverts to
the training-MJCF default (3 legs / 2 arms) — up to 3x the training upper bound —
so the loaded policy meets a stiffer plant than it ever saw and collapses.

This also explains the g1_recipe asymmetry: g1_recipe sets
``dr_interval_period_s = 10.0`` so its DR terms are ``interval_dr`` (NOT in the
eval removal set), so its damping survives eval; base (joystick.py) leaves the
terms at ``reset_dr`` and loses them.

What this prints
----------------
Phase 1: the actual ``dof_damping`` in the loaded eval env (train dist = U(0,1)).
Phase 2: rollout the loaded policy under two otherwise-identical conditions:
  A (as-is eval)         : dof_damping = whatever eval left it (expected ~3/2)
  B (damping forced 0.5) : dof_damping = train-center, everything else identical

Verdict: A collapses & B survives -> the reset_dr removal (dof_damping default)
IS the cause. Both collapse equally -> damping is NOT the cause; move the search
to action_distribution / action_clip / reward.

Run (JAX-backed -> jaxpy):
    jaxpy -m rlworld.scripts.diag.k1_eval_dr_ablation_diag \
        --wandb_run_path jsw7460/K1_Joystick/0wz9el80 --num_envs 64 --steps 500
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from rlworld.rl.envs.mdp.events.dr import unified as unified_dr
from rlworld.rl.evals.evaluator import PolicyEvaluator
from rlworld.scripts.diag.k1_damping_dr_diag import _read_damping


def _force_damping(env, value: float) -> None:
    """Force every actuated joint's passive dof_damping to a fixed absolute value."""
    env_ids = torch.arange(env.num_envs, device=env.device)
    unified_dr.randomize_joint_damping(env, env_ids=env_ids, damping_range=(value, value), operation="abs")


def _rollout(ev, n_steps: int, force_value: float | None) -> dict:
    """Deterministic rollout of the loaded policy; record first-termination step per env.

    ``force_value`` is None for condition A (leave eval's damping as-is) or a float
    for condition B (pin dof_damping to that value, re-applied after every reset
    since eval strips the reset_dr writer).
    """
    env = ev.env
    num = env.num_envs

    env.reset()
    if force_value is not None:
        _force_damping(env, force_value)

    obs = env.obs_manager.get_observation()
    robot_states = env.get_robot_state()

    first_fall = np.full(num, -1, dtype=np.int64)  # step index of first termination
    n_term = 0
    cmd_abs_sum = np.zeros(3, dtype=np.float64)  # running |command| to see if it even walks
    for t in range(n_steps):
        action = ev.policy.get_action(obs, robot_states)
        obs, _, terminated, truncated, _ = env.step(action)
        robot_states = env.get_robot_state()

        # get_commands_tensor() is [velocity(3) | gait_phase(2)]; velocity is first.
        cmd = env.command_manager.get_commands_tensor().detach().cpu().numpy()[:, :3]  # (N, 3)
        cmd_abs_sum += np.abs(cmd).mean(axis=0)

        term_np = terminated.detach().cpu().numpy().astype(bool)
        newly = term_np & (first_fall < 0)
        first_fall[newly] = t
        n_term += int(newly.sum())

        done = terminated | truncated
        if done.any():
            if force_value is not None:
                _force_damping(env, force_value)
            ev.policy.notify_reset(done.detach().cpu().numpy())

    fell = first_fall >= 0
    return {
        "fall_rate": float(fell.mean()),
        "n_term": n_term,
        "mean_first_fall": float(first_fall[fell].mean()) if fell.any() else float("nan"),
        "median_first_fall": float(np.median(first_fall[fell])) if fell.any() else float("nan"),
        "cmd_abs_mean": cmd_abs_sum / n_steps,  # mean |vx|,|vy|,|wz| over rollout
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--wandb_run_path", required=True)
    p.add_argument("--wandb_checkpoint_iter", type=int, default=None)
    p.add_argument("--num_envs", type=int, default=64)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--center", type=float, default=0.5, help="train-center damping for condition B")
    args = p.parse_args()

    ev = PolicyEvaluator(
        wandb_run_path=args.wandb_run_path,
        wandb_checkpoint_iter=args.wandb_checkpoint_iter,
        num_evals=1,
        use_logging=False,
        save_data=False,
        record_video=False,
        use_rich_display=False,
        extra_overrides={"env": {"num_envs": args.num_envs}},
    )
    sim = ev.env.sim_type
    print(f"\n=== loaded checkpoint | sim_type={sim} num_envs={ev.env.num_envs} ===")

    # ── Phase 1: actual dof_damping in the eval env (train dist = U(0, 1)) ──
    ev.env.reset()
    d = _read_damping(sim, ev.env)  # (num_envs, n_joint)
    print("\n[Phase 1] actual dof_damping in eval env  (training draw = U(0, 1))")
    print(f"  per-joint (env0): {np.array2string(d[0].numpy(), precision=3, max_line_width=120)}")
    print(f"  min={d.min():.3f}  max={d.max():.3f}  mean={d.mean():.3f}  std={d.std():.3f}")
    in_train = (float(d.min()) >= -1e-6) and (float(d.max()) <= 1.0 + 1e-6)
    print(f"  within train dist [0, 1]?  {in_train}")
    if not in_train:
        print("  -> dof_damping ESCAPED the train dist (expected MJCF default 3 legs / 2 arms).")
        print("     eval-defaults stripped reset_dr, so the abs-DR writer never ran.")
    else:
        print("  -> dof_damping is inside the train dist; damping hypothesis is already weakened.")

    # ── Phase 2: rollout ablation (damping is the ONLY difference between A and B) ──
    print(f"\n[Phase 2] rollout ablation  (steps={args.steps}, num_envs={ev.env.num_envs})")
    a = _rollout(ev, args.steps, force_value=None)
    b = _rollout(ev, args.steps, force_value=args.center)
    print(
        f"  A  (as-is eval, damping~{d.mean():.2f}) : fall_rate={a['fall_rate']:.3f}  "
        f"n_term={a['n_term']}  mean_first_fall={a['mean_first_fall']:.1f}  median={a['median_first_fall']:.1f}"
    )
    print(
        f"  B  (damping forced {args.center})        : fall_rate={b['fall_rate']:.3f}  "
        f"n_term={b['n_term']}  mean_first_fall={b['mean_first_fall']:.1f}  median={b['median_first_fall']:.1f}"
    )
    print(
        f"  command |vx,vy,wz| mean over rollout (A): {np.array2string(a['cmd_abs_mean'], precision=3)}  "
        "(near 0 -> policy is basically standing, not walking; fall stats are meaningless)"
    )

    print("\n[Verdict]")
    if a["fall_rate"] > 0.5 and b["fall_rate"] < a["fall_rate"] * 0.5:
        print("  A collapses, B survives -> dof_damping default (reset_dr removal) IS the cause.")
    elif a["fall_rate"] > 0.5 and b["fall_rate"] > 0.5:
        print("  Both collapse -> dof_damping is NOT the cause. Move to action_distribution / clip / reward.")
    else:
        print("  Inconclusive at this fall_rate; inspect the numbers above.")


if __name__ == "__main__":
    main()
