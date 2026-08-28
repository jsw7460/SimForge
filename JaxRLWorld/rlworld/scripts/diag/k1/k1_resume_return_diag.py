"""Why does a resumed (pretrained) run report scratch-level returns at first?

Hypothesis under test: the wandb/console "return" metric is the mean of
``EpisodeStatsCollector.return_history`` — a window holding COMPLETED
episodes only. Right after a resume every env starts a fresh episode; a
good policy survives the full episode (20 s = ``max_episode_length``
control steps), so its episodes first complete at iteration
``max_episode_length / num_steps_per_env`` (~42 with 1000/24). Until
then the window can only contain episodes that ended early — i.e. the
FAILURES — so the reported return is structurally scratch-like no
matter how good the policy is. Eval looks great because eval measures
the policy, not this census.

Three tests, each with an explicit verdict:

  1. LOAD IDENTITY — deserialise ``model.eqx`` again and compare every
     inexact-array leaf with the runner's loaded model (max |diff| must
     be exactly 0), and show the obs-normalizer count is a trained-run
     count (fresh init = 1e-4). Kills "checkpoint silently not loaded /
     normalizer reset" as causes.
  2. BEHAVIOR IN THE TRAINING WORLD — roll the loaded policy in the
     REAL training env (DR, pushes, delays all on), deterministic and
     stochastic, plus a fresh-init policy for reference. Reports
     fall(termination) counts and per-step reward. Kills "the policy is
     actually bad in the training world" (or confirms it).
  3. CENSUS REPLAY — reproduce the logged metric exactly: a fresh
     ``EpisodeStatsCollector`` fed by the stochastic loaded policy for
     N pseudo-iterations of ``num_steps_per_env`` steps, printing the
     window-mean (the reported "return") per iteration. Expected: the
     metric sits at failure-level until the first timeout wave lands at
     iteration ~``max_episode_length/num_steps_per_env``, then jumps.

Run (GPU box; checkpoint = the SAME dir passed as --runner.resume_path):

    jaxpy -m rlworld.scripts.diag.k1.k1_resume_return_diag \
        --checkpoint /path/to/checkpoint-iterXXXXX --preset calib
"""

from __future__ import annotations

import argparse
import os

import equinox as eqx
import jax
import numpy as np

from rlworld.rl.algorithms.ppo.ppo import PPO
from rlworld.rl.configs.presets.k1_joystick.calib import K1CalibConfig
from rlworld.rl.configs.presets.k1_joystick.g1_recipe import K1G1RecipeConfig
from rlworld.rl.envs.stats_collector import EpisodeStatsCollector
from rlworld.rl.modules.normalization import EmpiricalNormalization
from rlworld.rl.runners import BaseRunner
from rlworld.rl.utils.jax_utils import jax_to_torch, torch_to_jax
from rlworld.rl.utils.wandb_checkpoint import get_wandb_checkpoint

_PRESETS = {
    "calib": K1CalibConfig,
    "g1": K1G1RecipeConfig,  # g1-recipe now includes the mirror-symmetry loss
}


def _act_input(obs_dict) -> PPO.ActInput:
    return PPO.ActInput(
        actor_obs=torch_to_jax(obs_dict["actor"]),
        critic_obs=torch_to_jax(obs_dict["critic"]),
    )


def _rollout(runner, steps: int, deterministic: bool) -> dict:
    """Roll the runner's CURRENT policy in its training env."""
    env = runner.env
    env.reset()
    obs = env.get_observation()
    n_term = 0
    n_trunc = 0
    rew_sum = 0.0
    for _ in range(steps):
        actions = runner.alg.act(_act_input(obs), deterministic=deterministic)
        actions_torch = jax_to_torch(runner._process_action_for_env(actions), runner.device)
        obs, rewards, terminated, truncated, infos = env.step(actions_torch)
        n_term += int(terminated.sum())
        n_trunc += int(truncated.sum())
        rew_sum += float(rewards.mean())
    return {
        "terminations": n_term,
        "truncations": n_trunc,
        "per_step_reward": rew_sum / steps,
        "falls_per_env": n_term / env.num_envs,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default=None, help="checkpoint dir (the --runner.resume_path value)")
    p.add_argument(
        "--wandb_run_path",
        default=None,
        help="wandb run (e.g. jsw7460/K1_Joystick/w965hake) — " "downloads/caches its latest checkpoint",
    )
    p.add_argument("--wandb_checkpoint_iter", type=int, default=None)
    p.add_argument("--preset", choices=sorted(_PRESETS), default="calib")
    p.add_argument("--sim", default="newton")
    p.add_argument("--num_envs", type=int, default=512)
    p.add_argument(
        "--census_iters",
        type=int,
        default=60,
        help="pseudo-iterations for test 3 (must exceed " "max_episode_length/num_steps_per_env)",
    )
    args = p.parse_args()

    if (args.checkpoint is None) == (args.wandb_run_path is None):
        raise SystemExit("pass exactly one of --checkpoint / --wandb_run_path")
    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint, was_cached = get_wandb_checkpoint(
            wandb_run_path=args.wandb_run_path,
            iteration=args.wandb_checkpoint_iter,
        )
        print(f"[diag] wandb checkpoint ({'cached' if was_cached else 'downloaded'}): {checkpoint}")

    cfgs = _PRESETS[args.preset](sim_type=args.sim, num_envs=args.num_envs).build()
    cfgs.runner.resume_path = checkpoint
    runner = BaseRunner.create_with_env(cfgs, use_wandb=False)
    env = runner.env

    # ── TEST 1: load identity ─────────────────────────────────────────
    print("\n[TEST 1] checkpoint load identity")
    file_model = eqx.tree_deserialise_leaves(os.path.join(checkpoint, "model.eqx"), runner.alg.train_state.model)
    a = jax.tree_util.tree_leaves(eqx.filter(runner.alg.train_state.model, eqx.is_inexact_array))
    b = jax.tree_util.tree_leaves(eqx.filter(file_model, eqx.is_inexact_array))
    assert len(a) == len(b), f"leaf count mismatch: {len(a)} vs {len(b)}"
    max_diff = max(float(jax.numpy.abs(x - y).max()) for x, y in zip(a, b))
    print(f"  model leaves: {len(a)}, max |loaded - file| = {max_diff:.3e}")

    norms = [
        leaf
        for leaf in jax.tree_util.tree_leaves(
            runner.alg.train_state.model,
            is_leaf=lambda m: isinstance(m, EmpiricalNormalization),
        )
        if isinstance(leaf, EmpiricalNormalization)
    ]
    for i, nm in enumerate(norms):
        print(
            f"  normalizer[{i}]: count={float(nm.count):.3e}  "
            f"|mean|={float(jax.numpy.linalg.norm(nm.mean)):.3f}  (fresh: count=1e-4, |mean|=0)"
        )
    v1 = max_diff == 0.0 and all(float(nm.count) > 1.0 for nm in norms)
    print(f"  VERDICT 1: {'PASS — weights+normalizer restored bit-exactly' if v1 else 'FAIL — LOAD IS BROKEN'}")

    # ── TEST 2: behavior in the training world ────────────────────────
    steps = int(env.max_episode_length)
    print(
        f"\n[TEST 2] training-world rollout ({steps} steps = 1 episode length, "
        f"num_envs={env.num_envs}, DR/pushes ON)"
    )
    det = _rollout(runner, steps, deterministic=True)
    sto = _rollout(runner, steps, deterministic=False)
    fresh = type(runner)(env=env, cfgs=cfgs, use_wandb=False)
    scr = _rollout(fresh, steps, deterministic=False)
    for name, r in (("loaded/deterministic", det), ("loaded/stochastic", sto), ("fresh-init/stochastic", scr)):
        print(
            f"  {name:22s} falls/env={r['falls_per_env']:6.2f}  "
            f"terminations={r['terminations']:6d}  truncations={r['truncations']:6d}  "
            f"per-step reward={r['per_step_reward']:.4f}"
        )
    v2 = sto["falls_per_env"] < 0.5 * scr["falls_per_env"] and sto["per_step_reward"] > scr["per_step_reward"]
    print(
        f"  VERDICT 2: {'PASS — loaded policy is genuinely good in the training world' if v2 else 'FAIL — the policy really is scratch-like here (investigate world/plant mismatch)'}"
    )

    # ── TEST 3: census replay of the logged metric ────────────────────
    spe = int(cfgs.algorithm.num_steps_per_env)
    expect_at = steps / spe
    print(f"\n[TEST 3] logged-metric census replay: {args.census_iters} iterations x {spe} steps")
    print(f"  (survivor episodes can first complete at iteration ~{expect_at:.0f})")
    stats = EpisodeStatsCollector(
        num_envs=env.num_envs,
        max_episode_length=steps,
        device=env.device,
        gamma=float(cfgs.algorithm.gamma),
    )
    env.reset()
    obs = env.get_observation()
    early_mean = None
    for it in range(1, args.census_iters + 1):
        it_rew = 0.0
        for _ in range(spe):
            actions = runner.alg.act(_act_input(obs), deterministic=False)
            actions_torch = jax_to_torch(runner._process_action_for_env(actions), runner.device)
            obs, rewards, terminated, truncated, infos = env.step(actions_torch)
            stats.update(reward_info=infos["rewards_per_type"], dones=terminated | truncated)
            it_rew += float(rewards.mean())
        hist = list(stats.return_history)
        window_mean = float(np.mean(hist)) if hist else float("nan")
        if it == 5:
            early_mean = window_mean
        if it <= 10 or it % 5 == 0 or abs(it - expect_at) <= 2:
            print(
                f"  iter {it:3d}: REPORTED return={window_mean:9.3f}  "
                f"completed-episodes-in-window={len(hist):3d}  "
                f"per-step reward={it_rew / spe:.4f}"
            )
    late_mean = window_mean
    v3 = (
        early_mean is not None
        and np.isfinite(late_mean)
        and (not np.isfinite(early_mean) or late_mean > 2.0 * max(early_mean, 1e-9))
    )
    print(
        f"  VERDICT 3: {'PASS — early reported return is a completed-episode census artifact (failures only), true level appears once survivors hit timeout' if v3 else 'INCONCLUSIVE — early and late reported returns are similar; the census artifact does not explain it'}"
    )

    print("\n[SUMMARY]")
    print(f"  1 load identity : {'OK' if v1 else 'BROKEN'}")
    print(f"  2 policy quality: {'OK' if v2 else 'BAD IN TRAINING WORLD'}")
    print(f"  3 metric census : {'ARTIFACT CONFIRMED' if v3 else 'NOT THE (ONLY) CAUSE'}")


if __name__ == "__main__":
    main()
