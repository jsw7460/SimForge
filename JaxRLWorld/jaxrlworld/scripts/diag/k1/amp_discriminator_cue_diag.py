"""What does the motion-prior discriminator use to tell the policy from the data?

A discriminator at 0.99 accuracy on both sides gives the policy no style
gradient. That is expected while the policy cannot walk yet, and fatal if
the two sides differ by a cue the policy can never remove (a velocity
convention, a jitter level, a normalization gap). This diagnostic rolls the
checkpoint's policy out once, takes the very windows the discriminator was
trained on, and asks where the separation lives.

Sections (policy windows = one rollout of the loaded policy, expert windows
= the prior's reference set):

* **A  marginals** -- per feature block, mean / std / quantiles on both
  sides, and the per-feature standardized mean gap.
* **B  temporal texture** -- within each 10-frame window, the mean absolute
  change of every block between consecutive frames (jitter); a sim
  velocity read against an interpolated-mocap velocity shows up here.
* **C  linear separability** -- a logistic regression trained here on
  single frames (30-D) and on whole windows (300-D): if single frames
  already separate at 0.99, the cue is a marginal, not the gait.
* **D  the discriminator's own cue** -- the checkpoint's discriminator
  scores the policy windows, then again with ONE block (all 10 frames)
  replaced by that block from random expert windows; the block whose
  replacement raises the score most is the cue the discriminator learned.
  Also: the score of expert windows with one block replaced by policy
  values (which block, when wrong, costs the most).

Run on the GPU box::

    jaxpy -m jaxrlworld.scripts.diag.k1.amp_discriminator_cue_diag --checkpoint outputs/models/<date>/<time>/checkpoint_<it>

Without ``--checkpoint`` the fresh (untrained) policy and discriminator are used.
"""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
import numpy as np

from jaxrlworld.rl.algorithms.amp_ppo.amp_ppo import AmpPPO
from jaxrlworld.rl.algorithms.amp_ppo.discriminator import discriminator_logits, minibatch_std
from jaxrlworld.rl.algorithms.amp_ppo.expert_motion import amp_feature_layout
from jaxrlworld.rl.configs.presets.k1_velocity.base import K1VelocityConfig
from jaxrlworld.rl.runners.base_runner import BaseRunner
from jaxrlworld.rl.runners.on_policy_runner import OnPolicyRunner

SAMPLE = 20000


def logistic_regression(
    x_pos: np.ndarray, x_neg: np.ndarray, steps: int = 400, lr: float = 0.5
) -> tuple[float, np.ndarray]:
    """Standardized features, full-batch gradient descent; returns held-out accuracy and |w| per feature."""
    x = np.concatenate([x_pos, x_neg]).astype(np.float64)
    y = np.concatenate([np.ones(len(x_pos)), np.zeros(len(x_neg))])
    rng = np.random.default_rng(0)
    order = rng.permutation(len(x))
    x, y = x[order], y[order]
    n_train = int(0.8 * len(x))
    mu, sd = x[:n_train].mean(0), x[:n_train].std(0) + 1e-6
    x = (x - mu) / sd
    w = np.zeros(x.shape[1])
    b = 0.0
    for _ in range(steps):
        p = 1.0 / (1.0 + np.exp(-(x[:n_train] @ w + b)))
        g = p - y[:n_train]
        w -= lr * (x[:n_train].T @ g / n_train + 1e-3 * w)
        b -= lr * g.mean()
    pred = (x[n_train:] @ w + b) > 0
    return float((pred == (y[n_train:] > 0.5)).mean()), np.abs(w)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None, help="checkpoint directory; omit for the untrained policy")
    parser.add_argument("--sim", choices=["mujoco", "newton", "genesis"], default="mujoco")
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=500, help="policy steps run before the windows are collected")
    args = parser.parse_args()

    # The config is rebuilt from the preset here rather than restored from
    # the checkpoint's config.yaml: a run started before managers took their
    # own copies of the config's terms saved the selectors already resolved
    # (index tensors in place of name patterns), which the strict restore
    # refuses. The policy, discriminator and normalizer load from the
    # checkpoint either way; only the env is built from this preset.
    cfgs = K1VelocityConfig(sim_type=args.sim, num_envs=args.num_envs, use_amp=True).build()
    if args.checkpoint is None:
        runner = BaseRunner.create_with_env(cfgs, use_wandb=False)
    else:
        runner = OnPolicyRunner.load_checkpoint(args.checkpoint, cfgs=cfgs, use_wandb=False)
    alg = runner.alg
    if not isinstance(alg, AmpPPO):
        raise SystemExit(f"the loaded algorithm is {type(alg).__name__}, not AMP_PPO")
    prior = alg.amp
    layout = amp_feature_layout(runner.env.obs_manager, prior.cfg.amp_group)
    K, D = prior.num_steps, prior.feature_dim
    names = [t.name for t in layout]
    starts = np.concatenate([[0], np.cumsum([t.width for t in layout])])
    print(
        f"policy: {'fresh' if args.checkpoint is None else args.checkpoint}; K={K} D={D} blocks {list(zip(names, np.diff(starts)))}"
    )

    # Warm up first: right after the global reset every window is backfilled
    # with its reset frame, which IS a reference frame from the pose pool, so
    # the first rollout scores like expert data and says nothing about the
    # gait. Run the policy for ``--warmup`` steps, discarding those windows,
    # then keep one rollout of mid-episode windows.
    obs = runner._get_initial_obs()
    steps_done = 0
    while steps_done < args.warmup:
        data = runner._collect_experience(obs=obs, ep_infos=[])
        obs = type(obs)(data["last_obs"]["actor_obs"], data["last_obs"]["critic_obs"])
        steps_done += runner.num_steps_per_env
        alg.storage.clear()  # the update normally empties it between rollouts
        prior._current = []
    runner._collect_experience(obs=obs, ep_infos=[])
    print(f"warmed up {steps_done} steps ({steps_done * runner.env.control_dt:.1f} s) before collecting")
    policy = np.asarray(jnp.concatenate(prior._current, axis=0))  # (T*N, K*D)
    prior._current = []
    expert = np.asarray(prior.expert_windows)
    rng = np.random.default_rng(0)
    pol = policy[rng.choice(len(policy), min(SAMPLE, len(policy)), replace=False)]
    exp = expert[rng.choice(len(expert), min(SAMPLE, len(expert)), replace=False)]
    print(f"windows: policy {policy.shape} (sampled {len(pol)}), expert {expert.shape} (sampled {len(exp)})")
    pol_f, exp_f = pol.reshape(-1, K, D), exp.reshape(-1, K, D)

    print("\n=== A. marginals per block (last frame of each window) ===")
    print(
        f"    {'block':18s} {'policy mean':>12s} {'expert mean':>12s} {'policy std':>11s} {'expert std':>11s} {'max |Δmean|/std':>16s}"
    )
    for k, name in enumerate(names):
        a, b = pol_f[:, -1, starts[k] : starts[k + 1]], exp_f[:, -1, starts[k] : starts[k + 1]]
        gap = np.abs(a.mean(0) - b.mean(0)) / (b.std(0) + 1e-6)
        print(f"    {name:18s} {a.mean():+12.3f} {b.mean():+12.3f} {a.std():11.3f} {b.std():11.3f} {gap.max():16.2f}")
    for k, name in enumerate(names):
        a, b = pol_f[:, -1, starts[k] : starts[k + 1]], exp_f[:, -1, starts[k] : starts[k + 1]]
        qa, qb = np.percentile(np.abs(a), [50, 95, 99]), np.percentile(np.abs(b), [50, 95, 99])
        print(
            f"    |{name}| p50/p95/p99: policy {qa[0]:.3f}/{qa[1]:.3f}/{qa[2]:.3f}   expert {qb[0]:.3f}/{qb[1]:.3f}/{qb[2]:.3f}"
        )

    print("\n=== B. temporal texture: mean |x(t) - x(t-1)| inside a window, per block ===")
    for k, name in enumerate(names):
        a = np.abs(np.diff(pol_f[:, :, starts[k] : starts[k + 1]], axis=1)).mean()
        b = np.abs(np.diff(exp_f[:, :, starts[k] : starts[k + 1]], axis=1)).mean()
        # Second difference: curvature / jitter beyond a smooth trend.
        a2 = np.abs(np.diff(pol_f[:, :, starts[k] : starts[k + 1]], n=2, axis=1)).mean()
        b2 = np.abs(np.diff(exp_f[:, :, starts[k] : starts[k + 1]], n=2, axis=1)).mean()
        print(
            f"    {name:18s} Δ: policy {a:.4f} expert {b:.4f} (x{a / max(b, 1e-9):.2f})   Δ²: policy {a2:.4f} expert {b2:.4f} (x{a2 / max(b2, 1e-9):.2f})"
        )

    print("\n=== C. linear separability (held-out accuracy of a logistic regression trained here) ===")
    acc_frame, w_frame = logistic_regression(exp_f[:, -1], pol_f[:, -1])
    print(f"    single frame (D={D}): {acc_frame:.3f}")
    top = np.argsort(w_frame)[::-1][:6]
    print(
        "      most weighted features: "
        + ", ".join(
            f"{names[np.searchsorted(starts, i, side='right') - 1]}[{i - starts[np.searchsorted(starts, i, side='right') - 1]}]={w_frame[i]:.2f}"
            for i in top
        )
    )
    acc_win, _ = logistic_regression(exp, pol, steps=300)
    print(f"    whole window (K*D={K * D}): {acc_win:.3f}")
    for k, name in enumerate(names):
        acc_b, _ = logistic_regression(
            exp_f[:, :, starts[k] : starts[k + 1]].reshape(len(exp), -1),
            pol_f[:, :, starts[k] : starts[k + 1]].reshape(len(pol), -1),
            steps=300,
        )
        print(f"    window, {name} block only: {acc_b:.3f}")

    print("\n=== D. the trained discriminator's cue ===")
    norm = prior.normalizer

    def score(x: np.ndarray) -> np.ndarray:
        xj = jnp.asarray(x)
        xn = xj if norm is None else norm.normalize(xj)
        return np.asarray(discriminator_logits(prior.disc, xn))

    def minibatch_std_of(x: np.ndarray) -> jax.Array:
        xj = jnp.asarray(x)
        xn = xj if norm is None else norm.normalize(xj)
        return minibatch_std(prior.disc.trunk(xn))

    s_pol, s_exp = score(pol), score(exp)
    sig = lambda z: 1.0 / (1.0 + np.exp(-z))  # noqa: E731 - one-line local helper
    print(
        f"    pure batches (as the reward is computed: policy-only): policy logit {s_pol.mean():+.3f} (sigmoid {sig(s_pol).mean():.3f}), expert logit {s_exp.mean():+.3f} (sigmoid {sig(s_exp).mean():.3f})"
    )
    print(
        f"    style reward softplus(logit): policy {np.log1p(np.exp(s_pol)).mean():.4f}, expert {np.log1p(np.exp(s_exp)).mean():.4f}"
    )
    # The training batch is 2B policy (current + replay) + B expert; the
    # minibatch-std feature sees that composition, so the accuracies the
    # run logs are measured in it. Reproduce it with B = 4096.
    B = 4096
    mixed = np.concatenate([pol[rng.choice(len(pol), 2 * B)], exp[rng.choice(len(exp), B)]])
    s_mix = score(mixed)
    acc_p, acc_e = (sig(s_mix[: 2 * B]) < 0.5).mean(), (sig(s_mix[2 * B :]) >= 0.5).mean()
    print(
        f"    mixed batch (2B policy + B expert, as trained): policy sigmoid {sig(s_mix[:2 * B]).mean():.3f}, expert sigmoid {sig(s_mix[2 * B:]).mean():.3f}; accuracy policy {acc_p:.3f} expert {acc_e:.3f}"
    )
    print(
        f"    minibatch-std feature: policy-only batch {float(minibatch_std_of(pol)):.4f}, expert-only {float(minibatch_std_of(exp)):.4f}, mixed {float(minibatch_std_of(mixed)):.4f}; head weight on it {float(prior.disc.head.weight[0, -1]):+.3f}"
    )
    donor = exp[rng.choice(len(exp), len(pol))]
    print("    policy windows with one block taken from expert windows -> policy logit:")
    for k, name in enumerate(names):
        idx = np.concatenate([np.arange(f * D + starts[k], f * D + starts[k + 1]) for f in range(K)])
        swapped = pol.copy()
        swapped[:, idx] = donor[:, idx]
        s = score(swapped)
        print(f"      {name:18s} {s.mean():+.3f}  (Δ {s.mean() - s_pol.mean():+.3f})")
    donor_p = pol[rng.choice(len(pol), len(exp))]
    print("    expert windows with one block taken from policy windows -> expert logit:")
    for k, name in enumerate(names):
        idx = np.concatenate([np.arange(f * D + starts[k], f * D + starts[k + 1]) for f in range(K)])
        swapped = exp.copy()
        swapped[:, idx] = donor_p[:, idx]
        s = score(swapped)
        print(f"      {name:18s} {s.mean():+.3f}  (Δ {s.mean() - s_exp.mean():+.3f})")
    # Temporal structure alone: shuffle the frame order inside expert windows.
    shuffled = exp_f[:, rng.permutation(K)].reshape(len(exp), -1)
    print(
        f"    expert windows with frames shuffled in time -> logit {score(shuffled).mean():+.3f} (Δ {score(shuffled).mean() - s_exp.mean():+.3f}); a large drop means the discriminator reads the motion, not the pose"
    )

    print("\n=== E. the style signal the policy receives ===")
    # Spread: a reward that is the same for every policy window carries no
    # gradient however large or small it is.
    r_pol = np.log1p(np.exp(s_pol))
    q = np.percentile(r_pol, [1, 10, 50, 90, 99])
    print(
        f"    policy style reward percentiles p1/p10/p50/p90/p99: {q[0]:.3f} / {q[1]:.3f} / {q[2]:.3f} / {q[3]:.3f} / {q[4]:.3f}   (expert p50 {np.percentile(np.log1p(np.exp(s_exp)), 50):.3f})"
    )
    # What the high-reward policy windows look like: correlation of the
    # reward with the window's posture and speed statistics.
    k_g = names.index("projected_gravity")
    k_q = names.index("joint_pos")
    k_v = names.index("base_lin_vel")
    lean = np.linalg.norm(pol_f[:, :, starts[k_g] : starts[k_g] + 2], axis=2).mean(1)
    amp_q = np.abs(pol_f[:, :, starts[k_q] : starts[k_q + 1]]).mean((1, 2))
    speed = np.linalg.norm(pol_f[:, :, starts[k_v] : starts[k_v] + 2], axis=2).mean(1)
    jitter = np.abs(
        np.diff(pol_f[:, :, starts[names.index("joint_vel")] : starts[names.index("joint_vel") + 1]], n=2, axis=1)
    ).mean((1, 2))
    for label, stat in (
        ("trunk lean |g_xy|", lean),
        ("joint amplitude", amp_q),
        ("planar speed", speed),
        ("joint_vel jitter", jitter),
    ):
        c = np.corrcoef(stat, r_pol)[0, 1]
        hi, lo = stat[r_pol >= q[3]].mean(), stat[r_pol <= q[1]].mean()
        print(
            f"    corr(style reward, {label:18s}) = {c:+.3f};  top-10% windows {hi:.3f} vs bottom-10% {lo:.3f}  (expert {np.mean(np.linalg.norm(exp_f[:, :, starts[k_g] : starts[k_g] + 2], axis=2)) if label.startswith('trunk') else np.abs(exp_f[:, :, starts[k_q] : starts[k_q + 1]]).mean() if label.startswith('joint amp') else np.linalg.norm(exp_f[:, :, starts[k_v] : starts[k_v] + 2], axis=2).mean() if label.startswith('planar') else np.abs(np.diff(exp_f[:, :, starts[names.index('joint_vel')] : starts[names.index('joint_vel') + 1]], n=2, axis=1)).mean():.3f})"
        )

    # Input gradient of the reward: how much a unit change of each block
    # (in normalized units) moves the reward, on policy windows.
    def reward_of(x_norm_row, s_const):
        h = prior.disc.trunk(x_norm_row)
        if prior.disc.use_minibatch_std:
            h = jnp.concatenate([h, s_const[None]], axis=-1)
        return jax.nn.softplus(prior.disc.head(h)[0])

    sub = jnp.asarray(pol[:4096])
    sub_n = sub if norm is None else norm.normalize(sub)
    s_const = jax.lax.stop_gradient(minibatch_std(prior.disc.trunk(sub_n)))
    grads = np.asarray(jax.vmap(jax.grad(reward_of), in_axes=(0, None))(sub_n, s_const)).reshape(-1, K, D)
    total = np.linalg.norm(grads.reshape(len(grads), -1), axis=1)
    print(
        f"    |d reward / d window| on policy windows: p50 {np.percentile(total, 50):.4f}, p90 {np.percentile(total, 90):.4f} (normalized units)"
    )
    for k, name in enumerate(names):
        share = (grads[:, :, starts[k] : starts[k + 1]] ** 2).sum((1, 2)) / np.maximum(total**2, 1e-12)
        print(
            f"      gradient energy in {name:18s} {share.mean():6.1%}   last frame's share of the block {((grads[:, -1, starts[k] : starts[k + 1]] ** 2).sum(1) / np.maximum((grads[:, :, starts[k] : starts[k + 1]] ** 2).sum((1, 2)), 1e-12)).mean():6.1%}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
