"""What the mirror data augmentation feeds PPO on the FIRST update, at a given init.

Mittal et al. (ICRA 2024, Eq. 6) weight a mirrored sample ``(L o, K a)`` by
``pi_theta(K a | L o) / pi_old(a | o)``: the new policy on the mirrored
pair over the old policy on the ORIGINAL pair. For the originals this ratio
is exactly 1 on the first update. For the mirrored rows it is the policy's
own left/right asymmetry, and the paper's Sec. IV-D reports that a policy
initialised with large weights is asymmetric enough for this to break
training, while a small-weight init starts "roughly equivalent to its
symmetric counterpart" and trains well.

This script measures that on the real recipe: it builds the runner at the
requested actor head gain, collects one rollout with the fresh policy, and
reports the mirrored-row ratio distribution, how much of the clipped
surrogate's gradient mass the mirrored rows carry, the policy's asymmetry,
then runs ONE augmented PPO update and prints its KL and clip fraction.

Run per gain (one env per process)::

    jaxpy -m jaxrlworld.scripts.diag.k1.symmetry_augmentation_ratios_diag --sim mujoco --output_gain 1.0
    jaxpy -m jaxrlworld.scripts.diag.k1.symmetry_augmentation_ratios_diag --sim mujoco --output_gain 0.01
"""

from __future__ import annotations

import argparse
import math

import jax
import jax.numpy as jnp
import numpy as np

from jaxrlworld.rl.algorithms.ppo.symmetry import mirror
from jaxrlworld.rl.configs.presets.k1_velocity.base import K1VelocityConfig
from jaxrlworld.rl.runners.base_runner import BaseRunner

CHUNK = 8192


def percentiles(x: np.ndarray) -> str:
    q = np.percentile(x, [0, 1, 5, 25, 50, 75, 95, 99, 100])
    return " ".join(f"{v:+.2f}" for v in q)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", choices=["mujoco", "newton", "genesis"], required=True)
    parser.add_argument("--num_envs", type=int, default=1024)
    parser.add_argument("--output_gain", type=float, required=True, help="actor head orthogonal gain")
    args = parser.parse_args()

    cfgs = K1VelocityConfig(sim_type=args.sim, num_envs=args.num_envs).build()
    # Set on the BUILT config, the same path the training CLI override takes,
    # so this does not depend on the preset file carrying the field.
    cfgs.algorithm.symmetry_cfg.use_data_augmentation = True
    cfgs.nn.policy.actor.init.output_gain = args.output_gain
    print(f"built symmetry_cfg: {cfgs.algorithm.symmetry_cfg}")
    runner = BaseRunner.create_with_env(cfgs)
    alg = runner.alg
    spec = alg.symmetry_spec
    if spec is None or not alg.symmetry_augment:
        raise SystemExit(
            "the runner did not enable data augmentation although the built config asks for it "
            f"(symmetry_spec={spec is not None}, symmetry_augment={alg.symmetry_augment}). "
            "JaxRLWorld/jaxrlworld/rl/runners/on_policy_runner.py on this machine predates "
            "use_data_augmentation: sync it together with rl/algorithms/ppo/{ppo,update,symmetry}.py and "
            "rl/configs/algorithms/ppo.py. Any training run launched with the flag on this machine trained "
            "WITHOUT augmentation."
        )
    clip = alg.clip_param
    print(
        f"actor head output gain {args.output_gain}, clip {clip}, num_envs {args.num_envs}, "
        f"steps/env {runner.num_steps_per_env}"
    )

    # One rollout with the fresh policy, exactly as training would collect it.
    obs = runner._get_initial_obs()
    data = runner._collect_experience(obs=obs, ep_infos=[])
    alg.compute_returns(data["last_obs"]["critic_obs"])
    flat = alg.storage.get_flat_batch()
    model = alg.train_state.model
    key = jax.random.PRNGKey(0)

    n = int(flat.actions.shape[0])
    adv = np.asarray(flat.advantages)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    old_lp = np.asarray(flat.old_log_probs)

    lp_orig, lp_mir, mu_orig, mu_mir_obs, sig = [], [], [], [], []
    for s in range(0, n, CHUNK):
        o = flat.actor_observations[s : s + CHUNK]
        a = flat.actions[s : s + CHUNK]
        lp, _, mu, sigma, _ = model.evaluate_actions(o, a, key=key)
        lpm, _, mum, _, _ = model.evaluate_actions(
            mirror(o, spec.actor_perm, spec.actor_sign), mirror(a, spec.action_perm, spec.action_sign), key=key
        )
        lp_orig.append(np.asarray(lp))
        lp_mir.append(np.asarray(lpm))
        mu_orig.append(np.asarray(mu))
        mu_mir_obs.append(np.asarray(mum))
        sig.append(np.asarray(sigma))
    lp_orig, lp_mir = np.concatenate(lp_orig), np.concatenate(lp_mir)
    mu_orig, mu_mir_obs, sig = np.concatenate(mu_orig), np.concatenate(mu_mir_obs), np.concatenate(sig)

    print("\n=== policy at init ===")
    k_mu = np.asarray(mirror(jnp.asarray(mu_orig), spec.action_perm, spec.action_sign))
    print(
        f"  RMS |mu(o)|                    {np.sqrt(np.mean(mu_orig**2)):.3f}   (raw action units; scale 0.12..0.38 rad per unit)"
    )
    print(
        f"  RMS |K mu(o) - mu(L o)|         {np.sqrt(np.mean((k_mu - mu_mir_obs)**2)):.3f}   (left/right asymmetry of the mean)"
    )
    print(f"  mean sigma                      {sig.mean():.3f}")

    log_r_orig = lp_orig - old_lp
    log_r_mir = lp_mir - old_lp
    print("\n=== importance ratios on the first update (log space) ===")
    print("           percentiles:   0%    1%    5%   25%   50%   75%   95%   99%  100%")
    print(f"  originals  log r     {percentiles(log_r_orig)}   (must be ~0: same policy)")
    print(f"  mirrored   log r     {percentiles(log_r_mir)}")
    r_mir = np.exp(np.clip(log_r_mir, -50, 50))
    lo, hi = math.log(1 - clip), math.log(1 + clip)
    print(f"  mirrored rows outside the clip band |r-1|>{clip}: {np.mean((log_r_mir < lo) | (log_r_mir > hi)):.1%}")
    print(
        f"  mirrored rows with r > 5: {np.mean(r_mir > 5):.2%}, r > 100: {np.mean(r_mir > 100):.3%}, max r {r_mir.max():.3g}"
    )

    # Gradient mass of the clipped surrogate. A row contributes a gradient
    # only where the unclipped branch is active: A>0 and r<1+clip, or A<0 and
    # r>1-clip; its magnitude is |A| r. Originals sit at r=1 on the first
    # update, so their mass is sum |A|. Mirrored rows with A<0 and r >> 1 are
    # the explosive ones the paper warns about.
    active_mir = ((adv > 0) & (r_mir < 1 + clip)) | ((adv < 0) & (r_mir > 1 - clip))
    mass_orig = float(np.abs(adv).sum())
    mass_mir = float((np.abs(adv) * r_mir)[active_mir].sum())
    explosive = (adv < 0) & (r_mir > 1 + clip)
    mass_expl = float((np.abs(adv) * r_mir)[explosive].sum())
    print("\n=== gradient mass of the policy surrogate on the first update ===")
    print(f"  originals (all at r=1):                      {mass_orig:12.1f}")
    print(
        f"  mirrored, gradient-active rows:              {mass_mir:12.1f}   ({mass_mir / mass_orig:.2f}x the originals)"
    )
    print(
        f"  of which A<0 and r>1+clip (pushed DOWN hard): {mass_expl:12.1f}   from {explosive.mean():.2%} of the rows"
    )
    top = np.sort((np.abs(adv) * r_mir)[active_mir])[::-1]
    if top.size:
        share = top[: max(1, top.size // 1000)].sum() / max(mass_mir, 1e-9)
        print(f"  share of the mirrored mass carried by its top 0.1% rows: {share:.1%}")

    # One augmented update, as training would do it.
    metrics = alg.update()
    print("\n=== one augmented PPO update ===")
    print(
        f"  approx_kl {metrics.kl.approx_kl:.4f}   clip_fraction {metrics.kl.clip_fraction:.3f}   "
        f"policy_loss {metrics.actor.policy_loss:.4f}   value_loss {metrics.critic.value_loss:.4f}"
    )
    model2 = alg.train_state.model
    mu2, mu2m = [], []
    for s in range(0, min(n, 4 * CHUNK), CHUNK):
        o = flat.actor_observations[s : s + CHUNK]
        a = flat.actions[s : s + CHUNK]
        mu2.append(np.asarray(model2.evaluate_actions(o, a, key=key)[2]))
        mu2m.append(np.asarray(model2.evaluate_actions(mirror(o, spec.actor_perm, spec.actor_sign), a, key=key)[2]))
    mu2, mu2m = np.concatenate(mu2), np.concatenate(mu2m)
    k_mu2 = np.asarray(mirror(jnp.asarray(mu2), spec.action_perm, spec.action_sign))
    print(
        f"  after the update: RMS |mu| {np.sqrt(np.mean(mu2**2)):.3f}, RMS asymmetry {np.sqrt(np.mean((k_mu2 - mu2m)**2)):.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
