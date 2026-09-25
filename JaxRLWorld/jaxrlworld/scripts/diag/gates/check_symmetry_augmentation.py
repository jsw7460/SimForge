"""Prove the PPO mirror data augmentation computes exactly what it claims.

No simulator. The mirror operator here is synthetic (a random involution
with signs), so nothing about a robot is assumed; what is under test is the
update arithmetic in ``compute_batch_loss(symmetry_augment=True)`` against
the rule it reproduces -- rsl_rl's ``Symmetry.augment_batch`` as used by
Mittal et al. (ICRA 2024):

    rows [:N]   originals          rows [N:]  mirrors (L o, L c, K a)
    policy loss, value loss        over all 2N rows, old log-prob / value /
                                   advantage / return of the ORIGINAL row
    entropy, analytical KL,        over the N originals only
    approx KL, bound penalty

Sections:

  1. Layout. The augmented arrays are the inputs followed by their exact
     mirrors; the scalars are the inputs repeated. Bitwise.
  2. Reference arithmetic. A float64 NumPy re-implementation of the rule on
     a tanh-MLP Gaussian stub reproduces every loss component the
     implementation reports.
  3. Same-graph arithmetic. The rule written out in JAX with the same model
     calls matches the implementation to float32 round-off, in value and in
     gradient. The flag OFF matches the plain loss the same way.
  4. Set invariance. Mirroring the whole input batch permutes the augmented
     row set, so policy and value loss cannot change.
  5. Equivariant fixed point. For a policy/critic that is exactly mirror
     equivariant, augmentation is a no-op: every mirrored row's surrogate
     equals its original's, so loss and gradient equal the plain ones.
  6. Real actor-critic. The framework's ``PPOActorCritic`` (obs normalizer
     on) runs the augmented loss on 2N rows and satisfies (4).

Run::

    jaxpy -m jaxrlworld.scripts.diag.gates.check_symmetry_augmentation
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from jaxrlworld.rl.algorithms.ppo import update as U
from jaxrlworld.rl.algorithms.ppo.symmetry import MirrorSpec, augment_minibatch, mirror
from jaxrlworld.rl.configs.common_config_classes import PPOPolicyConfig
from jaxrlworld.rl.modules.policies.ppo_ac import PPOActorCritic
from jaxrlworld.rl.storages.rollout_storage import RolloutBatch

OBS_A, OBS_C, ACT, N = 10, 14, 6, 64
CLIP, VLC, ENT = 0.2, 0.7, 0.013
_LOG_2PI = float(np.log(2.0 * np.pi))


class Checker:
    def __init__(self) -> None:
        self.fails: list[str] = []

    def __call__(self, name: str, ok: bool, detail: str = "") -> bool:
        ok = bool(ok)
        print(f"  [{'OK' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
        if not ok:
            self.fails.append(name)
        return ok


def maxdiff(a, b) -> float:
    return float(jnp.max(jnp.abs(jnp.asarray(a) - jnp.asarray(b))))


# ── synthetic involutive mirror ──────────────────────────────────────


def random_involution(key, dim: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """A random pairing permutation (some fixed points) with pair-consistent signs.

    Involutive by construction: perm[perm] == id and sign * sign[perm] == 1.
    """
    k1, k2, k3 = jax.random.split(key, 3)
    order = np.asarray(jax.random.permutation(k1, dim))
    perm = np.arange(dim)
    sign = np.ones(dim)
    n_pairs = dim // 3
    for i in range(n_pairs):
        a, b = int(order[2 * i]), int(order[2 * i + 1])
        perm[a], perm[b] = b, a
        s = -1.0 if float(jax.random.uniform(jax.random.fold_in(k2, i))) < 0.5 else 1.0
        sign[a] = sign[b] = s
    for j in order[2 * n_pairs :]:
        sign[int(j)] = -1.0 if float(jax.random.uniform(jax.random.fold_in(k3, int(j)))) < 0.5 else 1.0
    return jnp.asarray(perm, dtype=jnp.int32), jnp.asarray(sign, dtype=jnp.float32)


def make_spec(key) -> MirrorSpec:
    ka, kc, kj = jax.random.split(key, 3)
    ap, as_ = random_involution(ka, OBS_A)
    cp, cs = random_involution(kc, OBS_C)
    jp, js = random_involution(kj, ACT)
    return MirrorSpec(actor_perm=ap, actor_sign=as_, action_perm=jp, action_sign=js, critic_perm=cp, critic_sign=cs)


# ── stub actor-critic (protocol of compute_batch_loss) ───────────────


class StubAC(eqx.Module):
    """tanh-MLP Gaussian policy with state-independent std, tanh-MLP critic."""

    w1: jax.Array
    b1: jax.Array
    w2: jax.Array
    b2: jax.Array
    log_std: jax.Array
    vw1: jax.Array
    vb1: jax.Array
    vw2: jax.Array
    vb2: jax.Array

    def __init__(self, key):
        k = jax.random.split(key, 5)
        self.w1 = jax.random.normal(k[0], (OBS_A, 32)) * 0.3
        self.b1 = jax.random.normal(k[1], (32,)) * 0.1
        self.w2 = jax.random.normal(k[2], (32, ACT)) * 0.3
        self.b2 = jnp.zeros(ACT)
        self.log_std = jax.random.normal(k[3], (ACT,)) * 0.2
        self.vw1 = jax.random.normal(k[4], (OBS_C, 32)) * 0.3
        self.vb1 = jnp.zeros(32)
        self.vw2 = jax.random.normal(jax.random.fold_in(k[4], 1), (32, 1)) * 0.3
        self.vb2 = jnp.zeros(1)

    def mu(self, obs):
        return jnp.tanh(obs @ self.w1 + self.b1) @ self.w2 + self.b2

    def sigma(self, n: int):
        return jnp.broadcast_to(jnp.exp(self.log_std), (n, ACT))

    def evaluate_actions(self, actor_obs, actions, *, key=None):
        mu = self.mu(actor_obs)
        sigma = self.sigma(actions.shape[0])
        z = (actions - mu) / sigma
        log_probs = (-0.5 * z**2 - jnp.log(sigma) - 0.5 * _LOG_2PI).sum(-1)
        entropy = (0.5 * (1.0 + _LOG_2PI) + jnp.log(sigma)).sum(-1)
        return log_probs, entropy, mu, sigma, {}

    def evaluate_value(self, critic_obs):
        return jnp.tanh(critic_obs @ self.vw1 + self.vb1) @ self.vw2 + self.vb2, {}


class SymmetrizedAC(eqx.Module):
    """Exactly mirror-equivariant wrapper: mu(o) = (mu(o) + K mu(L o)) / 2,
    std symmetric across mirror pairs, V(c) = (V(c) + V(L c)) / 2."""

    base: StubAC
    spec: MirrorSpec = eqx.field(static=False)

    @property
    def _spec(self) -> MirrorSpec:
        # The operator is a constant of the construction, not a parameter:
        # differentiating through its sign arrays would break the identity
        # mu(L o) == K mu(o) at first order and fake a gradient mismatch.
        return jax.tree.map(jax.lax.stop_gradient, self.spec)

    def mu(self, obs):
        s = self._spec
        direct = self.base.mu(obs)
        mirrored = mirror(self.base.mu(mirror(obs, s.actor_perm, s.actor_sign)), s.action_perm, s.action_sign)
        return 0.5 * (direct + mirrored)

    def sigma(self, n: int):
        s = self._spec
        ls = 0.5 * (self.base.log_std + self.base.log_std[s.action_perm])
        return jnp.broadcast_to(jnp.exp(ls), (n, ACT))

    def evaluate_actions(self, actor_obs, actions, *, key=None):
        mu = self.mu(actor_obs)
        sigma = self.sigma(actions.shape[0])
        z = (actions - mu) / sigma
        log_probs = (-0.5 * z**2 - jnp.log(sigma) - 0.5 * _LOG_2PI).sum(-1)
        entropy = (0.5 * (1.0 + _LOG_2PI) + jnp.log(sigma)).sum(-1)
        return log_probs, entropy, mu, sigma, {}

    def evaluate_value(self, critic_obs):
        s = self._spec
        v, _ = self.base.evaluate_value(critic_obs)
        vm, _ = self.base.evaluate_value(mirror(critic_obs, s.critic_perm, s.critic_sign))
        return 0.5 * (v + vm), {}


# ── batch construction ───────────────────────────────────────────────


def make_batch(key, n: int = N) -> RolloutBatch:
    ks = jax.random.split(key, 9)
    return RolloutBatch(
        actor_observations=jax.random.normal(ks[0], (n, OBS_A)),
        critic_observations=jax.random.normal(ks[1], (n, OBS_C)),
        actions=jax.random.normal(ks[2], (n, ACT)),
        values=jax.random.normal(ks[3], (n,)),
        advantages=jax.random.normal(ks[4], (n,)) * 2.0,
        returns=jax.random.normal(ks[5], (n,)),
        old_log_probs=jax.random.normal(ks[6], (n,)) * 0.5 - 8.0,
        old_mu=jax.random.normal(ks[7], (n, ACT)),
        old_sigma=jnp.abs(jax.random.normal(ks[8], (n, ACT))) + 0.5,
    )


def mirror_batch(batch: RolloutBatch, spec: MirrorSpec) -> RolloutBatch:
    """The physically mirrored rollout: obs/actions/old_mu mirrored, old_sigma permuted."""
    return batch._replace(
        actor_observations=mirror(batch.actor_observations, spec.actor_perm, spec.actor_sign),
        critic_observations=mirror(batch.critic_observations, spec.critic_perm, spec.critic_sign),
        actions=mirror(batch.actions, spec.action_perm, spec.action_sign),
        old_mu=mirror(batch.old_mu, spec.action_perm, spec.action_sign),
        old_sigma=batch.old_sigma[..., spec.action_perm],
    )


def run_loss(model, batch: RolloutBatch, spec, augment: bool, key, bound: float = 0.0):
    params, static = eqx.partition(model, eqx.is_inexact_array)

    def loss_fn(p):
        return U.compute_batch_loss(p, static, batch, CLIP, VLC, ENT, True, True, key, spec, 0.0, bound, augment)

    (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    return loss, info, grads


def manual_loss_jax(model, batch: RolloutBatch, spec, augment: bool, key):
    """The rule written out with the same model calls, independent of update.py."""
    adv = batch.advantages
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    n = batch.actions.shape[0]
    ao, co, ac = batch.actor_observations, batch.critic_observations, batch.actions
    olp, ov, ret = batch.old_log_probs, batch.values, batch.returns
    if augment:
        ao = jnp.concatenate([ao, mirror(ao, spec.actor_perm, spec.actor_sign)])
        co = jnp.concatenate([co, mirror(co, spec.critic_perm, spec.critic_sign)])
        ac = jnp.concatenate([ac, mirror(ac, spec.action_perm, spec.action_sign)])
        olp, ov, adv, ret = (jnp.concatenate([x, x]) for x in (olp, ov, adv, ret))
    lp, ent, mu, sig, _ = model.evaluate_actions(ao, ac, key=key)
    v, _ = model.evaluate_value(co)
    v = v.squeeze(-1)
    ratio = jnp.exp(lp - olp)
    pl = jnp.maximum(-adv * ratio, -adv * jnp.clip(ratio, 1 - CLIP, 1 + CLIP)).mean()
    vc = ov + jnp.clip(v - ov, -CLIP, CLIP)
    vl = jnp.maximum((v - ret) ** 2, (vc - ret) ** 2).mean()
    e = ent[:n].mean()
    kl = (
        (
            jnp.log(sig[:n] / (batch.old_sigma + 1e-5) + 1e-5)
            + (batch.old_sigma**2 + (batch.old_mu - mu[:n]) ** 2) / (2.0 * sig[:n] ** 2)
            - 0.5
        )
        .sum(-1)
        .mean()
    )
    total = pl + VLC * vl - ENT * e
    return total, dict(policy_loss=pl, value_loss=vl, entropy=e, analytical_kl=kl)


def manual_loss_numpy(model: StubAC, batch: RolloutBatch, spec: MirrorSpec) -> dict:
    """rsl_rl's rule in float64 NumPy from the stub's raw weights."""
    f = lambda x: np.asarray(x, dtype=np.float64)  # noqa: E731
    P = lambda x: np.asarray(x)  # noqa: E731
    ap, as_ = P(spec.actor_perm), f(spec.actor_sign)
    cp, cs = P(spec.critic_perm), f(spec.critic_sign)
    jp, js = P(spec.action_perm), f(spec.action_sign)
    ao, co, ac = f(batch.actor_observations), f(batch.critic_observations), f(batch.actions)
    adv = f(batch.advantages)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    olp, ov, ret = f(batch.old_log_probs), f(batch.values), f(batch.returns)
    n = ac.shape[0]
    ao2 = np.concatenate([ao, ao[:, ap] * as_])
    co2 = np.concatenate([co, co[:, cp] * cs])
    ac2 = np.concatenate([ac, ac[:, jp] * js])
    olp2, ov2, adv2, ret2 = (np.concatenate([x, x]) for x in (olp, ov, adv, ret))
    w1, b1, w2, b2 = f(model.w1), f(model.b1), f(model.w2), f(model.b2)
    mu = np.tanh(ao2 @ w1 + b1) @ w2 + b2
    sig = np.exp(f(model.log_std))[None, :].repeat(2 * n, 0)
    z = (ac2 - mu) / sig
    lp = (-0.5 * z**2 - np.log(sig) - 0.5 * _LOG_2PI).sum(-1)
    ratio = np.exp(lp - olp2)
    pl = np.maximum(-adv2 * ratio, -adv2 * np.clip(ratio, 1 - CLIP, 1 + CLIP)).mean()
    v = (np.tanh(co2 @ f(model.vw1) + f(model.vb1)) @ f(model.vw2) + f(model.vb2))[:, 0]
    vc = ov2 + np.clip(v - ov2, -CLIP, CLIP)
    vl = np.maximum((v - ret2) ** 2, (vc - ret2) ** 2).mean()
    e = (0.5 * (1.0 + _LOG_2PI) + np.log(sig[:n])).sum(-1).mean()
    om, os_ = f(batch.old_mu), f(batch.old_sigma)
    kl = (
        (np.log(sig[:n] / (os_ + 1e-5) + 1e-5) + (os_**2 + (om - mu[:n]) ** 2) / (2.0 * sig[:n] ** 2) - 0.5)
        .sum(-1)
        .mean()
    )
    approx_kl = ((ratio[:n] - 1) - (lp[:n] - olp2[:n])).mean()  # originals only
    clip_frac = (np.abs(ratio - 1.0) > CLIP).mean()
    return dict(
        total=pl + VLC * vl - ENT * e,
        policy_loss=pl,
        value_loss=vl,
        entropy=e,
        analytical_kl=kl,
        approx_kl=approx_kl,
        clip_fraction=clip_frac,
    )


def grad_maxdiff(g1, g2) -> float:
    return max(
        float(jnp.max(jnp.abs(a - b))) for a, b in zip(jax.tree_util.tree_leaves(g1), jax.tree_util.tree_leaves(g2))
    )


# ── sections ─────────────────────────────────────────────────────────


def section_1_layout(chk, spec, batch):
    print("\n=== 1. augmented layout ===")
    out = augment_minibatch(
        spec,
        batch.actor_observations,
        batch.critic_observations,
        batch.actions,
        batch.old_log_probs,
        batch.values,
        batch.advantages,
        batch.returns,
    )
    ao, co, ac, olp, ov, adv, ret = out
    chk("2N rows everywhere", all(x.shape[0] == 2 * N for x in out), str([x.shape[0] for x in out]))
    chk(
        "rows [:N] are the inputs, bitwise",
        bool(jnp.array_equal(ao[:N], batch.actor_observations))
        and bool(jnp.array_equal(co[:N], batch.critic_observations))
        and bool(jnp.array_equal(ac[:N], batch.actions)),
    )
    chk(
        "rows [N:] are the exact mirrors (gather + sign), bitwise",
        bool(jnp.array_equal(ao[N:], batch.actor_observations[:, spec.actor_perm] * spec.actor_sign))
        and bool(jnp.array_equal(co[N:], batch.critic_observations[:, spec.critic_perm] * spec.critic_sign))
        and bool(jnp.array_equal(ac[N:], batch.actions[:, spec.action_perm] * spec.action_sign)),
    )
    chk(
        "old log-prob / value / advantage / return repeated, bitwise",
        all(
            bool(jnp.array_equal(x[:N], y)) and bool(jnp.array_equal(x[N:], y))
            for x, y in ((olp, batch.old_log_probs), (ov, batch.values), (adv, batch.advantages), (ret, batch.returns))
        ),
    )
    twice = mirror(mirror(batch.actor_observations, spec.actor_perm, spec.actor_sign), spec.actor_perm, spec.actor_sign)
    chk("mirror is an involution on this spec", bool(jnp.array_equal(twice, batch.actor_observations)))
    spec_no_critic = spec._replace(critic_perm=None, critic_sign=None)
    try:
        augment_minibatch(
            spec_no_critic,
            batch.actor_observations,
            batch.critic_observations,
            batch.actions,
            batch.old_log_probs,
            batch.values,
            batch.advantages,
            batch.returns,
        )
        chk("refuses a spec without the critic operator", False)
    except ValueError as exc:
        chk("refuses a spec without the critic operator", True, str(exc)[:60])


def section_2_reference(chk, spec, batch, model, key):
    print("\n=== 2. float64 reference of the rsl_rl rule ===")
    loss, info, _ = run_loss(model, batch, spec, True, key)
    ref = manual_loss_numpy(model, batch, spec)
    got = dict(
        total=float(loss),
        policy_loss=float(info.policy_loss),
        value_loss=float(info.value_loss),
        entropy=float(info.entropy),
        analytical_kl=float(info.analytical_kl),
        approx_kl=float(info.approx_kl),
        clip_fraction=float(info.clip_fraction),
    )
    for k in ref:
        d = abs(got[k] - ref[k])
        tol = 1e-5 * max(1.0, abs(ref[k]))
        chk(f"{k}: implementation == float64 reference", d <= tol, f"got {got[k]:.7f} ref {ref[k]:.7f} |d| {d:.2e}")


def section_3_same_graph(chk, spec, batch, model, key):
    print("\n=== 3. same-graph arithmetic, value and gradient ===")
    params, static = eqx.partition(model, eqx.is_inexact_array)
    for augment in (True, False):
        tag = "augment ON" if augment else "augment OFF (plain loss)"
        loss, info, grads = run_loss(model, batch, spec, augment, key)

        def manual(p, augment=augment):
            t, parts = manual_loss_jax(eqx.combine(p, static), batch, spec, augment, key)
            return t, parts

        (mt, parts), mgrads = jax.value_and_grad(manual, has_aux=True)(params)
        chk(f"{tag}: total loss", maxdiff(loss, mt) < 1e-6, f"|d| {maxdiff(loss, mt):.2e}")
        for k in ("policy_loss", "value_loss", "entropy", "analytical_kl"):
            chk(
                f"{tag}: {k}",
                maxdiff(getattr(info, k), parts[k]) < 1e-6,
                f"|d| {maxdiff(getattr(info, k), parts[k]):.2e}",
            )
        chk(
            f"{tag}: parameter gradients",
            grad_maxdiff(grads, mgrads) < 1e-6,
            f"max|d| {grad_maxdiff(grads, mgrads):.2e}",
        )
    # Flag off must not depend on the critic operator at all.
    spec_no_critic = spec._replace(critic_perm=None, critic_sign=None)
    l1, _, _ = run_loss(model, batch, spec, False, key)
    l2, _, _ = run_loss(model, batch, spec_no_critic, False, key)
    chk("augment OFF ignores the spec entirely (bitwise)", bool(jnp.array_equal(l1, l2)))
    # Bound penalty stays on the originals.
    _, info_b, _ = run_loss(model, batch, spec, True, key, bound=1.0)
    mu = model.mu(batch.actor_observations)
    bl = jnp.square(jnp.maximum(mu - 1.0, 0.0)).mean() + jnp.square(jnp.minimum(mu + 1.0, 0.0)).mean()
    chk(
        "bound penalty computed on the N originals only",
        maxdiff(info_b.aux["bound_loss"], bl) < 1e-6,
        f"|d| {maxdiff(info_b.aux['bound_loss'], bl):.2e}",
    )


def section_4_invariance(chk, spec, batch, model, key):
    print("\n=== 4. set invariance under mirroring the input batch ===")
    _, a, _ = run_loss(model, batch, spec, True, key)
    _, b, _ = run_loss(model, mirror_batch(batch, spec), spec, True, key)
    # approx_kl is excluded: it is measured on the originals only, and
    # mirroring the batch swaps which rows are the originals.
    for k in ("policy_loss", "value_loss", "entropy", "clip_fraction"):
        d = maxdiff(getattr(a, k), getattr(b, k))
        chk(f"{k}(B) == {k}(mirror B)", d < 1e-5, f"|d| {d:.2e}")
    _, a0, _ = run_loss(model, batch, spec, False, key)
    _, b0, _ = run_loss(model, mirror_batch(batch, spec), spec, False, key)
    d0 = maxdiff(a0.policy_loss, b0.policy_loss)
    chk("control: WITHOUT augmentation the same mirroring changes the policy loss", d0 > 1e-3, f"|d| {d0:.2e}")


def section_5_equivariant(chk, spec, batch, base, key):
    print("\n=== 5. equivariant policy: augmentation is a no-op ===")
    model = SymmetrizedAC(base=base, spec=spec)
    s = spec
    o = batch.actor_observations
    equi = maxdiff(model.mu(mirror(o, s.actor_perm, s.actor_sign)), mirror(model.mu(o), s.action_perm, s.action_sign))
    chk("wrapper is exactly equivariant: mu(L o) == K mu(o)", equi < 1e-6, f"|d| {equi:.2e}")
    # old stats consistent with an equivariant behaviour policy: take them
    # from the model itself so the mirrored rows are genuinely on-policy.
    lp, _, mu, sig, _ = model.evaluate_actions(o, batch.actions, key=key)
    b = batch._replace(old_log_probs=lp - 0.1, old_mu=mu, old_sigma=sig)
    l_aug, i_aug, g_aug = run_loss(model, b, spec, True, key)
    l_pl, i_pl, g_pl = run_loss(model, b, spec, False, key)
    for k in ("policy_loss", "value_loss", "entropy", "analytical_kl", "approx_kl", "clip_fraction"):
        d = maxdiff(getattr(i_aug, k), getattr(i_pl, k))
        chk(f"{k}: augmented == plain", d < 1e-5, f"|d| {d:.2e}")
    chk("total loss: augmented == plain", maxdiff(l_aug, l_pl) < 1e-5, f"|d| {maxdiff(l_aug, l_pl):.2e}")
    chk("gradients: augmented == plain", grad_maxdiff(g_aug, g_pl) < 1e-5, f"max|d| {grad_maxdiff(g_aug, g_pl):.2e}")
    # And the base (non-equivariant) stub does NOT enjoy this, so the
    # check above is not vacuous.
    lb, _, _ = run_loss(base, b, spec, True, key)
    lb0, _, _ = run_loss(base, b, spec, False, key)
    chk(
        "control: for the non-equivariant base, augmented != plain",
        maxdiff(lb, lb0) > 1e-4,
        f"|d| {maxdiff(lb, lb0):.2e}",
    )


def section_6_real_model(chk, spec, batch, key):
    print("\n=== 6. framework PPOActorCritic ===")
    pc = PPOPolicyConfig()
    model = PPOActorCritic(
        num_actor_obs=OBS_A,
        num_critic_obs=OBS_C,
        num_actions=ACT,
        actor_cfg=pc.actor,
        critic_cfg=pc.critic,
        init_noise_std=pc.init_noise_std,
        std_type=pc.std_type,
        distribution_type=pc.distribution_type,
        key=key,
        obs_normalization=True,
        obs_shapes={"actor": (OBS_A,), "critic": (OBS_C,)},
    )
    params, static = eqx.partition(
        model, eqx.is_inexact_array, is_leaf=lambda x: type(x).__name__ == "EmpiricalNormalization"
    )

    def loss_fn(p, bt, aug):
        return U.compute_batch_loss(p, static, bt, CLIP, VLC, ENT, True, True, key, spec, 0.0, 0.0, aug)

    (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, batch, True)
    finite = bool(jnp.isfinite(loss)) and all(bool(jnp.all(jnp.isfinite(g))) for g in jax.tree_util.tree_leaves(grads))
    chk("augmented loss and gradients are finite", finite, f"loss {float(loss):.5f}")
    (_, info_m), _ = jax.value_and_grad(loss_fn, has_aux=True)(params, mirror_batch(batch, spec), True)
    for k in ("policy_loss", "value_loss"):
        d = maxdiff(getattr(info, k), getattr(info_m, k))
        chk(f"{k}(B) == {k}(mirror B) with the real model", d < 1e-5, f"|d| {d:.2e}")
    # KL on originals only: recompute from the model on the N originals.
    m = eqx.combine(params, static)
    _, _, mu, sig, _ = m.evaluate_actions(batch.actor_observations, batch.actions, key=key)
    kl_ref = U.compute_analytical_kl(mu, sig, batch.old_mu, batch.old_sigma)
    chk(
        "analytical KL uses the N originals",
        maxdiff(info.analytical_kl, kl_ref) < 1e-6,
        f"|d| {maxdiff(info.analytical_kl, kl_ref):.2e}",
    )


def main() -> int:
    key = jax.random.PRNGKey(0)
    k_spec, k_batch, k_model, k_loss, k_real = jax.random.split(key, 5)
    spec = make_spec(k_spec)
    batch = make_batch(k_batch)
    model = StubAC(k_model)
    chk = Checker()
    print(f"synthetic dims: actor {OBS_A}, critic {OBS_C}, action {ACT}, N {N}")
    section_1_layout(chk, spec, batch)
    section_2_reference(chk, spec, batch, model, k_loss)
    section_3_same_graph(chk, spec, batch, model, k_loss)
    section_4_invariance(chk, spec, batch, model, k_loss)
    section_5_equivariant(chk, spec, batch, model, k_loss)
    section_6_real_model(chk, spec, batch, k_real)
    print("\n=== RESULT:", "ALL OK" if not chk.fails else f"{len(chk.fails)} FAIL: {chk.fails}", "===")
    return 1 if chk.fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
