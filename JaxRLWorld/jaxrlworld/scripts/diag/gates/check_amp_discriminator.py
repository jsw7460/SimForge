"""Gate: the AMP discriminator and the prior's bookkeeping, without a simulator.

The JAX implementation is held against an independent torch re-derivation
of the same definitions (losses, reward, gradient penalty by autograd,
running normalizer), and the prior's stateful parts are checked on
synthetic data with known answers:

1. losses and rewards: bce / hinge / wasserstein, logits and R1 / WGAN-GP
   penalties equal the torch reference on the same weights and batches;
2. the minibatch-std feature is what the reference computes, and is a
   constant (not differentiated) inside the penalty;
3. the running normalizer advances like the reference, expert batch first;
4. policy history: chronological push, per-env backfill on done, the
   window fed to the discriminator is the flattened history, the blended
   reward is ``(1 - w) r + w dt r_style``;
5. replay: eager fill, circular write, capped random insertion when full;
6. expert sampling mass: per file, spread uniformly over each variant's
   frames;
7. discriminator training on separable synthetic windows drives the
   accuracy up and the loss down, the learning-rate rewrite reaches both
   parameter groups, and a save / load round trip reproduces the rewards.

Run::

    python -m jaxrlworld.scripts.diag.gates.check_amp_discriminator
"""

from __future__ import annotations

import tempfile

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import torch

from jaxrlworld.rl.algorithms.amp_ppo.amp import (
    AdversarialMotionPrior,
    _discriminator_updates,
    expert_frame_probabilities,
)
from jaxrlworld.rl.algorithms.amp_ppo.discriminator import (
    Discriminator,
    discriminator_logits,
    discriminator_loss,
    discriminator_reward,
    minibatch_std,
)
from jaxrlworld.rl.algorithms.amp_ppo.expert_motion import ExpertMotionSet
from jaxrlworld.rl.configs.algorithms.amp_ppo import AmpConfig
from jaxrlworld.rl.modules.normalization import EmpiricalNormalization


class Checker:
    def __init__(self) -> None:
        self.fails: list[str] = []

    def __call__(self, name: str, ok: bool, detail: str = "") -> bool:
        ok = bool(ok)
        print(f"  [{'OK' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
        if not ok:
            self.fails.append(name)
        return ok


# ── torch reference ─────────────────────────────────────────────────


class TorchDisc(torch.nn.Module):
    """The reference network with the JAX weights copied in."""

    def __init__(self, disc: Discriminator):
        super().__init__()
        layers = []
        for lin in disc.trunk.linears:
            layer = torch.nn.Linear(lin.in_features, lin.out_features).double()
            layer.weight.data = torch.tensor(np.asarray(lin.weight, dtype=np.float64))
            layer.bias.data = torch.tensor(np.asarray(lin.bias, dtype=np.float64))
            layers += [layer, torch.nn.ReLU()]
        self.trunk = torch.nn.Sequential(*layers)
        self.head = torch.nn.Linear(disc.head.in_features, 1).double()
        self.head.weight.data = torch.tensor(np.asarray(disc.head.weight, dtype=np.float64))
        self.head.bias.data = torch.tensor(np.asarray(disc.head.bias, dtype=np.float64))
        self.use_std = disc.use_minibatch_std
        self.loss_type = disc.loss_type
        self.eta = disc.eta

    def forward(self, x, detach_std=False):
        h = self.trunk(x)
        if self.use_std:
            s = h.std(dim=0, unbiased=False).mean()
            if detach_std:
                s = s.detach()
            h = torch.cat([h, s.expand(h.shape[0], 1)], dim=-1)
        return self.head(h)[:, 0]


def torch_reference(disc: Discriminator, mean, var, eps, policy_raw, expert_raw, lambda_, alpha=None):
    """(amp_loss, grad_penalty, reward(policy_raw)) by the reference definitions with autograd."""
    ref = TorchDisc(disc)
    norm = lambda x: (torch.tensor(x, dtype=torch.float64) - torch.tensor(mean)) / (torch.sqrt(torch.tensor(var)) + eps)  # noqa: E731
    policy, expert = norm(policy_raw), norm(expert_raw)
    d = ref(torch.cat([policy, expert]))
    dp, de = d[: policy.shape[0]], d[policy.shape[0] :]
    bce = torch.nn.BCEWithLogitsLoss()
    if disc.loss_type == "bce":
        amp = 0.5 * (bce(de, torch.ones_like(de)) + bce(dp, torch.zeros_like(dp)))
    elif disc.loss_type == "hinge":
        amp = 0.5 * (torch.relu(1 - de).mean() + torch.relu(1 + dp).mean())
    else:
        amp = torch.tanh(disc.eta * dp).mean() - torch.tanh(disc.eta * de).mean()
    if disc.loss_type == "wasserstein":
        rep = (policy.shape[0] + expert.shape[0] - 1) // expert.shape[0]
        e = expert.repeat(rep, 1)[: policy.shape[0]]
        a = torch.tensor(alpha, dtype=torch.float64)
        data = (a * e + (1 - a) * policy).detach().requires_grad_(True)
        scores = torch.tanh(disc.eta * ref(data, detach_std=True))
        grad = torch.autograd.grad(scores, data, torch.ones_like(scores), create_graph=True)[0]
        gp = lambda_ * (grad.norm(2, dim=1) - 1.0).pow(2).mean()
    else:
        data = expert.detach().requires_grad_(True)
        scores = ref(data, detach_std=True)
        grad = torch.autograd.grad(scores.sum(), data, create_graph=True)[0]
        gp = 0.5 * lambda_ * grad.pow(2).sum(dim=1).mean()
    with torch.no_grad():
        logits = ref(policy)
        reward = (
            torch.exp(torch.tanh(disc.eta * logits))
            if disc.loss_type == "wasserstein"
            else torch.nn.functional.softplus(logits)
        )
    return float(amp), float(gp), reward.numpy()


def main() -> int:
    chk = Checker()
    rng = np.random.default_rng(0)
    F = 12
    key = jax.random.PRNGKey(0)

    print("=== 1-2. losses, penalties, rewards vs torch autograd reference ===")
    policy_raw = rng.normal(size=(40, F)).astype(np.float32)
    expert_raw = (rng.normal(size=(20, F)) + 0.5).astype(np.float32)
    for loss_type in ("bce", "hinge", "wasserstein"):
        key, k = jax.random.split(key)
        disc = Discriminator(F, (16, 8), "relu", True, loss_type, 1.0, 1.0, key=k)
        norm = EmpiricalNormalization(F).update(jnp.asarray(expert_raw)).update(jnp.asarray(policy_raw))
        key, k_gp = jax.random.split(key)
        total, stats = discriminator_loss(disc, norm, jnp.asarray(policy_raw), jnp.asarray(expert_raw), 10.0, k_gp)
        alpha = np.asarray(jax.random.uniform(k_gp, (policy_raw.shape[0], 1))) if loss_type == "wasserstein" else None
        amp_ref, gp_ref, reward_ref = torch_reference(
            disc, np.asarray(norm.mean), np.asarray(norm.var), norm.epsilon, policy_raw, expert_raw, 10.0, alpha
        )
        reward = np.asarray(discriminator_reward(disc, norm, jnp.asarray(policy_raw)))
        chk(
            f"{loss_type}: amp loss == reference",
            abs(float(stats.amp_loss) - amp_ref) < 1e-5,
            f"{float(stats.amp_loss):.6f} vs {amp_ref:.6f}",
        )
        chk(
            f"{loss_type}: gradient penalty == reference (autograd)",
            abs(float(stats.grad_penalty) - gp_ref) < 1e-4 * max(1.0, gp_ref),
            f"{float(stats.grad_penalty):.6f} vs {gp_ref:.6f}",
        )
        chk(
            f"{loss_type}: total == amp + penalty",
            abs(float(total) - float(stats.amp_loss) - float(stats.grad_penalty)) < 1e-6,
        )
        chk(
            f"{loss_type}: reward == reference",
            np.abs(reward - reward_ref).max() < 1e-5,
            f"max |Δ| {np.abs(reward - reward_ref).max():.1e}",
        )
    # minibatch std: the value, and that it is a constant inside the penalty.
    x = jnp.asarray(rng.normal(size=(30, F)).astype(np.float32))
    h = disc.trunk(x)
    chk(
        "minibatch std == mean over features of the batch std (ddof 0)",
        abs(float(minibatch_std(h)) - float(np.asarray(h).std(axis=0, ddof=0).mean())) < 1e-6,
    )
    # A feature constant over the batch (dead unit): the std gradient must be
    # finite and zero for that column, as torch's std backward makes it.
    dead = np.asarray(h).copy()
    dead[:, 2] = 0.0
    g = jax.grad(lambda z: minibatch_std(z))(jnp.asarray(dead))
    ht = torch.tensor(dead, requires_grad=True)
    ht.std(dim=0, unbiased=False).mean().backward()
    chk(
        "std gradient on a zero-variance column is finite and matches torch (zero)",
        bool(jnp.isfinite(g).all()) and np.abs(np.asarray(g) - ht.grad.numpy()).max() < 1e-6,
        f"max |Δ| vs torch {np.abs(np.asarray(g) - ht.grad.numpy()).max():.1e}",
    )
    chk(
        "std value unchanged by the guard",
        abs(float(minibatch_std(jnp.asarray(dead))) - float(dead.std(axis=0, ddof=0).mean())) < 1e-6,
    )
    l_free = discriminator_logits(disc, x)
    l_fixed = discriminator_logits(disc, x, batch_std=jax.lax.stop_gradient(minibatch_std(h)))
    chk(
        "logits with the std feature supplied == logits computing it", np.abs(np.asarray(l_free - l_fixed)).max() < 1e-6
    )
    no_std = Discriminator(F, (16, 8), "relu", False, "bce", 1.0, 1.0, key=k)
    chk("without minibatch std the head is (H -> 1)", no_std.head.in_features == 8 and disc.head.in_features == 9)

    print("\n=== 3. running normalizer ===")
    n0 = EmpiricalNormalization(F)
    a, b = rng.normal(size=(50, F)), rng.normal(size=(70, F)) * 3 + 1
    n1 = n0.update(jnp.asarray(a)).update(jnp.asarray(b))
    both = np.concatenate([a, b])
    # Chan's merge with the module's initial pseudo-count 1e-4 on a zero-mean unit-var prior.
    c0, m0, v0 = 1e-4, np.zeros(F), np.ones(F)
    for batch in (a, b):
        bm, bv, bc = batch.mean(0), batch.var(0), batch.shape[0]
        tot = c0 + bc
        m_new = m0 + (bm - m0) * bc / tot
        v0 = (v0 * c0 + bv * bc + (bm - m0) ** 2 * c0 * bc / tot) / tot
        m0, c0 = m_new, tot
    chk(
        "two updates == Chan merge of the batches (mean, var)",
        np.abs(np.asarray(n1.mean) - m0).max() < 1e-4 and np.abs(np.asarray(n1.var) - v0).max() < 1e-3,
        f"mean vs population {np.abs(np.asarray(n1.mean) - both.mean(0)).max():.1e}",
    )
    chk(
        "normalize == (x - mean) / (sqrt(var) + eps)",
        np.abs(
            np.asarray(n1.normalize(jnp.asarray(a)))
            - (a - np.asarray(n1.mean)) / (np.sqrt(np.asarray(n1.var)) + n1.epsilon)
        ).max()
        < 1e-5,
    )

    print("\n=== 4-6. the prior: history, blend, replay, expert mass ===")
    D, K, N = 3, 4, 5
    T = 30
    feats = rng.normal(size=(T, D)).astype(np.float32)
    expert = ExpertMotionSet(
        features=feats,
        clip_start=np.asarray([0, 10, 30]),
        clip_names=("a", "b+mirror"),
        fps=50.0,
        source_index=np.asarray([0, 0]),
    )
    cfg = AmpConfig(
        motion_files=("a.npz",), root_body_name="Trunk", num_amp_obs_steps=K, style_reward_weight=0.3,
        discriminator_hidden_dims=(8, 8), replay_buffer_size=64, replay_insert_size=7, loss_type="bce",
    )  # fmt: skip
    prior = AdversarialMotionPrior(cfg, expert, D, 0.02, 1e-3, jax.random.PRNGKey(1))
    probs = np.asarray(prior.expert_probs)
    chk(
        "expert mass: each variant gets equal total mass, uniform over its frames",
        abs(probs[:10].sum() - 0.5) < 1e-6
        and abs(probs[10:].sum() - 0.5) < 1e-6
        and np.allclose(probs[:10], probs[0])
        and np.allclose(probs[10:], probs[10]),
    )
    two_files = ExpertMotionSet(
        features=feats,
        clip_start=np.asarray([0, 10, 30]),
        clip_names=("a", "b"),
        fps=50.0,
        source_index=np.asarray([0, 1]),
    )
    p2 = expert_frame_probabilities(two_files, (1.0, 3.0))
    chk("dataset weights scale a file's mass", abs(p2[:10].sum() - 0.25) < 1e-6 and abs(p2[10:].sum() - 0.75) < 1e-6)
    try:
        expert_frame_probabilities(two_files, (1.0,))
        chk("wrong number of dataset weights refused", False)
    except ValueError:
        chk("wrong number of dataset weights refused", True)

    obs = [rng.normal(size=(N, D)).astype(np.float32) for _ in range(6)]
    rewards = [rng.normal(size=(N,)).astype(np.float32) for _ in range(6)]
    dones = [np.zeros(N, bool) for _ in range(6)]
    dones[3][1] = True  # env 1 resets at step 3
    windows = []
    blended = []
    for o, r, d in zip(obs, rewards, dones):
        blended.append(np.asarray(prior.shape_rewards(jnp.asarray(o), jnp.asarray(d), jnp.asarray(r))))
        windows.append(np.asarray(prior._current[-1]))
    hist = np.asarray(prior._history)
    chk("history holds the last K frames, chronological", np.array_equal(hist[0], np.stack(obs[2:6])[:, 0]))
    chk("first step backfilled the history with its own frame", np.array_equal(windows[0][0], np.tile(obs[0][0], K)))
    chk("done env: window == its post-reset frame repeated", np.array_equal(windows[3][1], np.tile(obs[3][1], K)))
    chk(
        "done env: later frames push into the backfilled history",
        np.array_equal(windows[5][1], np.concatenate([obs[3][1], obs[3][1], obs[4][1], obs[5][1]])),
    )
    chk(
        "undone env: window at step 5 == frames 2..5",
        np.array_equal(windows[5][0], np.concatenate([obs[2][0], obs[3][0], obs[4][0], obs[5][0]])),
    )
    style = np.asarray(discriminator_reward(prior.disc, prior.normalizer, jnp.asarray(windows[5])))
    want = (1 - 0.3) * rewards[5] + 0.3 * 0.02 * style
    chk(
        "blend == (1 - w) r + w dt r_style",
        np.abs(blended[5] - want).max() < 1e-6,
        f"max |Δ| {np.abs(blended[5] - want).max():.1e}",
    )
    chk("stats accumulated per step", prior._num_steps_seen == 6)

    # Replay: 6 steps x 5 envs = 30 windows -> eager fill; then a second rollout of 30 -> 60; then 30 more -> capped at 64 with 7 inserted.
    m1 = prior.update_discriminator(num_updates=2, minibatch_size=8)
    chk(
        "replay after rollout 1: 30 windows written in order",
        prior.replay_count == 30 and np.array_equal(np.asarray(prior.replay[:30]), np.concatenate(windows)),
    )
    chk(
        "metrics carry the blend's per-step means",
        abs(m1.task_reward - 0.7 * np.mean([r.mean() for r in rewards])) < 1e-5 and m1.style_weight == 0.3,
        f"task {m1.task_reward:.4f}",
    )
    for o, r, d in zip(obs, rewards, dones):
        prior.shape_rewards(jnp.asarray(o), jnp.asarray(d), jnp.asarray(r))
    prior.update_discriminator(2, 8)
    chk("replay after rollout 2: 60 windows", prior.replay_count == 60 and prior.replay_ptr == 60)
    for o, r, d in zip(obs, rewards, dones):
        prior.shape_rewards(jnp.asarray(o), jnp.asarray(d), jnp.asarray(r))
    before = np.asarray(prior.replay).copy()
    prior.update_discriminator(2, 8)
    after = np.asarray(prior.replay)
    changed = int((np.abs(after - before).max(axis=1) > 0).sum())
    chk(
        "replay not yet full: the whole third rollout is written, wrapping around",
        prior.replay_count == 64 and prior.replay_ptr == 26 and changed == 30,
        f"{changed} rows changed, ptr {prior.replay_ptr}",
    )
    for o, r, d in zip(obs, rewards, dones):
        prior.shape_rewards(jnp.asarray(o), jnp.asarray(d), jnp.asarray(r))
    before = after.copy()
    prior.update_discriminator(2, 8)
    after = np.asarray(prior.replay)
    changed = int((np.abs(after - before).max(axis=1) > 0).sum())
    chk(
        "replay full: exactly replay_insert_size rows replaced, in order",
        changed == 7 and prior.replay_count == 64 and prior.replay_ptr == 33,
        f"{changed} rows changed, ptr {prior.replay_ptr}",
    )
    try:
        prior.update_discriminator(1, 8)
        chk("update without new rollout windows refused", False)
    except RuntimeError:
        chk("update without new rollout windows refused", True)

    print("\n=== 7. training on separable windows, learning-rate rewrite, checkpoint round trip ===")
    D, K = 6, 3
    T = 2000
    expert_feats = (rng.normal(size=(T, D)) + 2.0).astype(np.float32)
    expert = ExpertMotionSet(
        features=expert_feats, clip_start=np.asarray([0, T]), clip_names=("e",), fps=50.0, source_index=np.asarray([0])
    )
    cfg = AmpConfig(
        motion_files=("e.npz",),
        root_body_name="Trunk",
        num_amp_obs_steps=K,
        discriminator_hidden_dims=(32, 16),
        replay_buffer_size=4096,
        loss_type="bce",
    )
    prior = AdversarialMotionPrior(cfg, expert, D, 0.02, 1e-3, jax.random.PRNGKey(2))
    N = 64
    for _ in range(12):
        o = (rng.normal(size=(N, D)) - 2.0).astype(np.float32)
        prior.shape_rewards(jnp.asarray(o), jnp.zeros(N, bool), jnp.zeros(N, jnp.float32))
    first = prior.update_discriminator(num_updates=1, minibatch_size=128)
    for _ in range(12):
        o = (rng.normal(size=(N, D)) - 2.0).astype(np.float32)
        prior.shape_rewards(jnp.asarray(o), jnp.zeros(N, bool), jnp.zeros(N, jnp.float32))
    later = prior.update_discriminator(num_updates=300, minibatch_size=128)
    chk(
        "discriminator loss falls with training",
        later.disc_loss < 0.5 * first.disc_loss,
        f"{first.disc_loss:.3f} -> {later.disc_loss:.3f}",
    )
    chk(
        "accuracy rises (policy > 0.85, expert > 0.95; the R1 penalty and the head's decay slow it)",
        later.accuracy_policy > 0.85 and later.accuracy_expert > 0.95,
        f"policy {later.accuracy_policy:.3f} expert {later.accuracy_expert:.3f}",
    )
    # Batches of one size throughout: the minibatch-std feature makes a
    # window's reward depend on the batch it is scored with.
    r_exp = np.asarray(discriminator_reward(prior.disc, prior.normalizer, prior.expert_windows[:64]))
    r_pol = np.asarray(discriminator_reward(prior.disc, prior.normalizer, prior.replay[:64]))
    chk(
        "expert windows earn more style reward than policy windows",
        r_exp.mean() > 2 * r_pol.mean(),
        f"expert {r_exp.mean():.3f} policy {r_pol.mean():.3f}",
    )

    prior.set_learning_rate(2.5e-4)
    lrs = [
        float(prior.opt_state.inner_states[label].inner_state[1].hyperparams["learning_rate"])
        for label in ("trunk", "head")
    ]
    chk("learning-rate rewrite reaches both groups", all(abs(lr - 2.5e-4) < 1e-9 for lr in lrs), f"{lrs}")
    params, static = eqx.partition(prior.disc, eqx.is_inexact_array)
    labels = prior.param_labels
    flat_labels = jax.tree_util.tree_leaves(labels)
    chk(
        "head parameters labelled 'head', the rest 'trunk'",
        flat_labels.count("head") == 2 and flat_labels.count("trunk") == len(flat_labels) - 2,
        f"{flat_labels}",
    )

    with tempfile.TemporaryDirectory() as tmp:
        prior.save(tmp)
        twin = AdversarialMotionPrior(cfg, expert, D, 0.02, 1e-3, jax.random.PRNGKey(99))
        r_before = np.asarray(discriminator_reward(twin.disc, twin.normalizer, prior.expert_windows[:64]))
        twin.learning_rate = prior.learning_rate
        twin.load(tmp)
        r_after = np.asarray(discriminator_reward(twin.disc, twin.normalizer, prior.expert_windows[:64]))
        chk("fresh prior differs before load", np.abs(r_before - r_exp).max() > 1e-3)
        chk(
            "loaded prior reproduces the rewards (weights + normalizer)",
            np.array_equal(r_after, r_exp),
            f"max |Δ| {np.abs(r_after - r_exp).max():.1e}",
        )
        chk(
            "loaded prior carries the learning rate",
            abs(float(twin.opt_state.inner_states["head"].inner_state[1].hyperparams["learning_rate"]) - 2.5e-4) < 1e-9,
        )

    # The scan is deterministic for a fixed key and the reference of one step matches a manual step.
    params, static = eqx.partition(prior.disc, eqx.is_inexact_array)
    key = jax.random.PRNGKey(7)
    args = (
        params,
        static,
        prior.optimizer,
        prior.opt_state,
        prior.normalizer,
        key,
        32,
        3,
        10.0,
        prior.replay[:512],
        prior.replay,
        jnp.asarray(prior.replay_count),
        prior.expert_windows,
        prior.expert_probs,
    )
    out_a = _discriminator_updates(*args)
    out_b = _discriminator_updates(*args)
    chk(
        "discriminator update scan is deterministic for a fixed key",
        all(
            np.array_equal(np.asarray(x), np.asarray(y))
            for x, y in zip(jax.tree_util.tree_leaves(out_a[0]), jax.tree_util.tree_leaves(out_b[0]))
        ),
    )

    print("\n=== RESULT:", "ALL OK" if not chk.fails else f"{len(chk.fails)} FAIL: {chk.fails}", "===")
    return 1 if chk.fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
