"""Bitwise-equivalence check for the rollout-storage minibatch refactor.

The refactor replaces ``get_stacked_batches`` (which materialized
``num_epochs`` full shuffled copies of the rollout on device) with
``get_flat_batch`` + ``get_minibatch_indices`` (the update's scan gathers
one minibatch per step).  Same permutations, same minibatch membership —
so the ENTIRE update output (new params, optimizer state, per-batch
losses/KL, final RNG key) must match the old path bit for bit, on this
machine's own backend.

Two-phase, auto-detected from which storage API is present:

  1. With the OLD code (before syncing the refactor), running this dumps
     the old path's outputs:      -> baseline npz written
  2. With the NEW code, running it again compares the new path against
     that dump bitwise:           -> PASS/FAIL

Covers a vector-obs case and a vision-style dict-obs case (state vector
+ image group).

Run (same command both times):
    jaxpy -m jaxrlworld.scripts.diag.gates.check_ppo_minibatch_bitwise
"""

import os

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from jaxrlworld.rl.algorithms.ppo import update as U
from jaxrlworld.rl.storages.rollout_storage import RolloutStorage

OBS, ACT, N, T = 24, 8, 32, 12
IMG = (16, 16)
_LOG_2PI = float(np.log(2.0 * np.pi))
_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_minibatch_bitwise_baseline")


class StubAC(eqx.Module):
    """Minimal actor-critic implementing the protocol compute_batch_loss uses."""

    w1: jax.Array
    b1: jax.Array
    w2: jax.Array
    b2: jax.Array
    log_std: jax.Array
    vw1: jax.Array
    vb1: jax.Array
    vw2: jax.Array
    vb2: jax.Array
    dict_obs: bool = eqx.field(static=True)

    def __init__(self, key, in_dim, dict_obs):
        k = jax.random.split(key, 4)
        self.dict_obs = dict_obs
        self.w1 = jax.random.normal(k[0], (in_dim, 64)) * 0.1
        self.b1 = jnp.zeros(64)
        self.w2 = jax.random.normal(k[1], (64, ACT)) * 0.1
        self.b2 = jnp.zeros(ACT)
        self.log_std = jnp.zeros(ACT)
        self.vw1 = jax.random.normal(k[2], (in_dim, 64)) * 0.1
        self.vb1 = jnp.zeros(64)
        self.vw2 = jax.random.normal(k[3], (64, 1)) * 0.1
        self.vb2 = jnp.zeros(1)

    def _vec(self, obs):
        if self.dict_obs:
            return jnp.concatenate([obs["state"], obs["cam"].reshape(obs["cam"].shape[0], -1)], axis=-1)
        return obs

    def _mu(self, obs):
        h = jnp.tanh(self._vec(obs) @ self.w1 + self.b1)
        return h @ self.w2 + self.b2

    def evaluate_actions(self, actor_obs, actions, *, key=None):
        mu = self._mu(actor_obs)
        sigma = jnp.exp(self.log_std)
        z = (actions - mu) / sigma
        log_probs = (-0.5 * z**2 - jnp.log(sigma) - 0.5 * _LOG_2PI).sum(-1)
        entropy = (0.5 * (1.0 + _LOG_2PI) + jnp.log(sigma)).sum() * jnp.ones(actions.shape[0])
        return log_probs, entropy, mu, jnp.broadcast_to(sigma, mu.shape), {}

    def evaluate_value(self, critic_obs):
        h = jnp.tanh(self._vec(critic_obs) @ self.vw1 + self.vb1)
        return h @ self.vw2 + self.vb2, {}


def make_case(dict_obs: bool):
    if dict_obs:
        obs_shape = {"state": (OBS,), "cam": IMG}
        in_dim = OBS + IMG[0] * IMG[1]
    else:
        obs_shape = (OBS,)
        in_dim = OBS
    st = RolloutStorage(
        num_envs=N,
        num_steps=T,
        actor_obs_shape=obs_shape,
        critic_obs_shape=obs_shape,
        action_shape=(ACT,),
    )
    key = jax.random.PRNGKey(42)
    ks = iter(jax.random.split(key, 16))

    def fill(shape):
        return jax.random.normal(next(ks), shape)

    if dict_obs:
        st.actor_obs = {"state": fill((T, N, OBS)), "cam": fill((T, N) + IMG)}
        st.critic_obs = {"state": fill((T, N, OBS)), "cam": fill((T, N) + IMG)}
    else:
        st.actor_obs = fill((T, N, OBS))
        st.critic_obs = fill((T, N, OBS))
    st.actions = fill((T, N, ACT))
    st.values = fill((T, N))
    st.log_probs = fill((T, N))
    st.mu = fill((T, N, ACT))
    st.sigma = jnp.abs(fill((T, N, ACT))) + 0.5
    st.advantages = fill((T, N))
    st.returns = fill((T, N))

    model = StubAC(next(ks), in_dim, dict_obs)
    params, static = eqx.partition(model, eqx.is_inexact_array)
    optimizer = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(1e-3))
    opt_state = optimizer.init(params)
    return st, params, static, optimizer, opt_state, jax.random.PRNGKey(7)


def _old_update_all_batches(params, static, opt_state, optimizer, stacked, key):
    """The pre-refactor update, verbatim: a scan over PRE-GATHERED
    minibatches.  Same ``compute_batch_loss``, same early-stop/cond
    structure — only the xs differ (stacked data instead of index rows),
    which is exactly what the refactor changed.  Kept here so the
    old-vs-new comparison runs on any machine without the old files."""

    def scan_fn(carry, batch):
        params, opt_state, key, early_stopped = carry
        key, subkey = jax.random.split(key)

        def loss_fn(p):
            return U.compute_batch_loss(p, static, batch, 0.2, 1.0, 0.01, False, True, subkey, None, 0.0)

        (loss, loss_info), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        should_stop = False & (loss_info.approx_kl > 1.5 * 1e10)
        do_update = ~early_stopped & ~should_stop

        def apply_update(_):
            updates, new_opt = optimizer.update(grads, opt_state, params)
            return optax.apply_updates(params, updates), new_opt

        def skip_update(_):
            return params, opt_state

        new_params, new_opt_state = jax.lax.cond(do_update, apply_update, skip_update, operand=None)
        new_carry = (new_params, new_opt_state, key, early_stopped | should_stop)
        output = U.ScanOutput(
            policy_loss=loss_info.policy_loss,
            value_loss=loss_info.value_loss,
            entropy=loss_info.entropy,
            approx_kl=loss_info.approx_kl,
            analytical_kl=loss_info.analytical_kl,
            clip_fraction=loss_info.clip_fraction,
            did_update=do_update,
            aux=loss_info.aux,
        )
        return new_carry, output

    init = (params, opt_state, key, jnp.array(False))
    (fp, fo, fk, _), outputs = jax.lax.scan(scan_fn, init, stacked)
    return fp, fo, outputs, fk


def leaves_of(new_params, new_opt_state, outputs, new_key):
    return [
        np.asarray(x)
        for x in (
            jax.tree_util.tree_leaves(new_params)
            + jax.tree_util.tree_leaves(new_opt_state)
            + jax.tree_util.tree_leaves(outputs._asdict())
            + [new_key]
        )
    ]


def run_case(dict_obs: bool, old_api: bool):
    st, params, static, optimizer, opt_state, key = make_case(dict_obs)
    common = (params, static, opt_state, optimizer, 0.2, 1.0, 0.01, False, True, False, 1e10, None, 0.0)
    if old_api:
        batches = st.get_stacked_batches(num_minibatches=4, num_epochs=2, key=key)
        out = U.update_all_batches(*common, batches, key)
    else:
        flat = st.get_flat_batch()
        idx = st.get_minibatch_indices(num_minibatches=4, num_epochs=2, key=key)
        out = U.update_all_batches(*common, flat, idx, key)
    return leaves_of(*out)


def run_case_old_emulated(dict_obs: bool):
    """Old path reproduced from the NEW storage: pre-gather the stacked
    minibatches with the same index rows (value-identical to what
    ``get_stacked_batches`` produced) and run the verbatim old scan."""
    st, params, static, optimizer, opt_state, key = make_case(dict_obs)
    flat = st.get_flat_batch()
    idx = st.get_minibatch_indices(num_minibatches=4, num_epochs=2, key=key)
    stacked = jax.tree.map(lambda x: x[idx], flat)
    out = jax.jit(_old_update_all_batches, static_argnums=(3,))(params, static, opt_state, optimizer, stacked, key)
    return leaves_of(*out)


def main() -> int:
    old_api = hasattr(RolloutStorage, "get_stacked_batches")
    backend = jax.default_backend()
    print("=" * 78)
    print(f"  PPO MINIBATCH REFACTOR — BITWISE EQUIVALENCE ({backend})")
    print(f"  detected storage API: {'OLD (pre-refactor)' if old_api else 'NEW (refactored)'}")
    print("=" * 78)

    failed = False
    for case, dict_obs in (("vec", False), ("dict", True)):
        path = f"{_BASE}_{backend}_{case}.npz"
        leaves = run_case(dict_obs, old_api)
        if old_api:
            np.savez(path, *leaves)
            print(f"  [{case}] baseline written ({len(leaves)} leaves): {path}")
            print("          -> now sync the refactor and run this again to compare")
            continue

        # Primary check: the verbatim old scan over pre-gathered stacked
        # minibatches, reproduced in-process, versus the shipped indexed
        # update — same machine, same backend, no old files needed.
        old_leaves = run_case_old_emulated(dict_obs)
        bad = [i for i, (a, b) in enumerate(zip(old_leaves, leaves)) if not np.array_equal(a, b)]
        if len(old_leaves) != len(leaves) or bad:
            print(f"  [{case}] FAIL vs emulated old path — differing leaves: {bad} (of {len(leaves)})")
            failed = True
        else:
            print(f"  [{case}] BIT-IDENTICAL vs emulated old path — all {len(leaves)} leaves")

        # Secondary: a baseline file dumped by a genuinely-old checkout,
        # when one exists on this machine.
        if os.path.isfile(path):
            ref = np.load(path)
            bad = [
                i for i, (name, x) in enumerate(zip(ref.files, leaves)) if not np.array_equal(ref[name], np.asarray(x))
            ]
            if len(ref.files) != len(leaves) or bad:
                print(f"  [{case}] FAIL vs stored old-code baseline — differing leaves: {bad}")
                failed = True
            else:
                print(f"  [{case}] BIT-IDENTICAL vs stored old-code baseline")

    print("=" * 78)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
