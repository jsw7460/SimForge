"""Does changing the learning rate keep the optimizer's memory?

PPO's adaptive schedule moves the actor learning rate whenever the KL
leaves its target band — often, and most often early in training where
the KL swings hardest. It used to do that by rebuilding the optimizer
and calling ``init`` again, which returns ZEROED Adam moments: every
adjustment silently restarted the optimizer from scratch. rsl_rl assigns
to ``param_group["lr"]`` instead and keeps its state.

Nothing about that failure is visible in a training curve. It shows up
as a run that learns more slowly than it should, which is
indistinguishable from a task that is simply hard.

So it is checked directly:

* the rate written is the rate the optimizer state holds;
* the Adam moments are the SAME arrays before and after — not merely
  present, but unchanged, since a rebuild that happened to run one step
  would also leave them non-zero;
* the critic's rate is untouched, because the adaptive schedule is the
  actor's;
* an update still runs afterwards and moves the parameters.

    python -m jaxrlworld.scripts.diag.gates.ppo_adaptive_lr_diag
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

from jaxrlworld.rl.algorithms.base import create_optimizer_with_labels


def _is(actual: float, expected: float) -> bool:
    """Is ``actual`` the learning rate that was asked for?

    The rate is stored as a float32 leaf inside the optimizer state, so it
    comes back rounded: 1e-3 reads as 0.0010000000474974513. Comparing
    against the Python float exactly can never pass, so the expected value
    is put through the same rounding first.
    """
    return float(jnp.float32(expected)) == actual


def _moments(opt_state) -> dict[str, jnp.ndarray]:
    """Every Adam moment array in the state, flattened and labelled."""
    out: dict[str, jnp.ndarray] = {}
    leaves = jax.tree_util.tree_leaves_with_path(opt_state)
    for path, leaf in leaves:
        if not hasattr(leaf, "shape"):
            continue
        key = ".".join(str(p) for p in path)
        if "mu" in key or "nu" in key:
            out[key] = leaf
    return out


def _injected_lrs(opt_state) -> dict[str, float]:
    """The learning rate held inside the state, per label.

    Reached exactly the way ``PPO._write_injected_lr`` reaches it —
    ``opt_state[1]`` is the multi_transform state — so this check fails if
    that structural assumption ever stops holding.
    """
    inner_states = opt_state[1].inner_states
    return {label: float(w.inner_state.hyperparams["learning_rate"]) for label, w in inner_states.items()}


def main() -> int:
    import equinox as eqx

    print("=" * 78)
    print("PPO ADAPTIVE LEARNING RATE — does the optimizer keep its state?")
    print("=" * 78)

    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)

    # A stand-in for the actor/critic split: two labelled parameter groups,
    # which is what the real optimizer is built over.
    class Model(eqx.Module):
        actor: jnp.ndarray
        critic: jnp.ndarray

        def __init__(self, ka, kc):
            self.actor = jax.random.normal(ka, (4, 4))
            self.critic = jax.random.normal(kc, (4, 4))

    model = Model(k1, k2)

    def label_fn(path):
        return "actor" if "actor" in ".".join(str(p) for p in path) else "critic"

    optimizer, _ = create_optimizer_with_labels(
        model=model,
        label_fn=label_fn,
        lr_config={"actor": 1e-3, "critic": 5e-4},
        max_grad_norm=1.0,
        optimizer_class=optax.adam,
    )
    params, static = eqx.partition(model, eqx.is_inexact_array)
    opt_state = optimizer.init(params)

    results: dict[str, bool] = {}

    lrs = _injected_lrs(opt_state)
    print(f"\n  learning rates inside the state : {lrs}")
    results["the_state_carries_the_rate"] = set(lrs) == {"actor", "critic"}
    results["each_label_keeps_its_own_rate"] = _is(lrs["actor"], 1e-3) and _is(lrs["critic"], 5e-4)

    # Take a few steps so the moments hold something worth losing.
    grads = jax.tree.map(lambda p: jnp.ones_like(p), params)
    for _ in range(3):
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = eqx.apply_updates(params, updates)

    before = _moments(opt_state)
    nonzero = {k: float(jnp.abs(v).sum()) for k, v in before.items()}
    print(f"  moment arrays after 3 steps     : {len(before)}, total |.| = {sum(nonzero.values()):.4f}")
    results["moments_are_populated_before"] = sum(nonzero.values()) > 0

    # The adjustment itself: write the rate in place, as the schedule does.
    opt_state[1].inner_states["actor"].inner_state.hyperparams["learning_rate"] = jnp.asarray(2e-3, jnp.float32)

    after_lrs = _injected_lrs(opt_state)
    after = _moments(opt_state)
    print(f"  learning rates after the change : {after_lrs}")
    results["the_new_rate_is_what_was_written"] = _is(after_lrs["actor"], 2e-3)
    results["the_critic_rate_is_untouched"] = after_lrs["critic"] == lrs["critic"]

    same = all(jnp.array_equal(before[k], after[k]) for k in before)
    print(f"  moment arrays unchanged         : {same}")
    results["the_moments_survive_the_change"] = same and len(after) == len(before)

    # And the optimizer still works afterwards.
    updates, opt_state = optimizer.update(grads, opt_state, params)
    new_params = eqx.apply_updates(params, updates)
    moved = float(jnp.abs(new_params.actor - params.actor).sum())
    print(f"  an update after the change moves the actor by {moved:.6f}")
    results["updates_still_apply_after_the_change"] = moved > 0

    print("=" * 78)
    ok = True
    for k, v in results.items():
        print(f"  {k:<40}: {'PASS' if v else 'FAIL'}")
        ok = ok and v
    print(f"  {'OVERALL':<40}: {'PASS' if ok else 'FAIL'}")
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
