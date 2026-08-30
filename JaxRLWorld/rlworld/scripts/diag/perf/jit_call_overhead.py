"""How much of a jitted call is Python, and how much is the GPU.

A batch-1 forward through a three-layer MLP is a few microseconds of
arithmetic. Measured through ``eqx.filter_jit`` it costs closer to a
hundred, and an off-policy algorithm makes several such calls per
environment step, so the difference is most of the step.

The suspicion is ``filter_jit`` itself: it walks and partitions the
whole argument pytree on every call, so the cost tracks leaf count
rather than FLOPs — which is what the step breakdown showed (act, a
small tree, 88 us; the fused update, six times the leaves, 583 us).

Three ways of running the same arithmetic, timed with a single
synchronisation at the end of each so what is measured is the rate the
host can issue work:

  filter_jit    what the code does today
  jit(params)   the module partitioned once up front, static captured in
                a closure, so each call passes only arrays
  bare          a jitted add on one array — the floor for a dispatch

``jit(params)`` is what a refactor would buy; ``bare`` says how much of
what remains is JAX's own dispatch and therefore not recoverable.

Run on the training box:
    jaxpy -m rlworld.scripts.diag.perf.jit_call_overhead
    jaxpy -m rlworld.scripts.diag.perf.jit_call_overhead --batch-size 256
"""

import argparse
import time

import equinox as eqx
import jax
import jax.numpy as jnp

from rlworld.rl.configs.common_config_classes import Activation, DefaultInit, MLPActorCfg, MLPCriticCfg
from rlworld.rl.modules.policies.sac_ac import SACActorCritic


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--obs-dim", type=int, default=17)
    ap.add_argument("--act-dim", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--calls", type=int, default=5000)
    ap.add_argument("--warmup", type=int, default=500)
    args = ap.parse_args()

    model = SACActorCritic(
        num_actor_obs=args.obs_dim,
        num_critic_obs=args.obs_dim,
        num_actions=args.act_dim,
        actor_cfg=MLPActorCfg(hidden_dims=[256, 256], activation=Activation.RELU, init=DefaultInit()),
        critic_cfg=MLPCriticCfg(hidden_dims=[256, 256], activation=Activation.RELU, init=DefaultInit()),
        key=jax.random.PRNGKey(0),
    )
    obs = jnp.zeros((args.batch_size, args.obs_dim), jnp.float32)
    key = jax.random.PRNGKey(1)

    params, static = eqx.partition(model, eqx.is_inexact_array)
    n_leaves = len(jax.tree.leaves(params))

    print("=" * 78)
    print("  JIT CALL OVERHEAD — PYTHON VS GPU")
    print(f"  equinox {eqx.__version__}   jax {jax.__version__}   backend {jax.default_backend()}")
    print(f"  batch {args.batch_size}   obs {args.obs_dim}   model leaves {n_leaves}   calls {args.calls:,}")
    print("=" * 78)

    @eqx.filter_jit
    def via_filter_jit(m, x, k):
        actions, _ = m.act(x, key=k, deterministic=False)
        return actions

    @jax.jit
    def via_params(p, x, k):
        actions, _ = eqx.combine(p, static).act(x, key=k, deterministic=False)
        return actions

    @jax.jit
    def bare(x):
        return x + 1.0

    def timed(label: str, body) -> float:
        for _ in range(args.warmup):
            out = body()
        jax.block_until_ready(out)
        t0 = time.perf_counter()
        for _ in range(args.calls):
            out = body()
        jax.block_until_ready(out)
        dt = (time.perf_counter() - t0) / args.calls * 1e6
        print(f"  {label:<14s} {dt:8.1f} us/call")
        return dt

    t_filter = timed("filter_jit", lambda: via_filter_jit(model, obs, key))
    t_params = timed("jit(params)", lambda: via_params(params, obs, key))
    t_bare = timed("bare", lambda: bare(obs))

    print("\n" + "-" * 78)
    saved = t_filter - t_params
    print(f"  partitioning the module on every call costs {saved:7.1f} us")
    print(f"  what is left above a bare dispatch is       {t_params - t_bare:7.1f} us")
    print(f"  a bare dispatch is                          {t_bare:7.1f} us")
    print("-" * 78)
    if saved > 0.3 * t_filter:
        print("  Worth refactoring: partition once at init, keep static in a closure,")
        print("  and pass only arrays. It pays off on every act and every update.")
    else:
        print("  Not worth refactoring: filter_jit is not what the calls are spending")
        print("  their time on. Look at the dispatch floor and the arithmetic instead.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
