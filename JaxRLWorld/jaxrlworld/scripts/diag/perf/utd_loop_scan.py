"""Is the update-to-data loop worth scanning?

An off-policy iteration takes ``num_gradient_steps`` updates — 200 on
go2/newton/sac — and the runner drives them from Python: sample, update,
sample, update. Each pass is three jit calls, so an iteration issues six
hundred of them, and between each the accelerator has nothing queued
while Python decides what to do next.

PPO does not work this way. Its whole update, ten epochs over thirty-two
minibatches, is a single ``lax.scan`` inside a single call, which is why
its learn time is what it is.

The same shape is available to SAC now that the replay buffer can live
on the device: with sampling no longer leaving the accelerator, the
whole UTD loop can be one program. What that is worth depends on how
much of an update is Python rather than arithmetic, which is the thing
to measure rather than assume.

Both paths do exactly the same work on the same buffer — sample a batch,
update, repeat — and are timed with one synchronisation at the end, so
the numbers are the rate the host can drive the loop.

Defaults are go2/newton/sac's shapes. Stop any training first; this
wants the GPU to itself.

Run:
    jaxpy -m jaxrlworld.scripts.diag.perf.utd_loop_scan
    jaxpy -m jaxrlworld.scripts.diag.perf.utd_loop_scan --updates 50 --batch-size 4096
"""

import argparse
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from jaxrlworld.rl.algorithms.sac.sac import SAC
from jaxrlworld.rl.algorithms.sac.update import update_all
from jaxrlworld.rl.configs.common_config_classes import Activation, DefaultInit, MLPActorCfg, MLPCriticCfg
from jaxrlworld.rl.modules.policies.sac_ac import SACActorCritic
from jaxrlworld.rl.storages.device_replay_buffer import _gather_batch, _sample_indices


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num-envs", type=int, default=1024)
    ap.add_argument("--obs-dim", type=int, default=48)
    ap.add_argument("--act-dim", type=int, default=12)
    ap.add_argument("--hidden", type=int, nargs="+", default=[1024, 512, 256])
    ap.add_argument("--buffer-size", type=int, default=5_000_000)
    ap.add_argument("--n-steps", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--updates", type=int, default=200, help="num_gradient_steps")
    ap.add_argument("--reps", type=int, default=5, help="timed iterations per path")
    args = ap.parse_args()

    size_per_env = args.buffer_size // args.num_envs
    nets = dict(
        actor_cfg=MLPActorCfg(hidden_dims=list(args.hidden), activation=Activation.SILU, init=DefaultInit()),
        critic_cfg=MLPCriticCfg(hidden_dims=list(args.hidden), activation=Activation.SILU, init=DefaultInit()),
    )
    model = SACActorCritic(
        num_actor_obs=args.obs_dim,
        num_critic_obs=args.obs_dim,
        num_actions=args.act_dim,
        key=jax.random.PRNGKey(0),
        **nets,
    )
    alg = SAC(actor_critic=model, batch_size=args.batch_size, gamma=0.97, tau=0.003, key=jax.random.PRNGKey(1))
    alg.init_storage(
        {
            "num_envs": args.num_envs,
            "actor_obs_shape": [args.obs_dim],
            "critic_obs_shape": [args.obs_dim],
            "actions_shape": [args.act_dim],
            "size_per_env": size_per_env,
            "n_steps": args.n_steps,
            "seed": 0,
            "buffer_device": "device",
        }
    )

    print("=" * 78)
    print("  UTD LOOP — PYTHON LOOP VS ONE SCAN")
    print(f"  backend {jax.default_backend()}   envs {args.num_envs:,}   ring {size_per_env:,}")
    print(f"  net {args.hidden}   batch {args.batch_size:,}   updates/iteration {args.updates}")
    print("=" * 78)

    rng = np.random.default_rng(0)
    fill = min(size_per_env, 4 * args.n_steps + 64)
    for _ in range(fill):
        o = jnp.asarray(rng.normal(size=(args.num_envs, args.obs_dim)), jnp.float32)
        alg.replay_buffer.store_parallel(
            o,
            o,
            jnp.asarray(rng.normal(size=(args.num_envs, args.act_dim)), jnp.float32),
            jnp.asarray(rng.normal(size=(args.num_envs,)), jnp.float32),
            o,
            o,
            jnp.asarray(rng.random(args.num_envs) < 0.02),
            jnp.asarray(rng.random(args.num_envs) < 0.02),
        )
    buf = alg.replay_buffer
    print(f"  buffer filled to {buf.filled_size:,} steps/env\n")

    # ---- path A: what the runner does today -------------------------
    def python_loop():
        for _ in range(args.updates):
            alg.update(alg.sample_batch(args.batch_size), build_metrics=False)

    # ---- path B: the same work as one program -----------------------
    sample_static = (buf.num_envs, buf.size_per_env, buf.n_steps, args.batch_size, buf.filled_size >= buf.size_per_env)
    fill_state = (jnp.asarray(buf.ptr), jnp.asarray(buf.filled_size))

    @eqx.filter_jit
    def scanned(state, buffers, sample_key):
        def body(carry, _):
            model, tc1, tc2, cos, aos, alos, lec, key, skey = carry
            skey, sub = jax.random.split(skey)
            indices = _sample_indices(sub, fill_state, *sample_static)
            batch = _gather_batch(buffers, indices, buf.size_per_env, buf.n_steps, buf.gamma)
            model, tc1, tc2, cos, aos, alos, lec, key, *_ = update_all(
                model,
                tc1,
                tc2,
                cos,
                aos,
                alos,
                lec,
                batch,
                key,
                alg.critic_optimizer,
                alg.actor_optimizer,
                alg.alpha_optimizer,
                alg.gamma,
                alg.tau,
                alg.target_entropy,
                0.0,
                True,
                alg.auto_entropy,
            )
            return (model, tc1, tc2, cos, aos, alos, lec, key, skey), None

        out, _ = jax.lax.scan(body, state, None, length=args.updates)
        return out

    ts = alg.train_state
    scan_state = (
        ts.model,
        ts.target_critic1_params,
        ts.target_critic2_params,
        ts.critic_opt_state,
        ts.actor_opt_state,
        ts.alpha_opt_state,
        ts.log_ent_coef,
        ts.key,
        jax.random.PRNGKey(7),
    )

    def timed(label, body, warmup=1):
        for _ in range(warmup):
            out = body()
        jax.block_until_ready(out if out is not None else alg.train_state.model)
        t0 = time.perf_counter()
        for _ in range(args.reps):
            out = body()
        jax.block_until_ready(out if out is not None else alg.train_state.model)
        dt = (time.perf_counter() - t0) / args.reps
        print(f"  {label:<14s} {dt * 1e3:8.1f} ms/iteration   {dt / args.updates * 1e3:6.2f} ms/update")
        return dt

    t_loop = timed("python loop", lambda: python_loop())
    t_scan = timed("one scan", lambda: scanned(scan_state, buf.buffers, jax.random.PRNGKey(7)))
    # The two above are the A/B of the strategies. This one is what the
    # runner actually calls, so it says whether the shipped code takes
    # the scan or quietly falls back to the loop.
    t_prod = timed("update_many", lambda: alg.update_many(args.updates, args.batch_size, build_metrics=False))

    print("\n" + "-" * 78)
    saved = t_loop - t_scan
    print(f"  scanning saves {saved * 1e3:7.1f} ms per iteration  ({100 * saved / t_loop:+.0f}%)")
    print(f"  which is {saved / args.updates * 1e6:6.0f} us per update of host-side cost")
    took_scan = abs(t_prod - t_scan) < abs(t_prod - t_loop)
    print(
        f"  update_many took the {'scan' if took_scan else 'LOOP'} path"
        f" ({t_prod * 1e3:.0f} ms against {t_scan * 1e3:.0f} scanned / {t_loop * 1e3:.0f} looped)"
    )
    print("-" * 78)
    if saved > 0.1 * t_loop:
        print("  Worth doing: drive the UTD loop from lax.scan when num_gradient_steps > 1.")
    else:
        print("  Not worth doing: the loop is already spending its time on arithmetic.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
