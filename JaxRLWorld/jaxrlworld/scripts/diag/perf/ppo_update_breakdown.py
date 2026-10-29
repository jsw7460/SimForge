"""Where does PPO's learn time go?

At 16384 envs x 24 steps our learn is 0.215 s against rsl_rl's 0.157 s
on the same batch shape (5 epochs x 4 minibatches). This times the
learn phase's pieces separately, each fully drained:

  1. compute_returns          (GAE scan over the rollout)
  2. get_flat_batch/indices   (host-side views + permutation build)
  3. update_all_batches       (the 20-step scan: gather + grad + adam)
  4. _compute_metrics         (single jitted metrics program + transfer)

Collection runs once through the real runner to fill the storage with
real shapes; the update pieces then repeat on the same data.

  5. the actor alone: the rollout's act over one step's observations,
     fwd+bwd over one minibatch, and the auxiliary step when the actor
     declares one -- the rows that separate an actor architecture's
     cost from the rest of the update.

Usage:
    jaxpy -m jaxrlworld.scripts.diag.perf.ppo_update_breakdown --preset g1_29dof --sim mujoco --num-envs 16384
    jaxpy -m jaxrlworld.scripts.diag.perf.ppo_update_breakdown --preset <module>:<ConfigClass> --sim mujoco --num-envs 2048
"""

from __future__ import annotations

import argparse
import importlib
import statistics
import time

import jax
import jax.numpy as jnp

from jaxrlworld.rl.algorithms.ppo.metrics import PPOMetrics

_PRESETS: dict[str, tuple[str, str]] = {
    "g1_29dof": ("jaxrlworld.rl.configs.presets.g1_29dof.base", "G1FlatConfig"),
    "go2": ("jaxrlworld.rl.configs.presets.go2.base", "Go2FlatConfig"),
}


def _build_runner(preset: str, sim: str, num_envs: int):
    from jaxrlworld.rl.runners import BaseRunner

    if ":" in preset:
        mod_path, cls_name = preset.split(":", 1)
    else:
        mod_path, cls_name = _PRESETS[preset]
    cfg_cls = getattr(importlib.import_module(mod_path), cls_name)
    cfgs = cfg_cls(sim_type=sim, num_envs=num_envs).build()
    return BaseRunner.create_with_env(cfgs, use_wandb=False)


def _timed(fn, reps: int) -> list[float]:
    out = []
    for _ in range(reps):
        t0 = time.perf_counter()
        result = fn()
        jax.block_until_ready(result)
        out.append((time.perf_counter() - t0) * 1e3)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="g1_29dof")
    ap.add_argument("--sim", default="mujoco", choices=("genesis", "newton", "mujoco"))
    ap.add_argument("--num-envs", type=int, default=16384)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument(
        "--matmul-precision",
        default=None,
        choices=("default", "tensorfloat32", "float32", "highest"),
        help="Wrap the scan-only measurement in jax.default_matmul_precision.",
    )
    ap.add_argument(
        "--scan-epochs",
        type=int,
        default=None,
        help="Epoch count for the scan-only measurement (default: the preset's).",
    )
    args = ap.parse_args()

    runner = _build_runner(args.preset, args.sim, args.num_envs)
    alg = runner.alg

    # One real collection to fill the storage.
    obs = runner._get_initial_obs()
    data = runner._collect_experience(obs=obs, ep_infos=[])
    last_critic = data["last_obs"]["critic_obs"]

    # Keep the storage full across repeated update() calls.
    alg.storage.clear = lambda: None

    # ---- piece 1: compute_returns ------------------------------------
    # compute_returns returns None; wait on what it wrote (advantages,
    # returns and, under value normalization, the rewritten values), or
    # the row measures dispatch only and the GAE lands in piece 2.
    def returns():
        alg.compute_returns(last_critic)
        return (alg.storage.advantages, alg.storage.returns, alg.storage.values)

    t_returns = _timed(returns, args.reps)

    # ---- piece 2: flat batch + indices -------------------------------
    def flat_and_indices():
        flat = alg.storage.get_flat_batch()
        idx = alg.storage.get_minibatch_indices(
            num_minibatches=alg.num_mini_batches,
            num_epochs=alg.num_learning_epochs,
            key=jax.random.PRNGKey(0),
        )
        return idx

    t_batch = _timed(flat_and_indices, args.reps)

    # ---- piece 3 + 4: the full update (scan + metrics), then update
    # alone with metrics suppressed to split the two.
    t_update_full = _timed(lambda: (alg.update(), alg.train_state.opt_state)[1], args.reps)

    orig_metrics = alg._compute_metrics
    alg._compute_metrics = lambda *a, **k: PPOMetrics()
    t_update_nometrics = _timed(lambda: (alg.update(), alg.train_state.opt_state)[1], args.reps)
    alg._compute_metrics = orig_metrics

    # ---- piece 5: the jitted scan alone, python wrapper excluded -----
    # Replicates update()'s call to update_all_batches with fixed inputs
    # so the number is the compiled program's own wall time.
    import equinox as eqx

    from jaxrlworld.rl.algorithms.ppo.ppo import EmpiricalNormalization
    from jaxrlworld.rl.algorithms.ppo.update import update_all_batches

    flat_batch = alg.storage.get_flat_batch()
    batch_indices = alg.storage.get_minibatch_indices(
        num_minibatches=alg.num_mini_batches,
        num_epochs=args.scan_epochs or alg.num_learning_epochs,
        key=jax.random.PRNGKey(0),
    )
    desired_kl = alg.desired_kl if alg.desired_kl is not None else 1e10

    def scan_only():
        params, static = eqx.partition(
            alg.train_state.model,
            eqx.is_inexact_array,
            is_leaf=lambda x: isinstance(x, EmpiricalNormalization),
        )
        # opt_state is donated: rebuild a fresh one per rep so repeated
        # calls do not consume an invalidated buffer.
        opt_state = alg.optimizer.init(params)
        out = update_all_batches(
            params,
            static,
            opt_state,
            alg.optimizer,
            alg.clip_param,
            alg.value_loss_coef,
            alg.entropy_coef,
            alg.use_clipped_value_loss,
            alg.normalize_advantage_per_minibatch,
            alg.use_early_stop,
            desired_kl,
            alg.symmetry_spec,
            alg.symmetry_coef,
            alg.bound_loss_coef,
            alg.symmetry_augment,
            flat_batch,
            batch_indices,
            jax.random.PRNGKey(1),
        )
        return out[0]

    if args.matmul_precision:
        with jax.default_matmul_precision(args.matmul_precision):
            t_scan = _timed(scan_only, args.reps)
    else:
        t_scan = _timed(scan_only, args.reps)

    # ---- piece 6: the pre-refactor layout — gather every minibatch up
    # front, scan over stacked data with no in-scan gather. Splits the
    # lazy path's cost into [one big gather] + [gather-free scan] so the
    # in-scan gather's share of the 10.6 ms/grad-step is measurable.
    from jaxrlworld.scripts.diag.gates.check_ppo_minibatch_bitwise import _old_update_all_batches

    def pregather():
        return jax.tree.map(lambda x: x[batch_indices], flat_batch)

    t_pregather = _timed(pregather, args.reps)
    stacked = pregather()

    old_scan_jit = jax.jit(_old_update_all_batches, static_argnums=(3,))

    def old_scan():
        params, static = eqx.partition(
            alg.train_state.model,
            eqx.is_inexact_array,
            is_leaf=lambda x: isinstance(x, EmpiricalNormalization),
        )
        opt_state = alg.optimizer.init(params)
        out = old_scan_jit(params, static, opt_state, alg.optimizer, stacked, jax.random.PRNGKey(1))
        return out[0]

    t_old_scan = _timed(old_scan, args.reps)

    # ---- piece 7: the actor alone. One rollout step's act (what every
    # env step pays), the actor's fwd+bwd over one minibatch (what every
    # grad step pays on top of the critic), and the auxiliary step the
    # update runs once when the actor declares one.
    model = alg.train_state.model
    step_obs = jax.tree.map(lambda x: x[: args.num_envs], flat_batch.actor_observations)
    minibatch_obs = jax.tree.map(lambda x: x[batch_indices[-1]], flat_batch.actor_observations)

    @eqx.filter_jit
    def act_step(m, obs, key):
        return m.act(obs, key)[0]

    t_act = _timed(lambda: act_step(model, step_obs, jax.random.PRNGKey(2)), args.reps)

    def actor_loss(params, static, obs, key):
        m = eqx.combine(params, static)
        keys = jax.random.split(key, jax.tree.leaves(obs)[0].shape[0])
        mean, _ = jax.vmap(m.actor)(m._normalize_actor_obs(obs), key=keys)
        return jnp.mean(jnp.square(mean))

    actor_grad = eqx.filter_jit(eqx.filter_grad(actor_loss))
    params, static = eqx.partition(model, eqx.is_inexact_array, is_leaf=lambda x: isinstance(x, EmpiricalNormalization))
    t_actor_grad = _timed(lambda: actor_grad(params, static, minibatch_obs, jax.random.PRNGKey(3)), args.reps)

    t_aux = None
    if model.actor.has_auxiliary_loss:

        def aux():
            opt_state = alg.optimizer.init(params)
            return alg._auxiliary_step(params, static, opt_state, minibatch_obs)[0]

        t_aux = _timed(aux, args.reps)

    def rep(name, samples):
        print(
            f"  {name:<28} mean {statistics.mean(samples):8.2f} ms   "
            f"median {statistics.median(samples):8.2f} ms   min {min(samples):8.2f}"
        )

    print(f"\nPPO UPDATE BREAKDOWN  [{args.preset} {args.sim} @{args.num_envs}]")
    print(f"  epochs x minibatches: {alg.num_learning_epochs} x {alg.num_mini_batches}")
    rep("compute_returns", t_returns)
    rep("flat batch + indices", t_batch)
    rep("update() full", t_update_full)
    rep("update() metrics off", t_update_nometrics)
    label = f"scan only (epochs={args.scan_epochs or alg.num_learning_epochs}, mm={args.matmul_precision or '-'})"
    rep(label, t_scan)
    rep("pre-gather (all minibatches)", t_pregather)
    rep("old scan (no in-scan gather)", t_old_scan)
    rep("actor act (one env step)", t_act)
    rep("actor fwd+bwd (one minibatch)", t_actor_grad)
    if t_aux is not None:
        rep("auxiliary step (per update)", t_aux)
    else:
        print("  auxiliary step               (actor declares none)")
    print("  (metrics cost = full - metrics-off; scan cost ~= metrics-off - returns pieces)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
