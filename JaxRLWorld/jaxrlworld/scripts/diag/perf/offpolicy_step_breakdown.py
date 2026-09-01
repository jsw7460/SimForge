"""Where an off-policy step's time actually goes.

An off-policy iteration is one environment step, so anything the loop
does per iteration is paid per step, and the pieces are small enough
that guessing which one dominates has been wrong more often than right.
This times them apart instead, by running the same loop with pieces
removed:

  update      updates on an already-filled buffer, one sampled batch
              reused, so nothing touches the replay buffer or the env
  sample      a fresh batch per update — adds the host-side gather and
              the upload
  collect     the whole loop — adds the env step, the action handoff,
              and storing the transition

Each phase's cost is the difference from the one above it. That says
where to spend effort: moving the replay buffer onto the device only
helps if ``sample`` and the store half of ``collect`` are where the time
is, and doing the work first to find out has already cost a day once.

Deliberately built on a plain SyncVectorEnv rather than the runner, so
it measures the algorithm and the buffer without a simulator in the way.

Run on the training box:
    jaxpy -m jaxrlworld.scripts.diag.perf.offpolicy_step_breakdown
    jaxpy -m jaxrlworld.scripts.diag.perf.offpolicy_step_breakdown --algo td3 --steps 4000
"""

import argparse
import time

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium.vector import AutoresetMode, SyncVectorEnv

from jaxrlworld.rl.algorithms.base import ActInput
from jaxrlworld.rl.algorithms.sac.sac import SAC
from jaxrlworld.rl.algorithms.td3.td3 import TD3
from jaxrlworld.rl.configs.common_config_classes import Activation, DefaultInit, MLPActorCfg, MLPCriticCfg
from jaxrlworld.rl.modules.policies.sac_ac import SACActorCritic
from jaxrlworld.rl.modules.policies.td3_ac import TD3ActorCritic


def _build(algo: str, obs_dim: int, act_dim: int, batch_size: int):
    nets = dict(
        actor_cfg=MLPActorCfg(hidden_dims=[256, 256], activation=Activation.RELU, init=DefaultInit()),
        critic_cfg=MLPCriticCfg(hidden_dims=[256, 256], activation=Activation.RELU, init=DefaultInit()),
    )
    if algo == "sac":
        model = SACActorCritic(
            num_actor_obs=obs_dim, num_critic_obs=obs_dim, num_actions=act_dim, key=jax.random.PRNGKey(0), **nets
        )
        return SAC(actor_critic=model, batch_size=batch_size, key=jax.random.PRNGKey(1))
    model = TD3ActorCritic(
        num_actor_obs=obs_dim, num_critic_obs=obs_dim, num_actions=act_dim, key=jax.random.PRNGKey(0), **nets
    )
    return TD3(actor_critic=model, batch_size=batch_size, key=jax.random.PRNGKey(1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--algo", choices=("sac", "td3"), default="sac")
    ap.add_argument("--task", default="HalfCheetah-v5")
    ap.add_argument("--steps", type=int, default=3000, help="timed steps per phase")
    ap.add_argument("--warmup", type=int, default=200, help="steps dropped before timing (compilation)")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--prefill", type=int, default=5000)
    args = ap.parse_args()

    env = SyncVectorEnv([lambda: gym.make(args.task)], autoreset_mode=AutoresetMode.SAME_STEP)
    obs, _ = env.reset(seed=0)
    obs_dim, act_dim = obs.shape[1], env.single_action_space.shape[0]
    low, high = env.single_action_space.low, env.single_action_space.high

    alg = _build(args.algo, obs_dim, act_dim, args.batch_size)
    alg.init_storage(
        {
            "num_envs": 1,
            "actor_obs_shape": [obs_dim],
            "critic_obs_shape": [obs_dim],
            "actions_shape": [act_dim],
            "size_per_env": 1_000_000,
            "n_steps": 1,
            "seed": 0,
        }
    )

    print("=" * 78)
    print("  OFF-POLICY STEP BREAKDOWN")
    print(f"  algo: {args.algo}   task: {args.task}   batch: {args.batch_size}")
    print(f"  jax backend: {jax.default_backend()}   timed steps/phase: {args.steps:,}")
    print("=" * 78)

    # Fill the buffer with real transitions so sampling reads a realistic
    # spread of the ring rather than one hot row.
    rng = np.random.default_rng(0)
    actor_obs = jnp.asarray(obs, jnp.float32)
    for _ in range(args.prefill):
        action = rng.uniform(low, high, size=(1, act_dim)).astype(np.float32)
        nxt, rew, term, trunc, _ = env.step(action)
        nxt_j = jnp.asarray(nxt, jnp.float32)
        alg.replay_buffer.store_parallel(
            actor_obs,
            actor_obs,
            jnp.asarray(action),
            jnp.asarray(rew, jnp.float32),
            nxt_j,
            nxt_j,
            jnp.asarray(term),
            jnp.asarray(trunc),
        )
        actor_obs = nxt_j
    print(f"  buffer prefilled with {alg.replay_buffer.size:,} transitions\n")

    def timed(label: str, body, n: int) -> float:
        for _ in range(args.warmup):
            body()
        jax.block_until_ready(alg.train_state.model)
        t0 = time.perf_counter()
        for _ in range(n):
            body()
        jax.block_until_ready(alg.train_state.model)
        dt = (time.perf_counter() - t0) / n * 1e6
        print(f"  {label:<10s} {dt:8.1f} us/step")
        return dt

    fixed_batch = alg.sample_batch(args.batch_size)

    state_obs = actor_obs

    def phase_act():
        # One tiny forward: batch 1, three dense layers. Anything much
        # above a few tens of microseconds here is per-call host work,
        # not arithmetic — which is the point of measuring it.
        alg.act(ActInput(state_obs, state_obs), deterministic=False)

    def phase_update():
        alg.update(fixed_batch, build_metrics=False)

    def phase_sample():
        alg.update(alg.sample_batch(args.batch_size), build_metrics=False)

    state = {"obs": actor_obs}

    def phase_collect():
        action = alg.act(ActInput(state["obs"], state["obs"]), deterministic=False)
        nxt, rew, term, trunc, _ = env.step(np.clip(np.asarray(action), low, high))
        nxt_j = jnp.asarray(nxt, jnp.float32)
        alg.replay_buffer.store_parallel(
            state["obs"],
            state["obs"],
            action,
            jnp.asarray(rew, jnp.float32),
            nxt_j,
            nxt_j,
            jnp.asarray(term),
            jnp.asarray(trunc),
        )
        state["obs"] = nxt_j
        alg.update(alg.sample_batch(args.batch_size), build_metrics=False)

    t_act = timed("act only", phase_act, args.steps)
    t_update = timed("update", phase_update, args.steps)
    t_sample = timed("+sample", phase_sample, args.steps)
    t_collect = timed("+collect", phase_collect, args.steps)

    print("\n" + "-" * 78)
    print("  attribution (difference from the phase above)")
    print(f"    update      {t_update:8.1f} us  {100 * t_update / t_collect:5.1f}%")
    print(f"      of which one act() call costs {t_act:.1f} us — a batch-1 forward")
    print(f"    sampling    {t_sample - t_update:8.1f} us  {100 * (t_sample - t_update) / t_collect:5.1f}%")
    print(f"    env+store   {t_collect - t_sample:8.1f} us  {100 * (t_collect - t_sample) / t_collect:5.1f}%")
    print(f"    total       {t_collect:8.1f} us  -> {1e6 / t_collect:,.0f} steps/s")
    print("-" * 78)
    print("  Moving the replay buffer onto the device attacks 'sampling' and the")
    print("  store half of 'env+store'. If those are small, it is not worth doing.")
    print("  The phases only synchronise at the end, so these are host issue rates:")
    print("  a number far above the arithmetic means the GPU is waiting on Python.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
