"""Does ``PPO.act`` sample what it did when the key split ran eagerly?

``PPO.act`` used to split the stored RNG key eagerly (one dispatch), keep
the first half, and hand the second half to the jitted forward, which
split it again. The first split now runs inside the forward, which
returns the half to store. Same two halves, same consumers, so every
sample, log-probability and stored key must be bit for bit what the old
sequence produced. This replays that sequence -- the eager split around a
jitted copy of the old forward body -- next to the production ``act``,
over a chain of steps, for the Gaussian and the squashed Gaussian head,
stochastic and deterministic.

    python -m jaxrlworld.scripts.diag.gates.check_ppo_act_key_fold
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from jaxrlworld.rl.algorithms.ppo import PPO
from jaxrlworld.rl.configs.common_config_classes import DistributionType, PPOPolicyConfig
from jaxrlworld.rl.modules.policies.ppo_ac import PPOActorCritic

OBS_A, OBS_C, ACT, NUM_ENVS, NUM_STEPS = 6, 8, 3, 16, 4
CHAIN = 12


@eqx.filter_jit
def _old_forward(model, actor_obs, critic_obs, key):
    """The forward body as it was: ``key`` is already the second half."""
    key, subkey = jax.random.split(key)
    dist, _ = model.get_distribution(actor_obs, key=subkey)
    if dist.is_squashed:
        raw = dist.sample_raw(key)
        env = jnp.tanh(raw)
        log_prob = dist.log_prob_raw(raw)
    else:
        raw = dist.sample(key)
        env = raw
        log_prob = dist.log_prob(raw)
    values, _ = model.evaluate_value(critic_obs)
    return env, raw, log_prob, values.squeeze(-1)


@eqx.filter_jit
def _old_forward_deterministic(model, actor_obs, critic_obs, key):
    dist, _ = model.get_distribution(actor_obs, key=key)
    raw = dist.mean
    if dist.is_squashed:
        env = jnp.tanh(raw)
        log_prob = dist.log_prob_raw(raw)
    else:
        env = raw
        log_prob = dist.log_prob(raw)
    values, _ = model.evaluate_value(critic_obs)
    return env, raw, log_prob, values.squeeze(-1)


def make_ppo(distribution_type: DistributionType) -> PPO:
    pc = PPOPolicyConfig(distribution_type=distribution_type)
    model = PPOActorCritic(
        num_actor_obs=OBS_A,
        num_critic_obs=OBS_C,
        num_actions=ACT,
        actor_cfg=pc.actor,
        critic_cfg=pc.critic,
        init_noise_std=1.0,
        std_type=pc.std_type,
        distribution_type=pc.distribution_type,
        key=jax.random.PRNGKey(0),
        obs_normalization=False,
        obs_shapes={"actor": (OBS_A,), "critic": (OBS_C,)},
    )
    alg = PPO(actor_critic=model, num_learning_epochs=1, num_mini_batches=1, key=jax.random.PRNGKey(1))
    alg.init_storage(
        dict(
            num_envs=NUM_ENVS,
            num_transitions_per_env=NUM_STEPS,
            actor_obs_shape=[OBS_A],
            critic_obs_shape=[OBS_C],
            actions_shape=[ACT],
        )
    )
    return alg


def check_chain(chk, distribution_type: DistributionType, deterministic: bool) -> None:
    label = f"{distribution_type.value} {'deterministic' if deterministic else 'stochastic'}"
    alg = make_ppo(distribution_type)
    model = alg.train_state.model
    key = alg.train_state.key
    obs_key = jax.random.PRNGKey(3)
    worst = 0
    for step in range(CHAIN):
        obs_key, ka, kc = jax.random.split(obs_key, 3)
        a = jax.random.normal(ka, (NUM_ENVS, OBS_A))
        c = jax.random.normal(kc, (NUM_ENVS, OBS_C))

        # The old sequence: eager split, second half into the forward.
        key, subkey = jax.random.split(key)
        old = _old_forward_deterministic if deterministic else _old_forward
        env_ref, raw_ref, logp_ref, values_ref = old(model, a, c, subkey)

        env_new = alg.act(PPO.ActInput(a, c), deterministic=deterministic)
        t = alg.transition
        got = (env_new, t.actions, t.actions_log_prob, t.values)
        ref = (env_ref, raw_ref, logp_ref, values_ref)
        for x, y in zip(got, ref):
            worst = max(worst, int((np.asarray(x) != np.asarray(y)).sum()))
        worst = max(worst, int((np.asarray(alg.train_state.key) != np.asarray(key)).sum()))
    chk(
        f"{label}: {CHAIN} chained acts bit-identical (actions, raw, log_prob, values, key)",
        worst == 0,
        f"{worst} elements differ",
    )


def main() -> int:
    failures: list[str] = []

    def chk(name: str, ok: bool, detail: str = "") -> None:
        print(f"  [{'OK' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
        if not ok:
            failures.append(name)

    print("=" * 78)
    print("PPO ACT KEY FOLD")
    print("=" * 78)
    for dist_type in (DistributionType.GAUSSIAN, DistributionType.SQUASHED_GAUSSIAN):
        for deterministic in (False, True):
            check_chain(chk, dist_type, deterministic)
    print("=" * 78)
    print("=== RESULT: ALL OK ===" if not failures else f"=== RESULT: {len(failures)} FAILED ===")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
