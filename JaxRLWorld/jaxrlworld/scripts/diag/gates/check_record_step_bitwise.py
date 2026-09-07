"""Is the one-dispatch step record bitwise what the eager sequence wrote?

``RolloutStorage.record_step`` folds the flag casts, ``dones``, the
timeout bootstrap bonus, the episode-start row and the row write into
one jitted program. This replays the sequence it replaced — the eager
ops of the old ``PPO.process_env_step`` / ``_handle_timeout`` around
``_write_step`` — on the same random inputs and demands every rollout
buffer match bit for bit, on this machine's own backend.

Cases: bootstrap with an env-provided mask, with the
``truncated & ~terminated`` fallback, a step without a terminal
observation, the per-epoch-GAE path (mask row instead of bonus), and a
dict (vision-style) observation.

    jaxpy -m jaxrlworld.scripts.diag.gates.check_record_step_bitwise
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from jaxrlworld.rl.storages.rollout_storage import RolloutStorage

OBS, ACT, N, T = 24, 8, 64, 12
IMG = (6, 6)
GAMMA = 0.99


def _old_process(storage, last_dones, fields, rewards, terminated, truncated, infos, recompute_gae):
    """The eager sequence ``process_env_step`` ran before ``record_step``."""
    terminated = terminated.astype(jnp.bool_)
    truncated = truncated.astype(jnp.bool_)
    dones = terminated | truncated
    episode_starts = jnp.zeros_like(dones) if last_dones is None else last_dones
    if recompute_gae:
        bootstrap_mask = infos.get("bootstrap_mask")
        if bootstrap_mask is None:
            bootstrap_mask = truncated & ~terminated
        else:
            bootstrap_mask = bootstrap_mask.astype(jnp.bool_)
        trunc_no_reset = infos.get("trunc_no_reset_mask")
        mask = bootstrap_mask if trunc_no_reset is None else (bootstrap_mask | trunc_no_reset.astype(jnp.bool_))
        storage.trunc_masks = storage.trunc_masks.at[storage.step].set(mask)
    elif "bootstrap_values" in infos:
        bootstrap_mask = infos.get("bootstrap_mask")
        if bootstrap_mask is None:
            bootstrap_mask = truncated & ~terminated
        else:
            bootstrap_mask = bootstrap_mask.astype(jnp.bool_)
        bonus = bootstrap_mask.astype(rewards.dtype) * (GAMMA * infos["bootstrap_values"])
        rewards = rewards + bonus
    actor_obs, critic_obs, actions, values, log_probs, mu, sigma = fields
    storage.add_transition(actor_obs, critic_obs, actions, rewards, dones, episode_starts, values, log_probs, mu, sigma)
    return dones


def _buffers(storage):
    return {
        "actor_obs": storage.actor_obs,
        "critic_obs": storage.critic_obs,
        "actions": storage.actions,
        "rewards": storage.rewards,
        "dones": storage.dones,
        "episode_starts": storage.episode_starts,
        "values": storage.values,
        "log_probs": storage.log_probs,
        "mu": storage.mu,
        "sigma": storage.sigma,
        "trunc_masks": storage.trunc_masks,
    }


def _random_step(key, dict_obs: bool):
    ks = jax.random.split(key, 12)
    if dict_obs:
        actor = {"actor": jax.random.normal(ks[0], (N, OBS)), "cam": jax.random.normal(ks[1], (N, *IMG))}
        critic = {"critic": jax.random.normal(ks[2], (N, OBS + 4)), "cam": actor["cam"]}
    else:
        actor = jax.random.normal(ks[0], (N, OBS))
        critic = jax.random.normal(ks[2], (N, OBS + 4))
    fields = (
        actor,
        critic,
        jax.random.normal(ks[3], (N, ACT)),
        jax.random.normal(ks[4], (N,)),
        jax.random.normal(ks[5], (N,)),
        jax.random.normal(ks[6], (N, ACT)),
        jax.nn.softplus(jax.random.normal(ks[7], (N, ACT))),
    )
    rewards = jax.random.normal(ks[8], (N,))
    terminated = (jax.random.uniform(ks[9], (N,)) < 0.15).astype(jnp.uint8)
    truncated = (jax.random.uniform(ks[10], (N,)) < 0.15).astype(jnp.uint8)
    values = jax.random.normal(ks[11], (N,)) * 5.0
    return fields, rewards, terminated, truncated, values


def run_case(name: str, dict_obs: bool, mode: str, seed: int) -> None:
    key = jax.random.PRNGKey(seed)
    actor_shape = {"actor": (OBS,), "cam": IMG} if dict_obs else (OBS,)
    critic_shape = {"critic": (OBS + 4,), "cam": IMG} if dict_obs else (OBS + 4,)
    old = RolloutStorage(N, T, actor_shape, critic_shape, (ACT,))
    new = RolloutStorage(N, T, actor_shape, critic_shape, (ACT,))
    recompute = mode == "recompute_gae"
    last_old = None if mode == "no_last_dones" else jnp.ones((N,), dtype=jnp.bool_)
    last_new = last_old
    for t in range(T):
        key, sub = jax.random.split(key)
        fields, rewards, terminated, truncated, values = _random_step(sub, dict_obs)
        has_final = t % 3 != 1
        with_mask = mode == "env_mask" or (mode == "recompute_gae" and t % 2 == 0)
        key, km, kt = jax.random.split(key, 3)
        env_mask = (jax.random.uniform(km, (N,)) < 0.2).astype(jnp.uint8) if with_mask else None
        trunc_no_reset = (jax.random.uniform(kt, (N,)) < 0.1).astype(jnp.uint8) if recompute and t % 4 == 0 else None

        infos = {}
        if has_final:
            infos["bootstrap_values"] = values
            if env_mask is not None:
                infos["bootstrap_mask"] = env_mask
        if trunc_no_reset is not None:
            infos["trunc_no_reset_mask"] = trunc_no_reset
        last_old = _old_process(old, last_old, fields, rewards, terminated, truncated, infos, recompute)

        # As in the runner: the env mask travels with the terminal
        # observation, so a step without one falls back on both paths.
        last_new = new.record_step(
            *fields,
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            last_dones=last_new,
            bootstrap_values=values if (has_final and not recompute) else None,
            bootstrap_mask=infos.get("bootstrap_mask"),
            trunc_no_reset=trunc_no_reset,
            gamma=GAMMA,
            recompute_gae=recompute,
        )
        assert np.array_equal(np.asarray(last_old), np.asarray(last_new)), f"{name}: dones differ at t={t}"

    assert old.step == new.step == T and int(new._index) == T
    for field, a in _buffers(old).items():
        b = _buffers(new)[field]
        flat_a, flat_b = jax.tree.leaves(a), jax.tree.leaves(b)
        for x, y in zip(flat_a, flat_b, strict=True):
            assert np.array_equal(np.asarray(x), np.asarray(y)), f"{name}: {field} differs"
    print(f"  {name:<28} {T} steps bitwise")


def main() -> None:
    print(f"record_step vs eager sequence ({jax.default_backend()})")
    run_case("fallback mask, vector obs", False, "fallback", 0)
    run_case("env mask, vector obs", False, "env_mask", 1)
    run_case("first step (no last_dones)", False, "no_last_dones", 2)
    run_case("recompute_gae mask row", False, "recompute_gae", 3)
    run_case("env mask, dict obs", True, "env_mask", 4)
    print("PASS")


if __name__ == "__main__":
    main()
