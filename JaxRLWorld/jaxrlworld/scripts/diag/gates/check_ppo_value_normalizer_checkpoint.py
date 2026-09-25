"""Gate: PPO's value-target normalizer survives a checkpoint round trip.

With ``use_value_normalization=True`` the critic learns normalized returns
and its outputs are only meaningful against the normalizer's running
statistics, which live outside the model. No simulator:

1. a PPO whose normalizer has advanced is saved and loaded into a fresh
   PPO; the statistics come back exactly;
2. loading a checkpoint without the statistics into a normalizing PPO is
   refused;
3. loading a normalizing checkpoint into a PPO with normalization off is
   refused;
4. a PPO without normalization saves and loads as before.

Run::

    python -m jaxrlworld.scripts.diag.gates.check_ppo_value_normalizer_checkpoint
"""

from __future__ import annotations

import os
import sys
import tempfile

import jax
import jax.numpy as jnp
import numpy as np

from jaxrlworld.rl.algorithms.ppo import PPO
from jaxrlworld.rl.configs.common_config_classes import PPOPolicyConfig
from jaxrlworld.rl.modules.policies.ppo_ac import PPOActorCritic

OBS_A, OBS_C, ACT = 6, 8, 3


def make_ppo(key, use_value_normalization: bool) -> PPO:
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
    return PPO(actor_critic=model, use_value_normalization=use_value_normalization, key=key)


def stats(ppo: PPO) -> tuple[np.ndarray, np.ndarray, float]:
    n = ppo.value_normalizer
    return np.asarray(n.mean), np.asarray(n.var), float(n.count)


def main() -> int:
    failures: list[str] = []

    def chk(name: str, ok: bool, detail: str = "") -> None:
        print(f"  [{'OK' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
        if not ok:
            failures.append(name)

    key = jax.random.PRNGKey(0)

    print("=== 1. statistics round trip ===")
    trained = make_ppo(key, use_value_normalization=True)
    returns = jax.random.normal(jax.random.PRNGKey(1), (512, 1)) * 40.0 + 120.0
    trained.value_normalizer = trained.value_normalizer.update(returns)
    mean0, var0, count0 = stats(trained)
    chk(
        "normalizer advanced before save",
        abs(mean0[0] - 120.0) < 10 and count0 > 500,
        f"mean {mean0[0]:.2f} count {count0}",
    )
    with tempfile.TemporaryDirectory() as d:
        meta = trained.save_train_state(d)
        chk("metadata flags value normalization", meta["value_normalization"] is True)
        chk("value_normalizer.eqx written", os.path.isfile(os.path.join(d, "value_normalizer.eqx")))
        fresh = make_ppo(jax.random.PRNGKey(5), use_value_normalization=True)
        mean_f, _, count_f = stats(fresh)
        chk("fresh PPO starts from default statistics", count_f < 1.0 and abs(mean_f[0]) < 1e-6)
        fresh.load_train_state(d, meta)
        mean1, var1, count1 = stats(fresh)
        chk(
            "loaded statistics equal the saved ones",
            np.array_equal(mean0, mean1) and np.array_equal(var0, var1) and count0 == count1,
            f"mean {mean1[0]:.4f} var {var1[0]:.4f} count {count1}",
        )
        x = jnp.array([[150.0]])
        chk(
            "loaded normalizer normalizes like the saved one",
            float(jnp.abs(trained.value_normalizer.normalize(x) - fresh.value_normalizer.normalize(x)).max()) == 0.0,
        )

        print("\n=== 2. missing statistics are refused ===")
        os.remove(os.path.join(d, "value_normalizer.eqx"))
        try:
            make_ppo(key, use_value_normalization=True).load_train_state(d, meta)
            chk("missing value_normalizer.eqx raises", False, "loaded")
        except FileNotFoundError as e:
            chk("missing value_normalizer.eqx raises", True, str(e)[:70])

        print("\n=== 3. normalization mismatch is refused ===")
        try:
            make_ppo(key, use_value_normalization=False).load_train_state(d, meta)
            chk("normalizing checkpoint into a non-normalizing PPO raises", False, "loaded")
        except ValueError as e:
            chk("normalizing checkpoint into a non-normalizing PPO raises", True, str(e)[:70])

    print("\n=== 4. no normalization: unchanged path ===")
    plain = make_ppo(key, use_value_normalization=False)
    with tempfile.TemporaryDirectory() as d:
        meta = plain.save_train_state(d)
        chk("no normalizer file written", not os.path.isfile(os.path.join(d, "value_normalizer.eqx")))
        chk("metadata flag false", meta["value_normalization"] is False)
        make_ppo(jax.random.PRNGKey(9), use_value_normalization=False).load_train_state(d, meta)
        chk("plain round trip loads", True)

    print(f"\n=== RESULT: {'ALL OK' if not failures else f'{len(failures)} FAILED: {failures}'} ===")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
