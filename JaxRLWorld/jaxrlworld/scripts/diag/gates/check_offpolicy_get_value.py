"""Gate: the off-policy algorithms' public ``get_value`` works and agrees.

``SAC.get_value`` took the critic observation alone and its jitted helper
called ``model.evaluate(critic_obs, key=key)``; ``SACActorCritic.evaluate``
needs the actor observation too (the value is ``min(Q1, Q2)`` at the
deterministic action), so every call raised a TypeError. Nothing in the
tree called it, which is how it stayed broken. TD3 and FastTD3 already
took ``(actor_obs, critic_obs)``.

For each of SAC, TD3 and FastTD3 on a small MLP model: ``get_value``
returns one finite value per row (``(N, 1)``; FastTD3's distributional
critic returns ``(N,)``), matches ``model.evaluate`` on the same inputs
and key to float32 rounding (the jitted and the eager reduction over the
C51 atoms differ in the last bits), and for SAC equals ``min(Q1, Q2)``
at the model's own deterministic action.

Run::

    python -m jaxrlworld.scripts.diag.gates.check_offpolicy_get_value
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from jaxrlworld.rl.algorithms.fast_td3.fast_td3 import FastTD3
from jaxrlworld.rl.algorithms.sac.sac import SAC
from jaxrlworld.rl.algorithms.td3.td3 import TD3
from jaxrlworld.rl.configs.common_config_classes import Activation, DefaultInit, MLPActorCfg, MLPCriticCfg
from jaxrlworld.rl.modules.policies.fast_td3_ac import FastTD3ActorCritic
from jaxrlworld.rl.modules.policies.sac_ac import SACActorCritic
from jaxrlworld.rl.modules.policies.td3_ac import TD3ActorCritic

OBS_A, OBS_C, ACT, N = 6, 8, 3, 16


def nets() -> dict:
    return dict(
        actor_cfg=MLPActorCfg(hidden_dims=[32, 32], activation=Activation.RELU, init=DefaultInit()),
        critic_cfg=MLPCriticCfg(hidden_dims=[32, 32], activation=Activation.RELU, init=DefaultInit()),
    )


def build(name: str):
    key = jax.random.PRNGKey(0)
    if name == "sac":
        model = SACActorCritic(num_actor_obs=OBS_A, num_critic_obs=OBS_C, num_actions=ACT, key=key, **nets())
        return SAC(actor_critic=model, batch_size=8, key=jax.random.PRNGKey(1))
    if name == "td3":
        model = TD3ActorCritic(num_actor_obs=OBS_A, num_critic_obs=OBS_C, num_actions=ACT, key=key, **nets())
        return TD3(actor_critic=model, batch_size=8, key=jax.random.PRNGKey(1))
    model = FastTD3ActorCritic(num_actor_obs=OBS_A, num_critic_obs=OBS_C, num_actions=ACT, key=key, **nets())
    return FastTD3(actor_critic=model, num_envs=N, batch_size=8, key=jax.random.PRNGKey(1))


def main() -> int:
    failures: list[str] = []

    def chk(name: str, ok: bool, detail: str = "") -> None:
        print(f"  [{'OK' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
        if not ok:
            failures.append(name)

    print("=" * 78)
    print("OFF-POLICY get_value")
    print("=" * 78)
    actor_obs = jax.random.normal(jax.random.PRNGKey(2), (N, OBS_A))
    critic_obs = jax.random.normal(jax.random.PRNGKey(3), (N, OBS_C))

    for name in ("sac", "td3", "fast_td3"):
        print(f"\n-- {name} --")
        alg = build(name)
        key_before = alg.train_state.key
        values = alg.get_value(actor_obs, critic_obs)
        chk(
            f"{name}: returns one finite value per row",
            values.shape[0] == N and values.size == N and bool(jnp.all(jnp.isfinite(values))),
            str(values.shape),
        )
        # The algorithm split its key once; evaluate with the half it used.
        _, subkey = jax.random.split(key_before)
        ref = alg.train_state.model.evaluate(actor_obs, critic_obs, key=subkey)
        chk(
            f"{name}: matches model.evaluate on the same inputs",
            values.shape == ref.shape and np.allclose(np.asarray(values), np.asarray(ref), rtol=1e-5, atol=1e-6),
            f"max |d| = {float(jnp.abs(values - ref).max()):.2e}",
        )
        if name == "sac":
            model = alg.train_state.model
            actions, _ = model.act(actor_obs, key=subkey, deterministic=True)
            q = jnp.minimum(model.critic1_forward(critic_obs, actions), model.critic2_forward(critic_obs, actions))
            chk(
                "sac: equals min(Q1, Q2) at the deterministic action", np.array_equal(np.asarray(values), np.asarray(q))
            )

    print("=" * 78)
    print("=== RESULT: ALL OK ===" if not failures else f"=== RESULT: {len(failures)} FAILED ===")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
