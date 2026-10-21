"""Gate: a resumed off-policy run continues its optimizer, bit for bit.

SAC, TD3, FastTD3 and TD-MPC2 checkpoints held the networks and their
targets and not the optimizer states, so a resume re-initialized every
Adam: zero moments, step count 1. PPO had the same gap, closed earlier;
the four now save and restore through the shared helpers in
``algorithms.base`` (``optimizer_state.eqx`` next to the model, flagged
in the metadata).

No simulator: each algorithm is fed synthetic replay batches that are a
deterministic function of the update index, so an uninterrupted run and
a saved-then-resumed run must agree exactly. For each algorithm:

1. after some updates the checkpoint flags and writes the optimizer
   state, and a fresh algorithm reads it back leaf for leaf (the fresh
   state differed before the load);
2. the resumed and the uninterrupted algorithm, fed the same further
   batches, end with bit-identical parameters, optimizer states and key;
3. the old behaviour (optimizer re-initialized) does not reproduce the
   uninterrupted run;
4. a checkpoint without the flag loads with the fresh state and says so;
   a flag without the file is refused.

Run::

    python -m jaxrlworld.scripts.diag.gates.check_offpolicy_optimizer_state_checkpoint
"""

from __future__ import annotations

import os
import tempfile

import jax
import jax.numpy as jnp
import numpy as np

from jaxrlworld.rl.algorithms.base import OPTIMIZER_STATE_FILE
from jaxrlworld.rl.algorithms.fast_td3.fast_td3 import FastTD3
from jaxrlworld.rl.algorithms.sac.sac import SAC
from jaxrlworld.rl.algorithms.td3.td3 import TD3
from jaxrlworld.rl.algorithms.tdmpc2.tdmpc2 import TDMPC2
from jaxrlworld.rl.configs.common_config_classes import Activation, DefaultInit, MLPActorCfg, MLPCriticCfg
from jaxrlworld.rl.modules.policies.fast_td3_ac import FastTD3ActorCritic
from jaxrlworld.rl.modules.policies.sac_ac import SACActorCritic
from jaxrlworld.rl.modules.policies.td3_ac import TD3ActorCritic
from jaxrlworld.rl.modules.policies.tdmpc2_world_model import TDMPC2WorldModel
from jaxrlworld.rl.storages.replay_buffer import ReplayBatch
from jaxrlworld.rl.storages.sequence_replay_buffer import SequenceBatch

OBS_A, OBS_C, ACT, BATCH, HORIZON = 6, 8, 3, 16, 3
UPDATES_BEFORE_SAVE, UPDATES_AFTER = 4, 4
DATA_KEY = jax.random.PRNGKey(1234)
ALGORITHMS = ("sac", "td3", "fast_td3", "tdmpc2")


def nets() -> dict:
    return dict(
        actor_cfg=MLPActorCfg(hidden_dims=[32, 32], activation=Activation.RELU, init=DefaultInit()),
        critic_cfg=MLPCriticCfg(hidden_dims=[32, 32], activation=Activation.RELU, init=DefaultInit()),
    )


def build(name: str, seed: int):
    key = jax.random.PRNGKey(seed)
    if name == "sac":
        model = SACActorCritic(num_actor_obs=OBS_A, num_critic_obs=OBS_C, num_actions=ACT, key=key, **nets())
        return SAC(actor_critic=model, batch_size=BATCH, key=jax.random.PRNGKey(seed + 1))
    if name == "td3":
        model = TD3ActorCritic(num_actor_obs=OBS_A, num_critic_obs=OBS_C, num_actions=ACT, key=key, **nets())
        return TD3(actor_critic=model, batch_size=BATCH, key=jax.random.PRNGKey(seed + 1))
    if name == "fast_td3":
        model = FastTD3ActorCritic(num_actor_obs=OBS_A, num_critic_obs=OBS_C, num_actions=ACT, key=key, **nets())
        return FastTD3(actor_critic=model, num_envs=BATCH, batch_size=BATCH, key=jax.random.PRNGKey(seed + 1))
    model = TDMPC2WorldModel(obs_dim=OBS_A, action_dim=ACT, latent_dim=16, mlp_dim=16, num_q=2, num_bins=5, key=key)
    return TDMPC2(model, num_envs=BATCH, batch_size=BATCH, num_bins=5, key=jax.random.PRNGKey(seed + 1))


def replay_batch(index: int) -> ReplayBatch:
    k = jax.random.split(jax.random.fold_in(DATA_KEY, index), 6)
    return ReplayBatch(
        actor_observations=jax.random.normal(k[0], (BATCH, OBS_A)),
        critic_observations=jax.random.normal(k[1], (BATCH, OBS_C)),
        actions=jnp.tanh(jax.random.normal(k[2], (BATCH, ACT))),
        rewards=jax.random.normal(k[3], (BATCH, 1)),
        next_actor_observations=jax.random.normal(k[4], (BATCH, OBS_A)),
        next_critic_observations=jax.random.normal(k[5], (BATCH, OBS_C)),
        terminated=jnp.zeros((BATCH, 1)),
        truncated=jnp.zeros((BATCH, 1)),
        gamma_power=jnp.ones((BATCH, 1)),
    )


def sequence_batch(index: int) -> SequenceBatch:
    k = jax.random.split(jax.random.fold_in(DATA_KEY, 10_000 + index), 3)
    return SequenceBatch(
        observations=jax.random.normal(k[0], (HORIZON + 1, BATCH, OBS_A)),
        actions=jnp.tanh(jax.random.normal(k[1], (HORIZON, BATCH, ACT))),
        rewards=jax.random.normal(k[2], (HORIZON, BATCH, 1)),
        terminated=jnp.zeros((HORIZON, BATCH, 1)),
    )


def run_updates(name: str, alg, first: int, count: int) -> None:
    for i in range(first, first + count):
        alg.update(sequence_batch(i) if name == "tdmpc2" else replay_batch(i), build_metrics=False)


def opt_states(name: str, alg) -> tuple:
    s = alg.train_state
    if name == "tdmpc2":
        return (s.wm_opt_state, s.pi_opt_state)
    if name == "sac":
        return (s.actor_opt_state, s.critic_opt_state, s.alpha_opt_state)
    return (s.actor_opt_state, s.critic_opt_state)


def leaves(tree) -> list[np.ndarray]:
    return [np.asarray(x) for x in jax.tree_util.tree_leaves(tree)]


def trees_equal(x, y) -> tuple[bool, str]:
    """Same structure, and every leaf equal in shape, dtype and value."""
    tx, ty = jax.tree_util.tree_structure(x), jax.tree_util.tree_structure(y)
    if tx != ty:
        return False, "tree structures differ"
    for i, (p, q) in enumerate(zip(leaves(x), leaves(y))):
        if p.shape != q.shape or p.dtype != q.dtype:
            return False, f"leaf {i}: {p.shape}/{p.dtype} vs {q.shape}/{q.dtype}"
        if not np.array_equal(p, q):
            return False, f"leaf {i}: max |d| = {np.max(np.abs(p.astype(np.float64) - q.astype(np.float64)))}"
    return True, f"{len(leaves(x))} leaves"


def main() -> int:
    failures: list[str] = []

    def chk(name: str, ok: bool, detail: str = "") -> None:
        print(f"  [{'OK' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
        if not ok:
            failures.append(name)

    for name in ALGORITHMS:
        print("=" * 78)
        print(f"{name.upper()}: optimizer state across a checkpoint")
        print("=" * 78)
        reference = build(name, 0)
        run_updates(name, reference, 0, UPDATES_BEFORE_SAVE)
        with tempfile.TemporaryDirectory() as d:
            meta = reference.save_train_state(d)
            chk("metadata flags optimizer state", meta.get("optimizer_state") is True)
            chk(f"{OPTIMIZER_STATE_FILE} written", os.path.isfile(os.path.join(d, OPTIMIZER_STATE_FILE)))

            resumed = build(name, 99)
            ok, _ = trees_equal(opt_states(name, reference), opt_states(name, resumed))
            chk("a fresh algorithm's optimizer state differs from the saved one", not ok)
            resumed.load_train_state(d, meta)
            ok, detail = trees_equal(opt_states(name, reference), opt_states(name, resumed))
            chk("restored optimizer state equals the saved one", ok, detail)
            ok, detail = trees_equal(reference.train_state.model, resumed.train_state.model)
            chk("restored model equals the saved one", ok, detail)

            run_updates(name, reference, UPDATES_BEFORE_SAVE, UPDATES_AFTER)
            run_updates(name, resumed, UPDATES_BEFORE_SAVE, UPDATES_AFTER)
            ok, detail = trees_equal(reference.train_state.model, resumed.train_state.model)
            chk("parameters bit-identical after further updates", ok, detail)
            ok, detail = trees_equal(opt_states(name, reference), opt_states(name, resumed))
            chk("optimizer states bit-identical after further updates", ok, detail)
            chk(
                "keys bit-identical after further updates",
                bool(np.array_equal(np.asarray(reference.train_state.key), np.asarray(resumed.train_state.key))),
            )

            reinit = build(name, 5)
            reinit.load_train_state(d, {**meta, "optimizer_state": False})
            run_updates(name, reinit, UPDATES_BEFORE_SAVE, UPDATES_AFTER)
            ok, detail = trees_equal(reference.train_state.model, reinit.train_state.model)
            chk("a re-initialized optimizer diverges from the uninterrupted run", not ok, detail)

            legacy_meta = {k: v for k, v in meta.items() if k != "optimizer_state"}
            os.remove(os.path.join(d, OPTIMIZER_STATE_FILE))
            legacy = build(name, 6)
            fresh_states = opt_states(name, legacy)
            legacy.load_train_state(d, legacy_meta)
            # The template is a fresh init over the LOADED parameters, so it
            # has the fresh structure and zero moments.
            chk(
                "checkpoint without the flag loads with a fresh optimizer state",
                jax.tree_util.tree_structure(opt_states(name, legacy)) == jax.tree_util.tree_structure(fresh_states),
            )
            try:
                build(name, 7).load_train_state(d, meta)
                chk("flag without file is refused", False, "loaded")
            except FileNotFoundError as e:
                chk("flag without file is refused", True, str(e)[:60])
        print()

    print("=== RESULT: ALL OK ===" if not failures else f"=== RESULT: {len(failures)} FAILED: {failures} ===")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
