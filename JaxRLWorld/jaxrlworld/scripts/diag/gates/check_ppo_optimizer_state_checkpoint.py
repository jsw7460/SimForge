"""Gate: a resumed PPO run is the run it continues, bit for bit.

A PPO checkpoint used to hold the model and not the optimizer state, so a
resume re-initialized Adam: zero moments, step count 1, the config's
learning rates in place of the ones the run had reached. No simulator; the
experience stream is synthetic and a deterministic function of the
iteration and step, so two PPOs fed the same stream must agree exactly.

1. a PPO trained for a few iterations is saved; its optimizer state comes
   back leaf for leaf (shapes, dtypes, values) in a freshly built PPO;
2. the resumed PPO and the uninterrupted one, fed the same further
   experience, end with bit-identical parameters, optimizer state and key;
3. the old behaviour (model restored, optimizer re-initialized) does NOT
   reproduce the uninterrupted run — so 2 is a real test, not a tautology;
4. a checkpoint written before the optimizer state was saved (no flag, no
   file) loads with zero moments and says so; a checkpoint whose flag
   promises the file but lacks it is refused;
5. the AMP discriminator's optimizer state round-trips the same way.

Run::

    python -m jaxrlworld.scripts.diag.gates.check_ppo_optimizer_state_checkpoint
"""

from __future__ import annotations

import os
import sys
import tempfile

import jax
import jax.numpy as jnp
import numpy as np

from jaxrlworld.rl.algorithms.amp_ppo.amp import AdversarialMotionPrior
from jaxrlworld.rl.algorithms.amp_ppo.expert_motion import ExpertMotionSet
from jaxrlworld.rl.algorithms.ppo import PPO
from jaxrlworld.rl.configs.algorithms.amp_ppo import AmpConfig
from jaxrlworld.rl.configs.common_config_classes import PPOPolicyConfig
from jaxrlworld.rl.modules.policies.ppo_ac import PPOActorCritic

OBS_A, OBS_C, ACT = 6, 8, 3
NUM_ENVS, NUM_STEPS = 8, 4
ITERS_BEFORE_SAVE, ITERS_AFTER = 3, 3
DATA_KEY = jax.random.PRNGKey(1234)

LAYOUT = (
    ("actor", 0, OBS_A),
    ("critic", OBS_A, OBS_C),
    ("reward", OBS_A + OBS_C, 1),
    ("terminated", OBS_A + OBS_C + 1, 1),
    ("truncated", OBS_A + OBS_C + 2, 1),
)


def make_ppo(key) -> PPO:
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
    # The adaptive schedule writes the actor rate into the optimizer state,
    # so a restored state has to carry the reached rate, not the config's.
    alg = PPO(actor_critic=model, num_learning_epochs=2, num_mini_batches=2, schedule="adaptive", key=key)
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


def step_data(iteration: int, step: int) -> jax.Array:
    """The packed env step for (iteration, step): the same on every run."""
    k1, k2, k3 = jax.random.split(jax.random.fold_in(DATA_KEY, iteration * NUM_STEPS + step), 3)
    actor = jax.random.normal(k1, (NUM_ENVS, OBS_A))
    critic = jax.random.normal(k2, (NUM_ENVS, OBS_C))
    reward = jax.random.normal(k3, (NUM_ENVS, 1))
    flags = jnp.zeros((NUM_ENVS, 2))
    return jnp.concatenate([actor, critic, reward, flags], axis=1)


def run_iterations(alg: PPO, obs: tuple[jax.Array, jax.Array], first: int, count: int) -> tuple[jax.Array, jax.Array]:
    a, c = obs
    for it in range(first, first + count):
        for step in range(NUM_STEPS):
            alg.act(PPO.ActInput(a, c))
            a, c = alg.process_env_step_packed(step_data(it, step), LAYOUT)
        alg.compute_returns(c)
        alg.update()
    return a, c


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


def adam_counts(alg: PPO) -> list[int]:
    """Step count of every parameter group's Adam state."""
    return [int(s.inner_state.inner_state[0].count) for s in alg.train_state.opt_state[1].inner_states.values()]


def make_amp(key) -> AdversarialMotionPrior:
    feature_dim, frames = 5, 40
    features = np.asarray(jax.random.normal(jax.random.PRNGKey(7), (frames, feature_dim)), dtype=np.float32)
    expert = ExpertMotionSet(
        features=features,
        clip_start=np.array([0, frames], dtype=np.int64),
        clip_names=("synthetic",),
        fps=50.0,
        source_index=np.array([0]),
    )
    cfg = AmpConfig(num_amp_obs_steps=2, discriminator_hidden_dims=(16,), replay_buffer_size=64, replay_insert_size=16)
    return AdversarialMotionPrior(cfg, expert, feature_dim, control_dt=0.02, learning_rate=1e-4, key=key)


def drive_amp(amp: AdversarialMotionPrior, key) -> None:
    for i in range(4):
        obs = jax.random.normal(jax.random.fold_in(key, i), (NUM_ENVS, amp.feature_dim))
        amp.shape_rewards(obs, jnp.zeros(NUM_ENVS, dtype=bool), jnp.zeros(NUM_ENVS))
    amp.update_discriminator(num_updates=2, minibatch_size=8)


def main() -> int:
    failures: list[str] = []

    def chk(name: str, ok: bool, detail: str = "") -> None:
        print(f"  [{'OK' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
        if not ok:
            failures.append(name)

    obs0 = (jnp.ones((NUM_ENVS, OBS_A)), jnp.ones((NUM_ENVS, OBS_C)))

    print("=== 1. optimizer state round trip ===")
    reference = make_ppo(jax.random.PRNGKey(0))
    obs_at_save = run_iterations(reference, obs0, 0, ITERS_BEFORE_SAVE)
    counts = adam_counts(reference)
    chk("moments advanced before save", all(c == ITERS_BEFORE_SAVE * 4 for c in counts), f"adam counts {counts}")
    with tempfile.TemporaryDirectory() as d:
        meta = reference.save_train_state(d)
        chk("metadata flags optimizer state", meta.get("optimizer_state") is True)
        chk("optimizer_state.eqx written", os.path.isfile(os.path.join(d, "optimizer_state.eqx")))

        resumed = make_ppo(jax.random.PRNGKey(99))
        chk("fresh PPO starts from zero moments", all(c == 0 for c in adam_counts(resumed)))
        resumed.load_train_state(d, meta)
        ok, detail = trees_equal(reference.train_state.opt_state, resumed.train_state.opt_state)
        chk("restored optimizer state equals the saved one", ok, detail)
        ok, detail = trees_equal(reference.train_state.model, resumed.train_state.model)
        chk("restored model equals the saved one", ok, detail)
        chk(
            "restored key equals the saved one",
            bool(np.array_equal(np.asarray(reference.train_state.key), np.asarray(resumed.train_state.key))),
        )
        chk(
            "restored actor rate is the reached one",
            resumed.actor_lr == reference.actor_lr,
            f"{resumed.actor_lr} vs {reference.actor_lr}",
        )

        print("\n=== 2. resumed run == uninterrupted run ===")
        run_iterations(reference, obs_at_save, ITERS_BEFORE_SAVE, ITERS_AFTER)
        run_iterations(resumed, obs_at_save, ITERS_BEFORE_SAVE, ITERS_AFTER)
        ok, detail = trees_equal(reference.train_state.model, resumed.train_state.model)
        chk("parameters bit-identical after further training", ok, detail)
        ok, detail = trees_equal(reference.train_state.opt_state, resumed.train_state.opt_state)
        chk("optimizer state bit-identical after further training", ok, detail)
        chk(
            "keys bit-identical after further training",
            bool(np.array_equal(np.asarray(reference.train_state.key), np.asarray(resumed.train_state.key))),
        )

        print("\n=== 3. the old behaviour is told apart ===")
        reinit = make_ppo(jax.random.PRNGKey(5))
        reinit.load_train_state(d, {**meta, "optimizer_state": False})
        chk("re-initialized optimizer starts from zero moments", all(c == 0 for c in adam_counts(reinit)))
        run_iterations(reinit, obs_at_save, ITERS_BEFORE_SAVE, ITERS_AFTER)
        ok, detail = trees_equal(reference.train_state.model, reinit.train_state.model)
        chk("re-initialized optimizer diverges from the uninterrupted run", not ok, detail)

        print("\n=== 4. legacy and corrupt checkpoints ===")
        legacy_meta = {k: v for k, v in meta.items() if k != "optimizer_state"}
        os.remove(os.path.join(d, "optimizer_state.eqx"))
        legacy = make_ppo(jax.random.PRNGKey(6))
        legacy.load_train_state(d, legacy_meta)
        chk("checkpoint without the flag loads with zero moments", all(c == 0 for c in adam_counts(legacy)))
        ok, detail = trees_equal(reference.train_state.model, legacy.train_state.model)
        chk("...and its model is still the saved one", not ok and "leaf" in detail, "(reference trained on)")
        try:
            make_ppo(jax.random.PRNGKey(7)).load_train_state(d, meta)
            chk("flag without file is refused", False, "loaded")
        except FileNotFoundError as e:
            chk("flag without file is refused", True, str(e)[:60])

    print("\n=== 5. AMP discriminator optimizer state ===")
    amp = make_amp(jax.random.PRNGKey(0))
    drive_amp(amp, jax.random.PRNGKey(3))
    with tempfile.TemporaryDirectory() as d:
        amp.save(d)
        chk("amp_optimizer_state.eqx written", os.path.isfile(os.path.join(d, "amp_optimizer_state.eqx")))
        fresh = make_amp(jax.random.PRNGKey(11))
        fresh.load(d, optimizer_state=True)
        ok, detail = trees_equal(amp.opt_state, fresh.opt_state)
        chk("restored discriminator optimizer state equals the saved one", ok, detail)
        ok, detail = trees_equal(amp.disc, fresh.disc)
        chk("restored discriminator equals the saved one", ok, detail)
        legacy = make_amp(jax.random.PRNGKey(12))
        legacy.load(d, optimizer_state=False)
        ok, _ = trees_equal(amp.opt_state, legacy.opt_state)
        chk("legacy load keeps zero moments", not ok)
        os.remove(os.path.join(d, "amp_optimizer_state.eqx"))
        try:
            make_amp(jax.random.PRNGKey(13)).load(d, optimizer_state=True)
            chk("AMP flag without file is refused", False, "loaded")
        except FileNotFoundError as e:
            chk("AMP flag without file is refused", True, str(e)[:60])

    print(f"\n=== RESULT: {'ALL OK' if not failures else f'{len(failures)} FAILED: {failures}'} ===")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
