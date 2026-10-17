"""Gate: every n-step window a replay buffer hands out lies on written rows.

The replay rings start zero-filled. A start position drawn too close to the
write pointer gives a window that runs off the written rows into those
zeros: the reward sum stops early, the bootstrap observation is a zero
vector, no boundary flag ends the window, and the discount is gamma to the
full n. Nothing about it is loud — the shapes fit and the loss falls.

``DeviceReplayBuffer.sample_batch`` refused to sample before every env held
``n_steps`` rows; ``batch_sampler``, the traced sampler SAC's scanned
update uses, did not, so a device-buffer run with ``learning_starts``
below ``n_steps`` rows per env trained its first updates on such windows.
No simulator. Every row written here carries its env index, its absolute
step and a marker, so a sampled window can be decoded and checked against
the rows it claims to summarize:

1. before ``n_steps`` rows per env exist, all three samplers (host
   ``sample_batch``, device ``sample_batch``, device ``batch_sampler``)
   refuse — and the unguarded index math is shown to produce windows over
   unwritten rows, so the refusal guards something real;
2. from ``n_steps`` rows up to a full ring, and after the ring wraps, every
   window from every sampler covers written rows only, never crosses the
   write pointer, and its reward, bootstrap observation, flags and discount
   equal a NumPy reference computed from the stored rows;
3. the off-policy runner's gate holds the update back until every env has
   ``n_steps`` rows even when ``learning_starts`` is already met.

Run::

    python -m jaxrlworld.scripts.diag.gates.check_replay_nstep_window_validity
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from jaxrlworld.rl.storages.device_replay_buffer import DeviceReplayBuffer, _gather_batch, _sample_indices
from jaxrlworld.rl.storages.replay_buffer import ReplayBatch, ReplayBuffer

NUM_ENVS, SIZE, OBS, ACT = 3, 16, 3, 2
GAMMA = 0.9
BATCH = 256
REWARD_TOL = 1e-5  # NumPy's and XLA's reduction order may differ by ulps; see check_replay_buffer_parity.


def row_fields(step: int) -> list[np.ndarray]:
    """The transition every env stores at absolute ``step``: self-describing rows."""
    env = np.arange(NUM_ENVS, dtype=np.float32)
    obs = np.stack([env, np.full(NUM_ENVS, step, np.float32), np.ones(NUM_ENVS, np.float32)], axis=1)
    next_obs = np.stack([env, np.full(NUM_ENVS, step + 1, np.float32), np.ones(NUM_ENVS, np.float32)], axis=1)
    act = np.stack([np.full(NUM_ENVS, step, np.float32), env], axis=1)
    rew = (1.0 + step / 100.0 + env / 1000.0).astype(np.float32)
    terminated = ((step * 7 + np.arange(NUM_ENVS)) % 5 == 0).astype(np.float32)
    truncated = ((step * 3 + np.arange(NUM_ENVS)) % 11 == 0).astype(np.float32)
    return [obs, obs.copy(), act, rew, next_obs, next_obs.copy(), terminated, truncated]


class Log:
    """Every row written so far, by (env, absolute step), for the reference."""

    def __init__(self) -> None:
        self.rows: dict[tuple[int, int], list[np.ndarray]] = {}
        self.steps = 0

    def write(self, fields: list[np.ndarray]) -> None:
        for e in range(NUM_ENVS):
            self.rows[(e, self.steps)] = [f[e] for f in fields]
        self.steps += 1

    def reference(self, env: int, start: int, n: int) -> tuple[np.ndarray, np.ndarray, float, float, float]:
        """n-step target from the log: reward sum to the first boundary, bootstrap row, flags, gamma^n_eff."""
        reward, used = 0.0, 0
        for k in range(n):
            if (env, start + k) not in self.rows:
                raise KeyError(f"env {env} step {start + k} was never written")
            _, _, _, rew, _, _, term, trunc = self.rows[(env, start + k)]
            reward += (GAMMA**k) * float(rew)
            used = k + 1
            if term + trunc > 0:
                break
        _, _, _, _, next_obs, _, term, trunc = self.rows[(env, start + used - 1)]
        return np.float32(reward), next_obs, float(term), float(trunc), GAMMA**used


def decode(batch: ReplayBatch) -> tuple[np.ndarray, np.ndarray]:
    """(env, absolute start step) of every sampled window, read off its first observation."""
    obs = np.asarray(batch.actor_observations)
    return obs[:, 0].astype(int), obs[:, 1].astype(int)


def check_batch(batch: ReplayBatch, log: Log, n: int, label: str, chk) -> None:
    envs, starts = decode(batch)
    obs = np.asarray(batch.actor_observations)
    chk(
        f"{label}: sampled rows are written rows",
        bool(np.all(obs[:, 2] == 1.0)),
        f"{int((obs[:, 2] != 1.0).sum())} unwritten",
    )
    newest = log.steps - 1
    oldest = log.steps - min(log.steps, SIZE)
    in_ring = np.all((starts >= oldest) & (starts + n - 1 <= newest))
    chk(
        f"{label}: windows stay inside the written ring",
        bool(in_ring),
        f"starts {starts.min()}..{starts.max()}, ring {oldest}..{newest}",
    )
    if not in_ring:
        return
    worst = {"rewards": 0.0, "next_obs": 0.0, "terminated": 0.0, "truncated": 0.0, "gamma_power": 0.0}
    rewards = np.asarray(batch.rewards)[:, 0]
    next_obs = np.asarray(batch.next_actor_observations)
    next_critic = np.asarray(batch.next_critic_observations)
    terminated = np.asarray(batch.terminated)[:, 0]
    truncated = np.asarray(batch.truncated)[:, 0]
    gamma_power = np.asarray(batch.gamma_power)[:, 0]
    for i, (e, s) in enumerate(zip(envs, starts)):
        r, nxt, term, trunc, gp = log.reference(int(e), int(s), n)
        worst["rewards"] = max(worst["rewards"], abs(float(rewards[i]) - float(r)))
        worst["next_obs"] = max(
            worst["next_obs"], float(np.abs(next_obs[i] - nxt).max()), float(np.abs(next_critic[i] - nxt).max())
        )
        worst["terminated"] = max(worst["terminated"], abs(float(terminated[i]) - term))
        worst["truncated"] = max(worst["truncated"], abs(float(truncated[i]) - trunc))
        worst["gamma_power"] = max(worst["gamma_power"], abs(float(gamma_power[i]) - np.float32(gp)))
    chk(f"{label}: reward sum matches the reference", worst["rewards"] <= REWARD_TOL, f"max |d| {worst['rewards']:.3g}")
    chk(
        f"{label}: bootstrap obs / flags / discount match exactly",
        worst["next_obs"] == 0.0
        and worst["terminated"] == 0.0
        and worst["truncated"] == 0.0
        and worst["gamma_power"] <= 1e-6,
        f"next_obs {worst['next_obs']:.3g} term {worst['terminated']:.3g} trunc {worst['truncated']:.3g} gamma^n {worst['gamma_power']:.3g}",
    )


def traced_batches(dev: DeviceReplayBuffer, num: int) -> list[ReplayBatch]:
    """Batches the way SAC's scanned update draws them: the traced sampler inside lax.scan."""
    sampler, state = dev.batch_sampler(BATCH)

    def body(key, _):
        key, k = jax.random.split(key)
        return key, sampler(state, k)

    _, stacked = jax.lax.scan(body, jax.random.PRNGKey(42), None, length=num)
    return [ReplayBatch(*(np.asarray(f[i]) for f in stacked)) for i in range(num)]


def make_buffers(n: int) -> tuple[ReplayBuffer, DeviceReplayBuffer]:
    host = ReplayBuffer(NUM_ENVS, OBS, OBS, ACT, SIZE, n, GAMMA, seed=0)
    dev = DeviceReplayBuffer(NUM_ENVS, OBS, OBS, ACT, SIZE, n, GAMMA, seed=0)
    return host, dev


def store(host: ReplayBuffer, dev: DeviceReplayBuffer, log: Log) -> None:
    fields = row_fields(log.steps)
    host.store_parallel(*[jnp.asarray(f) for f in fields])
    dev.store_parallel(*[jnp.asarray(f) for f in fields])
    log.write(fields)


def expect_raise(fn, arg, name: str, chk) -> None:
    try:
        fn(arg)
        chk(name, False, "sampled")
    except ValueError as e:
        chk(name, True, str(e)[:60])


def main() -> int:
    failures: list[str] = []

    def chk(name: str, ok: bool, detail: str = "") -> None:
        print(f"  [{'OK' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
        if not ok:
            failures.append(name)

    for n in (3, 5):
        print(f"\n=== n_steps={n}: 1. fewer than n rows per env ===")
        host, dev = make_buffers(n)
        log = Log()
        for _ in range(n - 1):
            store(host, dev, log)
        expect_raise(host.sample_batch, BATCH, "host sample_batch refuses", chk)
        expect_raise(dev.sample_batch, BATCH, "device sample_batch refuses", chk)
        expect_raise(dev.batch_sampler, BATCH, "device batch_sampler refuses", chk)
        # What the refusal guards: the unguarded index math at this fill.
        idx = _sample_indices(
            jax.random.PRNGKey(0), (jnp.asarray(dev.ptr), jnp.asarray(dev.filled_size)), NUM_ENVS, SIZE, n, BATCH, False
        )
        raw = _gather_batch(dev.buffers, idx, SIZE, n, GAMMA)
        # Every window drawn here ends past the written rows; how many
        # actually bootstrap off a zero row depends on where the planted
        # boundaries cut the window short.
        past_written = int((np.asarray(idx[1]) + n - 1 >= dev.filled_size).sum())
        zero_bootstrap = int((np.asarray(raw.next_actor_observations)[:, 2] != 1.0).sum())
        chk(
            "unguarded windows at this fill run past the written rows",
            past_written == BATCH,
            f"{past_written}/{BATCH} windows; {zero_bootstrap} bootstrap off a zero row",
        )

        print(f"\n=== n_steps={n}: 2. every fill from n rows to a wrapped ring ===")
        store(host, dev, log)  # now exactly n rows per env
        checked_fills = []
        while log.steps <= int(SIZE * 1.5):
            fill = min(log.steps, SIZE)
            at = {"host": host.sample_batch(BATCH), "device": dev.sample_batch(BATCH)}
            traced = traced_batches(dev, 3)
            for label, b in at.items():
                check_batch(b, log, n, f"fill {fill:2d} {label}", chk)
            for i, b in enumerate(traced):
                check_batch(b, log, n, f"fill {fill:2d} traced[{i}]", chk)
            checked_fills.append(fill)
            store(host, dev, log)
        chk(
            "fills covered n..full and wrapped",
            checked_fills[0] == n and checked_fills[-1] == SIZE and SIZE in checked_fills[:-1],
            str(checked_fills),
        )

    print("\n=== 3. runner gate waits for n_steps rows per env ===")
    # The real iteration method on a stand-in runner: 2048 transitions in
    # the buffer satisfy learning_starts=0 after one step of 2048 envs, as
    # the go2 SAC preset does, but one row per env is not an n-step window.
    from jaxrlworld.rl.runners.off_policy_runner import OffPolicyRunner

    def iteration(size: int, filled_size: int, learning_starts: int) -> bool:
        calls: list[int] = []
        runner = SimpleNamespace(
            alg=SimpleNamespace(
                replay_buffer=SimpleNamespace(size=size, filled_size=filled_size, n_steps=5),
                update_many=lambda *a, **k: calls.append(1),
            ),
            cfgs=SimpleNamespace(
                algorithm=SimpleNamespace(batch_size=8, learning_starts=learning_starts, get=lambda k, d: 1)
            ),
            runner_cfg=SimpleNamespace(log_interval=10),
            env=SimpleNamespace(num_envs=2048),
            num_steps_per_env=1,
            _collect_experience=lambda **k: {"collection_time": 0.0, "last_obs": None},
        )
        OffPolicyRunner._run_training_iteration(runner, obs=None, iteration=1, ep_infos=[])
        return bool(calls)

    chk("learning_starts met, 1 row per env, n_steps 5: no update", not iteration(2048, 1, 0))
    chk("learning_starts met, 4 rows per env: no update", not iteration(4 * 2048, 4, 0))
    chk("learning_starts met, 5 rows per env: update runs", iteration(5 * 2048, 5, 0))
    chk("5 rows per env but learning_starts not met: no update", not iteration(5 * 2048, 5, 100_000))

    print(f"\n=== RESULT: {'ALL OK' if not failures else f'{len(failures)} FAILED: {failures[:6]}'} ===")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
