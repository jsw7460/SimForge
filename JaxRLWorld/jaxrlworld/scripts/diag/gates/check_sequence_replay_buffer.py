"""Gate: ``SequenceReplayBuffer`` never hands out a sequence across a reset.

No simulator. Observations encode ``(env, step)`` so a sampled sequence can
be checked against the stored episode structure:

1. every sampled sequence lies inside one episode, with consecutive steps
   and ``observations[H]`` equal to the last transition's ``next_obs``;
2. a buffer holding exactly ``horizon`` transitions of one episode yields
   that sequence (the earlier bound demanded ``horizon + 1``);
3. once the buffer has wrapped, the newest sequence (the one ending at the
   write pointer) is sampled, and no sequence straddles the pointer;
4. when no within-episode sequence exists the sampler raises instead of
   returning boundary-crossing data.

Run::

    python -m jaxrlworld.scripts.diag.gates.check_sequence_replay_buffer
"""

from __future__ import annotations

import sys

import jax
import numpy as np

from jaxrlworld.rl.storages.sequence_replay_buffer import SequenceReplayBuffer

NUM_ENVS = 3
OBS_DIM = 2  # (env index, global step)
ACT_DIM = 1


def store_steps(buf: SequenceReplayBuffer, num_steps: int, done_at: dict[int, set[int]], t0: int = 0) -> None:
    """Store ``num_steps`` steps; ``done_at[env]`` are the global steps that end an episode."""
    for t in range(t0, t0 + num_steps):
        obs = np.stack([np.array([e, t], dtype=np.float32) for e in range(NUM_ENVS)])
        next_obs = obs + np.array([0.0, 1.0], dtype=np.float32)
        done = np.array([t in done_at.get(e, set()) for e in range(NUM_ENVS)])
        buf.store_parallel(
            obs, np.zeros((NUM_ENVS, ACT_DIM)), np.zeros(NUM_ENVS), next_obs, done, np.zeros(NUM_ENVS, bool)
        )


def check_batch(batch, horizon: int, done_at: dict[int, set[int]]) -> tuple[bool, str]:
    obs = np.asarray(batch.observations)  # [H+1, B, 2]
    envs = obs[0, :, 0].astype(int)
    steps = obs[:, :, 1]
    for b in range(obs.shape[1]):
        e = int(envs[b])
        seq = steps[:, b]
        if not np.all(obs[:, b, 0] == e):
            return False, f"sample {b}: env index changes along the sequence"
        if not np.all(np.diff(seq) == 1):
            return False, f"sample {b}: steps not consecutive: {seq}"
        # A done at step s ends the episode after transition s, so a reset
        # lies inside the sequence iff some done falls on steps[0 .. H-2].
        inner = set(int(s) for s in seq[: horizon - 1])
        if inner & done_at.get(e, set()):
            return False, f"sample {b}: env {e} steps {seq[:horizon]} contain a reset at {inner & done_at[e]}"
    return True, f"{obs.shape[1]} sequences checked"


def main() -> int:
    failures: list[str] = []

    def chk(name: str, ok: bool, detail: str = "") -> None:
        print(f"  [{'OK' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
        if not ok:
            failures.append(name)

    horizon = 3
    key = jax.random.PRNGKey(0)

    print("=== 1. sequences stay inside one episode ===")
    buf = SequenceReplayBuffer(NUM_ENVS, OBS_DIM, ACT_DIM, size_per_env=64, horizon=horizon)
    done_at = {0: {2, 7, 8, 15}, 1: {0, 5, 11}, 2: {3, 4, 5, 6, 12}}
    store_steps(buf, 20, done_at)
    batch = buf.sample_batch(512, key)
    ok, detail = check_batch(batch, horizon, done_at)
    chk("512 samples: same env, consecutive, no reset inside", ok, detail)
    last_next = np.asarray(batch.observations)[horizon, :, 1]
    prev = np.asarray(batch.observations)[horizon - 1, :, 1]
    chk("observations[H] is next_obs of the last transition", bool(np.all(last_next == prev + 1)))

    print("\n=== 2. exactly `horizon` transitions suffice ===")
    buf = SequenceReplayBuffer(NUM_ENVS, OBS_DIM, ACT_DIM, size_per_env=64, horizon=horizon)
    store_steps(buf, horizon, {})
    try:
        batch = buf.sample_batch(8, key)
        steps = np.asarray(batch.observations)[:, :, 1]
        chk(
            "sampled the only sequence",
            bool(np.all(steps[0] == 0) and np.all(steps[horizon] == horizon)),
            f"steps {steps[:, 0]}",
        )
    except ValueError as e:
        chk("sampled the only sequence", False, str(e))

    print("\n=== 3. wrapped buffer: newest sequence reachable, none straddles the pointer ===")
    size = 16
    buf = SequenceReplayBuffer(NUM_ENVS, OBS_DIM, ACT_DIM, size_per_env=size, horizon=horizon)
    total = size + 5  # wrap: rows 0..4 hold steps 16..20, rows 5..15 hold steps 5..15
    store_steps(buf, total, {})
    batch = buf.sample_batch(4096, jax.random.PRNGKey(1))
    steps = np.asarray(batch.observations)[:, :, 1]
    ok, detail = check_batch(batch, horizon, {})
    chk("4096 samples consecutive (no wrap through the overwritten rows)", ok, detail)
    newest_start = total - horizon
    chk(
        "newest sequence (ending at the write pointer) is sampled",
        bool(np.any(steps[0] == newest_start)),
        f"starts seen up to {int(steps[0].max())}, newest {newest_start}",
    )
    chk(
        "oldest surviving step is the first row after the pointer",
        int(steps[0].min()) == total - size,
        f"min start {int(steps[0].min())}",
    )

    print("\n=== 4. no within-episode sequence -> refuse, never pad ===")
    buf = SequenceReplayBuffer(NUM_ENVS, OBS_DIM, ACT_DIM, size_per_env=64, horizon=horizon)
    store_steps(buf, 12, {e: set(range(12)) for e in range(NUM_ENVS)})  # every step ends an episode
    try:
        buf.sample_batch(16, key)
        chk("all-terminal buffer raises", False, "returned a batch")
    except RuntimeError as e:
        chk("all-terminal buffer raises", True, str(e)[:80])
    buf = SequenceReplayBuffer(NUM_ENVS, OBS_DIM, ACT_DIM, size_per_env=64, horizon=horizon)
    store_steps(buf, 12, {e: set(range(1, 12, 2)) for e in range(NUM_ENVS)})  # episodes of length 2 < horizon
    try:
        buf.sample_batch(16, key)
        chk("episodes shorter than the horizon raise", False, "returned a batch")
    except RuntimeError as e:
        chk("episodes shorter than the horizon raise", True, str(e)[:80])

    print(f"\n=== RESULT: {'ALL OK' if not failures else f'{len(failures)} FAILED: {failures}'} ===")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
