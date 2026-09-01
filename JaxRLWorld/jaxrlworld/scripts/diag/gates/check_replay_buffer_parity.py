"""Do the host and device replay buffers agree, field for field?

``DeviceReplayBuffer`` exists so transitions born on the accelerator stay
there. It is a second implementation of semantics that already existed,
and the n-step half of those semantics is fiddly — a discounted sum
truncated at the first episode boundary in the window, with the
bootstrap taken from the last transition actually used and the discount
raised to however many steps that was.

Nothing about getting it subtly wrong is loud. The shapes still fit, the
losses still fall, and the policy is trained on slightly wrong targets.
So the two implementations are compared directly: identical transitions
in, the *same* indices handed to both, every field of the resulting
batch compared exactly.

Feeding both the same indices is the point — comparing two sampled
batches would only ever test the random number generators. The gate
reaches past ``sample_batch`` into the gather for that reason.

Episode boundaries are planted on purpose, at several densities, since
a window with no boundary in it exercises none of the interesting code.

Run:
    jaxpy -m jaxrlworld.scripts.diag.gates.check_replay_buffer_parity
    jaxpy -m jaxrlworld.scripts.diag.gates.check_replay_buffer_parity --n-steps 5 --num-envs 8
"""

import argparse

import jax
import jax.numpy as jnp
import numpy as np

from jaxrlworld.rl.storages.device_replay_buffer import DeviceReplayBuffer, _gather_batch
from jaxrlworld.rl.storages.replay_buffer import ReplayBatch, ReplayBuffer

# Every field but one is a gather: the batch either reads the row the
# indices point at or it does not, so any difference at all means the two
# implementations disagree about *which* transition to use, and the
# tolerance is zero. ``rewards`` is the exception — a discounted sum
# across the n-step window, where NumPy's reduction and XLA's are free to
# associate differently. Float32 has about seven digits, the terms are
# order 1, and there are at most n_steps of them, so a few ulps is the
# honest allowance; anything larger is a different sum, not a rounding.
FIELD_TOLERANCE = {"rewards": 1e-5}


def _host_batch(buf: ReplayBuffer, env_indices: np.ndarray, positions: np.ndarray) -> ReplayBatch:
    """What ``ReplayBuffer.sample_batch`` would return for these indices."""
    actor_obs = buf.actor_obs_buf[env_indices, positions]
    critic_obs = buf.critic_obs_buf[env_indices, positions]
    actions = buf.acts_buf[env_indices, positions]

    if buf.n_steps > 1:
        (rewards, next_actor, next_critic, terminated, truncated, gamma_power) = buf._compute_nstep_data(
            env_indices, positions
        )
    else:
        rewards = buf.rews_buf[env_indices, positions]
        next_actor = buf.next_actor_obs_buf[env_indices, positions]
        next_critic = buf.next_critic_obs_buf[env_indices, positions]
        terminated = buf.terminated_buf[env_indices, positions]
        truncated = buf.truncated_buf[env_indices, positions]
        gamma_power = np.full((env_indices.shape[0], 1), buf.gamma, dtype=np.float32)

    return ReplayBatch(
        actor_obs, critic_obs, actions, rewards, next_actor, next_critic, terminated, truncated, gamma_power
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num-envs", type=int, default=4)
    ap.add_argument("--obs-dim", type=int, default=6)
    ap.add_argument("--act-dim", type=int, default=3)
    ap.add_argument("--size-per-env", type=int, default=64)
    ap.add_argument("--n-steps", type=int, nargs="+", default=[1, 3, 5])
    ap.add_argument("--gamma", type=float, default=0.97)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--trials", type=int, default=20)
    args = ap.parse_args()

    print("=" * 78)
    print("  REPLAY BUFFER PARITY — HOST VS DEVICE")
    print(f"  jax backend: {jax.default_backend()}   envs {args.num_envs}   ring {args.size_per_env}")
    print(f"  n_steps {args.n_steps}   batch {args.batch_size}   index draws {args.trials}")
    print("=" * 78)

    failures = 0
    for n_steps in args.n_steps:
        for boundary_p in (0.0, 0.02, 0.2):
            host = ReplayBuffer(
                args.num_envs, args.obs_dim, args.obs_dim, args.act_dim, args.size_per_env, n_steps, args.gamma
            )
            dev = DeviceReplayBuffer(
                args.num_envs, args.obs_dim, args.obs_dim, args.act_dim, args.size_per_env, n_steps, args.gamma
            )

            # Wrap the ring once and a bit, so ptr sits mid-buffer and the
            # modular arithmetic in both paths is actually exercised.
            rng = np.random.default_rng(0)
            steps = int(args.size_per_env * 1.5)
            for t in range(steps):
                fields = [
                    rng.normal(size=(args.num_envs, args.obs_dim)).astype(np.float32),
                    rng.normal(size=(args.num_envs, args.obs_dim)).astype(np.float32),
                    rng.normal(size=(args.num_envs, args.act_dim)).astype(np.float32),
                    rng.normal(size=(args.num_envs,)).astype(np.float32),
                    rng.normal(size=(args.num_envs, args.obs_dim)).astype(np.float32),
                    rng.normal(size=(args.num_envs, args.obs_dim)).astype(np.float32),
                    (rng.random(args.num_envs) < boundary_p),
                    (rng.random(args.num_envs) < boundary_p),
                ]
                host.store_parallel(*[jnp.asarray(f) for f in fields])
                dev.store_parallel(*[jnp.asarray(f) for f in fields])

            worst = {}
            for trial in range(args.trials):
                env_idx = rng.integers(0, args.num_envs, size=args.batch_size)
                pos = rng.integers(0, args.size_per_env, size=args.batch_size)

                h = _host_batch(host, env_idx, pos)
                d = _gather_batch(
                    dev.buffers,
                    (jnp.asarray(env_idx), jnp.asarray(pos)),
                    args.size_per_env,
                    n_steps,
                    args.gamma,
                )
                for name, hv, dv in zip(h._fields, h, d):
                    diff = float(np.abs(np.asarray(hv, np.float64) - np.asarray(dv, np.float64)).max())
                    worst[name] = max(worst.get(name, 0.0), diff)

            bad = {k: v for k, v in worst.items() if v > FIELD_TOLERANCE.get(k, 0.0)}
            label = f"n_steps={n_steps} boundary_p={boundary_p}"
            if bad:
                failures += 1
                print(f"  {label:<34s} MISMATCH")
                for k, v in sorted(bad.items(), key=lambda kv: -kv[1]):
                    print(f"      {k:<26s} max |diff| {v:.6g}  (tolerance {FIELD_TOLERANCE.get(k, 0.0):g})")
            else:
                gathers = sum(1 for k, v in worst.items() if k not in FIELD_TOLERANCE and v == 0.0)
                summed = max((worst[k] for k in FIELD_TOLERANCE if k in worst), default=0.0)
                print(f"  {label:<34s} ok — {gathers} gathers bit-exact, sum within {summed:.2g}")

    print("=" * 78)
    if failures:
        print(f"  FAIL — {failures} configuration(s) disagree. The two buffers are not")
        print("  interchangeable, and the device one would train on different targets.")
        print("=" * 78)
        return 1
    print("  PASS — every gathered field bit-exact, the discounted sum within float32,")
    print("         across boundary densities and a wrapped ring.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
