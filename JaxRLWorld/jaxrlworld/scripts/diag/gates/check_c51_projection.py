"""Gate: the C51 target projection used by the FastTD3 critic loss.

``project_distribution_batched`` is held against a scalar NumPy
re-derivation of the categorical projection (Bellemare et al. 2017,
Algorithm 1) on random inputs and on the cases where the earlier
implementation went wrong:

1. identity: reward 0, discount 1, bootstrap 1 maps every distribution
   onto itself (every target lands exactly on its own atom);
2. targets landing exactly on interior atoms keep their mass there;
3. targets clipped to v_min / v_max pile onto the end atoms;
4. batch size 1 and inputs with a trailing unit axis are accepted;
5. random batches match the reference to float32 round-off and every row
   keeps its mass.

Run::

    python -m jaxrlworld.scripts.diag.gates.check_c51_projection
"""

from __future__ import annotations

import sys

import jax
import jax.numpy as jnp
import numpy as np

from jaxrlworld.rl.modules.policies.fast_td3_ac import project_distribution_batched


def reference_projection(
    next_probs: np.ndarray,
    rewards: np.ndarray,
    bootstrap: np.ndarray,
    discount: np.ndarray,
    num_atoms: int,
    v_min: float,
    v_max: float,
) -> np.ndarray:
    """Scalar loop over batch rows and source atoms, float64."""
    delta_z = (v_max - v_min) / (num_atoms - 1)
    support = np.linspace(v_min, v_max, num_atoms)
    out = np.zeros((next_probs.shape[0], num_atoms), dtype=np.float64)
    for i in range(next_probs.shape[0]):
        for j in range(num_atoms):
            tz = min(max(rewards[i] + bootstrap[i] * discount[i] * support[j], v_min), v_max)
            b = (tz - v_min) / delta_z
            lo = int(np.floor(b))
            hi = int(np.ceil(b))
            if lo == hi:
                out[i, lo] += next_probs[i, j]
            else:
                out[i, lo] += next_probs[i, j] * (hi - b)
                out[i, hi] += next_probs[i, j] * (b - lo)
    return out


def main() -> int:
    failures: list[str] = []

    def chk(name: str, ok: bool, detail: str = "") -> None:
        print(f"  [{'OK' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
        if not ok:
            failures.append(name)

    def run(probs, rewards, bootstrap, discount, num_atoms, v_min, v_max):
        return np.asarray(
            project_distribution_batched(
                jnp.asarray(probs, jnp.float32),
                jnp.asarray(rewards, jnp.float32),
                jnp.asarray(bootstrap, jnp.float32),
                jnp.asarray(discount, jnp.float32),
                num_atoms,
                v_min,
                v_max,
            )
        )

    print("=== 1. identity projection ===")
    probs = np.array([[0.0, 1.0, 0.0], [0.2, 0.5, 0.3]])
    got = run(probs, [0.0, 0.0], [1.0, 1.0], [1.0, 1.0], 3, -1.0, 1.0)
    chk("[0,1,0] stays [0,1,0]", np.allclose(got[0], [0, 1, 0], atol=1e-7), f"got {got[0]}")
    chk("[0.2,0.5,0.3] stays put", np.allclose(got[1], probs[1], atol=1e-7), f"got {got[1]}")

    print("\n=== 2. targets on interior atoms ===")
    # support [-2,-1,0,1,2]; reward +1 with discount 1 shifts every atom up by exactly one.
    probs = np.array([[0.1, 0.2, 0.3, 0.4, 0.0]])
    got = run(probs, [1.0], [1.0], [1.0], 5, -2.0, 2.0)
    chk(
        "shift by one atom, top atom absorbs the clip",
        np.allclose(got[0], [0, 0.1, 0.2, 0.3, 0.4], atol=1e-7),
        f"got {got[0]}",
    )

    print("\n=== 3. clipping to the ends ===")
    probs = np.array([[0.25, 0.25, 0.25, 0.25]])
    got = run(probs, [100.0], [1.0], [1.0], 4, -1.0, 1.0)
    chk("huge reward -> all mass on v_max", np.allclose(got[0], [0, 0, 0, 1], atol=1e-7), f"got {got[0]}")
    got = run(probs, [-100.0], [1.0], [1.0], 4, -1.0, 1.0)
    chk("huge negative reward -> all mass on v_min", np.allclose(got[0], [1, 0, 0, 0], atol=1e-7), f"got {got[0]}")
    got = run(probs, [0.3], [0.0], [1.0], 4, -1.0, 1.0)
    ref = reference_projection(probs, np.array([0.3]), np.array([0.0]), np.array([1.0]), 4, -1.0, 1.0)
    chk("terminal (bootstrap 0): delta at the reward", np.allclose(got, ref, atol=1e-6), f"got {got[0]} ref {ref[0]}")

    print("\n=== 4. batch size 1 and trailing unit axis ===")
    got1 = run(probs, np.array([[0.3]]), np.array([[1.0]]), np.array([[0.9]]), 4, -1.0, 1.0)
    chk("[1,1]-shaped scalars accepted", got1.shape == (1, 4), f"shape {got1.shape}")

    print("\n=== 5. random batches vs reference ===")
    rng = np.random.default_rng(0)
    worst = 0.0
    worst_mass = 0.0
    for trial in range(20):
        batch = int(rng.integers(1, 64))
        num_atoms = int(rng.integers(2, 128))
        v_min = float(rng.uniform(-50, 0))
        v_max = float(rng.uniform(1, 50))
        logits = rng.normal(size=(batch, num_atoms))
        p = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        r = rng.uniform(-60, 60, size=batch)
        boot = (rng.uniform(size=batch) > 0.2).astype(np.float64)
        disc = rng.uniform(0.5, 1.0, size=batch) ** rng.integers(1, 4, size=batch)
        got = run(p.astype(np.float32), r, boot, disc, num_atoms, v_min, v_max)
        ref = reference_projection(p.astype(np.float32).astype(np.float64), r, boot, disc, num_atoms, v_min, v_max)
        worst = max(worst, float(np.abs(got - ref).max()))
        worst_mass = max(worst_mass, float(np.abs(got.sum(axis=1) - 1.0).max()))
    chk("max |impl - reference| over 20 random batches", worst < 1e-4, f"{worst:.2e}")
    chk("every row keeps unit mass", worst_mass < 1e-5, f"max |sum - 1| {worst_mass:.2e}")

    print("\n=== 6. jit + gradient-free target ===")
    jitted = jax.jit(project_distribution_batched, static_argnums=(4, 5, 6))
    p = jnp.asarray(probs, jnp.float32)
    out = jitted(p, jnp.array([0.3]), jnp.array([1.0]), jnp.array([0.9]), 4, -1.0, 1.0)
    chk("jit output matches eager", np.allclose(np.asarray(out), run(probs, [0.3], [1.0], [0.9], 4, -1.0, 1.0)))

    print(f"\n=== RESULT: {'ALL OK' if not failures else f'{len(failures)} FAILED: {failures}'} ===")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
