"""Gate: the TD-MPC2 two-hot encoding puts its mass where the target is.

``two_hot`` measured the fractional offset from the unclipped lower bin
and then clipped that bin to ``num_bins - 2``. For a target at or above
``vmax`` (any TD target the symlog clip saturates) the lower bin became
``num_bins - 2`` with offset 0, so the whole mass sat one bin below the
top: the critic was trained toward a value one bin size lower than its
target. The offset is now measured from the clipped bin.

1. every grid point ``vmin + k * bin_size`` (``k = 0 .. num_bins - 1``,
   the last included) encodes to the one-hot of bin ``k``;
2. targets beyond ``vmax`` encode to the last bin, beyond ``vmin`` to
   the first;
3. for random targets the encoding is a distribution (rows sum to 1,
   two adjacent non-zero entries at most) whose expectation over the
   bin centres is the clipped symlog target;
4. ``soft_ce`` of logits that put all mass on the right bins is near
   zero for grid targets, including the last bin.

Run::

    python -m jaxrlworld.scripts.diag.gates.check_tdmpc2_two_hot
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jaxrlworld.rl.algorithms.tdmpc2.math import make_two_hot_config, soft_ce, symexp, symlog, two_hot

NUM_BINS, VMIN, VMAX = 101, -10.0, 10.0


def main() -> int:
    failures: list[str] = []

    def chk(name: str, ok: bool, detail: str = "") -> None:
        print(f"  [{'OK' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
        if not ok:
            failures.append(name)

    print("=" * 78)
    print("TD-MPC2 TWO-HOT")
    print("=" * 78)
    cfg = make_two_hot_config(NUM_BINS, VMIN, VMAX)
    bins = jnp.linspace(VMIN, VMAX, NUM_BINS)

    # 1. grid points, in the raw (pre-symlog) domain the function takes.
    grid_targets = symexp(bins)[:, None]
    enc = two_hot(grid_targets, cfg)
    expected = jnp.eye(NUM_BINS)
    worst = float(jnp.abs(enc - expected).max())
    chk(
        "every grid point is the one-hot of its bin (last bin included)",
        worst < 1e-5,
        f"max |enc - onehot| = {worst:.2e}",
    )
    last = two_hot(symexp(jnp.array([VMAX]))[:, None], cfg)[0]
    chk("the top grid point lands on the last bin", float(last[-1]) > 1 - 1e-5, f"last bin mass {float(last[-1]):.6f}")

    # 2. saturation.
    above = two_hot(jnp.array([[1e6], [symexp(jnp.array(VMAX + 3.0))]]), cfg)
    below = two_hot(jnp.array([[-1e6]]), cfg)
    chk("targets beyond vmax encode to the last bin", bool(jnp.all(above[:, -1] > 1 - 1e-5)))
    chk("targets beyond vmin encode to the first bin", bool(below[0, 0] > 1 - 1e-5))

    # 3. random targets: a distribution whose mean is the clipped symlog target.
    x = jax.random.uniform(jax.random.PRNGKey(0), (4096, 1), minval=-5e4, maxval=5e4)
    enc = two_hot(x, cfg)
    row_sums = enc.sum(axis=-1)
    nonzero = (enc > 0).sum(axis=-1)
    mean = (enc * bins).sum(axis=-1)
    target = jnp.clip(symlog(x[:, 0]), VMIN, VMAX)
    chk("rows sum to one", float(jnp.abs(row_sums - 1).max()) < 1e-5)
    chk("at most two bins carry mass", bool(jnp.all(nonzero <= 2)))
    err = float(jnp.abs(mean - target).max())
    chk(
        "the expectation over bin centres is the clipped symlog target",
        err < 1e-4,
        f"max |E[bins] - target| = {err:.2e}",
    )

    # 4. soft_ce against confident logits on the right bins.
    logits = 50.0 * expected
    ce = soft_ce(logits, grid_targets, cfg)
    # Float32 rounding of the grid leaves ~1e-5 of mass on a neighbour,
    # which the 50-logit margin turns into ~5e-4 of cross-entropy.
    chk(
        "soft_ce is ~0 for confident correct logits on every grid point",
        float(ce.max()) < 1e-3,
        f"max ce = {float(ce.max()):.2e}",
    )

    print("=" * 78)
    print("=== RESULT: ALL OK ===" if not failures else f"=== RESULT: {len(failures)} FAILED ===")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
