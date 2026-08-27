"""JAX vs torch, same MLP forward+backward, width sweep.

The PPO update investigation ended at "the compiled scan's grad step is
~30% slower than rsl_rl's on the same shapes, and every framework-level
suspect (jit structure, precision, gather) measured clean". If that
residual really is small-GEMM kernel efficiency, the gap must close —
or invert — as the network grows into the compute-bound regime. This
bench tests exactly that hypothesis with no RL code in the loop:

  batch 98304 (the g1 minibatch), obs 100 -> [W, W, W] -> 29, ELU,
  mean-square loss, forward+backward, both frameworks fully synced.

torch runs twice: with its default full-f32 matmul and with TF32
enabled (rsl_rl inherits torch defaults). JAX runs at its default
(TF32 on this GPU, as the precision A/B showed).

Usage:
    jaxpy -m rlworld.scripts.diag.mlp_fwdbwd_jax_vs_torch
    jaxpy -m rlworld.scripts.diag.mlp_fwdbwd_jax_vs_torch --widths 512,2048,8192
"""

from __future__ import annotations

import argparse
import statistics
import time

import jax
import jax.numpy as jnp
import torch


def _flops(batch: int, dims: list[int]) -> float:
    fwd = sum(2 * a * b for a, b in zip(dims[:-1], dims[1:]))
    return 3.0 * batch * fwd  # fwd + ~2x for backward


def bench_jax(batch: int, dims: list[int], reps: int) -> float:
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, len(dims))
    ws = [jax.random.normal(k, (a, b), dtype=jnp.float32) * 0.02 for k, a, b in zip(keys, dims[:-1], dims[1:])]
    bs = [jnp.zeros((b,), dtype=jnp.float32) for b in dims[1:]]
    x = jax.random.normal(key, (batch, dims[0]), dtype=jnp.float32)
    y = jax.random.normal(key, (batch, dims[-1]), dtype=jnp.float32)

    def loss_fn(params):
        ws_, bs_ = params
        h = x
        for i, (w, b) in enumerate(zip(ws_, bs_)):
            h = h @ w + b
            if i < len(ws_) - 1:
                h = jax.nn.elu(h)
        return jnp.mean((h - y) ** 2)

    grad_fn = jax.jit(jax.grad(loss_fn))
    params = (ws, bs)
    jax.block_until_ready(grad_fn(params))  # compile
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(grad_fn(params))
        samples.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(samples)


def bench_torch(batch: int, dims: list[int], reps: int, tf32: bool) -> float:
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    device = "cuda"
    layers = []
    for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
        layers.append(torch.nn.Linear(a, b))
        if i < len(dims) - 2:
            layers.append(torch.nn.ELU())
    net = torch.nn.Sequential(*layers).to(device)
    x = torch.randn(batch, dims[0], device=device)
    y = torch.randn(batch, dims[-1], device=device)

    def step():
        net.zero_grad(set_to_none=True)
        loss = torch.mean((net(x) - y) ** 2)
        loss.backward()

    step()
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        step()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(samples)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=98304)
    ap.add_argument("--obs", type=int, default=100)
    ap.add_argument("--out", type=int, default=29)
    ap.add_argument("--widths", default="128,256,512,1024,2048,4096")
    ap.add_argument("--reps", type=int, default=20)
    args = ap.parse_args()

    widths = [int(w) for w in args.widths.split(",")]
    print(f"\nMLP fwd+bwd, batch={args.batch}, obs={args.obs} -> [W,W,W] -> {args.out}, ELU")
    print(f"{'W':>6} | {'jax ms':>8} {'TFLOP/s':>8} | {'torch f32 ms':>12} | {'torch tf32 ms':>13} | jax/tf32")
    for w in widths:
        dims = [args.obs, w, w, w, args.out]
        fl = _flops(args.batch, dims)
        t_j = bench_jax(args.batch, dims, args.reps)
        t_t32 = bench_torch(args.batch, dims, args.reps, tf32=False)
        t_ttf = bench_torch(args.batch, dims, args.reps, tf32=True)
        print(f"{w:>6} | {t_j:8.3f} {fl / t_j / 1e9:8.1f} | {t_t32:12.3f} | {t_ttf:13.3f} | {t_j / t_ttf:8.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
