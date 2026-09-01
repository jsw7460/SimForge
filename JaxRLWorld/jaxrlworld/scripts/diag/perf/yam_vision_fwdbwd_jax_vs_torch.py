"""yam_lift vision policy, JAX vs torch: identical network, measured speed.

The MLP width sweep (``mlp_fwdbwd_jax_vs_torch``) attributed the PPO
learn-time gap to small-GEMM kernel selection: XLA loses ~1.24x to
cuBLAS-TF32 at W=512 and converges to ~1.08x at W=4096. The yam_lift
vision policy is the workload where that verdict matters next — a conv
encoder + spatial softmax in front of a [256, 256, 128] trunk — and
convolutions exercise a different kernel path (cuDNN vs XLA conv)
entirely.

This bench compares the REAL policy, not a stand-in:

  1. The JAX side is built through the same factory training uses
     (``build_actor`` / ``build_critic`` with the yam_lift vision
     preset's actual cfgs), with the observation shapes read from a
     briefly-built env.
  2. The torch side mirrors it layer by layer (Conv2d with the same
     rsl_rl ceil-mode padding, ELU, spatial softmax, Linear trunk),
     the weights are COPIED across, and a forward-equivalence gate
     (max |Δ| with TF32 off) must pass before anything is timed —
     otherwise the two frameworks are not running the same network and
     the comparison is void.
  3. Timed at the real training shapes: inference forward at the
     rollout batch (num_envs) and forward+backward at the PPO
     minibatch (num_envs * num_steps / num_minibatches). torch runs
     both full-f32 and TF32 (mjlab/rsl_rl enable TF32); JAX default
     matmul precision is already TF32-class on this hardware.

Deliberately omitted (identical scalar work on both sides, irrelevant
to the kernel question): obs normalisation, the log-std head, and the
PPO loss algebra.

Usage:
    jaxpy -m jaxrlworld.scripts.diag.perf.yam_vision_fwdbwd_jax_vs_torch
    jaxpy -m jaxrlworld.scripts.diag.perf.yam_vision_fwdbwd_jax_vs_torch \
        --no-from-env --actor-dim 39 --critic-dim 52 --actions 7
"""

from __future__ import annotations

import argparse
import statistics
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import torch

from jaxrlworld.rl.configs.presets.yam_lift.vision import YamLiftVisionConfig
from jaxrlworld.rl.modules.architectures.actor_registry import build_actor, build_critic

_IMAGE_GROUP = "camera"
_ACTOR_GROUP = "actor"
_CRITIC_GROUP = "critic"


# ── shape discovery ──────────────────────────────────────────────────


def _shapes_from_env() -> tuple[dict[str, tuple[int, ...]], int]:
    """Build the mujoco env briefly and read every obs group's per-env shape."""
    from jaxrlworld.rl.runners import BaseRunner

    cfgs = YamLiftVisionConfig(sim_type="mujoco", num_envs=64).build()
    runner = BaseRunner.create_with_env(cfgs)
    env = runner.env
    env.reset()
    obs_shapes = {name: tuple(t.shape[1:]) for name, t in env.obs_manager.obs_dict.items()}
    num_actions = env.num_actions
    del runner, env
    torch.cuda.empty_cache()
    return obs_shapes, num_actions


# ── torch mirror ─────────────────────────────────────────────────────


class TorchSpatialSoftmax(torch.nn.Module):
    def __init__(self, height: int, width: int, temperature: float):
        super().__init__()
        pos_h, pos_w = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height),
            torch.linspace(-1.0, 1.0, width),
            indexing="ij",
        )
        self.register_buffer("pos_h", pos_h.reshape(1, -1))
        self.register_buffer("pos_w", pos_w.reshape(1, -1))
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c = x.shape[0], x.shape[1]
        features = x.reshape(n, c, -1)
        weights = torch.softmax(features / self.temperature, dim=-1)
        expected_h = (weights * self.pos_h).sum(dim=-1)
        expected_w = (weights * self.pos_w).sum(dim=-1)
        return torch.stack([expected_h, expected_w], dim=-1).reshape(n, c * 2)


class TorchVisionNet(torch.nn.Module):
    """Layer-by-layer mirror of one jax VisionActor/VisionCritic."""

    def __init__(self, jax_encoder, jax_mlp):
        super().__init__()
        convs = []
        for conv in jax_encoder.convs:
            pad = tuple(p[0] for p in conv.padding)  # eqx ((h,h),(w,w)) -> (h,w)
            convs.append(
                torch.nn.Conv2d(
                    in_channels=conv.in_channels,
                    out_channels=conv.out_channels,
                    kernel_size=conv.kernel_size,
                    stride=conv.stride,
                    padding=pad,
                    dilation=conv.dilation,
                )
            )
        self.convs = torch.nn.ModuleList(convs)
        sps = jax_encoder.spatial_softmax
        if sps is None:
            raise ValueError("The yam_lift vision encoder uses spatial softmax; the mirror expects it.")
        # Recover (H, W) from the coordinate grid rather than assuming square.
        pos_h = np.asarray(sps.pos_h).reshape(-1)
        height = int(np.unique(np.round(pos_h, 6)).size)
        width = pos_h.size // height
        self.spatial_softmax = TorchSpatialSoftmax(height, width, sps.temperature)
        self.linears = torch.nn.ModuleList(
            [torch.nn.Linear(lin.in_features, lin.out_features) for lin in jax_mlp.linears]
        )
        self._copy_weights(jax_encoder, jax_mlp)

    @torch.no_grad()
    def _copy_weights(self, jax_encoder, jax_mlp) -> None:
        for tconv, jconv in zip(self.convs, jax_encoder.convs, strict=True):
            tconv.weight.copy_(torch.from_numpy(np.array(jconv.weight)))
            tconv.bias.copy_(torch.from_numpy(np.array(jconv.bias).reshape(-1)))
        for tlin, jlin in zip(self.linears, jax_mlp.linears, strict=True):
            tlin.weight.copy_(torch.from_numpy(np.array(jlin.weight)))
            tlin.bias.copy_(torch.from_numpy(np.array(jlin.bias).reshape(-1)))

    def forward(self, vec: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        x = image
        for conv in self.convs:
            x = torch.nn.functional.elu(conv(x))
        latent = self.spatial_softmax(x)
        h = torch.cat([vec, latent], dim=-1)
        for i, lin in enumerate(self.linears):
            h = lin(h)
            if i < len(self.linears) - 1:
                h = torch.nn.functional.elu(h)
        return h


# ── jax side ─────────────────────────────────────────────────────────


def _jax_forward_fns(actor, critic):
    """Batched jitted forwards and a fwd+bwd grad step for both nets."""

    def actor_fwd(model, vec, img):
        return jax.vmap(lambda v, i: model({_ACTOR_GROUP: v, _IMAGE_GROUP: i})[0])(vec, img)

    def critic_fwd(model, vec, img):
        return jax.vmap(lambda v, i: model({_CRITIC_GROUP: v, _IMAGE_GROUP: i})[0])(vec, img)

    @eqx.filter_jit
    def infer(model, vec, img):
        return actor_fwd(model, vec, img)

    def loss_fn(models, batch):
        a, c = models
        av, ai, cv, ci = batch
        act = actor_fwd(a, av, ai)
        val = critic_fwd(c, cv, ci)
        return jnp.mean(jnp.square(act)) + jnp.mean(jnp.square(val))

    grad_step = eqx.filter_jit(eqx.filter_grad(loss_fn))
    return infer, grad_step


def _median_ms(fn, reps: int) -> float:
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(samples)


def _set_tf32(on: bool) -> None:
    torch.backends.cuda.matmul.allow_tf32 = on
    torch.backends.cudnn.allow_tf32 = on


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-env", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--actor-dim", type=int, default=None, help="Actor state-vector width (with --no-from-env).")
    ap.add_argument("--critic-dim", type=int, default=None, help="Critic state-vector width (with --no-from-env).")
    ap.add_argument("--actions", type=int, default=None, help="Action dimension (with --no-from-env).")
    ap.add_argument("--num-envs", type=int, default=4096, help="Rollout batch for the inference row.")
    ap.add_argument("--num-steps", type=int, default=24)
    ap.add_argument("--num-minibatches", type=int, default=4)
    ap.add_argument("--reps", type=int, default=30)
    args = ap.parse_args()

    cfg = YamLiftVisionConfig(sim_type="mujoco", num_envs=args.num_envs)
    nn = cfg._build_nn_config()
    actor_cfg, critic_cfg = nn.policy.actor, nn.policy.critic

    if args.from_env:
        obs_shapes, num_actions = _shapes_from_env()
    else:
        if args.actor_dim is None or args.critic_dim is None or args.actions is None:
            raise SystemExit("--no-from-env requires --actor-dim, --critic-dim and --actions.")
        obs_shapes = {
            _ACTOR_GROUP: (args.actor_dim,),
            _CRITIC_GROUP: (args.critic_dim,),
            _IMAGE_GROUP: (1, cfg.camera_height, cfg.camera_width),
        }
        num_actions = args.actions

    actor_dim = obs_shapes[_ACTOR_GROUP][0]
    critic_dim = obs_shapes[_CRITIC_GROUP][0]
    img_shape = obs_shapes[_IMAGE_GROUP]
    minibatch = args.num_envs * args.num_steps // args.num_minibatches

    print(f"\nyam_lift vision policy: actor vec {actor_dim}, critic vec {critic_dim}, image {img_shape}, ")
    print(f"actions {num_actions}, cnn {actor_cfg.cnn.output_channels} k{actor_cfg.cnn.kernel_size} ")
    print(f"s{actor_cfg.cnn.stride} + spatial softmax, trunk {actor_cfg.trunk.hidden_dims}")
    print(f"inference batch {args.num_envs}, update minibatch {minibatch}\n")

    key = jax.random.PRNGKey(0)
    k_a, k_c = jax.random.split(key)
    jax_actor = build_actor(
        actor_cfg, num_obs=actor_dim, num_actions=num_actions, key=k_a, obs_shapes=obs_shapes, vector_group=_ACTOR_GROUP
    )
    jax_critic = build_critic(
        critic_cfg, num_obs=critic_dim, key=k_c, obs_shapes=obs_shapes, vector_group=_CRITIC_GROUP
    )

    torch_actor = TorchVisionNet(jax_actor.encoders[_IMAGE_GROUP], jax_actor.trunk.net).cuda()
    torch_critic = TorchVisionNet(jax_critic.encoders[_IMAGE_GROUP], jax_critic.trunk.net).cuda()

    n_jax = sum(int(np.asarray(x).size) for x in jax.tree_util.tree_leaves(eqx.filter(jax_actor, eqx.is_array)))
    # The spatial-softmax coordinate grids are jax array leaves but torch
    # buffers; count both sides with them included so the totals match.
    n_torch = sum(p.numel() for p in torch_actor.parameters()) + sum(b.numel() for b in torch_actor.buffers())
    print(f"actor arrays (incl. coordinate grids): jax {n_jax}, torch {n_torch}")
    if n_jax != n_torch:
        raise SystemExit("Array counts differ — the mirror is not the same network.")

    # ── equivalence gate (full f32 on BOTH sides: exactness before speed) ──
    # torch's TF32 switch is turned off below; jax needs the same, since
    # its DEFAULT matmul/conv precision on this hardware is already
    # TF32-class — comparing jax-TF32 against torch-f32 reads ~1e-3 of
    # pure rounding and would fail the gate with identical weights.
    _set_tf32(False)
    rng = np.random.default_rng(0)
    vec_np = rng.standard_normal((256, actor_dim), dtype=np.float32)
    img_np = rng.standard_normal((256, *img_shape), dtype=np.float32)
    with jax.default_matmul_precision("highest"):
        out_jax = np.asarray(
            jax.vmap(lambda v, i: jax_actor({_ACTOR_GROUP: v, _IMAGE_GROUP: i})[0])(
                jnp.asarray(vec_np), jnp.asarray(img_np)
            )
        )
    with torch.no_grad():
        out_torch = torch_actor(torch.from_numpy(vec_np).cuda(), torch.from_numpy(img_np).cuda()).cpu().numpy()
    max_dev = float(np.abs(out_jax - out_torch).max())
    print(f"forward equivalence, max |jax - torch|: {max_dev:.2e}")
    if max_dev > 2e-4:
        raise SystemExit("The two frameworks are not computing the same network; refusing to time them.")

    # ── benches ──────────────────────────────────────────────────────
    infer_jax, grad_jax = _jax_forward_fns(jax_actor, jax_critic)

    def make_jax_batch(batch):
        ks = jax.random.split(jax.random.PRNGKey(1), 4)
        return (
            jax.random.normal(ks[0], (batch, actor_dim), dtype=jnp.float32),
            jax.random.normal(ks[1], (batch, *img_shape), dtype=jnp.float32),
            jax.random.normal(ks[2], (batch, critic_dim), dtype=jnp.float32),
            jax.random.normal(ks[3], (batch, *img_shape), dtype=jnp.float32),
        )

    def make_torch_batch(batch):
        return (
            torch.randn(batch, actor_dim, device="cuda"),
            torch.randn(batch, *img_shape, device="cuda"),
            torch.randn(batch, critic_dim, device="cuda"),
            torch.randn(batch, *img_shape, device="cuda"),
        )

    # inference forward, rollout batch
    av, ai, _, _ = make_jax_batch(args.num_envs)
    jax.block_until_ready(infer_jax(jax_actor, av, ai))
    t_inf_jax = _median_ms(lambda: jax.block_until_ready(infer_jax(jax_actor, av, ai)), args.reps)

    tav, tai, _, _ = make_torch_batch(args.num_envs)

    def torch_infer():
        with torch.no_grad():
            torch_actor(tav, tai)
        torch.cuda.synchronize()

    rows = [("inference fwd", args.num_envs, t_inf_jax)]
    torch_inf = {}
    for tf32 in (False, True):
        _set_tf32(tf32)
        torch_infer()
        torch_inf[tf32] = _median_ms(torch_infer, args.reps)

    # update fwd+bwd, PPO minibatch
    jb = make_jax_batch(minibatch)
    models = (jax_actor, jax_critic)

    def run_grad():
        grads = grad_jax(models, jb)
        jax.block_until_ready(jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_array)))

    run_grad()  # compile
    t_upd_jax = _median_ms(run_grad, args.reps)

    tb = make_torch_batch(minibatch)

    def torch_update():
        torch_actor.zero_grad(set_to_none=True)
        torch_critic.zero_grad(set_to_none=True)
        loss = torch.mean(torch_actor(tb[0], tb[1]) ** 2) + torch.mean(torch_critic(tb[2], tb[3]) ** 2)
        loss.backward()
        torch.cuda.synchronize()

    torch_upd = {}
    for tf32 in (False, True):
        _set_tf32(tf32)
        torch_update()
        torch_upd[tf32] = _median_ms(torch_update, args.reps)
    rows.append(("update fwd+bwd", minibatch, t_upd_jax))

    print(f"\n{'stage':<16} {'batch':>8} | {'jax ms':>8} | {'torch f32':>10} | {'torch tf32':>10} | jax/tf32")
    torch_vals = [(torch_inf[False], torch_inf[True]), (torch_upd[False], torch_upd[True])]
    for (name, batch, t_j), (t_f32, t_tf32) in zip(rows, torch_vals, strict=True):
        print(f"{name:<16} {batch:>8} | {t_j:8.3f} | {t_f32:10.3f} | {t_tf32:10.3f} | {t_j / t_tf32:8.2f}")
    print(
        "\nReading: jax/tf32 > 1 means torch-TF32 wins on this workload; the\n"
        "f32 column is torch without mjlab's TF32 switch. The equivalence\n"
        "gate above proves both frameworks ran the same weights."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
