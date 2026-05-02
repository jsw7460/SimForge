"""NPMP distillation trainer.

Wires together the env, the multi-expert dispatcher, the rolling
trajectory buffer, and the NPMP module + ELBO loss into a single
online-BC training loop.

Per outer iteration:

  1. **Rollout phase** — clear the buffer, then run
     ``cfg.rollout_steps`` env steps. Each step gathers
     ``(decoder_input, encoder_input, mu_E, ep_starts)`` for every env
     and applies ``mu_E + DART noise`` to the env. ``ep_starts`` is
     True at env reset *or* motion-command rollover so the encoder's
     AR(1) z-prior resets at every kinematic-motion discontinuity.

  2. **Update phase** — ``cfg.num_grad_steps`` minibatches, each
     ``cfg.batch_traj`` length-``cfg.traj_len`` time-contiguous
     trajectories sampled from the buffer. Each minibatch runs the
     ELBO loss + clipped Adam step on the NPMP module.

Checkpoints persist (a) the NPMP weights via
``eqx.tree_serialise_leaves``, and (b) a small YAML metadata sidecar
that captures the architecture + train-state counters needed to
reload the module standalone — i.e. without re-instantiating the full
distillation config — so downstream HL-controller training can pick
up just the motor primitive module.
"""
from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from rlworld.imitation.npmp.buffer import NPMPBuffer
from rlworld.imitation.npmp.config import T1NPMPDistillConfig
from rlworld.imitation.npmp.expert_dispatch import MultiExpertDispatcher
from rlworld.imitation.npmp.loss import NPMPLossInfo, npmp_elbo_loss
from rlworld.imitation.npmp.module import NPMPModule
from rlworld.rl.utils.jax_utils import jax_to_torch, torch_to_jax
from rlworld.rl.utils.yaml_io import dump_yaml, load_yaml

if TYPE_CHECKING:
    from rlworld.rl.envs.world import World


__all__ = ["NPMPTrainer"]


# ── JIT-compiled gradient step ──────────────────────────────────────


def _build_optimizer(learning_rate: float, max_grad_norm: float) -> optax.GradientTransformation:
    return optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(learning_rate),
    )


@eqx.filter_jit
def _update_step(
    params: Any,
    static: Any,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    batch,
    beta: float,
    key: jax.Array,
) -> tuple[Any, optax.OptState, NPMPLossInfo]:
    """One ELBO gradient step. Returns (new_params, new_opt_state, info)."""

    def loss_fn(p):
        module = eqx.combine(p, static)
        loss, info = npmp_elbo_loss(module, batch, beta=beta, key=key)
        return loss, info

    (_loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, info


# ── Trainer ─────────────────────────────────────────────────────────


class NPMPTrainer:
    """Online BC distillation of N tracking experts into one NPMP module."""

    def __init__(
        self,
        cfg: T1NPMPDistillConfig,
        env: "World",
        dispatcher: MultiExpertDispatcher,
        key: jax.Array,
    ):
        self._cfg = cfg
        self._env = env
        self._dispatcher = dispatcher

        obs_dim = env.calculate_obs_dim()
        for required in ("decoder_input", "encoder_input"):
            if required not in obs_dim:
                raise ValueError(
                    f"NPMPTrainer expects env obs to include the "
                    f"{required!r} group; got {sorted(obs_dim)}."
                )
        self._s_dim = obs_dim["decoder_input"]
        self._x_dim = obs_dim["encoder_input"]
        self._action_dim = env.num_actions

        # Buffer.
        self._buffer = NPMPBuffer(
            num_envs=env.num_envs,
            max_steps=cfg.rollout_steps,
            s_dim=self._s_dim,
            x_dim=self._x_dim,
            action_dim=self._action_dim,
        )

        # NPMP module.
        key_module, self._key = jax.random.split(key)
        self._module = NPMPModule(
            s_dim=self._s_dim,
            x_dim=self._x_dim,
            action_dim=self._action_dim,
            latent_dim=cfg.latent_dim,
            encoder_hidden=tuple(cfg.encoder_hidden),
            decoder_hidden=tuple(cfg.decoder_hidden),
            ar1_alpha=cfg.ar1_alpha,
            decoder_log_std_init=cfg.decoder_log_std_init,
            key=key_module,
        )

        # Optimizer.
        self._optimizer = _build_optimizer(
            learning_rate=cfg.learning_rate,
            max_grad_norm=cfg.max_grad_norm,
        )
        self._params, self._static = eqx.partition(
            self._module, eqx.is_inexact_array,
        )
        self._opt_state = self._optimizer.init(self._params)

        # Counters / per-env state.
        self._iteration = 0
        self._total_env_steps = 0
        self._just_reset = jnp.ones(env.num_envs, dtype=jnp.bool_)

        self._env.reset()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def buffer(self) -> NPMPBuffer:
        return self._buffer

    @property
    def module(self) -> NPMPModule:
        """Up-to-date NPMP module (params combined with static)."""
        return eqx.combine(self._params, self._static)

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------

    def _rollout_one_step(self, key: jax.Array) -> jax.Array:
        cmd = self._env.command_manager.get_term("motion")

        obs = self._env.obs_manager.get_observation()
        actor_obs = torch_to_jax(obs["actor"])
        decoder_s = torch_to_jax(obs["decoder_input"])
        encoder_x = torch_to_jax(obs["encoder_input"])
        motion_ids = torch_to_jax(cmd.motion_ids)
        ep_starts = self._just_reset

        prev_time_steps = cmd.time_steps.clone()

        mu_E = self._dispatcher.deterministic_mean(actor_obs, motion_ids)
        self._buffer.add(decoder_s, encoder_x, mu_E, ep_starts)

        noise_key, key = jax.random.split(key)
        noise = jax.random.normal(noise_key, mu_E.shape) * self._cfg.expert_noise_std
        action_to_env = jax_to_torch(mu_E + noise, self._env.device)
        _, _, terminated, truncated, _ = self._env.step(action_to_env)

        new_time_steps = cmd.time_steps
        rollover = new_time_steps != (prev_time_steps + 1)
        env_reset = terminated | truncated
        self._just_reset = torch_to_jax(env_reset | rollover)

        return key

    def rollout_iteration(self) -> None:
        self._buffer.clear()
        for _ in range(self._cfg.rollout_steps):
            self._key = self._rollout_one_step(self._key)
        self._total_env_steps += self._cfg.rollout_steps * self._env.num_envs

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update_iteration(self) -> NPMPLossInfo:
        """Run ``cfg.num_grad_steps`` ELBO gradient steps. Returns last info."""
        info = None
        for _ in range(self._cfg.num_grad_steps):
            sample_key, step_key, self._key = jax.random.split(self._key, 3)
            batch = self._buffer.sample_trajectories(
                n_traj=self._cfg.batch_traj,
                traj_len=self._cfg.traj_len,
                key=sample_key,
            )
            self._params, self._opt_state, info = _update_step(
                self._params,
                self._static,
                self._opt_state,
                self._optimizer,
                batch,
                self._cfg.beta,
                step_key,
            )
        assert info is not None  # num_grad_steps validated >=1 in cfg
        return info

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------

    def train(
        self,
        num_iterations: int,
        save_dir: str | None = None,
    ) -> None:
        """Drive ``num_iterations`` rollout + update cycles.

        If ``save_dir`` is given, snapshots NPMP weights every
        ``cfg.save_interval`` iterations into
        ``{save_dir}/checkpoint_{it}/`` plus a rolling
        ``{save_dir}/checkpoint_latest/``.
        """
        env_steps_per_iter = self._cfg.rollout_steps * self._env.num_envs

        for it in range(self._iteration, self._iteration + num_iterations):
            self._iteration = it

            t0 = time.time()
            self.rollout_iteration()
            t_rollout = time.time() - t0

            t0 = time.time()
            info = self.update_iteration()
            t_update = time.time() - t0

            print(
                f"[npmp it {it:5d}] "
                f"loss={float(info.loss):8.4f} "
                f"recon={float(info.recon):8.4f} "
                f"kl={float(info.kl):8.4f} "
                f"dec_log_std_mean={float(info.decoder_log_std.mean()):+.3f} "
                f"| rollout {t_rollout:5.2f}s update {t_update:5.2f}s "
                f"| env_steps={self._total_env_steps:,}"
            )

            if save_dir and (
                (it + 1) % self._cfg.save_interval == 0
                or (it + 1) == self._iteration + num_iterations
            ):
                ckpt_dir = os.path.join(save_dir, f"checkpoint_{it + 1}")
                self.save_checkpoint(ckpt_dir)
                latest = os.path.join(save_dir, "checkpoint_latest")
                self.save_checkpoint(latest)

        self._iteration += 1  # advance past last completed iter

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save_checkpoint(self, save_dir: str) -> None:
        """Persist NPMP module weights + arch metadata under ``save_dir``."""
        os.makedirs(save_dir, exist_ok=True)

        model_path = os.path.join(save_dir, "model.eqx")
        eqx.tree_serialise_leaves(model_path, self.module)

        meta = {
            "iteration": int(self._iteration),
            "total_env_steps": int(self._total_env_steps),
            "key": np.asarray(self._key).tolist(),
            "arch": {
                "s_dim": int(self._s_dim),
                "x_dim": int(self._x_dim),
                "action_dim": int(self._action_dim),
                "latent_dim": int(self._cfg.latent_dim),
                "encoder_hidden": list(self._cfg.encoder_hidden),
                "decoder_hidden": list(self._cfg.decoder_hidden),
                "ar1_alpha": float(self._cfg.ar1_alpha),
                "decoder_log_std_init": float(self._cfg.decoder_log_std_init),
            },
        }
        dump_yaml(os.path.join(save_dir, "npmp_meta.yaml"), meta)
        print(f"[npmp ckpt] saved → {save_dir}")

    # ------------------------------------------------------------------
    # Standalone module loader (for downstream HL controller etc.)
    # ------------------------------------------------------------------

    @classmethod
    def load_module(cls, checkpoint_dir: str) -> NPMPModule:
        """Reconstruct an :class:`NPMPModule` from a saved checkpoint
        directory, without needing the original :class:`T1NPMPDistillConfig`.

        Reads ``npmp_meta.yaml`` for architecture, builds an empty
        :class:`NPMPModule`, then deserialises ``model.eqx`` into it.
        """
        meta = load_yaml(os.path.join(checkpoint_dir, "npmp_meta.yaml"))
        arch = meta["arch"]
        empty = NPMPModule(
            s_dim=arch["s_dim"],
            x_dim=arch["x_dim"],
            action_dim=arch["action_dim"],
            latent_dim=arch["latent_dim"],
            encoder_hidden=tuple(arch["encoder_hidden"]),
            decoder_hidden=tuple(arch["decoder_hidden"]),
            ar1_alpha=arch["ar1_alpha"],
            decoder_log_std_init=arch["decoder_log_std_init"],
            key=jax.random.PRNGKey(0),
        )
        return eqx.tree_deserialise_leaves(
            os.path.join(checkpoint_dir, "model.eqx"), empty,
        )
