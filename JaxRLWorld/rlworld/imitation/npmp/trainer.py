"""NPMP distillation trainer.

Wires together the env, the multi-expert dispatcher, the rolling
trajectory buffer, and the NPMP module + ELBO loss into a single
online-BC training loop. Logging mirrors the RL stack's
``ConsoleWriter`` / ``WandbLogger`` / :class:`IterationData` pattern
so distillation runs render the same boxed iteration display the user
already sees from PPO training (run info, performance, algorithm
metrics, ETA) and stream the same flat dicts to wandb.

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

  3. **Log + (every save_interval) checkpoint** — :class:`NPMPMetrics`
     populates an :class:`IterationData`; ``ConsoleWriter`` and
     ``WandbLogger`` consume it identically to the PPO path. Saved
     checkpoints are uploaded as wandb artifacts so downstream HL
     controller training can pull them by run path.
"""
from __future__ import annotations

import dataclasses
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
from rlworld.imitation.npmp.metrics import NPMPMetrics
from rlworld.imitation.npmp.module import NPMPModule
from rlworld.rl.runners import BaseRunner
from rlworld.rl.runners.iteration_data import EpisodeStats, IterationData
from rlworld.rl.utils.jax_utils import jax_to_torch, torch_to_jax
from rlworld.rl.utils.logger import ConsoleWriter, WandbLogger
from rlworld.rl.utils.utils import setup_log_dir
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
        self._train_start_time = time.time()
        self._just_reset = jnp.ones(env.num_envs, dtype=jnp.bool_)

        self._env.reset()

        # Eval env (lazy). Built smaller than train env for cheap
        # periodic evaluation; reused across all in-training eval calls.
        self._eval_env: "World | None" = None
        self._last_eval_stats = None

        # Logging — log dirs, console writer, wandb.
        self._model_log_dir, self._wandb_log_dir = setup_log_dir(
            output_dir="auto",
        )
        self._console_writer = ConsoleWriter()
        self._wandb_logger: WandbLogger | None = None
        if cfg.use_wandb:
            wandb_cfg = self._cfg_to_wandb_dict()
            self._wandb_logger = WandbLogger(
                log_dir=self._wandb_log_dir,
                project_name=cfg.wandb_project,
                group_name=cfg.wandb_group or cfg.run_name,
                run_name=f"{cfg.run_name}_seed{env.seed}",
                cfg=wandb_cfg,
            )

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

    @property
    def model_log_dir(self) -> str:
        return self._model_log_dir

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
        assert info is not None  # cfg.num_grad_steps validated >= 1
        return info

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------

    def train(self, num_iterations: int) -> None:
        """Drive ``num_iterations`` rollout + update cycles, logging via
        the shared :class:`ConsoleWriter` / :class:`WandbLogger` infra
        and writing checkpoints under :attr:`model_log_dir` every
        ``cfg.save_interval`` iterations.
        """
        env_steps_per_iter = self._cfg.rollout_steps * self._env.num_envs
        end_iter = self._iteration + num_iterations

        for it in range(self._iteration, end_iter):
            self._iteration = it

            t0 = time.time()
            self.rollout_iteration()
            t_rollout = time.time() - t0

            t0 = time.time()
            info = self.update_iteration()
            t_update = time.time() - t0

            # Periodic in-training evaluation (deterministic NPMP rollout
            # in a separate env). Folded into the iteration's logged
            # data so the standard ConsoleWriter / WandbLogger pipeline
            # picks it up alongside the training metrics.
            if self._cfg.eval_interval > 0 and (
                (it + 1) % self._cfg.eval_interval == 0
                or (it + 1) == end_iter
                or it == self._iteration  # also at first iter for early signal
            ):
                self._last_eval_stats = self._run_evaluation()

            iter_data = self._build_iteration_data(
                info=info,
                rollout_time=t_rollout,
                update_time=t_update,
                env_steps_per_iter=env_steps_per_iter,
            )
            context = self._build_console_context(end_iter)

            self._console_writer.write_iteration(iter_data, context)
            if self._wandb_logger is not None:
                self._wandb_logger.log_iteration(
                    iter_data, step=self._total_env_steps,
                )
                # Also push the rich eval stats dict directly to wandb
                # (per-motion / action_gap / z diagnostics that don't
                # fit cleanly into IterationData).
                if self._last_eval_stats is not None:
                    self._wandb_logger.run.log(
                        self._last_eval_stats.to_wandb_dict(prefix="Eval"),
                        step=self._total_env_steps,
                    )

            if (
                (it + 1) % self._cfg.save_interval == 0
                or (it + 1) == end_iter
            ):
                ckpt_dir = os.path.join(
                    self._model_log_dir, f"checkpoint_{it + 1}",
                )
                self.save_checkpoint(ckpt_dir)
                latest = os.path.join(
                    self._model_log_dir, "checkpoint_latest",
                )
                self.save_checkpoint(latest)
                if (
                    self._wandb_logger is not None
                    and self._cfg.upload_checkpoint_artifact
                ):
                    self._wandb_logger.upload_checkpoint_artifact(
                        ckpt_dir, iteration=it + 1,
                        metadata={
                            "iteration": it + 1,
                            "total_env_steps": self._total_env_steps,
                        },
                    )

        self._iteration = end_iter

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _get_or_create_eval_env(self) -> "World":
        """Lazily build a smaller eval-only env separate from training."""
        if self._eval_env is not None:
            return self._eval_env

        eval_cfg = dataclasses.replace(
            self._cfg,
            num_envs=self._cfg.eval_num_envs,
        )
        cfgs_for_run = eval_cfg.build()
        # Disable obs noise during eval — matches RL pipeline's
        # ``eval_disable_noise`` default.
        if hasattr(cfgs_for_run.observation, "enable_noise"):
            cfgs_for_run.observation.enable_noise = False
        self._eval_env = BaseRunner._create_env_from_config(cfgs_for_run)
        return self._eval_env

    def _run_evaluation(self):
        """Deterministic NPMP rollout in the eval env. Returns
        :class:`NPMPEvalStats`. Lazy-imports the evaluator helper to
        avoid the trainer↔evaluator import cycle (evaluator imports
        :class:`NPMPTrainer.load_module`).
        """
        from rlworld.imitation.npmp.evaluator import (
            NPMPEvalStats, run_npmp_eval,
        )
        eval_env = self._get_or_create_eval_env()
        dispatcher = (
            self._dispatcher
            if self._cfg.eval_compute_action_gap else None
        )
        return run_npmp_eval(
            module=self.module,
            env=eval_env,
            num_steps=self._cfg.eval_steps,
            dispatcher=dispatcher,
            per_motion=True,
        )

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _build_iteration_data(
        self,
        info: NPMPLossInfo,
        rollout_time: float,
        update_time: float,
        env_steps_per_iter: int,
    ) -> IterationData:
        decoder_log_std = np.asarray(info.decoder_log_std)
        metrics = NPMPMetrics(
            loss=float(info.loss),
            recon=float(info.recon),
            kl=float(info.kl),
            beta=float(self._cfg.beta),
            decoder_log_std_mean=float(decoder_log_std.mean()),
            decoder_log_std_min=float(decoder_log_std.min()),
            decoder_log_std_max=float(decoder_log_std.max()),
            encoder_q_log_std_mean=float(info.q_log_std_mean),
        )

        total_step_time = rollout_time + update_time
        fps = env_steps_per_iter / total_step_time if total_step_time > 0 else 0.0

        # Fold the most recent eval stats (if any) into EpisodeStats so
        # the shared ConsoleWriter renders the standard "Episode Stats"
        # + "Reward Breakdown" sections under the iteration headline,
        # and WandbLogger emits ``Train/mean_return`` etc. unmodified.
        if self._last_eval_stats is not None:
            ev = self._last_eval_stats
            episode_stats = EpisodeStats(
                return_buffer=[ev.tracking_reward_mean],
                length_buffer=[ev.episode_length_mean],
                reward_stats={
                    name: {"mean": val}
                    for name, val in ev.reward_terms.items()
                },
            )
        else:
            episode_stats = EpisodeStats(
                return_buffer=[], length_buffer=[], reward_stats={},
            )

        return IterationData(
            collection_time=rollout_time,
            learning_time=update_time,
            episode_stats=episode_stats,
            fps=fps,
            metrics=metrics,
            buffer_size=self._buffer.num_filled,
            iteration=self._iteration,
            total_timesteps=self._total_env_steps,
            total_time=time.time() - self._train_start_time,
        )

    def _build_console_context(self, end_iter: int) -> dict[str, Any]:
        sim_name = getattr(self._env, "sim_name", "Newton")
        task_name = getattr(self._env, "task_name", "T1_NPMP_Distill")
        ctx: dict[str, Any] = {
            "total_iterations": end_iter,
            "log_dir": self._model_log_dir,
            "simulator": sim_name,
            "task_name": task_name,
            "wandb_run_name": self._cfg.run_name,
        }
        if self._wandb_logger is not None:
            ctx["wandb_url"] = self._wandb_logger.wandb_url
            ctx["wandb_run_path"] = self._wandb_logger.run.path
        return ctx

    def _cfg_to_wandb_dict(self) -> dict[str, Any]:
        """Best-effort dataclass → dict for wandb config logging.

        Falls back to ``str()`` for any non-serialisable value (e.g. a
        ``CheckpointRef`` that resolves to a wandb artifact URI). The
        wandb side stores whatever it gets; we just want the
        hyperparams visible in the run config UI.
        """
        try:
            return dataclasses.asdict(self._cfg)
        except TypeError:
            return {"run_name": self._cfg.run_name}

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
