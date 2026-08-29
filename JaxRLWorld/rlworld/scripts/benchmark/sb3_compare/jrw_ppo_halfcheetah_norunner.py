"""JaxRLWorld PPO on HalfCheetah-v5, driven WITHOUT the runner stack.

Control experiment for the SB3 comparison. ``jrw_ppo_halfcheetah.py``
goes through ``BaseRunner`` + ``GymnasiumEnv`` (torch tensors on the
Genesis device, JAX<->torch DLPack handoff, preset-derived configs);
this file drives ``PPO`` directly on a plain ``SyncVectorEnv`` with
numpy<->JAX conversions and nothing else. Same hyperparameter table
(``PPO_HP``), same env construction, same collection semantics
(clip -> step -> timeout bootstrap -> GAE -> update), same 100-episode
return window.

A gap between the two therefore isolates the runner/env plumbing; an
equal result points at the learner or the platform instead. That is not
hypothetical: this file is what localised the DLPack lifetime bug fixed
in ``torch_to_jax`` — it reached SB3's return on the same GPU where the
runner reached half of it. Reach for it again whenever the runner path
itself falls under suspicion.

Run:  jaxpy rlworld/scripts/benchmark/sb3_compare/jrw_ppo_halfcheetah_norunner.py
"""

from __future__ import annotations

import math
import time

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium.vector import AutoresetMode, SyncVectorEnv

import wandb
from rlworld.rl.algorithms.ppo.ppo import PPO
from rlworld.rl.configs.common_config_classes import (
    Activation,
    DistributionType,
    MLPActorCfg,
    MLPCriticCfg,
    OrthoInit,
    StdType,
)
from rlworld.rl.modules.policies.ppo_ac import PPOActorCritic
from rlworld.scripts.benchmark.sb3_compare._common import PPO_HP, PPO_ITERS, PROJECT, TASKS, _run_name, _seed

TASK_KEY = "halfcheetah"
WINDOW = 100  # matches EpisodeStatsCollector's window


def main() -> None:
    task = TASKS[TASK_KEY]
    seed = _seed()
    hp = PPO_HP
    n_envs, n_steps = hp["n_envs"], hp["n_steps"]

    # Env construction copied from BaseRunner._create_env_from_config's
    # Gymnasium branch: bare gym.make per lane, sub-env seed = lane index,
    # SAME_STEP autoreset (so ``final_obs`` carries the terminal state).
    def make_env(lane_seed: int):
        def _init():
            e = gym.make(task)
            e.action_space.seed(lane_seed)
            e.observation_space.seed(lane_seed)
            return e

        return _init

    env = SyncVectorEnv([make_env(i) for i in range(n_envs)], autoreset_mode=AutoresetMode.SAME_STEP)
    obs, _ = env.reset(seed=seed)
    obs_dim = obs.shape[1]
    act_dim = env.single_action_space.shape[0]
    low, high = env.single_action_space.low, env.single_action_space.high

    actor_critic = PPOActorCritic(
        num_actor_obs=obs_dim,
        num_critic_obs=obs_dim,
        num_actions=act_dim,
        actor_cfg=MLPActorCfg(
            hidden_dims=list(hp["net"]),
            activation=Activation.TANH,
            init=OrthoInit(output_gain=0.01, hidden_gain=math.sqrt(2.0)),
        ),
        critic_cfg=MLPCriticCfg(
            hidden_dims=list(hp["net"]),
            activation=Activation.TANH,
            init=OrthoInit(output_gain=1.0, hidden_gain=math.sqrt(2.0)),
        ),
        init_noise_std=1.0,
        std_type=StdType.STATE_INDEPENDENT,
        distribution_type=DistributionType.GAUSSIAN,
        obs_normalization=False,
        key=jax.random.PRNGKey(seed),
    )
    rollout = n_envs * n_steps
    alg = PPO(
        actor_critic=actor_critic,
        num_learning_epochs=hp["n_epochs"],
        num_mini_batches=rollout // hp["minibatch_size"],
        clip_param=hp["clip_range"],
        gamma=hp["gamma"],
        lam=hp["gae_lambda"],
        value_loss_coef=hp["vf_coef"],
        entropy_coef=hp["ent_coef"],
        actor_lr=hp["lr"],
        critic_lr=hp["lr"],
        max_grad_norm=hp["max_grad_norm"],
        use_clipped_value_loss=False,
        schedule="fixed",
        desired_kl=None,
        use_early_stop=False,
        optimizer_eps=1e-5,
        normalize_advantage_per_minibatch=True,
        key=jax.random.PRNGKey(seed + 1),
    )
    alg.init_storage(
        {
            "num_envs": n_envs,
            "num_transitions_per_env": n_steps,
            "actor_obs_shape": [obs_dim],
            "critic_obs_shape": [obs_dim],
            "actions_shape": [act_dim],
        }
    )

    wandb.init(
        project=PROJECT,
        name=_run_name("JRW-norunner", "ppo", TASK_KEY, seed),
        config={"task": task, "seed": seed, **hp},
    )

    returns: list[float] = []
    running = np.zeros(n_envs)
    actor_obs = jnp.asarray(obs, jnp.float32)
    t0 = time.time()

    for it in range(PPO_ITERS):
        for _ in range(n_steps):
            actions = alg.act(PPO.ActInput(actor_obs, actor_obs))
            obs, rew, term, trunc, info = env.step(np.clip(np.asarray(actions), low, high))

            running += rew
            done = term | trunc
            infos: dict = {}
            if done.any():
                final = obs.copy()
                for i in np.nonzero(done)[0]:
                    final[i] = info["final_obs"][i]
                    returns.append(float(running[i]))
                    running[i] = 0.0
                infos["final_observation"] = {"critic": jnp.asarray(final, jnp.float32)}

            actor_obs = jnp.asarray(obs, jnp.float32)
            alg.process_env_step(
                jnp.asarray(rew, jnp.float32),
                jnp.asarray(term),
                jnp.asarray(trunc),
                infos,
                next_actor_obs=actor_obs,
                next_critic_obs=actor_obs,
            )

        alg.compute_returns(actor_obs)
        metrics = alg.update()

        if returns:
            mean_return = float(np.mean(returns[-WINDOW:]))
            wandb.log({"Train/mean_return": mean_return}, step=(it + 1) * rollout)
            if (it + 1) % 50 == 0:
                print(
                    f"iter {it + 1:4d}  steps {(it + 1) * rollout:8d}  ret {mean_return:9.1f}"
                    f"  std {metrics.actor.std:.3f}  kl {metrics.kl.approx_kl:.4f}"
                    f"  clip {metrics.kl.clip_fraction:.3f}  ({time.time() - t0:.0f}s)",
                    flush=True,
                )

    print("FINAL", float(np.mean(returns[-WINDOW:])), flush=True)
    wandb.finish()


if __name__ == "__main__":
    main()
