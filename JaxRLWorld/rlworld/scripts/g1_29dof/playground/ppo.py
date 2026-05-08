"""
Diagnostic: PPO on MuJoCo Playground G1JoystickFlatTerrain.

Uses our PPO algorithm directly with the Playground environment,
bypassing OnPolicyRunner. This isolates whether the issue is in the
algorithm or in the reward/obs of Newton/Genesis/MuJoCo environments.
"""

import os
import sys

os.environ["MUJOCO_GL"] = "egl" if sys.platform != "darwin" else "glfw"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"

import time

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import optax
import torch
from mujoco_playground import registry, wrapper_torch

from rlworld.rl.algorithms.base import ActInput
from rlworld.rl.algorithms.ppo.ppo import PPO
from rlworld.rl.modules.policies.ppo_ac import PPOActorCritic

# ===================== Config =====================

ENV_NAME = "G1JoystickFlatTerrain"
NUM_ENVS = 8192
NUM_EVAL_ENVS = 1024
SEED = 1
DEVICE_RANK = 0

# PPO hyperparameters
NUM_STEPS_PER_ENV = 20
MAX_ITERATIONS = 15000
GAMMA = 0.97
LAM = 0.95
CLIP_PARAM = 0.2
ACTOR_LR = 3e-4
CRITIC_LR = 3e-4
NUM_LEARNING_EPOCHS = 4
NUM_MINI_BATCHES = 32
ENTROPY_COEF = 0.005
VALUE_LOSS_COEF = 1.0
MAX_GRAD_NORM = 1.0
SCHEDULE = "fixed"
DESIRED_KL = 0.01
USE_CLIPPED_VALUE_LOSS = True
OBS_NORMALIZATION = True

# Network
ACTOR_HIDDEN = [512, 256, 128]
CRITIC_HIDDEN = [512, 256, 128]
INIT_NOISE_STD = 0.5

EVAL_INTERVAL = 100
USE_WANDB = True


def torch_to_jax(t: torch.Tensor) -> jax.Array:
    return jnp.asarray(t.detach().cpu().numpy())


def jax_to_torch(a: jax.Array, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(np.asarray(a), device=device)


def main():
    device = torch.device(f"cuda:{DEVICE_RANK}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ===================== Wandb =====================
    if USE_WANDB:
        import wandb

        wandb.init(
            project="PPO-Playground-Diagnostic",
            name=f"{ENV_NAME}__ours__seed{SEED}",
            config={
                "env_name": ENV_NAME,
                "num_envs": NUM_ENVS,
                "num_steps_per_env": NUM_STEPS_PER_ENV,
                "gamma": GAMMA,
                "lam": LAM,
                "clip_param": CLIP_PARAM,
                "actor_lr": ACTOR_LR,
                "num_learning_epochs": NUM_LEARNING_EPOCHS,
                "num_mini_batches": NUM_MINI_BATCHES,
                "entropy_coef": ENTROPY_COEF,
                "schedule": SCHEDULE,
                "desired_kl": DESIRED_KL,
                "obs_normalization": OBS_NORMALIZATION,
            },
        )

    # ===================== Training Environment =====================
    train_env_cfg = registry.get_default_config(ENV_NAME)
    train_env_cfg.push_config.enable = False
    train_env_cfg.push_config.magnitude_range = [0.0, 0.0]
    raw_train_env = registry.load(ENV_NAME, config=train_env_cfg)
    train_env = wrapper_torch.RSLRLBraxWrapper(
        raw_train_env,
        NUM_ENVS,
        SEED,
        train_env_cfg.episode_length,
        train_env_cfg.action_repeat,
        device_rank=DEVICE_RANK,
    )
    max_episode_steps = train_env_cfg.episode_length

    n_obs = train_env.num_obs if isinstance(train_env.num_obs, int) else train_env.num_obs[0]
    n_act = train_env.num_actions
    asymmetric = train_env.asymmetric_obs
    if asymmetric:
        n_critic_obs = (
            train_env.num_privileged_obs
            if isinstance(train_env.num_privileged_obs, int)
            else train_env.num_privileged_obs[0]
        )
    else:
        n_critic_obs = n_obs

    print(f"Env: {ENV_NAME}")
    print(f"  obs_dim={n_obs}, critic_obs_dim={n_critic_obs}, act_dim={n_act}")
    print(f"  asymmetric_obs={asymmetric}")
    print(f"  episode_length={max_episode_steps}")

    # ===================== Eval Environment =====================
    eval_env_cfg = registry.get_default_config(ENV_NAME)
    eval_env_cfg.push_config.enable = False
    eval_env_cfg.push_config.magnitude_range = [0.0, 0.0]
    raw_eval_env = registry.load(ENV_NAME, config=eval_env_cfg)

    eval_jit_reset = jax.jit(jax.vmap(raw_eval_env.reset))
    eval_jit_step = jax.jit(jax.vmap(raw_eval_env.step))
    eval_key = jax.random.PRNGKey(SEED + 100)
    gpu_devices = jax.devices("gpu")
    if gpu_devices:
        eval_key = jax.device_put(eval_key, gpu_devices[DEVICE_RANK])
    eval_key_reset = jax.random.split(eval_key, NUM_EVAL_ENVS)

    # ===================== Render Environment (1 env) =====================
    render_env_cfg = registry.get_default_config(ENV_NAME)
    render_env_cfg.push_config.enable = False
    render_env_cfg.push_config.magnitude_range = [0.0, 0.0]
    raw_render_env = registry.load(ENV_NAME, config=render_env_cfg)
    render_jit_reset = jax.jit(raw_render_env.reset)
    render_jit_step = jax.jit(raw_render_env.step)
    render_key = jax.random.PRNGKey(SEED + 200)
    if gpu_devices:
        render_key = jax.device_put(render_key, gpu_devices[DEVICE_RANK])

    # ===================== Algorithm =====================
    key = jax.random.PRNGKey(SEED)
    key, model_key = jax.random.split(key)

    actor_critic = PPOActorCritic(
        num_actor_obs=n_obs,
        num_critic_obs=n_critic_obs,
        num_actions=n_act,
        actor_class_name="MLPActor",
        init_noise_std=INIT_NOISE_STD,
        std_type="state_dependent",
        distribution_type="squashed_gaussian",
        obs_normalization=OBS_NORMALIZATION,
        key=model_key,
        actor_kwargs={
            "hidden_dims": ACTOR_HIDDEN,
            "activation": "elu",
            "ortho_init": True,
            "output_gain": 0.01,
        },
        critic_kwargs={
            "hidden_dims": CRITIC_HIDDEN,
            "activation": "elu",
            "ortho_init": True,
        },
    )

    alg = PPO(
        actor_critic=actor_critic,
        num_learning_epochs=NUM_LEARNING_EPOCHS,
        num_mini_batches=NUM_MINI_BATCHES,
        clip_param=CLIP_PARAM,
        gamma=GAMMA,
        lam=LAM,
        value_loss_coef=VALUE_LOSS_COEF,
        entropy_coef=ENTROPY_COEF,
        actor_lr=ACTOR_LR,
        critic_lr=CRITIC_LR,
        max_grad_norm=MAX_GRAD_NORM,
        use_clipped_value_loss=USE_CLIPPED_VALUE_LOSS,
        schedule=SCHEDULE,
        desired_kl=DESIRED_KL,
        use_early_stop=False,
        optimizer_class=optax.adam,
        key=key,
    )

    alg.init_storage(
        {
            "num_envs": NUM_ENVS,
            "num_transitions_per_env": NUM_STEPS_PER_ENV,
            "actor_obs_shape": [n_obs],
            "critic_obs_shape": [n_critic_obs],
            "actions_shape": [n_act],
        }
    )

    # ===================== Evaluate =====================
    def evaluate():
        state = eval_jit_reset(eval_key_reset)
        episode_returns = jnp.zeros(NUM_EVAL_ENVS)
        episode_lengths = jnp.zeros(NUM_EVAL_ENVS)
        done_masks = jnp.zeros(NUM_EVAL_ENVS, dtype=bool)

        for _ in range(max_episode_steps):
            if asymmetric:
                actor_obs_jax = jnp.asarray(state.obs["state"])
                critic_obs_jax = jnp.asarray(state.obs["privileged_state"])
            else:
                actor_obs_jax = jnp.asarray(state.obs)
                critic_obs_jax = actor_obs_jax
            act_input = ActInput(actor_obs=actor_obs_jax, critic_obs=critic_obs_jax)
            actions = alg.act(act_input, deterministic=True)
            actions_jax = jnp.asarray(actions)
            state = eval_jit_step(state, actions_jax)

            rewards = state.reward
            dones = state.done
            episode_returns = jnp.where(~done_masks, episode_returns + rewards, episode_returns)
            episode_lengths = jnp.where(~done_masks, episode_lengths + 1, episode_lengths)
            done_masks = done_masks | (dones > 0)
            if done_masks.all():
                break

        return float(episode_returns.mean()), float(episode_lengths.mean())

    def render_with_rollout():
        """Single-env rollout, collect states, render video."""
        state = render_jit_reset(render_key)
        state.info["command"] = jnp.array([1.0, 0.0, 0.0])
        trajectory = [state]

        for i in range(max_episode_steps):
            if asymmetric:
                actor_obs = jnp.asarray(state.obs["state"])[None]
                critic_obs_r = jnp.asarray(state.obs["privileged_state"])[None]
            else:
                actor_obs = jnp.asarray(state.obs)[None]
                critic_obs_r = actor_obs
            act_in = ActInput(actor_obs=actor_obs, critic_obs=critic_obs_r)
            actions = alg.act(act_in, deterministic=True)
            actions_jax = jnp.asarray(actions)[0]
            state = render_jit_step(state, actions_jax)
            state.info["command"] = jnp.array([1.0, 0.0, 0.0])
            if i % 2 == 0:
                trajectory.append(state)
            if state.done > 0:
                break

        scene_option = mujoco.MjvOption()
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
        frames = raw_render_env.render(
            trajectory,
            camera="track",
            height=480,
            width=640,
            scene_option=scene_option,
        )
        return frames

    # ===================== Training Loop =====================
    obs_td = train_env.reset()  # TensorDict
    obs = obs_td["state"]
    if asymmetric:
        critic_obs = obs_td["privileged_state"]

    total_timesteps = 0
    start_time = time.time()

    for iteration in range(MAX_ITERATIONS):
        # ---- Rollout phase ----
        for step in range(NUM_STEPS_PER_ENV):
            obs_jax = torch_to_jax(obs)
            if asymmetric:
                critic_obs_jax = torch_to_jax(critic_obs)
            else:
                critic_obs_jax = obs_jax
            act_input = ActInput(actor_obs=obs_jax, critic_obs=critic_obs_jax)
            actions = alg.act(act_input, deterministic=False)

            # Step env
            actions_torch = jax_to_torch(actions, device)
            next_obs_td, rewards, dones, _infos = train_env.step(actions_torch.float())

            next_obs = next_obs_td["state"]
            if asymmetric:
                next_critic_obs = next_obs_td["privileged_state"]

            # Official wrapper doesn't provide terminal obs on truncation,
            # so disable truncation bootstrapping (treat all dones as termination).
            # This matches Brax PPO behavior which zeros out advantage at truncation.
            dones_bool = dones.bool()
            terminated_jax = jnp.asarray(dones_bool.cpu().numpy())
            truncated_jax = jnp.zeros_like(terminated_jax)

            rewards_jax = torch_to_jax(rewards)
            next_obs_jax = torch_to_jax(next_obs)
            if asymmetric:
                next_critic_obs_jax = torch_to_jax(next_critic_obs)
            else:
                next_critic_obs_jax = next_obs_jax

            alg.process_env_step(
                rewards=rewards_jax,
                terminated=terminated_jax,
                truncated=truncated_jax,
                infos={},
                next_actor_obs=next_obs_jax,
                next_critic_obs=next_critic_obs_jax,
            )

            obs = next_obs  # plain tensor (state)
            if asymmetric:
                critic_obs = next_critic_obs  # plain tensor (privileged_state)
            total_timesteps += NUM_ENVS

        # ---- Update phase ----
        obs_jax = torch_to_jax(obs)
        if asymmetric:
            critic_obs_jax = torch_to_jax(critic_obs)
        else:
            critic_obs_jax = obs_jax
        alg.compute_returns(last_critic_obs=critic_obs_jax)
        metrics = alg.update()

        # ---- Logging ----
        elapsed = time.time() - start_time
        fps = total_timesteps / elapsed if elapsed > 0 else 0
        m = metrics.to_wandb_dict()
        print(
            f"[iter {iteration:>5d}/{MAX_ITERATIONS}] "
            f"steps={total_timesteps:>9d} | "
            f"fps={fps:.0f} | "
            f"policy_loss={metrics.actor.policy_loss:.4f} | "
            f"value_loss={metrics.critic.value_loss:.4f} | "
            f"entropy={metrics.actor.entropy:.4f} | "
            f"kl={metrics.kl.approx_kl:.5f} | "
            f"lr={metrics.learning_rate:.1e}"
        )
        if USE_WANDB:
            log_data = {
                "speed": fps,
                "total_timesteps": total_timesteps,
                **m,
            }
            wandb.log(log_data, step=iteration)

        # ---- Evaluation + render ----
        if EVAL_INTERVAL > 0 and iteration % EVAL_INTERVAL == 0:
            eval_return, eval_length = evaluate()
            print(f"  >>> EVAL: return={eval_return:.2f}, length={eval_length:.1f}")
            if USE_WANDB:
                eval_log = {"eval/return": eval_return, "eval/length": eval_length}
                try:
                    frames = render_with_rollout()
                    video = wandb.Video(
                        np.array(frames).transpose(0, 3, 1, 2),
                        fps=30,
                        format="gif",
                    )
                    eval_log["eval/video"] = video
                except Exception as e:
                    print(f"  Render failed: {e}")
                wandb.log(eval_log, step=iteration)

    # Final eval
    eval_return, eval_length = evaluate()
    print(f"\nFinal eval: return={eval_return:.2f}, length={eval_length:.1f}")
    if USE_WANDB:
        wandb.log(
            {"eval/return": eval_return, "eval/length": eval_length},
            step=MAX_ITERATIONS,
        )
        wandb.finish()


if __name__ == "__main__":
    main()
