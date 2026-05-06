"""
Diagnostic: FastTD3 on MuJoCo Playground G1JoystickFlatTerrain.

Uses our FastTD3 algorithm directly with the Playground environment,
bypassing OffPolicyRunner. This isolates whether the issue is in the
algorithm or in the reward/obs of Newton/Genesis/MuJoCo environments.

Hyperparameters match the original author's G1JoystickFlatTerrain config exactly.
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
import torch
from mujoco_playground import registry, wrapper_torch

from rlworld.rl.algorithms.base import ActInput
from rlworld.rl.algorithms.fast_td3.fast_td3 import FastTD3
from rlworld.rl.modules.policies.fast_td3_ac import FastTD3ActorCritic

# ===================== Config (matches original G1JoystickFlatTerrain exactly) =====================

ENV_NAME = "G1JoystickFlatTerrain"
NUM_ENVS = 1024
NUM_EVAL_ENVS = 1024
SEED = 1
DEVICE_RANK = 0
TOTAL_TIMESTEPS = 100_000
LEARNING_STARTS = 10
NUM_UPDATES = 2  # Original: num_updates=2
BATCH_SIZE = 32768
GAMMA = 0.97
TAU = 0.1
TARGET_POLICY_NOISE = 0.001
NOISE_CLIP = 0.5
NOISE_MIN = 0.001
NOISE_MAX = 0.4
BUFFER_SIZE_PER_ENV = 1024 * 10  # Original: 10240
N_STEPS = 1
NUM_ATOMS = 101
V_MIN = -10.0
V_MAX = 10.0
ACTOR_HIDDEN = [512, 256, 128]  # Original: actor_hidden_dim=512 -> [512, 256, 128]
CRITIC_HIDDEN = [1024, 512, 256]  # Original: critic_hidden_dim=1024 -> [1024, 512, 256]
INIT_SCALE = 0.01
EVAL_INTERVAL = 5000
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
            project="FastTD3-Playground-Diagnostic",
            name=f"{ENV_NAME}__ours__seed{SEED}",
            config={
                "env_name": ENV_NAME,
                "num_envs": NUM_ENVS,
                "batch_size": BATCH_SIZE,
                "gamma": GAMMA,
                "tau": TAU,
                "target_policy_noise": TARGET_POLICY_NOISE,
                "noise_min": NOISE_MIN,
                "noise_max": NOISE_MAX,
                "buffer_size_per_env": BUFFER_SIZE_PER_ENV,
                "n_steps": N_STEPS,
                "num_updates": NUM_UPDATES,
                "num_atoms": NUM_ATOMS,
                "v_min": V_MIN,
                "v_max": V_MAX,
                "use_target_actor": False,
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

    actor_critic = FastTD3ActorCritic(
        num_actor_obs=n_obs,
        num_critic_obs=n_critic_obs,
        num_actions=n_act,
        num_atoms=NUM_ATOMS,
        v_min=V_MIN,
        v_max=V_MAX,
        is_squashed=True,
        obs_normalization=True,
        key=model_key,
        actor_kwargs={
            "hidden_dims": ACTOR_HIDDEN,
            "ortho_init": True,
            "activation": "relu",
            "output_gain": INIT_SCALE,
        },
        critic_kwargs={
            "hidden_dims": CRITIC_HIDDEN,
            "ortho_init": True,
            "activation": "relu",
            "output_gain": INIT_SCALE,
        },
    )

    alg = FastTD3(
        actor_critic=actor_critic,
        num_envs=NUM_ENVS,
        gamma=GAMMA,
        tau=TAU,
        batch_size=BATCH_SIZE,
        policy_delay=2,
        target_policy_noise=TARGET_POLICY_NOISE,
        target_noise_clip=NOISE_CLIP,
        noise_min=NOISE_MIN,
        noise_max=NOISE_MAX,
        use_cdq=True,
        use_target_actor=False,
        key=key,
    )

    alg.init_storage(
        {
            "num_envs": NUM_ENVS,
            "actor_obs_shape": [n_obs],
            "critic_obs_shape": [n_critic_obs],
            "actions_shape": [n_act],
            "size_per_env": BUFFER_SIZE_PER_ENV,
            "n_steps": N_STEPS,
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
            actions_jax = wrapper_torch._torch_to_jax(jax_to_torch(actions, device))
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
                actor_obs = jnp.asarray(state.obs["state"])[None]  # [1, obs_dim]
                critic_obs_r = jnp.asarray(state.obs["privileged_state"])[None]
            else:
                actor_obs = jnp.asarray(state.obs)[None]
                critic_obs_r = actor_obs
            act_in = ActInput(actor_obs=actor_obs, critic_obs=critic_obs_r)
            actions = alg.act(act_in, deterministic=True)
            actions_jax = jnp.asarray(actions)[0]  # [act_dim]
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
    obs = train_env.reset()  # torch tensor
    if asymmetric:
        # First step: get critic_obs from a dummy step or use obs as placeholder.
        # RSLRLBraxWrapper returns critic_obs in infos on step(), not on reset().
        # We'll initialize critic_obs after the first step; for step 0, use zeros.
        critic_obs = torch.zeros(NUM_ENVS, n_critic_obs, device=device)
    start_time = time.time()
    measure_start = None
    measure_step = 0

    for global_step in range(TOTAL_TIMESTEPS):
        # Always use policy (matching original — no random warmup)
        obs_jax = torch_to_jax(obs)
        if asymmetric:
            critic_obs_jax = torch_to_jax(critic_obs)
        else:
            critic_obs_jax = obs_jax
        act_input = ActInput(actor_obs=obs_jax, critic_obs=critic_obs_jax)
        actions = alg.act(act_input, deterministic=False)

        # Step env
        actions_torch = jax_to_torch(actions, device)
        next_obs, rewards, dones, infos = train_env.step(actions_torch.float())
        truncations = infos["time_outs"]

        if asymmetric:
            next_critic_obs = infos["observations"]["critic"]

        # Pre-reset obs for ALL done environments (matching original exactly)
        true_next_obs = torch.where(
            dones[:, None] > 0,
            infos["observations"]["raw"]["obs"],
            next_obs,
        )
        if asymmetric:
            true_next_critic_obs = torch.where(
                dones[:, None] > 0,
                infos["observations"]["raw"]["critic_obs"],
                next_critic_obs,
            )

        # Convert
        next_obs_jax = torch_to_jax(true_next_obs)
        if asymmetric:
            next_critic_obs_jax = torch_to_jax(true_next_critic_obs)
        else:
            next_critic_obs_jax = next_obs_jax
        rewards_jax = torch_to_jax(rewards)
        dones_bool = dones.bool()
        trunc_bool = truncations.bool()
        terminated_jax = jnp.asarray((dones_bool & ~trunc_bool).cpu().numpy())
        truncated_jax = jnp.asarray(trunc_bool.cpu().numpy())

        # Store transition
        alg.store_transition(
            actor_obs=obs_jax,
            critic_obs=critic_obs_jax,
            action=actions,
            reward=rewards_jax,
            next_actor_obs=next_obs_jax,
            next_critic_obs=next_critic_obs_jax,
            terminated=terminated_jax,
            truncated=truncated_jax,
        )

        # Process env step (normalizer update + noise resample)
        alg.process_env_step(rewards_jax, terminated_jax, truncated_jax, {})

        # Use auto-reset obs for next step (NOT true_next_obs)
        obs = next_obs
        if asymmetric:
            critic_obs = next_critic_obs

        # Training
        if global_step >= LEARNING_STARTS:
            if measure_start is None:
                measure_start = time.time()
                measure_step = global_step

            for _ in range(NUM_UPDATES):
                key, subkey = jax.random.split(key)
                batch = alg.sample_batch(BATCH_SIZE, subkey)
                metrics = alg.update(batch)

            # Logging
            if global_step % 100 == 0:
                elapsed = time.time() - measure_start if measure_start else 1
                speed = (global_step - measure_step) / elapsed if elapsed > 0 else 0
                m = metrics.to_wandb_dict()
                log_data = {
                    "speed": speed,
                    "env_reward_mean": float(rewards.mean()),
                    **m,
                }
                print(
                    f"[{global_step:>6d}/{TOTAL_TIMESTEPS}] "
                    f"speed={speed:.0f} sps | "
                    f"env_rew={float(rewards.mean()):.3f} | "
                    f"critic_loss={m.get('critic/loss', 0):.4f} | "
                    f"q1={m.get('critic/q1_mean', 0):.2f}"
                )
                if USE_WANDB:
                    wandb.log(log_data, step=global_step)

            # Evaluation + render
            if EVAL_INTERVAL > 0 and global_step % EVAL_INTERVAL == 0:
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
                    wandb.log(eval_log, step=global_step)

    # Final eval
    eval_return, eval_length = evaluate()
    print(f"\nFinal eval: return={eval_return:.2f}, length={eval_length:.1f}")
    if USE_WANDB:
        wandb.log(
            {"eval/return": eval_return, "eval/length": eval_length},
            step=TOTAL_TIMESTEPS,
        )
        wandb.finish()


if __name__ == "__main__":
    main()
