"""
Diagnostic: FastTD3 on MuJoCo Playground G1JoystickFlatTerrain (Fully JAX).

Same as fast_td3.py but bypasses RSLRLBraxWrapper entirely.
Uses raw JAX Brax environment directly — zero torch↔jax conversions.

This should be significantly faster than the torch-wrapper version.
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

from mujoco_playground import registry
from mujoco_playground._src.wrapper import wrap_for_brax_training

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
NUM_UPDATES = 2
BATCH_SIZE = 32768
GAMMA = 0.97
TAU = 0.1
TARGET_POLICY_NOISE = 0.001
NOISE_CLIP = 0.5
NOISE_MIN = 0.001
NOISE_MAX = 0.4
BUFFER_SIZE_PER_ENV = 1024 * 10
N_STEPS = 1
NUM_ATOMS = 101
V_MIN = -10.0
V_MAX = 10.0
ACTOR_HIDDEN = [512, 256, 128]
CRITIC_HIDDEN = [1024, 512, 256]
INIT_SCALE = 0.01
EVAL_INTERVAL = 5000
USE_WANDB = True


def _extract_obs(state_obs, asymmetric: bool):
    """Extract actor_obs and critic_obs from Brax state.obs."""
    if asymmetric:
        return state_obs["state"], state_obs["privileged_state"]
    else:
        obs = state_obs
        return obs, obs


def main():
    gpu_devices = jax.devices("gpu")
    if gpu_devices:
        jax_device = gpu_devices[DEVICE_RANK]
    else:
        jax_device = jax.devices("cpu")[0]
    print(f"JAX device: {jax_device}")

    # ===================== Wandb =====================
    if USE_WANDB:
        import wandb

        wandb.init(
            project="FastTD3-Playground-Diagnostic",
            name=f"{ENV_NAME}__jax__seed{SEED}",
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
                "backend": "fully_jax",
            },
        )

    # ===================== Training Environment (raw JAX) =====================
    train_env_cfg = registry.get_default_config(ENV_NAME)
    train_env_cfg.push_config.enable = False
    train_env_cfg.push_config.magnitude_range = [0.0, 0.0]
    raw_train_env = registry.load(ENV_NAME, config=train_env_cfg)
    max_episode_steps = train_env_cfg.episode_length

    # Wrap with VmapWrapper + EpisodeWrapper + BraxAutoResetWrapper (provides raw_obs)
    train_env = wrap_for_brax_training(
        raw_train_env,
        episode_length=train_env_cfg.episode_length,
        action_repeat=train_env_cfg.action_repeat,
    )
    train_jit_reset = jax.jit(train_env.reset)
    train_jit_step = jax.jit(train_env.step)

    # Determine obs dims from a test reset
    test_key = jax.random.PRNGKey(0)
    test_key = jax.device_put(test_key, jax_device)
    test_state = train_jit_reset(test_key)

    asymmetric = isinstance(test_state.obs, dict)
    if asymmetric:
        n_obs = test_state.obs["state"].shape[-1]
        n_critic_obs = test_state.obs["privileged_state"].shape[-1]
    else:
        n_obs = test_state.obs.shape[-1]
        n_critic_obs = n_obs
    n_act = raw_train_env.action_size

    print(f"Env: {ENV_NAME} (fully JAX)")
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
    if gpu_devices:
        eval_key = jax.device_put(eval_key, jax_device)
    eval_key_reset = jax.random.split(eval_key, NUM_EVAL_ENVS)
    # Note: eval uses raw vmap (no auto-reset wrapper) since we track done manually

    # ===================== Render Environment (1 env) =====================
    render_env_cfg = registry.get_default_config(ENV_NAME)
    render_env_cfg.push_config.enable = False
    render_env_cfg.push_config.magnitude_range = [0.0, 0.0]
    raw_render_env = registry.load(ENV_NAME, config=render_env_cfg)
    render_jit_reset = jax.jit(raw_render_env.reset)
    render_jit_step = jax.jit(raw_render_env.step)
    render_key = jax.random.PRNGKey(SEED + 200)
    if gpu_devices:
        render_key = jax.device_put(render_key, jax_device)

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

    alg.init_storage({
        "num_envs": NUM_ENVS,
        "actor_obs_shape": [n_obs],
        "critic_obs_shape": [n_critic_obs],
        "actions_shape": [n_act],
        "size_per_env": BUFFER_SIZE_PER_ENV,
        "n_steps": N_STEPS,
    })

    # ===================== Evaluate =====================
    def evaluate():
        state = eval_jit_reset(eval_key_reset)
        episode_returns = jnp.zeros(NUM_EVAL_ENVS)
        episode_lengths = jnp.zeros(NUM_EVAL_ENVS)
        done_masks = jnp.zeros(NUM_EVAL_ENVS, dtype=bool)

        for _ in range(max_episode_steps):
            actor_obs, critic_obs_eval = _extract_obs(state.obs, asymmetric)
            act_input = ActInput(actor_obs=actor_obs, critic_obs=critic_obs_eval)
            actions = alg.act(act_input, deterministic=True)
            state = eval_jit_step(state, actions)

            episode_returns = jnp.where(~done_masks, episode_returns + state.reward, episode_returns)
            episode_lengths = jnp.where(~done_masks, episode_lengths + 1, episode_lengths)
            done_masks = done_masks | (state.done > 0)
            if done_masks.all():
                break

        return float(episode_returns.mean()), float(episode_lengths.mean())

    def render_with_rollout():
        """Single-env rollout, collect states, render video."""
        state = render_jit_reset(render_key)
        state.info["command"] = jnp.array([1.0, 0.0, 0.0])
        trajectory = [state]

        for i in range(max_episode_steps):
            actor_obs, critic_obs_r = _extract_obs(state.obs, asymmetric)
            actor_obs = actor_obs[None]  # [1, obs_dim]
            critic_obs_r = critic_obs_r[None]
            act_in = ActInput(actor_obs=actor_obs, critic_obs=critic_obs_r)
            actions = alg.act(act_in, deterministic=True)
            state = render_jit_step(state, actions[0])
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
            trajectory, camera="track", height=480, width=640,
            scene_option=scene_option,
        )
        return frames

    # ===================== Training Loop (fully JAX) =====================
    train_key = jax.random.PRNGKey(SEED)
    if gpu_devices:
        train_key = jax.device_put(train_key, jax_device)
    state = train_jit_reset(train_key)

    measure_start = None
    measure_step = 0

    for global_step in range(TOTAL_TIMESTEPS):
        # Extract obs (all JAX, no conversion)
        actor_obs, critic_obs = _extract_obs(state.obs, asymmetric)
        act_input = ActInput(actor_obs=actor_obs, critic_obs=critic_obs)
        actions = alg.act(act_input, deterministic=False)

        # Step env (all JAX)
        state = train_jit_step(state, actions)

        # Extract raw_obs (pre-reset observation for replay buffer)
        raw_actor_obs, raw_critic_obs = _extract_obs(state.info["raw_obs"], asymmetric)

        # Build true_next_obs: use raw_obs for done envs, post-reset obs for others
        next_actor_obs, next_critic_obs = _extract_obs(state.obs, asymmetric)
        dones = state.done
        truncations = state.info["truncation"]

        true_next_actor = jnp.where(dones[:, None] > 0, raw_actor_obs, next_actor_obs)
        true_next_critic = jnp.where(dones[:, None] > 0, raw_critic_obs, next_critic_obs)

        terminated = (dones > 0) & ~(truncations > 0)
        truncated = truncations > 0

        # Store transition (all JAX arrays, zero copies)
        alg.store_transition(
            actor_obs=actor_obs,
            critic_obs=critic_obs,
            action=actions,
            reward=state.reward,
            next_actor_obs=true_next_actor,
            next_critic_obs=true_next_critic,
            terminated=terminated,
            truncated=truncated,
        )

        # Process env step
        alg.process_env_step(state.reward, terminated, truncated, {})

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
                    "env_reward_mean": float(state.reward.mean()),
                    **m,
                }
                print(
                    f"[{global_step:>6d}/{TOTAL_TIMESTEPS}] "
                    f"speed={speed:.0f} sps | "
                    f"env_rew={float(state.reward.mean()):.3f} | "
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
                            fps=30, format="gif",
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
