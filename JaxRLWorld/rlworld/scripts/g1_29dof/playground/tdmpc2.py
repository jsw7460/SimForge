"""
Diagnostic: TDMPC2 on MuJoCo Playground G1JoystickFlatTerrain.

Uses our TDMPC2 algorithm directly with the Playground environment,
bypassing ModelBasedRunner. This isolates whether the issue is in the
algorithm or in the reward/obs of Newton/Genesis/MuJoCo environments.

Evaluation uses MPPI planning (not direct policy output).
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

from mujoco_playground import registry
from mujoco_playground import wrapper_torch

from rlworld.rl.algorithms.tdmpc2 import TDMPC2
from rlworld.rl.modules.policies.tdmpc2_world_model import TDMPC2WorldModel


# ===================== Config =====================

ENV_NAME = "G1JoystickFlatTerrain"
NUM_ENVS = 1024
NUM_EVAL_ENVS = 1024
SEED = 1
DEVICE_RANK = 0
TOTAL_TIMESTEPS = 100_000
LEARNING_STARTS = 1024       # Random-action warmup steps
BATCH_SIZE = 4096
EPISODE_LENGTH = 1000

# Discount
GAMMA = 0.99
DISCOUNT_MIN = 0.95
DISCOUNT_MAX = 0.995
DISCOUNT_DENOM = 5.0

# Target network
TAU = 0.01

# Learning rates
LR = 3e-4
PI_LR = 3e-4

# Planning (MPPI)
MPC = True
HORIZON = 3
NUM_SAMPLES = 512
NUM_PI_TRAJS = 24
NUM_ELITES = 64
NUM_ITERATIONS = 6
TEMPERATURE = 0.5
MIN_STD = 0.05
MAX_STD = 2.0

# Loss coefficients
CONSISTENCY_COEF = 20.0
REWARD_COEF = 0.1
VALUE_COEF = 0.1
ENTROPY_COEF = 1e-4
RHO = 0.5

# World model architecture
LATENT_DIM = 512
MLP_DIM = 512
NUM_ENC_LAYERS = 2
NUM_Q = 5
NUM_BINS = 101
V_MIN = -10.0
V_MAX = 10.0
SIMNORM_DIM = 8
DROPOUT = 0.01

# Episodic termination
EPISODIC = True
TERMINATION_COEF = 5.0

# Training
NUM_GRADIENT_STEPS = 1
BUFFER_SIZE_PER_ENV = 1024 * 10
GRAD_CLIP_NORM = 20.0

# Eval / logging
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
            project="TDMPC2-Playground-Diagnostic",
            name=f"{ENV_NAME}__tdmpc2__seed{SEED}",
            config={
                "env_name": ENV_NAME,
                "num_envs": NUM_ENVS,
                "batch_size": BATCH_SIZE,
                "gamma": GAMMA,
                "tau": TAU,
                "lr": LR,
                "pi_lr": PI_LR,
                "mpc": MPC,
                "horizon": HORIZON,
                "num_samples": NUM_SAMPLES,
                "num_elites": NUM_ELITES,
                "num_iterations": NUM_ITERATIONS,
                "latent_dim": LATENT_DIM,
                "num_q": NUM_Q,
                "num_bins": NUM_BINS,
                "episodic": EPISODIC,
                "learning_starts": LEARNING_STARTS,
                "buffer_size_per_env": BUFFER_SIZE_PER_ENV,
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

    print(f"Env: {ENV_NAME}")
    print(f"  obs_dim={n_obs}, act_dim={n_act}")
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

    world_model = TDMPC2WorldModel(
        obs_dim=n_obs,
        action_dim=n_act,
        latent_dim=LATENT_DIM,
        mlp_dim=MLP_DIM,
        num_enc_layers=NUM_ENC_LAYERS,
        num_q=NUM_Q,
        num_bins=NUM_BINS,
        simnorm_dim=SIMNORM_DIM,
        dropout=DROPOUT,
        squash_action=True,
        obs_normalization=True,
        episodic=EPISODIC,
        key=model_key,
    )

    key, alg_key = jax.random.split(key)
    alg = TDMPC2(
        world_model=world_model,
        num_envs=NUM_ENVS,
        gamma=GAMMA,
        episode_length=EPISODE_LENGTH,
        discount_min=DISCOUNT_MIN,
        discount_max=DISCOUNT_MAX,
        discount_denom=DISCOUNT_DENOM,
        lr=LR,
        pi_lr=PI_LR,
        tau=TAU,
        mpc=MPC,
        horizon=HORIZON,
        num_samples=NUM_SAMPLES,
        num_pi_trajs=NUM_PI_TRAJS,
        num_elites=NUM_ELITES,
        num_iterations=NUM_ITERATIONS,
        temperature=TEMPERATURE,
        min_std=MIN_STD,
        max_std=MAX_STD,
        consistency_coef=CONSISTENCY_COEF,
        reward_coef=REWARD_COEF,
        value_coef=VALUE_COEF,
        entropy_coef=ENTROPY_COEF,
        rho=RHO,
        num_bins=NUM_BINS,
        vmin=V_MIN,
        vmax=V_MAX,
        batch_size=BATCH_SIZE,
        grad_clip_norm=GRAD_CLIP_NORM,
        episodic=EPISODIC,
        termination_coef=TERMINATION_COEF,
        key=alg_key,
    )

    alg.init_storage({
        "num_envs": NUM_ENVS,
        "obs_dim": n_obs,
        "action_dim": n_act,
        "size_per_env": BUFFER_SIZE_PER_ENV,
    })

    # ===================== Action Scaling =====================
    # squash_action=True: world model outputs in [-1, 1], scale to env range
    action_low = torch.full((n_act,), -1.0, device=device)
    action_high = torch.full((n_act,), 1.0, device=device)
    # RSLRLBraxWrapper expects actions in env's native range
    # With squash_action=True, model outputs [-1, 1] which we scale
    env_action_low = train_env.clip_actions_low if hasattr(train_env, "clip_actions_low") else -1.0
    env_action_high = train_env.clip_actions_high if hasattr(train_env, "clip_actions_high") else 1.0

    def scale_action(actions_jax: jax.Array) -> jax.Array:
        """Scale actions from [-1, 1] to env range."""
        low = jnp.array(env_action_low) if not isinstance(env_action_low, float) else env_action_low
        high = jnp.array(env_action_high) if not isinstance(env_action_high, float) else env_action_high
        return low + (actions_jax + 1.0) * 0.5 * (high - low)

    # ===================== Evaluate (MPPI) =====================
    def evaluate():
        """Evaluate using MPPI planning, not direct policy output."""
        state = eval_jit_reset(eval_key_reset)
        episode_returns = jnp.zeros(NUM_EVAL_ENVS)
        episode_lengths = jnp.zeros(NUM_EVAL_ENVS)
        done_masks = jnp.zeros(NUM_EVAL_ENVS, dtype=bool)

        # Resize MPPI prev_mean for eval envs
        _, horizon, action_dim = alg._prev_mean.shape
        orig_prev_mean = alg._prev_mean
        alg._prev_mean = np.zeros((NUM_EVAL_ENVS, horizon, action_dim), dtype=np.float32)
        t0_mask = np.ones(NUM_EVAL_ENVS, dtype=bool)

        for _ in range(max_episode_steps):
            obs_jax = jnp.asarray(state.obs)

            # MPPI planning
            actions = alg.act_with_t0(obs_jax, t0_mask=t0_mask, eval_mode=True)

            # Scale and step
            actions_scaled = scale_action(actions)
            actions_for_env = wrapper_torch._torch_to_jax(jax_to_torch(actions_scaled, device))
            state = eval_jit_step(state, actions_for_env)

            rewards = state.reward
            dones = state.done

            # Reset MPPI warm-start for done envs
            t0_mask = np.array(dones > 0)

            episode_returns = jnp.where(~done_masks, episode_returns + rewards, episode_returns)
            episode_lengths = jnp.where(~done_masks, episode_lengths + 1, episode_lengths)
            done_masks = done_masks | (dones > 0)
            if done_masks.all():
                break

        # Restore training prev_mean
        alg._prev_mean = orig_prev_mean

        return float(episode_returns.mean()), float(episode_lengths.mean())

    def render_with_rollout():
        """Single-env MPPI rollout for video rendering."""
        state = render_jit_reset(render_key)
        state.info["command"] = jnp.array([1.0, 0.0, 0.0])
        trajectory = [state]

        # Single-env MPPI: resize prev_mean
        _, horizon, action_dim = alg._prev_mean.shape
        orig_prev_mean = alg._prev_mean
        alg._prev_mean = np.zeros((1, horizon, action_dim), dtype=np.float32)
        t0_mask = np.ones(1, dtype=bool)

        for i in range(max_episode_steps):
            obs_jax = jnp.asarray(state.obs)[None]  # [1, obs_dim]
            actions = alg.act_with_t0(obs_jax, t0_mask=t0_mask, eval_mode=True)
            actions_scaled = scale_action(actions)
            action_single = jnp.asarray(actions_scaled)[0]  # [act_dim]
            state = render_jit_step(state, action_single)
            state.info["command"] = jnp.array([1.0, 0.0, 0.0])
            t0_mask = np.array([False])
            if i % 2 == 0:
                trajectory.append(state)
            if state.done > 0:
                break

        alg._prev_mean = orig_prev_mean

        scene_option = mujoco.MjvOption()
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
        frames = raw_render_env.render(
            trajectory, camera="track", height=480, width=640,
            scene_option=scene_option,
        )
        return frames

    # ===================== Training Loop =====================
    obs = train_env.reset()  # torch tensor
    t0_mask = np.ones(NUM_ENVS, dtype=bool)  # All envs start fresh
    start_time = time.time()
    measure_start = None
    measure_step = 0
    pretrained = False

    for global_step in range(TOTAL_TIMESTEPS):
        # Warmup: random actions in [-1, 1] (squash_action=True)
        if global_step < LEARNING_STARTS:
            key, subkey = jax.random.split(key)
            actions = jax.random.uniform(
                subkey,
                shape=(NUM_ENVS, n_act),
                minval=-1.0,
                maxval=1.0,
            )
        else:
            # MPPI planning
            obs_jax = torch_to_jax(obs)
            actions = alg.act_with_t0(obs_jax, t0_mask=t0_mask, eval_mode=False)

        # Scale actions and step env
        actions_scaled = scale_action(actions)
        actions_torch = jax_to_torch(actions_scaled, device)
        next_obs, rewards, dones, infos = train_env.step(actions_torch.float())
        truncations = infos["time_outs"]

        # Pre-reset obs for done envs (terminal observation for buffer)
        true_next_obs = torch.where(
            dones[:, None] > 0,
            infos["observations"]["raw"]["obs"],
            next_obs,
        )

        # Convert to JAX
        obs_jax = torch_to_jax(obs) if global_step < LEARNING_STARTS else obs_jax
        next_obs_jax = torch_to_jax(true_next_obs)
        rewards_jax = torch_to_jax(rewards)
        dones_bool = dones.bool()
        trunc_bool = truncations.bool()
        terminated_jax = jnp.asarray((dones_bool & ~trunc_bool).cpu().numpy())
        truncated_jax = jnp.asarray(trunc_bool.cpu().numpy())

        # Store transition
        alg.store_transition(
            obs=obs_jax,
            action=actions,  # Store pre-scaled actions (model space, [-1, 1])
            reward=rewards_jax,
            next_obs=next_obs_jax,
            terminated=terminated_jax,
            truncated=truncated_jax,
        )

        # Update t0_mask: done envs get MPPI warm-start reset
        t0_mask = dones.cpu().numpy().astype(bool)

        # Use auto-reset obs for next step (NOT true_next_obs)
        obs = next_obs

        # Training
        min_buffer_size = max(LEARNING_STARTS, BATCH_SIZE)
        if alg.replay_buffer.size >= min_buffer_size:
            if measure_start is None:
                measure_start = time.time()
                measure_step = global_step

            # Pretrain on seed data (matches original TDMPC2 author)
            if not pretrained:
                num_updates = LEARNING_STARTS // NUM_ENVS
                print(f"Pretraining on seed data ({num_updates} updates)...")
                pretrained = True
            else:
                num_updates = NUM_GRADIENT_STEPS

            for i in range(num_updates):
                key, subkey = jax.random.split(key)
                batch = alg.sample_batch(BATCH_SIZE, subkey)
                is_last = (i == num_updates - 1)
                metrics = alg.update(batch, build_metrics=is_last)

            # Logging
            if global_step % 100 == 0 and metrics is not None:
                elapsed = time.time() - measure_start if measure_start else 1
                speed = (global_step - measure_step) / elapsed if elapsed > 0 else 0
                m = metrics.to_wandb_dict()
                log_data = {
                    "speed": speed,
                    "env_reward_mean": float(rewards.mean()),
                    "buffer_size": alg.replay_buffer.size,
                    **m,
                }
                print(
                    f"[{global_step:>6d}/{TOTAL_TIMESTEPS}] "
                    f"speed={speed:.0f} sps | "
                    f"env_rew={float(rewards.mean()):.3f} | "
                    f"consistency={m.get('world_model/consistency_loss', 0):.4f} | "
                    f"reward_loss={m.get('world_model/reward_loss', 0):.4f} | "
                    f"pi_loss={m.get('policy/pi_loss', 0):.4f}"
                )
                if USE_WANDB:
                    wandb.log(log_data, step=global_step)

            # Evaluation + render
            if EVAL_INTERVAL > 0 and global_step % EVAL_INTERVAL == 0:
                eval_return, eval_length = evaluate()
                print(f"  >>> EVAL (MPPI): return={eval_return:.2f}, length={eval_length:.1f}")
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
    print(f"\nFinal eval (MPPI): return={eval_return:.2f}, length={eval_length:.1f}")
    if USE_WANDB:
        wandb.log(
            {"eval/return": eval_return, "eval/length": eval_length},
            step=TOTAL_TIMESTEPS,
        )
        wandb.finish()


if __name__ == "__main__":
    main()