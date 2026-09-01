"""TDMPC2 reproduction on dm_control ``cheetah-run``.

Single-task entry point.  Other DMC tasks reuse the same structure —
copy this file, change ``TASK_NAME`` (and any TDMPC2 hyperparameters
that differ for that task), keep everything else identical.  The
gymnasium wrapper chain and the eval-env routing both live in
:mod:`jaxrlworld.rl.envs.gymnasium`, so train and eval envs are always
built through the same factory.
"""

import os

os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"
os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"


from gymnasium.vector import AutoresetMode, SyncVectorEnv  # noqa: E402

from jaxrlworld.rl.configs.algorithms import TDMPC2Config  # noqa: E402
from jaxrlworld.rl.configs.presets.dmc import get_config  # noqa: E402
from jaxrlworld.rl.envs import GymnasiumEnv  # noqa: E402
from jaxrlworld.rl.envs.gymnasium import make_dmc_env_factory  # noqa: E402
from jaxrlworld.rl.runners import ModelBasedRunner  # noqa: E402

TASK_NAME = "dm_control/cheetah-run-v0"
ACTION_REPEAT = 2
MAX_EPISODE_STEPS = 1000


def main():
    # All TDMPC2 hyperparameters live on this dataclass instance so the
    # call site is fully self-describing — no preset lookup, no
    # ``cfgs.algorithm.<field> = ...`` dance after the fact.
    algorithm_cfg = TDMPC2Config(
        batch_size=256,
        buffer_size=1_000_000,
        learning_starts=2500,
        dropout=0.01,
        vmin=-10.0,
        vmax=10.0,
        num_bins=101,
        episode_length=500,
    )

    cfgs_for_run = get_config(
        task_name=TASK_NAME,
        algorithm_cfg=algorithm_cfg,
        action_repeat=ACTION_REPEAT,
        max_episode_steps=MAX_EPISODE_STEPS,
        num_envs=1,
        seed=42,
        run_name="HalfCheetah_TDMPC2",
    )
    cfgs_for_run.runner.log_interval = 500
    cfgs_for_run.runner.max_iterations = 1_000_000
    cfgs_for_run.runner.save_interval = 100_000
    cfgs_for_run.runner.eval_interval = 2500

    # Single factory drives BOTH the training vector env (built below)
    # and the eval env (built by ``ModelBasedRunner`` via the
    # ``runner.gym_env_factory`` hook), so the wrapper chain — action
    # repeat → flatten dict obs → seed — never drifts between them.
    env_factory = make_dmc_env_factory(
        task_name=cfgs_for_run.env.task_name,
        action_repeat=ACTION_REPEAT,
        max_episode_steps=MAX_EPISODE_STEPS,
    )

    env_gym = SyncVectorEnv(
        [lambda i=i: env_factory(i) for i in range(cfgs_for_run.env.num_envs)],
        autoreset_mode=AutoresetMode.SAME_STEP,
    )
    env = GymnasiumEnv(
        env_gym,
        env_cfg=cfgs_for_run.env,
        scene_cfg=cfgs_for_run.scene,
        obs_cfg=cfgs_for_run.observation,
        act_cfg=cfgs_for_run.action,
        reward_cfg=cfgs_for_run.reward,
        command_cfg=cfgs_for_run.command,
        seed=cfgs_for_run.env.seed,
    )

    runner = ModelBasedRunner(
        env=env,
        cfgs=cfgs_for_run,
        use_wandb=True,
        seed=cfgs_for_run.env.seed,
    )
    runner.gym_env_factory = env_factory

    runner.learn(
        num_learning_iterations=cfgs_for_run.runner.max_iterations,
        init_at_random_ep_len=cfgs_for_run.runner.init_at_random_ep_len,
    )


if __name__ == "__main__":
    main()
