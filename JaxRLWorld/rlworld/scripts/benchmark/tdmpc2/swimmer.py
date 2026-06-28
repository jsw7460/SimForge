"""TDMPC2 on Gymnasium ``Swimmer-v5``.

Mirrors ``tdmpc2/halfcheetah.py`` but targets the real gymnasium Swimmer-v5
task (not dm_control). ``make_dmc_env_factory`` is generic — it just wraps
``gym.make(task_name) -> ActionRepeat -> FlattenObservation -> seed`` — so it
drives Swimmer-v5 too (FlattenObservation is a no-op on its already-flat obs).
The same factory builds both the training vector env and the eval env (via
``runner.gym_env_factory``), so their wrapper chains never drift.

``ACTION_REPEAT=1`` -> effective episode length 1000, so ``episode_length`` on
the algorithm config is set to 1000. Try ACTION_REPEAT=4 (and episode_length
250) if learning stalls — a common Swimmer trick.

Run (GPU box):
    python rlworld/scripts/benchmark/tdmpc2/swimmer.py
"""

import os

os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"
os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"


from gymnasium.vector import AutoresetMode, SyncVectorEnv  # noqa: E402

from rlworld.rl.configs.algorithms import TDMPC2Config  # noqa: E402
from rlworld.rl.configs.presets.dmc import get_config  # noqa: E402
from rlworld.rl.envs import GymnasiumEnv  # noqa: E402
from rlworld.rl.envs.gymnasium import make_dmc_env_factory  # noqa: E402
from rlworld.rl.runners import ModelBasedRunner  # noqa: E402

TASK_NAME = "Swimmer-v5"
ACTION_REPEAT = 1
MAX_EPISODE_STEPS = 1000


def main():
    algorithm_cfg = TDMPC2Config(
        batch_size=256,
        buffer_size=1_000_000,
        learning_starts=2500,
        dropout=0.01,
        vmin=-10.0,
        vmax=10.0,
        num_bins=101,
        episode_length=MAX_EPISODE_STEPS // ACTION_REPEAT,
    )

    cfgs_for_run = get_config(
        task_name=TASK_NAME,
        algorithm_cfg=algorithm_cfg,
        action_repeat=ACTION_REPEAT,
        max_episode_steps=MAX_EPISODE_STEPS,
        num_envs=1,
        seed=42,
        run_name="Swimmer_TDMPC2",
    )
    cfgs_for_run.runner.log_interval = 500
    cfgs_for_run.runner.max_iterations = 1_000_000
    cfgs_for_run.runner.save_interval = 100_000
    cfgs_for_run.runner.eval_interval = 2500

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
