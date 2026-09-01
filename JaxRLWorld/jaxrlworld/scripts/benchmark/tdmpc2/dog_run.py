"""TDMPC2 reproduction on dm_control ``dog-run``.

Sibling of ``halfcheetah.py`` / ``walker_walk.py`` / ``humanoid_walk.py``
— identical structure, only ``TASK_NAME`` and ``run_name`` differ.
TDMPC2's design principle is a *single* hyperparameter set across the
entire DMControl suite, so the ``TDMPC2Config`` is left bit-identical
to the other DMC task scripts to keep the paper-vs-our-impl
comparison fair.

``dog-run`` is the **hardest DMC locomotion task** in the TDMPC2 paper:
a 38-DOF quadruped (high-DOF, complex contact dynamics) where SAC /
PPO baselines fail entirely and TDMPC2 is the only method that
reaches non-trivial return.  Reproducing it is the strongest
demonstration of the model-based planning + value ensemble pipeline.

Cost warning (num_envs=1):

* Saturated return per paper Fig. 2-3: ~400-600 at ~3-5M env steps
* Per-step cost ~2x cheetah (38 DOF MPPI rollout)
* 1M iter: ~24-36 hours; 3M iter: ~3-5 days per seed
* 5-seed reproduction: ~2-3 weeks GPU time

The ``max_iterations`` below is left at 1M for the initial run so
the first results land in a reasonable time; bump to 3-5M for the
full paper-matching curve.
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

TASK_NAME = "dm_control/dog-run-v0"
ACTION_REPEAT = 2
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
        episode_length=500,
    )

    cfgs_for_run = get_config(
        task_name=TASK_NAME,
        algorithm_cfg=algorithm_cfg,
        action_repeat=ACTION_REPEAT,
        max_episode_steps=MAX_EPISODE_STEPS,
        num_envs=1,
        seed=42,
        run_name="DogRun_TDMPC2",
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
