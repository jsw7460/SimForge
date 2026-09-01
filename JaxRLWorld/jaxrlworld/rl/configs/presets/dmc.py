"""One-shot config builder for dm_control suite tasks (via shimmy).

Single entry point :func:`get_config` returns a fully populated
:class:`GymnasiumConfigsForRun` for any task name accepted by
``gym.make`` (e.g. ``"dm_control/cheetah-run-v0"``).  The caller passes
an *algorithm config instance* — ``TDMPC2Config(...)`` / ``SACConfig(...)``
/ ``PPOConfig(...)`` — so per-task hyperparameter overrides stay
explicit at the call site.

Bench scripts that target several DMC tasks (TDMPC2 reproduction,
SAC sweep, ...) should call ``get_config`` per task with the same
algorithm dataclass, optionally tweaking the few fields that differ
between tasks::

    from jaxrlworld.rl.configs.algorithms import TDMPC2Config
    from jaxrlworld.rl.configs.presets.dmc import get_config

    cfgs = get_config(
        task_name="dm_control/cheetah-run-v0",
        algorithm_cfg=TDMPC2Config(batch_size=256, ...),
        action_repeat=2,
        max_episode_steps=1000,
        num_envs=1,
        seed=42,
        run_name="cheetah_run_tdmpc2",
    )

The wrapper chain (action repeat + ``FlattenObservation`` + seeding)
is *not* baked into the config — it lives in
``runner.gym_env_factory`` (see
``jaxrlworld.rl.envs.gymnasium.make_dmc_env_factory``).  Both training
and eval envs share that single factory, so eval metrics stay
comparable to training.
"""

from __future__ import annotations

from jaxrlworld.rl.configs.base_config import BaseConfig
from jaxrlworld.rl.configs.gymnasium_config_classes import (
    GymnasiumConfigsForRun,
    GymnasiumEnvConfig,
)

_DEFAULT_RUN_NAME_PREFIX = "DMC"


def _default_run_name(task_name: str) -> str:
    """Derive a sensible wandb run name from the task id.

    ``"dm_control/cheetah-run-v0"`` → ``"DMC_cheetah_run"``.
    """
    slug = task_name.split("/")[-1]
    # Strip the ``-v<N>`` suffix gymnasium env ids carry.
    if "-v" in slug:
        slug = slug.rsplit("-v", 1)[0]
    slug = slug.replace("-", "_")
    return f"{_DEFAULT_RUN_NAME_PREFIX}_{slug}"


def get_config(
    task_name: str,
    algorithm_cfg: BaseConfig,
    action_repeat: int = 2,
    max_episode_steps: int = 1000,
    num_envs: int = 1,
    seed: int = 42,
    run_name: str | None = None,
) -> GymnasiumConfigsForRun:
    """Build a complete :class:`GymnasiumConfigsForRun` for one DMC task.

    Args:
        task_name: gymnasium env id, e.g. ``"dm_control/cheetah-run-v0"``.
            Must be loadable via ``gym.make(task_name)`` — the dm_control
            namespace requires ``shimmy[dm-control]`` installed.
        algorithm_cfg: instantiated algorithm config (TDMPC2Config,
            SACConfig, ...).  All algorithm-side hyperparameters
            (``batch_size``, ``buffer_size``, ``learning_starts``,
            ``vmin`` / ``vmax`` / ``num_bins`` for TDMPC2, ...) belong
            here so the call site is fully self-describing.
        action_repeat: forwarded to the env factory; affects how many
            simulator substeps one policy decision spans.  DMC
            convention is 2.  Recorded on the env cfg so it shows up
            in serialised configs / wandb.
        max_episode_steps: forwarded to the env factory's ``gym.make``.
            DMC convention is 1000.
        num_envs: number of parallel envs in the ``SyncVectorEnv``.
            ``1`` is the standard DMC/TDMPC2 benchmark setting.
        seed: env seed; also propagated to the runner's seed.
        run_name: wandb / log run name.  ``None`` derives from
            ``task_name`` — e.g. ``"DMC_cheetah_run"``.

    Returns:
        ``GymnasiumConfigsForRun`` with all sim-side cfgs at their
        empty defaults and ``env`` / ``runner`` / ``algorithm``
        populated from the args.  The caller pairs this with
        ``make_dmc_env_factory(task_name, action_repeat,
        max_episode_steps)`` for the env, and assigns the same factory
        to ``runner.gym_env_factory`` so eval uses the identical chain.
    """
    cfgs = GymnasiumConfigsForRun()

    cfgs.env = GymnasiumEnvConfig(
        env_name="GymnasiumEnv",
        task_name=task_name,
        num_envs=num_envs,
        seed=seed,
        episode_length_s=float(max_episode_steps),
        # Stored for traceability / serialisation; the actual factory
        # reads these via its own kwargs in the bench script.
        gym_make_kwargs={
            "max_episode_steps": max_episode_steps,
            "action_repeat": action_repeat,
        },
    )

    cfgs.runner.run_name = run_name if run_name is not None else _default_run_name(task_name)
    cfgs.algorithm = algorithm_cfg

    return cfgs
