"""Gymnasium simulator initializer."""

from typing import Any

import gymnasium as gym
import torch

from rlworld.rl.evals.sim_initializers import SimInitializer


class GymnasiumInitializer(SimInitializer):

    def init_device(self) -> torch.device:
        import genesis as gs
        return gs.device

    def prepare_configs(
        self,
        policy_path: str,
        extra_overrides: dict | None,
        metadata: dict,
        record_video: bool,
        video_dir: str | None,
    ) -> Any:
        from rlworld.rl.configs.genesis_config_classes import GenesisConfigsForRun

        eval_cfgs = GenesisConfigsForRun.from_dict(metadata['config'])

        if extra_overrides is not None:
            eval_cfgs.apply_overrides(**extra_overrides)

        eval_cfgs.visualization.show_viewer = False
        eval_cfgs.visualization.record_video = record_video
        eval_cfgs.visualization.video_dir = video_dir

        return eval_cfgs

    def init_environment(self, eval_cfgs: Any, **kwargs) -> Any:
        from gymnasium.vector import SyncVectorEnv
        from rlworld.rl.envs import GymnasiumEnv

        seed = kwargs.get('seed', 42)

        def make_env(env_seed):
            def _init():
                return gym.make(eval_cfgs.env.task_name, max_episode_steps=100)
            return _init

        num_envs = eval_cfgs.env.num_envs
        env_gym = SyncVectorEnv([make_env(i) for i in range(num_envs)])
        return GymnasiumEnv(
            env_gym,
            env_cfg=eval_cfgs.env,
            scene_cfg=eval_cfgs.scene,
            obs_cfg=eval_cfgs.observation,
            act_cfg=eval_cfgs.action,
            reward_cfg=eval_cfgs.reward,
            command_cfg=eval_cfgs.command,
            seed=seed,
        )
