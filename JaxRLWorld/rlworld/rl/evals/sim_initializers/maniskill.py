"""ManiSkill simulator initializer."""

import os
from typing import Any

import gymnasium as gym
import torch

from rlworld.rl.evals.sim_initializers import SimInitializer
from rlworld.rl.utils.console import print_info, print_success


class ManiSkillInitializer(SimInitializer):

    @property
    def supports_success_tracking(self) -> bool:
        return True

    def init_device(self) -> torch.device:
        import genesis as gs
        return gs.device

    def prepare_configs(
        self,
        policy_path: str,
        eval_env_cfgs: dict | None,
        extra_overrides: dict | None,
        metadata: dict,
        show_viewer: bool,
        record_video: bool,
        video_dir: str | None,
    ) -> Any:
        # ManiSkill uses Genesis config classes
        from rlworld.rl.configs.genesis_config_classes import GenesisConfigsForRun, EnvConfig
        from rlworld.rl.utils import compare_dicts

        train_cfgs = GenesisConfigsForRun.from_dict(metadata['config'])

        if eval_env_cfgs is not None:
            compare_dicts(eval_env_cfgs, train_cfgs.env.to_dict(), "eval_env_cfgs", "train_cfgs.env")
            eval_cfgs = train_cfgs
            eval_cfgs.env = EnvConfig.from_dict(eval_env_cfgs)
        else:
            eval_cfgs = train_cfgs

        if extra_overrides is not None:
            eval_cfgs.apply_overrides(**extra_overrides)

        eval_cfgs.visualization.show_viewer = show_viewer
        eval_cfgs.visualization.record_video = record_video
        eval_cfgs.visualization.video_dir = video_dir

        return eval_cfgs

    def init_environment(self, eval_cfgs: Any, **kwargs) -> Any:
        record_video = kwargs.get('record_video', False)
        video_dir = kwargs.get('video_dir', None)
        record_steps = kwargs.get('record_steps', 1000)
        seed = kwargs.get('seed', 42)

        from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
        from mani_skill.utils.wrappers.record import RecordEpisode
        from rlworld.rl.envs import ManiSkillEnv

        env_kwargs = eval_cfgs.env.gym_make_kwargs
        env = gym.make(eval_cfgs.env.task_name, num_envs=eval_cfgs.env.num_envs, **env_kwargs)

        if record_video and video_dir:
            video_dir_only = os.path.dirname(video_dir)
            env = RecordEpisode(
                env,
                output_dir=video_dir_only,
                save_trajectory=False,
                save_video=True,
                max_steps_per_video=record_steps,
                video_fps=30
            )

        env = ManiSkillVectorEnv(env, eval_cfgs.env.num_envs, auto_reset=True, ignore_terminations=False)
        env = ManiSkillEnv(
            env,
            env_cfg=eval_cfgs.env,
            scene_cfg=eval_cfgs.scene,
            obs_cfg=eval_cfgs.observation,
            act_cfg=eval_cfgs.action,
            reward_cfg=eval_cfgs.reward,
            command_cfg=eval_cfgs.command,
            seed=seed,
        )
        return env

    def stop_recording(self, env: Any) -> None:
        print_info("Saving ManiSkill video...")
        env.gym_env.close()
        print_success("Video saved!")

    def cleanup(self, env: Any) -> None:
        # ManiSkill RecordEpisode handles its own cleanup
        pass
