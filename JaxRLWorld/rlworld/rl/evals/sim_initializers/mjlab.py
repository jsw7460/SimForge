"""MjlabEnv (MuJoCo) simulator initializer."""

from typing import Any

import torch

from rlworld.rl.evals.sim_initializers import SimInitializer
from rlworld.rl.utils.console import print_info, print_success, print_error


class MjlabInitializer(SimInitializer):

    def init_device(self) -> torch.device:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def prepare_configs(
        self,
        policy_path: str,
        extra_overrides: dict | None,
        metadata: dict,
        record_video: bool,
        video_dir: str | None,
    ) -> Any:
        from rlworld.rl.configs.mujoco_config_classes import MujocoConfigsForRun

        eval_cfgs = MujocoConfigsForRun.from_dict(metadata['config'])

        if extra_overrides is not None:
            eval_cfgs.apply_overrides(**extra_overrides)

        eval_cfgs.visualization.show_viewer = False
        eval_cfgs.visualization.record_video = record_video
        eval_cfgs.visualization.video_dir = video_dir

        # entities field is required — scene manager builds SceneCfg internally
        if getattr(eval_cfgs.scene, "entities", None) is None:
            raise ValueError(
                "MuJoCo evaluation requires 'entities' in scene config. "
                "Re-train with unified EntityCfg to embed entity info in checkpoints."
            )

        return eval_cfgs

    def init_environment(self, eval_cfgs: Any, **kwargs) -> Any:
        from rlworld.rl.envs import MjlabEnv

        return MjlabEnv(
            num_envs=eval_cfgs.env.num_envs,
            env_cfg=eval_cfgs.env,
            scene_cfg=eval_cfgs.scene,
            visualization_cfg=eval_cfgs.visualization,
            obs_cfg=eval_cfgs.observation,
            act_cfg=eval_cfgs.action,
            reward_cfg=eval_cfgs.reward,
            command_cfg=eval_cfgs.command,
            event_cfg=eval_cfgs.event,
        )

    def cleanup(self, env: Any) -> None:
        if hasattr(env, 'visualization_manager'):
            print_info("Closing Mjlab viewer...")
            try:
                env.visualization_manager.close()
                print_success("Mjlab viewer closed!")
            except Exception as e:
                print_error(f"Error closing viewer: {e}")