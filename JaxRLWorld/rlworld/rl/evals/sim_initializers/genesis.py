"""Genesis simulator initializer."""

from typing import Any

import torch

from rlworld.rl.evals.sim_initializers import SimInitializer
from rlworld.rl.utils import compare_dicts
from rlworld.rl.utils.console import print_info, print_success, print_error


class GenesisInitializer(SimInitializer):

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
        from rlworld.rl.configs.genesis_config_classes import GenesisConfigsForRun, EnvConfig

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
        from rlworld.rl import envs

        env_class_name = eval_cfgs.env.env_name
        if hasattr(envs, env_class_name):
            env_class = getattr(envs, env_class_name)
            return env_class(
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
        raise NotImplementedError(f"Undefined env class name {env_class_name}")

    def start_recording(self, env: Any) -> None:
        env.vis_manager.start_recording()
        print_info("Video recording started")

    def stop_recording(self, env: Any) -> None:
        env.vis_manager.stop_recording()
        print_success("Video recording stopped")

    def cleanup(self, env: Any) -> None:
        if hasattr(env, 'vis_manager'):
            print_info("Saving video before exit...")
            try:
                env.vis_manager.stop_recording()
                print_success("Video saved successfully!")
            except Exception as e:
                print_error(f"Error saving video: {e}")
