"""Newton simulator initializer."""

from typing import Any

import torch

from rlworld.rl.evals.sim_initializers import SimInitializer
from rlworld.rl.utils.console import print_info, print_success, print_error


class NewtonInitializer(SimInitializer):

    @property
    def video_extension(self) -> str:
        return ".bin"

    def init_device(self) -> torch.device:
        import warp as wp
        from warp.torch import device_to_torch
        return device_to_torch(wp.get_device())

    def prepare_configs(
        self,
        policy_path: str,
        extra_overrides: dict | None,
        metadata: dict,
        record_video: bool,
        video_dir: str | None,
    ) -> Any:
        from rlworld.rl.utils.checkpoint import load_config_from_checkpoint

        eval_cfgs = load_config_from_checkpoint(metadata)

        if extra_overrides is not None:
            eval_cfgs.apply_overrides(**extra_overrides)

        eval_cfgs.visualization.show_viewer = False
        eval_cfgs.visualization.record_video = record_video
        eval_cfgs.visualization.video_dir = video_dir

        return eval_cfgs

    def init_environment(self, eval_cfgs: Any, **kwargs) -> Any:
        from rlworld.rl import envs

        env_class_name = eval_cfgs.env.env_name
        env_class = getattr(envs, env_class_name)

        kw = dict(
            num_envs=eval_cfgs.env.num_envs,
            env_cfg=eval_cfgs.env,
            scene_cfg=eval_cfgs.scene,
            visualization_cfg=eval_cfgs.visualization,
            obs_cfg=eval_cfgs.observation,
            act_cfg=eval_cfgs.action,
            reward_cfg=eval_cfgs.reward,
            command_cfg=eval_cfgs.command,
            event_cfg=eval_cfgs.event,
            curriculum_cfg=eval_cfgs.curriculum,
        )
        gait_cfg = getattr(eval_cfgs, "gait", None)
        if gait_cfg is not None:
            kw["gait_cfg"] = gait_cfg
        return env_class(**kw)

    def create_play_scene(self, env: Any):
        from rlworld.rl.vis.viser.bridges import NewtonBridge
        from rlworld.rl.vis.viser.play_scene import BridgePlayScene
        return BridgePlayScene(NewtonBridge(env.scene_manager))

    def start_recording(self, env: Any) -> None:
        # Newton ViewerFile records automatically
        print_info("Newton recording active")

    def stop_recording(self, env: Any) -> None:
        env.vis_manager.stop_recording()
        print_success("Newton recording saved!")

    def cleanup(self, env: Any) -> None:
        if hasattr(env, 'vis_manager'):
            print_info("Closing Newton viewer...")
            try:
                env.vis_manager.close()
                print_success("Newton viewer closed!")
            except Exception as e:
                print_error(f"Error closing viewer: {e}")
