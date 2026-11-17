"""Newton simulator initializer."""

from typing import Any

import torch
import warp as wp

from jaxrlworld.rl import envs
from jaxrlworld.rl.evals.sim_initializers import SimInitializer
from jaxrlworld.rl.utils.checkpoint import load_config_from_checkpoint
from jaxrlworld.rl.utils.console import print_info, print_success
from jaxrlworld.rl.vis.viser.bridges.newton_bridge import NewtonBridge
from jaxrlworld.rl.vis.viser.play_scene import BridgePlayScene


class NewtonInitializer(SimInitializer):
    @property
    def video_extension(self) -> str:
        return ".bin"

    def init_device(self) -> torch.device:
        return wp.device_to_torch(wp.get_device())

    def prepare_configs(
        self,
        policy_path: str,
        extra_overrides: dict | None,
        metadata: dict,
        record_video: bool,
        video_dir: str | None,
    ) -> Any:
        eval_cfgs = load_config_from_checkpoint(metadata)

        if extra_overrides is not None:
            eval_cfgs.apply_overrides(**extra_overrides)

        eval_cfgs.visualization.show_viewer = False
        eval_cfgs.visualization.record_video = record_video
        eval_cfgs.visualization.video_dir = video_dir

        return eval_cfgs

    def init_environment(self, eval_cfgs: Any, **kwargs) -> Any:
        env_class_name = eval_cfgs.env.env_name
        env_class = getattr(envs, env_class_name)  # dynamic-attr: env class named by config

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
        gait_cfg = eval_cfgs.gait
        if gait_cfg is not None:
            kw["gait_cfg"] = gait_cfg
        return env_class(**kw)

    def create_play_scene(self, env: Any):
        scene_cfg = env.visualization_cfg.viser_scene
        return BridgePlayScene(NewtonBridge(env.scene_manager), scene_config=scene_cfg)

    def start_recording(self, env: Any) -> None:
        # Newton ViewerFile records automatically
        print_info("Newton recording active")

    def stop_recording(self, env: Any) -> None:
        env.vis_manager.stop_recording()
        print_success("Newton recording saved!")

    def cleanup(self, env: Any) -> None:
        if env.vis_manager is not None:
            print_info("Closing Newton viewer...")
            env.vis_manager.close()
            print_success("Newton viewer closed!")
