"""Genesis simulator initializer."""

from typing import Any

import torch

from rlworld.rl.evals.sim_initializers import SimInitializer
from rlworld.rl.utils.console import print_error, print_info, print_success


class GenesisInitializer(SimInitializer):
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
        if hasattr(envs, env_class_name):
            env_class = getattr(envs, env_class_name)
            kwargs = dict(
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
                kwargs["gait_cfg"] = gait_cfg
            return env_class(**kwargs)
        raise NotImplementedError(f"Undefined env class name {env_class_name}")

    def create_play_scene(self, env: Any):
        from rlworld.rl.vis.viser.bridges import GenesisBridge
        from rlworld.rl.vis.viser.play_scene import BridgePlayScene

        scene_cfg = getattr(getattr(env, "visualization_cfg", None), "viser_scene", None)
        return BridgePlayScene(GenesisBridge(env.scene_manager), scene_config=scene_cfg)

    def try_stop_mid_episode_recording(self, env: Any, target_steps: int) -> bool:
        # Genesis writes the video file frame-by-frame, so we have to
        # call stop_recording before the episode actually ends if the
        # caller asked for a fixed number of recorded steps.
        if env.env_step_counter >= target_steps - 1:
            self.stop_recording(env)
            return True
        return False

    def start_recording(self, env: Any) -> None:
        env.vis_manager.start_recording()
        print_info("Video recording started")

    def stop_recording(self, env: Any) -> None:
        env.vis_manager.stop_recording()
        print_success("Video recording stopped")

    def cleanup(self, env: Any) -> None:
        if hasattr(env, "vis_manager"):
            print_info("Saving video before exit...")
            try:
                env.vis_manager.stop_recording()
                print_success("Video saved successfully!")
            except Exception as e:
                print_error(f"Error saving video: {e}")
