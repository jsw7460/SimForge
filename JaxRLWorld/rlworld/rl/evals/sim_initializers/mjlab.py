"""MjlabEnv (Mujoco) simulator initializer with auto-resolve for mjlab_scene_cfg."""

import importlib
from typing import Any

import torch

from rlworld.rl.evals.sim_initializers import SimInitializer
from rlworld.rl.utils.console import print_info, print_success, print_error, print_warning

# Registry: preset class name → (module path, class name)
# Includes base configs and all subclasses (e.g. MLP variants)
MUJOCO_PRESET_REGISTRY: dict[str, tuple[str, str]] = {
    # G1
    "G1FlatMujocoConfig": ("rlworld.rl.configs.presets.g1_29dof.mujoco.base", "G1FlatMujocoConfig"),
    "G1MLPConfig": ("rlworld.rl.configs.presets.g1_29dof.mujoco.mlp", "G1MLPConfig"),
    # Go1
    "Go1FlatMujocoConfig": ("rlworld.rl.configs.presets.go1.mujoco.base", "Go1FlatMujocoConfig"),
    "Go1MLPConfig": ("rlworld.rl.configs.presets.go1.mujoco.mlp", "Go1MLPConfig"),
}


def auto_resolve_mjlab_configs(preset_class_name: str | None, preset_module_path: str | None = None):
    """Try to reconstruct mjlab_scene_cfg and mjlab_sim_cfg from preset info.

    Both are non-serializable mjlab objects that get lost during checkpoint save.

    Resolution order:
      1. Direct import via preset_module_path (new checkpoints)
      2. Registry fallback via MUJOCO_PRESET_REGISTRY (old checkpoints)

    Returns (mjlab_scene_cfg, mjlab_sim_cfg) or (None, None).
    """
    if not preset_class_name:
        return None, None

    # 1st: direct import via module path (new checkpoints)
    if preset_module_path:
        mod = importlib.import_module(preset_module_path)
        cls = getattr(mod, preset_class_name, None)
        if cls is not None:
            preset = cls()
            cfgs = preset.build()
            return cfgs.scene.mjlab_scene_cfg, cfgs.scene.mjlab_sim_cfg

    # 2nd: registry fallback (old checkpoints)
    if preset_class_name in MUJOCO_PRESET_REGISTRY:
        module_path, class_name = MUJOCO_PRESET_REGISTRY[preset_class_name]
        mod = importlib.import_module(module_path)
        preset = getattr(mod, class_name)()
        cfgs = preset.build()
        return cfgs.scene.mjlab_scene_cfg, cfgs.scene.mjlab_sim_cfg

    return None, None


class MjlabInitializer(SimInitializer):

    def init_device(self) -> torch.device:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def prepare_configs(
        self,
        policy_path: str,
        extra_overrides: dict | None,
        metadata: dict,
        show_viewer: bool,
        record_video: bool,
        video_dir: str | None,
    ) -> Any:
        from rlworld.rl.configs.mujoco_config_classes import MujocoConfigsForRun

        eval_cfgs = MujocoConfigsForRun.from_dict(metadata['config'])

        if extra_overrides is not None:
            eval_cfgs.apply_overrides(**extra_overrides)

        eval_cfgs.visualization.show_viewer = show_viewer
        eval_cfgs.visualization.record_video = record_video
        eval_cfgs.visualization.video_dir = video_dir

        # Auto-resolve non-serializable mjlab objects if not provided
        if eval_cfgs.scene.mjlab_scene_cfg is None:
            # New checkpoints: preset info stored in scene config
            preset_name = getattr(eval_cfgs.scene, 'preset_class_name', None)
            preset_module = getattr(eval_cfgs.scene, 'preset_module_path', None)

            # Old checkpoints fallback: preset_class_name in metadata top-level
            if not preset_name:
                preset_name = metadata.get('preset_class_name')

            scene_cfg, sim_cfg = auto_resolve_mjlab_configs(preset_name, preset_module)
            if scene_cfg is not None:
                eval_cfgs.scene.mjlab_scene_cfg = scene_cfg
                if sim_cfg is not None:
                    eval_cfgs.scene.mjlab_sim_cfg = sim_cfg
                source = "module path" if preset_module else "registry"
                print_success(
                    f"Auto-resolved mjlab configs from preset: "
                    f"{preset_name} (via {source})"
                )
            else:
                raise ValueError(
                    "mjlab_scene_cfg is required for MjlabEnv evaluation but was not found "
                    "in the checkpoint and could not be auto-resolved.\n"
                    "  - For new checkpoints: re-train to embed preset info in scene config.\n"
                    "  - For existing checkpoints: provide it via extra_overrides, e.g.:\n"
                    "      PolicyEvaluator(\n"
                    "          ...,\n"
                    "          extra_overrides={'scene': {'mjlab_scene_cfg': your_config.scene.mjlab_scene_cfg}},\n"
                    "      )"
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
