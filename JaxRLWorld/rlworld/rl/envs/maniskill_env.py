import genesis as gs
import mujoco
import numpy as np
import torch

from rlworld.rl.configs import ActionConfig, CommandConfig, EnvConfig, ObservationConfig, RewardConfig, SceneConfig
from rlworld.rl.configs.robots.kinematic_tree import KinematicTree
from rlworld.rl.envs import World


class ManiSkillEnv(World):
    """Wrapper to make vectorized Maniskill envs compatible with RLEnv interface"""

    sim_name: str = "ManiSkill"

    def __init__(
        self,
        gym_env,  # gym.Env
        env_cfg: EnvConfig,
        scene_cfg: SceneConfig,
        obs_cfg: ObservationConfig,
        act_cfg: ActionConfig,
        reward_cfg: RewardConfig,
        command_cfg: CommandConfig,
        max_episode_length: int = 1000,
        seed: int = 0,
    ):
        super().__init__()
        # Check if vectorized or not
        self.gym_env = gym_env
        self.num_envs = gym_env.num_envs

        self.env_cfg = env_cfg
        self.scene_cfg = scene_cfg
        self.obs_cfg = obs_cfg
        self.act_cfg = act_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.is_vectorized = True

        self.mj_model: mujoco.MjModel | None = None
        self.device = gs.device

        # Required attributes
        self.seed = seed
        self.max_episode_length = max_episode_length

        # Action/observation spaces
        self.num_actions = self.gym_env.unwrapped.single_action_space.shape[0]
        self._obs_dim = self.gym_env.unwrapped.single_observation_space.shape[0]
        self.action_low = torch.from_numpy(self.gym_env.single_action_space.low).to(self.device)
        self.action_high = torch.from_numpy(self.gym_env.single_action_space.high).to(self.device)

        # Cache for observations
        self._current_obs = None

        # Create minimal obs_manager
        self.obs_manager = self._create_obs_manager()
        self.scene_manager = self._create_scene_manager_with_kinematic_tree(self.gym_env)
        self._initial_seed_set = False
        self._reset_counter = 0

    @property
    def robot(self):
        return None

    @property
    def robot_data(self):
        return None

    def get_robot_data(self, entity_name: str = "robot"):
        return None

    def _build_scene(self):
        pass

    def _build_sim_managers(self):
        pass

    def _step_physics(self):
        pass

    def _create_scene_manager_with_kinematic_tree(self, gym_env):
        """Extract kinematic tree from backend simulator"""

        class DummySceneManager:
            def __init__(self, kinematic_tree):
                self.trees = {"robot": kinematic_tree}

        # Unwrap to base environment
        if hasattr(gym_env, "envs"):
            first_env = gym_env.envs[0]
        else:
            first_env = gym_env

        base_env = first_env
        while hasattr(base_env, "env"):
            base_env = base_env.env

        kinematic_tree = self._extract_maniskill_kinematic_tree(base_env._env)
        print(f"Extracted ManiSkill kinematic tree: {kinematic_tree}")
        return DummySceneManager(kinematic_tree)

    def _extract_maniskill_kinematic_tree(self, base_env):
        """Extract kinematic tree from ManiSkill environment"""
        if not hasattr(base_env, "agent"):
            raise ValueError("ManiSkill environment has no agent")

        agent = base_env.agent

        if hasattr(agent, "robot"):
            # Try URDF first, then MJCF
            if hasattr(agent, "urdf_path") and agent.urdf_path is not None:
                return KinematicTree(urdf_path=agent.urdf_path)
            elif hasattr(agent, "mjcf_path") and agent.mjcf_path is not None:
                return KinematicTree(mjcf_path=agent.mjcf_path)
            else:
                raise ValueError("Agent has neither urdf_path nor mjcf_path")

        raise ValueError("Cannot extract kinematic tree from ManiSkill agent")

    def _create_obs_manager(self):
        """Create minimal obs manager for interface compatibility"""

        class DummyObsManager:
            def __init__(self, wrapper):
                self.wrapper = wrapper

            def calculate_obs_dim(self) -> dict[str, int]:
                """Return observation dimensions for actor and critic"""
                obs_dim = self.wrapper._obs_dim
                return {
                    "actor": obs_dim,
                    "critic": obs_dim,  # Same as actor for standard gym
                    "robot_state": 0,  # Not used in gym envs
                    "estimator": 0,  # Not used in gym envs
                }

            def get_observation(self) -> dict[str, torch.Tensor]:
                """Return current observation in required format"""
                if self.wrapper._current_obs is None:
                    # Initial call before first step
                    obs, _ = self.wrapper.gym_env.reset()
                    self.wrapper._current_obs = torch.from_numpy(obs).float().to(self.wrapper.device)

                obs_tensor = self.wrapper._current_obs
                return {
                    "actor": obs_tensor,
                    "critic": obs_tensor,  # Share same obs
                }

            def get_robot_state(self):
                """Not applicable for gym envs"""
                return None

        return DummyObsManager(self)

    def get_robot_state(self):
        return None

    def step(self, actions: torch.Tensor):
        """Execute actions in the environment

        Args:
            actions: (num_envs, num_actions) tensor

        Returns:
            Tuple of (obs_dict, privileged_obs, rewards, dones, info)
        """
        # Action clipping
        actions = torch.clamp(actions, self.action_low, self.action_high)

        # Convert to numpy
        actions_np = actions.cpu().numpy()

        obs, rewards, terminated, truncated, info = self.gym_env.step(actions_np)

        # Handle both numpy and torch tensor
        if isinstance(obs, torch.Tensor):
            obs_tensor = obs.float().to(self.device)
        else:
            obs_tensor = torch.from_numpy(obs).float().to(self.device)

        rewards_tensor = (
            torch.from_numpy(rewards).float().to(self.device)
            if isinstance(rewards, np.ndarray)
            else rewards.float().to(self.device)
        )
        terminated_tensor = (
            torch.from_numpy(terminated).to(self.device)
            if isinstance(terminated, np.ndarray)
            else terminated.to(self.device)
        )
        truncated_tensor = (
            torch.from_numpy(truncated).to(self.device)
            if isinstance(truncated, np.ndarray)
            else truncated.to(self.device)
        )

        # Compute dones
        dones_tensor = terminated_tensor | truncated_tensor

        # Final observation for bootstrap (Gymnasium style)
        final_observation = None
        if dones_tensor.any() and "final_observation" in info:
            final_obs = info["final_observation"]

            if isinstance(final_obs, np.ndarray):
                final_obs_tensor = torch.from_numpy(final_obs).float().to(self.device)
            else:
                final_obs_tensor = final_obs.float().to(self.device)

            final_observation = {
                "actor": final_obs_tensor,
                "critic": final_obs_tensor,
            }

        # Update cache
        self._current_obs = obs_tensor

        # Format observations
        obs_dict = {
            "actor": obs_tensor.clone(),
            "critic": obs_tensor.clone(),
        }

        privileged_obs = None

        # Format info
        formatted_info = {}

        # Merge original info (exclude final_observation to avoid confusion)
        if isinstance(info, dict):
            for k, v in info.items():
                if k not in ["final_observation"]:
                    formatted_info[k] = v

        formatted_info.update(
            {
                "final_observation": final_observation,  # (num_envs, obs_dim), valid for done envs
                "rewards_per_type": {"total_reward": rewards_tensor},
            }
        )

        if dones_tensor.any():
            if "success" in info["final_info"]:
                formatted_info["success"] = info["final_info"]["success"]

        self._update_num_step_calls()
        return obs_dict, rewards, terminated, truncated, info

    def reset(self):
        """Reset all environments

        Returns:
            Tuple of (obs_dict, info)
        """
        obs, info = self.gym_env.reset(seed=self.seed + self._reset_counter)
        self._reset_counter += 1

        # Handle both numpy and torch tensor
        if isinstance(obs, torch.Tensor):
            obs_tensor = obs.float().to(self.device)
        else:
            obs_tensor = torch.from_numpy(obs).float().to(self.device)

        # Update cache
        self._current_obs = obs_tensor

        # Format observations
        obs_dict = {
            "actor": obs_tensor,
            "critic": obs_tensor,
        }

        formatted_info = {
            "rewards_per_type": {"total": torch.zeros(self.num_envs, device=self.device)},
        }

        return obs_dict, formatted_info
