from typing import Optional, Any

import mujoco
import torch

import numpy as np

import genesis as gs
from rlworld.rl.configs import EnvConfig, SceneConfig, ObservationConfig, ActionConfig, RewardConfig, CommandConfig
from rlworld.rl.envs import World
from rlworld.rl.envs.managers.genesis.scene import KinematicTree


class MuJoCoKinematicTree:
    """
    Extract kinematic tree from MuJoCo model.
    Compatible with KinematicTree interface.
    Excludes world body (body 0) to match URDF structure.
    """

    def __init__(self, mj_model):
        """
        Args:
            mj_model: mujoco.MjModel object
        """
        self.model = mj_model
        self.name = mj_model.names.decode('utf-8').split('\x00')[0] if mj_model.names else "mujoco_robot"

        # Extract robot bodies (excluding world)
        self.links = self._extract_links()
        self.parent_indices = self._extract_parent_indices()
        self.children_indices = self._build_children_indices()
        self.joints = self._extract_joints()

        self._bottom_up_order = None
        self._depth_cache = None

        self.traverse_bottom_up()

    def _extract_links(self):
        """
        Extract link (body) information, EXCLUDING world body (body 0).

        Returns:
            List of link dictionaries with index, name, mass
        """
        links = []

        # Start from body 1 (skip world body 0)
        for mj_body_idx in range(1, self.model.nbody):
            # Get body name
            body_name_start = self.model.name_bodyadr[mj_body_idx]
            body_name_end = self.model.names[body_name_start:].find(b'\x00')
            body_name = self.model.names[body_name_start:body_name_start + body_name_end].decode('utf-8')

            # Get body mass
            mass = self.model.body_mass[mj_body_idx]

            # Get body position (relative to parent)
            pos = self.model.body_pos[mj_body_idx].copy()

            # Get body inertia
            inertia = self.model.body_inertia[mj_body_idx].copy()

            # New index starts from 0 (shifted by -1 from MuJoCo index)
            links.append({
                "index": len(links),  # Our index (0-based)
                "mujoco_index": mj_body_idx,  # Original MuJoCo index
                "name": body_name,
                "mass": float(mass),
                "pos": pos,
                "inertia": inertia
            })

        return links

    def _extract_parent_indices(self):
        """
        Extract parent indices, remapped to exclude world body.

        Returns:
            List where parent_indices[i] = parent of body i in our indexing
        """
        parent_indices = []

        # Build mapping: MuJoCo body index → Our body index
        mj_to_our_idx = {}
        for link in self.links:
            mj_to_our_idx[link['mujoco_index']] = link['index']

        for link in self.links:
            mj_body_idx = link['mujoco_index']
            mj_parent_idx = self.model.body_parentid[mj_body_idx]

            if mj_parent_idx == 0:
                # Parent is world → this is root in our tree
                parent_indices.append(-1)
            else:
                # Parent is another body → remap to our indexing
                parent_indices.append(mj_to_our_idx[mj_parent_idx])

        return parent_indices

    def _build_children_indices(self):
        """Build children list for each body"""
        children = [[] for _ in range(len(self.parent_indices))]

        for child_idx, parent_idx in enumerate(self.parent_indices):
            if parent_idx != -1:
                children[parent_idx].append(child_idx)

        return children

    def _extract_joints(self):
        """
        Extract joint information from MuJoCo model.
        Only include joints between robot bodies (not world).

        Returns:
            List of joint dictionaries
        """
        if self.model.njnt == 0:
            return []

        # Build mapping: MuJoCo body index → Our body index
        mj_to_our_idx = {}
        for link in self.links:
            mj_to_our_idx[link['mujoco_index']] = link['index']

        joints = []

        for jnt_idx in range(self.model.njnt):
            mj_child_body = self.model.jnt_bodyid[jnt_idx]

            # Skip joints connected to world body
            if mj_child_body == 0:
                continue

            # Skip if child is not in our robot bodies
            if mj_child_body not in mj_to_our_idx:
                continue

            mj_parent_body = self.model.body_parentid[mj_child_body]

            # Get our indices
            child_idx = mj_to_our_idx[mj_child_body]

            # Parent might be world (body 0) → this is root joint
            if mj_parent_body == 0:
                parent_idx = -1  # Root parent (will be filtered later if needed)
            elif mj_parent_body in mj_to_our_idx:
                parent_idx = mj_to_our_idx[mj_parent_body]
            else:
                continue  # Skip if parent not in robot

            # Get joint name
            joint_name_start = self.model.name_jntadr[jnt_idx]
            joint_name_end = self.model.names[joint_name_start:].find(b'\x00')
            joint_name = self.model.names[joint_name_start:joint_name_start + joint_name_end].decode('utf-8')

            # Get joint type
            joint_type = self.model.jnt_type[jnt_idx]
            joint_type_names = {
                0: 'free',
                1: 'ball',
                2: 'slide',
                3: 'hinge'
            }
            joint_type_str = joint_type_names.get(joint_type, f'unknown({joint_type})')

            # Get joint range (limits)
            joint_range = None
            if self.model.jnt_limited[jnt_idx]:
                joint_range = self.model.jnt_range[jnt_idx].copy()

            # Get joint axis
            joint_axis = self.model.jnt_axis[jnt_idx].copy()

            # Store joint info (only if parent != -1, i.e., not root joint)
            if parent_idx != -1:
                joints.append({
                    "index": len(joints),
                    "parent_link": int(parent_idx),
                    "child_link": int(child_idx),
                    "name": joint_name,
                    "type": joint_type_str,
                    "type_id": int(joint_type),
                    "axis": joint_axis,
                    "range": joint_range,
                    "limited": bool(self.model.jnt_limited[jnt_idx]),
                    "mujoco_index": jnt_idx
                })

        return joints

    @property
    def num_bodies(self) -> int:
        return len(self.links)

    @property
    def root_idx(self) -> int:
        for i, parent in enumerate(self.parent_indices):
            if parent == -1:
                return i
        return 0

    @property
    def num_joints(self) -> int:
        return len(self.joints)

    @property
    def max_joint_idx(self) -> int:
        return len(self.joints) - 1 if self.joints else -1

    def get_joint_parent_link(self, joint_idx: int) -> int:
        if joint_idx >= len(self.joints):
            raise ValueError(f"No joint at index {joint_idx}")
        return self.joints[joint_idx]['parent_link']

    def get_joint_child_link(self, joint_idx: int) -> int:
        if joint_idx >= len(self.joints):
            raise ValueError(f"No joint at index {joint_idx}")
        return self.joints[joint_idx]['child_link']

    def get_active_joint_indices(self):
        return list(range(len(self.joints)))

    def get_parent(self, body_idx: int) -> int:
        return self.parent_indices[body_idx]

    def get_children(self, body_idx: int):
        return self.children_indices[body_idx]

    def traverse_bottom_up(self):
        """Get body indices in bottom-up order (leaves → root)"""
        if self._bottom_up_order is not None:
            return self._bottom_up_order

        visited = [False] * self.num_bodies
        order = []

        def dfs_post_order(node):
            if visited[node]:
                return
            visited[node] = True
            for child in self.children_indices[node]:
                dfs_post_order(child)
            order.append(node)

        dfs_post_order(self.root_idx)
        self._bottom_up_order = order
        return order

    def get_ancestor_chain(self, body_idx: int):
        """Get ancestor chain from root to body_idx (inclusive)"""
        chain = []
        current = body_idx

        while current != -1:
            chain.append(current)
            current = self.parent_indices[current]

        return list(reversed(chain))

    def get_depth(self, body_idx: int) -> int:
        """Get depth level of body in the tree (root = 0)"""
        if self._depth_cache is None:
            self._compute_depths()
        return self._depth_cache[body_idx]

    def _compute_depths(self):
        """Compute and cache depth for all bodies"""
        self._depth_cache = [0] * self.num_bodies

        for body_idx in range(self.num_bodies):
            depth = 0
            current = body_idx
            while self.parent_indices[current] != -1:
                depth += 1
                current = self.parent_indices[current]
            self._depth_cache[body_idx] = depth

    def get_bodies_at_depth(self, depth: int):
        """Get all body indices at a specific depth level"""
        if self._depth_cache is None:
            self._compute_depths()
        return [i for i in range(self.num_bodies) if self._depth_cache[i] == depth]

    def get_max_depth(self) -> int:
        """Get maximum depth of the tree"""
        if self._depth_cache is None:
            self._compute_depths()
        return max(self._depth_cache)

    def get_depth_groups(self) -> dict[int, list[int]]:
        """
        Group body indices by depth level for parallel processing.

        Returns:
            Dictionary mapping depth → list of body indices at that depth
        """
        if self._depth_cache is None:
            self._compute_depths()

        depth_groups = {}
        for body_idx in range(self.num_bodies):
            depth = self._depth_cache[body_idx]
            if depth not in depth_groups:
                depth_groups[depth] = []
            depth_groups[depth].append(body_idx)

        return depth_groups

    def get_adjacency_matrix(self) -> torch.Tensor:
        """
        Get adjacency matrix for the kinematic tree.

        Returns:
            adjacency_matrix: (num_bodies, num_bodies) binary tensor
        """
        num_bodies = self.num_bodies
        adjacency = torch.zeros(num_bodies, num_bodies, dtype=torch.float32)

        # Mark parent-child connections as edges
        for child_idx, parent_idx in enumerate(self.parent_indices):
            if parent_idx != -1:
                # Undirected edge: both directions
                adjacency[parent_idx, child_idx] = 1.0
                adjacency[child_idx, parent_idx] = 1.0

        return adjacency

    def print_tree_structure(self):
        """Print tree structure for debugging"""
        print(f"\n{'=' * 60}")
        print(f"Kinematic Tree: {self.name}")
        print(f"{'=' * 60}")
        print(f"Total bodies: {self.num_bodies}")
        print(f"Total joints: {self.num_joints}")
        print(f"Root: {self.root_idx} ({self.links[self.root_idx]['name']})")
        print(f"\nBodies:")
        for link in self.links:
            parent_idx = self.parent_indices[link['index']]
            parent_name = self.links[parent_idx]['name'] if parent_idx != -1 else 'None'
            print(f"  [{link['index']}] {link['name']}: mass={link['mass']:.3f}kg, parent={parent_name}")

        print(f"\nJoints:")
        for joint in self.joints:
            print(f"  [{joint['index']}] {joint['name']}: "
                  f"type={joint['type']}, "
                  f"parent={self.links[joint['parent_link']]['name']} → "
                  f"child={self.links[joint['child_link']]['name']}")

    def __repr__(self) -> str:
        return (f"MuJoCoKinematicTree(name='{self.name}', "
                f"num_bodies={self.num_bodies}, "
                f"num_joints={self.num_joints}, "
                f"root={self.root_idx})")


class GymnasiumEnv(World):
    """Wrapper to make vectorized Gymnasium envs compatible with RLEnv interface"""
    sim_name = "Gymnasium"

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
        seed: int = 0
    ):
        from rlworld.rl.utils import set_seed
        set_seed(seed)
        super().__init__()
        # Check if vectorized or not
        gym_env.action_space.seed(seed)
        gym_env.observation_space.seed(seed)
        self.gym_env = gym_env
        self.num_envs = gym_env.num_envs
        self.is_vectorized = True

        self.env_cfg = env_cfg
        self.scene_cfg = scene_cfg
        self.obs_cfg = obs_cfg
        self.act_cfg = act_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.mj_model: Optional[mujoco.MjModel] = None
        self.device = gs.device

        # Required attributes
        self.seed = seed
        # self.max_episode_length = max_episode_length

        # Action/observation spaces
        # if self.is_vectorized:
        # self.num_actions = self.gym_env.single_action_space.shape[0]
        self._obs_dim = self.gym_env.single_observation_space.shape[0]
        # self.action_low = torch.from_numpy(self.gym_env.single_action_space.low).to(self.device)
        # self.action_high = torch.from_numpy(self.gym_env.single_action_space.high).to(self.device)

        # Episode tracking
        self._episode_length_buf = torch.zeros(self.num_envs, dtype=torch.int, device=self.device)
        self._reset_buf = torch.ones(self.num_envs, device=self.device)

        # Cache for observations
        self._current_obs = None

        # Create minimal obs_manager
        self.obs_manager = self._create_obs_manager()
        self.scene_manager = self._create_scene_manager_with_kinematic_tree(self.gym_env)
        self._initial_seed_set = False
        self._reset_counter = 0

        # self.mj_model.opt.timestep = 0.001

    @property
    def reset_buf(self) -> torch.Tensor:
        return self._reset_buf

    @property
    def max_episode_length(self) -> int:
        return 1000

    @property
    def action_low(self):
        return torch.from_numpy(self.gym_env.single_action_space.low).to(self.device)

    @property
    def action_high(self):
        return torch.from_numpy(self.gym_env.single_action_space.high).to(self.device)

    @property
    def num_actions(self) -> int:
        return self.gym_env.single_action_space.shape[0]

    def _setup_environment(self) -> None:
        pass

    def _step_physics(self) -> None:
        pass

    def robot(self) -> Any:
        pass

    def robot_data(self):
        return None

    def get_robot_data(self, entity_name: str = "robot"):
        return None

    def heading_w(self) -> torch.Tensor:
        pass

    def _create_scene_manager_with_kinematic_tree(self, gym_env):
        """Extract kinematic tree from MuJoCo model"""

        class DummySceneManager:
            def __init__(self, kinematic_tree):
                self.trees = {"robot": kinematic_tree}

        if hasattr(gym_env, 'envs'):
            first_env = gym_env.envs[0]
        else:
            first_env = gym_env

        base_env = first_env.unwrapped

        if hasattr(base_env, 'physics'):
            # dm_control env (via shimmy)
            mj_model = base_env.physics.model._model
            kinematic_tree = MuJoCoKinematicTree(mj_model)
        elif hasattr(base_env, 'model'):
            # Standard MuJoCo gym env
            mj_model = base_env.model
            kinematic_tree = KinematicTree(mjcf_path=base_env.fullpath)
        else:
            raise ValueError("Cannot extract MuJoCo model from environment.")

        self.mj_model = mj_model
        print(f"✓ Extracted kinematic tree: {kinematic_tree}")
        return DummySceneManager(kinematic_tree)

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
        # if self.is_vectorized:
        # Vectorized env
        obs, rewards, terminated, truncated, info = self.gym_env.step(actions_np)

        obs_tensor = torch.from_numpy(obs).float().to(self.device)
        rewards_tensor = torch.from_numpy(rewards).float().to(self.device)
        terminated_tensor = torch.from_numpy(terminated).to(self.device)
        truncated_tensor = torch.from_numpy(truncated).to(self.device)

        # Compute dones
        dones_tensor = terminated_tensor | truncated_tensor
        self._reset_buf = dones_tensor

        final_observation = None
        if dones_tensor.any():
            final_obs_arr = info["final_obs"]  # object array, None for non-done envs

            # Start from reset obs (current obs_tensor)
            final_obs_tensor = obs_tensor.clone()

            # Only overwrite done envs
            done_indices = dones_tensor.nonzero(as_tuple=True)[0].cpu().numpy()
            for i in done_indices:
                final_obs_tensor[i] = torch.from_numpy(
                    final_obs_arr[i].astype(np.float32)
                ).to(self.device)

            final_observation = {
                "actor": final_obs_tensor,
                "critic": final_obs_tensor,
            }

        # Update cache
        self._current_obs = obs_tensor

        # Update episode length
        self._episode_length_buf += 1
        self._episode_length_buf[dones_tensor] = 0

        # Format observations
        obs_dict = {
            "actor": obs_tensor.clone(),
            "critic": obs_tensor.clone(),
        }

        formatted_info = {}
        # Merge original info
        if isinstance(info, dict):
            formatted_info.update(info)

        # Format info
        formatted_info.update({
            "final_observation": final_observation,
            "rewards_per_type": {"total_reward": rewards_tensor},
        })

        self._update_num_step_calls()
        return obs_dict, rewards_tensor, terminated_tensor, truncated_tensor, formatted_info

    def reset(self):
        """Reset all environments

        Returns:
            Tuple of (obs_dict, info)
        """
        obs, info = self.gym_env.reset(seed=self.seed + self._reset_counter)
        self._reset_counter += 1
        obs_tensor = torch.from_numpy(obs).float().to(self.device)

        # Update cache
        self._current_obs = obs_tensor

        # Reset episode lengths
        self._episode_length_buf.zero_()

        # Format observations
        obs_dict = {
            "actor": obs_tensor,
            "critic": obs_tensor,
        }

        formatted_info = {
            "rewards_per_type": {"total": torch.zeros(self.num_envs, device=self.device)},
        }

        return obs_dict, formatted_info
