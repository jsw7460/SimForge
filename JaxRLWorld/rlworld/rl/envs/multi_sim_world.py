"""MultiSimWorld: Run multiple simulators in parallel as a single environment.

Wraps multiple World instances (e.g., Genesis, Newton, MuJoCo) into a unified
environment that presents a single interface to the Runner. Actions are split
and dispatched to each sub-environment; observations, rewards, and termination
signals are concatenated back.

Joint ordering across simulators may differ (e.g., Genesis uses URDF order
while Newton/MuJoCo use pattern order).  This class automatically detects
ordering mismatches and applies permutations so the Runner sees a single
consistent (canonical) joint order.

Usage:
    from rlworld.rl.envs.multi_sim_world import MultiSimWorld

    multi_env = MultiSimWorld([genesis_env, newton_env, mujoco_env])
    runner = OnPolicyRunner(env=multi_env, cfgs=ppo_cfg)
    runner.learn(num_learning_iterations=30000)
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Tuple, TYPE_CHECKING

import numpy as np
import torch
from gymnasium import spaces

if TYPE_CHECKING:
    from rlworld.rl.envs import World


# =====================================================================
# Joint-indexed observation function names.
# These produce tensors of shape (num_envs, num_joints) whose columns
# follow each simulator's internal joint ordering and therefore need
# permutation when orderings differ.
# =====================================================================
_JOINT_INDEXED_OBS_NAMES = frozenset({
    "dof_pos",
    "dof_vel",
    "dof_pos_nominal_difference",
    "prev_processed_actions",
    "raw_actions",
})


# =====================================================================
# _JointPermutation
# =====================================================================

class _JointPermutation:
    """Maps between a sub-environment's joint order and the canonical order.

    Canonical order is defined by the *first* environment passed to
    ``MultiSimWorld``.

    Attributes:
        is_identity:  True when no reordering is needed (fast path).
        action_perm:  Index tensor to reorder canonical actions → sim order.
        obs_perms:    Per obs-group index tensor to reorder sim obs → canonical.
    """

    def __init__(
        self,
        canonical_names: List[str],
        sim_names: List[str],
        obs_group_joint_slices: Dict[str, List[Tuple[int, int]]],
        obs_group_dims: Dict[str, int],
        device: torch.device,
    ):
        n = len(canonical_names)
        assert len(sim_names) == n, (
            f"Joint count mismatch: canonical={n}, sim={len(sim_names)}"
        )

        # Strip prefix (Newton adds "g1_29dof/" etc.)
        def _bare(name: str) -> str:
            return name.rsplit("/", 1)[-1]

        canonical_bare = [_bare(n) for n in canonical_names]
        sim_bare = [_bare(n) for n in sim_names]

        # Validate same set of joints
        if set(canonical_bare) != set(sim_bare):
            only_canonical = set(canonical_bare) - set(sim_bare)
            only_sim = set(sim_bare) - set(canonical_bare)
            raise ValueError(
                f"Joint name mismatch!\n"
                f"  Only in canonical: {only_canonical}\n"
                f"  Only in sim: {only_sim}"
            )

        # ── Build permutation indices ──
        # sim_to_canon[s] = c  means sim_names[s] == canonical_names[c]
        canonical_idx = {name: i for i, name in enumerate(canonical_bare)}
        sim_to_canon_list = [canonical_idx[b] for b in sim_bare]
        self._sim_to_canon = torch.tensor(sim_to_canon_list, device=device, dtype=torch.long)

        # canon_to_sim[c] = s  means canonical_names[c] == sim_names[s]
        sim_idx = {name: i for i, name in enumerate(sim_bare)}
        canon_to_sim_list = [sim_idx[b] for b in canonical_bare]
        self._canon_to_sim = torch.tensor(canon_to_sim_list, device=device, dtype=torch.long)

        # ── Identity check ──
        identity = torch.arange(n, device=device, dtype=torch.long)
        self.is_identity = bool(torch.equal(self._sim_to_canon, identity))

        # ── Action permutation (canonical → sim) ──
        # sim_actions[:, s] = canonical_actions[:, sim_to_canon[s]]
        # => sim_actions = canonical_actions[:, sim_to_canon]
        self.action_perm = self._sim_to_canon

        # ── Obs permutation per group (sim → canonical) ──
        self.obs_perms: Dict[str, torch.Tensor] = {}
        for group_name, joint_slices in obs_group_joint_slices.items():
            obs_dim = obs_group_dims[group_name]
            perm = torch.arange(obs_dim, device=device, dtype=torch.long)
            if not self.is_identity:
                for start, end in joint_slices:
                    # perm[start + c] = start + canon_to_sim[c]
                    for c in range(n):
                        perm[start + c] = start + self._canon_to_sim[c].item()
            self.obs_perms[group_name] = perm

    def permute_actions(self, canonical_actions: torch.Tensor) -> torch.Tensor:
        """Reorder actions from canonical joint order to this sim's order."""
        if self.is_identity:
            return canonical_actions
        return canonical_actions[:, self.action_perm]

    def permute_obs(self, sim_obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Reorder observations from this sim's joint order to canonical."""
        if self.is_identity:
            return sim_obs
        return {
            group: tensor[:, self.obs_perms[group]]
            if group in self.obs_perms else tensor
            for group, tensor in sim_obs.items()
        }


# =====================================================================
# Proxy managers
# =====================================================================

class _TerminationManagerProxy:
    """Proxy that concatenates termination state across sub-environments."""

    def __init__(self, envs: List[World], splits: List[int]):
        self._envs = envs
        self._splits = splits

    @property
    def max_episode_length(self) -> int:
        return min(e.termination_manager.max_episode_length for e in self._envs)

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return torch.cat(
            [e.termination_manager.episode_length_buf for e in self._envs], dim=0
        )

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        parts = torch.split(value, self._splits, dim=0)
        for env, part in zip(self._envs, parts):
            env.termination_manager.episode_length_buf = part

    @property
    def reset_buf(self) -> torch.Tensor:
        return torch.cat(
            [e.termination_manager.reset_buf for e in self._envs], dim=0
        )

    @property
    def episode_count(self) -> torch.Tensor:
        return torch.cat(
            [e.termination_manager.episode_count for e in self._envs], dim=0
        )

    @property
    def extras(self) -> dict:
        merged: dict = {}
        for env in self._envs:
            merged.update(env.termination_manager.extras)
        return merged


class _ObsManagerProxy:
    """Proxy that concatenates observations across sub-environments."""

    def __init__(
        self,
        envs: List[World],
        splits: List[int],
        joint_perms: List[_JointPermutation],
    ):
        self._envs = envs
        self._splits = splits
        self._joint_perms = joint_perms

    def get_observation(self) -> Dict[str, torch.Tensor]:
        obs_list = []
        for env, jp in zip(self._envs, self._joint_perms):
            obs = env.obs_manager.get_observation()
            obs_list.append(jp.permute_obs(obs))
        keys = obs_list[0].keys()
        return {k: torch.cat([o[k] for o in obs_list], dim=0) for k in keys}

    def calculate_obs_dim(self) -> Dict[str, int]:
        return self._envs[0].obs_manager.calculate_obs_dim()

    def get_robot_state(self) -> torch.Tensor | None:
        states = [e.obs_manager.get_robot_state() for e in self._envs]
        if states[0] is None:
            return None
        return torch.cat(states, dim=0)

    @property
    def extras(self) -> dict:
        merged: dict = {}
        for env in self._envs:
            merged.update(env.obs_manager.extras)
        return merged


class _ActManagerProxy:
    """Proxy that exposes action metadata from the first sub-environment."""

    def __init__(self, envs: List[World]):
        self._envs = envs
        self._primary = envs[0].act_manager

    @property
    def total_action_dim(self) -> int:
        return self._primary.total_action_dim

    @property
    def _clip_low(self) -> torch.Tensor:
        return self._primary._clip_low

    @property
    def _clip_high(self) -> torch.Tensor:
        return self._primary._clip_high

    @property
    def clip(self):
        return getattr(self._primary, "clip", None)

    @property
    def clip_actions(self):
        return getattr(self._primary, "clip_actions", None)


# =====================================================================
# MultiSimWorld
# =====================================================================

class MultiSimWorld:
    """Wraps multiple simulator environments into a single unified environment.

    From the Runner's perspective this behaves exactly like a single ``World``
    with ``sum(num_envs)`` parallel environments.  Internally it dispatches
    actions to each sub-environment and concatenates the results.

    Joint ordering is automatically aligned: the first environment's ordering
    is used as the canonical order, and permutations are applied for any
    sub-environment whose joint order differs.
    """

    sim_name: str = "MultiSim"
    sim_type: str = "multi_sim"

    def __init__(self, envs: List[World]):
        if not envs:
            raise ValueError("At least one environment is required.")

        self.envs = envs
        self.splits: List[int] = [e.num_envs for e in envs]
        self.num_envs: int = sum(self.splits)

        # ── Validate compatibility ──
        self._validate_compatibility()

        # ── Core attributes ──
        self._primary = envs[0]
        self.device: torch.device = self._primary.device
        self.seed: int = self._primary.seed
        self.physics_dt: float = self._primary.physics_dt
        self.control_dt: float = self._primary.control_dt
        self.decimation: int = getattr(self._primary, "decimation", 1)

        # ── Joint permutations ──
        self._joint_perms = self._build_joint_permutations()

        # ── Proxy managers ──
        self.obs_manager = _ObsManagerProxy(envs, self.splits, self._joint_perms)
        self.termination_manager = _TerminationManagerProxy(envs, self.splits)
        self.act_manager = _ActManagerProxy(envs)
        self.scene_manager = getattr(self._primary, "scene_manager", None)

        # ── Buffers ──
        self.rew_buf = torch.zeros(self.num_envs, device=self.device)
        self.episode_sums: defaultdict = defaultdict(
            lambda: torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        )
        self.rew_buf_per_type: defaultdict = defaultdict(
            lambda: torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        )
        self.extras: dict = {}

        # ── Pretty name ──
        sim_names = [e.sim_name for e in envs]
        env_counts = [str(e.num_envs) for e in envs]
        self.sim_name = "MultiSim(" + "+".join(
            f"{n}x{c}" for n, c in zip(sim_names, env_counts)
        ) + ")"

        self._task_name = getattr(self._primary, "task_name", "multi_sim")

        import ipdb; ipdb.set_trace()

    # ------------------------------------------------------------------
    # Joint permutation setup
    # ------------------------------------------------------------------

    def _build_joint_permutations(self) -> List[_JointPermutation]:
        """Build permutation mappings for each sub-environment.

        The canonical joint order is defined by envs[0].
        """
        canonical_names = self._get_joint_names(self.envs[0])
        num_actions = self.envs[0].num_actions

        perms = []
        for env in self.envs:
            sim_names = self._get_joint_names(env)
            joint_slices = self._find_joint_obs_slices(env, num_actions)
            obs_dims = env.obs_manager.calculate_obs_dim()

            jp = _JointPermutation(
                canonical_names=canonical_names,
                sim_names=sim_names,
                obs_group_joint_slices=joint_slices,
                obs_group_dims=obs_dims,
                device=self.device,
            )
            perms.append(jp)

        # Log permutation info
        needs_perm = [not jp.is_identity for jp in perms]
        if any(needs_perm):
            print(f"[MultiSimWorld] Joint permutation needed for: "
                  f"{[self.envs[i].sim_name for i, need in enumerate(needs_perm) if need]}")
            print(f"[MultiSimWorld] Canonical order (from {self.envs[0].sim_name}): "
                  f"{canonical_names[:5]}... ({len(canonical_names)} joints)")
        else:
            print(f"[MultiSimWorld] All sub-environments have identical joint ordering.")

        return perms

    @staticmethod
    def _get_joint_names(env: World) -> List[str]:
        """Get actuated joint names from an environment."""
        return list(env.act_manager.actuated_joint_names)

    @staticmethod
    def _find_joint_obs_slices(
        env: World, num_actions: int
    ) -> Dict[str, List[Tuple[int, int]]]:
        """Find slices in each obs group's flat vector that are joint-indexed.

        A term is considered joint-indexed if:
          1. Its function name is in _JOINT_INDEXED_OBS_NAMES, OR
          2. Its dimension equals num_actions (heuristic fallback)
             AND its function name suggests joint data.
        """
        # Ensure term indices are built
        if not env.obs_manager._is_term_indices_built:
            env.obs_manager._build_term_indices()
            env.obs_manager._is_term_indices_built = True

        result: Dict[str, List[Tuple[int, int]]] = {}

        for group_name, terms in env.obs_manager.config.obs_group.items():
            slices = []
            term_indices = env.obs_manager._group_term_indices.get(group_name, {})

            for term_idx, obs_term in enumerate(terms):
                func_name = getattr(obs_term.func, "__name__", f"term_{term_idx}")

                if func_name in _JOINT_INDEXED_OBS_NAMES:
                    if func_name in term_indices:
                        start, end = term_indices[func_name]
                        # Verify dimension matches num_actions
                        term_dim = end - start
                        if term_dim == num_actions:
                            slices.append((start, end))
                        # If history is used, dim = num_actions * history_length
                        elif term_dim % num_actions == 0:
                            history_len = term_dim // num_actions
                            for h in range(history_len):
                                slices.append((
                                    start + h * num_actions,
                                    start + (h + 1) * num_actions,
                                ))

            result[group_name] = slices

        return result

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_compatibility(self) -> None:
        """Ensure all sub-environments are compatible."""
        ref = self.envs[0]

        # Device
        devices = {str(e.device) for e in self.envs}
        if len(devices) > 1:
            raise ValueError(
                f"All sub-environments must use the same device. Got: {devices}"
            )

        # Observation dimensions
        ref_obs_dim = ref.calculate_obs_dim()
        for i, env in enumerate(self.envs[1:], 1):
            obs_dim = env.calculate_obs_dim()
            if obs_dim != ref_obs_dim:
                raise ValueError(
                    f"Obs dim mismatch: env[0]={ref_obs_dim}, env[{i}]={obs_dim}"
                )

        # Action dimensions
        ref_act = ref.num_actions
        for i, env in enumerate(self.envs[1:], 1):
            if env.num_actions != ref_act:
                raise ValueError(
                    f"Action dim mismatch: env[0]={ref_act}, env[{i}]={env.num_actions}"
                )

        # Joint names (same set, possibly different order)
        def _bare(name: str) -> str:
            return name.rsplit("/", 1)[-1]

        ref_joints = set(_bare(n) for n in ref.act_manager.actuated_joint_names)
        for i, env in enumerate(self.envs[1:], 1):
            env_joints = set(_bare(n) for n in env.act_manager.actuated_joint_names)
            if ref_joints != env_joints:
                raise ValueError(
                    f"Joint name mismatch between env[0] and env[{i}]!\n"
                    f"  Only in env[0]: {ref_joints - env_joints}\n"
                    f"  Only in env[{i}]: {env_joints - ref_joints}"
                )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def task_name(self) -> str:
        return self._task_name

    @property
    def num_actions(self) -> int:
        return self.act_manager.total_action_dim

    @property
    def max_episode_length(self) -> int:
        return self.termination_manager.max_episode_length

    @property
    def reset_buf(self) -> torch.Tensor:
        return self.termination_manager.reset_buf

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.termination_manager.episode_length_buf

    @property
    def action_low(self) -> torch.Tensor:
        return self.act_manager._clip_low

    @property
    def action_high(self) -> torch.Tensor:
        return self.act_manager._clip_high

    @property
    def action_space(self) -> spaces.Box:
        num_actions = self.act_manager.total_action_dim
        act_mgr = self.act_manager._primary
        if hasattr(act_mgr, "clip") and act_mgr.clip is not None:
            low, high = act_mgr.clip
        elif hasattr(act_mgr, "clip_actions") and act_mgr.clip_actions is not None:
            low, high = act_mgr.clip_actions
        else:
            low, high = -np.inf, np.inf
        return spaces.Box(
            low=np.float32(low), high=np.float32(high),
            shape=(num_actions,), dtype=np.float32,
        )

    @property
    def observation_space(self) -> Dict[str, spaces.Box]:
        obs_dims = self.obs_manager.calculate_obs_dim()
        return {
            name: spaces.Box(low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32)
            for name, dim in obs_dims.items()
        }

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def calculate_obs_dim(self) -> Dict[str, int]:
        return self.obs_manager.calculate_obs_dim()

    def get_observation(self) -> Dict[str, torch.Tensor]:
        return self.obs_manager.get_observation()

    def get_observation_dims(self) -> Dict[str, int]:
        return self.obs_manager.calculate_obs_dim()

    def get_robot_state(self) -> torch.Tensor | None:
        return self.obs_manager.get_robot_state()

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        split_actions = torch.split(actions, self.splits, dim=0)

        all_obs: List[Dict[str, torch.Tensor]] = []
        all_rew: List[torch.Tensor] = []
        all_term: List[torch.Tensor] = []
        all_trunc: List[torch.Tensor] = []
        all_infos: List[Dict[str, Any]] = []

        for env, act, jp in zip(self.envs, split_actions, self._joint_perms):
            # Permute actions: canonical → sim order
            sim_act = jp.permute_actions(act)
            obs, rew, terminated, truncated, info = env.step(sim_act)
            # Permute observations: sim → canonical order
            all_obs.append(jp.permute_obs(obs))
            all_rew.append(rew)
            all_term.append(terminated)
            all_trunc.append(truncated)
            all_infos.append(info)

        obs_keys = all_obs[0].keys()
        merged_obs = {
            k: torch.cat([o[k] for o in all_obs], dim=0) for k in obs_keys
        }
        merged_rew = torch.cat(all_rew, dim=0)
        merged_term = torch.cat(all_term, dim=0)
        merged_trunc = torch.cat(all_trunc, dim=0)
        merged_info = self._merge_infos(all_infos)

        return merged_obs, merged_rew, merged_term, merged_trunc, merged_info

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(
        self, *, seed=None, options=None
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        all_obs: List[Dict[str, torch.Tensor]] = []

        for env, jp in zip(self.envs, self._joint_perms):
            obs, _ = env.reset(seed=seed, options=options)
            all_obs.append(jp.permute_obs(obs))

        obs_keys = all_obs[0].keys()
        merged_obs = {
            k: torch.cat([o[k] for o in all_obs], dim=0) for k in obs_keys
        }
        merged_extras = {
            "time_outs": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            "terminal_observations": None,
            "terminal_env_ids": None,
            "rewards_per_type": self.rew_buf_per_type,
        }
        return merged_obs, merged_extras

    # ------------------------------------------------------------------
    # Info merging
    # ------------------------------------------------------------------

    def _merge_infos(self, all_infos: List[Dict[str, Any]]) -> Dict[str, Any]:
        merged: Dict[str, Any] = {}

        # rewards_per_type
        if "rewards_per_type" in all_infos[0]:
            reward_keys = set()
            for info in all_infos:
                reward_keys.update(info["rewards_per_type"].keys())
            merged_rpt: Dict[str, torch.Tensor] = {}
            for k in reward_keys:
                parts = []
                for info, n in zip(all_infos, self.splits):
                    rpt = info["rewards_per_type"]
                    parts.append(rpt[k] if k in rpt else torch.zeros(n, device=self.device))
                merged_rpt[k] = torch.cat(parts, dim=0)
            merged["rewards_per_type"] = merged_rpt

        # episode_reward_sums
        if "episode_reward_sums" in all_infos[0]:
            sum_keys = set()
            for info in all_infos:
                if info.get("episode_reward_sums") is not None:
                    sum_keys.update(info["episode_reward_sums"].keys())
            merged_sums: Dict[str, torch.Tensor] = {}
            for k in sum_keys:
                parts = []
                for info, n in zip(all_infos, self.splits):
                    sums = info.get("episode_reward_sums")
                    parts.append(sums[k] if sums is not None and k in sums
                                 else torch.zeros(n, device=self.device))
                merged_sums[k] = torch.cat(parts, dim=0)
            merged["episode_reward_sums"] = merged_sums

        # final_observation (permuted to canonical order)
        merged["final_observation"] = self._merge_final_observations(all_infos)
        merged["final_info"] = self._merge_final_info(all_infos)
        merged["terminal_env_ids"] = self._merge_terminal_env_ids(all_infos)

        return merged

    def _merge_final_observations(
        self, all_infos: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor] | None:
        has_any = any(info.get("final_observation") is not None for info in all_infos)
        if not has_any:
            return None

        obs_keys = None
        for info in all_infos:
            fo = info.get("final_observation")
            if fo is not None:
                obs_keys = list(fo.keys())
                break
        if obs_keys is None:
            return None

        merged: Dict[str, torch.Tensor] = {}
        for k in obs_keys:
            parts = []
            for info, env, jp in zip(all_infos, self.envs, self._joint_perms):
                fo = info.get("final_observation")
                if fo is not None:
                    # Permute final obs to canonical order
                    parts.append(jp.permute_obs({k: fo[k]})[k])
                else:
                    current_obs = env.obs_manager.get_observation()
                    parts.append(jp.permute_obs({k: current_obs[k]})[k])
            merged[k] = torch.cat(parts, dim=0)

        return merged

    def _merge_final_info(
        self, all_infos: List[Dict[str, Any]]
    ) -> Dict[str, Any] | None:
        has_any = any(info.get("final_info") is not None for info in all_infos)
        if not has_any:
            return None

        sum_keys: set = set()
        for info in all_infos:
            fi = info.get("final_info")
            if fi is not None and "episode_reward_sums" in fi:
                sum_keys.update(fi["episode_reward_sums"].keys())
        if not sum_keys:
            return None

        merged_sums: Dict[str, torch.Tensor] = {}
        for k in sum_keys:
            parts = []
            for info, env in zip(all_infos, self.envs):
                fi = info.get("final_info")
                parts.append(
                    fi["episode_reward_sums"][k]
                    if fi is not None and k in fi.get("episode_reward_sums", {})
                    else torch.zeros(env.num_envs, device=self.device)
                )
            merged_sums[k] = torch.cat(parts, dim=0)
        return {"episode_reward_sums": merged_sums}

    def _merge_terminal_env_ids(
        self, all_infos: List[Dict[str, Any]]
    ) -> torch.Tensor | None:
        parts = []
        offset = 0
        for info, n in zip(all_infos, self.splits):
            ids = info.get("terminal_env_ids")
            if ids is not None:
                parts.append(ids + offset)
            offset += n
        return torch.cat(parts, dim=0) if parts else None

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        lines = [f"MultiSimWorld ({self.num_envs} total environments)"]
        for i, (env, jp) in enumerate(zip(self.envs, self._joint_perms)):
            perm_str = " (permuted)" if not jp.is_identity else ""
            lines.append(f"  [{i}] {env.sim_name}: {env.num_envs} envs{perm_str}")
        return "\n".join(lines)
