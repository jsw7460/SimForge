"""ManiSkill -> JaxRLWorld adapter.

Wraps a ``ManiSkillVectorEnv`` (ManiSkill 3.x, GPU/PhysX) so it satisfies the
JaxRLWorld runner contract -- the same one ``rlworld.rl.envs.gymnasium_env.
GymnasiumEnv`` implements. ManiSkill *owns* the task: scene, robot, reward,
success / failure, termination, and (through the vector wrapper) Gymnasium-style
same-step auto-reset. This adapter only reshapes the vector env's outputs into
the ``(obs_dict, rewards, terminated, truncated, infos)`` tuple the runners
consume, keeping every tensor on the simulator's GPU device.

Correctness contract (fully exercised by
``rlworld/scripts/diag/check_maniskill_adapter.py``):

* ``infos["final_observation"]`` carries the TRUE terminal observation for done
  envs. ManiSkill's vector wrapper clones ``obs`` *before* the partial reset,
  so the tensor's done rows hold the terminal obs and its non-done rows equal
  the returned (current) obs. We surface it as ``{"actor", "critic"}`` and set
  it to ``None`` on steps where no env is done -- exactly what the runners gate
  on for truncation bootstrap.
* ``terminated`` (task success | fail) is kept distinct from ``truncated`` (time
  limit) so on/off-policy bootstrap is applied on truncation only.
* Every returned tensor stays on ``gym_env.device`` (PhysX CUDA) -- no host
  round-trips, preserving ManiSkill's GPU throughput.

The constructor signature matches the ManiSkill branch of
``rlworld.rl.runners.base_runner._create_env_from_config`` so the runner can
build it directly from a config.
"""

from __future__ import annotations

from typing import Any

import torch

from rlworld.rl.envs.world import World


class _DummySceneManager:
    """Minimal scene-manager stub satisfying the runner's tree lookup.

    The on-policy runner does ``env.scene_manager.trees.get("robot", None)`` to
    optionally wire a kinematic-tree-aware policy. State-based ManiSkill tasks
    use a flat observation + MLP, so there is no tree: an empty mapping makes
    the lookup return ``None`` without parsing any URDF/MJCF. A bare ``None``
    scene_manager would crash that lookup (``None.trees``), so the attribute
    must exist and expose ``.trees``.
    """

    def __init__(self) -> None:
        self.trees: dict[str, Any] = {}


class ManiSkillEnv(World):
    """Adapter making a vectorized ManiSkill env speak the JaxRLWorld interface."""

    sim_name: str = "ManiSkill"

    def __init__(
        self,
        gym_env,  # mani_skill.vector.wrappers.gymnasium.ManiSkillVectorEnv
        env_cfg,
        scene_cfg,
        obs_cfg,
        act_cfg,
        reward_cfg,
        command_cfg,
        seed: int = 0,
    ):
        super().__init__()

        self.gym_env = gym_env
        self.num_envs = int(gym_env.num_envs)
        # ManiSkillVectorEnv.device -> base_env.device (PhysX CUDA). This is the
        # single source of truth for placement; never init another simulator's
        # runtime just to obtain a device.
        self.device = gym_env.device

        # Stored for interface symmetry with the physics backends and for
        # checkpoint/eval reconstruction; the adapter itself dereferences none
        # of them because ManiSkill owns the MDP.
        self.env_cfg = env_cfg
        self.scene_cfg = scene_cfg
        self.obs_cfg = obs_cfg
        self.act_cfg = act_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg
        self.seed = seed

        single_action_space = gym_env.single_action_space
        single_obs_space = gym_env.single_observation_space
        # ``num_actions`` / ``action_low`` / ``action_high`` / ``reset_buf`` /
        # ``episode_length_buf`` are read-only ``@property`` on ``World`` (they
        # delegate to the act/termination managers this adapter does not have),
        # so back them with private fields and override the properties below.
        self._num_actions = int(single_action_space.shape[0])
        # obs_mode="state" yields a flat Box observation.
        self._obs_dim = int(single_obs_space.shape[0])
        self._action_low = torch.as_tensor(single_action_space.low, dtype=torch.float32, device=self.device)
        self._action_high = torch.as_tensor(single_action_space.high, dtype=torch.float32, device=self.device)

        self._current_obs: torch.Tensor | None = None
        self._reset_counter = 0
        self._reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Maintained for logging parity only; ManiSkill itself owns the
        # authoritative episode clock and decides truncation.
        self._episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self.obs_manager = self._make_obs_manager()
        self.scene_manager = _DummySceneManager()

    # ------------------------------------------------------------------ #
    # World abstract methods. ManiSkill owns physics / scene / state, so #
    # these are intentionally inert.                                     #
    # ------------------------------------------------------------------ #
    @property
    def robot(self) -> Any:
        return None

    def get_robot_data(self, entity_name: str = "robot") -> Any:
        return None

    def get_robot_state_writer(self, entity_name: str = "robot") -> Any:
        return None

    def _build_scene(self) -> None:
        pass

    def _build_sim_managers(self) -> None:
        pass

    def _step_physics(self) -> None:
        pass

    # ------------------------------------------------------------------ #
    # Read-only properties overriding World's manager-backed ones.        #
    # ------------------------------------------------------------------ #
    @property
    def num_actions(self) -> int:
        return self._num_actions

    @property
    def action_low(self):
        return self._action_low

    @property
    def action_high(self):
        return self._action_high

    @property
    def reset_buf(self) -> torch.Tensor:
        return self._reset_buf

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self._episode_length_buf

    @property
    def max_episode_length(self) -> int:
        limit = getattr(self.gym_env.spec, "max_episode_steps", None)
        return int(limit) if limit is not None else 1000

    # ------------------------------------------------------------------ #
    # Observation manager (mirrors GymnasiumEnv's minimal manager).      #
    # ------------------------------------------------------------------ #
    def _make_obs_manager(self):
        outer = self

        class _ObsManager:
            def calculate_obs_dim(self) -> dict[str, int]:
                return {"actor": outer._obs_dim, "critic": outer._obs_dim}

            def get_observation(self) -> dict[str, torch.Tensor]:
                if outer._current_obs is None:
                    outer.reset()
                obs = outer._current_obs
                return {"actor": obs, "critic": obs}

        return _ObsManager()

    def calculate_obs_dim(self) -> dict[str, int]:
        return self.obs_manager.calculate_obs_dim()

    # ------------------------------------------------------------------ #
    # Core loop.                                                         #
    # ------------------------------------------------------------------ #
    def reset(self):
        obs, _info = self.gym_env.reset(seed=self.seed + self._reset_counter)
        self._reset_counter += 1
        obs = obs.to(dtype=torch.float32)
        self._current_obs = obs
        self._reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._episode_length_buf.zero_()
        obs_dict = {"actor": obs, "critic": obs}
        info = {"rewards_per_type": {"total_reward": torch.zeros(self.num_envs, device=self.device)}}
        return obs_dict, info

    def step(self, actions: torch.Tensor):
        actions = torch.clamp(actions, self.action_low, self.action_high)
        obs, rewards, terminated, truncated, infos = self.gym_env.step(actions)

        obs = obs.to(dtype=torch.float32)
        rewards = rewards.to(dtype=torch.float32).reshape(self.num_envs)
        terminated = terminated.to(dtype=torch.bool).reshape(self.num_envs)
        truncated = truncated.to(dtype=torch.bool).reshape(self.num_envs)
        dones = terminated | truncated

        # True terminal observation for done envs. ManiSkill's vector wrapper
        # adds this key only when at least one env is done; it is a full-batch
        # tensor (done rows = pre-reset terminal obs, non-done rows = current
        # obs). Surface it verbatim wrapped as actor/critic; None otherwise so
        # the runners' ``infos.get("final_observation") is not None`` gate is
        # exact.
        final_observation = None
        raw_final = infos.get("final_observation", None)
        if raw_final is not None:
            raw_final = raw_final.to(dtype=torch.float32)
            final_observation = {"actor": raw_final, "critic": raw_final}

        # Episode-end success for logging. After the same-step auto-reset,
        # ``infos["success"]`` describes the freshly reset episode, so the
        # terminal value lives in ``infos["final_info"]``.
        success = infos.get("success", None)
        final_info = infos.get("final_info", None)
        if dones.any() and final_info is not None and "success" in final_info:
            terminal_success = final_info["success"].to(dtype=torch.bool).reshape(self.num_envs)
            if success is None:
                success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            else:
                success = success.to(dtype=torch.bool).reshape(self.num_envs).clone()
            success[dones] = terminal_success[dones]

        self._current_obs = obs
        self._reset_buf = dones
        self._episode_length_buf += 1
        self._episode_length_buf[dones] = 0

        obs_dict = {"actor": obs, "critic": obs}
        formatted_info: dict[str, Any] = {
            "final_observation": final_observation,
            "rewards_per_type": {"total_reward": rewards},
        }
        if success is not None:
            formatted_info["success"] = success

        # Bootstrap mask for the on-policy value bootstrap: bootstrap on
        # truncation (time limit) and on SUCCESS terminations (non-absorbing --
        # the agent would keep earning ManiSkill's dense reward), but NOT on
        # fail terminations (absorbing). PickCube has no fail, so terminated ==
        # success. Only set on done steps (alongside final_observation), which
        # is when the on-policy runner consumes it.
        if final_observation is not None:
            success_bool = (
                success if success is not None else torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            )
            formatted_info["bootstrap_mask"] = truncated | (terminated & success_bool)

        self._update_num_step_calls()
        return obs_dict, rewards, terminated, truncated, formatted_info
