"""Reusable ``gym.Wrapper`` building blocks for Gymnasium-side tasks."""

from __future__ import annotations

import gymnasium as gym


class ActionRepeatWrapper(gym.Wrapper):
    """Repeat each action ``repeat`` times, summing rewards.

    Frame skip / action repeat is the standard DMC benchmarking knob:
    TDMPC2, PPO, SAC reproductions on dm_control all run with
    ``repeat=2`` (or ``4``) so the policy issues one action per
    ``physics_dt * repeat`` seconds while the underlying simulator
    integrates at its native rate.

    Implementation matches the canonical DeepMind / TDMPC2 reference:
    early termination breaks the inner loop so the wrapper never
    crosses an episode boundary.
    """

    def __init__(self, env: gym.Env, repeat: int = 2):
        super().__init__(env)
        if repeat < 1:
            raise ValueError(f"ActionRepeatWrapper: repeat must be ≥ 1, got {repeat}.")
        self._repeat = int(repeat)

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        obs = None
        info: dict = {}
        for _ in range(self._repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info
