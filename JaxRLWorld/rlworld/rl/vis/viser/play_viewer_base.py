"""Base class for interactive play viewers with real-time pacing.

Ported from Mjlab's BaseViewer, adapted for JaxRLWorld's World/PolicyWrapper
interface. See Mjlab/src/mjlab/viewer/base.py for the original design and
budget accumulator documentation.

Budget Accumulator
==================

A single variable, ``_sim_budget``, tracks how much sim time has
accumulated but not yet been simulated. Each tick of the main loop:

  1. Measure real time elapsed since the last tick.
  2. Multiply by the speed setting and add to the budget.
  3. Call ``env.step()`` in a loop, subtracting ``control_dt`` from the
     budget each time, until the budget is less than one step.
  4. Carry the leftover to the next tick.

If physics is too slow to keep up, the budget grows without bound. A
real time deadline (one frame period) caps each burst so the renderer
always gets a turn.
"""

from __future__ import annotations

import signal
import time
import traceback
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import torch

if TYPE_CHECKING:
    from rlworld.rl.envs.world import World
    from rlworld.rl.evals.policy_wrappers import PolicyWrapper


@dataclass(frozen=True)
class ViewerStatus:
    paused: bool
    step_count: int
    speed_multiplier: float
    speed_label: str
    target_realtime: float
    actual_realtime: float
    smoothed_fps: float
    capped: bool
    last_error: str | None


class ViewerAction(Enum):
    RESET = "reset"
    TOGGLE_PAUSE = "toggle_pause"
    SINGLE_STEP = "single_step"
    RESET_SPEED = "reset_speed"
    SPEED_UP = "speed_up"
    SPEED_DOWN = "speed_down"
    PREV_ENV = "prev_env"
    NEXT_ENV = "next_env"
    SET_MOTION_CLIP = "set_motion_clip"   # payload: int (clip index)
    SET_MOTION_LOCK = "set_motion_lock"   # payload: bool (locked)


class PlayViewerBase(ABC):
    """Abstract base class for interactive play viewers.

    Subclasses implement setup(), sync_env_to_viewer(), sync_viewer_to_env(),
    close(), and is_running().
    """

    SPEED_MULTIPLIERS = [1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0, 2.0, 4.0, 8.0]

    def __init__(
        self,
        env: World,
        policy: PolicyWrapper,
        frame_rate: float = 60.0,
    ):
        self.env = env
        self.policy = policy
        self.frame_rate = frame_rate
        self.frame_time = 1.0 / frame_rate

        # State.
        self._is_paused = True
        self._step_count = 0
        self._last_error: str | None = None

        # Speed.
        self._speed_index = self.SPEED_MULTIPLIERS.index(1.0)
        self._time_multiplier = self.SPEED_MULTIPLIERS[self._speed_index]

        # Physics accumulator and render timer.
        self._sim_budget = 0.0
        self._time_until_next_render = 0.0
        self._last_tick_time = 0.0
        self._was_capped = False

        # Windowed stats, updated every 0.5s.
        self._stats_frames = 0
        self._stats_steps = 0
        self._stats_last_time = 0.0
        self._fps = 0.0
        self._sps = 0.0

        # Action queue, drained on main thread each tick.
        self._actions: deque[tuple[ViewerAction, Optional[Any]]] = deque()

    # Abstract hooks.

    @abstractmethod
    def setup(self) -> None: ...
    @abstractmethod
    def sync_env_to_viewer(self) -> None: ...
    @abstractmethod
    def sync_viewer_to_env(self) -> None: ...
    @abstractmethod
    def close(self) -> None: ...
    @abstractmethod
    def is_running(self) -> bool: ...

    # Thread-safe action requests.

    def request_reset(self) -> None:
        self._actions.append((ViewerAction.RESET, None))

    def request_toggle_pause(self) -> None:
        self._actions.append((ViewerAction.TOGGLE_PAUSE, None))

    def request_single_step(self) -> None:
        self._actions.append((ViewerAction.SINGLE_STEP, None))

    def request_speed_up(self) -> None:
        self._actions.append((ViewerAction.SPEED_UP, None))

    def request_speed_down(self) -> None:
        self._actions.append((ViewerAction.SPEED_DOWN, None))

    def request_reset_speed(self) -> None:
        self._actions.append((ViewerAction.RESET_SPEED, None))

    def request_set_motion_clip(self, clip_id: int) -> None:
        self._actions.append((ViewerAction.SET_MOTION_CLIP, int(clip_id)))

    def request_set_motion_lock(self, locked: bool) -> None:
        self._actions.append((ViewerAction.SET_MOTION_LOCK, bool(locked)))

    # ── Motion-command bridge ──────────────────────────────────────
    def _get_motion_command(self):
        """Return the env's 'motion' CommandTerm if present, else ``None``.

        Non-tracking tasks (g1_29dof locomotion, t1_getup, go2_flat, ...)
        don't register a 'motion' term, so this cleanly returns ``None``
        and the motion picker code paths become no-ops without branching
        at every call site.
        """
        terms = getattr(self.env.command_manager, "_terms", None)
        if not terms or "motion" not in terms:
            return None
        return terms["motion"]

    # Tracking-specific termination terms that interrupt a clip mid-
    # playback. With the motion lock ON these would keep resetting the
    # episode (which also re-teleports and picks a new random clip,
    # desyncing the viser dropdown), so we suspend them while locked
    # and restore them when the lock is released.
    _LOCK_SUSPENDED_TERMS: tuple[str, ...] = (
        "bad_anchor_pos_z_only",
        "bad_anchor_ori",
        "bad_motion_body_pos_z_only",
    )

    def _set_motion_clip(self, clip_id: int) -> None:
        motion_cmd = self._get_motion_command()
        if motion_cmd is None:
            return
        before = int(motion_cmd.motion_ids[0].item())
        motion_cmd.set_motion_clip(clip_id)
        after = int(motion_cmd.motion_ids[0].item())
        # Diagnostic: prints to the eval server's stdout so we can tell
        # whether a dropdown selection is actually reaching the env or
        # the viser callback is not firing.
        print(
            f"[PlayViewer] motion clip: env0 motion_ids {before} -> {after} "
            f"(requested {clip_id})"
        )

    def _set_motion_lock(self, locked: bool) -> None:
        motion_cmd = self._get_motion_command()
        if motion_cmd is None:
            return
        # Lock ON == suppress rollover teleport.
        motion_cmd.cfg.rollover_teleport = not bool(locked)

        term_mgr = getattr(self.env, "termination_manager", None)
        if term_mgr is None:
            return
        all_terms = getattr(term_mgr, "_all_terms", None)
        resolved = getattr(term_mgr, "_resolved_fns", None)
        if all_terms is None or resolved is None:
            return

        # Lazy per-viewer cache for suspended (cfg, fn) pairs.
        if not hasattr(self, "_suspended_term_cache"):
            self._suspended_term_cache: dict = {}

        for name in self._LOCK_SUSPENDED_TERMS:
            if locked:
                if name in all_terms and name not in self._suspended_term_cache:
                    self._suspended_term_cache[name] = (
                        all_terms.pop(name),
                        resolved.pop(name),
                    )
            else:
                if name in self._suspended_term_cache:
                    cfg, fn = self._suspended_term_cache.pop(name)
                    all_terms[name] = cfg
                    resolved[name] = fn
        print(
            f"[PlayViewer] motion lock {'ON' if locked else 'OFF'}: "
            f"rollover_teleport={motion_cmd.cfg.rollover_teleport}, "
            f"suspended_terms={sorted(self._suspended_term_cache)}"
        )

    # Speed controls.

    def increase_speed(self) -> None:
        if self._speed_index < len(self.SPEED_MULTIPLIERS) - 1:
            self._speed_index += 1
            self._time_multiplier = self.SPEED_MULTIPLIERS[self._speed_index]

    def decrease_speed(self) -> None:
        if self._speed_index > 0:
            self._speed_index -= 1
            self._time_multiplier = self.SPEED_MULTIPLIERS[self._speed_index]

    def reset_speed(self) -> None:
        self._speed_index = self.SPEED_MULTIPLIERS.index(1.0)
        self._time_multiplier = 1.0

    # Pause and resume.

    def pause(self) -> None:
        self._is_paused = True

    def resume(self) -> None:
        print("[PlayViewer] resume() called")
        self._is_paused = False
        self._last_error = None
        self._sim_budget = 0.0
        self._last_tick_time = time.perf_counter()

    def toggle_pause(self) -> None:
        if self._is_paused:
            self.resume()
        else:
            self.pause()

    # Core loop.

    def _execute_step(self) -> bool:
        """Run one obs → policy → step cycle. Returns True on success."""
        if self._step_count == 0:
            print("[PlayViewer] first _execute_step()")
        try:
            with torch.no_grad():
                obs = self.env.obs_manager.get_observation()
                robot_states = self.env.get_robot_state()
                action = self.policy.get_action(obs, robot_states)

                obs_out, rewards, terminated, truncated, infos = self.env.step(action)
                self._step_count += 1
                self._stats_steps += 1

                # Detect episode resets and notify policy.
                reset_idx = terminated | truncated
                if reset_idx.any():
                    self.policy.notify_reset(reset_idx.cpu().numpy())

                return True
        except Exception:
            self._last_error = traceback.format_exc()
            print(f"[PlayViewer] Exception during step:\n{self._last_error}")
            self.pause()
            return False

    def _step_physics(self, dt: float) -> None:
        """Run physics steps for this frame's sim-time budget."""
        step_dt = self.env.control_dt
        self._sim_budget += dt * self._time_multiplier
        self._was_capped = False

        if self._sim_budget < step_dt:
            return

        self.sync_viewer_to_env()
        deadline = time.perf_counter() + self.frame_time
        hit_deadline = False
        while self._sim_budget >= step_dt:
            if not self._execute_step():
                self._sim_budget = 0.0
                return
            self._sim_budget -= step_dt
            if time.perf_counter() > deadline:
                hit_deadline = True
                break

        if hit_deadline:
            self._was_capped = self._sim_budget >= step_dt
            self._sim_budget = min(self._sim_budget, step_dt)

    def _single_step(self) -> None:
        """Advance exactly one step while paused."""
        if not self._is_paused:
            return
        self.sync_viewer_to_env()
        self._execute_step()

    def reset_environment(self) -> None:
        self.env.reset()
        reset_fn = getattr(self.policy, "reset", None)
        if reset_fn is not None:
            reset_fn()
        self._step_count = 0
        self._sim_budget = 0.0
        self._last_error = None
        self._last_tick_time = time.perf_counter()

    def _process_actions(self) -> None:
        """Drain action queue. Runs on the main loop thread."""
        while self._actions:
            action, payload = self._actions.popleft()
            if action == ViewerAction.RESET:
                self.reset_environment()
            elif action == ViewerAction.TOGGLE_PAUSE:
                self.toggle_pause()
            elif action == ViewerAction.SINGLE_STEP:
                self._single_step()
            elif action == ViewerAction.RESET_SPEED:
                self.reset_speed()
            elif action == ViewerAction.SPEED_UP:
                self.increase_speed()
            elif action == ViewerAction.SPEED_DOWN:
                self.decrease_speed()
            elif action == ViewerAction.SET_MOTION_CLIP:
                self._set_motion_clip(int(payload))
            elif action == ViewerAction.SET_MOTION_LOCK:
                self._set_motion_lock(bool(payload))

    def tick(self) -> bool:
        """Advance one tick. Returns True when a render frame was produced."""
        now = time.perf_counter()
        dt = now - self._last_tick_time
        self._last_tick_time = now

        self._process_actions()

        if not self._is_paused:
            self._step_physics(dt)

        # Render at fixed frame rate.
        self._time_until_next_render -= dt
        if self._time_until_next_render > 0:
            return False

        self._time_until_next_render += self.frame_time
        if self._time_until_next_render < -self.frame_time:
            self._time_until_next_render = 0.0

        self.sync_env_to_viewer()
        self._stats_frames += 1
        return True

    def run(self, num_steps: int | None = None) -> None:
        """Main loop: setup, tick until done, close."""
        self._interrupted = False
        self.setup()
        now = time.perf_counter()
        self._stats_last_time = now
        self._last_tick_time = now

        prev_handler = None
        try:
            try:
                prev_handler = signal.signal(signal.SIGINT, self._sigint_handler)
            except ValueError:
                pass  # Non-main thread.

            while (
                self.is_running()
                and (num_steps is None or self._step_count < num_steps)
                and not self._interrupted
            ):
                if not self.tick():
                    time.sleep(0.001)
                self._update_stats()
        finally:
            self.close()
            if prev_handler is not None:
                signal.signal(signal.SIGINT, prev_handler)

    def _sigint_handler(self, signum, frame) -> None:
        self._interrupted = True
        print("\nCtrl+C received. Shutting down viewer...")
        signal.signal(signal.SIGINT, signal.SIG_DFL)

    # Stats.

    def _update_stats(self) -> None:
        if self._is_paused:
            return
        now = time.perf_counter()
        dt = now - self._stats_last_time
        if dt >= 0.5:
            self._fps = self._stats_frames / dt
            self._sps = self._stats_steps / dt
            self._stats_frames = 0
            self._stats_steps = 0
            self._stats_last_time = now

    @property
    def target_realtime(self) -> float:
        return self._time_multiplier

    @property
    def actual_realtime(self) -> float:
        return self._sps * self.env.control_dt

    @staticmethod
    def _format_speed(multiplier: float) -> str:
        if multiplier == 1.0:
            return "1x"
        inv = 1.0 / multiplier
        inv_rounded = round(inv)
        if abs(inv - inv_rounded) < 1e-9 and inv_rounded > 0:
            return f"1/{inv_rounded}x"
        return f"{multiplier:.3g}x"

    def get_status(self) -> ViewerStatus:
        return ViewerStatus(
            paused=self._is_paused,
            step_count=self._step_count,
            speed_multiplier=self._time_multiplier,
            speed_label=self._format_speed(self._time_multiplier),
            target_realtime=self.target_realtime,
            actual_realtime=self.actual_realtime,
            smoothed_fps=self._fps,
            capped=self._was_capped,
            last_error=self._last_error,
        )
