"""Where does ``env.step`` spend its time, section by section?

``g1_step_benchmark`` says how much of a wrapped step is NOT physics
(Genesis G1 flat at 4096 envs: 17.3 ms wrapped against 6.1 ms of bare
``scene.step``); ``collect_breakdown`` says how the training loop spends
its time around ``env.step``. Neither says which part of ``env.step``
itself is the cost, and that is the question when another framework's
whole step on the same engine, robot and timing is faster than ours.

This wraps the methods ``World.step`` calls, on the built env instance,
with a GPU-synchronised timer each, and prints the mean wall time per
control step of every section, nested the way the step nests them (the
reset's event applications are shown under ``_reset_idx``). Zero actions
drive the robot to its default pose, so the robot stands and the step
carries its normal contact load; episode ages are staggered so resets,
terminal observations and interval events happen at their training rate
instead of never.

Every section is synchronised before and after it is timed. That stops
work queued by one section from landing in a later one, but it also
removes the CPU/GPU overlap the real step enjoys, so the sections add up
to more than the un-instrumented step, which is measured first and
printed beside the sum: the inflation is the overlap the instrumented
numbers cannot see.

    jaxpy -m jaxrlworld.scripts.diag.perf.env_step_breakdown --sim genesis --num-envs 4096
    jaxpy -m jaxrlworld.scripts.diag.perf.env_step_breakdown --sim mujoco \\
        --config jaxrlworld.rl.configs.presets.go2.base:Go2FlatConfig
"""

from __future__ import annotations

import argparse
import importlib
import time
from collections import defaultdict
from collections.abc import Callable

import torch

from jaxrlworld.rl.evals.sim_initializers import get_initializer

_SIM_NAMES = {"genesis": "Genesis", "newton": "Newton", "mujoco": "MujocoEnv"}


class SectionTimer:
    """Wraps bound methods on live objects and accumulates their synchronised wall time."""

    def __init__(self) -> None:
        self.totals: dict[str, float] = defaultdict(float)
        self.calls: dict[str, int] = defaultdict(int)
        self.order: list[str] = []
        self.recording = False
        self._stack: list[str] = []

    def wrap(self, owner: object, method: str, label: str, mode_label: bool = False) -> None:
        original: Callable = getattr(owner, method)
        timer = self

        def timed(*args, **kwargs):
            name = f"{label}[{kwargs['mode']}]" if mode_label else label
            name = "/".join([*timer._stack, name])
            if name not in timer.order:
                timer.order.append(name)
            timer._stack.append(name.rsplit("/", 1)[-1])
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                torch.cuda.synchronize()
                timer._stack.pop()
                if timer.recording:
                    timer.totals[name] += time.perf_counter() - t0
                    timer.calls[name] += 1

        setattr(owner, method, timed)

    def report(self, steps: int, plain_ms: float) -> str:
        top_level_ms = sum(self.totals[name] for name in self.order if "/" not in name) / steps * 1e3
        lines = [
            f"{'ms/ctrl-step':>13}  {'calls/step':>10}  section",
            f"{plain_ms:13.3f}  {'':>10}  env.step, un-instrumented",
            f"{top_level_ms:13.3f}  {'':>10}  sum of top-level sections (synchronised)",
        ]
        for name in self.order:
            depth = name.count("/")
            lines.append(
                f"{self.totals[name] / steps * 1e3:13.3f}  {self.calls[name] / steps:10.2f}  "
                f"{'    ' * depth}{name.rsplit('/', 1)[-1]}"
            )
        return "\n".join(lines)


def build_env(config: str, sim: str, num_envs: int):
    mod_path, cls_name = config.split(":", 1)
    cfg_cls = getattr(importlib.import_module(mod_path), cls_name)
    cfgs = cfg_cls(sim_type=sim, num_envs=num_envs, seed=0).build()
    return get_initializer(_SIM_NAMES[sim]).init_environment(cfgs)


def instrument(env, timer: SectionTimer) -> None:
    timer.wrap(env.act_manager, "process_actions", "act_manager.process_actions")
    timer.wrap(env, "_step_physics", "_step_physics")
    timer.wrap(env.act_manager, "apply_actions", "act_manager.apply_actions")
    timer.wrap(env.scene_manager, "step", "scene_manager.step")
    timer.wrap(env.contact_manager, "advance", "contact_manager.advance")
    timer.wrap(env.termination_manager, "advance", "termination_manager.advance")
    timer.wrap(env, "_pre_termination_hook", "_pre_termination_hook")
    timer.wrap(env.termination_manager, "check_termination", "termination_manager.check_termination")
    timer.wrap(env, "_pre_reward_hook", "_pre_reward_hook")
    timer.wrap(env.reward_manager, "set_rewards", "reward_manager.set_rewards")
    timer.wrap(env.obs_manager, "process_observations", "obs_manager.process_observations (terminal obs)")
    timer.wrap(env.obs_manager, "rollback_last_history_append", "obs_manager.rollback_last_history_append")
    timer.wrap(env.event_manager, "apply", "event_manager.apply", mode_label=True)
    timer.wrap(env, "_reset_idx", "_reset_idx")
    timer.wrap(env.curriculum_manager, "compute", "curriculum_manager.compute")
    timer.wrap(env, "_reset_scene", "_reset_scene")
    timer.wrap(env.event_manager, "reset", "event_manager.reset")
    timer.wrap(env.termination_manager, "reset", "termination_manager.reset")
    timer.wrap(env.command_manager, "reset", "command_manager.reset")
    timer.wrap(env.act_manager, "reset", "act_manager.reset")
    timer.wrap(env.obs_manager, "reset", "obs_manager.reset")
    timer.wrap(env.contact_manager, "reset", "contact_manager.reset")
    timer.wrap(env.contact_manager, "refresh_after_reset", "contact_manager.refresh_after_reset")
    timer.wrap(env.reward_manager, "reset", "reward_manager.reset")
    timer.wrap(env.curriculum_manager, "reset", "curriculum_manager.reset")
    timer.wrap(env, "_post_reset_forward", "_post_reset_forward")
    timer.wrap(env.command_manager, "compute", "command_manager.compute")
    timer.wrap(env, "_forward_if_kinematics_stale", "_forward_if_kinematics_stale")
    timer.wrap(env, "_render_sensors", "_render_sensors")
    timer.wrap(env, "_advance_managers", "_advance_managers")
    timer.wrap(env.obs_manager, "advance", "obs_manager.advance")
    timer.wrap(env.reward_manager, "advance", "reward_manager.advance")
    timer.wrap(env.act_manager, "advance", "act_manager.advance")
    timer.wrap(env.obs_manager, "get_observation", "obs_manager.get_observation")


def time_plain_steps(env, actions: torch.Tensor, steps: int) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        env.step(actions)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / steps * 1e3


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="jaxrlworld.rl.configs.presets.g1_29dof.base:G1FlatConfig")
    ap.add_argument("--sim", default="genesis", choices=tuple(_SIM_NAMES))
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--no-stagger", action="store_true", help="start every episode together (no resets in the window)")
    args = ap.parse_args()

    t_build = time.perf_counter()
    env = build_env(args.config, args.sim, args.num_envs)
    env.reset()
    if not args.no_stagger:
        env.termination_manager.episode_length_buf = torch.randint_like(
            env.episode_length_buf, high=int(env.max_episode_length)
        )
    actions = torch.zeros((env.num_envs, env.num_actions), device=env.device)
    for _ in range(args.warmup):
        env.step(actions)
    torch.cuda.synchronize()
    print(
        f"[breakdown] {args.config} sim={args.sim} num_envs={args.num_envs} build+warmup {time.perf_counter() - t_build:.1f} s"
    )

    plain_ms = time_plain_steps(env, actions, args.steps)

    timer = SectionTimer()
    instrument(env, timer)
    for _ in range(args.warmup):
        env.step(actions)
    timer.recording = True
    for _ in range(args.steps):
        env.step(actions)
    timer.recording = False
    print(timer.report(args.steps, plain_ms))


if __name__ == "__main__":
    main()
