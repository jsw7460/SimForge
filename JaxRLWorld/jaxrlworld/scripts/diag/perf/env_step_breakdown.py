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

``--host-only`` times the sections without synchronising around them:
each section's time is then the host's own (its Python, its kernel
launches, and whatever waits for the device it contains), with the GPU
running behind as in the real step. The sections then add up to about
the un-instrumented step, and the table says where the host spends it;
a step whose host sum fills its wall time is bound by the host, however
busy or idle the GPU is.

``--sync-debug`` adds a census of host synchronisations: with
``torch.cuda.set_sync_debug_mode("warn")`` on during un-instrumented
steps, every synchronising CUDA call is attributed to the innermost
``jaxrlworld`` frame that led to it and the innermost frame below that
(the simulator's), and the per-step counts are printed. A sync inside
a setter is the whole cost of that setter when the GPU is otherwise
idle, so this is the list of places where the step waits for the GPU.

    jaxpy -m jaxrlworld.scripts.diag.perf.env_step_breakdown --sim genesis --num-envs 4096
    jaxpy -m jaxrlworld.scripts.diag.perf.env_step_breakdown --sim genesis --sync-debug
    jaxpy -m jaxrlworld.scripts.diag.perf.env_step_breakdown --sim mujoco \\
        --config jaxrlworld.rl.configs.presets.go2.base:Go2FlatConfig
"""

from __future__ import annotations

import argparse
import ast
import importlib
import time
import traceback
import warnings
from collections import Counter, defaultdict
from collections.abc import Callable

import torch

from jaxrlworld.rl.evals.sim_initializers import get_initializer

_SIM_NAMES = {"genesis": "Genesis", "newton": "Newton", "mujoco": "MujocoEnv"}


class SectionTimer:
    """Wraps bound methods on live objects and accumulates their synchronised wall time."""

    def __init__(self, synchronise: bool = True) -> None:
        self.totals: dict[str, float] = defaultdict(float)
        self.calls: dict[str, int] = defaultdict(int)
        self.order: list[str] = []
        self.recording = False
        self._stack: list[str] = []
        # Without synchronisation a section's time is the host's alone: its Python, its kernel launches and
        # whatever device waits it contains, with the GPU running behind as in the real step.
        self._sync = torch.cuda.synchronize if synchronise else (lambda: None)

    def wrap(
        self, owner: object, method: str, label: str, mode_label: bool = False, name_arg: int | None = None
    ) -> None:
        """Time ``owner.method``; ``mode_label`` suffixes the ``mode`` keyword, ``name_arg`` the positional arg."""
        original: Callable = getattr(owner, method)
        timer = self

        def timed(*args, **kwargs):
            if mode_label:
                name = f"{label}[{kwargs['mode']}]"
            elif name_arg is not None:
                name = f"{label}[{args[name_arg]}]"
            else:
                name = label
            name = "/".join([*timer._stack, name])
            if name not in timer.order:
                timer.order.append(name)
            timer._stack.append(name.rsplit("/", 1)[-1])
            timer._sync()
            t0 = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                timer._sync()
                timer._stack.pop()
                if timer.recording:
                    timer.totals[name] += time.perf_counter() - t0
                    timer.calls[name] += 1

        setattr(owner, method, timed)

    def report(self, steps: int, plain_ms: float) -> str:
        top_level_ms = sum(self.totals[name] for name in self.order if "/" not in name) / steps * 1e3
        how = "synchronised" if self._sync is torch.cuda.synchronize else "host time, no synchronisation"
        lines = [
            f"{'ms/ctrl-step':>13}  {'calls/step':>10}  section",
            f"{plain_ms:13.3f}  {'':>10}  env.step, un-instrumented",
            f"{top_level_ms:13.3f}  {'':>10}  sum of top-level sections ({how})",
        ]
        for name in self.order:
            depth = name.count("/")
            lines.append(
                f"{self.totals[name] / steps * 1e3:13.3f}  {self.calls[name] / steps:10.2f}  "
                f"{'    ' * depth}{name.rsplit('/', 1)[-1]}"
            )
        return "\n".join(lines)


def build_env(config: str, sim: str, num_envs: int, overrides: list[str]):
    mod_path, cls_name = config.split(":", 1)
    cfg_cls = getattr(importlib.import_module(mod_path), cls_name)
    cfgs = cfg_cls(sim_type=sim, num_envs=num_envs, seed=0).build()
    for item in overrides:
        path, value = item.split("=", 1)
        *parents, leaf = path.split(".")
        target = cfgs
        for name in parents:
            target = getattr(target, name)
        getattr(target, leaf)  # the field must exist: a typo must not silently add an attribute
        setattr(target, leaf, ast.literal_eval(value))
        print(f"[breakdown] override {path} = {getattr(target, leaf)!r}")
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
    timer.wrap(env.event_manager, "_call_event_fn", "event term", name_arg=0)
    writer = env.get_robot_state_writer("robot")
    for method in ("set_root_pose", "set_root_velocity", "set_dof_state"):
        timer.wrap(writer, method, f"writer.{method}")
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
    if env.sim_type == "genesis":
        collider = env.scene_manager.scene.rigid_solver.collider
        timer.wrap(collider, "detection", "collider.detection")
        timer.wrap(collider, "clear", "collider.clear")
        timer.wrap(collider, "get_contacts", "collider.get_contacts")
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


def _frame_label(frame: traceback.FrameSummary) -> str:
    return f"{frame.filename.rsplit('/', 1)[-1]}:{frame.lineno} {frame.name}"


def count_syncs(env, actions: torch.Tensor, steps: int) -> Counter:
    """Per-step count of synchronising CUDA calls, keyed by the frames that issued them."""
    counts: Counter = Counter()

    def record(message, category, filename, lineno, file=None, line=None) -> None:
        stack = traceback.extract_stack()[:-1]
        ours = [
            i for i, f in enumerate(stack) if "/jaxrlworld/" in f.filename and "env_step_breakdown" not in f.filename
        ]
        if not ours:
            return  # the census loop's own synchronize
        below = stack[ours[-1] + 1 :]
        callee = next((f for f in below if "/torch/" not in f.filename), None)
        key = _frame_label(stack[ours[-1]]) + ("  ->  " + _frame_label(callee) if callee else "")
        counts[key] += 1

    torch.cuda.synchronize()
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.showwarning = record
        torch.cuda.set_sync_debug_mode("warn")
        try:
            for _ in range(steps):
                env.step(actions)
        finally:
            torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="jaxrlworld.rl.configs.presets.g1_29dof.base:G1FlatConfig")
    ap.add_argument("--sim", default="genesis", choices=tuple(_SIM_NAMES))
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--no-stagger", action="store_true", help="start every episode together (no resets in the window)")
    ap.add_argument("--sync-debug", action="store_true", help="also count the host synchronisations per step")
    ap.add_argument(
        "--host-only",
        action="store_true",
        help="time the sections without synchronising: each section's host time (Python, launches, its own waits)",
    )
    ap.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help="override a built config field for an A/B, e.g. scene.rigid_options.enable_multi_contact=False; "
        "VALUE is a Python literal and the field must already exist",
    )
    args = ap.parse_args()

    t_build = time.perf_counter()
    env = build_env(args.config, args.sim, args.num_envs, args.set)
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

    if args.sync_debug:
        counts = count_syncs(env, actions, args.steps)
        print(f"[breakdown] host syncs: {sum(counts.values()) / args.steps:.1f} per step")
        for key, n in counts.most_common():
            print(f"{n / args.steps:10.2f}  {key}")

    timer = SectionTimer(synchronise=not args.host_only)
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
