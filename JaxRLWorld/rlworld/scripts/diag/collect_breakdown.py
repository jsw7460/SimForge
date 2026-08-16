"""Where does a collect step's time go, if not into the simulator?

Measured on yam_lift and k1_joystick at 8192 envs on mujoco: the arm's
``env.step`` is 13.8 ms against the humanoid's 27.0 ms — half the
physics — yet the arm trains at a LOWER throughput. So most of the arm's
collect time is spent somewhere other than stepping the simulator, and
"somewhere" is the whole question.

``_collect_experience`` is one loop of eight or so distinct operations
against three runtimes (torch, JAX, the simulator). A single wall-clock
number for the loop cannot say which of them is the cost, and the
answers differ by an order of magnitude in what they would imply: a slow
``env.step`` is a physics problem, a slow ``act`` is a policy-size
problem, and a slow ``.cpu()`` is a pipeline stall that no amount of
faster physics will fix.

The loop below mirrors ``OnPolicyRunner._collect_experience`` operation
for operation. It is a copy rather than instrumentation of the real one
on purpose: the env-var-gated profilers that used to live inside the
managers were removed, and nothing here runs unless this script is
invoked. The copy has to be kept in step with the original — if the two
drift, this measures a loop that no longer exists.

Every section synchronizes before it is timed. Without that, work queued
by one section lands in whichever later section happens to synchronize,
and the profile points at the messenger.

    jaxpy -m rlworld.scripts.diag.collect_breakdown --preset yam_lift
    jaxpy -m rlworld.scripts.diag.collect_breakdown --preset k1_joystick
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections import defaultdict

import jax.numpy as jnp
import torch

from rlworld.rl.algorithms.base import ActInput
from rlworld.rl.configs.presets.go2.base import Go2FlatConfig
from rlworld.rl.configs.presets.k1_joystick.base import K1JoystickConfig
from rlworld.rl.configs.presets.yam_lift.base import YamLiftConfig
from rlworld.rl.runners import BaseRunner
from rlworld.rl.utils.jax_utils import jax_to_torch, torch_to_jax

_PRESETS = {
    "go2": Go2FlatConfig,
    "k1_joystick": K1JoystickConfig,
    "yam_lift": YamLiftConfig,
}


class _Timer:
    """Wall-clock per section, with both runtimes drained before each read.

    JAX dispatches asynchronously and so does CUDA. Timing a section
    without draining first charges it for whatever the PREVIOUS section
    queued, which is how an innocent ``.cpu()`` ends up looking like the
    bottleneck: it is merely the first call that waits.
    """

    def __init__(self) -> None:
        self.samples: dict[str, list[float]] = defaultdict(list)
        self._t0 = 0.0

    @staticmethod
    def _drain(value=None) -> None:
        if value is not None:
            jnp.asarray(value).block_until_ready()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def start(self) -> None:
        self._drain()
        self._t0 = time.perf_counter()

    def stop(self, name: str, jax_result=None) -> None:
        self._drain(jax_result)
        self.samples[name].append((time.perf_counter() - self._t0) * 1e3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=list(_PRESETS), default="yam_lift")
    ap.add_argument("--sim", default="mujoco", choices=("genesis", "newton", "mujoco"))
    ap.add_argument("--num-envs", type=int, default=8192)
    ap.add_argument("--warmup", type=int, default=24)
    ap.add_argument("--steps", type=int, default=96)
    args = ap.parse_args()

    cfgs = _PRESETS[args.preset](sim_type=args.sim, num_envs=args.num_envs).build()
    runner = BaseRunner.create_with_env(cfgs, use_wandb=False)
    env = runner.env

    # The rollout buffer holds exactly one iteration, and add_transition
    # raises once it is full. The real runner empties it in update(); this
    # loop never updates, so it empties it on the same boundary.
    steps_per_iteration = runner.num_steps_per_env

    obs_dict, _ = env.reset()
    actor_obs = torch_to_jax(obs_dict["actor"])
    critic_obs = torch_to_jax(obs_dict["critic"])

    timer = _Timer()

    for step_i in range(args.warmup + args.steps):
        # Warmup covers JAX tracing and the first CUDA-graph capture, both
        # of which land on step 0 and would otherwise dominate the mean.
        record = step_i >= args.warmup
        if step_i % steps_per_iteration == 0:
            runner.alg.storage.clear()

        timer.start()
        actions = runner.alg.act(ActInput(actor_obs, critic_obs))
        if record:
            timer.stop("act (policy forward, JAX)", actions)

        timer.start()
        actions_for_env = runner._process_action_for_env(actions)
        actions_torch = jax_to_torch(actions_for_env, runner.device)
        if record:
            timer.stop("jax -> torch (actions)")

        timer.start()
        obs_dict, rewards, terminated, truncated, infos = env.step(actions_torch)
        if record:
            timer.stop("env.step (the simulator)")

        timer.start()
        dones = terminated | truncated
        actor_obs = torch_to_jax(obs_dict["actor"])
        critic_obs = torch_to_jax(obs_dict["critic"])
        rewards_jax = torch_to_jax(rewards)
        if record:
            timer.stop("torch -> jax (obs, rewards)", actor_obs)

        # The runner's own comment says DLPack cannot carry booleans, so
        # these two go through the host. Two forced device-to-host round
        # trips per step, every step.
        timer.start()
        terminated_jax = jnp.asarray(terminated.cpu().numpy())
        truncated_jax = jnp.asarray(truncated.cpu().numpy())
        if record:
            timer.stop("done flags via host (.cpu)", terminated_jax)

        timer.start()
        infos_jax = {}
        if infos.get("final_observation") is not None:
            infos_jax["final_observation"] = {
                "actor": torch_to_jax(infos["final_observation"]["actor"]),
                "critic": torch_to_jax(infos["final_observation"]["critic"]),
            }
            if infos.get("bootstrap_mask") is not None:
                infos_jax["bootstrap_mask"] = jnp.asarray(infos["bootstrap_mask"].cpu().numpy())
        if record:
            timer.stop("final_observation handling")

        timer.start()
        runner.alg.process_env_step(
            rewards_jax,
            terminated_jax,
            truncated_jax,
            infos_jax,
            next_actor_obs=actor_obs,
            next_critic_obs=critic_obs,
        )
        if record:
            timer.stop("process_env_step (storage)")

        timer.start()
        runner._update_reward_stats(
            reward_info=infos["rewards_per_type"],
            dones=dones,
            success=infos.get("success", None),
        )
        if record:
            timer.stop("reward statistics")

    print("=" * 78)
    print(f"COLLECT BREAKDOWN  [preset={args.preset}  sim={args.sim}  num_envs={args.num_envs}]")
    print(f"  {args.steps} steps after {args.warmup} warmup, {env.act_manager.num_actions} actions")
    print("=" * 78)

    means = {name: statistics.mean(vals) for name, vals in timer.samples.items()}
    total = sum(means.values())
    for name, mean_ms in sorted(means.items(), key=lambda kv: -kv[1]):
        median_ms = statistics.median(timer.samples[name])
        share = 100.0 * mean_ms / total
        print(f"  {name:<32} {mean_ms:7.3f} ms  (median {median_ms:7.3f})  {share:5.1f}%")
    print("-" * 78)
    print(f"  {'TOTAL per collect step':<32} {total:7.3f} ms")
    sim_share = 100.0 * means["env.step (the simulator)"] / total
    print(f"  {'of which the simulator':<32} {sim_share:5.1f}%")
    print("=" * 78)
    print("  Sections are drained before and after, so each number is that")
    print("  section's own work, not the previous section's queue.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
