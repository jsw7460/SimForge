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
from copy import deepcopy

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
    ap.add_argument(
        "--no-stagger",
        action="store_true",
        help="Start every episode together instead of spreading their ends out.",
    )
    args = ap.parse_args()

    cfgs = _PRESETS[args.preset](sim_type=args.sim, num_envs=args.num_envs).build()
    runner = BaseRunner.create_with_env(cfgs, use_wandb=False)
    env = runner.env

    # The rollout buffer holds exactly one iteration, and add_transition
    # raises once it is full. The real runner empties it in update(); this
    # loop never updates, so it empties it on the same boundary.
    steps_per_iteration = runner.num_steps_per_env

    obs_dict, _ = env.reset()

    # Spread the episode ends out, the way ``learn(init_at_random_ep_len=)``
    # does. Straight after a reset every environment is the same age, so
    # nothing finishes for a full episode and the loop measures only the
    # cheap case: no ``final_observation`` to build, no bootstrap value to
    # evaluate, no ``_reset_idx``. Training spends almost every step in
    # the other case — 8192 environments averaging 946 steps means about
    # nine of them end per step — so measuring without staggering
    # describes a loop the trainer is almost never in.
    if not args.no_stagger:
        env.termination_manager.episode_length_buf = torch.randint_like(
            env.episode_length_buf, high=int(env.max_episode_length)
        )
    actor_obs = torch_to_jax(obs_dict["actor"])
    critic_obs = torch_to_jax(obs_dict["critic"])

    timer = _Timer()
    done_steps: list[int] = []

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
            # Outside every timed section: how often the expensive branch
            # is taken is the first thing to check before believing any of
            # the numbers above.
            done_steps.append(int(dones.sum()))

    print("=" * 78)
    print(f"COLLECT BREAKDOWN  [preset={args.preset}  sim={args.sim}  num_envs={args.num_envs}]")
    print(f"  {args.steps} steps after {args.warmup} warmup, {env.act_manager.num_actions} actions")
    with_dones = sum(1 for d in done_steps if d > 0)
    print(
        f"  {with_dones}/{len(done_steps)} steps had at least one done, "
        f"{statistics.mean(done_steps):.1f} envs per step on average"
    )
    print("=" * 78)

    means = {name: statistics.mean(vals) for name, vals in timer.samples.items()}
    total = sum(means.values())

    # A mean far above the median is the interesting case, not a rounding
    # detail: it means a handful of steps cost hundreds of times what the
    # typical one does, and an average spreads that evenly across all of
    # them so it reads as a uniformly slow section. ``spikes`` counts the
    # steps responsible.
    print(f"  {'section':<32}{'mean':>9}{'median':>9}{'p95':>9}{'max':>9}{'spikes':>8}{'share':>8}")
    print("-" * 78)
    for name, mean_ms in sorted(means.items(), key=lambda kv: -kv[1]):
        vals = sorted(timer.samples[name])
        median_ms = statistics.median(vals)
        spikes = sum(1 for v in vals if v > 10.0 * max(median_ms, 1e-6))
        print(
            f"  {name:<32}{mean_ms:9.3f}{median_ms:9.3f}{vals[int(0.95 * len(vals))]:9.3f}"
            f"{vals[-1]:9.3f}{spikes:8d}{100.0 * mean_ms / total:7.1f}%"
        )
    print("-" * 78)
    medians_total = sum(statistics.median(v) for v in timer.samples.values())
    print(f"  {'TOTAL per collect step':<32}{total:9.3f}   (medians sum to {medians_total:.3f} ms)")
    sim_share = 100.0 * means["env.step (the simulator)"] / total
    print(f"  {'of which the simulator':<32}{sim_share:8.1f}%")
    print("=" * 78)
    print("  Sections are drained before and after, so each number is that")
    print("  section's own work, not the previous section's queue.")
    print("  spikes = steps costing more than 10x that section's median.")

    _probe_step_phases(env, timer, actions_torch)
    _probe_done_branch(env, timer)
    return 0


def _probe_step_phases(env, timer: _Timer, actions: torch.Tensor) -> None:
    """Split ``env.step`` into the phases it runs on EVERY step.

    ``env.step`` is 82% of a collect step, and the terminate/reset branch
    accounts for only about eight of its twenty-one milliseconds. The
    rest is physics, observations, rewards, terminations and commands,
    and which of those dominates decides whether the next thing worth
    optimising is the simulator or the manager layer above it.

    Ordered as ``World.step`` runs them. Physics is the decimation loop,
    so it already covers several solver steps.
    """
    repeats = 20
    print()
    print("=" * 78)
    print("env.step, PHASE BY PHASE  (the work every step does)")
    print("=" * 78)

    probes = {
        "act_manager.process_actions": lambda: env.act_manager.process_actions(actions),
        "_step_physics (decimation loop)": lambda: (env._step_physics(), env._invalidate_cache()),
        "termination.advance": lambda: env.termination_manager.advance(),
        "termination.check_termination": lambda: env.termination_manager.check_termination(),
        "reset_buf.nonzero": lambda: env.termination_manager.reset_buf.nonzero(as_tuple=False).flatten(),
        "reward_manager.set_rewards": lambda: env.reward_manager.set_rewards(
            reward_buffer=env.rew_buf,
            episode_sums=env.episode_sums,
            reward_buffer_per_type=env.rew_buf_per_type,
        ),
        "command_manager.compute": lambda: env.command_manager.compute(env.control_dt),
        "obs_manager.advance (the returned obs)": lambda: env.obs_manager.advance(),
        "reward_manager.advance": lambda: env.reward_manager.advance(),
        "act_manager.advance": lambda: env.act_manager.advance(),
    }
    _time_probes(timer, probes, repeats)
    _probe_observation_terms(env, timer, repeats)


def _probe_observation_terms(env, timer: _Timer, repeats: int) -> None:
    """Time each observation term's own function.

    Building the observations costs 4.37 ms and happens twice per step
    once anything terminates — 8.7 ms against physics' 8.3 ms. For an
    arm reporting seven joints, a cube pose and a command, that cannot be
    arithmetic, so the question is whether one term is expensive or every
    term costs a fixed amount and there are simply many of them.

    Only the term function is timed here, not the noise, scaling, delay,
    history and concat the manager wraps around it. The gap between the
    sum of these and the manager's own 4.37 ms is that wrapper.
    """
    print()
    print("=" * 78)
    print("OBSERVATION TERMS, one build")
    print("=" * 78)

    probes = {}
    for group_name, terms in env.obs_manager._group_terms.items():
        for term_name, obs_term in terms.items():
            func = env.obs_manager._resolved_fns[group_name][term_name]
            probes[f"{group_name}/{term_name}"] = lambda func=func, obs_term=obs_term: func(
                env.obs_manager.env, **obs_term.params
            )

    _time_probes(timer, probes, repeats)
    total = sum(statistics.median(timer.samples[name]) for name in probes)
    print(f"  {'sum of term functions':<38}{total:8.3f} ms")
    print("=" * 78)


def _probe_done_branch(env, timer: _Timer) -> None:
    """Time the pieces ``step`` runs only when something terminated.

    Staggering episode ends moved ``env.step`` from 15.3 ms to 23.0 ms
    while terminating about 8 environments out of 8192 — resetting a
    tenth of a percent of the batch costs more than stepping all of it.
    That branch is five operations, and four of them work on the whole
    batch regardless of how many environments actually ended, so knowing
    which one carries the 7.7 ms decides whether there is anything to fix.

    These are the calls ``step`` makes, invoked directly and out of their
    usual order. That is fine for timing and wrong for state, so nothing
    should read this environment afterwards.
    """
    env_ids = torch.arange(8, device=env.device)
    repeats = 20
    print()
    print("=" * 78)
    print(f"THE DONE BRANCH, PIECE BY PIECE  ({len(env_ids)} environments terminating)")
    print("=" * 78)

    probes = {
        "obs rebuild (whole batch)": lambda: (
            env.obs_manager.process_observations(update_history=True),
            env.obs_manager.rollback_last_history_append(),
        ),
        "clone the observations": lambda: {key: obs.clone() for key, obs in env.obs_manager.obs_dict.items()},
        "deepcopy(episode_sums)": lambda: deepcopy(env.episode_sums),
        "_reset_idx(env_ids)": lambda: env._reset_idx(env_ids),
        "_post_reset_forward (whole batch)": lambda: env._post_reset_forward(),
    }

    _time_probes(timer, probes, repeats)

    _probe_reset_managers(env, timer, env_ids, repeats)
    _probe_reset_events(env, timer, env_ids, repeats)


def _probe_reset_events(env, timer: _Timer, env_ids: torch.Tensor, repeats: int) -> None:
    """Split ``event.apply("reset")`` across its terms.

    It is 2.0 ms of ``_reset_idx``'s 2.9 ms — more than the other twelve
    managers together — and it is a plain Python loop over the reset
    terms, so its cost is however many there are and what each writes.
    """
    reset_terms = env.event_manager._terms_by_mode.get("reset", [])
    if not reset_terms:
        return

    print()
    print("=" * 78)
    print(f'event.apply("reset"), TERM BY TERM  ({len(env_ids)} environments)')
    print("=" * 78)
    probes = {
        name: (lambda name=name, term=term: env.event_manager._call_event_fn(name, term, env_ids=env_ids))
        for name, term in reset_terms
    }
    _time_probes(timer, probes, repeats)


def _time_probes(timer: _Timer, probes: dict, repeats: int) -> None:
    """Run each probe warm, then time it, then print mean and median."""
    for name, call in probes.items():
        for _ in range(3):
            call()
        timer.samples.pop(name, None)
        for _ in range(repeats):
            timer.start()
            call()
            timer.stop(name)
        vals = timer.samples[name]
        print(f"  {name:<38}{statistics.mean(vals):8.3f} ms   median {statistics.median(vals):8.3f}")
    print("=" * 78)


def _probe_reset_managers(env, timer: _Timer, env_ids: torch.Tensor, repeats: int) -> None:
    """Split ``_reset_idx`` across the managers it fans out to.

    Resetting 8 environments costs 2.99 ms; resetting all 8192 costs
    2.95 ms. A cost that ignores how much work it is given is not
    arithmetic — it is a fixed number of kernel launches and Python
    calls, and the only way to shrink it is to find which of the dozen
    managers issues the most of them.

    The list mirrors ``World._reset_idx`` in order. Anything it does
    conditionally is guarded the same way here, so an absent mode shows
    up as a missing row rather than a zero.
    """
    print()
    print("=" * 78)
    print(f"_reset_idx, MANAGER BY MANAGER  ({len(env_ids)} environments)")
    print("=" * 78)

    probes = {
        "curriculum.compute": lambda: env.curriculum_manager.compute(env_ids=env_ids),
        "_reset_scene (backend)": lambda: env._reset_scene(env_ids),
        "event.reset": lambda: env.event_manager.reset(env_ids),
    }
    for mode in ("reset", "reset_dr"):
        if mode in env.event_manager.available_modes:
            probes[f"event.apply({mode})"] = lambda mode=mode: env.event_manager.apply(mode=mode, env_ids=env_ids)
    probes.update(
        {
            "termination.reset": lambda: env.termination_manager.reset(env_ids),
            "command.reset": lambda: env.command_manager.reset(env_ids),
            "action.reset": lambda: env.act_manager.reset(env_ids),
            "observation.reset": lambda: env.obs_manager.reset(env_ids),
            "contact.reset": lambda: env.contact_manager.reset(env_ids),
            "contact.refresh_after_reset": lambda: env.contact_manager.refresh_after_reset(env_ids),
            "reward.reset": lambda: env.reward_manager.reset(env_ids),
            "curriculum.reset": lambda: env.curriculum_manager.reset(env_ids),
            "episode_sums index_fill": lambda: [
                env.episode_sums[key].index_fill_(0, env_ids, 0.0) for key in list(env.episode_sums.keys())
            ],
        }
    )

    _time_probes(timer, probes, repeats)


if __name__ == "__main__":
    raise SystemExit(main())
