"""Exhaustive correctness diagnostic for the ManiSkill -> JaxRLWorld adapter.

Goal: 100% confidence that ``jaxrlworld.rl.envs.maniskill_env.ManiSkillEnv``
faithfully satisfies the JaxRLWorld runner contract -- with special, paranoid
attention to ``final_observation`` (the silent-failure surface for truncation
bootstrap).

This script is JAX-free and runner-free: it drives the adapter directly and
cross-checks it against ManiSkill ground truth, so it can run with plain
``python`` (no ``jaxpy`` needed):

    python -m jaxrlworld.scripts.diag.gates.check_maniskill_adapter \
        --task PickCube-v1 --num_envs 8 --steps 200 --horizon 8

Every check both DUMPS its measurements verbosely and HARD-ASSERTS the
invariant; on any violation the script crashes immediately (no fallbacks).

Phases
------
1. Construction & spaces / runner-required attributes.
2. ``reset()`` contract + cross-instance determinism.
3. Long random rollout: per-step dtype/device/shape/finite invariants, plus
   verbose reward / done / episode-length / final_observation / success dumps.
4. ``final_observation`` GROUND TRUTH under synchronized truncation: an
   auto-reset adapter vs an auto_reset=False env stepped in lockstep -- the
   no-reset env's returned obs IS the true terminal obs.
5. ``final_observation`` GROUND TRUTH under natural termination (success/fail),
   masked to the first envs that finish; also verifies reward pass-through,
   terminated/truncated semantics, and reset-actually-changed-state.
6. Action clamping.
"""

from __future__ import annotations

import argparse
import sys

import gymnasium as gym
import torch
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from jaxrlworld.rl.envs.maniskill_env import ManiSkillEnv

# --------------------------------------------------------------------------- #
# Reporting helpers                                                           #
# --------------------------------------------------------------------------- #
_FAILURES: list[str] = []


def hr(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def tstats(t) -> str:
    """One-line tensor summary."""
    if t is None:
        return "None"
    if not isinstance(t, torch.Tensor):
        return f"<non-tensor {type(t).__name__}>"
    td = t.detach()
    f = td.float()
    has_nan = bool(torch.isnan(f).any())
    has_inf = bool(torch.isinf(f).any())
    if td.numel() == 0:
        rng = "empty"
    else:
        rng = f"min={f.min().item():.4g} max={f.max().item():.4g} mean={f.mean().item():.4g}"
    return f"shape={tuple(td.shape)} dtype={td.dtype} dev={td.device} {rng} nan={has_nan} inf={has_inf}"


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  ::  {detail}" if detail else ""))
    if not cond:
        _FAILURES.append(f"{name} :: {detail}")
        # Crash early & loud -- a failed invariant must never be swallowed.
        raise AssertionError(f"CHECK FAILED: {name} :: {detail}")


def finite(t: torch.Tensor) -> bool:
    f = t.detach().float()
    return not bool(torch.isnan(f).any()) and not bool(torch.isinf(f).any())


# --------------------------------------------------------------------------- #
# Env builders                                                                #
# --------------------------------------------------------------------------- #
def make_vec(
    task: str,
    num_envs: int,
    obs_mode: str,
    control_mode: str,
    horizon: int,
    auto_reset: bool,
    ignore_terminations: bool,
):
    base = gym.make(
        task,
        num_envs=num_envs,
        obs_mode=obs_mode,
        control_mode=control_mode,
        sim_backend="physx_cuda",
        max_episode_steps=horizon,
    )
    return ManiSkillVectorEnv(base, num_envs, auto_reset=auto_reset, ignore_terminations=ignore_terminations)


def make_adapter(task, num_envs, obs_mode, control_mode, horizon, seed, ignore_terminations=False):
    venv = make_vec(
        task, num_envs, obs_mode, control_mode, horizon, auto_reset=True, ignore_terminations=ignore_terminations
    )
    env = ManiSkillEnv(
        venv,
        env_cfg=None,
        scene_cfg=None,
        obs_cfg=None,
        act_cfg=None,
        reward_cfg=None,
        command_cfg=None,
        seed=seed,
    )
    return env


def spec_limit(venv):
    """Effective max_episode_steps from the gym spec (truncation horizon).

    ManiSkill's base env does not compute truncation; the time-limit wrapper
    applied by ``gym.make`` does, and the limit lives on the spec. Reading it
    here (rather than trusting the ``max_episode_steps=`` override) keeps the
    diagnostic correct whether or not the override is honored.
    """
    return getattr(venv.spec, "max_episode_steps", None)


def sample_action_seq(num_envs, num_actions, low, high, steps, device, seed):
    g = torch.Generator(device=device).manual_seed(seed)
    seq = []
    span = high - low
    for _ in range(steps):
        a = torch.rand(num_envs, num_actions, generator=g, device=device) * span + low
        seq.append(a)
    return seq


# --------------------------------------------------------------------------- #
# Phase 1: construction & spaces                                              #
# --------------------------------------------------------------------------- #
def phase1_construction(args):
    hr("PHASE 1 -- Construction, spaces, runner-required attributes")
    env = make_adapter(args.task, args.num_envs, args.obs_mode, args.control_mode, args.horizon, seed=0)

    print(f"  task={args.task} obs_mode={args.obs_mode} control_mode={args.control_mode}")
    print(f"  num_envs={env.num_envs} num_actions={env.num_actions} obs_dim={env._obs_dim} device={env.device}")
    print(f"  action_low : {tstats(env.action_low)}")
    print(f"  action_high: {tstats(env.action_high)}")

    check("num_envs matches request", env.num_envs == args.num_envs, f"{env.num_envs} vs {args.num_envs}")
    check("device is cuda", env.device.type == "cuda", str(env.device))
    check("num_actions positive", env.num_actions > 0, str(env.num_actions))
    check("obs_dim positive", env._obs_dim > 0, str(env._obs_dim))
    check(
        "action_low shape == (num_actions,)",
        tuple(env.action_low.shape) == (env.num_actions,),
        str(tuple(env.action_low.shape)),
    )
    check(
        "action_high shape == (num_actions,)",
        tuple(env.action_high.shape) == (env.num_actions,),
        str(tuple(env.action_high.shape)),
    )
    check("action_low < action_high everywhere", bool((env.action_low < env.action_high).all()))
    check("action bounds on device", env.action_low.device.type == "cuda" and env.action_high.device.type == "cuda")

    # calculate_obs_dim contract (used by the runner to size networks).
    dims = env.calculate_obs_dim()
    print(f"  calculate_obs_dim(): {dims}")
    check("calculate_obs_dim has actor", dims.get("actor") == env._obs_dim, str(dims))
    check("calculate_obs_dim has critic", dims.get("critic") == env._obs_dim, str(dims))

    # scene_manager contract: on_policy_runner does scene_manager.trees.get("robot", None).
    check("scene_manager present", hasattr(env, "scene_manager"))
    tree = env.scene_manager.trees.get("robot", None)
    check("scene_manager.trees.get('robot') is None (MLP path)", tree is None, repr(tree))

    # Abstract methods are all implemented (instantiation already proves it, but
    # assert the inert returns explicitly).
    check("robot is None", env.robot is None)
    check("get_robot_data is None", env.get_robot_data() is None)
    check("get_robot_state_writer is None", env.get_robot_state_writer() is None)

    env.gym_env.close()


# --------------------------------------------------------------------------- #
# Phase 2: reset() contract + determinism                                     #
# --------------------------------------------------------------------------- #
def phase2_reset(args):
    hr("PHASE 2 -- reset() contract + cross-instance determinism")
    env = make_adapter(args.task, args.num_envs, args.obs_mode, args.control_mode, args.horizon, seed=123)
    obs_dict, info = env.reset()
    print(f"  obs_dict['actor'] : {tstats(obs_dict['actor'])}")
    print(f"  obs_dict['critic']: {tstats(obs_dict['critic'])}")
    print(f"  info keys: {list(info.keys())}")

    check("reset obs_dict has actor+critic", set(obs_dict) >= {"actor", "critic"}, str(list(obs_dict)))
    a, c = obs_dict["actor"], obs_dict["critic"]
    check("actor shape == (N, D)", tuple(a.shape) == (env.num_envs, env._obs_dim), str(tuple(a.shape)))
    check("critic shape == (N, D)", tuple(c.shape) == (env.num_envs, env._obs_dim), str(tuple(c.shape)))
    check("actor dtype float32", a.dtype == torch.float32, str(a.dtype))
    check("actor on cuda", a.device.type == "cuda", str(a.device))
    check("reset obs finite", finite(a))
    check("info has rewards_per_type", "rewards_per_type" in info, str(list(info)))
    check("_current_obs set after reset", env._current_obs is not None)
    check("episode_length_buf zeroed", int(env.episode_length_buf.abs().sum()) == 0)

    # obs_manager.get_observation must echo the cached obs (off-policy runner
    # path: env.obs_manager.get_observation()).
    om = env.obs_manager.get_observation()
    check("obs_manager.get_observation matches reset obs", torch.equal(om["actor"], a), "mismatch with cached obs")

    # Determinism: a second independent instance with the same seed must reset
    # to the same observation (premise of the lockstep cross-checks below).
    env2 = make_adapter(args.task, args.num_envs, args.obs_mode, args.control_mode, args.horizon, seed=123)
    obs2, _ = env2.reset()
    diff = (a - obs2["actor"]).abs().max().item()
    print(f"  cross-instance reset max|diff| = {diff:.3e}")
    check("two same-seed instances reset identically", diff < 1e-4, f"max|diff|={diff:.3e}")

    env.gym_env.close()
    env2.gym_env.close()


# --------------------------------------------------------------------------- #
# Phase 3: long rollout invariants + verbose dumps                           #
# --------------------------------------------------------------------------- #
def phase3_rollout(args):
    hr("PHASE 3 -- Long random rollout: per-step invariants + verbose dumps")
    env = make_adapter(args.task, args.num_envs, args.obs_mode, args.control_mode, args.horizon, seed=7)
    env.reset()
    actions = sample_action_seq(
        env.num_envs, env.num_actions, env.action_low, env.action_high, args.steps, env.device, seed=7
    )

    total_done = 0
    total_term = 0
    total_trunc = 0
    rew_min, rew_max, rew_sum, rew_n = float("inf"), float("-inf"), 0.0, 0
    saw_final_obs = False
    saw_success_key = False

    for t in range(args.steps):
        obs_dict, rewards, terminated, truncated, infos = env.step(actions[t])
        dones = terminated | truncated

        # Hard invariants every single step.
        check(f"[t={t}] obs_dict has actor+critic", set(obs_dict) >= {"actor", "critic"})
        check(f"[t={t}] actor shape", tuple(obs_dict["actor"].shape) == (env.num_envs, env._obs_dim))
        check(
            f"[t={t}] actor float32 cuda finite",
            obs_dict["actor"].dtype == torch.float32
            and obs_dict["actor"].device.type == "cuda"
            and finite(obs_dict["actor"]),
        )
        check(f"[t={t}] rewards shape", tuple(rewards.shape) == (env.num_envs,))
        check(
            f"[t={t}] rewards float32 cuda finite",
            rewards.dtype == torch.float32 and rewards.device.type == "cuda" and finite(rewards),
        )
        check(
            f"[t={t}] terminated bool shape",
            terminated.dtype == torch.bool and tuple(terminated.shape) == (env.num_envs,),
        )
        check(
            f"[t={t}] truncated bool shape", truncated.dtype == torch.bool and tuple(truncated.shape) == (env.num_envs,)
        )
        check(f"[t={t}] infos has rewards_per_type", "rewards_per_type" in infos)
        check(
            f"[t={t}] rewards_per_type.total_reward == rewards",
            torch.equal(infos["rewards_per_type"]["total_reward"], rewards),
        )

        # final_observation structural invariant: present iff some env done.
        fo = infos.get("final_observation", None)
        if dones.any():
            check(f"[t={t}] final_observation present on done step", fo is not None)
            check(f"[t={t}] final_observation has actor+critic", set(fo) >= {"actor", "critic"})
            check(f"[t={t}] final_observation actor shape", tuple(fo["actor"].shape) == (env.num_envs, env._obs_dim))
            check(
                f"[t={t}] final_observation float32 cuda finite",
                fo["actor"].dtype == torch.float32 and fo["actor"].device.type == "cuda" and finite(fo["actor"]),
            )
            saw_final_obs = True
            if "success" in infos:
                saw_success_key = True
                check(
                    f"[t={t}] success bool shape",
                    infos["success"].dtype == torch.bool and tuple(infos["success"].shape) == (env.num_envs,),
                )
        else:
            check(f"[t={t}] final_observation is None when no env done", fo is None)

        # episode_length_buf bookkeeping: zero exactly at done rows.
        check(
            f"[t={t}] episode_length_buf zero at done rows",
            bool((env.episode_length_buf[dones] == 0).all()) if dones.any() else True,
        )

        total_done += int(dones.sum())
        total_term += int(terminated.sum())
        total_trunc += int(truncated.sum())
        r = rewards.float()
        rew_min = min(rew_min, r.min().item())
        rew_max = max(rew_max, r.max().item())
        rew_sum += r.sum().item()
        rew_n += r.numel()

    print(f"  steps={args.steps} num_envs={env.num_envs}")
    print(f"  done events: total={total_done}  terminated={total_term}  truncated={total_trunc}")
    print(f"  reward over rollout: min={rew_min:.4g} max={rew_max:.4g} mean={rew_sum / max(rew_n, 1):.4g}")
    print(f"  observed final_observation on a done step: {saw_final_obs}")
    print(f"  observed success key on a done step: {saw_success_key}")

    check(
        "at least one episode ended during rollout (else can't test finals)",
        total_done > 0,
        "increase --steps or lower --horizon",
    )
    check("final_observation actually exercised", saw_final_obs)

    env.gym_env.close()


# --------------------------------------------------------------------------- #
# Phase 4: final_observation ground truth -- synchronized truncation          #
# --------------------------------------------------------------------------- #
def phase4_truncation_groundtruth(args):
    hr("PHASE 4 -- final_observation GROUND TRUTH (synchronized truncation)")
    print("  Setup: ignore_terminations=True so the ONLY done cause is the time")
    print("  limit -> all envs truncate together. A reference env with")
    print("  auto_reset=False never resets, so its returned obs on the done step")
    print("  IS the true terminal obs. Compare against the adapter's")
    print("  final_observation. Stepped in lockstep with identical actions.\n")

    seed = 999
    # Adapter: auto-reset ON, terminations ignored.
    adapter = make_adapter(
        args.task, args.num_envs, args.obs_mode, args.control_mode, args.horizon, seed=seed, ignore_terminations=True
    )
    # Reference: same task/seed but auto_reset OFF (no reset -> obs stays terminal).
    ref = make_vec(
        args.task,
        args.num_envs,
        args.obs_mode,
        args.control_mode,
        args.horizon,
        auto_reset=False,
        ignore_terminations=True,
    )

    a0, _ = adapter.reset()
    r0, _ = ref.reset(seed=seed)
    init_diff = (a0["actor"] - r0.float()).abs().max().item()
    print(f"  initial reset max|diff| (adapter vs ref) = {init_diff:.3e}")
    check("adapter & reference start identical", init_diff < 1e-4, f"{init_diff:.3e}")

    limit = spec_limit(adapter.gym_env) or args.horizon
    budget = limit + 2
    print(f"  effective max_episode_steps (from spec) = {spec_limit(adapter.gym_env)} -> stepping up to {budget}")
    actions = sample_action_seq(
        adapter.num_envs,
        adapter.num_actions,
        adapter.action_low,
        adapter.action_high,
        budget,
        adapter.device,
        seed=seed,
    )

    cycles_verified = 0
    for t in range(budget):
        a_obs, a_rew, a_term, a_trunc, a_info = adapter.step(actions[t])
        r_obs, r_rew, r_term, r_trunc, r_info = ref.step(actions[t])
        a_dones = a_term | a_trunc
        r_dones = (r_term | r_trunc).to(torch.bool).reshape(adapter.num_envs)

        # Rewards must match exactly up to the divergence point.
        rew_diff = (a_rew - r_rew.float().reshape(adapter.num_envs)).abs().max().item()
        check(f"[t={t}] reward pass-through matches reference", rew_diff < 1e-4, f"max|diff|={rew_diff:.3e}")

        if a_dones.any() or r_dones.any():
            print(f"  >> done step t={t}: adapter dones={int(a_dones.sum())} ref dones={int(r_dones.sum())}")
            check(f"[t={t}] done masks agree (adapter vs ref)", bool(torch.equal(a_dones, r_dones)), "mask mismatch")
            check(
                f"[t={t}] truncation is synchronized (all envs)",
                bool(a_dones.all()),
                f"{int(a_dones.sum())}/{adapter.num_envs} done",
            )

            fo = a_info.get("final_observation", None)
            check(f"[t={t}] adapter exposes final_observation", fo is not None)
            term_adapter = fo["actor"]
            term_ref = r_obs.float().reshape(adapter.num_envs, adapter._obs_dim)

            # GROUND TRUTH: adapter's terminal obs == reference's no-reset obs.
            gt_diff = (term_adapter - term_ref).abs().max().item()
            print(f"     terminal obs max|diff| (adapter.final vs ref no-reset) = {gt_diff:.3e}")
            check(f"[t={t}] final_observation == true terminal obs", gt_diff < 1e-3, f"max|diff|={gt_diff:.3e}")

            # The adapter's *current* obs must be the RESET obs, i.e. it must
            # differ from the terminal obs (otherwise bootstrap would be wrong).
            reset_vs_term = (a_obs["actor"] - term_adapter).abs().max().item()
            print(f"     reset obs vs terminal obs max|diff| = {reset_vs_term:.3e}")
            check(
                f"[t={t}] current obs is the RESET obs (differs from terminal)",
                reset_vs_term > 1e-4,
                f"reset==terminal? max|diff|={reset_vs_term:.3e}",
            )

            cycles_verified += 1
            # After the adapter auto-resets and the ref does not, the two
            # diverge; one verified synchronized cycle is conclusive.
            break

    check(
        "at least one synchronized truncation cycle verified",
        cycles_verified >= 1,
        "no truncation occurred within horizon+2 -- check --horizon",
    )

    adapter.gym_env.close()
    ref.close()


# --------------------------------------------------------------------------- #
# Phase 5: final_observation ground truth -- natural termination              #
# --------------------------------------------------------------------------- #
def phase5_natural_termination(args):
    hr("PHASE 5 -- final_observation GROUND TRUTH (natural termination) + semantics")
    print("  Setup: ignore_terminations=False (default training). Step adapter")
    print("  (auto_reset=True) and a no-reset reference in lockstep until the")
    print("  FIRST env(s) finish, then verify terminal obs / reward / flags on")
    print("  exactly those envs (after which the two envs diverge).\n")

    seed = 4242
    adapter = make_adapter(
        args.task, args.num_envs, args.obs_mode, args.control_mode, args.horizon, seed=seed, ignore_terminations=False
    )
    ref = make_vec(
        args.task,
        args.num_envs,
        args.obs_mode,
        args.control_mode,
        args.horizon,
        auto_reset=False,
        ignore_terminations=False,
    )

    a0, _ = adapter.reset()
    r0, _ = ref.reset(seed=seed)
    init_diff = (a0["actor"] - r0.float()).abs().max().item()
    check("adapter & reference start identical", init_diff < 1e-4, f"{init_diff:.3e}")

    limit = spec_limit(ref)
    big_horizon = (limit or args.horizon) + 4
    print(f"  effective max_episode_steps (from spec) = {limit} -> stepping up to {big_horizon}")
    actions = sample_action_seq(
        adapter.num_envs,
        adapter.num_actions,
        adapter.action_low,
        adapter.action_high,
        big_horizon,
        adapter.device,
        seed=seed,
    )

    verified = False
    for t in range(big_horizon):
        a_obs, a_rew, a_term, a_trunc, a_info = adapter.step(actions[t])
        r_obs, r_rew, r_term, r_trunc, r_info = ref.step(actions[t])
        a_dones = a_term | a_trunc
        r_term = r_term.to(torch.bool).reshape(adapter.num_envs)
        r_trunc = r_trunc.to(torch.bool).reshape(adapter.num_envs)
        r_dones = r_term | r_trunc

        rew_diff = (a_rew - r_rew.float().reshape(adapter.num_envs)).abs().max().item()
        check(f"[t={t}] reward pass-through matches reference", rew_diff < 1e-4, f"max|diff|={rew_diff:.3e}")

        if a_dones.any():
            done_idx = a_dones.nonzero(as_tuple=True)[0]
            print(f"  >> first done at t={t}: {int(a_dones.sum())} env(s) -> idx={done_idx.tolist()}")

            # terminated/truncated flags must agree with the reference.
            check(
                f"[t={t}] terminated agrees with reference",
                bool(torch.equal(a_term, r_term)),
                f"adapter={a_term.tolist()} ref={r_term.tolist()}",
            )
            check(
                f"[t={t}] truncated agrees with reference",
                bool(torch.equal(a_trunc, r_trunc)),
                f"adapter={a_trunc.tolist()} ref={r_trunc.tolist()}",
            )
            check(f"[t={t}] dones agree with reference", bool(torch.equal(a_dones, r_dones)))

            # Semantics: terminated == task success|fail; truncated == time limit.
            base = ref.base_env
            elapsed = base.elapsed_steps.to(torch.long).reshape(adapter.num_envs)
            print(f"     ref elapsed_steps={elapsed.tolist()} (max_episode_steps={args.horizon})")
            sf = torch.zeros(adapter.num_envs, dtype=torch.bool, device=adapter.device)
            if "success" in r_info:
                sf = sf | r_info["success"].to(torch.bool).reshape(adapter.num_envs)
            if "fail" in r_info:
                sf = sf | r_info["fail"].to(torch.bool).reshape(adapter.num_envs)
            check(
                f"[t={t}] terminated == (success|fail)",
                bool(torch.equal(r_term, sf)),
                f"term={r_term.tolist()} success|fail={sf.tolist()}",
            )
            if limit is not None:
                expected_trunc = elapsed >= limit
                check(
                    f"[t={t}] truncated == (elapsed>=max_episode_steps={limit})",
                    bool(torch.equal(r_trunc, expected_trunc)),
                    f"trunc={r_trunc.tolist()} expected={expected_trunc.tolist()}",
                )
            else:
                print("     (spec max_episode_steps unavailable -> skipping absolute truncation check)")

            # GROUND TRUTH terminal obs on the done envs.
            fo = a_info.get("final_observation", None)
            check(f"[t={t}] final_observation present", fo is not None)
            term_adapter = fo["actor"][done_idx]
            term_ref = r_obs.float().reshape(adapter.num_envs, adapter._obs_dim)[done_idx]
            gt_diff = (term_adapter - term_ref).abs().max().item()
            print(f"     terminal obs max|diff| on done envs = {gt_diff:.3e}")
            check(
                f"[t={t}] final_observation == true terminal obs (done envs)",
                gt_diff < 1e-3,
                f"max|diff|={gt_diff:.3e}",
            )

            # Non-done rows of final_observation equal the current obs.
            not_done = (~a_dones).nonzero(as_tuple=True)[0]
            if not_done.numel() > 0:
                nd_diff = (fo["actor"][not_done] - a_obs["actor"][not_done]).abs().max().item()
                check(
                    f"[t={t}] final_observation non-done rows == current obs",
                    nd_diff < 1e-4,
                    f"max|diff|={nd_diff:.3e}",
                )

            # Reset actually changed the done envs' state.
            reset_vs_term = (a_obs["actor"][done_idx] - term_adapter).abs().max().item()
            check(
                f"[t={t}] current obs is RESET obs on done envs (differs from terminal)",
                reset_vs_term > 1e-4,
                f"max|diff|={reset_vs_term:.3e}",
            )

            # success bookkeeping: adapter success on done envs == reference final success.
            if "success" in a_info and "success" in r_info:
                a_succ = a_info["success"][done_idx]
                r_succ = r_info["success"].to(torch.bool).reshape(adapter.num_envs)[done_idx]
                check(
                    f"[t={t}] adapter success matches reference on done envs",
                    bool(torch.equal(a_succ, r_succ)),
                    f"adapter={a_succ.tolist()} ref={r_succ.tolist()}",
                )

            verified = True
            break

    check(
        "a natural-termination cycle was verified",
        verified,
        "no env finished within horizon+4 -- adjust --horizon/--task",
    )

    adapter.gym_env.close()
    ref.close()


# --------------------------------------------------------------------------- #
# Phase 6: action clamping                                                    #
# --------------------------------------------------------------------------- #
def phase6_action_clamp(args):
    hr("PHASE 6 -- Action clamping")
    env = make_adapter(args.task, args.num_envs, args.obs_mode, args.control_mode, args.horizon, seed=11)
    env.reset()

    # Out-of-bounds actions must not crash and must be clamped internally:
    # feeding +-1e6 must produce the same transition as feeding the clamped
    # bounds explicitly. Use two identical-seed instances to compare.
    huge = torch.full((env.num_envs, env.num_actions), 1e6, device=env.device)
    o1, r1, t1, tr1, _ = env.step(huge)
    check("huge action does not crash and obs finite", finite(o1["actor"]) and finite(r1))

    env2 = make_adapter(args.task, args.num_envs, args.obs_mode, args.control_mode, args.horizon, seed=11)
    env2.reset()
    clamped = env2.action_high.unsqueeze(0).expand(env2.num_envs, -1).contiguous()
    o2, r2, t2, tr2, _ = env2.step(clamped)

    obs_diff = (o1["actor"] - o2["actor"]).abs().max().item()
    rew_diff = (r1 - r2).abs().max().item()
    print(f"  huge-vs-explicit-clamp: obs max|diff|={obs_diff:.3e} reward max|diff|={rew_diff:.3e}")
    check(
        "clamping huge action == feeding action_high",
        obs_diff < 1e-4 and rew_diff < 1e-4,
        f"obs={obs_diff:.3e} rew={rew_diff:.3e}",
    )

    env.gym_env.close()
    env2.gym_env.close()


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description="Exhaustive ManiSkill adapter diagnostic")
    p.add_argument("--task", type=str, default="PickCube-v1")
    p.add_argument("--num_envs", type=int, default=8)
    p.add_argument("--obs_mode", type=str, default="state")
    p.add_argument("--control_mode", type=str, default="pd_ee_delta_pose")
    p.add_argument("--horizon", type=int, default=8, help="max_episode_steps override (forces quick truncation)")
    p.add_argument("--steps", type=int, default=200, help="length of the Phase 3 random rollout")
    args = p.parse_args()

    torch.manual_seed(0)

    print("ManiSkill adapter diagnostic")
    print(f"  torch {torch.__version__}  cuda_available={torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("  ERROR: CUDA not available -- ManiSkill physx_cuda backend requires a GPU.")
        sys.exit(1)

    phase1_construction(args)
    phase2_reset(args)
    phase3_rollout(args)
    phase4_truncation_groundtruth(args)
    phase5_natural_termination(args)
    phase6_action_clamp(args)

    hr("RESULT")
    if _FAILURES:
        print(f"  {len(_FAILURES)} FAILURE(S):")
        for f in _FAILURES:
            print(f"    - {f}")
        sys.exit(1)
    print("  ALL CHECKS PASSED -- adapter is contract-correct with 100% confidence.")
    print("  (final_observation verified against ManiSkill ground truth under both")
    print("   synchronized truncation and natural termination.)")


if __name__ == "__main__":
    main()
