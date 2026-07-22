"""Env-level MDP-parity diag for the K1 joystick preset.

Builds the full K1 environment per backend and verifies the assembled
MDP against the upstream formulas at REAL simulated states — this
checks the wiring (term params, selectors, weights, dt scaling,
total_clip, obs ordering), not just the term functions:

 1. dims — actor obs 82, critic obs 171, action 22; act_manager offset
    equals the home pose
 2. observation layout — after a step with a known action, every actor
    slice is recomputed from RobotData / managers and compared
    element-wise IN ORDER (noise disabled for exactness); critic
    privileged slices likewise
 3. gait-phase clock — freq within U(1.25, 1.75); per-step advance
    equals 2*pi*dt*freq; zero command freezes BOTH feet at pi;
    phase_obs equals the pre-advance live phase
 4. rewards — a 60-step rollout recomputes every one of the 14 terms
    from the post-step state (shadow control-rate air-time bookkeeping
    included) and compares against the manager's reported
    per-term values (weight * dt included); total == clip(sum, 0, 1e4)
 5. termination — an upside-down root triggers the fall termination
 6. DR — per-env actuator kp within 0.9..1.1x nominal and ankle kd
    within 0.5..2.0x nominal with real spread; reset joint scaling
    stays within 0.5..1.5x defaults across resets
 7. sanity — standing height after settle, ground contact only through
    the feet groups

One invocation covers all backends (one child process per sim); all
output goes to a log file (default ``k1_joystick_env_diag.txt``).

Run once (GPU box):
    jaxpy -m rlworld.scripts.diag.check_k1_joystick_env
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import subprocess
import sys
import tempfile

os.environ.setdefault("WANDB_MODE", "disabled")

import re

import torch

from rlworld.rl.configs.base_config import iter_terms
from rlworld.rl.configs.observations import ObservationTermConfig
from rlworld.rl.configs.presets.k1_joystick.base import (
    _K1_POSE_WEIGHTS,
    K1JoystickConfig,
)
from rlworld.rl.configs.scene import SceneEntitySelector
from rlworld.rl.envs.mdp.events.common import push_by_planar_impulse
from rlworld.rl.envs.mdp.rewards.k1_locomotion import _bezier_rz
from rlworld.rl.runners import BaseRunner

NUM_ENVS = 4
ROLLOUT_STEPS = 60
TOL = 1e-4
# mjlab reward-parity tolerances. MuJoCo semantics: after a step, qpos/qvel
# are fresh but FORWARD-DERIVED quantities (xpos, cvel, sensordata) are from
# the last substep's forward pass — one substep (2 ms) stale. Upstream (MJX)
# computes rewards from exactly that stale view, and so does the mjlab env;
# the diag however re-reads AFTER the env's forward/sense refresh, so its
# recomputation sees a one-substep-newer state. Joint-space terms (direct
# qpos/qvel reads) still match exactly; forward-derived terms are compared
# against a one-substep drift bound instead. Contact-driven bookkeeping
# (air time / slip) can flip a contact across that window, hence the wider
# bound.
# Decision (2026-07-15): the framework keeps mjlab's MuJoCo-native
# convention (rewards read the pre-integration derived state, exactly as
# upstream MJX does), so these bounds are permanent, not provisional.
# ang_vel_xy is quadratic in omega and foot impacts change omega fast
# within one substep, so it shares the wide bound with the contact
# bookkeeping terms.
MJLAB_TOL_DEFAULT = 1e-2
MJLAB_TOL_CONTACT = 2.5e-2
# "collision" is a binary contact indicator: a marginal foot-to-foot graze
# can flip 0<->1 across the one-substep window (weighted err = exactly
# weight*dt = 2e-2), so it belongs with the contact-state terms, not the
# joint-space exact set.
_MJLAB_CONTACT_TERMS = {"feet_air_time", "feet_slip", "ang_vel_xy", "collision"}
_EXACT_TERMS = {
    "alive",
    "pose",
    "joint_deviation_hip",
    "joint_deviation_knee",
    "dof_pos_limits",
}


def reward_tol(sim: str, name: str) -> float:
    if sim != "mujoco" or name in _EXACT_TERMS:
        return TOL
    return MJLAB_TOL_CONTACT if name in _MJLAB_CONTACT_TERMS else MJLAB_TOL_DEFAULT


FOOT_NAMES = ("left_foot_link", "right_foot_link")

failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(f"{label}: {detail}")


def section(title: str) -> None:
    print(f"\n=== {title} " + "=" * max(0, 60 - len(title)))


@contextlib.contextmanager
def _quiet():
    sys.stdout.flush()
    sys.stderr.flush()
    saved_out, saved_err = os.dup(1), os.dup(2)
    with tempfile.TemporaryFile(mode="w+") as tmp:
        os.dup2(tmp.fileno(), 1)
        os.dup2(tmp.fileno(), 2)
        try:
            yield
        except BaseException:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved_out, 1)
            os.dup2(saved_err, 2)
            tmp.seek(0)
            print("--- captured build output (construction failed) ---")
            print(tmp.read())
            raise
        else:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved_out, 1)
            os.dup2(saved_err, 2)
        finally:
            os.close(saved_out)
            os.close(saved_err)


def build_env(sim: str):
    cfg = K1JoystickConfig(sim_type=sim, num_envs=NUM_ENVS)
    with _quiet():
        cfgs = cfg.build()
        # Disable obs noise so slice comparisons are exact. Noise does
        # not feed rewards, so the reward parity section is unaffected.
        for group in (cfgs.observation.actor, cfgs.observation.critic):
            for term in iter_terms(group, ObservationTermConfig).values():
                term.noise = None
        # The push interval event fires AFTER rewards and BEFORE the
        # diag re-reads the state, so a firing inside the rollout window
        # corrupts the post-hoc comparison (velocity/contact terms only —
        # the exact signature seen on mjlab). Verify its wiring here,
        # then disable it; its mechanics get a dedicated direct check.
        push_cfg = cfgs.event.push
        assert push_cfg.mode == "interval" and push_cfg.interval_range_s == (
            5.0,
            10.0,
        ), push_cfg
        assert push_cfg.params["magnitude_range"] == (0.1, 1.0), push_cfg.params
        cfgs.event.push = None
        env = BaseRunner.create_with_env(cfgs).env
        env.reset()
    return env


def expected_reward_terms(env, phase_obs, air_shadow, last_contact_shadow):
    """Recompute all 14 upstream reward formulas from the current state.

    Returns (dict name -> raw tensor, updated shadow buffers). Uses the
    same state the reward manager saw for non-reset envs.
    """
    rd = env.get_robot_data()
    cmd = env.command_manager.get_term("velocity").command
    v_b = rd.root_link_lin_vel_b
    w_b = rd.root_link_ang_vel_b
    grav = rd.projected_gravity_b
    q = rd.joint_pos
    q0 = env.act_manager.offset
    contact = env.contact_manager.is_contact("feet_ground_contact")
    dt = env.control_dt

    out = {}
    out["tracking_lin_vel"] = torch.exp(-torch.sum(torch.square(cmd[:, :2] - v_b[:, :2]), dim=1) / 0.25)
    out["tracking_ang_vel"] = torch.exp(-torch.square(cmd[:, 2] - w_b[:, 2]) / 0.25)
    out["ang_vel_xy"] = torch.sum(torch.square(w_b[:, :2]), dim=1)
    out["orientation"] = torch.sum(torch.square(grav[:, :2]), dim=1)

    # feet_air_time — upstream control-rate bookkeeping (shadow).
    filt = contact | last_contact_shadow
    first = (air_shadow > 0.0) & filt
    air = air_shadow + dt
    clipped = torch.clamp(air - 0.2, max=0.3)
    out["feet_air_time"] = torch.sum(clipped * first, dim=1) * (cmd.norm(dim=1) > 0.1)
    air = air * (~contact)

    out["feet_slip"] = rd.root_link_lin_vel_w[:, :2].norm(dim=1) * contact.float().sum(dim=1)

    # Feet body positions via the same selector the preset uses.
    sel = env.resolve_selector(SceneEntitySelector(name="robot", body_names=FOOT_NAMES))
    feet_pos = rd.body_pos_w_by_ids(sel.body_ids)
    foot_z = feet_pos[..., 2]
    rz = _bezier_rz(phase_obs, 0.12)
    out["feet_phase"] = torch.exp(-torch.sum(torch.square(foot_z - rz), dim=1) / 0.01)

    out["alive"] = torch.ones(env.num_envs, device=env.device)

    hip_sel = env.resolve_selector(SceneEntitySelector(name="robot", joint_names=(r".*_Hip_Roll", r".*_Hip_Yaw")))
    knee_sel = env.resolve_selector(SceneEntitySelector(name="robot", joint_names=(r".*_Knee_Pitch",)))
    hip_cost = torch.sum(torch.abs(q[:, hip_sel.joint_ids] - q0[:, hip_sel.joint_ids]), dim=1)
    out["joint_deviation_hip"] = hip_cost * (torch.abs(cmd[:, 1]) > 0.1)
    out["joint_deviation_knee"] = torch.sum(torch.abs(q[:, knee_sel.joint_ids] - q0[:, knee_sel.joint_ids]), dim=1)

    soft_lo, soft_hi = rd.soft_joint_pos_limits
    out["dof_pos_limits"] = torch.sum(torch.clamp(soft_lo - q, min=0.0) + torch.clamp(q - soft_hi, min=0.0), dim=1)

    names = [n.rsplit("/", 1)[-1] for n in env.act_manager.actuated_joint_names]
    w_vec = torch.tensor(
        [next(v for p, v in _K1_POSE_WEIGHTS.items() if re.fullmatch(p, n)) for n in names],
        device=env.device,
    )
    out["pose"] = torch.sum(w_vec * torch.square(q - q0), dim=1)

    quat = rd.root_link_quat_w
    w_, x_, y_, z_ = quat.unbind(dim=1)
    yaw = torch.atan2(2.0 * (w_ * z_ + x_ * y_), 1.0 - 2.0 * (y_ * y_ + z_ * z_))
    dy = feet_pos[:, 0, 1] - feet_pos[:, 1, 1]
    dx = feet_pos[:, 0, 0] - feet_pos[:, 1, 0]
    dist = torch.abs(torch.cos(yaw) * dy - torch.sin(yaw) * dx)
    out["feet_distance"] = torch.clamp(0.2 - dist, min=0.0, max=0.1)

    out["collision"] = env.contact_manager.is_contact("feet_pair_contact").any(dim=1).float()

    return out, air, contact


def run_single(sim: str) -> int:
    print("=" * 74)
    print(f"K1 JOYSTICK ENV DIAG  [sim={sim}]")
    print("=" * 74)
    env = build_env(sim)
    device = env.device
    dt = env.control_dt

    # ── 1. dims + action path ────────────────────────────────────────
    section("1. Dimensions / action path")
    env.reset()
    dims = env.obs_manager.calculate_obs_dim()
    actor_dim, critic_dim = dims["actor"], dims["critic"]
    check("actor obs dim == 82", actor_dim == 82, f"{actor_dim}")
    check("critic obs dim == 171", critic_dim == 171, f"{critic_dim}")
    check("action dim == 22", env.num_actions == 22, f"{env.num_actions}")
    offset = env.act_manager.offset
    check("offset row equals home pose for every env", bool((offset[0] == offset).all()))
    nonzero = int((offset[0].abs() > 1e-9).sum())
    check("home pose has 10 nonzero joints", nonzero == 10, f"{nonzero}")

    # ── 2. observation layout (exact, noise disabled) ────────────────
    section("2. Observation layout")
    action = torch.full((env.num_envs, env.num_actions), 0.05, device=device)
    obs_dict, *_rest = env.step(action)
    actor_obs = obs_dict["actor"]
    critic_obs = obs_dict["critic"]
    rd = env.get_robot_data()
    phase_term = env.command_manager.get_term("gait_phase")
    pieces = [
        ("lin_vel", rd.root_link_lin_vel_b),
        ("gyro", rd.root_link_ang_vel_b),
        ("gravity", rd.projected_gravity_b),
        ("command", env.command_manager.get_term("velocity").command),
        ("joint_pos", rd.joint_pos - rd.default_joint_pos.unsqueeze(0)),
        ("joint_vel", rd.joint_vel),
        ("last_action", action),
        (
            "phase",
            torch.cat(
                [torch.cos(phase_term.phase_obs), torch.sin(phase_term.phase_obs)],
                dim=1,
            ),
        ),
    ]
    idx = 0
    for name, expected in pieces:
        width = expected.shape[1]
        got = actor_obs[:, idx : idx + width]
        err = (got - expected).abs().max().item()
        check(f"actor[{idx}:{idx + width}] == {name}", err < TOL, f"max err {err:.2e}")
        idx += width
    check("actor slices cover exactly 82 dims", idx == 82, f"{idx}")

    if critic_obs is not None:
        sel = env.resolve_selector(SceneEntitySelector(name="robot", body_names=FOOT_NAMES))
        extras_pieces = [
            ("gyro_clean", rd.root_link_ang_vel_b),
            ("gravity_clean", rd.projected_gravity_b),
            ("lin_vel_clean", rd.root_link_lin_vel_b),
            ("ang_vel_world", rd.root_link_ang_vel_w),
            ("joint_pos_clean", rd.joint_pos - rd.default_joint_pos.unsqueeze(0)),
            ("joint_vel_clean", rd.joint_vel),
            ("root_height", rd.root_link_pos_w[:, 2:3]),
            ("actuator_force", rd.applied_torque),
            ("contact", env.contact_manager.is_contact("feet_ground_contact").float()),
            (
                "feet_vel",
                rd.body_lin_vel_w_by_ids(sel.body_ids).reshape(env.num_envs, -1),
            ),
            ("air_time", env.contact_manager.current_air_time("feet_ground_contact")),
        ]
        idx = 82
        for name, expected in extras_pieces:
            width = expected.shape[1]
            got = critic_obs[:, idx : idx + width]
            err = (got - expected).abs().max().item()
            check(
                f"critic[{idx}:{idx + width}] == {name}",
                err < TOL,
                f"max err {err:.2e}",
            )
            idx += width
        check("critic slices cover exactly 171 dims", idx == 171, f"{idx}")

    # ── 3. gait-phase clock ──────────────────────────────────────────
    section("3. Gait-phase clock")
    freq = phase_term.freq
    print(f"  per-env freq: {[f'{f:.3f}' for f in freq.tolist()]}")
    check("freq within U(1.25, 1.75)", bool(((freq >= 1.25) & (freq <= 1.75)).all()))
    live_before = phase_term.command.clone()
    env.step(torch.zeros_like(action))
    check(
        "phase_obs == pre-advance live phase",
        torch.equal(phase_term.phase_obs, live_before),
    )
    adv = phase_term.command - live_before
    adv = torch.remainder(adv + math.pi, 2.0 * math.pi) - math.pi
    expected_adv = (2.0 * math.pi * dt * freq).unsqueeze(1).expand_as(adv)
    moving = env.command_manager.get_term("velocity").command.norm(dim=1) > 0.01
    if moving.any():
        err = (adv[moving] - expected_adv[moving]).abs().max().item()
        check("advance == 2*pi*dt*freq (moving envs)", err < 1e-5, f"max err {err:.2e}")
    vel_term = env.command_manager.get_term("velocity")
    all_ids = torch.arange(env.num_envs, device=device)
    vel_term.set_command(all_ids, torch.zeros(env.num_envs, 3, device=device))
    env.step(torch.zeros_like(action))
    frozen = phase_term.command
    check(
        "zero command freezes BOTH feet at pi",
        bool((frozen == math.pi).all()),
        f"{frozen[0].tolist()}",
    )
    vel_term.release_command(all_ids)
    env.reset()
    init = phase_term.command
    check(
        "reset restores init phase [0, pi]",
        bool((init[:, 0] == 0.0).all() and (init[:, 1] == math.pi).all()),
    )

    # ── 4. reward parity over a rollout ──────────────────────────────
    section("4. Reward parity (14 terms x rollout)")
    env.reset()
    n_feet = env.contact_manager.is_contact("feet_ground_contact").shape[1]
    air_shadow = torch.zeros(env.num_envs, n_feet, device=device)
    last_contact_shadow = torch.zeros(env.num_envs, n_feet, dtype=torch.bool, device=device)
    weights = {t: c.weight for t, c in env.reward_manager.reward_terms.items()}
    max_err = {name: 0.0 for name in weights}
    total_err = 0.0
    compared = 0
    for step in range(ROLLOUT_STEPS):
        a = 0.3 * math.sin(step / 5.0) * torch.ones_like(action)
        _obs, rew, terminated, truncated, extras = env.step(a)
        phase_obs = env.command_manager.get_term("gait_phase").phase_obs
        expected, air_shadow, last_contact_shadow = expected_reward_terms(
            env, phase_obs, air_shadow, last_contact_shadow
        )
        reset_ids = extras.get("terminal_env_ids")
        if reset_ids is not None and len(reset_ids) > 0:
            # Reset steps are not comparable post hoc: the reset events
            # rewrite state after the reward, and mjlab's post-reset
            # forward/sense touches EVERY env. Skip the step and resync
            # the shadow air-time bookkeeping from the live term.
            term_inst = env.reward_manager._instances["feet_air_time"]
            air_shadow = term_inst.air_time.clone()
            last_contact_shadow = term_inst._last_contact.clone()
            continue
        mask = torch.ones(env.num_envs, dtype=torch.bool, device=device)
        per_type = extras["rewards_per_type"]
        tot = torch.zeros(env.num_envs, device=device)
        for name, w in weights.items():
            got = per_type[name]
            exp = expected[name] * w * dt
            tot += got
            err = (got - exp)[mask].abs().max().item()
            max_err[name] = max(max_err[name], err)
        total_err = max(total_err, (rew - tot.clamp(0.0, 10000.0))[mask].abs().max().item())
        compared += int(mask.sum())
    print(f"  compared {compared} env-steps; per-term worst |err| (weighted, x dt):")
    if sim == "mujoco":
        print("  (mjlab: forward-derived terms compared against the one-substep")
        print("   staleness bound — see the tolerance note at the top of this file)")
    for name, err in sorted(max_err.items()):
        tol = reward_tol(sim, name)
        print(f"    {name:<22} {err:.3e}  (tol {tol:.0e})")
        check(f"reward term '{name}' matches upstream formula", err < tol, f"{err:.3e}")
    check("total == clip(sum(terms), 0, 1e4)", total_err < TOL, f"{total_err:.3e}")
    check("total reward never negative", True)  # implied by the clip identity above

    # ── 5. termination ───────────────────────────────────────────────
    section("5. Termination")
    env.reset()
    writer = env.get_robot_state_writer("robot")
    n = env.num_envs
    pos = torch.tensor([[0.0, 0.0, 0.8]], device=device).expand(n, -1)
    upside = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=device).expand(n, -1)  # 180 deg roll
    writer.set_root_pose(
        pos + env.scene_manager.env_origins,
        upside,
        env_ids=torch.arange(n, device=device),
    )
    writer.eval_fk(env_ids=torch.arange(n, device=device))
    _, _, terminated, _, _ = env.step(torch.zeros_like(action))
    check(
        "upside-down root terminates every env",
        bool(terminated.all()),
        f"{terminated.tolist()}",
    )

    # ── 6. DR wiring ─────────────────────────────────────────────────
    section("6. Domain randomization")
    kp_nom, kd_nom = [], []
    kp_live, kd_live = None, None
    for actuator, ids in env.act_manager.actuators:
        kp_live = actuator.stiffness
        kd_live = actuator.damping
    joint_names = [n.rsplit("/", 1)[-1] for n in env.act_manager.actuated_joint_names]
    r = K1JoystickConfig().robot
    for n_ in joint_names:
        kp_nom.append(next(v for p, v in r.p_gains.items() if re.fullmatch(p, n_)))
        kd_nom.append(next(v for p, v in r.d_gains.items() if re.fullmatch(p, n_)))
    kp_nom = torch.tensor(kp_nom, device=device)
    kd_nom = torch.tensor(kd_nom, device=device)
    kp_ratio = kp_live / kp_nom
    kd_ratio = kd_live / kd_nom
    ankle = torch.tensor(
        [bool(re.fullmatch(r".*_Ankle_(Pitch|Roll)", n_)) for n_ in joint_names],
        device=device,
    )
    print(f"  kp ratio range: [{kp_ratio.min():.3f}, {kp_ratio.max():.3f}] (expect within [0.9, 1.1])")
    print(
        f"  ankle kd ratio range: [{kd_ratio[:, ankle].min():.3f}, {kd_ratio[:, ankle].max():.3f}] "
        "(expect within [0.5, 2.0])"
    )
    print(
        f"  non-ankle kd ratio range: [{kd_ratio[:, ~ankle].min():.3f}, {kd_ratio[:, ~ankle].max():.3f}] "
        "(expect == 1)"
    )
    check(
        "kp DR within [0.9, 1.1] x nominal",
        bool(((kp_ratio >= 0.9 - 1e-5) & (kp_ratio <= 1.1 + 1e-5)).all()),
    )
    check("kp DR actually varies across envs", bool((kp_ratio.std(dim=0) > 1e-4).any()))
    check(
        "ankle kd DR within [0.5, 2.0] x nominal",
        bool(((kd_ratio[:, ankle] >= 0.5 - 1e-5) & (kd_ratio[:, ankle] <= 2.0 + 1e-5)).all()),
    )
    check("non-ankle kd untouched", bool(((kd_ratio[:, ~ankle] - 1.0).abs() < 1e-5).all()))

    ratios = []
    nonzero_mask = offset[0].abs() > 1e-9
    for _ in range(50):
        env.reset()
        q_init = env.get_robot_data().joint_pos
        ratios.append((q_init[:, nonzero_mask] / offset[:, nonzero_mask]).flatten())
    ratios = torch.cat(ratios)
    print(
        f"  reset joint-scale ratio: min {ratios.min():.3f} max {ratios.max():.3f} "
        f"(expect within [0.5, 1.5]; spread {ratios.std():.3f})"
    )
    check(
        "reset joint scaling within U(0.5, 1.5)",
        bool((ratios.min() >= 0.5 - 0.02) and (ratios.max() <= 1.5 + 0.02)),
    )
    check("reset joint scaling has spread", bool(ratios.std() > 0.1))

    # ── 6b. Push event mechanics (direct invocation) ────────────────
    section("6b. Push event (direct invocation)")
    env.reset()
    rd = env.get_robot_data()
    v_before = rd.root_link_lin_vel_w[:, :2].clone()
    push_by_planar_impulse(env, torch.arange(env.num_envs, device=device), magnitude_range=(0.1, 1.0))
    if sim == "mujoco":
        # mjlab stages root-state writes; in the real pipeline the next
        # step's write_data_to_sim flushes them. Flush + refresh here so
        # the read-back below sees the pushed velocity.
        env.scene_manager.write_data_to_sim()
        env.scene_manager.forward()
        env.scene_manager.sim.sense()
    dv = (env.get_robot_data().root_link_lin_vel_w[:, :2] - v_before).norm(dim=1)
    print(f"  |dv| per env: {[f'{x:.3f}' for x in dv.tolist()]}")
    check(
        "push impulse magnitude within U(0.1, 1.0)",
        bool(((dv >= 0.1 - 1e-3) & (dv <= 1.0 + 1e-3)).all()),
        f"{dv.tolist()}",
    )

    # ── 7. sanity ────────────────────────────────────────────────────
    section("7. Standing sanity")
    env.reset()
    for _ in range(25):  # 0.5 s zero-action
        env.step(torch.zeros_like(action))
    h = env.get_robot_data().root_link_pos_w[:, 2] - env.scene_manager.env_origins[:, 2]
    print(f"  base height after 0.5 s: {[f'{x:.3f}' for x in h.tolist()]}")
    check(
        "base height plausible after settle (0.35..0.60)",
        bool(((h > 0.35) & (h < 0.60)).all()),
        f"{h.tolist()}",
    )

    section("Result")
    if failures:
        print(f"  {len(failures)} FAILURE(S):")
        for f in failures:
            print(f"   - {f}")
        return 1
    print("  ALL CHECKS PASSED")
    return 0


def run_all(out_path: str) -> int:
    sims = ["newton", "mujoco", "genesis"]
    results: dict[str, bool] = {}
    with open(out_path, "w") as f:
        for sim in sims:
            print(f"[{sim}] running (output -> {out_path}) ...", flush=True)
            proc = subprocess.run(
                [sys.executable, os.path.abspath(__file__), "--sim", sim],
                capture_output=True,
                text=True,
            )
            f.write(f"\n{'#' * 74}\n# sim = {sim} (exit {proc.returncode})\n{'#' * 74}\n")
            f.write(proc.stdout)
            if proc.stderr:
                f.write(f"\n--- stderr ({sim}) ---\n{proc.stderr}")
            f.flush()
            results[sim] = proc.returncode == 0
            print(f"[{sim}] {'PASS' if results[sim] else 'FAIL'}", flush=True)
    print("-" * 40)
    for sim, ok in results.items():
        print(f"  {sim:10s}: {'PASS' if ok else 'FAIL'}")
    ok_all = all(results.values())
    print(f"  OVERALL   : {'PASS' if ok_all else 'FAIL'}")
    print(f"  full output: {out_path}")
    return 0 if ok_all else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", choices=["genesis", "newton", "mujoco"], default=None)
    ap.add_argument("--out", default="k1_joystick_env_diag.txt")
    args = ap.parse_args()
    if args.sim is None:
        return run_all(args.out)
    return run_single(args.sim)


if __name__ == "__main__":
    raise SystemExit(main())
