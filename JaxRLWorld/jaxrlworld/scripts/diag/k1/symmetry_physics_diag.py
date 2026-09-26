"""Prove the K1 mirror operator against the simulator itself, one backend at a time.

The mirror data augmentation feeds the policy ``(L o, K a)`` for every
sample ``(o, a)`` and treats it as a real transition. That is only true if
``L`` and ``K`` are the observation and action images of the robot's actual
left/right symmetry. This script checks that with the physics as oracle
instead of by inspecting index tables:

  A. Structure. Every actor and critic term has a mirror rule (the build
     raises otherwise), the operators are involutions, and the joint
     pairing is printed for the eye.
  B. Kinematic oracle. Random states ``s`` are written into envs ``[:N]``
     and their physical mirrors ``L s`` (y flipped, orientation reflected,
     joints swapped with roll/yaw negated, command mirrored) into envs
     ``[N:]``. After forward kinematics the observation of env ``N+i`` must
     equal the mirror operator applied to env ``i``'s observation, every
     block, to float32 precision.
     Contact-derived blocks are skipped here: without a physics step the
     contact manager holds whatever the last step left, so they are not a
     function of the written state. They are covered by C.
  C. Dynamic oracle. From the home pose at rest on the floor, settled under
     zero action, envs ``[:N]`` receive a random action sequence and envs
     ``[N:]`` its mirror ``K a``. Every observation block and every reward
     term must stay mirror-equal, step by step, to within what the physics
     can reproduce at all: the same procedure is first run with exact
     COPIES in ``[N:]`` (same actions), and that "twin" disagreement -- env
     origins already place identical robots at different float32 world
     coordinates -- is the floor the mirror pairing is held to. This
     exercises the contact-dependent terms a kinematic write cannot reach
     and checks the reward set for left/right symmetry. 0/1 flags and step
     counters are compared by mismatch rate, since they flip on noise.

Two assets. The shipped ``k1.xml`` is not an exact mirror image of
itself (see ``symmetrize_k1_asset``: inertia 0.27%, forearm collision
cylinder 6.7 mm off), so its physics cannot be exactly mirror symmetric and
C can only report the residual that asymmetry produces. The proof of the
operators runs on the symmetrized twin (``--asset symmetrized``, the
default), where every pass criterion is held to float precision; the
source asset (``--asset source``) quantifies how far the augmentation's
premise holds on the plant that is actually trained.

Run one backend per process::

    jaxpy -m jaxrlworld.scripts.diag.k1.symmetry_physics_diag --sim mujoco
    jaxpy -m jaxrlworld.scripts.diag.k1.symmetry_physics_diag --sim mujoco --asset source
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from jaxrlworld.rl.algorithms.ppo.symmetry import build_mirror_spec
from jaxrlworld.rl.configs.common_config_classes import disable_corruption
from jaxrlworld.rl.configs.presets.k1_velocity.base import K1VelocityConfig
from jaxrlworld.rl.envs.mdp.terminations.k1_velocity import tilt_angle
from jaxrlworld.rl.runners.base_runner import BaseRunner
from jaxrlworld.rl.utils.quat_utils import quat_from_angle_axis_wxyz, quat_mul_wxyz
from jaxrlworld.scripts.diag.k1.symmetrize_asset import audit, write_symmetrized

# float32 kinematics agree to this after a mirror.
KINEMATIC_TOL = 1e-5
# Contact-derived observation terms: a function of the last physics step,
# not of a written state, so section B cannot test them.
CONTACT_TERM_FUNCS = {"foot_air_time", "foot_contact_indicator", "foot_contact_forces"}
# Section C. Two pairings run with identical inputs: "twin" (envs [N:] are
# exact copies of [:N], same actions) measures how reproducible the physics
# itself is across env slots -- env origins put the same robot at different
# world coordinates, so float32 rounding already differs -- and "mirror"
# (envs [N:] are the mirrors, actions K a) is the test. The mirror pairing is
# held to the twin pairing's floor: a mirror error is a symmetry failure
# only where it exceeds what identical copies already disagree by.
CONTINUOUS_FLOOR = 1e-4  # below this a mirror error is float32 noise, whatever the controls did
TWIN_RATIO = 10.0  # mirror error may exceed the control floor by this factor
REWARD_FLOOR = 1e-4
DISCRETE_MISMATCH_RATE = 0.05  # 0/1 contact flags and one-step air-time counters flip on noise
# A contact solver iterates over contacts in a fixed order, so from an exactly
# symmetric state it returns a slightly asymmetric solution. That
# self-asymmetry, mirror(obs) - obs at the end of settling, is a property of
# the simulator and the seed the scripted rollout then amplifies. It has to
# be small in absolute terms (an operator error shows here as O(0.1..1)),
# and the rollout is then held against a "perturbed twin": copies offset by
# that same magnitude, which measures the amplification alone.
# Both are also allowed up to TWIN_RATIO times the twin pairing's own
# settle-end disagreement: a solver whose identical copies already differ
# while standing still (mjlab's contact pipeline does, from atomics in its
# collision counting) cannot show a self-asymmetry below that noise.
SELF_ASYMMETRY_KINEMATIC_TOL = 1e-2
SELF_ASYMMETRY_CONTACT_TOL = 0.5  # log1p-scaled force
# Per-foot terms that are 0/1 flags or step counters: compared by mismatch rate.
DISCRETE_TERM_FUNCS = {"foot_contact_indicator", "foot_air_time"}
# On the source asset the self-asymmetry includes the asset's own left/right
# asymmetry and is reported, not judged; the amplification-floor checks
# apply as on the twin, with the perturbed twin seeded from that larger value.
SETTLE_STEPS = 100
HORIZON = 20
ACTION_AMPLITUDE = 0.15
# The training gains cannot hold the home pose without a policy: the ankle's
# 50 N m/rad is below the gravitational stiffness of a 20 kg body at 0.44 m
# (about 86 N m/rad), so a policy-less robot tips over within 1.5 s and the
# settle never reaches an equilibrium. The leg groups are stiffened here,
# kd scaled to keep the damping ratio. Mirror symmetry of the operators and
# of the physics does not depend on the gain values, and the gains are
# themselves left/right symmetric, so nothing under test changes. Arm and
# head gains stay: the arm kd sits near the explicit-damping stability limit.
LEG_GAIN_SCALE = 5.0
# Home keyframe base height plus the sole's drop below the link origin, plus
# a hair: the feet start 0.2 mm above the floor, at rest, so no impact seeds
# the rollout.
SOLE_BELOW_ORIGIN = 0.0058
START_HEIGHT = 0.5125 + SOLE_BELOW_ORIGIN + 0.0002


class Checker:
    def __init__(self) -> None:
        self.fails: list[str] = []

    def __call__(self, name: str, ok: bool, detail: str = "") -> bool:
        ok = bool(ok)
        print(f"  [{'OK' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
        if not ok:
            self.fails.append(name)
        return ok


def leaf(name: str) -> str:
    return name.rsplit("/", 1)[-1]


# ── env ──────────────────────────────────────────────────────────────


def build_env(sim: str, num_envs: int, mjcf_path: str):
    cfg = K1VelocityConfig(sim_type=sim, num_envs=num_envs)
    cfg.robot.mjcf_path = mjcf_path
    leg = lambda pattern: any(k in pattern for k in ("Hip", "Knee", "Ankle"))  # noqa: E731
    cfg.robot.p_gains = {k: v * (LEG_GAIN_SCALE if leg(k) else 1.0) for k, v in cfg.robot.p_gains.items()}
    cfg.robot.d_gains = {k: v * (LEG_GAIN_SCALE**0.5 if leg(k) else 1.0) for k, v in cfg.robot.d_gains.items()}
    cfg.action_delay_min = 0
    cfg.action_delay_max = 0
    # Commands are written by this script and must stay put: no heading
    # regeneration from the (mirrored) yaw, no resampling inside the horizon.
    cfg.heading_command = False
    cfg.command_resampling_time_range = (1e6, 1e6)
    cfgs = cfg.build()
    for name in list(vars(cfgs.event)):
        if name.startswith("dr_") or name == "push":
            delattr(cfgs.event, name)
    disable_corruption(cfgs.observation)
    env = BaseRunner._create_env_from_config(cfgs)
    env.reset()
    return env


class Mirror:
    """Torch-side mirror of states and observations, from the JAX MirrorSpec."""

    def __init__(self, spec, device):
        t = lambda x, dt: torch.as_tensor(np.asarray(x), dtype=dt, device=device)  # noqa: E731
        self.actor_perm, self.actor_sign = t(spec.actor_perm, torch.long), t(spec.actor_sign, torch.float32)
        self.critic_perm, self.critic_sign = t(spec.critic_perm, torch.long), t(spec.critic_sign, torch.float32)
        self.joint_perm, self.joint_sign = t(spec.action_perm, torch.long), t(spec.action_sign, torch.float32)
        self.y_flip = torch.tensor([1.0, -1.0, 1.0], device=device)
        self.rollyaw_flip = torch.tensor([-1.0, 1.0, -1.0], device=device)

    def actor(self, o):
        return o[..., self.actor_perm] * self.actor_sign

    def critic(self, o):
        return o[..., self.critic_perm] * self.critic_sign

    def joints(self, q):
        return q[..., self.joint_perm] * self.joint_sign

    def quat_wxyz(self, q):
        # Reflection across the y=0 plane conjugates the rotation: R -> M R M
        # with M = diag(1,-1,1), which on a unit quaternion is [w, -x, y, -z].
        return torch.stack([q[..., 0], -q[..., 1], q[..., 2], -q[..., 3]], dim=-1)

    def command(self, c):
        return c * torch.tensor([1.0, -1.0, -1.0], device=c.device)


# ── state injection ───────────────────────────────────────────────────


def random_states(n: int, home: torch.Tensor, gen: torch.Generator, device, standing: bool) -> dict:
    """N base states. ``standing``: home pose, feet at the ground, at rest."""
    u = lambda *shape, lo=-1.0, hi=1.0: (torch.rand(*shape, generator=gen, device="cpu") * (hi - lo) + lo).to(device)  # noqa: E731
    if standing:
        rel_pos = torch.zeros(n, 3, device=device)
        rel_pos[:, 2] = START_HEIGHT
        rpy = torch.zeros(n, 3, device=device)
        lin = torch.zeros(n, 3, device=device)
        ang = torch.zeros(n, 3, device=device)
        dq = torch.zeros(n, home.shape[-1], device=device)
        qd = torch.zeros(n, home.shape[-1], device=device)
    else:
        rel_pos = torch.stack([u(n, lo=-0.3, hi=0.3), u(n, lo=-0.3, hi=0.3), u(n, lo=0.45, hi=0.65)], 1)
        rpy = u(n, 3, lo=-0.3, hi=0.3)
        lin = u(n, 3, lo=-0.5, hi=0.5)
        ang = u(n, 3, lo=-1.0, hi=1.0)
        dq = u(n, home.shape[-1], lo=-0.3, hi=0.3)
        qd = u(n, home.shape[-1], lo=-2.0, hi=2.0)
    cmd = u(n, 3, lo=-1.0, hi=1.0)
    return dict(rel_pos=rel_pos, rpy=rpy, lin=lin, ang=ang, q=home + dq, qd=qd, cmd=cmd)


def rpy_to_quat_wxyz(rpy: torch.Tensor) -> torch.Tensor:
    dev = rpy.device
    ax = torch.tensor((1.0, 0.0, 0.0), device=dev)
    ay = torch.tensor((0.0, 1.0, 0.0), device=dev)
    az = torch.tensor((0.0, 0.0, 1.0), device=dev)
    return quat_mul_wxyz(
        quat_mul_wxyz(quat_from_angle_axis_wxyz(rpy[:, 2], az), quat_from_angle_axis_wxyz(rpy[:, 1], ay)),
        quat_from_angle_axis_wxyz(rpy[:, 0], ax),
    )


def inject_pair(env, m: Mirror, st: dict, mirror_pairing: bool = True, perturb: torch.Tensor | None = None) -> None:
    """Envs [:N] <- s, envs [N:] <- L s, or an exact copy of s, or a copy offset by ``perturb``.

    ``perturb`` (N, dof) is added to the copy's joint positions and velocities.
    """
    n = st["q"].shape[0]
    device = env.device
    origins = env.scene_manager.env_origins
    quat = rpy_to_quat_wxyz(st["rpy"])
    if mirror_pairing:
        pos = torch.cat([origins[:n] + st["rel_pos"], origins[n:] + st["rel_pos"] * m.y_flip], 0)
        quat2 = torch.cat([quat, m.quat_wxyz(quat)], 0)
        lin2 = torch.cat([st["lin"], st["lin"] * m.y_flip], 0)
        ang2 = torch.cat([st["ang"], st["ang"] * m.rollyaw_flip], 0)
        q2 = torch.cat([st["q"], m.joints(st["q"])], 0)
        qd2 = torch.cat([st["qd"], m.joints(st["qd"])], 0)
        cmd2 = torch.cat([st["cmd"], m.command(st["cmd"])], 0)
    else:
        delta = torch.zeros_like(st["q"]) if perturb is None else perturb
        pos = torch.cat([origins[:n] + st["rel_pos"], origins[n:] + st["rel_pos"]], 0)
        quat2 = torch.cat([quat, quat], 0)
        lin2 = torch.cat([st["lin"], st["lin"]], 0)
        ang2 = torch.cat([st["ang"], st["ang"]], 0)
        q2 = torch.cat([st["q"], st["q"] + delta], 0)
        qd2 = torch.cat([st["qd"], st["qd"] + delta], 0)
        cmd2 = torch.cat([st["cmd"], st["cmd"]], 0)
    env_ids = torch.arange(2 * n, device=device)
    writer = env.get_robot_state_writer("robot")
    writer.set_root_pose(pos, quat2, env_ids=env_ids)
    writer.set_root_velocity(lin2, ang2, env_ids=env_ids)
    writer.set_dof_state(q2, qd2, env_ids=env_ids)
    writer.eval_fk(env_ids=env_ids)
    env._post_reset_forward()
    env._invalidate_cache()
    env.command_manager.set_commands(env_ids, velocity=cmd2)


# ── comparison ───────────────────────────────────────────────────────


def blocks(env, group: str) -> list[tuple[str, int, int]]:
    out, start = [], 0
    for name, _func, width in env.obs_manager.term_layout(group):
        out.append((name, start, start + width))
        start += width
    return out


def contact_blocks(env, group: str) -> set[str]:
    return {name for name, func, _ in env.obs_manager.term_layout(group) if func.__name__ in CONTACT_TERM_FUNCS}


def compare_group(env, m: Mirror, group: str, n: int, tol: float, chk: Checker, tag: str, show_control: bool) -> None:
    obs = env.obs_manager.obs_dict[group]
    got = obs[n:]
    want = (m.actor if group == "actor" else m.critic)(obs[:n])
    skip = contact_blocks(env, group)
    header = f"     {'block':28s} {'max|mirror(o_i) - o_(N+i)|':>28s}" + (
        f" {'control |o_i - o_(N+i)|':>24s}" if show_control else ""
    )
    print(header)
    for name, s, e in blocks(env, group):
        if name in skip:
            print(f"     {name:28s} {'(contact-derived: section C)':>28s}")
            continue
        d = float((want[:, s:e] - got[:, s:e]).abs().max())
        line = f"     {name:28s} {d:28.3e}"
        if show_control:
            line += f" {float((obs[:n, s:e] - got[:, s:e]).abs().max()):24.3e}"
        print(line)
        chk(f"{tag} {group}.{name} mirror-equal", d <= tol, f"max|d| {d:.3e} > {tol:.0e}" if d > tol else "")


# ── sections ─────────────────────────────────────────────────────────


def section_a_structure(env, spec, chk: Checker) -> None:
    print("\n=== A. operator structure ===")
    joints = [leaf(n) for n in env.act_manager.actuated_joint_names]
    ap, as_ = np.asarray(spec.action_perm), np.asarray(spec.action_sign)
    print("  joint pairing:")
    for i, n in enumerate(joints):
        print(f"    {i:2d} {n:22s} -> {ap[i]:2d} {joints[ap[i]]:22s} sign {as_[i]:+.0f}")
    chk(
        "every Left joint maps to its Right mate and back",
        all(
            joints[ap[i]] == n.replace("Left", "Right")
            if "Left" in n
            else joints[ap[i]] == n.replace("Right", "Left")
            if "Right" in n
            else ap[i] == i
            for i, n in enumerate(joints)
        ),
    )
    chk(
        "roll/yaw joints flip sign, pitch joints keep it",
        all((as_[i] < 0) == (("Roll" in n) or ("Yaw" in n)) for i, n in enumerate(joints)),
    )
    for name, perm, sign in (
        ("actor", spec.actor_perm, spec.actor_sign),
        ("critic", spec.critic_perm, spec.critic_sign),
        ("action", spec.action_perm, spec.action_sign),
    ):
        p, s = np.asarray(perm), np.asarray(sign)
        chk(
            f"{name} operator is an involution",
            bool((p[p] == np.arange(len(p))).all()) and bool(np.allclose(s * s[p], 1.0)),
            f"dim {len(p)}",
        )
    for group in ("actor", "critic"):
        names = [n for n, _, _ in blocks(env, group)]
        print(f"  {group} blocks ({env.obs_manager.obs_dict[group].shape[-1]}-D): {names}")


def section_b_kinematic(env, m: Mirror, chk: Checker, home: torch.Tensor, n: int, rounds: int) -> None:
    print("\n=== B. kinematic oracle: obs(L s) == L obs(s) ===")
    gen = torch.Generator().manual_seed(0)
    for r in range(rounds):
        env.reset()
        st = random_states(n, home, gen, env.device, standing=False)
        inject_pair(env, m, st)
        env.obs_manager.process_observations()
        print(f"  round {r} (random free-space states)")
        compare_group(env, m, "actor", n, KINEMATIC_TOL, chk, f"B{r}", show_control=(r == 0))
        compare_group(env, m, "critic", n, KINEMATIC_TOL, chk, f"B{r}", show_control=False)
    env.reset()


def discrete_blocks(env, group: str) -> set[str]:
    return {name for name, func, _ in env.obs_manager.term_layout(group) if func.__name__ in DISCRETE_TERM_FUNCS}


def _pair_errors(env, m: Mirror, n: int, mirror_pairing: bool) -> tuple[dict[str, float], float]:
    """Per-block error between envs [N:] and the (mirrored or copied) envs [:N].

    Continuous blocks: max abs difference. Discrete blocks: mismatch rate.
    Also the unmirrored control (max |o_i - o_(N+i)|), meaningful for the
    mirror pairing only.
    """
    errs: dict[str, float] = {}
    control = 0.0
    for group in ("actor", "critic"):
        obs = env.obs_manager.obs_dict[group]
        ref = obs[:n]
        if mirror_pairing:
            ref = (m.actor if group == "actor" else m.critic)(ref)
        disc = discrete_blocks(env, group)
        for name, s_, e_ in blocks(env, group):
            diff = (ref[:, s_:e_] - obs[n:, s_:e_]).abs()
            if name in disc:
                errs[f"{group}.{name}"] = float((diff > 1e-6).float().mean())
            else:
                errs[f"{group}.{name}"] = float(diff.max())
            control = max(control, float((obs[:n, s_:e_] - obs[n:, s_:e_]).abs().max()))
    return errs, control


def describe_reset(env, step: int, phase: str) -> str:
    """Which termination fired on which envs, with the state that fired it."""
    fired = {
        name: mask.nonzero(as_tuple=False).flatten().tolist()
        for name, mask in env.termination_manager.term_dones.items()
        if bool(mask.any())
    }
    rd = env.get_entity_data("robot")
    tilt_deg = torch.rad2deg(tilt_angle(env))
    height = rd.root_link_pos_w[:, 2] - env.scene_manager.env_origins[:, 2]
    return (
        f"{phase} step {step}: {int(env.reset_buf.sum())} envs reset; fired {fired}; "
        f"tilt deg min/mean/max {float(tilt_deg.min()):.1f}/{float(tilt_deg.mean()):.1f}/{float(tilt_deg.max()):.1f}; "
        f"base height min/mean/max {float(height.min()):.3f}/{float(height.mean()):.3f}/{float(height.max()):.3f}"
    )


def run_pairing(env, m: Mirror, home: torch.Tensor, n: int, mirror_pairing: bool, perturb_scale: float = 0.0) -> dict:
    """Settle from the home pose at rest, then a scripted rollout; per-step errors.

    ``perturb_scale`` > 0 makes the copies in [N:] a perturbed twin: every
    joint position and velocity offset by +-perturb_scale (random signs),
    injected AFTER settling so the offset is measured against a settled
    state, exactly like the solver's own self-asymmetry is.
    """
    gen = torch.Generator().manual_seed(1)
    env.reset()
    st = random_states(n, home, gen, env.device, standing=True)
    inject_pair(env, m, st, mirror_pairing=mirror_pairing)
    num_actions = env.num_actions
    zero = torch.zeros(2 * n, num_actions, device=env.device)
    settle_curve = []
    for t in range(SETTLE_STEPS):
        env.step(zero)
        if bool(env.reset_buf.any()):
            raise RuntimeError(describe_reset(env, t, "settling"))
        if t % 20 == 19:
            errs, _ = _pair_errors(env, m, n, mirror_pairing)
            contact_keys = {f"critic.{k}" for k in contact_blocks(env, "critic")}
            kin = max(d for k, d in errs.items() if k not in contact_keys)
            con = max(d for k, d in errs.items() if k in contact_keys)
            tilt_deg = float(torch.rad2deg(tilt_angle(env)).max())
            height = float(
                (env.get_entity_data("robot").root_link_pos_w[:, 2] - env.scene_manager.env_origins[:, 2]).mean()
            )
            settle_curve.append((t + 1, kin, con, tilt_deg, height))
    settled, _ = _pair_errors(env, m, n, mirror_pairing)
    if perturb_scale > 0.0:
        # Offset the copies' joints by the given magnitude, from the settled state.
        rd = env.get_entity_data("robot")
        signs = (torch.randint(0, 2, (n, num_actions), generator=gen, device="cpu") * 2 - 1).to(env.device).float()
        delta = signs * perturb_scale
        q = rd.joint_pos.clone()
        qd = rd.joint_vel.clone()
        ids = torch.arange(n, 2 * n, device=env.device)
        writer = env.get_robot_state_writer("robot")
        writer.set_dof_state(q[n:] + delta, qd[n:] + delta, env_ids=ids)
        writer.eval_fk(env_ids=ids)
        env._post_reset_forward()
        env._invalidate_cache()
    # Drawn on the CPU generator explicitly: Genesis makes CUDA torch's default device.
    actions = (torch.rand(HORIZON, n, num_actions, generator=gen, device="cpu") * 2 - 1).to(
        env.device
    ) * ACTION_AMPLITUDE
    per_step = []
    control = 0.0
    for t in range(HORIZON):
        second = m.joints(actions[t]) if mirror_pairing else actions[t]
        env.step(torch.cat([actions[t], second], 0))
        if bool(env.reset_buf.any()):
            raise RuntimeError(describe_reset(env, t, "rollout"))
        errs, ctrl = _pair_errors(env, m, n, mirror_pairing)
        control = max(control, ctrl)
        rew = {name: float((value[:n] - value[n:]).abs().max()) for name, value in env.rew_buf_per_type.items()}
        per_step.append((errs, rew))
    env.reset()
    return dict(settle_curve=settle_curve, settled=settled, per_step=per_step, control=control)


def section_c_dynamic(env, m: Mirror, chk: Checker, home: torch.Tensor, n: int, twin_asset: bool) -> None:
    print(
        f"\n=== C. dynamic oracle: settle {SETTLE_STEPS} steps at rest on the floor, then {HORIZON} scripted steps ==="
    )
    print("     pairing 'twin': envs [N:] are exact copies (same actions) -> the physics' own reproducibility floor")
    print("     pairing 'mirror': envs [N:] are the mirrors (actions K a)  -> the test")
    # Three pairings with identical inputs. The settle phase starts from the
    # home pose, which is its own mirror image, so during settling the twin
    # and mirror pairings are the same physical experiment: the mirror error
    # there is mirror(obs) - obs, the solver's own asymmetry on a symmetric
    # state. The perturbed twin is seeded with that magnitude and measures
    # how far the scripted rollout amplifies it.
    try:
        twin = run_pairing(env, m, home, n, mirror_pairing=False)
        mir = run_pairing(env, m, home, n, mirror_pairing=True)
        contact_keys = {f"critic.{k}" for k in contact_blocks(env, "critic")}
        self_kin = max(d for k, d in mir["settled"].items() if k not in contact_keys)
        self_con = max(d for k, d in mir["settled"].items() if k in contact_keys)
        pert = run_pairing(env, m, home, n, mirror_pairing=False, perturb_scale=max(self_kin, CONTINUOUS_FLOOR))
    except RuntimeError as exc:
        chk("no env reset during settling or the rollouts", False, str(exc))
        return
    chk("no env reset during settling or the rollouts", True, f"3 x {SETTLE_STEPS + HORIZON} steps")

    print(
        f"     settling (kinematic | contact):  {'step':>5s} {'twin':>21s} {'mirror':>21s} {'tilt':>6s} {'height':>7s}"
    )
    for (t, k1, c1, _, _), (_, k2, c2, tilt, height) in zip(twin["settle_curve"], mir["settle_curve"]):
        print(
            f"                                     {t:5d} {k1:10.2e} {c1:10.2e} {k2:10.2e} {c2:10.2e} {tilt:6.1f} {height:7.3f}"
        )
    print(
        f"     solver self-asymmetry on the symmetric settled state: kinematic {self_kin:.3e}, contact {self_con:.3e}"
    )
    print(
        f"     perturbed twin seeded with +-{max(self_kin, CONTINUOUS_FLOOR):.3e} on every joint position and velocity"
    )

    keys = list(mir["settled"])
    disc = {f"{g}.{k}" for g in ("actor", "critic") for k in discrete_blocks(env, g)}

    def worst(run, k):
        # Continuous blocks: worst step. Discrete flags/counters: the mismatch
        # rate over the whole horizon, since a threshold event flipping on one
        # step is noise, a flag disagreeing throughout is not.
        vals = [step[0][k] for step in run["per_step"]]
        return sum(vals) / len(vals) if k in disc else max(vals)

    twin_settle_kin = max(d for k, d in twin["settled"].items() if k not in contact_keys)
    twin_settle_con = max(d for k, d in twin["settled"].items() if k in contact_keys)
    print(
        f"     twin settle-end disagreement (identical copies): kinematic {twin_settle_kin:.3e}, contact {twin_settle_con:.3e}"
    )
    print(f"     {'obs block':28s} {'twin':>11s} {'perturbed':>11s} {'mirror':>11s}  {'kind':10s}")
    for k in keys:
        print(
            f"     {k:28s} {worst(twin, k):11.3e} {worst(pert, k):11.3e} {worst(mir, k):11.3e}  {'mean rate' if k in disc else 'max|d|'}"
        )
    print(f"     control: unmirrored max|o_i - o_(N+i)| over the mirror rollout = {mir['control']:.3e}")
    rew_names = list(mir["per_step"][0][1])
    rworst = lambda run, r: max(step[1][r] for step in run["per_step"])  # noqa: E731
    print(f"     {'reward term':28s} {'twin':>11s} {'perturbed':>11s} {'mirror':>11s}")
    for r in rew_names:
        print(f"     {r:28s} {rworst(twin, r):11.3e} {rworst(pert, r):11.3e} {rworst(mir, r):11.3e}")

    if twin_asset:
        kin_bound = max(SELF_ASYMMETRY_KINEMATIC_TOL, TWIN_RATIO * twin_settle_kin)
        con_bound = max(SELF_ASYMMETRY_CONTACT_TOL, TWIN_RATIO * twin_settle_con)
        chk(
            "C solver self-asymmetry on a symmetric state is small (kinematic)",
            self_kin <= kin_bound,
            f"{self_kin:.3e} vs {kin_bound:.3e}",
        )
        chk(
            "C solver self-asymmetry on a symmetric state is small (contact, log force)",
            self_con <= con_bound,
            f"{self_con:.3e} vs {con_bound:.3e}",
        )
    else:
        print(
            "     (source asset: the self-asymmetry above includes the asset's own left/right asymmetry; reported, not judged)"
        )
    for k in keys:
        floor = max(worst(twin, k), worst(pert, k))
        if k in disc:
            chk(
                f"C {k}: mirror mismatch rate within the amplification floor",
                worst(mir, k) <= max(DISCRETE_MISMATCH_RATE, TWIN_RATIO * floor),
                f"{worst(mir, k):.3f} (floor {floor:.3f})",
            )
        else:
            bound = max(CONTINUOUS_FLOOR, TWIN_RATIO * floor)
            chk(
                f"C {k}: mirror within the amplification floor",
                worst(mir, k) <= bound,
                f"{worst(mir, k):.3e} vs {bound:.3e} (twin {worst(twin, k):.1e}, perturbed {worst(pert, k):.1e})",
            )
    for r in rew_names:
        floor = max(rworst(twin, r), rworst(pert, r))
        bound = max(REWARD_FLOOR, TWIN_RATIO * floor)
        chk(
            f"C reward {r}: mirror within the amplification floor",
            rworst(mir, r) <= bound,
            f"{rworst(mir, r):.3e} vs {bound:.3e}",
        )
    chk(
        "control: the two halves differ by far more than any tolerance when NOT mirrored",
        mir["control"] > 100 * CONTINUOUS_FLOOR,
        f"{mir['control']:.3e}",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", choices=["mujoco", "newton", "genesis"], required=True)
    parser.add_argument("--num_envs", type=int, default=16, help="even; half originals, half mirrors")
    parser.add_argument("--rounds", type=int, default=3, help="random-state rounds in section B")
    parser.add_argument(
        "--asset",
        choices=["symmetrized", "source"],
        default="symmetrized",
        help="symmetrized twin (the proof) or the shipped asset (the residual)",
    )
    args = parser.parse_args()
    if args.num_envs % 2:
        raise SystemExit("--num_envs must be even")
    n = args.num_envs // 2

    source = K1VelocityConfig().robot.mjcf_path
    if args.asset == "symmetrized":
        src = Path(source)
        mjcf_path = str(write_symmetrized(src, src.with_name("k1_symmetrized.xml")))
    else:
        mjcf_path = source
    print(f"asset: {mjcf_path}")
    print("  left/right asymmetry of this asset:")
    for k, v in audit(mjcf_path).items():
        print(f"    {k:18s} {v:.3e}")

    env = build_env(args.sim, args.num_envs, mjcf_path)
    joints = list(env.act_manager.actuated_joint_names)
    spec = build_mirror_spec(env.obs_manager, joints, include_critic=True)
    m = Mirror(spec, env.device)
    home = env.act_manager.offset
    if home.dim() == 2:
        home = home[0]
    home = home.to(env.device)

    chk = Checker()
    section_a_structure(env, spec, chk)
    section_b_kinematic(env, m, chk, home, n, args.rounds)
    section_c_dynamic(env, m, chk, home, n, twin_asset=(args.asset == "symmetrized"))
    print("\n=== RESULT:", "ALL OK" if not chk.fails else f"{len(chk.fails)} FAIL: {chk.fails}", "===")
    return 1 if chk.fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
