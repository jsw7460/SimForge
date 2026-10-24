"""Gate: a reward term switched on mid-run works under the compiled chain.

With ``compile_terms`` the reward manager's first call runs eagerly through
recorders to learn what each term reads from the engine; every later step
snapshots exactly those reads and runs the terms through ``torch.compile``.
A pure term whose weight is zero is not run, so a term that starts at zero
(a curriculum stage, a schedule rising from zero) left nothing in the
record, and the step it came on raised ``RobotData.<attr> was not read
during the recording pass`` out of the compiled chain.

No simulator: a tensor fixture stands in for the engine and the real
``RewardManager`` runs on it, compiled and eager side by side on the same
inputs.

1. after the first call the record holds only what the active terms read;
2. switching two zero-weight terms on by the curriculum's route
   (``get_term_cfg(name).weight = ...``) and letting a ``LinearSchedule``
   cross zero does not raise, extends the record, and each extension
   costs exactly one eager step;
3. on every step the compiled chain's per-term values and total equal the
   eager manager's to float32 rounding (a fused reduction may associate
   differently; the shadow gate allows the same);
4. a finite-difference term (state on the manager) switched on reports
   zero on its first active step, as the eager path does — a recording
   that had run it speculatively at step 0 would report a jump here;
5. a term reading ``contact_force_history`` (the substep ring the contact
   penalties and K1's self-collision count use) is served from the
   snapshot, re-indexed by ``order=``, and matches eager;
6. a term reading sites (``site_pos_w_by_ids`` / ``site_lin_vel_w_by_ids``,
   the mujoco feet of go2) is composed on the snapshot from the body
   arrays exactly as ``SiteReaderMixin`` composes it on the engine;
7. ``processed_action_rate_l2_scaled`` (the go2 SAC smoothness term) finds
   its per-joint scale on the reward manager from inside the compiled
   chain — a cache keyed by ``id(env)`` missed on the snapshot view and
   put a tensor ``if`` into the traced graph;
8. the contact group is declared in the opposite order to the sensor's
   native columns (``lf, rf`` native, ``rf, lf`` declared), and a term
   reading one named column of ``is_contact``, ``contact_force`` and the
   history gets the same column compiled as eager, and the column the
   name denotes. The snapshot used to serve the native columns to an
   ``order=`` re-index computed against the declared names.

Run::

    python -m jaxrlworld.scripts.diag.gates.check_reward_compile_zero_weight_activation
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import torch

from jaxrlworld.rl.configs.common_config_classes import RewardConfig
from jaxrlworld.rl.configs.rewards import LinearSchedule, RewardTermConfig
from jaxrlworld.rl.envs.managers.common.contact import BaseContactManager
from jaxrlworld.rl.envs.managers.common.reward import RewardManager
from jaxrlworld.rl.envs.managers.common.reward_view import RecordingEnvView
from jaxrlworld.rl.envs.mdp.rewards.common.reward_terms import _fd_foot_velocity, processed_action_rate_l2_scaled
from jaxrlworld.rl.envs.site_frames import SiteFrame, SiteReaderMixin

NUM_ENVS, NUM_JOINTS = 4, 3
CONTROL_DT = 0.02
STEPS, SWITCH_STEP = 24, 8
ZERO_CROSSING = 12  # LinearSchedule(-0.1, 0.1, 24) is exactly 0.0 at step 12.
# Terms here are sums of a few O(1) products: a fused reduction may differ
# from the eager one by a float32 ulp or two, never more.
TOL = 1e-6


def pos_term(env):
    return env.get_entity_data("robot").joint_pos.square().sum(-1)


def vel_term(env):
    return env.get_entity_data("robot").joint_vel.abs().sum(-1)


def fd_term(env):
    foot = env.get_entity_data("robot").body_pos_w_all[:, 0:1, :]
    return _fd_foot_velocity(env, "gate_fd", foot).norm(dim=-1).sum(-1)


def quat_term(env):
    return env.get_entity_data("robot").root_quat_w[:, 0]


def history_term(env):
    history = env.contact_manager.contact_force_history("feet", order=["rf", "lf"])
    return history.norm(dim=-1).sum(dim=(1, 2))


def contact_term(env):
    """Column ``rf`` of every contact read — one named column, not a sum
    over feet, so a snapshot that hands out the wrong column shows."""
    cm = env.contact_manager
    contact = cm.is_contact("feet", order=["rf", "lf"])[:, 0].float()
    force = cm.contact_force("feet", order=["rf", "lf"])[:, 0].norm(dim=-1)
    history = cm.contact_force_history("feet", order=["rf", "lf"])[:, 0].norm(dim=-1).sum(-1)
    return contact + force + history


def expected_contact_term(env) -> torch.Tensor:
    """``contact_term`` by hand from the fixture: ``rf`` is the sensor's
    native column 1 (the group is registered ``lf, rf``)."""
    rf = env.contact_manager.history[:, 1]  # (num_envs, H, 3)
    force = rf[:, 0].norm(dim=-1)
    return (force > 1e-6).float() + force + rf.norm(dim=-1).sum(-1)


SITE_IDS = torch.tensor([1, 0])


def site_term(env):
    rd = env.get_entity_data("robot")
    pos = rd.site_pos_w_by_ids(SITE_IDS)
    vel = rd.site_lin_vel_w_by_ids(SITE_IDS)
    return pos.norm(dim=-1).sum(-1) + vel[..., 0].sum(-1)


ACTION_SCALE = {"j[01]": 0.25, "j2": 0.5}


def act_rate_term(env):
    return processed_action_rate_l2_scaled(env, ACTION_SCALE)


class _Robot(SiteReaderMixin):
    """RobotData stand-in: plain tensors behind the properties the site mixin composes from."""

    _FRAMES = (
        SiteFrame("toe", "foot0", (0.1, 0.0, -0.02), (1.0, 0.0, 0.0, 0.0)),
        SiteFrame("heel", "foot1", (-0.05, 0.01, 0.0), (1.0, 0.0, 0.0, 0.0)),
    )

    def __init__(self):
        self.joint_pos = torch.zeros(NUM_ENVS, NUM_JOINTS)
        self.joint_vel = torch.zeros(NUM_ENVS, NUM_JOINTS)
        self.root_quat_w = torch.zeros(NUM_ENVS, 4)
        self._body_pos = torch.zeros(NUM_ENVS, 2, 3)
        self._body_quat = torch.zeros(NUM_ENVS, 2, 4)
        self._body_lin_vel = torch.zeros(NUM_ENVS, 2, 3)
        self._body_ang_vel = torch.zeros(NUM_ENVS, 2, 3)

    @property
    def _site_frames(self):
        return self._FRAMES

    def find_body_index(self, body_name: str) -> int:
        return {"foot0": 0, "foot1": 1}[body_name]

    @property
    def body_pos_w_all(self):
        return self._body_pos

    @property
    def body_quat_w_all(self):
        return self._body_quat

    @property
    def body_lin_vel_w_all(self):
        return self._body_lin_vel

    @property
    def body_ang_vel_w_all(self):
        return self._body_ang_vel


class _Contacts(BaseContactManager):
    """A contact manager whose per-substep force ring is a plain tensor fixture."""

    def __init__(self, env):
        super().__init__(env)
        # Native columns lf, rf; declared rf, lf — a permuted group, like
        # go2's feet on mjlab and Newton.
        self._register_group("feet", ["lf", "rf"], ("force",), patterns=("rf", "lf"))
        self.history = torch.zeros(NUM_ENVS, 2, 4, 3)

    def _compute_group_contact_force(self, group):
        return self.history[:, :, 0, :]

    def _compute_group_contact_force_history(self, group):
        return self.history


class Rewards(RewardConfig):
    pos = RewardTermConfig(func=pos_term, weight=1.0)
    vel = RewardTermConfig(func=vel_term, weight=0.0)  # switched on at SWITCH_STEP
    fd = RewardTermConfig(func=fd_term, weight=0.0)  # switched on at SWITCH_STEP
    sched = RewardTermConfig(func=quat_term, weight=LinearSchedule(-0.1, 0.1, STEPS))
    hist = RewardTermConfig(func=history_term, weight=1.0)
    contact = RewardTermConfig(func=contact_term, weight=1.0)
    site = RewardTermConfig(func=site_term, weight=1.0)
    act_rate = RewardTermConfig(func=act_rate_term, weight=1.0)


def make_env(compile_terms: bool):
    """A tensor stand-in for the engine with a real RewardManager on it."""
    robot = _Robot()
    env = SimpleNamespace(
        device=torch.device("cpu"),
        num_envs=NUM_ENVS,
        control_dt=CONTROL_DT,
        _env_step_counter=0,
        robot_entity_name="robot",
        episode_length_buf=torch.full((NUM_ENVS,), 5, dtype=torch.long),
        scene_manager=SimpleNamespace(),
        command_manager=SimpleNamespace(),
        act_manager=SimpleNamespace(
            actuated_joint_names=[f"j{i}" for i in range(NUM_JOINTS)],
            processed_actions=torch.zeros(NUM_ENVS, NUM_JOINTS),
            prev_processed_actions=torch.zeros(NUM_ENVS, NUM_JOINTS),
        ),
        get_robot_data=lambda name=None: robot,
        get_entity_data=lambda name=None: robot,
    )
    env.contact_manager = _Contacts(env)
    env.reward_manager = RewardManager(env, Rewards(compile_terms=compile_terms))
    return env, robot


def fill(env, robot, step: int) -> None:
    """The engine state at ``step``: the same on both managers."""
    g = torch.Generator().manual_seed(1000 + step)
    robot.joint_pos.copy_(torch.randn(NUM_ENVS, NUM_JOINTS, generator=g))
    robot.joint_vel.copy_(torch.randn(NUM_ENVS, NUM_JOINTS, generator=g))
    robot._body_pos.copy_(torch.randn(NUM_ENVS, 2, 3, generator=g))
    quat = torch.randn(NUM_ENVS, 2, 4, generator=g)
    robot._body_quat.copy_(quat / quat.norm(dim=-1, keepdim=True))
    robot._body_lin_vel.copy_(torch.randn(NUM_ENVS, 2, 3, generator=g))
    robot._body_ang_vel.copy_(torch.randn(NUM_ENVS, 2, 3, generator=g))
    robot.root_quat_w.copy_(torch.randn(NUM_ENVS, 4, generator=g))
    env.contact_manager.history.copy_(torch.randn(NUM_ENVS, 2, 4, 3, generator=g))
    env.act_manager.processed_actions.copy_(torch.randn(NUM_ENVS, NUM_JOINTS, generator=g))
    env.act_manager.prev_processed_actions.copy_(torch.randn(NUM_ENVS, NUM_JOINTS, generator=g))


def main() -> int:
    failures: list[str] = []

    def chk(name: str, ok: bool, detail: str = "") -> None:
        print(f"  [{'OK' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
        if not ok:
            failures.append(name)

    env_c, robot_c = make_env(compile_terms=True)
    env_e, robot_e = make_env(compile_terms=False)
    mgr_c, mgr_e = env_c.reward_manager, env_e.reward_manager

    eager_steps: list[int] = []
    # Every recording view is kept alive: the manager drops it right
    # before building the snapshot view, which then often lands on the
    # same address — a term cache keyed by ``id(env)`` would hit by that
    # accident here and miss on a real env (item 7).
    recording_views: list[RecordingEnvView] = []
    original = RewardManager._compute_stacked

    def counting(self, env):
        if isinstance(env, RecordingEnvView):
            eager_steps.append(self.env._env_step_counter)
            recording_views.append(env)
        return original(self, env)

    RewardManager._compute_stacked = counting

    worst_term, worst_total = 0.0, 0.0
    fd_first_active: float | None = None
    try:
        for step in range(STEPS):
            for env, robot in ((env_c, robot_c), (env_e, robot_e)):
                env._env_step_counter = step
                fill(env, robot, step)
                if step == SWITCH_STEP:
                    for name in ("vel", "fd"):
                        env.reward_manager.get_term_cfg(name).weight = 1.0
            buf_c, per_c = torch.zeros(NUM_ENVS), {}
            buf_e, per_e = torch.zeros(NUM_ENVS), {}
            mgr_c.set_rewards(buf_c, per_c)
            mgr_e.set_rewards(buf_e, per_e)

            if step == 0:
                rec = mgr_c._record
                chk(
                    "record after the first call holds only the active terms' reads",
                    rec.robot_attrs
                    == {
                        "joint_pos",
                        "root_quat_w",
                        "body_pos_w_all",
                        "body_quat_w_all",
                        "body_lin_vel_w_all",
                        "body_ang_vel_w_all",
                    },
                    f"robot attrs {sorted(rec.robot_attrs)}",
                )
                chk("contact_force_history read is recorded", rec.contact_force_history == {"feet"})
                chk(
                    "is_contact and contact_force reads are recorded",
                    rec.contact_is_contact == {"feet"} and rec.contact_force == {"feet"},
                )
                group = env_c.contact_manager._get_group("feet")
                chk(
                    "the fixture's contact group is permuted (native lf, rf; declared rf, lf)",
                    group.tracked_names == ["rf", "lf"] and group.native_perm.tolist() == [1, 0],
                    f"tracked {group.tracked_names}, perm {group.native_perm}",
                )
            if step == SWITCH_STEP:
                rec = mgr_c._record
                chk(
                    "switching the terms on extends the record",
                    "joint_vel" in rec.robot_attrs,
                    f"robot attrs {sorted(rec.robot_attrs)}",
                )
                fd_first_active = float(per_c["fd"].abs().max())
            if step == ZERO_CROSSING:
                chk("schedule is exactly zero at the crossing step", float(per_e["sched"].abs().max()) == 0.0)

            for name in mgr_c.reward_terms:
                worst_term = max(worst_term, float((per_c[name] - per_e[name]).abs().max()))
            worst_total = max(worst_total, float((buf_c - buf_e).abs().max()))
    finally:
        RewardManager._compute_stacked = original

    print("\n=== compiled vs eager, every step ===")
    chk("per-term values match eager on every step", worst_term <= TOL, f"max |d| {worst_term:.3g} (tol {TOL:g})")
    chk("totals match eager on every step", worst_total <= TOL, f"max |d| {worst_total:.3g} (tol {TOL:g})")
    act = env_c.act_manager
    # Per-term values carry weight * control_dt, as the manager stores them.
    delta = (act.processed_actions - act.prev_processed_actions) / torch.tensor([0.25, 0.25, 0.5])
    expected = -(delta**2).sum(-1) * CONTROL_DT
    chk(
        "action-rate term equals the hand-computed scaled L2 on the last step",
        float((per_c["act_rate"] - expected).abs().max()) <= TOL,
        f"max |d| {float((per_c['act_rate'] - expected).abs().max()):.3g}",
    )
    want = expected_contact_term(env_e) * CONTROL_DT
    chk(
        "contact term's named column is the sensor's rf column, compiled and eager",
        float((per_c["contact"] - want).abs().max()) <= TOL and float((per_e["contact"] - want).abs().max()) <= TOL,
        f"max |d| compiled {float((per_c['contact'] - want).abs().max()):.3g}, "
        f"eager {float((per_e['contact'] - want).abs().max()):.3g}",
    )
    chk(
        "eager recording steps are exactly the first call and the switch-on step",
        eager_steps == [0, SWITCH_STEP],
        f"{eager_steps}",
    )
    chk(
        "finite-difference term reports zero on its first active step",
        fd_first_active == 0.0,
        f"max |fd| at switch-on {fd_first_active}",
    )
    later = mgr_c._recorded_active
    chk(
        "recorded active set covers every term after the run",
        later == frozenset(mgr_c.reward_terms),
        str(sorted(later)),
    )
    chk("site term was computed through the snapshot on every step", "site" in per_c and "site" in per_e)

    print(f"\n=== RESULT: {'ALL OK' if not failures else f'{len(failures)} FAILED: {failures}'} ===")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
