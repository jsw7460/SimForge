"""Gate: feet rewards gate each foot by ITS OWN contact column.

``feet_slip``, ``feet_slip_fd`` and ``feet_swing_height`` multiply a
per-foot velocity or height column by a per-foot contact column. For the
MuJoCo go2 presets the feet are sites, ``("FR", "FL", "RR", "RL")``, and
``resolve_sites`` returns their ids in that selector order; the contact
sensor ``feet_ground_contact`` tracks the foot geoms in MODEL order,
``(FL, FR, RL, RR)``. ``_feet_contact_order`` returned ``None`` for
sites, so the contact tensor was never reordered and column ``i`` paired
FR's velocity with FL's contact (and RR with RL). In a trot the lateral
pair is in antiphase, so the SWING foot's speed was charged as slip —
the mujoco go2 ``feet_slip`` curve sat 30x below newton / genesis on
identical physics.

This gate rebuilds the exact go2 situation on the CPU without a
simulator: the real go2 MJCF through ``resolve_sites`` and MuJoCo's
compiled geom table, the real ``BaseContactManager`` reorder path, and
the real reward functions on a stub robot.

1. the premise: selector-order site ids are ``[2, 1, 4, 3]`` while the
   sensor's model-order geoms are ``(FL, FR, RL, RR)``;
2. ``_feet_contact_order`` maps each site to its own geom and the contact
   manager's reindex is ``[1, 0, 3, 2]``;
3. a trot frame (FR / RR swinging at 1 m/s, FL / RL standing at 3 cm/s,
   contacts FL and RL): ``penalize_feet_slip`` and ``_fd`` charge the
   stance feet only (``2 * 0.03^2``), and the same frame through the
   contact group's natural order charges the swing feet (``2 * 1.0^2``) —
   the bug, reproduced, 1100x;
4. ``FeetSwingHeightTracker`` on sites evaluates the landing foot's own
   peak;
5. bodies (Newton / Genesis) still reorder by the resolved body names and
   an explicit ``contact_order`` still wins, so the other backends'
   rewards are unchanged;
6. a site with no, several or a shared match raises instead of guessing,
   and the match is by whole ``_`` tokens (``foot_1`` does not claim
   ``foot_10_collision``);
7. every caller passes the env and group so no path can reach the
   natural order again, and the go2 builder still declares the selector
   this gate reproduces;
8. the manager itself exposes the columns in DECLARED order: registered
   with the builder's patterns, the same model-order sensor comes out as
   ``(FR, FL, RR, RL)``, every read (contact, timing, ``order=``) is
   permuted, a regex declaration keeps the model order, a declaration
   that does not cover the sensor raises, and a duplicated ``order``
   raises;
9. two finite-difference slip terms on one contact group keep separate
   previous positions (the state is keyed by the resolved selector, not
   the group).

Run::

    python -m jaxrlworld.scripts.diag.gates.check_feet_contact_column_alignment
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import mujoco
import torch

import jaxrlworld
from jaxrlworld.rl.configs.scene.entity_selector import ResolvedEntity, SceneEntitySelector
from jaxrlworld.rl.envs.managers.common.contact import BaseContactManager, ContactGroup
from jaxrlworld.rl.envs.mdp.rewards.common.reward_terms import (
    FeetSwingHeightTracker,
    _feet_contact_order,
    penalize_feet_slip,
    penalize_feet_slip_fd,
)
from jaxrlworld.rl.envs.site_frames import resolve_sites
from jaxrlworld.rl.utils.string import resolve_matching_names

ROOT = Path(jaxrlworld.__file__).parent
GO2_XML = ROOT / "assets" / "unitree_go2" / "xmls" / "go2.xml"
GO2_BUILDER = ROOT / "rl" / "configs" / "presets" / "go2" / "_mujoco_builders.py"
# What presets/go2/_mujoco_builders.py declares (checked against the source below).
SITE_NAMES = ("FR", "FL", "RR", "RL")
GEOM_PATTERNS = tuple(f"{name}_foot_collision" for name in SITE_NAMES)
GROUP = "feet_ground_contact"
NUM_ENVS = 3
DT = 0.02
SWING_SPEED = 1.0
STANCE_SPEED = 0.03
CALLERS = (
    "rl/envs/mdp/rewards/common/reward_terms.py",
    "scripts/diag/g1/check_g1_contact_force_parity.py",
)


class StubContactManager(BaseContactManager):
    """The real reorder path over a contact state this gate sets by hand."""

    def __init__(self, env, tracked: list[str], patterns: tuple[str, ...] | None = None) -> None:
        super().__init__(env)
        # ``tracked`` is the order the (stub) sensor produces; without
        # ``patterns`` the group exposes it as is, which is what every
        # backend did before the manager learned the declared order.
        self._register_group(GROUP, list(tracked), ("found", "force"), patterns=patterns)
        self.state = torch.zeros((env.num_envs, len(tracked)), dtype=torch.bool)

    def _compute_group_contact_force(self, group: ContactGroup) -> torch.Tensor | None:
        return None

    def _compute_group_is_contact(self, group: ContactGroup) -> torch.Tensor:
        return self.state

    def set_contacts(self, by_name: dict[str, bool]) -> None:
        """Contact per name, written into the sensor's NATIVE (model-order) columns."""
        native = self._get_group(GROUP).native_names
        self.state = torch.tensor([[by_name[name] for name in native]] * self.num_envs, dtype=torch.bool)

    def set_landed(self, by_name: dict[str, bool]) -> None:
        """Mark feet as having touched down this step (``compute_first_contact``)."""
        group = self._get_group(GROUP)
        tracked = group.tracked_names
        landed = torch.tensor([[by_name[name] for name in tracked]] * self.num_envs, dtype=torch.bool)
        group.current_contact_time.copy_(torch.where(landed, torch.full_like(group.current_contact_time, DT), 0.0))


class StubRobotData:
    """Site / body positions and velocities, indexed by the resolved ids."""

    def __init__(self, num_sites: int) -> None:
        self.pos = torch.zeros((NUM_ENVS, num_sites, 3))
        self.vel = torch.zeros((NUM_ENVS, num_sites, 3))

    def site_pos_w_by_ids(self, ids: torch.Tensor) -> torch.Tensor:
        return self.pos[:, ids]

    def site_lin_vel_w_by_ids(self, ids: torch.Tensor) -> torch.Tensor:
        return self.vel[:, ids]

    def body_pos_w_by_ids(self, ids: torch.Tensor) -> torch.Tensor:
        return self.pos[:, ids]

    def body_lin_vel_w_by_ids(self, ids: torch.Tensor) -> torch.Tensor:
        return self.vel[:, ids]


def model_order_geoms() -> list[str]:
    """The sensor's tracked names: the builder's patterns resolved against the
    compiled model in MODEL order, as mjlab's ``find_geoms`` (``preserve_order``
    False) does."""
    model = mujoco.MjModel.from_xml_path(str(GO2_XML))
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "" for i in range(model.ngeom)]
    _, matched = resolve_matching_names(list(GEOM_PATTERNS), names, preserve_order=False)
    return matched


def make_env(tracked: list[str], num_frames: int, patterns: tuple[str, ...] | None = None):
    env = SimpleNamespace(
        num_envs=NUM_ENVS,
        device="cpu",
        control_dt=DT,
        episode_length_buf=torch.full((NUM_ENVS,), 10, dtype=torch.long),
        command_manager=SimpleNamespace(
            lin_vel_x=torch.ones(NUM_ENVS),
            lin_vel_y=torch.zeros(NUM_ENVS),
            ang_vel=torch.zeros(NUM_ENVS),
        ),
        reward_manager=SimpleNamespace(_fd_prev_foot_pos={}),
        scene_manager=SimpleNamespace(
            config=SimpleNamespace(entities={"robot": SimpleNamespace(mjcf_path=str(GO2_XML))}, rigid_objects={})
        ),
        _env_step_counter=0,
    )
    env.contact_manager = StubContactManager(env, tracked, patterns)
    env.robot = StubRobotData(num_frames)
    env.get_entity_data = lambda name: env.robot
    return env


def resolve_go2_sites(env) -> ResolvedEntity:
    site_ids, site_names = resolve_sites(env, "robot", SITE_NAMES)
    return ResolvedEntity(
        name="robot",
        joint_ids=None,
        joint_ids_native=None,
        body_ids=None,
        geom_ids=None,
        site_ids=site_ids,
        actuator_ids=None,
        site_names=site_names,
        source_selector=SceneEntitySelector(name="robot", site_names=SITE_NAMES),
    )


def set_trot_frame(env, resolved: ResolvedEntity) -> None:
    """FR / RR swinging forward, FL / RL on the ground creeping; contacts on FL / RL."""
    speed = {"FR": SWING_SPEED, "FL": STANCE_SPEED, "RR": SWING_SPEED, "RL": STANCE_SPEED}
    for name, sid in zip(resolved.site_names, resolved.site_ids.tolist()):
        env.robot.vel[:, sid, 0] = speed[name]
    env.contact_manager.set_contacts({g: g.startswith(("FL", "RL")) for g in env.contact_manager.tracked_names(GROUP)})


def main() -> int:
    failures: list[str] = []

    def chk(name: str, ok: bool, detail: str = "") -> None:
        print(f"  [{'OK' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
        if not ok:
            failures.append(name)

    print("=" * 78)
    print("FEET CONTACT COLUMN ALIGNMENT (go2 MuJoCo sites)")
    print("=" * 78)

    tracked = model_order_geoms()
    num_sites = mujoco.MjModel.from_xml_path(str(GO2_XML)).nsite
    env = make_env(tracked, num_sites)
    resolved = resolve_go2_sites(env)
    ids = resolved.site_ids.tolist()

    print("\n-- 1. premise: selector order vs sensor order --")
    chk("site ids come back in selector order", resolved.site_names == list(SITE_NAMES), str(resolved.site_names))
    chk("... and are not sorted (the MJCF table lists FL before FR, RL before RR)", ids == [2, 1, 4, 3], str(ids))
    chk(
        "the sensor tracks the foot geoms in model order (FL, FR, RL, RR)",
        tracked == [f"{n}_foot_collision" for n in ("FL", "FR", "RL", "RR")],
        str(tracked),
    )
    chk("so the natural contact order differs from the selector order", [t[:2] for t in tracked] != list(SITE_NAMES))

    print("\n-- 2. the reorder --")
    order = _feet_contact_order(env, GROUP, resolved, None)
    chk("each site maps to its own geom", order == list(GEOM_PATTERNS), str(order))
    reindex = env.contact_manager._get_reindex(GROUP, order).tolist()
    chk("contact manager reindex swaps the lateral pairs", reindex == [1, 0, 3, 2], str(reindex))
    chk(
        "reordered is_contact has one column per site",
        tuple(env.contact_manager.is_contact(GROUP, order=order).shape) == (NUM_ENVS, len(SITE_NAMES)),
    )

    print("\n-- 3. trot frame through the real terms --")
    set_trot_frame(env, resolved)
    is_contact = env.contact_manager.is_contact(GROUP, order=order)
    expected = -2 * STANCE_SPEED**2
    wrong = -2 * SWING_SPEED**2
    slip = penalize_feet_slip(env, GROUP, command_threshold=0.05, asset_cfg=resolved)
    chk(
        "feet_slip charges the stance feet only",
        torch.allclose(slip, torch.full((NUM_ENVS,), expected)),
        f"{slip[0].item():.6f} (expected {expected:.6f})",
    )
    vel_xy_sq = torch.sum(torch.square(env.robot.vel[:, resolved.site_ids, :2]), dim=-1)
    natural = -torch.sum(vel_xy_sq * env.contact_manager.is_contact(GROUP, order=None).float(), dim=1)
    chk(
        "the natural order charges the swing feet (the bug, reproduced)",
        torch.allclose(natural, torch.full((NUM_ENVS,), wrong)),
        f"{natural[0].item():.6f} = {natural[0].item() / expected:.0f}x the correct penalty",
    )
    chk(
        "feet_slip gives the per-foot contact to the foot that owns it",
        torch.equal(is_contact[0], torch.tensor([False, True, False, True])),
        f"sites {list(SITE_NAMES)} -> contact {is_contact[0].tolist()}",
    )
    # Finite-difference variant: seed, then move the feet by one step's travel.
    env.robot.pos.zero_()
    penalize_feet_slip_fd(env, GROUP, command_threshold=0.05, asset_cfg=resolved)
    env.robot.pos[:, :, 0] = env.robot.vel[:, :, 0] * DT
    slip_fd = penalize_feet_slip_fd(env, GROUP, command_threshold=0.05, asset_cfg=resolved)
    chk(
        "feet_slip_fd charges the stance feet only",
        torch.allclose(slip_fd, torch.full((NUM_ENVS,), expected), atol=1e-7),
        f"{slip_fd[0].item():.6f}",
    )

    print("\n-- 4. swing height on sites --")
    env = make_env(tracked, num_sites)
    resolved = resolve_go2_sites(env)
    target = 0.1
    tracker = FeetSwingHeightTracker(env, GROUP, target_height=target, command_threshold=0.05, asset_cfg=resolved)
    chk("tracker holds the matched order", tracker._contact_order == list(GEOM_PATTERNS), str(tracker._contact_order))
    # Step 1: FR / RR in the air at 0.12 m (FL / RL on the ground), nobody lands.
    env.contact_manager.set_contacts({g: g.startswith(("FL", "RL")) for g in tracked})
    env.contact_manager.set_landed({g: False for g in tracked})
    for name, sid in zip(resolved.site_names, resolved.site_ids.tolist()):
        env.robot.pos[:, sid, 2] = 0.12 if name in ("FR", "RR") else 0.02
    first = tracker(env)
    chk("no landing, no cost", torch.equal(first, torch.zeros(NUM_ENVS)), str(first[0].item()))
    peaks = dict(zip(resolved.site_names, tracker.peak_heights[0].tolist()))
    chk(
        "peaks accumulate on the airborne feet (FR, RR)",
        all(abs(peaks[n] - h) < 1e-6 for n, h in {"FR": 0.12, "FL": 0.0, "RR": 0.12, "RL": 0.0}.items()),
        str(peaks),
    )
    # Step 2: FR / RR touch down.
    env.contact_manager.set_contacts({g: g.startswith(("FR", "RR")) for g in tracked})
    env.contact_manager.set_landed({g: g.startswith(("FR", "RR")) for g in tracked})
    for name, sid in zip(resolved.site_names, resolved.site_ids.tolist()):
        env.robot.pos[:, sid, 2] = 0.02
    cost = tracker(env)
    expected_sh = -2 * (0.12 / target - 1.0) ** 2
    chk(
        "landing charges the landing feet's own peak",
        torch.allclose(cost, torch.full((NUM_ENVS,), expected_sh)),
        f"{cost[0].item():.6f} (expected {expected_sh:.6f}; natural order would charge FL / RL peaks = 0)",
    )

    print("\n-- 5. bodies and explicit orders are untouched --")
    body_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    body_resolved = ResolvedEntity(
        name="robot",
        joint_ids=None,
        joint_ids_native=None,
        body_ids=torch.tensor([5, 9, 13, 17]),
        geom_ids=None,
        site_ids=None,
        actuator_ids=None,
        body_names=body_names,
    )
    body_env = make_env(body_names, 20)
    chk(
        "body selector reorders by its resolved body names",
        _feet_contact_order(body_env, GROUP, body_resolved, None) == body_names,
    )
    explicit = ["RR_foot", "RL_foot", "FR_foot", "FL_foot"]
    got = _feet_contact_order(body_env, GROUP, body_resolved, explicit)
    chk("explicit contact_order wins and is copied", got == explicit and got is not explicit)
    got = _feet_contact_order(env, GROUP, resolved, explicit)
    chk("... for sites too", got == explicit)

    print("\n-- 6. refusals --")

    def raises(fn) -> str | None:
        try:
            fn()
        except ValueError as e:
            return str(e)
        return None

    def sites(*names: str) -> ResolvedEntity:
        return ResolvedEntity(
            name="robot",
            joint_ids=None,
            joint_ids_native=None,
            body_ids=None,
            geom_ids=None,
            site_ids=torch.arange(len(names)),
            actuator_ids=None,
            site_names=list(names),
        )

    msg = raises(lambda: _feet_contact_order(env, GROUP, sites("FR", "FL", "RR", "XX"), None))
    chk("unmatched site raises", msg is not None and "'XX'" in msg, msg)
    msg = raises(lambda: _feet_contact_order(env, GROUP, sites("foot", "RR", "RL", "FL"), None))
    chk("ambiguous site (a token every foot carries) raises", msg is not None and "candidates" in msg, msg)
    shared_env = make_env(["FL_foot_collision", "hip"], 4)
    msg = raises(lambda: _feet_contact_order(shared_env, GROUP, sites("FL", "foot"), None))
    chk("two sites on one contact name raise", msg is not None and "same contact names" in msg, msg)
    numbered_env = make_env(["foot_10_collision", "foot_2_collision"], 4)
    msg = raises(lambda: _feet_contact_order(numbered_env, GROUP, sites("foot_1", "foot_2"), None))
    chk("tokens, not substrings: foot_1 does not claim foot_10", msg is not None and "'foot_1'" in msg, msg)
    chk(
        "a multi-token site still finds its geom (FR_foot -> FR_foot_collision)",
        _feet_contact_order(env, GROUP, sites("FR_foot", "FL_foot", "RR_foot", "RL_foot"), None) == list(GEOM_PATTERNS),
    )
    nameless = ResolvedEntity(
        name="robot",
        joint_ids=None,
        joint_ids_native=None,
        body_ids=None,
        geom_ids=None,
        site_ids=torch.arange(4),
        actuator_ids=None,
    )
    msg = raises(lambda: _feet_contact_order(env, GROUP, nameless, None))
    chk("site ids without site names raise", msg is not None)

    print("\n-- 7. callers and the preset --")
    call = re.compile(r"_feet_contact_order\(([^)]*)\)")
    for rel in CALLERS:
        src = (ROOT / rel).read_text()
        args = [m.group(1).strip() for m in call.finditer(src)]
        bad = [a for a in args if not a.startswith("env")]
        chk(f"{rel}: every call passes the env first ({len(args)} found)", args and not bad, str(bad))
    builder = GO2_BUILDER.read_text()
    chk(
        "go2 mujoco builder still declares the selector this gate reproduces",
        'site_names = ("FR", "FL", "RR", "RL")' in builder and 'f"{name}_foot_collision"' in builder,
    )

    print("\n-- 8. the manager exposes the DECLARED order --")
    declared_env = make_env(tracked, num_sites, patterns=GEOM_PATTERNS)
    group = declared_env.contact_manager._get_group(GROUP)
    chk("native names are the sensor's model order", group.native_names == tracked, str(group.native_names))
    chk(
        "tracked names are the declared order (FR, FL, RR, RL)",
        group.tracked_names == list(GEOM_PATTERNS),
        str(group.tracked_names),
    )
    chk("the permutation is the lateral swap", group.native_perm.tolist() == [1, 0, 3, 2], str(group.native_perm))
    declared_resolved = resolve_go2_sites(declared_env)
    set_trot_frame(declared_env, declared_resolved)
    contact = declared_env.contact_manager.is_contact(GROUP)
    chk(
        "is_contact comes out in declared order (FL, RL standing -> columns 1, 3)",
        torch.equal(contact[0], torch.tensor([False, True, False, True])),
        str(contact[0].tolist()),
    )
    declared_env.contact_manager.advance(DT)
    ct = declared_env.contact_manager.current_contact_time(GROUP)[0]
    chk(
        "timing buffers accumulate on the declared columns",
        torch.equal(ct > 0, torch.tensor([False, True, False, True])),
        str(ct.tolist()),
    )
    chk(
        "_feet_contact_order is the identity on a declared-order group",
        _feet_contact_order(declared_env, GROUP, declared_resolved, None) == group.tracked_names,
    )
    slip_declared = penalize_feet_slip(declared_env, GROUP, command_threshold=0.05, asset_cfg=declared_resolved)
    chk(
        "feet_slip gives the same penalty as on the native-order group",
        torch.allclose(slip_declared, torch.full((NUM_ENVS,), expected)),
        f"{slip_declared[0].item():.6f}",
    )
    by_native = declared_env.contact_manager.is_contact(GROUP, order=tracked)
    chk(
        "order= still reorders by name (asking for model order undoes the permutation)",
        torch.equal(by_native[0], torch.tensor([True, False, True, False])),
        str(by_native[0].tolist()),
    )
    regex_env = make_env(tracked, num_sites, patterns=(".*_foot_collision",))
    regex_group = regex_env.contact_manager._get_group(GROUP)
    chk(
        "a regex declaration keeps the model order and skips the gather",
        regex_group.tracked_names == tracked and regex_group.native_perm is None,
        str(regex_group.tracked_names),
    )
    msg = raises(lambda: make_env(tracked, num_sites, patterns=("FR_foot_collision", "FL_foot_collision")))
    chk("a declaration that does not cover the sensor raises", msg is not None and "unclaimed" in msg, msg)
    msg = raises(
        lambda: make_env(
            tracked,
            num_sites,
            patterns=("FR_foot_collision", "FL_foot_collision", "RR_foot_collision", "RL_foot_collision", "hip"),
        )
    )
    chk("a declaration the sensor does not match raises", msg is not None and "Not all regular expressions" in msg, msg)
    msg = raises(lambda: declared_env.contact_manager.is_contact(GROUP, order=["FR_foot_collision"] * 4))
    chk("a duplicated order raises", msg is not None and "more than once" in msg, msg)

    print("\n-- 9. finite-difference state is per term, not per group --")
    fd_env = make_env(tracked, num_sites, patterns=GEOM_PATTERNS)
    first_cfg = resolve_go2_sites(fd_env)
    second_cfg = resolve_go2_sites(fd_env)
    chk("two terms on one group hold distinct resolved selectors", first_cfg is not second_cfg)
    fd_env.contact_manager.set_contacts({g: True for g in tracked})
    fd_env.robot.pos.zero_()
    penalize_feet_slip_fd(fd_env, GROUP, command_threshold=0.05, asset_cfg=first_cfg)
    penalize_feet_slip_fd(fd_env, GROUP, command_threshold=0.05, asset_cfg=second_cfg)
    fd_env.robot.pos[:, :, 0] = STANCE_SPEED * DT
    first = penalize_feet_slip_fd(fd_env, GROUP, command_threshold=0.05, asset_cfg=first_cfg)
    second = penalize_feet_slip_fd(fd_env, GROUP, command_threshold=0.05, asset_cfg=second_cfg)
    expected_fd = -4 * STANCE_SPEED**2
    chk(
        "both terms see the step's travel (the second used to read zero)",
        torch.allclose(first, torch.full((NUM_ENVS,), expected_fd), atol=1e-7)
        and torch.allclose(second, torch.full((NUM_ENVS,), expected_fd), atol=1e-7),
        f"first {first[0].item():.6f}, second {second[0].item():.6f} (expected {expected_fd:.6f})",
    )
    chk(
        "the store holds one buffer per term",
        len([k for k in fd_env.reward_manager._fd_prev_foot_pos if k.startswith("feet_slip_fd:")]) == 2,
        str(list(fd_env.reward_manager._fd_prev_foot_pos)),
    )

    print("=" * 78)
    print("=== RESULT: ALL OK ===" if not failures else f"=== RESULT: {len(failures)} FAILED ===")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
