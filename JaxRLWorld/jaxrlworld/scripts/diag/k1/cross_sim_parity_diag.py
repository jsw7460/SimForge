"""Cross-sim fingerprint for the Booster K1 velocity port.

Answers, element by element, whether the same robot and the same MDP are
loaded on all three backends. Two halves:

- **Asset.** Joint order and limits, efforts and gains, action scale and
  offset, total and per-body mass, trunk centre of mass, and the collision
  geometry with its per-geom friction.
- **MDP.** With IDENTICAL states written into each backend and every source of
  randomness stripped, the 75-D actor observation, the 90-D critic observation
  and every weighted reward term. Any residual difference is then a backend
  difference, not a sampling one.

Randomness removed for the comparison: domain randomization, command delay,
observation noise, encoder bias, the stochastic fall draw (the term is run at
probability 1, where it short-circuits to deterministic), and the velocity
command itself, which is written and held.

What this deliberately does NOT answer: which geom pairs actually collide at
runtime, and whether the engines agree once they are allowed to integrate.
A short zero-action rollout is dumped as a reference number for the latter,
but it is reported, not asserted -- three solvers will diverge and that is not
a failure of this port.

One backend per process (the single-sim invariant), then compare::

    jaxpy -m jaxrlworld.scripts.diag.k1.cross_sim_parity_diag --sim mujoco
    jaxpy -m jaxrlworld.scripts.diag.k1.cross_sim_parity_diag --sim newton
    jaxpy -m jaxrlworld.scripts.diag.k1.cross_sim_parity_diag --sim genesis
    python -m jaxrlworld.scripts.diag.k1.cross_sim_parity_diag --compare

Dumps land in ``parity_out/<sim>.json`` next to this file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from jaxrlworld.rl.configs.common_config_classes import disable_corruption
from jaxrlworld.rl.configs.presets.k1_velocity.base import K1VelocityConfig
from jaxrlworld.rl.runners.base_runner import BaseRunner
from jaxrlworld.rl.utils.quat_utils import quat_from_angle_axis_wxyz, quat_mul_wxyz

OUT_DIR = Path(__file__).parent / "parity_out"
NUM_ENVS = 2
SIMS = ("mujoco", "newton", "genesis")

# Tolerances. Observations and rewards are pure functions of the injected
# state, so they should agree to float32 noise; masses and geometry come
# straight off the same file and should agree far tighter.
TOL_OBS = 2e-4
TOL_REWARD = 2e-4
TOL_ASSET = 1e-5
TOL_MASS = 1e-3

# Reward terms that cannot be produced by writing a state and reading it back.
# Each needs either contact, which only exists once the solver has stepped, or
# history that only accumulates across steps. Injecting a state leaves all of
# them at zero, so comparing them there is comparing zero against zero. They
# are measured over the stepped rollouts below instead.
# Reads the action history only, never the simulator, so it must agree
# exactly once the same action sequence is applied.
PHYSICS_FREE_TERMS = {"action_rate"}

CONTACT_DEPENDENT_TERMS = {
    "action_rate",  # previous versus current action
    "air_time",  # contact-manager air time
    "foot_slip",  # foot in contact and sliding
    "foot_swing_height",  # evaluated at landing
    "self_collisions",  # contact between the robot's own bodies
    "soft_landing",  # impact force at touchdown
}

# Those terms are priced off each engine's contact solve, so they are NOT
# expected to agree to float32 noise the way the state-pure terms do. What
# would matter is a term that is systematically several times larger on one
# backend -- the shape the self-collision substep-count error took. A factor
# of two is the line between solver disagreement and a different formula.
MAX_CONTACT_TERM_RATIO = 2.0

# Impact force is the most solver-sensitive quantity here: it is set by how a
# collision gets resolved across the substeps of one control step, and the
# three engines do that differently. Measured on the controlled drop, the
# spread is 2.0x (Newton) and 2.9x (Genesis) against MuJoCo, and lengthening
# the window moved both toward each other rather than to one. The allowance is
# that spread plus margin -- still far under the 4x a wrong reduction would
# show, which is the shape the self-collision substep-count error took.
TERM_RATIO_ALLOWANCE = {"soft_landing": 3.5}

# Below this the term barely fired and a ratio says nothing: two values a
# millionth of the reward scale apart differ by 5x and mean the same thing.
CONTACT_TERM_FLOOR = 1e-5

# Rollouts that put the robot through contact. Each starts from a written
# state so all three backends begin identically.
#
# ``action`` selects what is applied each step:
#   "zero"     - nothing, for terms that only need the robot to fall and land
#   "scripted" - a fixed sequence, identical on every backend
#   a dict     - a sustained per-joint action, held for the whole rollout
CONTACT_ROLLOUTS = {
    # Dropped from 15 cm up: the feet leave the ground (air time), fall
    # (clearance, swing height) and land (soft landing, slip).
    # Dropped from 15 cm up and left to settle. The window is long enough to
    # cover every landing event, not just the first: a single impulse is
    # partitioned across substeps differently by each solver, and comparing one
    # of them measures that partition rather than the term. Summed over the
    # whole settle the three agree far better.
    "drop_and_land": dict(
        state="home",
        lift=0.15,
        steps=140,
        cmd=[0.8, 0.0, 0.0],
        action="zero",
        compare_magnitudes=True,
    ),
    # A changing action, so the action-rate penalty has something to price.
    # Magnitudes are not compared: a gentle 30-step wiggle puts a foot right at
    # the edge of leaving the ground, and whether that marginal lift registers
    # as a landing differs between solvers. That is a knife-edge event, not a
    # measurement of the term.
    "scripted_actions": dict(
        state="home",
        lift=0.0,
        steps=30,
        cmd=[0.5, 0.0, 0.0],
        action="scripted",
        compare_magnitudes=False,
    ),
    # Arms driven to their inward limits and HELD there. Writing the pose is
    # not enough: the controller pulls the arms straight back to the home pose
    # on the next step, which is why the earlier version of this rollout never
    # produced a single self-contact.
    # Flailing under a seeded action sequence. This is the only rollout that
    # reaches self-collision: the shoulder roll axis points forward, so driving
    # it either way swings the arm up or down rather than across the body, and
    # folding an elbow cannot help because parent and child geoms never collide
    # with each other. Rather than keep guessing at a pose, this drives the
    # robot the way the random-action sweep does, where the group is known to
    # fire on every backend.
    #
    # Magnitudes are NOT compared here. Three solvers given the same actions
    # take three different trajectories within a few steps, so the sums measure
    # the trajectories, not the terms. What this rollout establishes is that
    # each term is reachable at all.
    "seeded_flailing": dict(
        state="home",
        lift=0.0,
        steps=50,
        cmd=[0.5, 0.0, 0.0],
        action="random",
        compare_magnitudes=False,
    ),
}

_MJ_GEOM_TYPES = {0: "plane", 2: "sphere", 3: "capsule", 4: "ellipsoid", 5: "cylinder", 6: "box", 7: "mesh"}

FOOT_BODIES = ["left_foot_link", "right_foot_link"]
TRUNK_BODY = "Trunk"

# The order the per-joint entries of STATES below are written in. The backends
# have agreed on joint order so far, but nothing guarantees they will, so the
# injected vectors are permuted into whatever order the environment reports
# rather than assuming this one.
CANONICAL_JOINTS = [
    "Head_Yaw",
    "Head_Pitch",
    "Left_Shoulder_Pitch",
    "Left_Shoulder_Roll",
    "Left_Elbow_Pitch",
    "Left_Elbow_Yaw",
    "Left_Hip_Pitch",
    "Left_Hip_Roll",
    "Left_Hip_Yaw",
    "Left_Knee_Pitch",
    "Left_Ankle_Pitch",
    "Left_Ankle_Roll",
    "Right_Shoulder_Pitch",
    "Right_Shoulder_Roll",
    "Right_Elbow_Pitch",
    "Right_Elbow_Yaw",
    "Right_Hip_Pitch",
    "Right_Hip_Roll",
    "Right_Hip_Yaw",
    "Right_Knee_Pitch",
    "Right_Ankle_Pitch",
    "Right_Ankle_Roll",
]

# Canonical injected states. Each pins the full root pose and velocity, the
# joint offsets from the home pose, the joint velocities, and the held command.
STATES = {
    "home": dict(
        z=0.5125,
        rpy=(0.0, 0.0, 0.0),
        lin=(0.0, 0.0, 0.0),
        ang=(0.0, 0.0, 0.0),
        dq=[0.0] * 22,
        qd=[0.0] * 22,
        cmd=[0.0, 0.0, 0.0],
    ),
    "walking": dict(
        z=0.50,
        rpy=(0.05, -0.08, 0.40),
        lin=(0.6, -0.15, 0.05),
        ang=(0.1, -0.05, 0.25),
        dq=[
            0.02,
            -0.01,
            0.10,
            -0.05,
            0.08,
            -0.03,
            -0.10,
            0.05,
            -0.08,
            0.03,
            0.25,
            -0.06,
            0.04,
            -0.30,
            0.12,
            -0.05,
            -0.20,
            0.06,
            -0.04,
            0.28,
            -0.10,
            0.05,
        ],
        qd=[
            0.1,
            -0.1,
            0.5,
            -0.3,
            0.4,
            -0.2,
            -0.5,
            0.3,
            -0.4,
            0.2,
            1.2,
            -0.4,
            0.3,
            -1.5,
            0.8,
            -0.3,
            -1.1,
            0.4,
            -0.3,
            1.4,
            -0.7,
            0.3,
        ],
        cmd=[0.8, 0.1, -0.2],
    ),
    "left_leg_lifted": dict(
        z=0.55,
        rpy=(0.0, 0.05, -1.20),
        lin=(0.1, 0.0, 0.0),
        ang=(0.0, 0.0, 0.1),
        dq=[0.0] * 10 + [-0.45, 0.10, 0.05, 0.60, -0.25, 0.05] + [0.0] * 6,
        qd=[0.0] * 22,
        cmd=[0.3, 0.0, 0.0],
    ),
    "running_tilted": dict(
        z=0.46,
        rpy=(-0.10, 0.22, 1.5707963),
        lin=(1.6, 0.35, -0.2),
        ang=(-0.3, 0.2, -0.6),
        dq=[
            0.05,
            0.10,
            0.30,
            -0.20,
            0.25,
            -0.15,
            -0.30,
            0.20,
            -0.25,
            0.15,
            0.40,
            -0.10,
            0.08,
            -0.50,
            0.20,
            -0.08,
            -0.35,
            0.10,
            -0.08,
            0.45,
            -0.18,
            0.08,
        ],
        qd=[
            0.3,
            -0.3,
            1.0,
            -0.8,
            0.9,
            -0.5,
            -1.0,
            0.8,
            -0.9,
            0.5,
            2.0,
            -0.8,
            0.6,
            -2.5,
            1.4,
            -0.6,
            -1.8,
            0.7,
            -0.5,
            2.3,
            -1.2,
            0.6,
        ],
        cmd=[1.7, -0.4, 0.8],
    ),
    # Arms driven hard inward past the trunk. Without a state like this the
    # self-collision term compares zero against zero on every backend, which
    # would pass while saying nothing -- and that term is the one whose
    # semantics were wrong on the first attempt.
    "arms_into_trunk": dict(
        z=0.5125,
        rpy=(0.0, 0.0, 0.0),
        lin=(0.0, 0.0, 0.0),
        ang=(0.0, 0.0, 0.0),
        dq=[
            0.0,
            0.0,
            0.0,
            1.9,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -1.9,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        qd=[0.0] * 22,
        cmd=[0.0, 0.0, 0.0],
    ),
    "standing_still_off_pose": dict(
        z=0.52,
        rpy=(0.02, 0.03, 2.5),
        lin=(0.0, 0.0, 0.0),
        ang=(0.0, 0.0, 0.0),
        dq=[0.3] * 22,
        qd=[0.0] * 22,
        cmd=[0.0, 0.0, 0.0],
    ),
}


def script_fingerprint() -> str:
    """Digest of this file, stamped into every dump.

    A backend whose dump crashes leaves its previous file on disk, and the
    comparison would then read a dump produced by different code and report
    the difference as a backend difference. Refusing to mix versions is the
    only way that failure mode announces itself.
    """
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]


def leaf(name: str) -> str:
    return name.rsplit("/", 1)[-1]


# ── environment with every randomness source stripped ────────────────


def build_parity_env(sim: str):
    cfg = K1VelocityConfig(sim_type=sim, num_envs=NUM_ENVS)
    # The stochastic fall term short-circuits to deterministic at probability 1.
    cfg.fall_probability = 1.0
    cfg.action_delay_min = 0
    cfg.action_delay_max = 0
    cfgs = cfg.build()

    # Drop every randomizer and the push disturbance. Reset events are
    # irrelevant: states are written on top of whatever reset produced.
    for name in list(vars(cfgs.event)):
        if name.startswith("dr_") or name == "push":
            delattr(cfgs.event, name)

    disable_corruption(cfgs.observation)

    env = BaseRunner._create_env_from_config(cfgs)
    env.reset()
    return env


# ── asset fingerprint ────────────────────────────────────────────────


def dump_asset(env) -> dict:
    rd = env.get_entity_data("robot")
    act = env.act_manager
    joints = [leaf(n) for n in act.actuated_joint_names]
    if sorted(joints) != sorted(CANONICAL_JOINTS):
        raise ValueError(
            f"joint set differs from the one the injected states are written for: "
            f"only-env {sorted(set(joints) - set(CANONICAL_JOINTS))}, "
            f"only-states {sorted(set(CANONICAL_JOINTS) - set(joints))}"
        )
    soft_lo, soft_hi = rd.soft_joint_pos_limits

    def row(value):
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value)
        value = value.double().cpu()
        return (value[0] if value.dim() > 1 else value).tolist()

    # Per-joint quantities are stored keyed by joint NAME, because the
    # backends do not agree on joint order.
    out = {
        "joints": joints,
        "soft_limit_lower": _by_name(joints, row(soft_lo)),
        "soft_limit_upper": _by_name(joints, row(soft_hi)),
        "action_offset": _by_name(joints, row(act.offset)),
        "default_joint_pos": _by_name(joints, row(rd.default_joint_pos)),
    }
    out["action_scale"] = _by_name(joints, row(act._scale))

    # ``act_manager._actuators`` holds (actuator, joint_indices) pairs, each
    # actuator carrying only the joints it drives.
    for attr in ("stiffness", "damping", "effort_limit"):
        full = [float("nan")] * len(joints)
        for actuator, joint_indices in getattr(act, "_actuators", []):
            value = getattr(actuator, attr, None)
            if not isinstance(value, torch.Tensor):
                continue
            values = row(value)
            for slot, index in enumerate(torch.as_tensor(joint_indices).cpu().tolist()):
                full[index] = values[slot]
        out[attr] = None if any(v != v for v in full) else _by_name(joints, full)

    out["joint_armature"] = _joint_armature(env, joints)

    masses, geoms, trunk_com = _backend_bodies_and_geoms(env)
    out["total_mass"] = float(sum(masses.values()))
    out["body_mass"] = dict(sorted(masses.items()))
    out["trunk_com_local"] = trunk_com
    out["collision_geoms"] = sorted(geoms, key=lambda g: (g["body"], g["type"], g["friction"]))
    out["geom_multiset"] = sorted(f"{g['body']}({_canon_type(g['type'])})" for g in geoms)
    return out


def _joint_armature(env, joints: list[str]) -> dict[str, float] | None:
    """Armature as each engine actually loaded it, keyed by joint name.

    Read from the simulator's own model rather than from the config that asked
    for it, so a backend that silently kept the asset's value instead of the
    configured one shows up here.
    """
    if env.sim_type == "mujoco":
        import mujoco

        mj = env.scene_manager.sim.mj_model
        model = env.scene_manager.sim.model
        # Joint names in the compiled model are prefixed with the entity they
        # belong to, so match on the leaf.
        dof_of = {}
        for j in range(mj.njnt):
            name = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_JOINT, j)
            if name:
                dof_of[leaf(name)] = int(mj.jnt_dofadr[j])
        missing = [n for n in joints if n not in dof_of]
        if missing:
            raise KeyError(f"joints absent from the compiled model: {missing}")
        return {name: float(model.dof_armature[0, dof_of[name]]) for name in joints}

    if env.sim_type == "newton":
        import warp as wp

        view = env.scene_manager.articulation_views["robot"]
        armature = wp.to_torch(view.get_attribute("joint_armature", env.scene_manager.model))
        # The view holds one entry per joint AXIS, and the free base joint
        # carries six of them, so the array is wider than the actuated joint
        # count. The action manager already maps canonical joint order onto
        # those columns; using its map rather than slicing by position is what
        # keeps a silent off-by-six from scrambling every value while still
        # comparing cleanly against itself.
        columns = env.act_manager.indexing.newton_qd_indices
        armature = armature.reshape(env.num_envs, -1)[0]
        if len(columns) != len(joints):
            raise ValueError(
                f"the action manager maps {len(columns)} dof columns for " f"{len(joints)} actuated joints"
            )
        selected = armature[torch.as_tensor(columns)].double().cpu().tolist()
        return {name: float(selected[i]) for i, name in enumerate(joints)}

    entity = env.scene_manager["robot"]
    armature = entity.get_dofs_armature(dofs_idx_local=env.act_manager._actuated_joint_indices)
    armature = armature.reshape(-1, len(joints))[0].double().cpu().tolist()
    return {name: float(armature[i]) for i, name in enumerate(joints)}


def _backend_bodies_and_geoms(env):
    """Per-body mass, trunk COM, and the collision geom set, per backend."""
    if env.sim_type == "mujoco":
        mj = env.scene_manager.sim.mj_model
        entity = env.scene_manager.get_entity("robot")
        body_ids = [int(b) for b in entity.indexing.body_ids.cpu().tolist()]
        masses = {leaf(mj.body(b).name): float(env.scene_manager.sim.model.body_mass[0, b]) for b in body_ids}
        trunk_local, _ = entity.find_bodies([TRUNK_BODY])
        trunk_gid = int(entity.indexing.body_ids[trunk_local[0]])
        trunk_com = env.scene_manager.sim.model.body_ipos[0, trunk_gid].cpu().tolist()
        geoms = []
        for b in body_ids:
            start, num = int(mj.body_geomadr[b]), int(mj.body_geomnum[b])
            for g in range(start, start + num):
                if int(mj.geom_contype[g]) == 0 and int(mj.geom_conaffinity[g]) == 0:
                    continue
                geoms.append(
                    {
                        "body": leaf(mj.body(b).name),
                        "type": _MJ_GEOM_TYPES.get(int(mj.geom_type[g]), str(int(mj.geom_type[g]))),
                        "friction": float(env.scene_manager.sim.model.geom_friction[0, g, 0]),
                    }
                )
        return masses, geoms, trunk_com

    if env.sim_type == "newton":
        import newton as _newton
        import warp as wp

        from jaxrlworld.rl.envs.utils.newton.body_cache import get_cache

        cache = get_cache(env)
        model = env.scene_manager.model
        per_env = cache.bodies_per_env
        labels = list(cache.body_names)
        mass_t = wp.to_torch(model.body_mass).reshape(env.num_envs, per_env)[0]
        masses = {labels[i]: float(mass_t[i]) for i in range(per_env)}
        trunk_com = (
            wp.to_torch(model.body_com).reshape(env.num_envs, per_env, 3)[0, labels.index(TRUNK_BODY)].cpu().tolist()
        )
        shape_body = wp.to_torch(model.shape_body).cpu()
        shape_type = wp.to_torch(model.shape_type).cpu()
        shape_mu = wp.to_torch(model.shape_material_mu).cpu()
        shape_flags = wp.to_torch(model.shape_flags).cpu()
        collide_bit = int(_newton.ShapeFlags.COLLIDE_SHAPES)
        type_names = {int(v): v.name.lower() for v in _newton.GeoType}
        geoms = []
        for s in range(shape_body.shape[0]):
            b = int(shape_body[s])
            if b < 0 or b >= per_env or not (int(shape_flags[s]) & collide_bit):
                continue
            geoms.append(
                {
                    "body": labels[b],
                    "type": type_names.get(int(shape_type[s]), str(int(shape_type[s]))),
                    "friction": float(shape_mu[s]),
                }
            )
        return masses, geoms, trunk_com

    import genesis as gs

    entity = env.scene_manager["robot"]
    masses, geoms = {}, []
    for link in entity.links:
        masses[leaf(link.name)] = _scalar(link.get_mass())
        for geom in link.geoms:
            geoms.append(
                {
                    "body": leaf(link.name),
                    "type": gs.GEOM_TYPE(int(geom.type)).name.lower(),
                    "friction": _scalar(geom.get_friction()),
                }
            )
    trunk_idx = env.get_entity_data("robot").find_body_index(TRUNK_BODY)
    trunk_com = entity.get_links_COM(links_idx_local=[trunk_idx])[0].squeeze(0).cpu().tolist()
    return masses, geoms, trunk_com


def _scalar(value) -> float:
    """First element of a value one backend reports per environment.

    Genesis is configured with batched link and dof info, so a link's mass and
    a geom's friction come back with an environment axis. Every environment
    carries the same value here because this diagnostic strips randomization,
    so the first is the value.
    """
    if isinstance(value, torch.Tensor):
        return float(value.flatten()[0])
    return float(value)


def _canon_type(name: str) -> str:
    """Collapse each backend's mesh flavour to one word.

    MuJoCo compiles an STL to a convex hull and Genesis names the same shape
    differently; the geometry is the same, only the label differs.
    """
    name = name.lower()
    return "mesh" if "mesh" in name else name


# ── injected-state fingerprint ───────────────────────────────────────


def inject_and_measure(env, spec: dict) -> dict:
    device = env.device
    n = env.num_envs
    env_ids = torch.arange(n, device=device)

    pos = env.scene_manager.env_origins.clone()
    pos[:, 2] = spec["z"]
    roll, pitch, yaw = spec["rpy"]
    ones = torch.ones(n, device=device)
    ax_x = torch.tensor((1.0, 0.0, 0.0), device=device)
    ax_y = torch.tensor((0.0, 1.0, 0.0), device=device)
    ax_z = torch.tensor((0.0, 0.0, 1.0), device=device)
    quat = quat_mul_wxyz(
        quat_mul_wxyz(
            quat_from_angle_axis_wxyz(yaw * ones, ax_z),
            quat_from_angle_axis_wxyz(pitch * ones, ax_y),
        ),
        quat_from_angle_axis_wxyz(roll * ones, ax_x),
    )

    writer = env.get_robot_state_writer("robot")
    writer.set_root_pose(pos, quat, env_ids=env_ids)
    writer.set_root_velocity(
        torch.tensor(spec["lin"], device=device).repeat(n, 1),
        torch.tensor(spec["ang"], device=device).repeat(n, 1),
        env_ids=env_ids,
    )

    default = env.act_manager.offset
    if default.dim() == 1:
        default = default.unsqueeze(0).repeat(n, 1)
    joints = [leaf(name) for name in env.act_manager.actuated_joint_names]
    order = _permutation(CANONICAL_JOINTS, joints)
    dq = [spec["dq"][i] for i in order]
    qd = [spec["qd"][i] for i in order]
    writer.set_dof_state(
        default + torch.tensor(dq, device=device),
        torch.tensor(qd, device=device).repeat(n, 1),
        env_ids=env_ids,
    )
    writer.eval_fk(env_ids=env_ids)
    env._post_reset_forward()
    env._invalidate_cache()

    env.command_manager.set_commands(env_ids, velocity=torch.tensor(spec["cmd"], device=device).repeat(n, 1))

    env.rew_buf[:] = 0.0
    env.reward_manager.set_rewards(reward_buffer=env.rew_buf, reward_buffer_per_type=env.rew_buf_per_type)
    rewards = {name: float(value[0]) for name, value in env.rew_buf_per_type.items()}

    env.obs_manager.process_observations()
    obs = env.obs_manager.obs_dict

    rd = env.get_entity_data("robot")
    actor = obs["actor"][0].detach().cpu().tolist()
    critic = obs["critic"][0].detach().cpu().tolist()
    return {
        "rewards": rewards,
        "total_reward": float(env.rew_buf[0]),
        "actor_obs": actor,
        "critic_obs": critic,
        # Sliced by the layout the preset declares, so a difference is
        # attributed to a block instead of to an index in a 90-long vector.
        # The three 22-long blocks are in this backend's joint order and are
        # permuted to a shared order before comparison.
        "obs_blocks": {
            "base_ang_vel": actor[0:3],
            "projected_gravity": actor[3:6],
            "joint_pos": actor[6:28],
            "joint_vel": actor[28:50],
            "actions": actor[50:72],
            "command": actor[72:75],
            "critic_base_lin_vel": critic[75:78],
            "critic_foot_height": critic[78:80],
            "critic_foot_air_time": critic[80:82],
            "critic_foot_contact": critic[82:84],
            "critic_foot_contact_forces": critic[84:90],
        },
        "derived": {
            "projected_gravity": rd.projected_gravity_b[0].cpu().tolist(),
            "root_lin_vel_b": rd.root_link_lin_vel_b[0].cpu().tolist(),
            "root_ang_vel_b": rd.root_link_ang_vel_b[0].cpu().tolist(),
            "foot_z": rd.body_pos_w(FOOT_BODIES)[0, :, 2].cpu().tolist(),
        },
    }


def _scripted_action(env, step: int) -> torch.Tensor:
    """A fixed action sequence, a function of the step index alone.

    Identical on every backend by construction, which is what makes the
    action-rate penalty comparable exactly rather than approximately.
    """
    index = torch.arange(env.num_actions, device=env.device, dtype=torch.float32)
    values = 0.4 * torch.sin(0.7 * step + 0.37 * index)
    return values.unsqueeze(0).repeat(env.num_envs, 1)


def _sustained_action(env, table: dict[str, float]) -> torch.Tensor:
    """A constant action, addressed by joint name."""
    joints = [leaf(name) for name in env.act_manager.actuated_joint_names]
    unknown = sorted(set(table) - set(joints))
    if unknown:
        raise KeyError(f"sustained action names no such joint: {unknown}")
    values = [table.get(name, 0.0) for name in joints]
    return torch.tensor(values, device=env.device).unsqueeze(0).repeat(env.num_envs, 1)


def contact_rollouts(env) -> dict:
    """Per-term reward sums over rollouts that actually produce contact.

    Each rollout is seeded with a written state so every backend starts from
    the same place. The sums are what the contact-dependent terms are compared
    on; nothing else reaches them.
    """
    out = {}
    for name, spec in CONTACT_ROLLOUTS.items():
        base = dict(STATES[spec["state"]])
        base["z"] = base["z"] + spec["lift"]
        base["cmd"] = spec["cmd"]
        inject_and_measure(env, base)

        action_spec = spec["action"]
        held = _sustained_action(env, action_spec) if isinstance(action_spec, dict) else None
        zero = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        # Drawn up front, off a generator of their own. Drawing inside the loop
        # instead makes the sequence depend on how much randomness the
        # environment itself consumed, and an episode that ends on one backend
        # a few steps earlier than on another consumes a different amount at
        # its reset -- which silently gave the three backends different actions.
        scripted_random = None
        if action_spec == "random":
            generator = torch.Generator(device=env.device)
            generator.manual_seed(12345)
            scripted_random = [
                (
                    torch.rand(
                        env.num_envs,
                        env.num_actions,
                        generator=generator,
                        device=env.device,
                    )
                    - 0.5
                )
                * 6.0
                for _ in range(spec["steps"])
            ]

        totals: dict[str, float] = {}
        resets = 0
        for step in range(spec["steps"]):
            if held is not None:
                action = held
            elif action_spec == "scripted":
                action = _scripted_action(env, step)
            elif scripted_random is not None:
                action = scripted_random[step]
            else:
                action = zero
            _obs, _rew, terminated, truncated, _extras = env.step(action)
            resets += int((terminated | truncated)[0])
            for term, value in env.rew_buf_per_type.items():
                totals[term] = totals.get(term, 0.0) + float(value[0])
        # The action-rate penalty reads the action history, and a reset clears
        # it. So the term is physics-free only for a rollout the robot survives;
        # once an episode ends mid-rollout, WHEN it ended is a physics question
        # and the sum inherits it.
        out[name] = {"rewards": totals, "resets": resets}
    return out


def zero_action_rollout(env, steps: int = 3) -> dict:
    """Reference only: how far the three engines drift once they integrate."""
    traj = []
    for _ in range(steps):
        env.step(torch.zeros(env.num_envs, env.num_actions, device=env.device))
        rd = env.get_entity_data("robot")
        origin = env.scene_manager.env_origins[0]
        traj.append(
            {
                "root_pos": (rd.root_link_pos_w[0] - origin).cpu().tolist(),
                "root_lin_vel": rd.root_link_lin_vel_w[0].cpu().tolist(),
                "q": rd.joint_pos[0].cpu().tolist(),
            }
        )
    return {"steps": traj}


# ── compare ──────────────────────────────────────────────────────────


def _maxdiff(a, b) -> float:
    # The device is stated because one backend sets torch's global default to
    # the GPU on import, and these are plain Python lists off a JSON file.
    ta = torch.tensor(a, dtype=torch.float64, device="cpu")
    tb = torch.tensor(b, dtype=torch.float64, device="cpu")
    if ta.shape != tb.shape:
        return float("inf")
    return float((ta - tb).abs().max()) if ta.numel() else 0.0


def _by_name(names: list[str], values) -> dict[str, float]:
    """Pair a per-joint vector with its joint names.

    The backends do not agree on joint ORDER, so every per-joint quantity is
    compared name by name. Comparing the raw vectors would report a pure
    ordering difference as a value difference.
    """
    return {name: float(v) for name, v in zip(names, values)}


def _permutation(src: list[str], dst: list[str]) -> list[int]:
    """Indices that reorder a vector laid out in ``src`` order into ``dst`` order."""
    index = {name: i for i, name in enumerate(src)}
    return [index[name] for name in dst]


def _canonical_blocks(record: dict, joints: list[str], canonical: list[str]) -> dict[str, list]:
    """Re-express the joint-ordered observation blocks in one shared order."""
    perm = _permutation(joints, canonical)
    out = {}
    for name, block in record["obs_blocks"].items():
        out[name] = [block[i] for i in perm] if len(block) == len(joints) else block
    return out


def compare() -> int:
    data = {}
    for sim in SIMS:
        path = OUT_DIR / f"{sim}.json"
        if path.exists():
            data[sim] = json.loads(path.read_text())
    if len(data) < 2:
        print(f"need at least two dumps in {OUT_DIR} (have {sorted(data)})")
        return 1

    current = script_fingerprint()
    stale = {sim: d.get("script") for sim, d in data.items() if d.get("script") != current}
    if stale:
        print("REFUSING TO COMPARE: these dumps were written by a different version of")
        print("this script, so any difference they show would be meaningless.")
        for sim, fingerprint in stale.items():
            print(f"  {sim}: {fingerprint or 'no fingerprint'} (this script: {current})")
        print("Re-run the dump for each backend listed above.")
        return 1

    ref_sim = next(iter(data))
    ref = data[ref_sim]
    fails: list[str] = []

    def chk(name, ok, detail=""):
        print(f"  [{'OK' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
        if not ok:
            fails.append(name)

    for sim, other in data.items():
        if sim == ref_sim:
            continue
        print(f"\n=== {ref_sim} vs {sim} ===")
        a, b = ref["asset"], other["asset"]

        chk(
            "same joint SET",
            set(a["joints"]) == set(b["joints"]),
            f"only-{ref_sim} {sorted(set(a['joints']) - set(b['joints']))} "
            f"only-{sim} {sorted(set(b['joints']) - set(a['joints']))}",
        )
        # Order is reported, not asserted. The backends index joints in their
        # own native order and every term is internally consistent with it, so
        # a difference here is not a porting error -- but it does mean a policy
        # cannot be moved between these two backends without permuting its
        # input and output, so it has to be visible.
        if a["joints"] != b["joints"]:
            moved = [j for j in a["joints"] if a["joints"].index(j) != b["joints"].index(j)]
            print(
                f"  [INFO] joint ORDER differs ({len(moved)} of {len(a['joints'])} joints move). "
                "Values below are compared by name; a policy needs a permutation to cross."
            )
            print(f"         {ref_sim}: {a['joints']}")
            print(f"         {sim}: {b['joints']}")
        else:
            print("  [INFO] joint order identical")

        for key in (
            "soft_limit_lower",
            "soft_limit_upper",
            "action_offset",
            "default_joint_pos",
            "action_scale",
            "stiffness",
            "damping",
            "effort_limit",
            "joint_armature",
        ):
            # A missing value fails rather than comparing equal: two Nones
            # would otherwise pass this check without either side ever
            # producing a number.
            if a.get(key) is None or b.get(key) is None:
                missing = [s for s, d in ((ref_sim, a), (sim, b)) if d.get(key) is None]
                chk(f"asset.{key}", False, f"not read on {', '.join(missing)}")
                continue
            names = sorted(set(a[key]) & set(b[key]))
            diff = _maxdiff([a[key][n] for n in names], [b[key][n] for n in names])
            chk(f"asset.{key} (per joint, by name)", diff <= TOL_ASSET, f"maxdiff {diff:.2e} over {len(names)} joints")

        chk(
            "total mass",
            abs(a["total_mass"] - b["total_mass"]) <= TOL_MASS,
            f"{a['total_mass']:.4f} vs {b['total_mass']:.4f}",
        )
        chk(
            "trunk COM",
            _maxdiff(a["trunk_com_local"], b["trunk_com_local"]) <= 1e-4,
            f"{a['trunk_com_local']} vs {b['trunk_com_local']}",
        )

        shared = sorted(set(a["body_mass"]) & set(b["body_mass"]))
        chk(
            "per-body mass over shared bodies",
            shared and _maxdiff([a["body_mass"][k] for k in shared], [b["body_mass"][k] for k in shared]) <= 1e-4,
            f"{len(shared)} shared; only-{ref_sim} {sorted(set(a['body_mass']) - set(b['body_mass']))}; "
            f"only-{sim} {sorted(set(b['body_mass']) - set(a['body_mass']))}",
        )

        chk(
            "collision geom multiset",
            a["geom_multiset"] == b["geom_multiset"],
            f"only-{ref_sim} {sorted(set(a['geom_multiset']) - set(b['geom_multiset']))} "
            f"only-{sim} {sorted(set(b['geom_multiset']) - set(a['geom_multiset']))}",
        )
        fa = sorted(g["friction"] for g in a["collision_geoms"])
        fb = sorted(g["friction"] for g in b["collision_geoms"])
        chk(
            "collision geom friction values",
            len(fa) == len(fb) and _maxdiff(fa, fb) <= 1e-4,
            f"{ref_sim} {sorted({round(v, 3) for v in fa})} vs {sim} {sorted({round(v, 3) for v in fb})}",
        )

        # One shared joint order for the observation comparison.
        canonical = a["joints"]
        for state in STATES:
            ra, rb = ref["states"][state], other["states"][state]
            blocks_a = _canonical_blocks(ra, a["joints"], canonical)
            blocks_b = _canonical_blocks(rb, b["joints"], canonical)
            worst_block, worst_diff = None, 0.0
            for block in blocks_a:
                diff = _maxdiff(blocks_a[block], blocks_b[block])
                if diff > worst_diff:
                    worst_block, worst_diff = block, diff
                if diff > TOL_OBS:
                    chk(f"{state}: obs block {block}", False, f"maxdiff {diff:.2e}")
            chk(
                f"{state}: observations (all blocks, joint-order canonicalized)",
                worst_diff <= TOL_OBS,
                f"worst {worst_diff:.2e} at {worst_block}",
            )

            keys = sorted(set(ra["rewards"]) | set(rb["rewards"]))
            missing = [k for k in keys if k not in ra["rewards"] or k not in rb["rewards"]]
            if missing:
                chk(f"{state}: reward term set", False, f"missing on one side: {missing}")
                continue
            worst_key = max(keys, key=lambda k: abs(ra["rewards"][k] - rb["rewards"][k]))
            worst = abs(ra["rewards"][worst_key] - rb["rewards"][worst_key])
            chk(f"{state}: reward terms", worst <= TOL_REWARD, f"maxdiff {worst:.2e} at {worst_key}")

        # A term that reads zero in every injected state is compared as zero
        # against zero, which passes while proving nothing about it. Say so,
        # because the states exist to exercise the terms.
        reach = {}
        for state in STATES:
            for name, value in ref["states"][state]["rewards"].items():
                reach[name] = max(reach.get(name, 0.0), abs(value))
        unexercised = sorted(
            n for n, v in reach.items() if v <= 0.0 and n != "total_reward" and n not in CONTACT_DEPENDENT_TERMS
        )
        chk(
            "every state-pure reward term is non-zero in at least one injected state",
            not unexercised,
            f"never fired: {unexercised}",
        )

        print(f"\n  per-term reward, {ref_sim} (weighted, one row per state):")
        terms = sorted(t for t in reach if t != "total_reward")
        header = "    " + f"{'state':24s}" + "".join(f"{t[:11]:>13s}" for t in terms)
        print(header)
        for state in STATES:
            values = ref["states"][state]["rewards"]
            print("    " + f"{state:24s}" + "".join(f"{values[t]:13.6f}" for t in terms))
        print()

        # Contact-dependent terms, over the stepped rollouts.
        print(f"\n  contact-dependent terms, summed over each rollout " f"({ref_sim} / {sim}, ratio):")
        for rollout, spec in CONTACT_ROLLOUTS.items():
            sums_a = ref["contact_rollouts"][rollout]["rewards"]
            sums_b = other["contact_rollouts"][rollout]["rewards"]
            note = "" if spec["compare_magnitudes"] else "   (reachability only, trajectories diverge)"
            print(f"    {rollout}{note}")
            for term in sorted(CONTACT_DEPENDENT_TERMS):
                va, vb = sums_a.get(term, 0.0), sums_b.get(term, 0.0)
                if abs(va) <= 1e-12 and abs(vb) <= 1e-12:
                    print(f"      {term:20s} {va:12.6f} {vb:12.6f}   both zero")
                    continue
                ratio = max(abs(va), abs(vb)) / max(min(abs(va), abs(vb)), 1e-12)
                print(f"      {term:20s} {va:12.6f} {vb:12.6f}   x{min(ratio, 9999):.2f}")

        fired = set()
        for rollout in CONTACT_ROLLOUTS:
            for term, value in ref["contact_rollouts"][rollout]["rewards"].items():
                if abs(value) > 1e-12:
                    fired.add(term)
        never = sorted(CONTACT_DEPENDENT_TERMS - fired)
        chk(
            "every contact-dependent term fires in at least one rollout",
            not never,
            f"never fired: {never}",
        )

        for rollout in CONTACT_ROLLOUTS:
            sums_a = ref["contact_rollouts"][rollout]["rewards"]
            sums_b = other["contact_rollouts"][rollout]["rewards"]
            resets_a = ref["contact_rollouts"][rollout]["resets"]
            resets_b = other["contact_rollouts"][rollout]["resets"]
            for term in sorted(PHYSICS_FREE_TERMS):
                va, vb = sums_a.get(term, 0.0), sums_b.get(term, 0.0)
                if abs(va) <= 1e-12 and abs(vb) <= 1e-12:
                    continue
                if resets_a or resets_b:
                    print(
                        f"    [INFO] {term} in {rollout}: {va:.6f} vs {vb:.6f}, not compared "
                        f"exactly -- the episode ended mid-rollout ({resets_a} vs {resets_b} "
                        "resets) and a reset clears the action history"
                    )
                    continue
                chk(
                    f"{term} in {rollout} agrees exactly (no reset, so it never reads the simulator)",
                    abs(va - vb) <= TOL_REWARD,
                    f"{va:.9f} vs {vb:.9f}",
                )

        worst_term, worst_ratio = None, 1.0
        for rollout, spec in CONTACT_ROLLOUTS.items():
            if not spec["compare_magnitudes"]:
                continue
            sums_a = ref["contact_rollouts"][rollout]["rewards"]
            sums_b = other["contact_rollouts"][rollout]["rewards"]
            for term in CONTACT_DEPENDENT_TERMS - PHYSICS_FREE_TERMS:
                va, vb = sums_a.get(term, 0.0), sums_b.get(term, 0.0)
                if max(abs(va), abs(vb)) < CONTACT_TERM_FLOOR:
                    continue
                ratio = max(abs(va), abs(vb)) / max(min(abs(va), abs(vb)), 1e-12)
                allowed = TERM_RATIO_ALLOWANCE.get(term, MAX_CONTACT_TERM_RATIO)
                excess = ratio / allowed
                if excess > worst_ratio:
                    worst_term = f"{term} in {rollout} (x{min(ratio, 9999):.2f}, allowed x{allowed:g})"
                    worst_ratio = excess
        chk(
            "no contact-dependent term exceeds its allowed ratio",
            worst_ratio <= 1.0,
            f"worst {worst_term}" if worst_term else "all within allowance",
        )

        za = ref["rollout"]["steps"][-1]
        zb = other["rollout"]["steps"][-1]
        print(
            f"  [ref] after 3 zero-action steps: root drift {_maxdiff(za['root_pos'], zb['root_pos']):.4f} m, "
            f"joint drift {_maxdiff(za['q'], zb['q']):.4f} rad "
            "(engine difference, not a failure)"
        )

    print("\n=== RESULT:", "ALL OK" if not fails else f"{len(fails)} FAIL: {fails}", "===")
    return 1 if fails else 0


def dump(sim: str) -> int:
    env = build_parity_env(sim)
    out = {"script": script_fingerprint(), "asset": dump_asset(env), "states": {}}
    for name, spec in STATES.items():
        out["states"][name] = inject_and_measure(env, spec)
    out["contact_rollouts"] = contact_rollouts(env)
    out["rollout"] = zero_action_rollout(env)

    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / f"{sim}.json"
    path.write_text(json.dumps(out, indent=1))

    print(f"wrote {path}")
    print("joints     :", out["asset"]["joints"])
    print("total mass :", round(out["asset"]["total_mass"], 4))
    print("geoms      :", len(out["asset"]["collision_geoms"]), "->", out["asset"]["geom_multiset"])
    for name, record in out["states"].items():
        print(f"{name:26s} total_reward {record['total_reward']:+.6f}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sim", choices=list(SIMS))
    group.add_argument("--compare", action="store_true")
    args = parser.parse_args()
    return compare() if args.compare else dump(args.sim)


if __name__ == "__main__":
    raise SystemExit(main())
