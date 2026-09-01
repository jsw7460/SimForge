"""Do mjlab and Newton hand the policy the same depth image?

A camera that works on one backend and disagrees with the other is the
worst kind of broken, because both pictures look fine. The only way to
know is to put the two simulators in the SAME state and compare the
images pixel by pixel.

The state is imposed rather than simulated. Two solvers integrating the
same actions drift apart within a few steps, and after that any depth
difference is a physics difference wearing a camera's clothes. So this
writes identical joint angles and identical object poses into both
scenes, runs forward kinematics, renders, and compares. What is left is
the camera alone.

The scene is deliberately cluttered: a bare table gives a depth image
that is one flat plane, which matches trivially. Cubes strewn across the
workspace put edges, occlusions and holes in the frame — the places
where two renderers actually disagree.

The checks build on each other and are reported in that order:

1. the two scenes hold the same joints and the same objects
2. the wrist link is therefore in the same place in both
3. the depth images agree

Failing 1 or 2 makes 3 meaningless, which is why they are separate.

Run::

    jaxpy -m jaxrlworld.scripts.diag.yam.wrist_camera_cross_sim_diag
    jaxpy -m jaxrlworld.scripts.diag.yam.wrist_camera_cross_sim_diag --samples 200 --clutter 24 --resolution 64
"""

from __future__ import annotations

import argparse
import os

# Two backends in one process is the whole point here, so the
# single-backend guard is bypassed BEFORE anything imports a simulator.
os.environ.setdefault("JAXRLWORLD_ALLOW_MULTI_SIM", "1")

import torch  # noqa: E402

from jaxrlworld.rl.configs.presets.yam_arm.base import CUBE_HALF, TABLE_TOP_Z  # noqa: E402
from jaxrlworld.rl.configs.presets.yam_lift.vision import CAMERA_SENSOR, YamLiftVisionConfig  # noqa: E402
from jaxrlworld.rl.configs.scene.unified_entity_config import InitialStateCfg, RigidObjectCfg  # noqa: E402
from jaxrlworld.rl.runners import BaseRunner  # noqa: E402

_CLUTTER_PREFIX = "clutter_"
"""Extra cubes, added to both scenes from the same URDF."""

# A box over the table, reachable by the arm and inside the camera's
# view for a good share of the sampled poses.
_CLUTTER_X = (0.15, 0.55)
_CLUTTER_Y = (-0.25, 0.25)
_CLUTTER_Z = (TABLE_TOP_Z + CUBE_HALF, TABLE_TOP_Z + 0.25)


def _parking_spot(index: int) -> tuple[float, float, float]:
    """Somewhere in the air, touching nothing, one per clutter cube."""
    row, column = divmod(index, 8)
    return (0.15 + 0.1 * column, -0.35 + 0.1 * row, TABLE_TOP_Z + 0.6 + 0.1 * row)


def _build_env(sim_type: str, args, clutter_names: list[str]):
    """The vision preset on one backend, with the clutter added."""
    cfg = YamLiftVisionConfig(
        sim_type=sim_type,
        num_envs=args.num_envs,
        camera_width=args.resolution,
        camera_height=args.resolution,
    )
    cfg.visible_geometry = args.geometry
    if args.near_clip is not None:
        cfg.near_clip = args.near_clip
    cfgs = cfg.build()
    cube_cfg = cfgs.scene.rigid_objects["cube"]
    for index, name in enumerate(clutter_names):
        cfgs.scene.rigid_objects[name] = RigidObjectCfg(
            urdf_path=cube_cfg.urdf_path,
            floating=True,
            # Parked in mid-air, well apart. Every sample overwrites this
            # pose, so it only has to touch nothing: cubes resting on the
            # table at build time put hundreds of contacts in the initial
            # state, and mjwarp sizes its contact buffers from exactly
            # that state.
            init_state=InitialStateCfg(pos=_parking_spot(index)),
        )
    # Nothing here steps physics, but mjwarp sizes its contact and
    # constraint buffers at build time from the reference state, and a
    # scene with a dozen extra bodies has far more of both than the
    # preset's defaults allow. Scaled with the clutter rather than set to
    # a number that silently stops being enough.
    bodies = len(clutter_names) + 1
    if getattr(cfgs.scene, "nconmax", None) is not None:
        cfgs.scene.nconmax = max(cfgs.scene.nconmax, 128 * bodies)
    if getattr(cfgs.scene, "njmax", None) is not None:
        cfgs.scene.njmax = max(cfgs.scene.njmax, 512 * bodies)
    env = BaseRunner._create_env_from_config(cfgs)
    env.reset()
    return env, cfg


def _sample_state(env, clutter_names: list[str], generator: torch.Generator, device, limits, joint_names):
    """One random arm pose and one random layout of the objects.

    Drawn on the CPU from a seeded generator so both backends are handed
    the same numbers, rather than each drawing its own.
    """
    # Limits come from Newton, which exposes the hard ones straight from
    # the model; mjlab keeps only soft limits and refuses this call. Both
    # load the same MJCF, so one set describes both arms.
    lower, upper = limits
    # Stay just inside the stops: a joint written exactly onto its limit
    # is where the two backends' clamping is most likely to differ, which
    # would show up as a physics disagreement rather than a camera one.
    middle = 0.5 * (lower + upper)
    half = 0.475 * (upper - lower)
    lower, upper = (middle - half).cpu(), (middle + half).cpu()
    # device="cpu" on every draw, spelled out: Genesis sets torch's
    # default device to the GPU, and the point of sampling here is that
    # both backends are handed the SAME numbers from one seeded CPU
    # generator rather than each drawing its own.
    shape = (env.num_envs, lower.shape[-1])
    unit = torch.rand(shape, generator=generator, device="cpu")
    joint_pos = lower + unit * (upper - lower)
    joint_by_name = {name: joint_pos[:, index] for index, name in enumerate(joint_names)}

    objects = {}
    for name in ["cube", *clutter_names]:
        pos = torch.empty((env.num_envs, 3), device="cpu")
        for axis, (low, high) in enumerate((_CLUTTER_X, _CLUTTER_Y, _CLUTTER_Z)):
            pos[:, axis] = low + torch.rand((env.num_envs,), generator=generator, device="cpu") * (high - low)
        yaw = torch.rand((env.num_envs,), generator=generator, device="cpu") * 6.2831853
        quat = torch.zeros((env.num_envs, 4), device="cpu")
        quat[:, 0] = torch.cos(0.5 * yaw)
        quat[:, 3] = torch.sin(0.5 * yaw)
        objects[name] = (pos, quat)

    return joint_by_name, {k: (p.to(device), q.to(device)) for k, (p, q) in objects.items()}


def _impose(env, joint_by_name: dict[str, torch.Tensor], objects: dict, device) -> None:
    """Write the sampled state into one backend and settle the kinematics.

    Joints are addressed BY NAME. Writing a bare vector assumes both
    backends order their actuated joints the same way, and if they do not
    the two arms end up in different poses from the same numbers — while
    reading the vector back, in each backend's own order, returns exactly
    what was written on both. The check that would catch it is the one
    the mistake defeats.
    """
    names = list(env.act_manager.actuated_joint_names)
    joint_pos = torch.stack([joint_by_name[name] for name in names], dim=-1).to(device)

    writer = env.get_robot_state_writer("robot")
    writer.set_dof_positions(joint_pos)
    writer.set_dof_velocities(torch.zeros_like(joint_pos))
    writer.eval_fk()

    for name, (pos, quat) in objects.items():
        object_writer = env.get_root_state_writer(name)
        object_writer.set_root_pose(pos.to(device) + env.scene_manager.env_origins, quat.to(device))
        object_writer.eval_fk()

    # Both hooks are needed, and each backend only implements one of
    # them. Newton does its forward kinematics in the writer's eval_fk;
    # mjlab's eval_fk is a no-op because it normally recomputes
    # everything inside Simulation.step(), and nothing here steps. Skip
    # this and mjlab renders the pose it was built with, forever, while
    # reading back exactly the joint angles that were written.
    env._post_reset_forward()

    # The observation is what the policy reads, and it is what carries
    # the rendered image, so go through the same path the env uses.
    env._render_sensors()
    env.obs_manager.process_observations(update_history=False)


def _depth(env) -> torch.Tensor:
    """Raw sensor depth, ``(num_envs, H, W)``, metres."""
    return env.scene_manager.camera_sensors[CAMERA_SENSOR].data.depth[..., 0].float()


def _wrist_pose(env, body_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Pose of the link the camera rides on, in its own env's frame.

    Env-local, not world: the two backends lay their environments out on
    different grids, so identical arms stand metres apart in world
    coordinates while being in exactly the same place as far as the
    camera is concerned. Both backends label an entity's bodies with the
    MJCF's own names, so one name works on either.
    """
    robot_data = env.get_robot_data("robot")
    index = robot_data.find_body_index(body_name)
    pos = robot_data.body_pos_w_all[:, index] - env.scene_manager.env_origins
    return pos, robot_data.body_quat_w_all[:, index]


def _edge_mask(depth: torch.Tensor, tolerance: float) -> torch.Tensor:
    """Pixels sitting on a depth discontinuity — a silhouette.

    At a silhouette a half-pixel difference in where a ray is aimed
    swings the answer between a near surface and whatever is behind it,
    so two independent renderers cannot be expected to agree there. At
    32x32 one pixel spans nearly two degrees, which at half a metre is
    over a centimetre of surface, so this is not a small effect and it
    is not a defect either.
    """
    padded = torch.nn.functional.pad(depth.unsqueeze(1), (1, 1, 1, 1), mode="replicate")
    local_max = torch.nn.functional.max_pool2d(padded, 3, stride=1)
    local_min = -torch.nn.functional.max_pool2d(-padded, 3, stride=1)
    return ((local_max - local_min) > tolerance).squeeze(1)


def _as_policy_sees(depth: torch.Tensor, cutoff: float, near_clip: float) -> torch.Tensor:
    """The observation term's own rule, so both backends meet one."""
    if near_clip > 0.0:
        depth = torch.where(depth < near_clip, torch.zeros_like(depth), depth)
    return torch.clamp(torch.clamp(depth, min=0.01, max=cutoff) / cutoff, 0.0, 1.0)


def _depth_profile(depth: torch.Tensor) -> str:
    """What one backend's depth looks like, in one line."""
    finite = depth[depth > 0.0]
    if finite.numel() == 0:
        return "every ray missed"
    q = torch.quantile(finite.float(), torch.tensor([0.5, 0.99], device=finite.device))
    far = float((depth > 2.0).float().mean())
    nohit = float((depth <= 0.0).float().mean())
    return (
        f"hit median {float(q[0]):.3f} m  p99 {float(q[1]):.3f} m  "
        f"max {float(depth.max()):.1f} m  beyond 2 m {100.0 * far:.1f}%  no-hit {100.0 * nohit:.1f}%"
    )


def _geometry_report(env, sim_type: str, label: str) -> None:
    """Count what one backend actually has to draw.

    Two backends disagreeing about a scene is meaningless until it is
    known which shapes each is even holding. Each states it its own way —
    mjlab by geom group, Newton by a per-shape visibility flag, Genesis
    through its own renderer — so each is counted the way it can be, from
    the BUILT model rather than from what the config asked for.
    """
    if sim_type == "mujoco":
        mj_model = env.scene_manager.sim.mj_model
        groups: dict[int, int] = {}
        for index in range(mj_model.ngeom):
            group = int(mj_model.geom_group[index])
            groups[group] = groups.get(group, 0) + 1
        print(f"  {label:<8} {mj_model.ngeom} geoms by group: {dict(sorted(groups.items()))}")
    elif sim_type == "newton":
        import warp as wp
        from newton import ShapeFlags

        flags = wp.to_torch(env.scene_manager.model.shape_flags)
        visible = (flags & int(ShapeFlags.VISIBLE)) != 0
        collides = (flags & int(ShapeFlags.COLLIDE_SHAPES)) != 0
        print(
            f"  {label:<8} {flags.numel()} shapes: visible {int(visible.sum())}, "
            f"colliding {int(collides.sum())}, visible-but-not-colliding {int((visible & ~collides).sum())}, "
            f"colliding-but-not-visible {int((collides & ~visible).sum())}"
        )
    else:
        print(f"  {label:<8} {sim_type}: no shape inventory wired up here")


def _chain_report(a_env, b_env) -> None:
    """Where along the arm do the two backends stop agreeing?

    A wrist that is in the wrong place with the joints identical is
    either a base placed differently or one joint's axis interpreted
    differently, and those have nothing to do with each other. Walking
    the chain names which.
    """
    a_bodies = a_env.get_robot_data("robot")
    b_bodies = b_env.get_robot_data("robot")
    a_origins = a_env.scene_manager.env_origins
    b_origins = b_env.scene_manager.env_origins

    print("\n-- 1a. where the two arms stop agreeing --")
    print(f"  {'body':<18} {'Δpos (m)':>12} {'Δrot (1-|dot|)':>16}")
    for name in ("arm", "link_1", "link_2", "link_3", "link_4", "link_5", "link_6"):
        try:
            a_index = a_bodies.find_body_index(name)
            b_index = b_bodies.find_body_index(name)
        except (ValueError, KeyError) as error:
            print(f"  {name:<18} not resolvable: {error}")
            continue
        a_pos = a_bodies.body_pos_w_all[:, a_index] - a_origins
        b_pos = (b_bodies.body_pos_w_all[:, b_index] - b_origins).to(a_pos.device)
        a_quat = a_bodies.body_quat_w_all[:, a_index]
        b_quat = b_bodies.body_quat_w_all[:, b_index].to(a_quat.device)
        pos_gap = float((a_pos - b_pos).abs().max())
        rot_gap = float((1.0 - (a_quat * b_quat).sum(-1).abs()).max())
        print(f"  {name:<18} {pos_gap:12.6f} {rot_gap:16.6f}")

    a_index = a_bodies.find_body_index("arm")
    b_index = b_bodies.find_body_index("arm")
    print("  env 0 base pose, env-local:")
    print(
        f"    mjlab  pos {[round(float(v), 5) for v in a_bodies.body_pos_w_all[0, a_index] - a_origins[0]]}"
        f"  quat {[round(float(v), 5) for v in a_bodies.body_quat_w_all[0, a_index]]}"
    )
    print(
        f"    Newton pos {[round(float(v), 5) for v in b_bodies.body_pos_w_all[0, b_index] - b_origins[0]]}"
        f"  quat {[round(float(v), 5) for v in b_bodies.body_quat_w_all[0, b_index]]}"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-a", default="mujoco", choices=("mujoco", "newton", "genesis"))
    ap.add_argument("--sim-b", default="newton", choices=("mujoco", "newton", "genesis"))
    ap.add_argument("--num-envs", type=int, default=8)
    ap.add_argument("--samples", type=int, default=64, help="Random arm poses to compare.")
    ap.add_argument("--clutter", type=int, default=12, help="Extra cubes strewn over the table.")
    ap.add_argument("--resolution", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--near-clip",
        type=float,
        default=None,
        help="Override the preset's near clip on Newton, metres. 0 disables it.",
    )
    ap.add_argument("--cutoff", type=float, default=0.5, help="Far plane the policy normalises against, metres.")
    ap.add_argument(
        "--geometry",
        default="collision",
        choices=("collision", "visual", "all"),
        help="Which of the asset's two shape descriptions both cameras are shown.",
    )
    ap.add_argument(
        "--tolerance",
        type=float,
        default=0.005,
        help="Metres a pixel may differ by and still count as agreeing.",
    )
    args = ap.parse_args()

    clutter_names = [f"{_CLUTTER_PREFIX}{i}" for i in range(args.clutter)]

    print("=" * 78)
    print(
        f"WRIST DEPTH: {args.sim_a} (A) vs {args.sim_b} (B)  [{args.num_envs} envs x {args.samples} poses, "
        f"{args.resolution}px, {args.clutter} extra cubes, {args.geometry} geometry]"
    )
    print("=" * 78)

    results: dict[str, bool] = {}

    print("\n-- building both backends --")
    a_env, cfg = _build_env(args.sim_a, args, clutter_names)
    b_env, _ = _build_env(args.sim_b, args, clutter_names)
    device = a_env.device
    print(f"  A = {args.sim_a} on {a_env.device}   B = {args.sim_b} on {b_env.device}")

    a_camera = a_env.scene_manager.camera_sensors[CAMERA_SENSOR]
    b_camera = b_env.scene_manager.camera_sensors[CAMERA_SENSOR]
    print(f"  A camera {type(a_camera).__name__}   B camera {type(b_camera).__name__}")
    # Our own cfg, not the backend's translation of it: mjlab converts
    # it into an mjlab CameraSensorCfg, which knows nothing about MJCF
    # resolution.
    our_cfg = next(c for c in a_env.scene_manager.config.cameras if c.name == CAMERA_SENSOR)
    mjcf_path = a_env.scene_manager.config.entities["robot"].mjcf_path
    link_name, offset, optics = our_cfg.resolve(mjcf_path)
    print(f"  both read the arm's MJCF camera as: link {link_name!r}")
    print(f"    offset pos  {[round(v, 6) for v in offset.pos]}")
    print(f"    offset quat {[round(v, 6) for v in offset.quat]}")
    print(f"    fovy {optics.fovy:.4f} deg   sensor {optics.sensorsize}   focal {optics.focal}")

    wrist_body = link_name
    print(f"  wrist body on both backends: {wrist_body!r}")

    a_joint_names = list(a_env.act_manager.actuated_joint_names)
    b_joint_names = list(b_env.act_manager.actuated_joint_names)
    print(f"  A actuated order {a_joint_names}")
    print(f"  B actuated order {b_joint_names}")
    if a_joint_names != b_joint_names:
        print("  -> the two orders DIFFER; joints are written by name, not by position")
    if sorted(a_joint_names) != sorted(b_joint_names):
        raise ValueError("The two backends do not even have the same actuated joints; nothing below is comparable.")

    near_clip = cfg.near_clip
    print(f"  near clip applied to both, in the observation: {near_clip} m")

    joint_limits = b_env.get_robot_data("robot").joint_pos_limits
    generator = torch.Generator().manual_seed(args.seed)

    worst = {
        "sample": -1,
        "max_diff": -1.0,
        "agree_fraction": 2.0,
    }
    totals = {
        "pixels": 0,
        "agreeing": 0,
        "abs_sum": 0.0,
        "joint_gap": 0.0,
        "wrist_pos_gap": 0.0,
        "wrist_quat_gap": 0.0,
        "nohit_a": 0,
        "nohit_b": 0,
        "nohit_both": 0,
        "nohit_either": 0,
        "policy_pixels": 0,
        "policy_agreeing": 0,
        "policy_abs_sum": 0.0,
        "near_pixels": 0,
        "near_agreeing": 0,
        "near_disputed": 0,
        "interior_pixels": 0,
        "interior_agreeing": 0,
        "interior_max": 0.0,
        "interior_policy_agreeing": 0,
        "edge_pixels": 0,
    }
    all_diffs = []
    worst_interior: dict = {"sample": -1, "count": 0, "mj": None, "nt": None, "interior": None, "shape": None}

    print(f"\n-- {args.samples} random poses --")
    for sample in range(args.samples):
        joint_by_name, objects = _sample_state(a_env, clutter_names, generator, device, joint_limits, b_joint_names)
        _impose(a_env, joint_by_name, objects, a_env.device)
        _impose(b_env, joint_by_name, objects, b_env.device)

        # 1. the two scenes really do hold the same state
        # Compared by name, so a differing order shows up as a mismatch
        # rather than hiding behind two consistent readbacks.
        a_joints = a_env.get_robot_data("robot").joint_pos
        b_joints = b_env.get_robot_data("robot").joint_pos.to(a_joints.device)
        nt_order = [b_joint_names.index(name) for name in a_joint_names]
        gap = (a_joints - b_joints[:, nt_order]).abs().max()
        totals["joint_gap"] = max(totals["joint_gap"], float(gap))

        # 2. so the wrist is in the same place
        a_pos, a_quat = _wrist_pose(a_env, wrist_body)
        b_pos, b_quat = _wrist_pose(b_env, wrist_body)
        b_pos = b_pos.to(a_pos.device)
        b_quat = b_quat.to(a_quat.device)
        totals["wrist_pos_gap"] = max(totals["wrist_pos_gap"], float((a_pos - b_pos).abs().max()))
        # Quaternions double-cover, so compare the rotation, not the sign.
        quat_gap = 1.0 - (a_quat * b_quat).sum(-1).abs()
        totals["wrist_quat_gap"] = max(totals["wrist_quat_gap"], float(quat_gap.max()))

        # 3. and the images should agree
        a_depth = _depth(a_env)
        b_depth = _depth(b_env).to(a_depth.device)
        diff = (a_depth - b_depth).abs()

        agreeing = int((diff <= args.tolerance).sum())
        pixels = diff.numel()
        totals["pixels"] += pixels
        totals["agreeing"] += agreeing
        totals["abs_sum"] += float(diff.sum())
        all_diffs.append(diff.flatten().cpu())

        # What the policy is actually handed: mjlab's own normalisation,
        # which saturates everything past the far plane. Two rays grazing
        # a ground plane hundreds of metres away differ enormously in
        # metres and not at all in what the network reads.
        cutoff = args.cutoff

        a_policy = _as_policy_sees(a_depth, cutoff, near_clip)
        b_policy = _as_policy_sees(b_depth, cutoff, near_clip)
        policy_diff = (a_policy - b_policy).abs()
        totals["policy_pixels"] += policy_diff.numel()
        totals["policy_agreeing"] += int((policy_diff <= args.tolerance / cutoff).sum())
        totals["policy_abs_sum"] += float(policy_diff.sum())

        # Near field only: the gripper, the table, the cubes. A pixel
        # counts as disputed when one backend sees something inside the
        # working range and the other does not — that is real geometry
        # missing from one of them, not a grazing-ray artefact.
        a_near = (a_depth > 0.0) & (a_depth <= cutoff)
        b_near = (b_depth > 0.0) & (b_depth <= cutoff)
        both_near = a_near & b_near
        totals["near_pixels"] += int(both_near.sum())
        totals["near_agreeing"] += int(((diff <= args.tolerance) & both_near).sum())
        totals["near_disputed"] += int((a_near ^ b_near).sum())

        # Away from silhouettes the question has one right answer, so
        # this is where a rendering port is either correct or not.
        edge = _edge_mask(a_depth, args.tolerance) | _edge_mask(b_depth, args.tolerance)
        interior = ~edge
        totals["edge_pixels"] += int(edge.sum())
        totals["interior_pixels"] += int(interior.sum())
        totals["interior_agreeing"] += int(((diff <= args.tolerance) & interior).sum())
        totals["interior_policy_agreeing"] += int(((policy_diff <= args.tolerance / cutoff) & interior).sum())
        if int(interior.sum()):
            totals["interior_max"] = max(totals["interior_max"], float(diff[interior].max()))
        # Keep the sample with the most interior disagreement, not the
        # last one: the last is whatever the loop happened to end on, and
        # inspecting it says nothing about the aggregate above.
        b_shape_index = getattr(b_camera.data, "shape_index", None)
        interior_bad = int(((diff > args.tolerance) & interior).sum())
        if interior_bad > worst_interior["count"]:
            worst_interior.update(
                sample=sample,
                count=interior_bad,
                mj=a_depth.clone(),
                nt=b_depth.clone(),
                interior=interior.clone(),
                # Newton alone can say WHICH shape a pixel hit, and that
                # is what turned the last mystery into an answer.
                shape=None if b_shape_index is None else b_shape_index.clone(),
            )

        a_nohit = a_depth <= 0.0
        b_nohit = b_depth <= 0.0
        totals["nohit_a"] += int(a_nohit.sum())
        totals["nohit_b"] += int(b_nohit.sum())
        totals["nohit_both"] += int((a_nohit & b_nohit).sum())
        totals["nohit_either"] += int((a_nohit | b_nohit).sum())

        max_diff = float(diff.max())
        fraction = agreeing / pixels
        if fraction < worst["agree_fraction"]:
            worst = {"sample": sample, "max_diff": max_diff, "agree_fraction": fraction}

        if sample < 3 or (sample + 1) % 16 == 0:
            print(
                f"  sample {sample:4d}: max |Δ| {max_diff:.5f} m   "
                f"within {args.tolerance * 1000:.0f} mm: {100.0 * fraction:.2f}%   "
                f"A depth [{float(a_depth.min()):.3f}, {float(a_depth.max()):.3f}]   "
                f"B [{float(b_depth.min()):.3f}, {float(b_depth.max()):.3f}]"
            )

    diffs = torch.cat(all_diffs)

    print("\n-- 0. what geometry each backend holds --")
    _geometry_report(a_env, args.sim_a, "A")
    _geometry_report(b_env, args.sim_b, "B")
    print("  (a backend drawing shapes the other does not hold cannot match it)")
    _chain_report(a_env, b_env)

    print("\n-- 1. are the two scenes in the same state --")
    print(f"  worst joint-angle gap      {totals['joint_gap']:.3e} rad")
    print(f"  worst wrist position gap   {totals['wrist_pos_gap']:.3e} m   (env-local)")
    print(f"  worst wrist rotation gap   {totals['wrist_quat_gap']:.3e} (1 - |dot|)")
    print("\n-- 1b. what each backend's camera actually sees --")
    print(f"  A  {_depth_profile(a_depth)}")
    print(f"  B  {_depth_profile(b_depth)}")
    print("  (last sample only; a large 'beyond 2 m' share means the camera is")
    print("   seeing a ground plane that stretches to the horizon)")
    results["the_same_joint_angles_land_in_both"] = totals["joint_gap"] < 1e-5
    results["the_wrist_ends_up_in_the_same_place"] = totals["wrist_pos_gap"] < 1e-4
    results["the_wrist_ends_up_in_the_same_pose"] = totals["wrist_quat_gap"] < 1e-6

    print("\n-- 2. do the depth images agree --")
    fraction = totals["agreeing"] / totals["pixels"]
    quantiles = torch.tensor([0.5, 0.9, 0.99, 0.999], device=diffs.device)
    q = torch.quantile(diffs.float(), quantiles)
    print(f"  pixels compared            {totals['pixels']:,}")
    print(f"  within {args.tolerance * 1000:.0f} mm             {100.0 * fraction:.3f}%")
    print(f"  mean |Δ|                   {totals['abs_sum'] / totals['pixels']:.6f} m")
    print(f"  median |Δ|                 {float(q[0]):.6f} m")
    print(f"  p90 / p99 / p99.9 |Δ|      {float(q[1]):.6f} / {float(q[2]):.6f} / {float(q[3]):.6f} m")
    print(f"  max |Δ|                    {float(diffs.max()):.6f} m")
    print(
        f"  worst pose was sample {worst['sample']} "
        f"({100.0 * worst['agree_fraction']:.2f}% within tolerance, max {worst['max_diff']:.5f} m)"
    )
    results["the_median_pixel_agrees"] = float(q[0]) <= args.tolerance

    print("\n-- 2a. away from silhouettes, where the answer is well defined --")
    interior_total = max(totals["interior_pixels"], 1)
    interior_fraction = totals["interior_agreeing"] / interior_total
    interior_policy = totals["interior_policy_agreeing"] / interior_total
    edge_share = totals["edge_pixels"] / max(totals["pixels"], 1)
    print(f"  interior pixels             {totals['interior_pixels']:,} ({100.0 * (1 - edge_share):.1f}% of the image)")
    print(f"  agreeing to {args.tolerance * 1000:.0f} mm            {100.0 * interior_fraction:.4f}%")
    print(f"  worst interior |Δ|          {totals['interior_max']:.6f} m")
    print(f"  agreeing as the policy sees them: {100.0 * interior_policy:.4f}%")
    # Thresholds, and why they are these numbers rather than 1.0.
    #
    # The raw metre comparison includes the band nearer than the sensor's
    # minimum range, where the camera is not looking at a surface but
    # buried in one and the two renderers answer differently by right —
    # so it is reported, not judged.
    #
    # The policy's view has that band clipped away on both sides, and
    # everything left has been traced: silhouettes, where a half pixel of
    # ray direction decides between a near surface and what is behind it.
    # 0.995 is the measured 0.9972 with room for the sampling to land
    # differently, not a number picked to make this pass; the figure
    # itself is printed above and a real regression moves it visibly.
    print("  (the raw figure is reported, not judged — it includes the band the policy never sees)")
    results["the_policy_sees_the_same_image_off_the_silhouettes"] = interior_policy > 0.995

    print(f"\n-- 2b. what the policy is handed (normalised against a {args.cutoff} m far plane) --")
    policy_fraction = totals["policy_agreeing"] / totals["policy_pixels"]
    print(f"  agreeing pixels            {100.0 * policy_fraction:.3f}%")
    print(f"  mean |Δ| (0-1 scale)       {totals['policy_abs_sum'] / totals['policy_pixels']:.6f}")
    print("  (silhouette pixels included, so this sits below the interior figure above)")

    print("\n-- 2c. the near field, where the task happens --")
    near_total = max(totals["near_pixels"], 1)
    near_fraction = totals["near_agreeing"] / near_total
    disputed = totals["near_disputed"] / max(totals["policy_pixels"], 1)
    print(f"  pixels both see within {args.cutoff} m:  {totals['near_pixels']:,}")
    print(f"  of those, agreeing to {args.tolerance * 1000:.0f} mm: {100.0 * near_fraction:.3f}%")
    print(f"  seen by one backend only:         {100.0 * disputed:.3f}% of all pixels")
    print("  (also silhouette-inclusive; a cube edge is one backend seeing it and the other")
    print("   seeing past it, which is the same sampling difference counted a second way)")

    print("\n-- 2d. WHERE the disagreement sits --")

    # A silhouette is where a half-pixel difference in ray direction
    # swings the answer between a near surface and whatever is behind
    # it, so edge pixels disagreeing is sampling, not modelling. Broad
    # patches disagreeing is one backend holding geometry the other does
    # not. Told apart by dilating the depth discontinuities and asking
    # how much of the disagreement lands on them.
    def _edges(depth: torch.Tensor) -> torch.Tensor:
        pad = torch.nn.functional.pad(depth.unsqueeze(1), (1, 1, 1, 1), mode="replicate")
        local_max = torch.nn.functional.max_pool2d(pad, 3, stride=1)
        local_min = -torch.nn.functional.max_pool2d(-pad, 3, stride=1)
        return ((local_max - local_min) > args.tolerance).squeeze(1)

    disagree = (a_depth - b_depth).abs() > args.tolerance
    edge = _edges(a_depth) | _edges(b_depth)
    disagreeing = int(disagree.sum())
    on_edge = int((disagree & edge).sum())
    share = on_edge / max(disagreeing, 1)
    print(f"  disagreeing pixels (last sample): {disagreeing:,}")
    print(f"  of those, on a depth edge:        {100.0 * share:.1f}%")
    print("  (high = the two renderers resolve silhouettes differently, which is sampling;")
    print("   low = one of them is holding geometry the other is not)")
    results["the_disagreement_is_confined_to_silhouettes"] = share > 0.9 or disagreeing == 0

    print("\n-- 2e. the worst interior pixels, spelled out --")
    # An aggregate cannot say WHAT is different. These are the actual
    # numbers at the pixels that disagree most while sitting away from
    # any depth discontinuity, with the neighbourhood each backend saw.
    if worst_interior["mj"] is None:
        print("  none: every interior pixel of every sample agrees")
        a_depth = b_depth = None
    else:
        a_depth = worst_interior["mj"]
        b_depth = worst_interior["nt"]
        print(f"  from sample {worst_interior['sample']}, which had {worst_interior['count']:,} of them")
    masked = (
        None
        if a_depth is None
        else torch.where(worst_interior["interior"], (a_depth - b_depth).abs(), torch.zeros_like(a_depth))
    )
    flat = None if masked is None else masked.flatten()
    count = 0 if flat is None else min(5, int((flat > args.tolerance).sum()))
    if count:
        for rank, index in enumerate(torch.topk(flat, count).indices.tolist()):
            env_index, row, column = (
                index // (a_depth.shape[1] * a_depth.shape[2]),
                (index // a_depth.shape[2]) % a_depth.shape[1],
                index % a_depth.shape[2],
            )
            a_value = float(a_depth[env_index, row, column])
            b_value = float(b_depth[env_index, row, column])
            rows = slice(max(row - 1, 0), row + 2)
            columns = slice(max(column - 1, 0), column + 2)
            print(
                f"  #{rank}: env {env_index} pixel ({row},{column})  "
                f"A {a_value:.4f} m   B {b_value:.4f} m   Δ {abs(a_value - b_value):.4f} m"
            )
            print(f"      A 3x3 {[round(float(v), 3) for v in a_depth[env_index, rows, columns].flatten()]}")
            print(f"      B 3x3 {[round(float(v), 3) for v in b_depth[env_index, rows, columns].flatten()]}")
            if worst_interior["shape"] is not None:
                # Newton alone reports which shape a pixel hit, and that
                # is what turned the last mystery into an answer.
                shape_id = int(worst_interior["shape"][env_index, row, column])
                keys = b_env.scene_manager.model.shape_label
                name = keys[shape_id] if 0 <= shape_id < len(keys) else "(nothing)"
                print(f"      B hit shape {shape_id}: {name}")
    print("  (two similar neighbourhoods offset by a pixel = a silhouette the edge mask")
    print("   was too narrow to catch; a flat patch differing = real geometry)")

    print("\n-- 2f. what A discards that B keeps --")
    # mjwarp builds its rays from a near plane (render_util.compute_ray
    # takes znear); a surface closer than that is not rendered and the
    # pixel comes back as a miss. Newton raytraces from the camera
    # origin with no such plane. If every pixel mjlab drops is nearer
    # than one distance, that distance IS the near plane, and the
    # difference is a documented property of the two renderers rather
    # than a fault in either.
    if args.sim_a == "mujoco":
        mj_model = a_env.scene_manager.sim.mj_model
        znear = float(mj_model.vis.map.znear) * float(mj_model.stat.extent)
        print(f"  A's near plane: znear {mj_model.vis.map.znear} x extent {mj_model.stat.extent:.4f} = {znear:.5f} m")
    else:
        znear = 0.0
        print("  A is not mjlab; no near plane to read")

    if worst_interior["mj"] is not None:
        a_worst, b_worst = worst_interior["mj"], worst_interior["nt"]
        # Interior only. At a silhouette mjlab legitimately reports a
        # miss where Newton reports the object, and counting those here
        # would drown the question in the answer already known.
        dropped = (a_worst <= 0.0) & (b_worst > 0.0) & worst_interior["interior"]
        kept = (a_worst > 0.0) & (b_worst > 0.0)
        if int(dropped.sum()):
            values = b_worst[dropped]
            quantiles = torch.quantile(values.float(), torch.tensor([0.5, 0.9, 1.0], device=values.device))
            print(
                f"  interior pixels mjlab missed and Newton hit: {int(dropped.sum()):,}\n"
                f"    Newton distance there: min {float(values.min()):.4f}  median {float(quantiles[0]):.4f}  "
                f"p90 {float(quantiles[1]):.4f}  max {float(quantiles[2]):.4f} m"
            )
            print(f"    beyond the near plane: {100.0 * float((values > znear).float().mean()):.1f}% of them")
            print(f"  for comparison, where both hit, A's nearest hit is {float(a_worst[kept].min()):.4f} m")
            print("    (reported, not judged: the near plane explains most of these but not all,")
            print("     and section 2g measures what actually removes them)")
        else:
            print("  none in the worst sample")

    print("\n-- 2g. how much a near clip on BOTH would buy --")
    # Within a few centimetres of the lens the two backends disagree
    # about which near surface is even there — one renders the camera's
    # own housing, the other clips it, and which of them does so swings
    # with the pose. The real sensor cannot measure that close at all,
    # so the honest fix is to discard the band on both sides rather than
    # to make one imitate the other.
    if worst_interior["mj"] is not None:
        a_worst, b_worst = worst_interior["mj"], worst_interior["nt"]
        interior_worst = worst_interior["interior"]
        for clip in (0.0, znear, 0.025, 0.04, 0.07):
            # Applied to BOTH. Clipping one backend only trades the
            # pixels it saw and the other did not for the pixels the
            # other saw and it did not, which is how the first attempt at
            # this made the agreement worse.
            a_clipped = torch.where(a_worst < clip, torch.zeros_like(a_worst), a_worst)
            b_clipped = torch.where(b_worst < clip, torch.zeros_like(b_worst), b_worst)
            agree = ((a_clipped - b_clipped).abs() <= args.tolerance) & interior_worst
            fraction = int(agree.sum()) / max(int(interior_worst.sum()), 1)
            label = " (mjlab's near plane)" if clip == znear else ""
            label = " (the real D405's minimum range)" if clip == 0.07 else label
            print(f"  clip at {clip:.5f} m -> interior agreement {100.0 * fraction:.3f}%{label}")
        print("  (worst sample only; a clip that helps here should be confirmed over all of them)")

    print("\n-- 3. do they agree about what they did NOT hit --")
    union = max(totals["nohit_either"], 1)
    iou = totals["nohit_both"] / union
    print(f"  A no-hit pixels            {totals['nohit_a']:,}")
    print(f"  B no-hit pixels            {totals['nohit_b']:,}")
    print(f"  agreement (IoU)            {iou:.4f}")
    print("  (a ray grazing the ground plane hits it hundreds of metres away in one backend")
    print("   and misses in the other; the policy saturates both to its far plane)")

    print("\n-- 4. is the comparison worth anything --")
    # A pair of images that are constant, or a scene the camera never
    # sees into, would pass every check above by being empty.
    spread = float(diffs.numel() and a_depth.std())
    print(f"  depth spread within one frame (A): {spread:.5f} m")
    print(f"  distinct depths in the last frame:     {int(torch.unique(a_depth).numel()):,}")
    results["the_image_has_structure_to_compare"] = spread > 1e-3

    print("\n" + "=" * 78)
    ok = True
    for name, passed in results.items():
        print(f"  {name:<48}: {'PASS' if passed else 'FAIL'}")
        ok = ok and passed
    print(f"  {'OVERALL':<48}: {'PASS' if ok else 'FAIL'}")
    print("=" * 78)
    if not ok:
        print("  With the arms provably in the same pose, an interior disagreement is a")
        print("  RENDERER difference. Suspect, in order: the geometry each backend was")
        print("  given (section 0), the field of view each derived, and the camera offset")
        print("  composed from the MJCF. Sections 2b-2d and 3 are reported, not judged:")
        print("  they count silhouette pixels, where two renderers cannot be made to agree.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
