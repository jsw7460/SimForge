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

    jaxpy -m rlworld.scripts.diag.wrist_camera_cross_sim_diag
    jaxpy -m rlworld.scripts.diag.wrist_camera_cross_sim_diag --samples 200 --clutter 24 --resolution 64
"""

from __future__ import annotations

import argparse
import os

# Two backends in one process is the whole point here, so the
# single-backend guard is bypassed BEFORE anything imports a simulator.
os.environ.setdefault("JAXRLWORLD_ALLOW_MULTI_SIM", "1")

import torch  # noqa: E402

from rlworld.rl.configs.presets.yam_arm.base import CUBE_HALF, TABLE_TOP_Z  # noqa: E402
from rlworld.rl.configs.presets.yam_lift.vision import CAMERA_SENSOR, YamLiftVisionConfig  # noqa: E402
from rlworld.rl.configs.scene.unified_entity_config import InitialStateCfg, RigidObjectCfg  # noqa: E402
from rlworld.rl.runners import BaseRunner  # noqa: E402

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
    return env


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
    shape = (env.num_envs, lower.shape[-1])
    unit = torch.rand(shape, generator=generator)
    joint_pos = lower + unit * (upper - lower)
    joint_by_name = {name: joint_pos[:, index] for index, name in enumerate(joint_names)}

    objects = {}
    for name in ["cube", *clutter_names]:
        pos = torch.empty((env.num_envs, 3))
        for axis, (low, high) in enumerate((_CLUTTER_X, _CLUTTER_Y, _CLUTTER_Z)):
            pos[:, axis] = low + torch.rand((env.num_envs,), generator=generator) * (high - low)
        yaw = torch.rand((env.num_envs,), generator=generator) * 6.2831853
        quat = torch.zeros((env.num_envs, 4))
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
    return env.scene_manager.sensors[CAMERA_SENSOR].data.depth[..., 0].float()


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


def _geometry_report(mj_env, nt_env) -> None:
    """Count what each backend actually has to draw.

    The two disagreeing about a scene is meaningless until it is known
    which shapes each one is even holding. mjlab filters by geom group;
    Newton by a per-shape visibility flag. Both are counted from the
    built model rather than from what the config asked for.
    """
    import warp as wp
    from newton import ShapeFlags

    print("\n-- 0. what geometry each backend holds --")

    mj_model = mj_env.scene_manager.sim.mj_model
    groups: dict[int, int] = {}
    for index in range(mj_model.ngeom):
        groups[int(mj_model.geom_group[index])] = groups.get(int(mj_model.geom_group[index]), 0) + 1
    print(f"  mjlab  {mj_model.ngeom} geoms by group: {dict(sorted(groups.items()))}")

    flags = wp.to_torch(nt_env.scene_manager.model.shape_flags)
    visible = (flags & int(ShapeFlags.VISIBLE)) != 0
    collides = (flags & int(ShapeFlags.COLLIDE_SHAPES)) != 0
    print(
        f"  Newton {flags.numel()} shapes: visible {int(visible.sum())}, "
        f"colliding {int(collides.sum())}, visible-but-not-colliding {int((visible & ~collides).sum())}, "
        f"colliding-but-not-visible {int((collides & ~visible).sum())}"
    )
    print("  (a backend drawing shapes the other does not hold cannot match it)")


def _chain_report(mj_env, nt_env) -> None:
    """Where along the arm do the two backends stop agreeing?

    A wrist that is in the wrong place with the joints identical is
    either a base placed differently or one joint's axis interpreted
    differently, and those have nothing to do with each other. Walking
    the chain names which.
    """
    mj_bodies = mj_env.get_robot_data("robot")
    nt_bodies = nt_env.get_robot_data("robot")
    mj_origins = mj_env.scene_manager.env_origins
    nt_origins = nt_env.scene_manager.env_origins

    print("\n-- 1a. where the two arms stop agreeing --")
    print(f"  {'body':<18} {'Δpos (m)':>12} {'Δrot (1-|dot|)':>16}")
    for name in ("arm", "link_1", "link_2", "link_3", "link_4", "link_5", "link_6"):
        try:
            mj_index = mj_bodies.find_body_index(name)
            nt_index = nt_bodies.find_body_index(name)
        except (ValueError, KeyError) as error:
            print(f"  {name:<18} not resolvable: {error}")
            continue
        mj_pos = mj_bodies.body_pos_w_all[:, mj_index] - mj_origins
        nt_pos = (nt_bodies.body_pos_w_all[:, nt_index] - nt_origins).to(mj_pos.device)
        mj_quat = mj_bodies.body_quat_w_all[:, mj_index]
        nt_quat = nt_bodies.body_quat_w_all[:, nt_index].to(mj_quat.device)
        pos_gap = float((mj_pos - nt_pos).abs().max())
        rot_gap = float((1.0 - (mj_quat * nt_quat).sum(-1).abs()).max())
        print(f"  {name:<18} {pos_gap:12.6f} {rot_gap:16.6f}")

    mj_index = mj_bodies.find_body_index("arm")
    nt_index = nt_bodies.find_body_index("arm")
    print("  env 0 base pose, env-local:")
    print(
        f"    mjlab  pos {[round(float(v), 5) for v in mj_bodies.body_pos_w_all[0, mj_index] - mj_origins[0]]}"
        f"  quat {[round(float(v), 5) for v in mj_bodies.body_quat_w_all[0, mj_index]]}"
    )
    print(
        f"    Newton pos {[round(float(v), 5) for v in nt_bodies.body_pos_w_all[0, nt_index] - nt_origins[0]]}"
        f"  quat {[round(float(v), 5) for v in nt_bodies.body_quat_w_all[0, nt_index]]}"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", type=int, default=8)
    ap.add_argument("--samples", type=int, default=64, help="Random arm poses to compare.")
    ap.add_argument("--clutter", type=int, default=12, help="Extra cubes strewn over the table.")
    ap.add_argument("--resolution", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
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
        f"WRIST DEPTH: mjlab vs Newton  [{args.num_envs} envs x {args.samples} poses, "
        f"{args.resolution}px, {args.clutter} extra cubes, {args.geometry} geometry]"
    )
    print("=" * 78)

    results: dict[str, bool] = {}

    print("\n-- building both backends --")
    mj_env = _build_env("mujoco", args, clutter_names)
    nt_env = _build_env("newton", args, clutter_names)
    device = mj_env.device
    print(f"  mjlab  device {mj_env.device}   Newton device {nt_env.device}")

    mj_camera = mj_env.scene_manager.sensors[CAMERA_SENSOR]
    nt_camera = nt_env.scene_manager.sensors[CAMERA_SENSOR]
    print(f"  mjlab  camera {type(mj_camera).__name__}  geom groups {mj_camera.cfg.enabled_geom_groups}")
    print(f"  Newton camera {type(nt_camera).__name__}")
    print(f"  Newton reads the arm's MJCF camera as: link {nt_camera._link_name!r}")
    print(f"    offset pos  {[round(float(v), 6) for v in nt_camera._offset_pos]}")
    print(f"    offset quat {[round(float(v), 6) for v in nt_camera._offset_quat]}")

    wrist_body = nt_camera._link_name
    print(f"  wrist body on both backends: {wrist_body!r}")

    mj_joint_names = list(mj_env.act_manager.actuated_joint_names)
    nt_joint_names = list(nt_env.act_manager.actuated_joint_names)
    print(f"  mjlab  actuated order {mj_joint_names}")
    print(f"  Newton actuated order {nt_joint_names}")
    if mj_joint_names != nt_joint_names:
        print("  -> the two orders DIFFER; joints are written by name, not by position")
    if sorted(mj_joint_names) != sorted(nt_joint_names):
        raise ValueError("The two backends do not even have the same actuated joints; nothing below is comparable.")

    joint_limits = nt_env.get_robot_data("robot").joint_pos_limits
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
        "nohit_mj": 0,
        "nohit_nt": 0,
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

    print(f"\n-- {args.samples} random poses --")
    for sample in range(args.samples):
        joint_by_name, objects = _sample_state(mj_env, clutter_names, generator, device, joint_limits, nt_joint_names)
        _impose(mj_env, joint_by_name, objects, mj_env.device)
        _impose(nt_env, joint_by_name, objects, nt_env.device)

        # 1. the two scenes really do hold the same state
        # Compared by name, so a differing order shows up as a mismatch
        # rather than hiding behind two consistent readbacks.
        mj_joints = mj_env.get_robot_data("robot").joint_pos
        nt_joints = nt_env.get_robot_data("robot").joint_pos.to(mj_joints.device)
        nt_order = [nt_joint_names.index(name) for name in mj_joint_names]
        gap = (mj_joints - nt_joints[:, nt_order]).abs().max()
        totals["joint_gap"] = max(totals["joint_gap"], float(gap))

        # 2. so the wrist is in the same place
        mj_pos, mj_quat = _wrist_pose(mj_env, wrist_body)
        nt_pos, nt_quat = _wrist_pose(nt_env, wrist_body)
        nt_pos = nt_pos.to(mj_pos.device)
        nt_quat = nt_quat.to(mj_quat.device)
        totals["wrist_pos_gap"] = max(totals["wrist_pos_gap"], float((mj_pos - nt_pos).abs().max()))
        # Quaternions double-cover, so compare the rotation, not the sign.
        quat_gap = 1.0 - (mj_quat * nt_quat).sum(-1).abs()
        totals["wrist_quat_gap"] = max(totals["wrist_quat_gap"], float(quat_gap.max()))

        # 3. and the images should agree
        mj_depth = _depth(mj_env)
        nt_depth = _depth(nt_env).to(mj_depth.device)
        diff = (mj_depth - nt_depth).abs()

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
        mj_policy = torch.clamp(torch.clamp(mj_depth, min=0.01, max=cutoff) / cutoff, 0.0, 1.0)
        nt_policy = torch.clamp(torch.clamp(nt_depth, min=0.01, max=cutoff) / cutoff, 0.0, 1.0)
        policy_diff = (mj_policy - nt_policy).abs()
        totals["policy_pixels"] += policy_diff.numel()
        totals["policy_agreeing"] += int((policy_diff <= args.tolerance / cutoff).sum())
        totals["policy_abs_sum"] += float(policy_diff.sum())

        # Near field only: the gripper, the table, the cubes. A pixel
        # counts as disputed when one backend sees something inside the
        # working range and the other does not — that is real geometry
        # missing from one of them, not a grazing-ray artefact.
        mj_near = (mj_depth > 0.0) & (mj_depth <= cutoff)
        nt_near = (nt_depth > 0.0) & (nt_depth <= cutoff)
        both_near = mj_near & nt_near
        totals["near_pixels"] += int(both_near.sum())
        totals["near_agreeing"] += int(((diff <= args.tolerance) & both_near).sum())
        totals["near_disputed"] += int((mj_near ^ nt_near).sum())

        # Away from silhouettes the question has one right answer, so
        # this is where a rendering port is either correct or not.
        edge = _edge_mask(mj_depth, args.tolerance) | _edge_mask(nt_depth, args.tolerance)
        interior = ~edge
        totals["edge_pixels"] += int(edge.sum())
        totals["interior_pixels"] += int(interior.sum())
        totals["interior_agreeing"] += int(((diff <= args.tolerance) & interior).sum())
        totals["interior_policy_agreeing"] += int(((policy_diff <= args.tolerance / cutoff) & interior).sum())
        if int(interior.sum()):
            totals["interior_max"] = max(totals["interior_max"], float(diff[interior].max()))

        mj_nohit = mj_depth <= 0.0
        nt_nohit = nt_depth <= 0.0
        totals["nohit_mj"] += int(mj_nohit.sum())
        totals["nohit_nt"] += int(nt_nohit.sum())
        totals["nohit_both"] += int((mj_nohit & nt_nohit).sum())
        totals["nohit_either"] += int((mj_nohit | nt_nohit).sum())

        max_diff = float(diff.max())
        fraction = agreeing / pixels
        if fraction < worst["agree_fraction"]:
            worst = {"sample": sample, "max_diff": max_diff, "agree_fraction": fraction}

        if sample < 3 or (sample + 1) % 16 == 0:
            print(
                f"  sample {sample:4d}: max |Δ| {max_diff:.5f} m   "
                f"within {args.tolerance * 1000:.0f} mm: {100.0 * fraction:.2f}%   "
                f"mjlab depth [{float(mj_depth.min()):.3f}, {float(mj_depth.max()):.3f}]   "
                f"Newton [{float(nt_depth.min()):.3f}, {float(nt_depth.max()):.3f}]"
            )

    diffs = torch.cat(all_diffs)

    _geometry_report(mj_env, nt_env)
    _chain_report(mj_env, nt_env)

    print("\n-- 1. are the two scenes in the same state --")
    print(f"  worst joint-angle gap      {totals['joint_gap']:.3e} rad")
    print(f"  worst wrist position gap   {totals['wrist_pos_gap']:.3e} m   (env-local)")
    print(f"  worst wrist rotation gap   {totals['wrist_quat_gap']:.3e} (1 - |dot|)")
    print("\n-- 1b. what each backend's camera actually sees --")
    print(f"  mjlab   {_depth_profile(mj_depth)}")
    print(f"  Newton  {_depth_profile(nt_depth)}")
    print("  (last sample only; a large 'beyond 2 m' share means the camera is")
    print("   seeing a ground plane that stretches to the horizon)")
    results["the_same_joint_angles_land_in_both"] = totals["joint_gap"] < 1e-5
    results["the_wrist_ends_up_in_the_same_place"] = totals["wrist_pos_gap"] < 1e-4
    results["the_wrist_ends_up_in_the_same_pose"] = totals["wrist_quat_gap"] < 1e-6

    print("\n-- 2. do the depth images agree --")
    fraction = totals["agreeing"] / totals["pixels"]
    quantiles = torch.tensor([0.5, 0.9, 0.99, 0.999])
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
    results["the_two_renderers_agree_off_the_silhouettes"] = interior_fraction > 0.999
    results["the_policy_sees_the_same_image_off_the_silhouettes"] = interior_policy > 0.999

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

    disagree = (mj_depth - nt_depth).abs() > args.tolerance
    edge = _edges(mj_depth) | _edges(nt_depth)
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
    interior_now = ~edge
    masked = torch.where(interior_now, (mj_depth - nt_depth).abs(), torch.zeros_like(mj_depth))
    flat = masked.flatten()
    count = min(5, int((flat > args.tolerance).sum()))
    if count == 0:
        print("  none: every interior pixel in this sample agrees")
    else:
        for rank, index in enumerate(torch.topk(flat, count).indices.tolist()):
            env_index, row, column = (
                index // (mj_depth.shape[1] * mj_depth.shape[2]),
                (index // mj_depth.shape[2]) % mj_depth.shape[1],
                index % mj_depth.shape[2],
            )
            mj_value = float(mj_depth[env_index, row, column])
            nt_value = float(nt_depth[env_index, row, column])
            rows = slice(max(row - 1, 0), row + 2)
            columns = slice(max(column - 1, 0), column + 2)
            print(
                f"  #{rank}: env {env_index} pixel ({row},{column})  "
                f"mjlab {mj_value:.4f} m   Newton {nt_value:.4f} m   Δ {abs(mj_value - nt_value):.4f} m"
            )
            print(f"      mjlab  3x3 {[round(float(v), 3) for v in mj_depth[env_index, rows, columns].flatten()]}")
            print(f"      Newton 3x3 {[round(float(v), 3) for v in nt_depth[env_index, rows, columns].flatten()]}")
    print("  (two similar neighbourhoods offset by a pixel = a silhouette the edge mask")
    print("   was too narrow to catch; a flat patch differing = real geometry)")

    print("\n-- 3. do they agree about what they did NOT hit --")
    union = max(totals["nohit_either"], 1)
    iou = totals["nohit_both"] / union
    print(f"  mjlab no-hit pixels        {totals['nohit_mj']:,}")
    print(f"  Newton no-hit pixels       {totals['nohit_nt']:,}")
    print(f"  agreement (IoU)            {iou:.4f}")
    print("  (a ray grazing the ground plane hits it hundreds of metres away in one backend")
    print("   and misses in the other; the policy saturates both to its far plane)")

    print("\n-- 4. is the comparison worth anything --")
    # A pair of images that are constant, or a scene the camera never
    # sees into, would pass every check above by being empty.
    spread = float(diffs.numel() and mj_depth.std())
    print(f"  depth spread within one frame (mjlab): {spread:.5f} m")
    print(f"  distinct depths in the last frame:     {int(torch.unique(mj_depth).numel()):,}")
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
