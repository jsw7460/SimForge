"""Does the wrist depth camera report what we think it reports?

Before a policy is asked to act on an image, the image has to be shown
to be of the right thing, in the right units, at the right time. Every
one of those fails silently. A camera rendered before the state it is
supposed to describe still returns a plausible picture; a depth buffer
in ray distance rather than forward distance still looks like depth; a
tensor aliasing the render buffer still holds the right values until
something else renders. None of it raises, and a policy trained on any
of them simply learns less well than it should, which is the hardest
failure to attribute.

So this checks, rather than prints:

* the sensor wraps the YAM's OWN ``camera_d405``, the real D405's
  extrinsics, rather than a fresh camera placed by hand;
* depth is finite, positive, and its "no hit" value is identified;
* the image CHANGES when the arm moves — if it does not, ``sense()`` is
  not running and every image is the reset frame;
* the depth convention: ray distance or projected onto the camera's
  forward axis. Newton offers both under separate names, so the two
  backends can only agree once this is known;
* whether the returned tensor ALIASES the render buffer, which decides
  whether an observation may keep a reference to it;
* what a render costs, per resolution and per environment count.

    jaxpy -m jaxrlworld.scripts.diag.yam.mjlab_wrist_camera_diag
    jaxpy -m jaxrlworld.scripts.diag.yam.mjlab_wrist_camera_diag --num-envs 64 --width 64 --height 48
"""

from __future__ import annotations

import argparse
import math
import statistics
import time
from dataclasses import dataclass

import mujoco
import torch

from jaxrlworld.rl.configs.common_config_classes import ObservationGroupConfig
from jaxrlworld.rl.configs.observations import ObservationTermConfig
from jaxrlworld.rl.configs.presets.yam_lift.base import YamLiftConfig
from jaxrlworld.rl.envs.mdp.observations.common import perception
from jaxrlworld.rl.runners import BaseRunner

_CAMERA_GROUP = "wrist"
_FLOOR_GROUP = "floor_ref"
_SKY_GROUP = "sky_ref"
# Well clear of the table at x=0.35, so the downward camera sees only the
# ground plane and every one of its rays lands on the SAME flat surface.
_REF_X = 2.0
_REF_H = 1.0


def _find_camera(mj_model, want: str) -> str:
    """The model's own camera whose bare name is ``want``.

    Discovered rather than hardcoded: mjlab prefixes an entity's
    elements with the entity name, and a diag that assumes the prefix
    passes for the wrong reason the day it changes.
    """
    names = [mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, i) or "" for i in range(mj_model.ncam)]
    matches = [n for n in names if n.rsplit("/", 1)[-1] == want]
    if not matches:
        raise KeyError(f"No camera named {want!r} in the model. Cameras: {names}")
    if len(matches) > 1:
        raise KeyError(f"Camera name {want!r} is ambiguous: {matches}")
    return matches[0]


def _depth_of(env, group: str = _CAMERA_GROUP) -> torch.Tensor:
    """(num_envs, H, W) depth for one camera group, squeezed."""
    return env.scene_manager.sensors[group].data.depth[..., 0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", type=int, default=16)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--height", type=int, default=48)
    ap.add_argument("--fovy", type=float, default=58.0, help="D405's vertical FOV, degrees.")
    args = ap.parse_args()

    from mjlab.sensor import CameraSensorCfg as MjCameraSensorCfg

    print("=" * 78)
    print(f"WRIST DEPTH CAMERA  [yam_lift / mjlab  num_envs={args.num_envs}  {args.width}x{args.height}]")
    print("=" * 78)

    results: dict[str, bool] = {}

    # Build once with no camera, purely to read the model's camera table
    # and confirm the asset carries the one we mean to wrap.
    config = YamLiftConfig(sim_type="mujoco", num_envs=args.num_envs)
    cfgs = config.build()
    probe = BaseRunner._create_env_from_config(cfgs)
    camera_name = _find_camera(probe.scene_manager.sim.mj_model, "camera_d405")
    print(f"\n  wrapping the asset's own camera: {camera_name!r}")
    results["the_arm_ships_its_own_wrist_camera"] = True
    del probe

    # Rebuild with the camera attached. Wrapping by name keeps the real
    # D405's mounting transform from the MJCF instead of re-deriving it.
    # Fresh cfgs per env: term params carry SceneEntitySelectors that the
    # managers resolve IN PLACE, so a cfgs object that has already built
    # an env holds that env's device tensors.
    def _build_cfgs(camera_group=None):
        cfgs = YamLiftConfig(sim_type="mujoco", num_envs=args.num_envs).build()
        cfgs.scene.sensors = tuple(cfgs.scene.sensors) + (
            MjCameraSensorCfg(
                name=_CAMERA_GROUP,
                camera_name=camera_name,
                width=args.width,
                height=args.height,
                fovy=args.fovy,
                data_types=("depth",),
            ),
            # Two reference cameras, in the worldbody, on bare floor away
            # from the table. The wrist camera cannot settle either
            # question below: it sees the gripper and the table at
            # angles, so no two of its pixels are on one plane, and
            # nothing it looks at is far enough away to miss. A MuJoCo
            # camera looks along its own -Z with identity orientation, so
            # this one points straight down.
            MjCameraSensorCfg(
                name=_FLOOR_GROUP,
                pos=(_REF_X, 0.0, _REF_H),
                quat=(1.0, 0.0, 0.0, 0.0),
                width=args.width,
                height=args.height,
                fovy=args.fovy,
                data_types=("depth",),
            ),
            # Rotated a half turn about X, so -Z points at the sky and
            # every ray misses.
            MjCameraSensorCfg(
                name=_SKY_GROUP,
                pos=(_REF_X, 0.0, _REF_H),
                quat=(0.0, 1.0, 0.0, 0.0),
                width=args.width,
                height=args.height,
                fovy=args.fovy,
                data_types=("depth",),
            ),
        )
        if camera_group is not None:
            cfgs.observation.camera = camera_group
        return cfgs

    env = BaseRunner._create_env_from_config(_build_cfgs())
    env.reset()

    # ── 1. shape, dtype, range ───────────────────────────────────────
    depth = _depth_of(env)
    print("\n-- 1. what comes back --")
    print(f"  shape {tuple(depth.shape)}  dtype {depth.dtype}  device {depth.device}")
    results["depth_is_one_plane_per_env"] = tuple(depth.shape) == (args.num_envs, args.height, args.width)
    results["depth_is_float32"] = depth.dtype == torch.float32

    finite = torch.isfinite(depth)
    results["every_pixel_is_finite"] = bool(finite.all())
    vals = depth[finite]
    lo, med, hi = (float(vals.min()), float(vals.median()), float(vals.max()))
    print(f"  min {lo:.4f}  median {med:.4f}  max {hi:.4f} m")
    # Just the mode, for shape of the distribution. It is NOT the no-hit
    # value: the wrist camera sees the gripper and the table, everything
    # within half a metre, and no ray of its misses. Check 3b answers that
    # with a camera pointed at the sky.
    mode = float(torch.mode(depth.reshape(-1)).values)
    share = float((depth == mode).float().mean())
    print(f"  most common value {mode:.4f} m, {100 * share:.1f}% of pixels")
    results["depth_is_positive"] = lo > 0.0

    # ── 2. does it follow the arm ────────────────────────────────────
    # The decisive check. A camera rendered from a stale state returns a
    # perfectly plausible image; only moving the thing it is bolted to
    # tells the two apart.
    print("\n-- 2. does the image follow the wrist --")
    before = depth.clone()
    action = torch.zeros(env.num_envs, env.act_manager.num_actions, device=env.device)
    action[:, 1] = 1.0  # one shoulder joint, enough to swing the wrist
    for _ in range(15):
        env.step(action)
    after = _depth_of(env).clone()
    moved = float((after - before).abs().mean())
    changed_frac = float(((after - before).abs() > 1e-4).float().mean())
    print(f"  mean |Δdepth| after 15 steps of arm motion: {moved:.5f} m over {100 * changed_frac:.1f}% of pixels")
    results["the_image_moves_with_the_arm"] = moved > 1e-3

    # ── 3. ray distance, or projected onto forward? ──────────────────
    # Newton returns both under separate names, so the two backends cannot
    # be made to agree until this is known — and the wrist camera cannot
    # answer it, because settling it needs two pixels on ONE flat surface.
    # Hence the downward reference camera: from a known height over bare
    # ground, every ray lands on the same plane, and a ray-distance buffer
    # reads 1/cos(theta) larger off-axis where a forward-projected one
    # reads flat.
    print("\n-- 3. which depth convention (camera looking straight down at the floor) --")
    floor = _depth_of(env, _FLOOR_GROUP)[0]
    h, w = floor.shape
    centre = float(floor[h // 2, w // 2])
    corner = float(floor[0, 0])
    ty = math.tan(math.radians(args.fovy) / 2.0)
    tx = ty * (w / h)
    ray_ratio = math.sqrt(1.0 + tx * tx + ty * ty)
    ratio = corner / max(centre, 1e-9)
    print(f"  camera {_REF_H:.2f} m above the ground plane at x={_REF_X}")
    print(f"  centre {centre:.4f} m   corner {corner:.4f} m   corner/centre {ratio:.4f}")
    print(f"  ray distance would give {ray_ratio:.4f}, forward-projected 1.0000")
    is_forward = abs(ratio - 1.0) < 0.02
    is_ray = abs(ratio - ray_ratio) < 0.02 * ray_ratio
    verdict = "forward-projected" if is_forward else ("ray distance" if is_ray else "NEITHER")
    print(f"  -> {verdict}")
    results["the_depth_convention_is_identified"] = is_forward or is_ray
    # The centre pixel reads the height directly under either convention,
    # so it also checks the units are metres rather than something scaled.
    print(f"  centre vs the true {_REF_H:.2f} m drop: error {abs(centre - _REF_H):.4f} m")
    results["depth_reads_metres"] = abs(centre - _REF_H) < 0.02

    # ── 3b. what a ray that hits nothing reports ─────────────────────
    # If it were 0, "empty space" and "touching the lens" would be the
    # same number, and a policy would read open air as an obstacle.
    print("\n-- 3b. the no-hit value (camera looking at the sky) --")
    sky = _depth_of(env, _SKY_GROUP)
    uniq = torch.unique(sky)
    print(f"  distinct values across {sky.numel():,} sky pixels: {uniq.numel()}")
    print(f"  min {float(sky.min()):.4f}  max {float(sky.max()):.4f}")
    if uniq.numel() <= 4:
        print(f"  values: {[round(float(v), 4) for v in uniq[:4]]}")
    no_hit = float(sky.median())
    print(f"  -> a ray that hits nothing reports {no_hit:.4f}")
    # Distinguishable from a real reading is what matters, not the value.
    results["no_hit_is_not_a_plausible_distance"] = no_hit <= 0.0 or no_hit > 5.0

    # ── 4. does the tensor alias the render buffer ───────────────────
    # mjlab's CameraSensorCfg.clone_data defaults to False, so what comes
    # back is a view. An observation term that returns it, or a history
    # buffer that stores it, would be silently rewritten by the next
    # render — with values that stay plausible throughout.
    print("\n-- 4. is the returned tensor a view of the render buffer --")
    held = _depth_of(env)
    snapshot = held.clone()
    for _ in range(5):
        env.step(action)
    drifted = float((held - snapshot).abs().max())
    print(f"  a reference kept across 5 renders changed by {drifted:.5f} m")
    results["a_kept_reference_aliases_the_buffer"] = drifted > 1e-6
    print("  (True is EXPECTED with clone_data=False — anything storing depth must clone it)")

    # ── 5. cost ──────────────────────────────────────────────────────
    print("\n-- 5. what a render costs --")
    sim = env.scene_manager.sim
    for _ in range(5):
        sim.sense()
    torch.cuda.synchronize()
    samples = []
    for _ in range(30):
        t0 = time.perf_counter()
        sim.sense()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1e3)
    print(f"  sim.sense(): {statistics.mean(samples):.3f} ms mean, {statistics.median(samples):.3f} median")
    print(f"  {args.num_envs} envs x {args.width}x{args.height} = {args.num_envs * args.width * args.height:,} pixels")

    # ── 6. the observation pipeline ──────────────────────────────────
    # Everything above says the RENDER is right. This says the number a
    # policy actually receives is that render, normalised the way mjlab
    # normalises it, in a group that does not disturb the vector groups.
    print("\n-- 6. what the policy receives --")
    cutoff = 0.5

    @dataclass
    class _CameraObsCfg(ObservationGroupConfig):
        enable_corruption: bool = False
        concatenate_dim: int = 0  # channel axis: several images stack, not laid side by side
        depth = ObservationTermConfig(
            func=perception.camera_depth,
            scale=1.0,
            params={"sensor_name": _CAMERA_GROUP, "cutoff_distance": cutoff},
        )

    cam_env = BaseRunner._create_env_from_config(_build_cfgs(_CameraObsCfg()))
    cam_env.reset()

    shapes = cam_env.obs_manager.calculate_obs_shapes()
    dims = cam_env.calculate_obs_dim()
    print(f"  group shapes: { {k: v for k, v in shapes.items()} }")
    print(f"  flat dims:    {dict(dims)}")
    results["camera_group_keeps_its_image_shape"] = shapes["camera"] == (1, args.height, args.width)
    results["camera_group_is_not_given_a_flat_width"] = "camera" not in dims
    results["vector_groups_are_untouched"] = dims["actor"] == env.calculate_obs_dim()["actor"]

    obs = cam_env.obs_manager.get_observation()["camera"]
    print(f"  camera obs {tuple(obs.shape)}  min {float(obs.min()):.4f}  max {float(obs.max()):.4f}")
    results["camera_obs_is_bchw"] = tuple(obs.shape) == (args.num_envs, 1, args.height, args.width)
    results["camera_obs_is_normalised"] = bool(torch.isfinite(obs).all() and obs.min() >= 0.0 and obs.max() <= 1.0)

    # The normalisation, recomputed by hand from the same raw buffer.
    raw = cam_env.scene_manager.sensors[_CAMERA_GROUP].data.depth.permute(0, 3, 1, 2)
    expected = torch.clamp(torch.clamp(raw, min=0.01, max=cutoff) / cutoff, 0.0, 1.0)
    gap = float((obs - expected).abs().max())
    print(f"  vs clamp(clamp(raw, 0.01, {cutoff}) / {cutoff}, 0, 1): max gap {gap:.3e}")
    results["camera_obs_is_mjlabs_normalisation"] = gap == 0.0

    # Check 4 established that the raw buffer is a view. The term must
    # hand back something the next render cannot rewrite.
    kept = cam_env.obs_manager.get_observation()["camera"]
    kept_snapshot = kept.clone()
    for _ in range(5):
        cam_env.step(action)
    results["camera_obs_survives_the_next_render"] = float((kept - kept_snapshot).abs().max()) == 0.0

    space = cam_env.observation_space["camera"]
    print(f"  observation_space['camera'] = {space.shape}")
    results["camera_observation_space_is_the_image"] = space.shape == (1, args.height, args.width)

    # Two image terms in one group stack on the channel axis rather than
    # being laid end to end — the mechanism a depth+mask policy needs.
    @dataclass
    class _TwoCameraObsCfg(_CameraObsCfg):
        depth_far = ObservationTermConfig(
            func=perception.camera_depth,
            scale=1.0,
            params={"sensor_name": _FLOOR_GROUP, "cutoff_distance": 2.0},
        )

    two_env = BaseRunner._create_env_from_config(_build_cfgs(_TwoCameraObsCfg()))
    two_env.reset()
    two_shape = two_env.obs_manager.calculate_obs_shapes()["camera"]
    print(f"  two image terms in one group -> {two_shape}")
    results["two_images_stack_on_the_channel_axis"] = two_shape == (2, args.height, args.width)

    print("\n" + "=" * 78)
    ok = True
    for name, passed in results.items():
        print(f"  {name:<44}: {'PASS' if passed else 'FAIL'}")
        ok = ok and passed
    print(f"  {'OVERALL':<44}: {'PASS' if ok else 'FAIL'}")
    print("=" * 78)
    print("  Check 4 reporting True is correct, not a failure: it records that")
    print("  the tensor is a view, which is what the observation layer must")
    print("  account for.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
