"""Standalone visualizer / inspector for MotionCommand-ready NPZ files.

Plays a converted NPZ (from ``booster_to_npz`` / ``csv_to_npz``) through
MuJoCo's interactive viewer or renders it to an MP4 video, **without
loading any RL framework**. Use this to sanity-check the conversion
pipeline before training: if the motion looks broken here, training
will not save it.

Three modes:

* ``--mode=viewer`` (default): launch ``mujoco.viewer.launch_passive``
  and step through frames at ``output_fps`` from the NPZ. Standard
  MuJoCo viewer controls (drag to orbit, scroll to zoom, space to pause).
* ``--mode=video --output-mp4=PATH``: offscreen render to MP4 via
  ``mujoco.Renderer`` + imageio.
* ``--mode=inspect``: skip rendering, print numerical sanity checks:
  shape/dtype, quat norms, joint pos / velocity ranges, body Z extents,
  first/last frame qpos. Useful for headless servers.

The viewer / video modes also drive the MJCF live so floating-base
poses use the *same* MuJoCo FK that produced the NPZ — i.e. you see
exactly what the replayer baked, not a re-derivation.

Usage::

    # interactive
    uv run python -m rlworld.tools.motion.visualize_npz \\
        --npz JaxRLWorld/rlworld/assets/motions/booster/booster_t1_converted/walking1.npz

    # save MP4
    uv run python -m rlworld.tools.motion.visualize_npz \\
        --npz .../walking1.npz --mode video --output-mp4 /tmp/walking1.mp4

    # numerical only
    uv run python -m rlworld.tools.motion.visualize_npz \\
        --npz .../walking1.npz --mode inspect
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Literal

import mujoco
import numpy as np
import tyro


_DEFAULT_MJCF = "./JaxRLWorld/rlworld/assets/menagerie_T1/t1.xml"


def _load_npz(path: str) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {k: np.asarray(data[k]) for k in data.files}


def _free_joint_qpos_adr(model: mujoco.MjModel) -> int:
    for jid in range(model.njnt):
        if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
            return int(model.jnt_qposadr[jid])
    raise ValueError("MJCF has no free joint.")


def _resolve_joint_qpos_adr(
    model: mujoco.MjModel, joint_names: list[str],
) -> np.ndarray:
    adr = []
    for name in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"Joint {name!r} not in MJCF.")
        adr.append(int(model.jnt_qposadr[jid]))
    return np.asarray(adr, dtype=np.int64)


def _set_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    npz: dict[str, np.ndarray],
    free_adr: int,
    joint_qpos_adr: np.ndarray,
    t: int,
) -> None:
    """Write frame ``t`` of the NPZ into MjData and run mj_forward."""
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    # Free-joint pose: pos(3) + quat_wxyz(4) — use the first NPZ body
    # (body_names[0] is conventionally the floating-base body, e.g.
    # "Trunk" for T1) to seed the root state, since the replayer wrote
    # body_pos_w / body_quat_w from data.xpos / data.xquat after FK.
    # That stored xpos/xquat for the root body equals the free-joint qpos
    # we wrote in the replayer, so we recover qpos by reading the same.
    body_names = [str(n) for n in npz["body_names"].tolist()]
    if "world" in body_names[0].lower():
        # MuJoCo prepends a "world" body at index 0 — skip it.
        root_idx = 1
    else:
        root_idx = 0
    data.qpos[free_adr:free_adr + 3] = npz["body_pos_w"][t, root_idx]
    data.qpos[free_adr + 3:free_adr + 7] = npz["body_quat_w"][t, root_idx]
    data.qpos[joint_qpos_adr] = npz["joint_pos"][t]
    mujoco.mj_forward(model, data)


def _inspect(npz: dict[str, np.ndarray]) -> None:
    """Print numerical sanity checks."""
    print("=" * 64)
    print("NPZ FIELDS")
    print("=" * 64)
    for k, v in npz.items():
        if v.ndim == 0:
            print(f"  {k:<20} scalar  {v.dtype}  = {v.item()}")
        else:
            print(f"  {k:<20} shape={tuple(v.shape)} dtype={v.dtype}")

    T = npz["joint_pos"].shape[0]
    J = npz["joint_pos"].shape[1]
    body_names = [str(n) for n in npz["body_names"].tolist()]
    joint_names = [str(n) for n in npz["joint_names"].tolist()]

    print()
    print("=" * 64)
    print(f"FRAMES: T = {T},  fps = {float(npz['fps'])}, duration = "
          f"{T / float(npz['fps']):.3f} s")
    print("=" * 64)

    print()
    print(f"BODIES ({len(body_names)}):")
    for i, n in enumerate(body_names):
        print(f"  [{i:2}] {n}")
    print()
    print(f"JOINTS ({len(joint_names)}):")
    for i, n in enumerate(joint_names):
        jp = npz["joint_pos"][:, i]
        print(
            f"  [{i:2}] {n:<30} pos: [{jp.min():+.3f}, {jp.max():+.3f}] "
            f"mean={jp.mean():+.3f}"
        )

    # Quaternion norm check
    q = npz["body_quat_w"]
    norms = np.linalg.norm(q, axis=-1)
    bad_q = np.where(np.abs(norms - 1.0) > 1e-3)
    print()
    print("QUATERNION NORM CHECK")
    print(f"  range: [{norms.min():.6f}, {norms.max():.6f}] (expect ~1)")
    print(f"  off-unit norm count: {len(bad_q[0])} / {T * len(body_names)}")

    # Body Z extents — sanity for ground contact
    print()
    print("BODY Z EXTENTS (world frame)")
    z = npz["body_pos_w"][..., 2]
    for i, n in enumerate(body_names):
        zi = z[:, i]
        print(f"  [{i:2}] {n:<30} z: [{zi.min():+.3f}, {zi.max():+.3f}] m")

    # Velocity magnitudes
    print()
    print("VELOCITY MAGNITUDES (per-frame max norm)")
    lv = np.linalg.norm(npz["body_lin_vel_w"], axis=-1)
    av = np.linalg.norm(npz["body_ang_vel_w"], axis=-1)
    print(f"  body_lin_vel: max {lv.max():.3f} m/s  (median {np.median(lv):.3f})")
    print(f"  body_ang_vel: max {av.max():.3f} rad/s (median {np.median(av):.3f})")
    jv = np.abs(npz["joint_vel"])
    print(f"  joint_vel:    max {jv.max():.3f} rad/s (median {np.median(jv):.3f})")

    # First / last frame for visual cross-check
    print()
    print("FIRST FRAME (t=0)")
    print(f"  base pos = {npz['body_pos_w'][0, 0]}")
    print(f"  base quat (wxyz) = {npz['body_quat_w'][0, 0]}")
    print(f"  joint_pos[:5] = {npz['joint_pos'][0, :5]}")
    print()
    print("LAST FRAME (t=-1)")
    print(f"  base pos = {npz['body_pos_w'][-1, 0]}")
    print(f"  base quat (wxyz) = {npz['body_quat_w'][-1, 0]}")
    print(f"  joint_pos[:5] = {npz['joint_pos'][-1, :5]}")


def _resolve_track_body_index(
    npz: dict[str, np.ndarray], track_body: "str | None",
) -> "int | None":
    """Find the NPZ-side index of the body whose world-pos drives the camera.

    Returns ``None`` when tracking is disabled. ``"auto"`` selects the
    first non-``"world"`` body (the floating-base body, which is what
    you want for almost every locomotion clip).
    """
    if track_body is None or track_body == "":
        return None
    body_names = [str(n) for n in npz["body_names"].tolist()]
    if track_body == "auto":
        for i, n in enumerate(body_names):
            if n.lower() != "world":
                return i
        return None
    if track_body not in body_names:
        raise ValueError(
            f"Tracking body {track_body!r} not in NPZ body_names. "
            f"Available: {body_names}"
        )
    return body_names.index(track_body)


def _run_viewer(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    npz: dict[str, np.ndarray],
    free_adr: int,
    joint_qpos_adr: np.ndarray,
    fps: float,
    loop: bool,
    track_body: "str | None",
    cam_distance: float,
    cam_azimuth: float,
    cam_elevation: float,
) -> None:
    import mujoco.viewer  # local: optional dep, viewer mode only

    T = npz["joint_pos"].shape[0]
    dt = 1.0 / fps
    track_idx = _resolve_track_body_index(npz, track_body)
    if track_idx is not None:
        print(
            f"[viewer] T={T}, fps={fps:.1f}, tracking '{npz['body_names'][track_idx]}'. "
            f"Press ESC to quit."
        )
    else:
        print(f"[viewer] T={T}, fps={fps:.1f}. Press ESC to quit.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        if track_idx is not None:
            viewer.cam.distance = cam_distance
            viewer.cam.azimuth = cam_azimuth
            viewer.cam.elevation = cam_elevation
        t = 0
        while viewer.is_running():
            step_start = time.time()
            _set_frame(model, data, npz, free_adr, joint_qpos_adr, t)
            if track_idx is not None:
                # lookat is a 3-vector property; assign elementwise to
                # avoid replacing the underlying memoryview.
                lookat = npz["body_pos_w"][t, track_idx]
                viewer.cam.lookat[0] = float(lookat[0])
                viewer.cam.lookat[1] = float(lookat[1])
                viewer.cam.lookat[2] = float(lookat[2])
            viewer.sync()
            t = (t + 1) % T if loop else min(t + 1, T - 1)
            elapsed = time.time() - step_start
            sleep = dt - elapsed
            if sleep > 0:
                time.sleep(sleep)
            if not loop and t == T - 1:
                # hold last frame briefly so the user sees it
                time.sleep(2.0)
                break


def _save_video(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    npz: dict[str, np.ndarray],
    free_adr: int,
    joint_qpos_adr: np.ndarray,
    fps: float,
    output_mp4: str,
    width: int,
    height: int,
    track_body: "str | None",
    cam_distance: float,
    cam_azimuth: float,
    cam_elevation: float,
) -> None:
    import imageio.v2 as imageio  # local: optional dep, video mode only

    # MJCFs typically declare a default offscreen framebuffer of 640x480
    # (mujoco.MjVisual.global_.offwidth/offheight). The Renderer rejects
    # render sizes larger than that buffer. Bump both to match the
    # requested resolution at runtime so the user can ask for any size.
    if model.vis.global_.offwidth < width:
        model.vis.global_.offwidth = width
    if model.vis.global_.offheight < height:
        model.vis.global_.offheight = height

    T = npz["joint_pos"].shape[0]
    track_idx = _resolve_track_body_index(npz, track_body)
    if track_idx is not None:
        print(
            f"[video] rendering {T} frames -> {output_mp4!r} ({width}x{height}), "
            f"tracking '{npz['body_names'][track_idx]}'."
        )
    else:
        print(
            f"[video] rendering {T} frames -> {output_mp4!r} ({width}x{height})."
        )

    # Free camera that we manually re-aim at the tracked body each frame.
    # mjvCamera's ``lookat`` is the orbit center; ``azimuth`` / ``elevation``
    # / ``distance`` define the relative pose of the camera around lookat.
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = cam_distance
    cam.azimuth = cam_azimuth
    cam.elevation = cam_elevation

    with mujoco.Renderer(model, height, width) as renderer:
        writer = imageio.get_writer(output_mp4, fps=int(round(fps)))
        try:
            for t in range(T):
                _set_frame(model, data, npz, free_adr, joint_qpos_adr, t)
                if track_idx is not None:
                    lookat = npz["body_pos_w"][t, track_idx]
                    cam.lookat[0] = float(lookat[0])
                    cam.lookat[1] = float(lookat[1])
                    cam.lookat[2] = float(lookat[2])
                    renderer.update_scene(data, camera=cam)
                else:
                    renderer.update_scene(data, camera=-1)
                pixels = renderer.render()
                writer.append_data(pixels)
                if t % 50 == 0:
                    print(f"  frame {t}/{T}")
        finally:
            writer.close()
    print(f"[video] wrote {output_mp4!r}")


def main(
    npz: str,
    mjcf: str = _DEFAULT_MJCF,
    mode: Literal["viewer", "video", "inspect"] = "viewer",
    output_mp4: str = "/tmp/motion.mp4",
    fps_override: "float | None" = None,
    loop: bool = True,
    video_width: int = 1280,
    video_height: int = 720,
    track_body: "str | None" = "auto",
    cam_distance: float = 3.5,
    cam_azimuth: float = 90.0,
    cam_elevation: float = -15.0,
) -> None:
    """Visualize / inspect a MotionCommand NPZ.

    Args:
        npz: Path to NPZ produced by ``booster_to_npz`` / ``csv_to_npz``.
        mjcf: MJCF file used to render the robot. Default points at the
            menagerie T1 model used by the conversion pipeline.
        mode: ``"viewer"`` (interactive, default), ``"video"`` (save MP4),
            or ``"inspect"`` (numerical only, no display required).
        output_mp4: Output path when ``mode == "video"``.
        fps_override: Force a different playback FPS. Defaults to the FPS
            stored in the NPZ.
        loop: When ``mode == "viewer"``, loop the clip indefinitely.
        video_width / video_height: Render resolution for ``mode=video``.
        track_body: Body name to lock the camera to so the robot stays
            in frame on translating motions like ``running.npz``.
            ``"auto"`` (default) picks the first non-world body
            (the floating-base body — Trunk for T1). ``""`` or ``None``
            disables tracking and uses the default free camera.
        cam_distance: Camera orbit distance from the tracked point (m).
        cam_azimuth: Camera azimuth in degrees (90 = looking down +Y).
        cam_elevation: Camera elevation in degrees (negative = looking down).
    """
    npz_data = _load_npz(npz)
    fps = float(fps_override) if fps_override is not None else float(
        npz_data["fps"]
    )

    if mode == "inspect":
        _inspect(npz_data)
        return

    model = mujoco.MjModel.from_xml_path(mjcf)
    data = mujoco.MjData(model)
    free_adr = _free_joint_qpos_adr(model)
    joint_names = [str(n) for n in npz_data["joint_names"].tolist()]
    joint_qpos_adr = _resolve_joint_qpos_adr(model, joint_names)

    if mode == "viewer":
        _run_viewer(
            model, data, npz_data, free_adr, joint_qpos_adr, fps, loop,
            track_body, cam_distance, cam_azimuth, cam_elevation,
        )
    elif mode == "video":
        Path(output_mp4).parent.mkdir(parents=True, exist_ok=True)
        _save_video(
            model, data, npz_data, free_adr, joint_qpos_adr, fps,
            output_mp4, video_width, video_height,
            track_body, cam_distance, cam_azimuth, cam_elevation,
        )
    else:
        raise ValueError(f"Unknown mode: {mode!r}")


if __name__ == "__main__":
    tyro.cli(main)
