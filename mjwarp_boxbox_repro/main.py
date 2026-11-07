"""box_box reports a contact ~1 m deep between boxes overlapping 0.1 mm.

A bar lies at rest on a table, sunk 0.1 mm into it. Every contact should
report -0.0001 m. MuJoCo reports exactly that. mujoco_warp reports those
and, for a few percent of yaw angles, one more at around -0.9 m, and the
solver undoing that invented penetration throws the bar off the table at
over 10 m/s.

    python main.py            # numbers, then a viewer
    python main.py --no-view  # numbers only
    mjpython main.py          # on macOS, for the viewer

Cause: collision_primitive_core.py:696

    if c3 < separation * (1.0 - 1e-12):

An edge axis must beat the best face axis by a margin, because when both
boxes are level four of the nine edge-cross axes ARE the face normal --
same direction written two ways -- so they tie to the last bit. The
constant is inherited from MuJoCo's C, where mjtNum is double and 1e-12
is ~4500 ULPs. In float32, 1.0 - 1e-12 rounds to 1.0 and the guard does
nothing, so rounding alone can hand a face-face contact to the edge-edge
routine, which returns a point outside the box.

A relative margin cannot work here regardless: the separation is a
difference of supports of order 0.4 m, so its float32 error is ~5e-8 m
absolute, while the separation itself is ~1e-4 m -- a relative error of
~5e-4. Raising the constant to 1e-6 or 1e-4 changes nothing. Slack
proportional to the box scale takes this from 92/2000 poses to 0/2000:

    edge_slack = wp.float32(1.0e-6) * (
      box1_size[0] + box1_size[1] + box1_size[2]
      + box2_size[0] + box2_size[1] + box2_size[2]
    )
    if c3 < separation - edge_slack:

nativeccd is disabled so that box-box uses the analytic collider rather
than CCD; with it enabled this scene is clean, and the same 2000 poses
give 0 bad contacts.

Note the bar body is declared BEFORE the table. For two geoms of the
same type the broadphase pair order is not normalised, and this only
misfires when the smaller box is box1. Declaring the table first hides
it completely.
"""

import argparse
import time

import mujoco
import numpy as np
import warp as wp

wp.config.log_level = wp.LOG_WARNING

import mujoco_warp as mjw  # noqa: E402  (warp log level is set above, before the import)

TABLE_HALF = (0.6, 0.4, 0.2)
TABLE_POS = (0.35, 0.0, 0.2)
BAR_HALF = (0.06, 0.003, 0.015)
OVERLAP = 1e-4

XML = f"""<mujoco>
  <option timestep="0.005">
    <!-- Routes box-box to the analytic collider instead of CCD. With
         nativeccd on this scene is clean; the bug is in that collider. -->
    <flag nativeccd="disable"/>
  </option>
  <worldbody>
    <light pos="0.35 -0.8 1.4" dir="0 0.5 -1"/>
    <camera name="watch" mode="targetbody" target="bar" pos="0.35 -1.5 1.05"/>
    <body name="bar" pos="0 0 1">
      <freejoint/>
      <geom name="bar" type="box" rgba="0.9 0.35 0.15 1"
            size="{BAR_HALF[0]} {BAR_HALF[1]} {BAR_HALF[2]}"/>
    </body>
    <body name="table" pos="{TABLE_POS[0]} {TABLE_POS[1]} {TABLE_POS[2]}">
      <geom name="table" type="box" rgba="0.55 0.55 0.6 1"
            size="{TABLE_HALF[0]} {TABLE_HALF[1]} {TABLE_HALF[2]}"/>
    </body>
  </worldbody>
</mujoco>"""


def resting_poses(n, seed=9):
    """n ways to lay the bar flat on the table, overlapping by OVERLAP."""
    rng = np.random.default_rng(seed)
    yaw = rng.uniform(-np.pi, np.pi, n)
    q = np.zeros((n, 7), dtype=np.float32)
    q[:, 0] = rng.uniform(0.14, 0.45, n)
    q[:, 1] = rng.uniform(-0.20, 0.20, n)
    q[:, 2] = TABLE_POS[2] + TABLE_HALF[2] + BAR_HALF[2] - OVERLAP
    q[:, 3] = np.cos(yaw / 2)
    q[:, 6] = np.sin(yaw / 2)
    return q


def scan(mjm, mjd, poses):
    """Collide every pose at once and return the qpos of the worst ones."""
    d = mjw.put_data(
        mjm, mjd, nworld=len(poses), nconmax=16 * len(poses), njmax=64 * len(poses)
    )
    d.qpos.assign(wp.array(poses, dtype=wp.float32, ndim=2))
    mjw.forward(mjw.put_model(mjm), d)

    n = int(d.nacon.numpy()[0])
    dist = d.contact.dist.numpy()[:n]
    world = d.contact.worldid.numpy()[:n]
    bad = sorted(
        set(world[dist < -0.01].tolist()), key=lambda w: dist[world == w].min()
    )
    return bad, [dist[world == w].copy() for w in bad]


def roll_out(mjm, qpos, steps, engine):
    """Step the scene from qpos and record the bar and its contacts."""
    mjd = mujoco.MjData(mjm)
    mjd.qpos[:] = qpos
    mujoco.mj_forward(mjm, mjd)
    frames, speeds, deepest = [], [], []

    if engine == "warp":
        m, d = mjw.put_model(mjm), mjw.put_data(mjm, mjd, nconmax=64, njmax=256)
    for _ in range(steps):
        if engine == "warp":
            mjw.step(m, d)
            mjw.get_data_into(mjd, mjm, d)
            dist = d.contact.dist.numpy()[: int(d.nacon.numpy()[0])]
        else:
            mujoco.mj_step(mjm, mjd)
            dist = np.array([mjd.contact[i].dist for i in range(mjd.ncon)])
        frames.append(mjd.qpos.copy())
        speeds.append(float(np.linalg.norm(mjd.qvel[:3])))
        deepest.append(float(dist.min()) if len(dist) else 0.0)
    return frames, speeds, deepest


def replay(mjm, mj_frames, wp_frames):
    """Loop MuJoCo's trajectory then mujoco_warp's, side by side in time."""
    import mujoco.viewer

    mjd = mujoco.MjData(mjm)
    with mujoco.viewer.launch_passive(mjm, mjd) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = mujoco.mj_name2id(
            mjm, mujoco.mjtObj.mjOBJ_CAMERA, "watch"
        )
        while viewer.is_running():
            for label, frames in (("MuJoCo", mj_frames), ("mujoco_warp", wp_frames)):
                print(f"  ... {label}")
                for qpos in frames:
                    if not viewer.is_running():
                        return
                    mjd.qpos[:] = np.nan_to_num(qpos, nan=0.0, posinf=0.0, neginf=0.0)
                    mujoco.mj_forward(mjm, mjd)
                    viewer.sync()
                    time.sleep(0.05)
                time.sleep(0.7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses", type=int, default=2000)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--no-view", action="store_true")
    args = ap.parse_args()

    mjm = mujoco.MjModel.from_xml_string(XML)
    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)

    print(
        f"bar   {2000 * BAR_HALF[0]:.0f} x {2000 * BAR_HALF[1]:.0f} x "
        f"{2000 * BAR_HALF[2]:.0f} mm, laid flat on a "
        f"{2000 * TABLE_HALF[0]:.0f} x {2000 * TABLE_HALF[1]:.0f} mm table"
    )
    print(
        f"sunk {1000 * OVERLAP:.1f} mm in, at rest; every contact should be "
        f"{-OVERLAP:+.6f} m"
    )
    print(f"device {wp.get_device()}\n")

    poses = resting_poses(args.poses)
    bad, dists = scan(mjm, mjd, poses)
    print(f"{len(bad)} of {args.poses} poses gave a contact deeper than 10 mm.")
    if not bad:
        return 0

    # Deepest is not the same as most disruptive, so simulate a handful.
    rolled = [
        (poses[w], dists[i], roll_out(mjm, poses[w], args.steps, "warp"))
        for i, w in enumerate(bad[:8])
    ]
    qpos, warp_dist, (wp_frames, wp_speeds, wp_deep) = max(
        rolled, key=lambda r: np.linalg.norm(np.nan_to_num(r[2][0][-1][:3]) - r[0][:3])
    )
    mj_frames, mj_speeds, mj_deep = roll_out(mjm, qpos, args.steps, "mujoco")

    mjd.qpos[:] = qpos
    mujoco.mj_forward(mjm, mjd)
    mj_dist = np.array([mjd.contact[i].dist for i in range(mjd.ncon)])

    print(
        f"\nbar at {np.array2string(qpos[:3], precision=4)}, "
        f"yaw {2 * np.arctan2(qpos[6], qpos[3]):+.6f} rad"
    )
    print(f"  MuJoCo      {np.array2string(np.sort(mj_dist), precision=6)}")
    print(f"  mujoco_warp {np.array2string(np.sort(warp_dist), precision=6)}")

    print(
        "\nstep |    MuJoCo speed | MuJoCo deepest |      warp speed |   warp deepest"
    )
    print("-----+-----------------+----------------+-----------------+---------------")
    for k in range(min(args.steps, 10)):
        print(
            f"{k:4d} | {mj_speeds[k]:11.4f} m/s | {1000 * mj_deep[k]:11.1f} mm |"
            f" {wp_speeds[k]:11.4f} m/s | {1000 * wp_deep[k]:10.1f} mm"
        )

    def moved(frames):
        return np.nan_to_num(frames[-1][:3], nan=0.0, posinf=0.0, neginf=0.0) - qpos[:3]

    finite = [s for s in wp_speeds if np.isfinite(s)]
    print(f"\nover {args.steps * 0.005:.2f} s from rest:")
    print(
        f"  MuJoCo      peak {max(mj_speeds):9.4f} m/s   moved "
        f"{1000 * np.linalg.norm(moved(mj_frames)):9.2f} mm"
    )
    print(
        f"  mujoco_warp peak {max(finite, default=float('nan')):9.4f} m/s   moved "
        f"{1000 * np.linalg.norm(moved(wp_frames)):9.2f} mm"
    )

    if not args.no_view:
        print("\nviewer: MuJoCo's trajectory, then mujoco_warp's, on a loop.")
        replay(mjm, mj_frames, wp_frames)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
