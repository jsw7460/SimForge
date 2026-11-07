"""Apply the box_box edge-axis fix to whichever mujoco_warp is installed here.

Run it inside the environment you train in, on each machine. It locates
mujoco_warp through the interpreter that is running it, so there is no
path to get wrong and no way to patch a copy the training run does not
use.

    python mjwarp_boxbox_repro/patch_installed.py            # apply, then verify
    python mjwarp_boxbox_repro/patch_installed.py --check    # report only
    python mjwarp_boxbox_repro/patch_installed.py --revert   # put it back

It is idempotent: run it twice and the second run says so and does
nothing. Re-run it after anything that reinstalls mujoco_warp -- pip
install, uv sync, a fresh venv -- since that restores the original file.

WHAT IT CHANGES

collision_primitive_core.py, in box_box(), the tie-break that decides
whether an edge axis beats the best face axis:

    -  if c3 < separation * (1.0 - 1e-12):
    +  if c3 < separation - edge_slack:

with edge_slack scaled by the box extents. The constant came from
MuJoCo's C, where mjtNum is double and 1e-12 is a few thousand ULPs; in
float32, 1.0 - 1e-12 rounds to exactly 1.0 and the guard does nothing.
When two boxes rest level against each other, four of the nine
edge-cross axes ARE the face normal, so they tie to the last bit and
rounding alone can hand a face-face contact to the edge-edge routine,
which returns a point outside the boxes at a depth of about 0.9 m for
boxes overlapping by 0.1 mm.

The verification below is not a claim that the file changed -- it calls
box_box at a pose that reproduces the fault and checks the numbers, so a
pass means the code the GPU will run is fixed, cache and all. Warp keys
its kernel cache on the source, so the edit forces a recompile; the first
run after patching takes a few seconds longer.

This is a local patch, not an upstream release. The same change is out as
a pull request against google-deepmind/mujoco_warp; when it lands, drop
this.
"""

from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys

import numpy as np
import warp as wp

wp.config.log_level = wp.LOG_WARNING

GUARD_OLD = "      if c3 < separation * (1.0 - 1e-12):"
GUARD_NEW = "      if c3 < separation - edge_slack:"
ANCHOR = "  axis_code = wp.int32(-1)\n"
SLACK = """
  # The separations are sums of box extents times rotation entries, so their
  # float32 error is ABSOLUTE and proportional to the extents, while a resting
  # contact's separation is near zero -- a relative margin cannot express that.
  edge_slack = wp.float32(1.0e-6) * (
    box1_size[0] + box1_size[1] + box1_size[2] + box2_size[0] + box2_size[1] + box2_size[2]
  )
"""

# The pose is what kinematics produces for a 120x6x30 mm bar resting on a
# 1.2x0.8x0.4 m table, overlapping it by 1e-4 m, at a yaw where the tie
# breaks the wrong way. Reproduces identically on x86_64 with CUDA and on
# arm64 on the CPU backend.
BAR_POS = (0.1917702853679657, 0.1877419650554657, 0.414900004863739)
BAR_ROT = (
    0.6132574081420898,
    -0.7898833751678467,
    0.0,
    0.7898833751678467,
    0.6132574081420898,
    0.0,
    0.0,
    0.0,
    1.000000238418579,
)
BAR_HALF = (0.06, 0.003, 0.015)
TABLE_POS = (0.3499999940395355, 0.0, 0.20000000298023224)
TABLE_ROT = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
TABLE_HALF = (0.6, 0.4, 0.2)
OVERLAP = 1e-4


def locate() -> pathlib.Path:
    """The collision source belonging to THIS interpreter's mujoco_warp.

    Found without importing the package. Importing it pulls the collision
    module into memory, and warp generates its kernels from the source of
    the function object it was handed -- so a verification run that
    imported before patching would compile the file it was trying to
    replace and report the fault as unfixed.
    """
    spec = importlib.util.find_spec("mujoco_warp")
    if spec is None or not spec.origin:
        raise SystemExit(f"No mujoco_warp importable from {sys.executable}.")
    path = (
        pathlib.Path(spec.origin).resolve().parent
        / "_src"
        / "collision_primitive_core.py"
    )
    if not path.is_file():
        raise SystemExit(f"mujoco_warp is installed but {path} is missing.")
    return path


def state(text: str) -> str:
    """``patched``, ``original``, or ``unknown``."""
    if GUARD_NEW in text and "edge_slack" in text:
        return "patched"
    if GUARD_OLD in text:
        return "original"
    return "unknown"


def verify() -> bool:
    """Call box_box at the failing pose and check every contact.

    mujoco_warp is imported HERE, after the file on disk has changed, so
    that warp generates the kernel from the patched source. warp itself is
    imported at the top, because the decorator below evaluates this
    kernel's annotations against module globals and would not find ``wp``
    among a function's locals.
    """
    from mujoco_warp._src.collision_primitive_core import MJ_MAXVAL, box_box

    # Eight plain floats rather than the vec8f box_box returns: the
    # decorator evaluates these annotations against module globals, and a
    # type imported inside this function is not one.
    @wp.kernel
    def probe(
        p1: wp.vec3,
        r1: wp.mat33,
        h1: wp.vec3,
        p2: wp.vec3,
        r2: wp.mat33,
        h2: wp.vec3,
        out: wp.array(dtype=float),
    ):
        dist, pos, normal = box_box(p1, r1, h1, p2, r2, h2, 0.0)
        for i in range(8):
            out[i] = dist[i]

    out = wp.zeros(8, dtype=float)
    wp.launch(
        probe,
        dim=1,
        inputs=[
            wp.vec3(*BAR_POS),
            wp.mat33(*BAR_ROT),
            wp.vec3(*BAR_HALF),
            wp.vec3(*TABLE_POS),
            wp.mat33(*TABLE_ROT),
            wp.vec3(*TABLE_HALF),
        ],
        outputs=[out],
    )
    dist = out.numpy()
    real = dist[dist < MJ_MAXVAL]
    print(f"  device   {wp.get_device()}")
    print(f"  contacts {np.array2string(np.sort(real), precision=6)}")
    print(
        f"  the bar overlaps the table by {OVERLAP:.6f} m, so every contact must be -{OVERLAP:.6f}"
    )
    return bool(real.size) and bool((np.abs(real + OVERLAP) < 1e-5).all())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report the state and stop")
    ap.add_argument("--revert", action="store_true", help="restore the original file")
    args = ap.parse_args()

    path = locate()
    backup = path.with_suffix(".py.orig")
    text = path.read_text()
    now = state(text)

    print(f"python  {sys.executable}")
    print(f"target  {path}")
    print(f"state   {now}")

    if now == "unknown":
        raise SystemExit(
            "\nThis file matches neither the original nor the patched form. It is a\n"
            "different mujoco_warp version than this patch was written against, or it\n"
            "has been edited by hand. Nothing was changed."
        )

    if args.check:
        return 0 if now == "patched" else 1

    if args.revert:
        if now == "original":
            print("\nAlready original; nothing to do.")
            return 0
        if not backup.is_file():
            raise SystemExit(f"\nNo backup at {backup}; reinstall mujoco_warp instead.")
        path.write_text(backup.read_text())
        backup.unlink()
        print("\nReverted.")
        return 0

    if now == "patched":
        print("\nAlready patched. Verifying anyway.\n")
    else:
        backup.write_text(text)
        path.write_text(
            text.replace(ANCHOR, ANCHOR + SLACK, 1).replace(GUARD_OLD, GUARD_NEW, 1)
        )
        print(f"\nPatched. Original kept at {backup}\n")

    print("Verifying against the pose that reproduces the fault:")
    ok = verify()
    print()
    if ok:
        print("PASS -- the running code no longer invents the contact.")
        return 0
    print("FAIL -- a contact deeper than the overlap is still reported.")
    print("Do not train on this; send the output back.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
