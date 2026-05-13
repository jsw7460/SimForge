"""Verify two candidates for adding a "foot-pad" kinematic frame to a Genesis MJCF.

Genesis discards MuJoCo ``<site>`` (``Genesis/utils/mjcf.py:753`` —
"Genesis does not implement site abstraction"). So if we want all three
sims to agree on a foot-pad position, Genesis needs an alternative
reference attached to the foot body at the same local offset that mjlab's
site uses. This script checks both candidates on a minimal MJCF
(one free-base ball with a marker attached at a non-trivial offset):

  Mode A — **Marker geom**: ``<geom contype="0" conaffinity="0" ...>``.
           Lands in ``entity.vgeoms`` (Genesis splits collision vs visual
           geoms). ``RigidVisGeom.get_pos()`` returns the world transform,
           but Genesis does NOT store the geom name — we have to match by
           ``(parent_link, init_pos)``.
  Mode B — **Dummy fixed body**: ``<body name="...marker"><geom .../></body>``
           with no joint (welded to the parent). Genesis preserves the body
           name, so ``entity.get_link(name).get_pos()`` works directly.

For each, we set the ball pose to a non-trivial (pos, quat), step kinematics,
read the marker world position, and compare with the analytical
``body_pos + R(body_quat) @ marker_local_offset``.

If Mode B PASSes, that's the recommended path (cleanest — name-addressable
and zero parsing tricks). Mode A is a useful backup.

Usage:
    python -m rlworld.scripts.diag.check_genesis_marker_geom
"""

from __future__ import annotations

import os
import tempfile
import textwrap

import numpy as np

_MARKER_LOCAL = np.array([0.10, 0.05, -0.03], dtype=np.float64)

_MJCF_MODE_A = textwrap.dedent(f"""
<mujoco model="marker_geom_test">
  <option gravity="0 0 0"/>
  <worldbody>
    <body name="ball" pos="0 0 1">
      <freejoint/>
      <geom name="ball_visual" type="sphere" size="0.05" rgba="0.6 0.4 0.4 1"/>
      <geom name="marker" type="sphere" size="0.002"
            pos="{_MARKER_LOCAL[0]} {_MARKER_LOCAL[1]} {_MARKER_LOCAL[2]}"
            contype="0" conaffinity="0" group="5" rgba="1 1 0 1"/>
    </body>
  </worldbody>
</mujoco>
""").strip()

_MJCF_MODE_B = textwrap.dedent(f"""
<mujoco model="marker_body_test">
  <option gravity="0 0 0"/>
  <worldbody>
    <body name="ball" pos="0 0 1">
      <freejoint/>
      <geom name="ball_visual" type="sphere" size="0.05" rgba="0.6 0.4 0.4 1"/>
      <body name="ball_marker"
            pos="{_MARKER_LOCAL[0]} {_MARKER_LOCAL[1]} {_MARKER_LOCAL[2]}">
        <inertial pos="0 0 0" mass="1e-6" diaginertia="1e-9 1e-9 1e-9"/>
        <geom type="sphere" size="0.002" contype="0" conaffinity="0" group="5" rgba="1 1 0 1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
""").strip()


def _q_rot_wxyz(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """R(q) @ v with q in (w, x, y, z) — forward rotate."""
    w = float(q[0])
    qv = np.array([float(q[1]), float(q[2]), float(q[3])])
    a = v * (2.0 * w * w - 1.0)
    b = np.cross(qv, v) * (2.0 * w)
    c = qv * (qv @ v) * 2.0
    return a + b + c


def _as_np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _cases():
    s = float(np.sqrt(0.5))
    return [
        (np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0, 0.0])),
        (np.array([1.2, -0.4, 0.6]), np.array([s, 0.0, 0.0, s])),  # 90° about z
        (np.array([0.0, 0.5, 0.9]), np.array([0.8660254, 0.5, 0.0, 0.0])),  # 60° about x
        (np.array([-0.3, 0.7, 1.4]), np.array([s, s, 0.0, 0.0])),  # 90° about x
    ]


def _run_marker_geom(gs, torch, path: str) -> float | None:
    """Mode A: marker is a vgeom. Returns max residual or None on infra failure."""
    print("\n" + "=" * 60)
    print("Mode A — marker as <geom contype='0' conaffinity='0'>")
    print("=" * 60)
    scene = gs.Scene(show_viewer=False)
    entity = scene.add_entity(gs.morphs.MJCF(file=path))
    scene.build(n_envs=1)

    print(f"entity.geoms (collision)  : {len(entity.geoms)}")
    print(f"entity.vgeoms (visual)    : {len(entity.vgeoms)}")
    print(f"entity.links              : {[getattr(L, 'name', None) for L in entity.links]}")

    ball_link = next((L for L in entity.links if getattr(L, "name", None) == "ball"), None)
    if ball_link is None:
        print("FAIL — 'ball' link not found.")
        return None

    # Marker has no name in Genesis. Match by parent link + init_pos.
    cand = []
    for vg in entity.vgeoms:
        parent = getattr(vg, "link", None) or getattr(vg, "_link", None)
        if parent is None or getattr(parent, "name", None) != "ball":
            continue
        init_pos = getattr(vg, "_init_pos", None)
        if init_pos is None:
            continue
        if np.linalg.norm(np.asarray(init_pos) - _MARKER_LOCAL) < 1e-4:
            cand.append(vg)
    if not cand:
        print("FAIL — no vgeom on 'ball' with init_pos matching the marker offset.")
        return None
    if len(cand) > 1:
        print(f"FAIL — multiple vgeoms matched the marker offset ({len(cand)}).")
        return None
    marker = cand[0]
    print(f"OK — found marker vgeom (idx={getattr(marker, '_idx', '?')}, init_pos={_as_np(marker._init_pos)}).")

    max_err = 0.0
    for i, (pos, quat) in enumerate(_cases()):
        entity.set_pos(torch.tensor(pos, dtype=torch.float32).unsqueeze(0))
        entity.set_quat(torch.tensor(quat, dtype=torch.float32).unsqueeze(0))
        scene.step()

        link_pos = _as_np(ball_link.get_pos()).reshape(-1)[:3]
        link_quat = _as_np(ball_link.get_quat()).reshape(-1)[:4]
        marker_pos = _as_np(marker.get_pos()).reshape(-1)[:3]
        expected = link_pos + _q_rot_wxyz(link_quat, _MARKER_LOCAL)
        err = float(np.linalg.norm(marker_pos - expected))
        max_err = max(max_err, err)
        print(
            f"  case {i}: link_pos={link_pos}  link_quat={link_quat}\n"
            f"          marker_pos={marker_pos}  expected={expected}  resid={err:.4g}"
        )
    return max_err


def _run_marker_body(gs, torch, path: str) -> float | None:
    """Mode B: marker is a fixed-welded child body. Returns max residual or None on infra failure."""
    print("\n" + "=" * 60)
    print("Mode B — marker as <body><geom .../></body> (welded child)")
    print("=" * 60)
    scene = gs.Scene(show_viewer=False)
    entity = scene.add_entity(gs.morphs.MJCF(file=path))
    scene.build(n_envs=1)

    print(f"entity.links              : {[getattr(L, 'name', None) for L in entity.links]}")

    ball_link = next((L for L in entity.links if getattr(L, "name", None) == "ball"), None)
    marker_link = next((L for L in entity.links if getattr(L, "name", None) == "ball_marker"), None)
    if ball_link is None or marker_link is None:
        print(
            f"FAIL — missing link(s). ball={ball_link is not None}, "
            f"ball_marker={marker_link is not None}. (Was the child body fused?)"
        )
        return None
    print("OK — both 'ball' and 'ball_marker' links present.")

    max_err = 0.0
    for i, (pos, quat) in enumerate(_cases()):
        entity.set_pos(torch.tensor(pos, dtype=torch.float32).unsqueeze(0))
        entity.set_quat(torch.tensor(quat, dtype=torch.float32).unsqueeze(0))
        scene.step()

        link_pos = _as_np(ball_link.get_pos()).reshape(-1)[:3]
        link_quat = _as_np(ball_link.get_quat()).reshape(-1)[:4]
        marker_pos = _as_np(marker_link.get_pos()).reshape(-1)[:3]
        expected = link_pos + _q_rot_wxyz(link_quat, _MARKER_LOCAL)
        err = float(np.linalg.norm(marker_pos - expected))
        max_err = max(max_err, err)
        print(
            f"  case {i}: link_pos={link_pos}  link_quat={link_quat}\n"
            f"          marker_pos={marker_pos}  expected={expected}  resid={err:.4g}"
        )
    return max_err


def main() -> int:
    import genesis as gs
    import torch

    gs.init(backend=gs.cpu, logging_level="warning")

    tmp = []
    try:
        for name, body in (("a", _MJCF_MODE_A), ("b", _MJCF_MODE_B)):
            with tempfile.NamedTemporaryFile("w", suffix=f"_{name}.xml", delete=False) as f:
                f.write(body)
                tmp.append(f.name)
        path_a, path_b = tmp

        tol = 1e-4
        err_a = _run_marker_geom(gs, torch, path_a)
        err_b = _run_marker_body(gs, torch, path_b)

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        ok_a = err_a is not None and err_a < tol
        ok_b = err_b is not None and err_b < tol
        print(f"  Mode A (marker geom)  : {'PASS' if ok_a else 'FAIL'}" f"  (max resid {err_a!r}, tol {tol:.0e})")
        print(f"  Mode B (welded body)  : {'PASS' if ok_b else 'FAIL'}" f"  (max resid {err_b!r}, tol {tol:.0e})")
        if ok_b:
            print("\n→ Use Mode B (dummy fixed body) — name-addressable, cleanest.")
            return 0
        if ok_a:
            print("\n→ Mode A works; Mode B failed (likely Genesis fused the child body).")
            print("  Either disable fuse via morph.merge_fixed_links=False (default), or use Mode A.")
            return 0
        print("\n→ Both failed. Need a different approach.")
        return 1
    finally:
        for p in tmp:
            try:
                os.unlink(p)
            except OSError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
