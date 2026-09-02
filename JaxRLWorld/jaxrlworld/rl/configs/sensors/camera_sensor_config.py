"""Simulator-agnostic camera sensor configuration.

The same config object must place the same camera on mjlab and on
Newton, and the two must return the same picture. That is harder than it
looks, because a camera is defined by three separate things and every
backend expresses each of them differently:

* **where it is** — mjlab rides an MJCF ``<camera>`` element that the
  physics moves for free; Newton has no such element and needs a
  camera-to-world transform handed to it every frame.
* **which way it points** — a camera looks along one axis of its own
  frame, and backends disagree about which. MuJoCo and Newton both use
  **-Z forward, +Y up** (Newton's ray kernel takes the camera forward as
  ``vec3f(0, 0, -1)``), so ``convention="opengl"`` is the only one
  implemented here. IsaacLab's ``CameraCfg.OffsetCfg`` carries the same
  field for the same reason, and names three.
* **how wide it sees** — MuJoCo writes a vertical FOV in DEGREES (or a
  sensor size and focal length); Newton wants vertical FOV in RADIANS.

Rather than have a preset write the extrinsics out by hand for the
backends that need them, ``camera_name`` names a ``<camera>`` element in
the entity's own MJCF and every backend reads its placement from that
one file. The asset stays the single source of truth, so the two
backends agree by construction rather than by someone keeping two sets
of numbers in step.

Backend support matrix
----------------------
========================  ======  ========  =========
field / value             mjlab   Newton    Genesis
========================  ======  ========  =========
camera_name (from MJCF)    yes     yes       yes
link_name + offset         yes     yes       yes
data_types={"depth"}       yes     yes       yes
data_types={"rgb"}         yes     no        no
enabled_geom_groups        yes     no        no
========================  ======  ========  =========

Genesis draws through Madrona's batch renderer, which has to be asked
for when the scene is constructed and requires ``gs_madrona``.

Depth contract
--------------
Every backend adapter must return depth as **forward-projected metres**
— the distance along the camera's view axis, which is what a real depth
camera reports — with **0.0 meaning the ray hit nothing**. mjlab is
already both. Newton is both provided the adapter reads
``forward_depth`` (not ``depth``, which is ray distance) and leaves
``ClearData.clear_depth`` at its 0.0 default. Genesis reports a miss as
+inf and its adapter converts it — measured, not assumed: the value was
read off the cross-sim diag.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

__all__ = [
    "CameraOffset",
    "CameraSensorCfg",
    "MjcfCameraPlacement",
    "resolve_mjcf_camera",
    "resolve_mjcf_geom_groups",
]


@dataclass
class CameraOffset:
    """Where the camera sits relative to the link it rides on."""

    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Translation in the parent link's frame, metres."""

    quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    """Rotation in the parent link's frame, ``(w, x, y, z)``."""

    convention: Literal["opengl"] = "opengl"
    """Which axis of the camera frame the camera looks along.

    ``"opengl"``: forward is **-Z**, up is **+Y**. This is MuJoCo's
    convention and Newton's, so it is the only one implemented. A
    backend that defines its camera differently must convert here rather
    than leave the discrepancy to be discovered in the images.
    """


@dataclass
class MjcfCameraPlacement:
    """A ``<camera>`` element's placement, resolved from the MJCF."""

    body: str
    """Name of the nearest enclosing body that the physics moves."""

    pos: tuple[float, float, float]
    """Camera position in that body's frame."""

    quat: tuple[float, float, float, float]
    """Camera orientation in that body's frame, ``(w, x, y, z)``."""

    fovy: float
    """Vertical field of view in DEGREES."""

    focal: tuple[float, float] | None
    """``(fx, fy)`` focal length, when the element states one."""

    sensorsize: tuple[float, float] | None
    """``(width, height)`` of the sensor, when the element states one.

    Present far more often than it looks: a real camera is not square,
    and a ``<camera>`` written from a datasheet describes it this way
    rather than with an fovy. Ignoring it and keeping only the vertical
    angle silently narrows the horizontal one to match the image's
    aspect ratio — on the D405 that is 90.5 degrees collapsed to 58.
    """


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of two ``(w, x, y, z)`` quaternions."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate ``v`` by the ``(w, x, y, z)`` quaternion ``q``."""
    w, x, y, z = q
    u = np.array([x, y, z])
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def _read_frame(element: ET.Element) -> tuple[np.ndarray, np.ndarray]:
    """A body's or camera's ``pos`` / ``quat`` attributes, with MJCF defaults."""
    pos = np.fromstring(element.get("pos", "0 0 0"), sep=" ")
    quat = np.fromstring(element.get("quat", "1 0 0 0"), sep=" ")
    if element.get("axisangle") or element.get("euler") or element.get("xyaxes") or element.get("zaxis"):
        raise NotImplementedError(
            f"MJCF element {element.get('name')!r} orients itself with something other than 'quat'. "
            "Only 'quat' is read here, so the camera would be placed wrongly and silently."
        )
    return pos, quat


def resolve_mjcf_camera(mjcf_path: str, camera_name: str) -> MjcfCameraPlacement:
    """Find a ``<camera>`` in an MJCF and express it in its body's frame.

    A camera is usually nested a body or two below the link that actually
    moves — on the YAM arm it hangs off ``link_6`` through a camera
    housing and a frame body. Those intermediate bodies are welded (they
    carry no joint), so the chain collapses into one fixed offset from
    the nearest jointed ancestor, which is what a backend without MJCF
    cameras needs to be told.
    """
    root = ET.parse(mjcf_path).getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"{mjcf_path} has no <worldbody>.")

    found: list[tuple[list[ET.Element], ET.Element]] = []

    def walk(node: ET.Element, chain: list[ET.Element]) -> None:
        for body in node.findall("body"):
            body_chain = chain + [body]
            for camera in body.findall("camera"):
                if camera.get("name") == camera_name:
                    found.append((body_chain, camera))
            walk(body, body_chain)

    walk(worldbody, [])
    if not found:
        names = [c.get("name") for c in root.iter("camera")]
        raise KeyError(f"No <camera name={camera_name!r}> in {mjcf_path}. Cameras: {names}")
    if len(found) > 1:
        raise ValueError(f"{mjcf_path} declares {len(found)} cameras named {camera_name!r}.")

    body_chain, camera = found[0]

    # Walk down from the last jointed body, composing the welded bodies
    # between it and the camera into a single offset.
    anchor_index = 0
    for index, body in enumerate(body_chain):
        if body.find("joint") is not None or body.find("freejoint") is not None:
            anchor_index = index
    anchor = body_chain[anchor_index]

    pos = np.zeros(3)
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    for body in body_chain[anchor_index + 1 :]:
        body_pos, body_quat = _read_frame(body)
        pos = pos + _quat_rotate(quat, body_pos)
        quat = _quat_mul(quat, body_quat)
    camera_pos, camera_quat = _read_frame(camera)
    pos = pos + _quat_rotate(quat, camera_pos)
    quat = _quat_mul(quat, camera_quat)

    sensorsize_attr = camera.get("sensorsize")
    focal_attr = camera.get("focal")
    sensorsize = tuple(np.fromstring(sensorsize_attr, sep=" ")) if sensorsize_attr else None
    focal = tuple(np.fromstring(focal_attr, sep=" ")) if focal_attr else None

    fovy_attr = camera.get("fovy")
    if fovy_attr is not None:
        fovy_deg = float(fovy_attr)
    elif sensorsize is not None and focal is not None:
        fovy_deg = float(np.degrees(2.0 * np.arctan(0.5 * sensorsize[1] / focal[1])))
    else:
        # MuJoCo's own default when a camera says nothing about its
        # field of view.
        fovy_deg = 45.0

    anchor_name = anchor.get("name")
    if anchor_name is None:
        raise ValueError(f"The body holding camera {camera_name!r} in {mjcf_path} has no name.")
    return MjcfCameraPlacement(
        body=anchor_name,
        pos=tuple(float(v) for v in pos),
        quat=tuple(float(v) for v in quat),
        fovy=fovy_deg,
        focal=None if focal is None else (float(focal[0]), float(focal[1])),
        sensorsize=None if sensorsize is None else (float(sensorsize[0]), float(sensorsize[1])),
    )


def resolve_mjcf_geom_groups(mjcf_path: str, visible_geometry: str) -> tuple[int, ...]:
    """Which MuJoCo geom groups hold the requested kind of geometry.

    A geom is visual when it cannot collide — ``contype`` and
    ``conaffinity`` both zero — which is MuJoCo's only formal statement
    of the distinction; the group number is a convention the asset picks.
    Defaults declared in ``<default class=...>`` blocks are resolved,
    since that is where assets usually put both.

    Raises when a group holds both kinds: mjlab's camera can only filter
    by group, so no setting expresses the intent, and picking one anyway
    would quietly show the camera the wrong robot.
    """
    if visible_geometry not in ("collision", "visual", "all"):
        raise ValueError(f"visible_geometry must be 'collision', 'visual' or 'all', got {visible_geometry!r}.")

    root = ET.parse(mjcf_path).getroot()

    # class name -> (group, collides), inherited through nested defaults.
    class_defaults: dict[str, tuple[int, bool]] = {}

    def walk_defaults(node: ET.Element, inherited: tuple[int, bool]) -> None:
        for default in node.findall("default"):
            group, collides = inherited
            geom = default.find("geom")
            if geom is not None:
                group = int(geom.get("group", group))
                contype = int(geom.get("contype", "1"))
                conaffinity = int(geom.get("conaffinity", "1"))
                collides = bool(contype or conaffinity)
            name = default.get("class")
            if name is not None:
                class_defaults[name] = (group, collides)
            walk_defaults(default, (group, collides))

    for defaults in root.findall("default"):
        walk_defaults(defaults, (0, True))
        geom = defaults.find("geom")
        if geom is not None and defaults.get("class") is None:
            class_defaults[""] = (
                int(geom.get("group", "0")),
                bool(int(geom.get("contype", "1")) or int(geom.get("conaffinity", "1"))),
            )

    groups: dict[int, set[bool]] = {}
    # Only geoms that exist in the scene. ``root.iter`` would also walk
    # the <geom> elements INSIDE <default> blocks, which are attribute
    # templates rather than geoms — and a nested default that sets only
    # friction (inheriting its group from the enclosing class, as
    # MuJoCo defaults do) then reads as an anonymous colliding geom in
    # group 0, failing the mixed-group check on an asset whose real
    # geoms are split perfectly.
    worldbody = root.find("worldbody")
    scene_geoms = worldbody.iter("geom") if worldbody is not None else ()
    for geom in scene_geoms:
        base_group, base_collides = class_defaults.get(geom.get("class", ""), (0, True))
        group = int(geom.get("group", base_group))
        contype = geom.get("contype")
        conaffinity = geom.get("conaffinity")
        if contype is None and conaffinity is None:
            collides = base_collides
        else:
            collides = bool(int(contype or "1") or int(conaffinity or "1"))
        groups.setdefault(group, set()).add(collides)

    mixed = sorted(group for group, kinds in groups.items() if len(kinds) > 1)
    if mixed and visible_geometry != "all":
        raise ValueError(
            f"{mjcf_path} puts both colliding and non-colliding geoms in group(s) {mixed}. "
            "mjlab's camera filters by group alone, so 'collision' and 'visual' cannot be told apart "
            "there. Split the groups in the asset, or pin enabled_geom_groups by hand."
        )

    if visible_geometry == "all":
        return tuple(sorted(groups))
    want_collides = visible_geometry == "collision"
    return tuple(sorted(group for group, kinds in groups.items() if want_collides in kinds))


@dataclass
class CameraSensorCfg:
    """A camera riding on one link of one entity.

    Either name a ``<camera>`` in the entity's MJCF (``camera_name``) and
    let every backend read its placement from there, or give
    ``link_name`` and ``offset`` directly.
    """

    name: str
    """Key this sensor is stored under, and the ``sensor_name`` an
    observation term refers to."""

    entity_name: str = "robot"
    """Entity carrying the camera."""

    camera_name: str | None = None
    """Name of a ``<camera>`` element in the entity's MJCF. When set,
    ``link_name`` / ``offset`` / ``fovy`` are read from the asset and
    must not be given."""

    link_name: str | None = None
    """Link the camera rides on, when placing it by hand."""

    offset: CameraOffset = field(default_factory=CameraOffset)
    """Placement in that link's frame, when placing it by hand."""

    width: int = 32
    height: int = 32
    """Image size in pixels. mjlab's own vision task uses 32x32."""

    fovy: float | None = None
    """Vertical field of view in DEGREES, MuJoCo's convention. None with
    ``camera_name`` means "whatever the asset says"."""

    data_types: tuple[str, ...] = ("depth",)
    """Which channels to render."""

    visible_geometry: Literal["collision", "visual", "all"] = "collision"
    """Which of an asset's shapes the camera can see.

    MuJoCo assets carry two descriptions of the same robot: detailed
    visual meshes that cannot collide, and coarse primitives that can.
    Which one a camera sees changes the image completely — on the YAM
    arm the visual meshes are the real shape and the colliders are
    capsules — and the two backends express the choice in different
    terms, so it is stated once here and translated.

    * mjlab filters by geom GROUP, so the groups are derived from the
      asset by ``resolve_mjcf_geom_groups``: a geom counts as visual when
      it cannot collide (``contype`` and ``conaffinity`` both zero). A
      group holding both kinds cannot be expressed as a filter, and that
      raises rather than being approximated.
    * Newton flags each shape ``VISIBLE`` and ``COLLIDE_SHAPES``
      independently, so the adapter sets the visibility bit directly and
      the classification is exact.

    ``"collision"`` matches mjlab's own vision task, which passes geom
    groups (0, 3) for this arm.
    """

    enabled_geom_groups: tuple[int, ...] | None = None
    """MuJoCo geom groups the camera can see. **mjlab only.**

    Leave as None to have them derived from ``visible_geometry``. Set it
    to pin the groups by hand, which bypasses that derivation entirely.

    A geom in a group excluded here is invisible to mjlab, and unless
    Newton is told the same thing it stays solid there — which is the
    single largest reason two backends disagree about the same scene.
    """

    def __post_init__(self):
        if (self.camera_name is None) == (self.link_name is None):
            raise ValueError(
                f"Camera {self.name!r} must be placed either by camera_name (read from the MJCF) "
                "or by link_name + offset, and not both."
            )
        if self.camera_name is not None and self.offset != CameraOffset():
            raise ValueError(
                f"Camera {self.name!r} names an MJCF camera and also sets an offset. "
                "The asset's placement is the whole point of camera_name; drop one."
            )
        unknown = set(self.data_types) - {"depth", "rgb"}
        if unknown:
            raise ValueError(f"Camera {self.name!r} asks for unsupported data types {sorted(unknown)}.")

    def resolve(self, mjcf_path: str | None) -> tuple[str, CameraOffset, MjcfCameraPlacement]:
        """Placement and optics, whatever way this camera was declared.

        Returns:
            ``(link_name, offset, optics)``. The optics carry the sensor
            size and focal length when the asset states them, because a
            vertical angle alone cannot describe a camera whose image is
            not square.
        """
        if self.camera_name is None:
            if self.fovy is None:
                raise ValueError(f"Camera {self.name!r} is placed by hand and so must state its fovy.")
            optics = MjcfCameraPlacement(
                body=self.link_name,
                pos=self.offset.pos,
                quat=self.offset.quat,
                fovy=self.fovy,
                focal=None,
                sensorsize=None,
            )
            return self.link_name, self.offset, optics
        if mjcf_path is None:
            raise ValueError(
                f"Camera {self.name!r} reads its placement from entity {self.entity_name!r}'s MJCF, "
                "but that entity has no mjcf_path."
            )
        placement = resolve_mjcf_camera(mjcf_path, self.camera_name)
        offset = CameraOffset(pos=placement.pos, quat=placement.quat)
        if self.fovy is not None:
            placement = MjcfCameraPlacement(
                body=placement.body, pos=placement.pos, quat=placement.quat, fovy=self.fovy, focal=None, sensorsize=None
            )
        return placement.body, offset, placement
