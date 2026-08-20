"""The tong-armed YAM: same arm, same spring, same grip on all three sims?

``assets/i2rt_yam/xmls/yam_tong.xml`` is the YAM with a spring tong where
its gripper used to be. Three separate things could differ between
backends and each would look like the others:

* the **kinematics** — the tong is mounted through a body frame rotated
  180 degrees about a diagonal axis, and a quaternion convention that is
  read differently puts the jaws somewhere plausible but wrong;
* the **spring** — verified standalone by
  ``tong_spring_cross_sim_diag``, but mounting changes the frame it lives
  in, and a backend that composes transforms differently could change the
  axis the hinge turns about;
* the **grip** — friction, contact softness and the solver's handling of
  a 12 g box squeezed between two pads, which is what actually decides
  whether the task is possible at all.

So all three are measured, separately, each against the other backends
and (where there is one) against an exact answer:

  0. **Parse.** Joint names in order, body masses, and every number on the
     tong hinge, as each backend stored them.
  1. **Kinematics, no physics.** Arm poses are imposed and the resulting
     link poses read back without stepping — the only way to compare
     kinematics without a solver's dynamics mixed in. Poses are compared
     in the ARM's OWN frame, since backends lay their worlds out
     differently.
  2. **The spring, mounted.** The staircase and release of the standalone
     tong diag, re-run on the arm with its six joints welded. The settled
     angle must still be the torque over the stiffness, and it must still
     spring fully open, exactly as it does off the arm.
  3. **The grip.** A 25 mm cube is placed between the pads and the tong
     is held shut on it. Here the arm points along +x and the jaws close
     along +-y, so gravity pulls the cube along neither: it is held by
     FRICTION alone, which is the honest version of this test. Then the
     torque is released and the cube must fall — a grip test that cannot
     fail is not measuring anything.

Run::

    jaxpy -m rlworld.scripts.diag.yam_tong_cross_sim_diag
    jaxpy -m rlworld.scripts.diag.yam_tong_cross_sim_diag --sims mujoco newton
"""

from __future__ import annotations

import argparse
import math
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ARM_XML = Path("./JaxRLWorld/rlworld/assets/i2rt_yam/xmls/yam_tong.xml")
"""Relative to the SimForge root, like every other asset path here."""

ARM_JOINTS = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
HINGE_JOINT = "tong_hinge"
TRACKED_BODIES = ("link_6", "tong_base", "tong_jaw")
ROOT_BODY = "arm"

DT = 0.002

IMPRATIO = 10.0
"""How stiff the friction cone is relative to the normal direction, taken
from the YAM preset's own solver settings rather than left at each
backend's default. It is not a detail here: at MuJoCo's default of 1 the
gripped cube creeps 9.8 mm in two seconds, at 10 it settles in 1.6 mm and
stops. A grip pass run on defaults would be measuring the solver's
tolerance for sliding rather than whether the tong holds."""


# ── What the asset says about the tong ────────────────────────────
def _tong_facts(path: Path) -> dict:
    """The spring and the jaws, read out of the arm asset itself.

    Parsed rather than copied. The tool's mass and spring were restated
    here once and went stale the day the jaws were made thicker: all
    three backends then reported the asset's own values and were marked
    wrong for it. ElementTree is none of the three engines, so it is
    still an independent reference — just one that cannot fall behind.
    """
    root = ET.parse(path).getroot()

    def find(tag: str, name: str):
        for element in root.iter(tag):
            if element.get("name") == name:
                return element
        raise ValueError(f"{path}: no <{tag}> named {name!r}")

    hinge = find("joint", HINGE_JOINT)
    return {
        "stiffness": float(hinge.get("stiffness")),
        "damping": float(hinge.get("damping")),
        "arm_mass": float(find("body", "tong_base").find("inertial").get("mass")),
    }


_TONG = _tong_facts(ARM_XML)

STIFFNESS = _TONG["stiffness"]  # N*m/rad
DAMPING = _TONG["damping"]  # N*m*s/rad
ARM_PIECE_MASS = _TONG["arm_mass"]  # kg, each tong jaw
OPEN_ANGLE = 0.0  # rad, spring rest and the open stop
CLOSED_ANGLE = 0.3244  # rad, pads touching
CUBE_ANGLE = 0.0695  # rad, closed on a 25 mm cube

# ── The cube, and where it goes ───────────────────────────────────
CUBE_SIZE = 0.025  # m
CUBE_MASS = 0.012  # kg, matching props/cube_25mm.urdf
CUBE_AT = (0.29519, 0.01852, 0.164)
"""The midpoint between the pads' inner faces with the arm at all-zero
and the hinge at CUBE_ANGLE, measured off the asset rather than guessed.
The faces are 25.4 mm apart there, so the cube starts just touching."""

GRIP_TORQUE = round(0.32 * STIFFNESS * CLOSED_ANGLE, 6)
"""N*m. The spring alone needs 0.047 at this angle, so this leaves about
0.05 to squeeze with — roughly 0.5 N at the pads against the 0.12 N the
cube weighs. Comfortable rather than marginal, on purpose: this pass asks
whether the three backends agree, not how close to slipping it can get."""

TORQUE_STAIRCASE = tuple(round(fraction * STIFFNESS * CLOSED_ANGLE, 6) for fraction in (0.1, 0.3, 0.6))
SETTLE_STEPS = 600
GRIP_STEPS = 1000
"""2 s. Long enough for a cube that is going to slip to have fallen far
enough to be unmistakable."""

# ── Tolerances ────────────────────────────────────────────────────
TOL_POSE = 1e-4
"""m, and radians for orientation. Kinematics is arithmetic — no solver,
no timestep — so the backends have nothing to disagree about beyond
float32 rounding. Anything above this is a convention mismatch."""

TOL_SETTLED = 2e-3
"""rad. As in the standalone tong diag: a steady state under a held
torque, with no integrator difference left in it."""

TOL_REOPEN = 1e-3

TOL_GRIP_POS = 3e-3
"""m. How differently the backends may hold the same cube. Contact is
where solvers genuinely differ — penetration depth, friction
regularisation — so this is looser than the kinematic bound and tighter
than anything that would change the task."""

HELD_DROP = 5e-3
"""m. Above this the cube has left the pads rather than settled into
them."""

DROPPED_FALL = 0.05
"""m. Below this the released cube has not actually fallen, which would
mean the pass proves nothing: it must be able to fail."""


@dataclass
class Reading:
    name: str
    parsed: dict = field(default_factory=dict)
    poses: np.ndarray | None = None  # (n_poses, n_bodies, 7)
    settled: dict = field(default_factory=dict)
    reopened: dict = field(default_factory=dict)
    held: np.ndarray | None = None
    dropped: np.ndarray | None = None


# ══════════════════════════════════════════════════════════════════
# Derived assets
# ══════════════════════════════════════════════════════════════════


def _load_tree() -> ET.ElementTree:
    tree = ET.parse(ARM_XML)
    # The asset's meshdir is relative to its own directory, and the
    # derived copies live in a temp directory. Made absolute here so the
    # meshes still resolve; every backend reads the same attribute.
    compiler = tree.getroot().find("compiler")
    compiler.set("meshdir", str((ARM_XML.parent / compiler.get("meshdir")).resolve()))
    return tree


def _write(tree: ET.ElementTree, name: str) -> str:
    out = Path(tempfile.mkdtemp(prefix="yam_tong_diag_")) / name
    tree.write(out, encoding="unicode")
    return str(out)


def full_asset() -> str:
    """The arm as authored, for the kinematics pass."""
    return _write(_load_tree(), "yam_tong.xml")


def welded_asset(with_cube: bool) -> str:
    """The arm with its six joints deleted, so only the tong can move.

    Welding rather than holding: writing the arm's joints back after
    every step does not stop the arm reacting to the tong DURING the
    step, so the hinge would be swinging against a moving frame and the
    torque-over-stiffness answer would stop being exact. With the joints
    gone the mounted tong is the same one-degree-of-freedom system the
    standalone diag measures, which is exactly the comparison wanted:
    mounting must not have changed the spring.

    All six freeze at zero, which points the arm along +x with the jaws
    closing along +-y — the pose CUBE_AT was measured in, and the one
    that makes the grip pass a friction test rather than a shelf.
    """
    tree = _load_tree()
    root = tree.getroot()
    removed = 0
    for body in root.iter("body"):
        for joint in list(body.findall("joint")):
            if joint.get("name") in ARM_JOINTS:
                body.remove(joint)
                removed += 1
    if removed != len(ARM_JOINTS):
        raise ValueError(f"Deleted {removed} arm joints, expected {len(ARM_JOINTS)}; the asset changed shape.")

    if with_cube:
        half = CUBE_SIZE / 2
        inertia = CUBE_MASS * CUBE_SIZE**2 / 6
        cube = ET.SubElement(root.find("worldbody"), "body")
        cube.set("name", "cube")
        cube.set("pos", " ".join(f"{v:.6f}" for v in CUBE_AT))
        ET.SubElement(cube, "freejoint", {"name": "cube_free"})
        ET.SubElement(
            cube,
            "inertial",
            {"pos": "0 0 0", "mass": str(CUBE_MASS), "diaginertia": f"{inertia:.6g} {inertia:.6g} {inertia:.6g}"},
        )
        ET.SubElement(
            cube,
            "geom",
            {
                "name": "cube",
                "type": "box",
                "size": f"{half} {half} {half}",
                "rgba": "0.85 0.4 0.2 1",
                "friction": "0.9 0.005 0.0001",
                "condim": "3",
            },
        )
    return _write(tree, "yam_tong_welded.xml")


# ══════════════════════════════════════════════════════════════════
# Backends
# ══════════════════════════════════════════════════════════════════


def _quat_angle(a: np.ndarray, b: np.ndarray) -> float:
    """Shortest rotation between two wxyz quaternions, in radians."""
    dot = abs(float(np.dot(a, b)))
    return 2.0 * math.acos(min(1.0, dot))


class MujocoArm:
    name = "mujoco"

    def __init__(self, xml: str, gravity: bool):
        import mujoco

        self._mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(xml)
        self.model.opt.timestep = DT
        self.model.opt.gravity[:] = (0.0, 0.0, -9.81 if gravity else 0.0)
        self.model.opt.impratio = IMPRATIO
        self.model.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
        self.data = mujoco.MjData(self.model)
        self._jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, HINGE_JOINT)
        self._qadr = int(self.model.jnt_qposadr[self._jid])
        self._dadr = int(self.model.jnt_dofadr[self._jid])
        self._arm_qadr = [
            int(self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)])
            for n in ARM_JOINTS
            if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) >= 0
        ]
        self._torque = 0.0
        mujoco.mj_forward(self.model, self.data)

    def describe(self) -> dict:
        m = self.model
        names = [self._mj.mj_id2name(m, self._mj.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)]
        bodies = {
            self._mj.mj_id2name(m, self._mj.mjtObj.mjOBJ_BODY, i): float(m.body_mass[i]) for i in range(1, m.nbody)
        }
        return {
            "joints": names,
            "tong_arm_masses": [bodies["tong_base"], bodies["tong_jaw"]],
            "total_mass": round(float(sum(m.body_mass)), 6),
            "stiffness": float(m.jnt_stiffness[self._jid]),
            "damping": float(m.dof_damping[self._dadr]),
            "armature": float(m.dof_armature[self._dadr]),
            "range": [float(v) for v in m.jnt_range[self._jid]],
            "spring_rest": float(m.qpos_spring[self._qadr]),
        }

    def set_arm(self, angles: np.ndarray) -> None:
        for adr, value in zip(self._arm_qadr, angles, strict=True):
            self.data.qpos[adr] = value
        self._mj.mj_forward(self.model, self.data)

    def body_poses(self) -> np.ndarray:
        self._mj.mj_forward(self.model, self.data)
        out = []
        root = self._mj.mj_name2id(self.model, self._mj.mjtObj.mjOBJ_BODY, ROOT_BODY)
        origin = self.data.xpos[root].copy()
        for name in TRACKED_BODIES:
            i = self._mj.mj_name2id(self.model, self._mj.mjtObj.mjOBJ_BODY, name)
            out.append(np.concatenate([self.data.xpos[i] - origin, self.data.xquat[i]]))
        return np.asarray(out)

    def set_hinge(self, q: float) -> None:
        self.data.qvel[:] = 0.0
        self.data.qpos[self._qadr] = q
        self._mj.mj_forward(self.model, self.data)

    def set_torque(self, tau: float) -> None:
        self._torque = tau

    def step(self) -> None:
        self.data.qfrc_applied[:] = 0.0
        self.data.qfrc_applied[self._dadr] = self._torque
        self._mj.mj_step(self.model, self.data)

    def hinge(self) -> float:
        return float(self.data.qpos[self._qadr])

    def cube_pos(self) -> np.ndarray:
        i = self._mj.mj_name2id(self.model, self._mj.mjtObj.mjOBJ_BODY, "cube")
        return self.data.xpos[i].copy()


class NewtonArm:
    name = "newton"

    def __init__(self, xml: str, gravity: bool):
        import newton
        import warp as wp

        from rlworld.rl.envs.utils.warp_logging import configure_warp_logging

        configure_warp_logging()
        self._newton, self._wp = newton, wp

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81 if gravity else 0.0))
        builder.add_mjcf(xml, floating=False, collapse_fixed_joints=False, parse_sites=False)
        # Newton always sums its own joint PD into the applied force; left
        # at import defaults it would pull every joint towards zero, which
        # on the hinge is indistinguishable from a stiffer spring and on
        # the arm would fight the pose being imposed.
        for i in range(len(builder.joint_target_ke)):
            builder.joint_target_ke[i] = 0.0
            builder.joint_target_kd[i] = 0.0

        self.model = builder.finalize()
        self.solver = newton.solvers.SolverMuJoCo(self.model, impratio=IMPRATIO, cone="pyramidal")
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self._labels = list(getattr(self.model, "joint_label", None) or self.model.joint_key)
        self._body_labels = list(getattr(self.model, "body_label", None) or self.model.body_key)
        self._q_start = wp.to_torch(self.model.joint_q_start).cpu().numpy()
        self._qd_start = wp.to_torch(self.model.joint_qd_start).cpu().numpy()
        self._qadr = int(self._q_start[self._joint_index(HINGE_JOINT)])
        self._dadr = int(self._qd_start[self._joint_index(HINGE_JOINT)])
        self._arm_qadr = [int(self._q_start[self._joint_index(n)]) for n in ARM_JOINTS if self._has_joint(n)]
        self._joint_f = wp.to_torch(self.control.joint_f)
        self._sync()

    def _has_joint(self, name: str) -> bool:
        return any(label.split("/")[-1] == name for label in self._labels)

    def _joint_index(self, name: str) -> int:
        matches = [i for i, label in enumerate(self._labels) if label.split("/")[-1] == name]
        if len(matches) != 1:
            raise ValueError(f"Newton has {len(matches)} joints named {name!r}; it must name exactly one.")
        return matches[0]

    def _body_index(self, name: str) -> int:
        matches = [i for i, label in enumerate(self._body_labels) if label.split("/")[-1] == name]
        if len(matches) != 1:
            raise ValueError(f"Newton has {len(matches)} bodies named {name!r}; it must name exactly one.")
        return matches[0]

    def _sync(self) -> None:
        self._joint_q = self._wp.to_torch(self.state_0.joint_q)
        self._joint_qd = self._wp.to_torch(self.state_0.joint_qd)

    def describe(self) -> dict:
        wp, model = self._wp, self.model
        mass = wp.to_torch(model.body_mass).cpu().numpy()
        lower = wp.to_torch(model.joint_limit_lower).cpu().numpy()
        upper = wp.to_torch(model.joint_limit_upper).cpu().numpy()
        springref = getattr(getattr(model, "mujoco", None), "dof_springref", None)
        stiffness = getattr(getattr(model, "mujoco", None), "dof_passive_stiffness", None)
        return {
            "joints": [label.split("/")[-1] for label in self._labels],
            "tong_arm_masses": [float(mass[self._body_index(n)]) for n in ("tong_base", "tong_jaw")],
            "total_mass": round(float(mass.sum()), 6),
            "stiffness": None if stiffness is None else float(wp.to_torch(stiffness).cpu().numpy()[self._dadr]),
            "damping": float(wp.to_torch(model.joint_damping).cpu().numpy()[self._dadr]),
            "armature": float(wp.to_torch(model.joint_armature).cpu().numpy()[self._dadr]),
            "range": [float(lower[self._dadr]), float(upper[self._dadr])],
            "spring_rest": None if springref is None else float(wp.to_torch(springref).cpu().numpy()[self._dadr]),
        }

    def _forward(self) -> None:
        self._newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

    def set_arm(self, angles: np.ndarray) -> None:
        for adr, value in zip(self._arm_qadr, angles, strict=True):
            self._joint_q[adr] = float(value)
        self._forward()

    def body_poses(self) -> np.ndarray:
        # Newton carries body transforms as (px, py, pz, qx, qy, qz, qw);
        # this protocol hands out wxyz, as the other two backends do.
        transforms = self._wp.to_torch(self.state_0.body_q).cpu().numpy()
        origin = self._root_origin(transforms)
        out = []
        for name in TRACKED_BODIES:
            t = transforms[self._body_index(name)]
            out.append(np.concatenate([t[:3] - origin, [t[6], t[3], t[4], t[5]]]))
        return np.asarray(out)

    def _root_origin(self, transforms: np.ndarray) -> np.ndarray:
        # The arm's base body is welded to the world and may have been
        # folded away by the importer, in which case the world origin is
        # the arm's origin.
        try:
            return transforms[self._body_index(ROOT_BODY)][:3]
        except ValueError:
            return np.zeros(3)

    def set_hinge(self, q: float) -> None:
        self._joint_qd[:] = 0.0
        self._joint_q[self._qadr] = q
        self._forward()

    def set_torque(self, tau: float) -> None:
        self._joint_f[:] = 0.0
        self._joint_f[self._dadr] = tau

    def step(self) -> None:
        self.solver.step(self.state_0, self.state_1, self.control, None, DT)
        self.state_0, self.state_1 = self.state_1, self.state_0
        self._sync()

    def hinge(self) -> float:
        return float(self._joint_q[self._qadr].item())

    def cube_pos(self) -> np.ndarray:
        transforms = self._wp.to_torch(self.state_0.body_q).cpu().numpy()
        return transforms[self._body_index("cube")][:3].copy()


class GenesisArm:
    name = "genesis"

    def __init__(self, xml: str, gravity: bool):
        import genesis as gs

        self._gs = gs
        if not gs._initialized:
            gs.init(logging_level="warning")
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=DT, gravity=(0.0, 0.0, -9.81 if gravity else 0.0)),
            # Only the contact settings are overridden; everything else is
            # left at the Genesis default, because the spring passes
            # already agree to the last digit on the defaults and an
            # integrator change would put that at risk for no reason.
            rigid_options=gs.options.RigidOptions(dt=DT, friction_cone=gs.friction_cone.pyramidal, impratio=IMPRATIO),
            show_viewer=False,
        )
        # default_armature=None for the reason spelled out in the Genesis
        # scene manager: Genesis otherwise adds 0.1 kg*m^2 to every joint
        # whose armature the file leaves out, which on this hinge is seven
        # hundred times the real thing.
        self.entity = self.scene.add_entity(gs.morphs.MJCF(file=xml, convexify=False, default_armature=None))
        self.scene.build(n_envs=1)

        self._hinge = list(self.entity.get_joint(HINGE_JOINT).dofs_idx_local)
        if len(self._hinge) != 1:
            raise ValueError(f"Genesis gives {HINGE_JOINT!r} {len(self._hinge)} DOFs; a hinge has exactly one.")
        self._arm_dofs = [list(self.entity.get_joint(n).dofs_idx_local)[0] for n in ARM_JOINTS if self._has_joint(n)]
        # Genesis keeps a per-DOF PD alongside the passive spring; zeroed
        # for the same reason as Newton's above.
        self.entity.set_dofs_kp([0.0] * self.entity.n_dofs, list(range(self.entity.n_dofs)))
        self.entity.set_dofs_kv([0.0] * self.entity.n_dofs, list(range(self.entity.n_dofs)))
        self._links = {link.name.split("/")[-1]: i for i, link in enumerate(self.entity.links)}

    def _has_joint(self, name: str) -> bool:
        try:
            return self.entity.get_joint(name) is not None
        except Exception:
            return False

    def _device(self):
        return self.entity.get_dofs_position(self._hinge).device

    def describe(self) -> dict:
        entity = self.entity
        limits = np.concatenate([_np(v).reshape(-1) for v in entity.get_dofs_limit(self._hinge)])
        return {
            "joints": [j.name.split("/")[-1] for j in entity.joints if j.n_dofs > 0],
            "tong_arm_masses": [float(entity.links[self._links[n]].inertial_mass) for n in ("tong_base", "tong_jaw")],
            "total_mass": round(float(sum(link.inertial_mass for link in entity.links)), 6),
            "stiffness": _scalar(entity.get_dofs_stiffness(self._hinge)),
            "damping": _scalar(entity.get_dofs_damping(self._hinge)),
            "armature": _scalar(entity.get_dofs_armature(self._hinge)),
            "range": [float(limits[0]), float(limits[1])],
            "spring_rest": 0.0,  # not a concept in Genesis; always zero
        }

    def set_arm(self, angles: np.ndarray) -> None:
        import torch

        self.entity.set_dofs_position(
            torch.tensor([[float(v) for v in angles]], dtype=torch.float32, device=self._device()),
            self._arm_dofs,
            zero_velocity=True,
        )

    def body_poses(self) -> np.ndarray:
        positions = _np(self.entity.get_links_pos()).reshape(len(self.entity.links), 3)
        quats = _np(self.entity.get_links_quat()).reshape(len(self.entity.links), 4)
        origin = positions[self._links[ROOT_BODY]] if ROOT_BODY in self._links else np.zeros(3)
        return np.asarray(
            [np.concatenate([positions[self._links[n]] - origin, quats[self._links[n]]]) for n in TRACKED_BODIES]
        )

    def set_hinge(self, q: float) -> None:
        import torch

        self.entity.set_dofs_position(
            torch.tensor([[q]], dtype=torch.float32, device=self._device()), self._hinge, zero_velocity=True
        )

    def set_torque(self, tau: float) -> None:
        import torch

        self.entity.control_dofs_force(torch.tensor([[tau]], dtype=torch.float32, device=self._device()), self._hinge)

    def step(self) -> None:
        self.scene.step()

    def hinge(self) -> float:
        return _scalar(self.entity.get_dofs_position(self._hinge))

    def cube_pos(self) -> np.ndarray:
        positions = _np(self.entity.get_links_pos()).reshape(len(self.entity.links), 3)
        return positions[self._links["cube"]].copy()


BACKENDS = {"mujoco": MujocoArm, "newton": NewtonArm, "genesis": GenesisArm}


def _np(value) -> np.ndarray:
    return np.asarray(value.detach().cpu() if hasattr(value, "detach") else value, dtype=np.float64)


def _scalar(value) -> float:
    return float(_np(value).reshape(-1)[0])


# ══════════════════════════════════════════════════════════════════
# Passes
# ══════════════════════════════════════════════════════════════════


def arm_poses(count: int) -> np.ndarray:
    """Arm configurations to compare kinematics at.

    Drawn from a fixed seed rather than chosen: a hand-picked pose is a
    pose somebody already believed in, and the failure this pass exists
    to catch — a quaternion convention that happens to agree at zero — is
    exactly the one a tidy pose hides. Home is included as the first row
    because it is the one every other diag quotes.
    """
    home = np.array([0.0, 1.047, 1.05, -0.9, 0.0, 0.0])
    rng = np.random.default_rng(0)
    return np.vstack([home, rng.uniform(-1.2, 1.2, size=(count - 1, len(ARM_JOINTS)))])


def run_kinematics(cls, xml: str, reading: Reading, poses: np.ndarray) -> None:
    backend = cls(xml, gravity=False)
    reading.parsed = backend.describe()
    out = []
    for angles in poses:
        backend.set_arm(angles)
        out.append(backend.body_poses())
    reading.poses = np.asarray(out)


def run_spring(cls, xml: str, reading: Reading) -> None:
    backend = cls(xml, gravity=False)
    for tau in TORQUE_STAIRCASE:
        backend.set_hinge(OPEN_ANGLE)
        backend.set_torque(tau)
        for _ in range(SETTLE_STEPS):
            backend.step()
        reading.settled[tau] = backend.hinge()
        backend.set_torque(0.0)
        for _ in range(SETTLE_STEPS):
            backend.step()
        reading.reopened[tau] = backend.hinge()


def run_grip(cls, xml: str, reading: Reading) -> None:
    backend = cls(xml, gravity=True)
    backend.set_hinge(CUBE_ANGLE)
    backend.set_torque(GRIP_TORQUE)
    for _ in range(GRIP_STEPS):
        backend.step()
    reading.held = backend.cube_pos()
    # Let go. Without this the pass cannot fail: a cube that was never
    # gripped and a cube that was would both simply be reported.
    backend.set_torque(0.0)
    for _ in range(GRIP_STEPS):
        backend.step()
    reading.dropped = backend.cube_pos()


# ══════════════════════════════════════════════════════════════════
# Reporting
# ══════════════════════════════════════════════════════════════════


def _fmt(value) -> str:
    if value is None:
        return "MISSING"
    if isinstance(value, list | tuple):
        if value and isinstance(value[0], str):
            return ",".join(value)
        return "[" + ", ".join(f"{float(v):.6g}" for v in value) + "]"
    return f"{float(value):.6g}"


def report_parse(readings: list[Reading]) -> bool:
    print("\n" + "=" * 78)
    print("  PASS 0 - what each backend parsed out of the tong-armed asset")
    print("=" * 78)
    keys = sorted({k for r in readings for k in r.parsed})
    width = max(len(k) for k in keys)
    for key in keys:
        print(f"  {key}")
        for reading in readings:
            print(f"      {reading.name:>8}  {_fmt(reading.parsed.get(key))}")
    del width
    print()
    ok = True
    for key, want in (
        ("stiffness", STIFFNESS),
        ("damping", DAMPING),
        ("spring_rest", OPEN_ANGLE),
        ("armature", 0.0),
    ):
        for reading in readings:
            got = reading.parsed.get(key)
            if got is None:
                print(f"  FAIL  {reading.name}: hinge {key} never arrived (asset says {want})")
                ok = False
            elif abs(float(got) - want) > 1e-6:
                print(f"  FAIL  {reading.name}: hinge {key} = {float(got):.6g}, asset says {want}")
                ok = False
    for reading in readings:
        joints = [j for j in reading.parsed["joints"] if j in (*ARM_JOINTS, HINGE_JOINT)]
        if joints != [*ARM_JOINTS, HINGE_JOINT]:
            print(f"  FAIL  {reading.name}: joints in order {joints}, expected {[*ARM_JOINTS, HINGE_JOINT]}")
            ok = False
        masses = reading.parsed["tong_arm_masses"]
        if any(abs(float(m) - ARM_PIECE_MASS) > 1e-6 for m in masses):
            print(f"  FAIL  {reading.name}: tong arm masses {_fmt(masses)}, asset says {ARM_PIECE_MASS} each")
            ok = False
        limits = reading.parsed["range"]
        if abs(float(limits[1]) - CLOSED_ANGLE) > 1e-4:
            print(f"  FAIL  {reading.name}: closed stop {float(limits[1]):.6g}, asset says {CLOSED_ANGLE}")
            ok = False
    print("  PASS 0: " + ("OK" if ok else "FAILED"))
    return ok


def report_kinematics(readings: list[Reading], poses: np.ndarray) -> bool:
    print("\n" + "=" * 78)
    print("  PASS 1 - imposed arm poses, link poses read back without stepping")
    print("=" * 78)
    print(f"  {len(poses)} poses x {len(TRACKED_BODIES)} bodies, in the arm's own frame")
    reference = readings[0]
    print(f"\n  {reference.name} at home:")
    for i, name in enumerate(TRACKED_BODIES):
        pose = reference.poses[0][i]
        print(
            f"    {name:<12} pos {np.array2string(pose[:3], precision=4)}  quat {np.array2string(pose[3:], precision=4)}"
        )
    print()
    ok = True
    for i, left in enumerate(readings):
        for right in readings[i + 1 :]:
            worst_pos, worst_ang, where = 0.0, 0.0, ""
            for p in range(len(poses)):
                for b, name in enumerate(TRACKED_BODIES):
                    dp = float(np.linalg.norm(left.poses[p][b][:3] - right.poses[p][b][:3]))
                    da = _quat_angle(left.poses[p][b][3:], right.poses[p][b][3:])
                    if dp > worst_pos or da > worst_ang:
                        where = f"pose {p}, {name}"
                    worst_pos, worst_ang = max(worst_pos, dp), max(worst_ang, da)
            verdict = "ok" if (worst_pos <= TOL_POSE and worst_ang <= TOL_POSE) else "FAIL"
            print(
                f"  {left.name} vs {right.name}: worst position {worst_pos:.7f} m, "
                f"orientation {worst_ang:.7f} rad  ({where})  [{verdict}]"
            )
            ok &= verdict == "ok"
    print("  PASS 1: " + ("OK" if ok else "FAILED"))
    return ok


def report_spring(readings: list[Reading]) -> bool:
    print("\n" + "=" * 78)
    print("  PASS 2 - the spring, with the tong mounted and the arm welded")
    print("=" * 78)
    print(f"  {'torque':>8} {'want q':>9}" + "".join(f"{r.name:>12}" for r in readings))
    ok = True
    for tau in TORQUE_STAIRCASE:
        want = tau / STIFFNESS
        print(f"  {tau:>8.3f} {want:>9.5f}" + "".join(f"{r.settled[tau]:>12.6f}" for r in readings))
        for reading in readings:
            if abs(reading.settled[tau] - want) > TOL_SETTLED:
                print(
                    f"    FAIL  {reading.name}: settled {reading.settled[tau]:.6f}, "
                    f"the spring says {want:.6f} (tol {TOL_SETTLED})"
                )
                ok = False
    print("\n  released again - back to fully open?")
    for reading in readings:
        worst = max(abs(v - OPEN_ANGLE) for v in reading.reopened.values())
        verdict = "ok" if worst <= TOL_REOPEN else "FAIL"
        print(f"    {reading.name:>8}: worst residual {worst:.6f} rad  [{verdict}]")
        ok &= verdict == "ok"
    print("  PASS 2: " + ("OK" if ok else "FAILED"))
    return ok


def report_grip(readings: list[Reading]) -> bool:
    print("\n" + "=" * 78)
    print("  PASS 3 - a 25 mm cube held by friction, then let go")
    print("=" * 78)
    start = np.asarray(CUBE_AT)
    print(f"  cube starts at {np.array2string(start, precision=4)}, tong held shut with {GRIP_TORQUE} N*m")
    print("  the jaws close along +-y and the arm points along +x, so nothing but friction holds it up")
    print()
    ok = True
    for reading in readings:
        drop = float(start[2] - reading.held[2])
        fall = float(reading.held[2] - reading.dropped[2])
        held_ok = abs(drop) <= HELD_DROP
        fell_ok = fall >= DROPPED_FALL
        print(
            f"  {reading.name:>8}: held at {np.array2string(reading.held, precision=4)}  "
            f"(slipped {1000 * drop:+.2f} mm)  [{'ok' if held_ok else 'FAIL'}]"
        )
        print(
            f"  {'':>8}  released -> {np.array2string(reading.dropped, precision=4)}  "
            f"(fell {1000 * fall:.0f} mm)  [{'ok' if fell_ok else 'FAIL'}]"
        )
        if not held_ok:
            print(f"    FAIL  {reading.name}: the cube left the pads while the tong was shut on it")
        if not fell_ok:
            print(f"    FAIL  {reading.name}: the cube did not fall when released, so nothing here was measured")
        ok &= held_ok and fell_ok
    for i, left in enumerate(readings):
        for right in readings[i + 1 :]:
            diff = float(np.linalg.norm(left.held - right.held))
            verdict = "ok" if diff <= TOL_GRIP_POS else "FAIL"
            print(f"  {left.name} vs {right.name}, held cube position: {diff:.6f} m  [{verdict}]")
            ok &= verdict == "ok"
    print("  PASS 3: " + ("OK" if ok else "FAILED"))
    return ok


# ══════════════════════════════════════════════════════════════════


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", nargs="+", default=list(BACKENDS), choices=list(BACKENDS))
    ap.add_argument("--poses", type=int, default=8)
    args = ap.parse_args()

    full = full_asset()
    welded = welded_asset(with_cube=False)
    with_cube = welded_asset(with_cube=True)
    poses = arm_poses(args.poses)

    print("=" * 78)
    print("  YAM + SPRING TONG: the same arm, spring and grip on every backend?")
    print("=" * 78)
    print(f"  asset    {ARM_XML}")
    print(f"  derived  {welded}  (arm joints deleted)")
    print(f"  derived  {with_cube}  (+ a {1000 * CUBE_SIZE:.0f} mm, {1000 * CUBE_MASS:.0f} g cube)")
    print(f"  dt       {DT} s, no decimation")

    readings: list[Reading] = []
    for name in args.sims:
        print(f"\n  running {name} ...")
        cls = BACKENDS[name]
        reading = Reading(name=name)
        run_kinematics(cls, full, reading, poses)
        run_spring(cls, welded, reading)
        run_grip(cls, with_cube, reading)
        readings.append(reading)

    results = [
        report_parse(readings),
        report_kinematics(readings, poses),
        report_spring(readings),
        report_grip(readings),
    ]

    print("\n" + "=" * 78)
    print("  OVERALL: " + ("PASS" if all(results) else "FAIL"))
    print("=" * 78)
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
