"""Is the spring tong the SAME tong on mjlab, Newton and Genesis?

The tong (``assets/props/tong.xml``) is held open by a torsion spring and
closes only while something pushes it shut. All of that behaviour is
carried by two numbers on one joint — ``stiffness`` and ``damping`` — and
nothing downstream announces when a backend drops one, adds its own PD on
top, or rests the spring at a different angle. The tong would still open
and close. It would open and close DIFFERENTLY, and a policy trained on
one backend would squeeze with the wrong force on another.

So each number is measured rather than trusted, in five passes, each with
its own verdict, and each checked against physics as well as against the
other backends — three backends agreeing is worthless if they agree on
the wrong answer:

  0. **Parse.** Mass, inertia, joint range, stiffness, damping and the
     spring's rest angle, as each backend actually stored them. A number
     that never arrived is caught here rather than four passes later.
  1. **Let go.** Held closed at 0.20 rad and released. Does it spring
     open on its own, and at the same rate everywhere? The swing down to
     the open stop is compared against the exact damped-oscillator
     solution and against the other backends; how far it then sinks into
     that stop is reported and compared separately, because a joint limit
     is a contact constraint rather than the spring and each solver
     softens it differently.
  2. **Ring down.** The same swing, but about a partly-closed
     equilibrium so it never touches either stop and the comparison is
     pure spring. The period and the decay between peaks are inverted
     back into a stiffness and a damping, which are then checked against
     the asset — this is what would catch all three backends sharing one
     wrong damping.
  3. **Squeeze a little, let go, repeat.** A staircase of closing
     torques, each held until settled and then released. The settled
     angle must be the torque over the stiffness; the released angle must
     be fully open again. Also reports the pad gap in millimetres, which
     is the number that decides whether the tong can hold a cube.
  4. **Gravity.** Passes 1-3 are weightless so the spring is measured
     alone. In use the arm's own weight closes it a little, and all three
     must agree how much — checked against the static balance.

The base is fixed for every pass. A tong in free fall feels no gravity
torque at all, and its two arms recoil against each other instead of one
swinging, so a floating measurement answers a different question than the
one that matters: the tong is always held by something. The fixed variant
is derived from the one authored asset at run time by dropping its free
joint, so the two cannot drift apart.

Run::

    jaxpy -m jaxrlworld.scripts.diag.yam.tong_spring_cross_sim_diag
    jaxpy -m jaxrlworld.scripts.diag.yam.tong_spring_cross_sim_diag --sims mujoco newton
"""

from __future__ import annotations

import argparse
import math
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

TONG_XML = Path("./JaxRLWorld/jaxrlworld/assets/props/tong.xml")
"""Relative to the SimForge root, like every other asset path here."""

HINGE_JOINT = "tong_hinge"

DT = 0.002
"""Seconds. One physics step, no decimation — this compares solvers, so
nothing is allowed to average over their differences."""

# ── What the asset says ───────────────────────────────────────────
#
# READ OUT OF THE XML, not copied from it. A backend that lost one of
# these is still caught against the asset rather than against another
# backend that might have lost the same one — which is the whole point of
# an independent reference — but the reference now cannot drift out of
# date, and an earlier copy of it did: the tool was made thicker and
# heavier and every number below stayed at the old tool's, so all three
# backends were reported as wrong for agreeing with the asset they had
# been given.
#
# Parsed with ElementTree, which is neither of the three physics engines
# and so is not an opinion any of them can share.


def _asset_facts(path: Path) -> dict:
    """The spring, the jaws and the pads, from the XML itself."""
    root = ElementTree.parse(path).getroot()

    def find(tag: str, name: str):
        for element in root.iter(tag):
            if element.get("name") == name:
                return element
        raise ValueError(f"{path}: no <{tag}> named {name!r}")

    hinge = find("joint", HINGE_JOINT)
    lo, hi = (float(v) for v in hinge.get("range").split())
    inertial = find("body", "tong_base").find("inertial")
    diag = [float(v) for v in inertial.get("diaginertia").split()]
    pad = find("geom", "tong_base_pad")
    jaw_quat = [float(v) for v in find("body", "tong_jaw").get("quat").split()]
    return {
        "stiffness": float(hinge.get("stiffness")),
        "damping": float(hinge.get("damping")),
        "open": lo,
        "closed": hi,
        "arm_mass": float(inertial.get("mass")),
        "arm_com": float(inertial.get("pos").split()[0]),
        # About the hinge's own axis, which is z. The other two turn the
        # jaw about axes the joint does not move it on.
        "arm_inertia_com": diag[2],
        "pad_reach": float(pad.get("pos").split()[0]),
        "pad_half": float(pad.get("size").split()[1]),
        # A rotation about -z, so the angle is what the w component says.
        "jaw_mount": 2.0 * math.acos(jaw_quat[0]),
    }


_ASSET = _asset_facts(TONG_XML)

STIFFNESS = _ASSET["stiffness"]  # N*m/rad
DAMPING = _ASSET["damping"]  # N*m*s/rad
ARM_MASS = _ASSET["arm_mass"]  # kg, each jaw
ARM_COM = _ASSET["arm_com"]  # m, pivot to a jaw's centre of mass
ARM_INERTIA_COM = _ASSET["arm_inertia_com"]  # kg*m^2, about the jaw's own centre of mass
OPEN_ANGLE = 0.0  # rad — the spring's rest position AND the open stop
CLOSED_ANGLE = _ASSET["closed"]  # rad, pads touching
JAW_MOUNT_ANGLE = _ASSET["jaw_mount"]  # rad, the jaw body's built-in rotation
PAD_REACH = _ASSET["pad_reach"]  # m, pivot to pad centre
PAD_HALF = _ASSET["pad_half"]  # m, pad half-thickness

RELEASE_FROM = 0.20
RINGDOWN_TORQUE = 0.25 * STIFFNESS * CLOSED_ANGLE
RINGDOWN_FROM = 0.50 * CLOSED_ANGLE
"""Rings about a quarter of the travel with an amplitude of a quarter, so
it stays clear of the open stop at 0 and the closed stop at the other
end. Touching either would turn a measurement of the spring into a
measurement of the stop. Scaled off the spring rather than stated, so a
stiffer tool still rings in the middle of its own range instead of
sitting on a stop."""

TORQUE_STAIRCASE = tuple(
    round(fraction * STIFFNESS * CLOSED_ANGLE, 6) for fraction in (0.125, 0.25, 0.375, 0.5, 0.625, 0.75)
)
"""N*m. Over the stiffness these are an eighth to three quarters of the
travel: from barely moving to nearly shut. Derived from the spring, so
the staircase still spans the tool's own range when the tool changes."""

SETTLE_STEPS = 600
"""1.2 s, about eleven undamped periods of this spring on this inertia,
so a reading is a settled value and not a phase."""

# ── Tolerances ────────────────────────────────────────────────────
#
# Absolute angles, because that is what a reader can judge: 1e-3 rad at
# the pivot is 0.11 mm at the pads, far below anything a grasp depends
# on. Two solvers integrating different formulations never agree bit for
# bit; they have to agree PHYSICALLY.
TOL_SETTLED = 2e-3
"""rad. A steady state under a held torque, with no integrator difference
left in it. The strictest test, and the one a wrong stiffness cannot
survive."""

TOL_TRAJECTORY = 1.5e-2
"""rad, worst point of a whole swing. Loose on purpose: dominated by the
first few steps, where integrators differ most and the physical answer
matters least."""

TOL_REOPEN = 1e-3
"""rad. How close to fully open it must return once released. Above this
is a spring that does not recover — a tong that gradually stops
working."""

TOL_STOP_PENETRATION = 0.05
"""rad. How differently the backends may sink INTO the open stop before
bouncing back off it.

Separate from the trajectory bound, and deliberately so: this is not the
spring. A joint limit is a contact constraint, each solver softens it its
own way (MuJoCo's ``solreflimit``, which the asset does not author, so all
three use their own default), and the tong arrives at the stop with real
momentum. Newton reproduces MuJoCo exactly, because it IS mjwarp; Genesis
barely penetrates at all where the other two reach -0.028 rad, which is
3 mm of extra opening at the pads for about 60 ms.

Judged apart rather than folded into one number, because a reader who
sees the release trace diverge deserves to know WHICH of the two things
it measures diverged. The part that decides whether the tong works — the
rate it opens at, and the angle it comes to rest at — is checked
strictly, above and below."""

TOL_STIFFNESS_REL = 0.08
TOL_DAMPING_REL = 0.15
"""Fractions, and set by the TIMESTEP rather than by sloppiness. With
``omega_n * dt`` about 0.12 here, an integrator shifts the damped period
by roughly 2%, and stiffness goes as the period squared, so ~4% is the
floor for any solver at 2 ms — MuJoCo lands there, and measuring the same
swing at 0.2 ms brings it back under 0.5%. Widened once more for the
decay ratio, which is the noisier of the two by construction.

Loose enough to survive discretisation, tight enough for the failures
worth catching: a stiffness wrong by a factor, or a damping that was
dropped on the way in and reads as zero."""


@dataclass
class Reading:
    """One backend's answers, kept together so they compare as a set."""

    name: str
    parsed: dict = field(default_factory=dict)
    release: np.ndarray | None = None
    ringdown: np.ndarray | None = None
    settled: dict = field(default_factory=dict)
    reopened: dict = field(default_factory=dict)
    gap_mm: dict = field(default_factory=dict)
    gravity_sag: float | None = None


# ══════════════════════════════════════════════════════════════════
# The asset, and the physics it describes
# ══════════════════════════════════════════════════════════════════


def fixed_base_asset() -> str:
    """The authored tong with its free joint removed, as a file path.

    Derived rather than authored twice. A second checked-in XML would be
    a copy that silently stops matching the first one the day somebody
    changes a mass, and every number here would then be measuring two
    different tongs. The tong carries no meshes, so a file written
    anywhere loads on all three backends — and it has to be a file,
    because Genesis takes a path and not XML text.
    """
    text = TONG_XML.read_text()
    stripped = re.sub(r"[ \t]*<freejoint[^>]*/>\n?", "", text)
    if stripped == text:
        raise ValueError(f"{TONG_XML} has no <freejoint/> to remove; the asset changed shape under this diag.")
    # Stood on edge, by a quarter turn about x, so the hinge axis is
    # HORIZONTAL. The tong is authored lying flat, where its hinge axis
    # is vertical and gravity has no moment about it — pass 4 would then
    # be measuring that a zero equals a zero. Every pass below assumes
    # this orientation, and it is the one the analytic references are
    # written for.
    stood, count = re.subn(
        r'(<body name="tong_base" pos="0 0 0")',
        r'\1 quat="0.7071068 -0.7071068 0 0"',
        stripped,
        count=1,
    )
    if count != 1:
        raise ValueError(f"{TONG_XML} no longer opens tong_base at the origin; this diag cannot stand it on edge.")
    stripped = stood
    out = Path(tempfile.mkdtemp(prefix="tong_diag_")) / "tong_fixed.xml"
    out.write_text(stripped)
    return str(out)


def hinge_inertia() -> float:
    """The jaw's inertia about the PIVOT, which is what the spring turns.

    A backend reports inertia about a body's centre of mass; the hinge is
    60 mm away, and the parallel-axis term is three times the central
    one. Comparing against the central value would predict a swing twice
    too fast and make all three backends look equally wrong together —
    the one failure an analytic reference exists to prevent.
    """
    return ARM_INERTIA_COM + ARM_MASS * ARM_COM**2


def analytic_swing(theta0: float, steps: int) -> np.ndarray:
    """The damped oscillator the asset describes, evaluated exactly.

    Weightless, base fixed, spring linear, stops untouched: this is not
    an approximation of the simulated system, it IS the simulated system.
    Any departure belongs to a solver, not to the model.
    """
    inertia = hinge_inertia()
    omega_n = math.sqrt(STIFFNESS / inertia)
    zeta = DAMPING / (2.0 * math.sqrt(STIFFNESS * inertia))
    if zeta >= 1.0:
        raise NotImplementedError(f"Overdamped (zeta={zeta:.3f}); this reference covers the underdamped case only.")
    omega_d = omega_n * math.sqrt(1.0 - zeta * zeta)
    t = np.arange(steps) * DT
    decay = np.exp(-zeta * omega_n * t)
    return theta0 * decay * (np.cos(omega_d * t) + (zeta / math.sqrt(1 - zeta * zeta)) * np.sin(omega_d * t))


def pad_gap(q: float) -> float:
    """Pad separation in metres, from the asset's geometry.

    Computed rather than read out of a backend: only MuJoCo exposes geom
    world positions cheaply, and a gap measured a different way on each
    backend would not be a comparison.
    """
    beta = JAW_MOUNT_ANGLE - q
    return PAD_REACH * math.sin(beta) - 2 * PAD_HALF * math.cos(beta) - 2 * PAD_HALF


def gravity_balance(q: float) -> tuple[float, float]:
    """The two sides of the static balance at a sagged angle.

    The spring holds ``k*q``; the jaw's own weight pulls with its
    horizontal lever arm. At rest they are equal.
    """
    return STIFFNESS * q, ARM_MASS * 9.81 * ARM_COM * math.cos(JAW_MOUNT_ANGLE - q)


# ══════════════════════════════════════════════════════════════════
# Backends
# ══════════════════════════════════════════════════════════════════


class MujocoTong:
    """The asset's native engine, and therefore the reference."""

    name = "mujoco"

    def __init__(self, xml: str, gravity: bool):
        import mujoco

        self._mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(xml)
        self.model.opt.timestep = DT
        self.model.opt.gravity[:] = (0.0, 0.0, -9.81 if gravity else 0.0)
        self.data = mujoco.MjData(self.model)
        self._jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, HINGE_JOINT)
        self._qadr = int(self.model.jnt_qposadr[self._jid])
        self._dadr = int(self.model.jnt_dofadr[self._jid])
        self._torque = 0.0

    def describe(self) -> dict:
        m = self.model
        return {
            "dof_count": int(m.nv),
            "arm_mass": [float(m.body_mass[1]), float(m.body_mass[2])],
            "arm_inertia_com_yy": [float(m.body_inertia[1][1]), float(m.body_inertia[2][1])],
            "stiffness": float(m.jnt_stiffness[self._jid]),
            "damping": float(m.dof_damping[self._dadr]),
            "range": [float(m.jnt_range[self._jid][0]), float(m.jnt_range[self._jid][1])],
            "spring_rest": float(m.qpos_spring[self._qadr]),
            "armature": float(m.dof_armature[self._dadr]),
        }

    def set_hinge(self, q: float) -> None:
        self.data.qpos[:] = 0.0
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


class NewtonTong:
    """Newton, from the same MJCF, through its MuJoCo-Warp solver."""

    name = "newton"

    def __init__(self, xml: str, gravity: bool):
        import newton
        import warp as wp

        from jaxrlworld.rl.envs.utils.warp_logging import configure_warp_logging

        # Warp logs one line per kernel module it loads, ~45 of them, which
        # buries the report this diag exists to print. Silenced through the
        # framework's own switch rather than a private one, so
        # JAXRLWORLD_BUILD_SUMMARY=1 brings them back here exactly as it
        # does in a training run. Imported inside the backend, like newton
        # and warp themselves, so a mujoco-only or genesis-only run needs
        # neither installed.
        configure_warp_logging()

        self._newton = newton
        self._wp = wp

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81 if gravity else 0.0))
        builder.add_mjcf(xml, floating=False, collapse_fixed_joints=False, parse_sites=False)
        # Newton ALWAYS sums its own joint PD into the applied force —
        # unlike Genesis, where calling a force setter switches the mode.
        # Left at their import defaults, those gains pull the hinge
        # towards a target of zero, which is indistinguishable from a
        # spring that is simply too stiff. Zeroed so the only restoring
        # torque is the authored one; both gains are reported in pass 0,
        # so a future Newton that ignores this is caught rather than
        # quietly believed.
        for i in range(len(builder.joint_target_ke)):
            builder.joint_target_ke[i] = 0.0
            builder.joint_target_kd[i] = 0.0

        self.model = builder.finalize()
        self.solver = newton.solvers.SolverMuJoCo(self.model)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        labels = list(getattr(self.model, "joint_label", None) or self.model.joint_key)
        matches = [i for i, label in enumerate(labels) if label.endswith(HINGE_JOINT)]
        if len(matches) != 1:
            raise ValueError(f"Newton labels {matches} match {HINGE_JOINT!r}; it must name exactly one joint.")
        self._qadr = int(wp.to_torch(self.model.joint_q_start).cpu().numpy()[matches[0]])
        self._dadr = int(wp.to_torch(self.model.joint_qd_start).cpu().numpy()[matches[0]])
        self._joint_f = wp.to_torch(self.control.joint_f)
        self._sync()

    def _sync(self) -> None:
        self._joint_q = self._wp.to_torch(self.state_0.joint_q)
        self._joint_qd = self._wp.to_torch(self.state_0.joint_qd)

    def describe(self) -> dict:
        wp, model = self._wp, self.model
        body_mass = wp.to_torch(model.body_mass).cpu().numpy()
        body_inertia = wp.to_torch(model.body_inertia).cpu().numpy()
        lower = wp.to_torch(model.joint_limit_lower).cpu().numpy()
        upper = wp.to_torch(model.joint_limit_upper).cpu().numpy()
        # ``damping`` is a first-class Newton field: the MJCF importer
        # reads it into ``JointDofConfig.damping`` (import_mjcf.py:2026),
        # which finalizes into ``model.joint_damping``. ``stiffness`` and
        # ``springref`` have no Newton equivalent and travel instead as
        # MuJoCo custom attributes (solver_mujoco.py:1053-1078) on their
        # way to mjwarp's ``jnt_stiffness`` / ``qpos_spring``.
        out = {
            "dof_count": int(model.joint_dof_count),
            "arm_mass": [float(m) for m in body_mass[:2]],
            "arm_inertia_com_yy": [float(body_inertia[i][1][1]) for i in range(min(2, len(body_inertia)))],
            "range": [float(lower[self._dadr]), float(upper[self._dadr])],
            "damping": float(wp.to_torch(model.joint_damping).cpu().numpy()[self._dadr]),
            "armature": float(wp.to_torch(model.joint_armature).cpu().numpy()[self._dadr]),
            "internal_pd_ke": float(wp.to_torch(model.joint_target_ke).cpu().numpy()[self._dadr]),
            "internal_pd_kd": float(wp.to_torch(model.joint_target_kd).cpu().numpy()[self._dadr]),
        }
        for key, attribute in (("stiffness", "dof_passive_stiffness"), ("spring_rest", "dof_springref")):
            values = self._custom_dof_attribute(attribute)
            out[key] = None if values is None else float(values[self._dadr])
        return out

    def _custom_dof_attribute(self, attribute: str) -> np.ndarray | None:
        mujoco_attrs = getattr(self.model, "mujoco", None)
        if mujoco_attrs is None:
            return None
        array = getattr(mujoco_attrs, attribute, None)
        if array is None:
            return None
        return self._wp.to_torch(array).cpu().numpy()

    def set_hinge(self, q: float) -> None:
        self._joint_q[:] = 0.0
        self._joint_qd[:] = 0.0
        self._joint_q[self._qadr] = q
        self._newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

    def set_torque(self, tau: float) -> None:
        self._joint_f[:] = 0.0
        self._joint_f[self._dadr] = tau

    def step(self) -> None:
        self.solver.step(self.state_0, self.state_1, self.control, None, DT)
        self.state_0, self.state_1 = self.state_1, self.state_0
        self._sync()

    def hinge(self) -> float:
        return float(self._joint_q[self._qadr].item())


class GenesisTong:
    """Genesis — the backend whose spring cannot rest anywhere but zero."""

    name = "genesis"

    def __init__(self, xml: str, gravity: bool):
        import genesis as gs

        self._gs = gs
        if not gs._initialized:
            gs.init(logging_level="warning")

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=DT, gravity=(0.0, 0.0, -9.81 if gravity else 0.0)),
            show_viewer=False,
        )
        # default_armature=None for the reason the scene manager states at
        # length: Genesis otherwise adds 0.1 kg*m^2 to any joint whose
        # armature the file leaves out, which on this hinge is seven hundred
        # times the real thing. Restated rather than inherited because this
        # diag builds its own morph, and a diag that quietly differs from the
        # scene it is vouching for is worse than no diag.
        self.entity = self.scene.add_entity(gs.morphs.MJCF(file=xml, convexify=False, default_armature=None))
        self.scene.build(n_envs=1)

        dofs = list(self.entity.get_joint(HINGE_JOINT).dofs_idx_local)
        if len(dofs) != 1:
            raise ValueError(f"Genesis gives {HINGE_JOINT!r} {len(dofs)} DOFs; a hinge has exactly one.")
        self._dofs = dofs
        # Genesis keeps a per-DOF PD alongside the passive spring, and an
        # MJCF with no actuator leaves those gains wherever the import put
        # them. Zeroed for the same reason as Newton's above.
        self.entity.set_dofs_kp([0.0], self._dofs)
        self.entity.set_dofs_kv([0.0], self._dofs)

    def describe(self) -> dict:
        entity = self.entity
        limits = np.concatenate([_numpy(v).reshape(-1) for v in entity.get_dofs_limit(self._dofs)])
        return {
            "dof_count": int(entity.n_dofs),
            # By name, not by index. Genesis gives the entity a base link
            # of its own on top of the two the asset declares, with a mass
            # of one float32 epsilon and no inertia -- harmless to the
            # physics, and every dynamic pass here agrees with the other
            # two backends to the digit, but enough to make a list
            # comparison read as a mass mismatch.
            "arm_mass": [float(link.inertial_mass) for link in self._jaw_links()],
            "arm_inertia_com_yy": [float(np.asarray(link.inertial_i)[1][1]) for link in self._jaw_links()],
            "stiffness": _scalar(entity.get_dofs_stiffness(self._dofs)),
            "damping": _scalar(entity.get_dofs_damping(self._dofs)),
            "range": [float(limits[0]), float(limits[1])],
            # Not a field Genesis has: its passive joint force is
            # -stiffness * q (forward_dynamics.py:985), so the spring
            # always rests at zero. Reported as the constant it is, which
            # is exactly why the asset puts "open" there.
            "spring_rest": 0.0,
            "armature": _scalar(entity.get_dofs_armature(self._dofs)),
        }

    def _jaw_links(self) -> list:
        """The two links the asset declares, in the order it declares them."""
        wanted = ("tong_base", "tong_jaw")
        found = {}
        for link in self.entity.links:
            for name in wanted:
                if str(link.name).endswith(name) and name not in found:
                    found[name] = link
        if len(found) != len(wanted):
            raise ValueError(f"Genesis links {[str(l.name) for l in self.entity.links]} do not cover {wanted}.")
        return [found[name] for name in wanted]

    def _device(self):
        return self.entity.get_dofs_position(self._dofs).device

    def set_hinge(self, q: float) -> None:
        import torch

        self.entity.set_dofs_position(
            torch.tensor([[q]], dtype=torch.float32, device=self._device()), self._dofs, zero_velocity=True
        )

    def set_torque(self, tau: float) -> None:
        import torch

        self.entity.control_dofs_force(torch.tensor([[tau]], dtype=torch.float32, device=self._device()), self._dofs)

    def step(self) -> None:
        self.scene.step()

    def hinge(self) -> float:
        return _scalar(self.entity.get_dofs_position(self._dofs))


BACKENDS = {"mujoco": MujocoTong, "newton": NewtonTong, "genesis": GenesisTong}


def _numpy(value) -> np.ndarray:
    return np.asarray(value.detach().cpu() if hasattr(value, "detach") else value)


def _scalar(value) -> float:
    return float(_numpy(value).reshape(-1)[0])


# ══════════════════════════════════════════════════════════════════
# Measurement
# ══════════════════════════════════════════════════════════════════


def _trace(backend, steps: int) -> np.ndarray:
    out = np.empty(steps, dtype=np.float64)
    for i in range(steps):
        out[i] = backend.hinge()
        backend.step()
    return out


def run_backend(cls, xml: str, reading: Reading, release_steps: int) -> None:
    backend = cls(xml, gravity=False)
    reading.parsed = backend.describe()

    backend.set_torque(0.0)
    backend.set_hinge(RELEASE_FROM)
    reading.release = _trace(backend, release_steps)

    backend.set_hinge(RINGDOWN_FROM)
    backend.set_torque(RINGDOWN_TORQUE)
    reading.ringdown = _trace(backend, release_steps) - RINGDOWN_TORQUE / STIFFNESS

    for tau in TORQUE_STAIRCASE:
        backend.set_hinge(OPEN_ANGLE)
        backend.set_torque(tau)
        for _ in range(SETTLE_STEPS):
            backend.step()
        reading.settled[tau] = backend.hinge()
        reading.gap_mm[tau] = 1000.0 * pad_gap(backend.hinge())
        # Let go. This is the half a torque sweep alone never tests: a
        # tong that closes correctly and does not spring back is not a
        # tong.
        backend.set_torque(0.0)
        for _ in range(SETTLE_STEPS):
            backend.step()
        reading.reopened[tau] = backend.hinge()

    heavy = cls(xml, gravity=True)
    heavy.set_torque(0.0)
    heavy.set_hinge(OPEN_ANGLE)
    for _ in range(SETTLE_STEPS * 3):
        heavy.step()
    reading.gravity_sag = heavy.hinge()


def invert_ringdown(trace: np.ndarray) -> tuple[float, float, float] | None:
    """Recover (period, stiffness, damping) from a free ring-down.

    The period gives the inertia the spring is actually turning, and the
    ratio between successive peaks gives the fraction of critical damping
    — together they name both authored numbers without being told either.
    That is the point of this pass: a stiffness read back out of a model
    only proves the file was parsed, while this proves the solver USED it.

    Peaks are located to sub-step precision by fitting a parabola through
    each one and its neighbours. Without that, a peak is only known to
    the nearest step, which on a 60-step period is 1.6% of the period and
    3% of the stiffness — the same size as the discretisation effect this
    is trying to measure, so the reading would be mostly quantisation.
    """
    peaks: list[tuple[float, float]] = []
    for i in range(1, len(trace) - 1):
        if not (trace[i] > trace[i - 1] and trace[i] >= trace[i + 1] and trace[i] > 0):
            continue
        before, here, after = trace[i - 1], trace[i], trace[i + 1]
        curvature = before - 2 * here + after
        offset = 0.5 * (before - after) / curvature if curvature != 0 else 0.0
        peaks.append((i + offset, here - 0.25 * (before - after) * offset))
    if len(peaks) < 2:
        return None
    (first_at, first_height), (second_at, second_height) = peaks[0], peaks[1]
    if second_height <= 0 or first_height <= second_height:
        return None
    period = (second_at - first_at) * DT
    decrement = math.log(first_height / second_height)
    zeta = decrement / math.sqrt(4 * math.pi**2 + decrement**2)
    omega_n = (2 * math.pi / period) / math.sqrt(1 - zeta * zeta)
    stiffness = hinge_inertia() * omega_n**2
    damping = 2 * zeta * math.sqrt(stiffness * hinge_inertia())
    return period, stiffness, damping


def analytic_damped_period() -> float:
    """The period the asset's own numbers predict, for the report."""
    inertia = hinge_inertia()
    omega_n = math.sqrt(STIFFNESS / inertia)
    zeta = DAMPING / (2.0 * math.sqrt(STIFFNESS * inertia))
    return 2 * math.pi / (omega_n * math.sqrt(1 - zeta * zeta))


# ══════════════════════════════════════════════════════════════════
# Reporting
# ══════════════════════════════════════════════════════════════════


def _fmt(value) -> str:
    if value is None:
        return "MISSING"
    if isinstance(value, list | tuple):
        return "[" + ", ".join(f"{float(v):.6g}" for v in value) + "]"
    return f"{float(value):.6g}"


def _pairwise(readings, extract, tolerance: float, label: str, quiet: bool = False) -> bool:
    ok = True
    for i, left in enumerate(readings):
        for right in readings[i + 1 :]:
            diff = float(np.max(np.abs(extract(left) - extract(right))))
            verdict = "ok" if diff <= tolerance else "FAIL"
            if not quiet or verdict == "FAIL":
                print(f"  {left.name} vs {right.name}, {label}: max |diff| {diff:.6f} rad  [{verdict}]")
            ok &= verdict == "ok"
    return ok


def report_parse(readings: list[Reading]) -> bool:
    print("\n" + "=" * 78)
    print("  PASS 0 - what each backend actually parsed out of the asset")
    print("=" * 78)
    keys = sorted({k for r in readings for k in r.parsed})
    width = max(len(k) for k in keys)
    print(f"  {'field':<{width}}  " + "  ".join(f"{r.name:>22}" for r in readings))
    for key in keys:
        print(f"  {key:<{width}}  " + "  ".join(f"{_fmt(r.parsed.get(key)):>22}" for r in readings))
    print()
    ok = True
    for key, want in (("stiffness", STIFFNESS), ("damping", DAMPING), ("spring_rest", OPEN_ANGLE)):
        for reading in readings:
            got = reading.parsed.get(key)
            if got is None:
                print(f"  FAIL  {reading.name}: {key} never arrived (asset says {want})")
                ok = False
            elif abs(float(got) - want) > 1e-6:
                print(f"  FAIL  {reading.name}: {key} = {float(got):.6g}, asset says {want}")
                ok = False
    for reading in readings:
        if int(reading.parsed["dof_count"]) != 1:
            print(f"  FAIL  {reading.name}: {reading.parsed['dof_count']} DOFs, the fixed-base tong has exactly 1")
            ok = False
        masses = reading.parsed.get("arm_mass") or []
        if len(masses) != 2 or any(abs(float(m) - ARM_MASS) > 1e-6 for m in masses):
            print(f"  FAIL  {reading.name}: arm masses {_fmt(masses)}, asset says [{ARM_MASS}, {ARM_MASS}]")
            ok = False
        limits = reading.parsed.get("range") or []
        if len(limits) == 2 and abs(float(limits[1]) - CLOSED_ANGLE) > 1e-4:
            print(f"  FAIL  {reading.name}: closed stop {float(limits[1]):.6g}, asset says {CLOSED_ANGLE}")
            ok = False
    print("  PASS 0: " + ("OK" if ok else "FAILED"))
    return ok


def report_release(readings: list[Reading]) -> bool:
    print("\n" + "=" * 78)
    print(f"  PASS 1 - let go from {RELEASE_FROM} rad: does it spring open, and how fast")
    print("=" * 78)
    steps = min(len(r.release) for r in readings)
    reference = analytic_swing(RELEASE_FROM, steps)
    # The analytic swing overshoots through zero; the real tong cannot,
    # because zero IS its open stop. Compared only up to that point, so
    # this measures the spring and not the stop. The stop is still
    # compared across backends over the whole trace, below.
    free = int(np.argmax(reference <= 0.0)) or steps
    zeta = DAMPING / (2 * math.sqrt(STIFFNESS * hinge_inertia()))
    print(
        f"  jaw inertia about the pivot {hinge_inertia():.6g} kg*m^2, "
        f"undamped period {2 * math.pi * math.sqrt(hinge_inertia() / STIFFNESS):.4f} s, zeta {zeta:.3f}"
    )
    print(f"  analytic comparison runs to step {free}, where the open stop takes over")
    print()
    marks = [m for m in (0, 5, 10, 20, 40, 80, 160, 400, steps - 1) if m < steps]
    print("  " + "step".rjust(6) + "".join(f"{r.name:>12}" for r in readings) + f"{'analytic':>12}")
    for m in marks:
        row = "  " + f"{m:>6}" + "".join(f"{r.release[m]:>12.6f}" for r in readings)
        print(row + f"{reference[m]:>12.6f}")
    print()
    ok = True
    for reading in readings:
        drift = float(np.max(np.abs(reading.release[:free] - reference[:free])))
        verdict = "ok" if drift <= TOL_TRAJECTORY else "FAIL"
        print(f"  {reading.name:>8} vs analytic (up to the stop): max |diff| {drift:.6f} rad  [{verdict}]")
        ok &= verdict == "ok"
        ended = reading.release[-1]
        if abs(ended - OPEN_ANGLE) > TOL_REOPEN:
            print(f"  FAIL  {reading.name}: ended at {ended:.6f} rad, not open (tol {TOL_REOPEN})")
            ok = False
    ok &= _pairwise(readings, lambda r: r.release[:free], TOL_TRAJECTORY, "swing down to the stop")
    print()
    print("  how far each one sinks into the open stop before bouncing back")
    for reading in readings:
        deepest = float(np.min(reading.release))
        print(
            f"    {reading.name:>8}: {deepest:.6f} rad  ({1000 * (pad_gap(deepest) - pad_gap(0.0)):+.2f} mm at the pads)"
        )
    ok &= _pairwise(readings, lambda r: np.array([np.min(r.release)]), TOL_STOP_PENETRATION, "stop penetration")
    print("  PASS 1: " + ("OK" if ok else "FAILED"))
    return ok


def report_ringdown(readings: list[Reading]) -> bool:
    print("\n" + "=" * 78)
    print("  PASS 2 - ring down about a partly-closed rest, clear of both stops")
    print("=" * 78)
    print(
        f"  holding {RINGDOWN_TORQUE} N*m (rest at {RINGDOWN_TORQUE / STIFFNESS:.3f} rad), "
        f"displaced to {RINGDOWN_FROM} rad"
    )
    steps = min(len(r.ringdown) for r in readings)
    reference = analytic_swing(RINGDOWN_FROM - RINGDOWN_TORQUE / STIFFNESS, steps)
    ok = True
    print()
    print(f"  the asset predicts a damped period of {analytic_damped_period():.5f} s")
    print(f"  {'':>8}{'period':>10}{'implied k':>12}{'implied c':>12}     recovered from the swing alone")
    for reading in readings:
        inverted = invert_ringdown(reading.ringdown[:steps])
        if inverted is None:
            print(f"  {reading.name:>8}   no two clean peaks in the trace  [FAIL]")
            ok = False
            continue
        period, stiffness, damping = inverted
        k_off = abs(stiffness - STIFFNESS) / STIFFNESS
        c_off = abs(damping - DAMPING) / DAMPING
        verdict = "ok" if (k_off <= TOL_STIFFNESS_REL and c_off <= TOL_DAMPING_REL) else "FAIL"
        print(
            f"  {reading.name:>8}{period:>10.4f}{stiffness:>12.5f}{damping:>12.6f}"
            f"     k off {100 * k_off:4.1f}%, c off {100 * c_off:4.1f}%  [{verdict}]"
        )
        ok &= verdict == "ok"
    print(f"  (the asset says k = {STIFFNESS}, c = {DAMPING})")
    print()
    for reading in readings:
        drift = float(np.max(np.abs(reading.ringdown[:steps] - reference)))
        verdict = "ok" if drift <= TOL_TRAJECTORY else "FAIL"
        print(f"  {reading.name:>8} vs analytic: max |diff| {drift:.6f} rad  [{verdict}]")
        ok &= verdict == "ok"
    ok &= _pairwise(readings, lambda r: r.ringdown[:steps], TOL_TRAJECTORY, "ring-down")
    print("  PASS 2: " + ("OK" if ok else "FAILED"))
    return ok


def report_staircase(readings: list[Reading]) -> bool:
    print("\n" + "=" * 78)
    print("  PASS 3 - squeeze a little, let go, repeat")
    print("=" * 78)
    header = f"  {'torque':>8} {'want q':>9}" + "".join(f"{r.name + ' q':>14}" for r in readings)
    header += "".join(f"{r.name + ' gap':>14}" for r in readings)
    print(header)
    ok = True
    for tau in TORQUE_STAIRCASE:
        want = tau / STIFFNESS
        row = f"  {tau:>8.3f} {want:>9.5f}"
        row += "".join(f"{r.settled[tau]:>14.6f}" for r in readings)
        row += "".join(f"{r.gap_mm[tau]:>12.2f}mm" for r in readings)
        print(row)
        for reading in readings:
            if abs(reading.settled[tau] - want) > TOL_SETTLED:
                print(
                    f"    FAIL  {reading.name}: settled {reading.settled[tau]:.6f}, "
                    f"the spring says {want:.6f} (tol {TOL_SETTLED})"
                )
                ok = False
        ok &= _pairwise(
            readings, lambda r, t=tau: np.array([r.settled[t]]), TOL_SETTLED, f"settled at {tau:.3f} N*m", quiet=True
        )
    print()
    print("  released again - did it return to fully open?")
    for reading in readings:
        worst = max(abs(v - OPEN_ANGLE) for v in reading.reopened.values())
        verdict = "ok" if worst <= TOL_REOPEN else "FAIL"
        print(f"    {reading.name:>8}: worst residual {worst:.6f} rad  [{verdict}]")
        ok &= verdict == "ok"
    print("  PASS 3: " + ("OK" if ok else "FAILED"))
    return ok


def report_gravity(readings: list[Reading]) -> bool:
    print("\n" + "=" * 78)
    print("  PASS 4 - with gravity: how far does the tong's own weight close it")
    print("=" * 78)
    ok = True
    for reading in readings:
        sag = reading.gravity_sag
        spring, weight = gravity_balance(sag)
        verdict = "ok" if abs(spring - weight) <= TOL_SETTLED * STIFFNESS else "FAIL"
        print(
            f"  {reading.name:>8}: {sag:.6f} rad, pad gap {1000 * pad_gap(sag):.2f} mm   "
            f"spring {spring:.6f} vs weight {weight:.6f} N*m  [{verdict}]"
        )
        ok &= verdict == "ok"
    ok &= _pairwise(readings, lambda r: np.array([r.gravity_sag]), TOL_SETTLED, "gravity sag")
    print("  PASS 4: " + ("OK" if ok else "FAILED"))
    return ok


# ══════════════════════════════════════════════════════════════════


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", nargs="+", default=list(BACKENDS), choices=list(BACKENDS))
    ap.add_argument("--release_steps", type=int, default=800)
    args = ap.parse_args()

    xml = fixed_base_asset()

    print("=" * 78)
    print("  SPRING TONG: is it the same tong on every backend?")
    print("=" * 78)
    print(f"  asset       {TONG_XML}")
    print(f"  fixed base  {xml}  (derived: <freejoint/> removed)")
    print(f"  dt          {DT} s, no decimation")
    print(f"  spring      k {STIFFNESS} N*m/rad, c {DAMPING} N*m*s/rad, rest at {OPEN_ANGLE} rad")
    print(
        f"  travel      {OPEN_ANGLE} rad open ({1000 * pad_gap(OPEN_ANGLE):.1f} mm) .. "
        f"{CLOSED_ANGLE} rad shut ({1000 * pad_gap(CLOSED_ANGLE):.1f} mm)"
    )

    readings: list[Reading] = []
    for name in args.sims:
        print(f"\n  running {name} ...")
        reading = Reading(name=name)
        run_backend(BACKENDS[name], xml, reading, args.release_steps)
        readings.append(reading)

    results = [
        report_parse(readings),
        report_release(readings),
        report_ringdown(readings),
        report_staircase(readings),
        report_gravity(readings),
    ]

    print("\n" + "=" * 78)
    print("  OVERALL: " + ("PASS" if all(results) else "FAIL"))
    print("=" * 78)
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
