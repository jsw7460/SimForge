"""Left/right symmetry audit and symmetrized twin of the Booster K1 asset.

The shipped ``k1.xml`` is not an exact mirror image of itself: the
explicit ``<inertial>`` entries of the arm and hip links differ left/right
by up to a few tenths of a percent (Booster's CAD export), and the forearm
collision cylinder is fitted to each side's mesh, which puts it 6.7 mm off
its mirror. The physics of such a model cannot be exactly mirror
symmetric, whatever the observation operator does.

:func:`audit` measures that asymmetry pair by pair. :func:`write_symmetrized`
writes a twin whose paired links carry the averaged inertia (each side the
mirror of the other) and whose paired primitive collision geoms are placed
explicitly as mirrors. On the twin the only remaining asymmetry is the foot
mesh itself (a few micrometres), so a mirror-equivariance test of the
observation and action operators can be held to float precision there. The
twin is a diagnostic artefact, not a training asset.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

Y = np.array([1.0, -1.0, 1.0])
M = np.diag(Y)
# Plain ints: the compiled model stores geom types as integers, and a
# pybind enum does not test equal to a numpy integer inside a set.
PRIMITIVE_TYPES = {
    int(mujoco.mjtGeom.mjGEOM_SPHERE),
    int(mujoco.mjtGeom.mjGEOM_CAPSULE),
    int(mujoco.mjtGeom.mjGEOM_CYLINDER),
    int(mujoco.mjtGeom.mjGEOM_BOX),
}


def mate(name: str) -> str | None:
    for a, b in (("Left", "Right"), ("left", "right")):
        if a in name:
            return name.replace(a, b)
        if b in name:
            return name.replace(b, a)
    return None


def _rot(quat_wxyz) -> np.ndarray:
    r = np.zeros(9)
    mujoco.mju_quat2Mat(r, np.asarray(quat_wxyz, dtype=np.float64))
    return r.reshape(3, 3)


def _body_inertia_tensor(model, body_id: int) -> np.ndarray:
    R = _rot(model.body_iquat[body_id])
    return R @ np.diag(model.body_inertia[body_id]) @ R.T


def audit(xml_path: str | Path) -> dict[str, float]:
    """Worst left/right mirror mismatch per quantity, from the compiled model."""
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    B, J, G = mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_JOINT, mujoco.mjtObj.mjOBJ_GEOM
    worst = {
        "mass_rel": 0.0,
        "inertia_rel": 0.0,
        "com_m": 0.0,
        "body_pos_m": 0.0,
        "joint_pos_m": 0.0,
        "joint_axis": 0.0,
        "joint_range": 0.0,
        "prim_geom_pos_m": 0.0,
        "prim_geom_size_m": 0.0,
        "prim_geom_axis": 0.0,
        "mesh_geom_pos_m": 0.0,
    }
    names = [mujoco.mj_id2name(model, B, i) for i in range(model.nbody)]
    for i, n in enumerate(names):
        mt = mate(n)
        if mt is None or mt not in names or n > mt:
            continue
        j = names.index(mt)
        worst["mass_rel"] = max(
            worst["mass_rel"], abs(model.body_mass[i] - model.body_mass[j]) / max(model.body_mass[i], 1e-12)
        )
        IL, IR = _body_inertia_tensor(model, i), _body_inertia_tensor(model, j)
        worst["inertia_rel"] = max(worst["inertia_rel"], np.abs(IL - M @ IR @ M).max() / max(np.abs(IL).max(), 1e-12))
        worst["com_m"] = max(worst["com_m"], np.abs(model.body_ipos[i] * Y - model.body_ipos[j]).max())
        worst["body_pos_m"] = max(worst["body_pos_m"], np.abs(model.body_pos[i] * Y - model.body_pos[j]).max())
    jnames = [mujoco.mj_id2name(model, J, i) for i in range(model.njnt)]
    for i, n in enumerate(jnames):
        mt = mate(n)
        if mt is None or mt not in jnames or n > mt:
            continue
        j = jnames.index(mt)
        flip = -1.0 if ("Roll" in n or "Yaw" in n) else 1.0
        worst["joint_pos_m"] = max(worst["joint_pos_m"], np.abs(model.jnt_pos[i] * Y - model.jnt_pos[j]).max())
        worst["joint_axis"] = max(worst["joint_axis"], np.abs(model.jnt_axis[i] - model.jnt_axis[j]).max())
        worst["joint_range"] = max(
            worst["joint_range"], np.abs(np.sort(flip * model.jnt_range[i]) - model.jnt_range[j]).max()
        )
    gnames = [mujoco.mj_id2name(model, G, i) for i in range(model.ngeom)]
    for i, n in enumerate(gnames):
        if model.geom_contype[i] == 0 and model.geom_conaffinity[i] == 0:
            continue
        mt = mate(n)
        if mt is None or mt not in gnames or n > mt:
            continue
        j = gnames.index(mt)
        dpos = np.abs(model.geom_pos[i] * Y - model.geom_pos[j]).max()
        if int(model.geom_type[i]) in PRIMITIVE_TYPES:
            worst["prim_geom_pos_m"] = max(worst["prim_geom_pos_m"], dpos)
            worst["prim_geom_size_m"] = max(
                worst["prim_geom_size_m"], np.abs(model.geom_size[i] - model.geom_size[j]).max()
            )
            axis_l = _rot(model.geom_quat[i])[:, 2]
            axis_r = _rot(model.geom_quat[j])[:, 2]
            # a primitive's axis is a line: +a and -a are the same geom
            worst["prim_geom_axis"] = max(
                worst["prim_geom_axis"], min(np.abs(axis_l * Y - axis_r).max(), np.abs(axis_l * Y + axis_r).max())
            )
        else:
            worst["mesh_geom_pos_m"] = max(worst["mesh_geom_pos_m"], dpos)
    return worst


def _fmt(v) -> str:
    return " ".join(f"{float(x):.9g}" for x in np.asarray(v).ravel())


def write_symmetrized(src: str | Path, dst: str | Path) -> Path:
    """Write the mirror-symmetrized twin of ``src`` to ``dst`` (mesh paths kept relative)."""
    src, dst = Path(src), Path(dst)
    if dst.parent.resolve() != src.parent.resolve():
        raise ValueError("write the twin next to the source so its relative meshdir still resolves")
    model = mujoco.MjModel.from_xml_path(str(src))
    B, G = mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_GEOM
    tree = ET.parse(src)
    root = tree.getroot()
    bodies = {el.get("name"): el for el in root.iter("body")}
    geoms = {el.get("name"): el for el in root.iter("geom") if el.get("name")}

    # Paired links: averaged inertia, each side the mirror of the other.
    for name, el in bodies.items():
        mt = mate(name)
        if mt is None or mt not in bodies or name > mt:
            continue
        i, j = mujoco.mj_name2id(model, B, name), mujoco.mj_name2id(model, B, mt)
        inertial_l, inertial_r = el.find("inertial"), bodies[mt].find("inertial")
        if inertial_l is None and inertial_r is None:
            # Inertia derived from the geoms on both sides (the hand end
            # balls): the audit reports those pairs at zero already.
            continue
        if inertial_l is None or inertial_r is None:
            raise ValueError(f"{name}/{mt}: <inertial> on one side only")
        I_sym = 0.5 * (_body_inertia_tensor(model, i) + M @ _body_inertia_tensor(model, j) @ M)
        pos_sym = 0.5 * (model.body_ipos[i] + Y * model.body_ipos[j])
        mass = 0.5 * (model.body_mass[i] + model.body_mass[j])
        for inertial, I, pos in ((inertial_l, I_sym, pos_sym), (inertial_r, M @ I_sym @ M, Y * pos_sym)):
            for attr in ("quat", "diaginertia", "fullinertia"):
                inertial.attrib.pop(attr, None)
            inertial.set("pos", _fmt(pos))
            inertial.set("mass", _fmt(mass))
            inertial.set("fullinertia", _fmt([I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]]))

    # Paired primitive collision geoms: explicit placement, right = mirror(left).
    for name, el in geoms.items():
        mt = mate(name)
        if mt is None or mt not in geoms or name > mt:
            continue
        i = mujoco.mj_name2id(model, G, name)
        if int(model.geom_type[i]) not in PRIMITIVE_TYPES:
            continue
        if model.geom_contype[i] == 0 and model.geom_conaffinity[i] == 0:
            continue
        pos, size, axis = model.geom_pos[i], model.geom_size[i], _rot(model.geom_quat[i])[:, 2]
        for geom, p, a in ((el, pos, axis), (geoms[mt], Y * pos, Y * axis)):
            for attr in ("mesh", "quat", "axisangle", "euler", "xyaxes", "fromto"):
                geom.attrib.pop(attr, None)
            geom.set("pos", _fmt(p))
            geom.set("size", _fmt(size))
            geom.set("zaxis", _fmt(a))

    root.insert(
        0,
        ET.Comment(
            " GENERATED by scripts/diag/k1/symmetrize_asset.py: left/right-symmetrized "
            "twin of k1.xml for the mirror-equivariance diagnostic. Not a training asset. "
        ),
    )
    tree.write(dst, encoding="unicode")
    return dst


if __name__ == "__main__":
    here = Path(__file__).resolve().parents[3] / "assets" / "K1"
    src = here / "k1.xml"
    dst = write_symmetrized(src, here / "k1_symmetrized.xml")
    print("source asset asymmetry:")
    for k, v in audit(src).items():
        print(f"  {k:18s} {v:.3e}")
    print(f"\nsymmetrized twin -> {dst}")
    for k, v in audit(dst).items():
        print(f"  {k:18s} {v:.3e}")
