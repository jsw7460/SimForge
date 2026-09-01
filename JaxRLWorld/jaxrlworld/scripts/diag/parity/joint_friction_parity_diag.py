"""What joint friction does each backend actually end up with?

``frictionloss`` used to default to ``0.0``, which cannot distinguish "no
friction" from "not specified". Newton and Genesis read it as unset and
kept the asset's value; the mjlab adapter passed the zero through and
mjlab wrote it, so the same go2 XML ran with its declared 0.2 N*m on two
backends and none on the third. The default is ``None`` now and all
three keep the asset's value.

That is a dynamics change on mjlab alone, and go2's gait-conditioned
policy trains there against exactly the quantities joint friction moves
— foot slip and foot placement. So the question is no longer whether the
value changed but whether it changed to the RIGHT thing: the same
magnitude, on the same joints, as the two backends that still train.

Reads the built model rather than the config, per backend, in its own
process — a config that agrees proves nothing about what the simulator
was handed.

    jaxpy -m jaxrlworld.scripts.diag.parity.joint_friction_parity_diag
    jaxpy -m jaxrlworld.scripts.diag.parity.joint_friction_parity_diag --preset go2_gait_mujoco
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from jaxrlworld.rl.configs.presets.go2.base import Go2FlatConfig
from jaxrlworld.rl.runners import BaseRunner

_PRESETS = {"go2": Go2FlatConfig}
_SIMS = ("genesis", "newton", "mujoco")


def _as_numpy(value) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "numpy"):
        return np.asarray(value.detach().cpu().numpy() if hasattr(value, "detach") else value.numpy())
    try:
        import warp as wp

        return wp.to_torch(value).detach().cpu().numpy()
    except Exception:
        return None


def _friction_per_dof(sim: str, env) -> np.ndarray:
    """Per-DOF joint friction, straight out of the built model."""
    if sim == "mujoco":
        return np.asarray(env.scene_manager.sim.mj_model.dof_frictionloss)
    if sim == "newton":
        model = env.scene_manager.model
        return _as_numpy(model.joint_friction)
    if sim == "genesis":
        entity = env.scene_manager["robot"]
        return _as_numpy(entity.get_dofs_frictionloss())
    raise ValueError(f"Unknown sim {sim!r}")


def run_single(preset: str, sim: str, num_envs: int) -> dict:
    cfgs = _PRESETS[preset](sim_type=sim, num_envs=num_envs).build()
    env = BaseRunner._create_env_from_config(cfgs)
    friction = _friction_per_dof(sim, env)
    if friction is not None and friction.ndim > 1:
        # Genesis reports per-env rows once batch_dofs_info is on; they are
        # identical unless friction is randomized, and one row is the model.
        friction = friction[0]

    values = [float(v) for v in np.asarray(friction).ravel()] if friction is not None else []
    nonzero = [v for v in values if v != 0.0]
    result = {
        "sim": sim,
        "num_dofs": len(values),
        "nonzero_count": len(nonzero),
        "distinct": sorted({round(v, 6) for v in values}),
        "actuated_joint_names": list(env.act_manager.actuated_joint_names),
    }
    print(
        f"  {sim:<9} dofs {result['num_dofs']:3d}   with friction {result['nonzero_count']:3d}   "
        f"values {result['distinct']}"
    )
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=list(_PRESETS), default="go2")
    ap.add_argument("--sim", choices=_SIMS, default=None)
    ap.add_argument("--result-json", default=None)
    ap.add_argument("--num-envs", type=int, default=16)
    args = ap.parse_args()

    if args.sim is not None:
        result = run_single(args.preset, args.sim, args.num_envs)
        if args.result_json:
            Path(args.result_json).write_text(json.dumps(result))
        return 0

    tmp = Path(tempfile.mkdtemp(prefix="joint_friction_"))
    out: dict[str, dict] = {}
    env_vars = dict(os.environ, JAXRLWORLD_ALLOW_MULTI_SIM="1")
    print("=" * 78)
    print(f"JOINT FRICTION, AS BUILT  [preset={args.preset}]")
    print("=" * 78)
    for sim in _SIMS:
        path = tmp / f"{sim}.json"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "jaxrlworld.scripts.diag.parity.joint_friction_parity_diag",
                "--preset",
                args.preset,
                "--sim",
                sim,
                "--num-envs",
                str(args.num_envs),
                "--result-json",
                str(path),
            ],
            env=env_vars,
            check=False,
        )
        if path.exists():
            out[sim] = json.loads(path.read_text())

    print("=" * 78)
    for sim, r in out.items():
        print(f"  {sim:<9} {r['nonzero_count']}/{r['num_dofs']} dofs carry friction, values {r['distinct']}")
    counts = {r["nonzero_count"] for r in out.values()}
    values = {tuple(r["distinct"]) for r in out.values()}
    if len(out) == len(_SIMS) and len(counts) == 1 and len(values) == 1:
        print("  The three agree: same magnitude, same number of joints.")
        print("  Then mjlab's change is the physics it always should have had,")
        print("  and a policy tuned without friction has to be retrained, not restored.")
    else:
        print("  THEY DISAGREE — the backends are not simulating the same robot.")
        print("  A backend applying friction to more DOFs than the others (a free")
        print("  base's six, say) would be a bug, not a physics correction.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
