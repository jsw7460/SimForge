"""Does the K1 getup preset present the velocity preset's policy contract?

The getup policy is meant to be deployed through the velocity policy's path
with no change to the deployment stack, which holds only if the two presets
agree on everything the policy sees and emits. This script compares the two
built configs field by field and names every difference, so that a drift in
either preset shows up here rather than on the robot.

Sections:

  A. Config. The actor and critic observation groups, the action config, the
     robot entity (actuators, initial state), the control timing, the network
     and the algorithm, compared as dictionaries. The only difference allowed
     is the action manager's ``settle_steps``, which is a reset-time hold and
     not part of the policy interface; anything else fails.
  B. Environment (``--env``). Both environments are built, one after the
     other, and the actor column layout, the action scale / offset / clip
     tensors and the actuator gains are compared numerically, so a config that
     resolves differently at runtime (a regex that matches another joint set,
     a default pose that lands in a different order) is caught too.

Run one backend per process::

    jaxpy -m jaxrlworld.scripts.diag.k1.getup_io_contract_diag --sim mujoco
    jaxpy -m jaxrlworld.scripts.diag.k1.getup_io_contract_diag --sim mujoco --env
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from jaxrlworld.rl.configs.presets.k1_getup.base import K1GetupConfig
from jaxrlworld.rl.configs.presets.k1_velocity.base import K1VelocityConfig

# Config paths that may differ between the two presets. Everything else
# under the compared sections must be identical.
_ALLOWED_DIFFS = {"action.settle_steps"}

_SECTIONS = (
    "observation.actor",
    "observation.critic",
    "action",
    "scene.entities.robot",
    "env.decimation",
    "nn",
    "algorithm",
)
_TIMING_FIELDS = {
    "mujoco": ("scene.physics_dt", "scene.substeps"),
    "newton": ("scene.dt", "scene.substeps"),
    # Genesis keeps its timing on ``gs.options`` objects, which the config
    # serializer leaves out; they are compared on the objects below.
    "genesis": (),
}


def _lookup(d: dict, path: str) -> Any:
    node: Any = d
    for key in path.split("."):
        node = node[key]
    return node


def _diff(a: Any, b: Any, path: str, out: list[str]) -> None:
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            sub = f"{path}.{key}" if path else key
            if key not in a:
                out.append(f"{sub}: missing in velocity")
            elif key not in b:
                out.append(f"{sub}: missing in getup")
            else:
                _diff(a[key], b[key], sub, out)
        return
    if isinstance(a, list | tuple) and isinstance(b, list | tuple):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} vs {len(b)}")
            return
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            _diff(x, y, f"{path}[{i}]", out)
        return
    if a != b:
        out.append(f"{path}: {a!r} vs {b!r}")


def section_a(sim: str, num_envs: int) -> bool:
    cfg_v = K1VelocityConfig(sim_type=sim, num_envs=num_envs).build()
    cfg_g = K1GetupConfig(sim_type=sim, num_envs=num_envs).build()
    velocity = cfg_v.recursive_to_dict()
    getup = cfg_g.recursive_to_dict()
    ok = True
    if sim == "genesis":
        timing_v = (cfg_v.scene.sim_options.dt, cfg_v.scene.sim_options.substeps, cfg_v.scene.rigid_options.dt)
        timing_g = (cfg_g.scene.sim_options.dt, cfg_g.scene.sim_options.substeps, cfg_g.scene.rigid_options.dt)
        same = timing_v == timing_g
        ok &= same
        print(f"[A] {'scene timing (gs.options)':28s} {'OK' if same else 'FAIL'}")
        if not same:
            print(f"      DIFF     {timing_v} vs {timing_g}")
    for path in _SECTIONS + _TIMING_FIELDS[sim]:
        diffs: list[str] = []
        _diff(_lookup(velocity, path), _lookup(getup, path), path, diffs)
        unexpected = [d for d in diffs if d.split(":")[0] not in _ALLOWED_DIFFS]
        allowed = [d for d in diffs if d.split(":")[0] in _ALLOWED_DIFFS]
        status = "OK" if not unexpected else "FAIL"
        ok &= not unexpected
        print(f"[A] {path:28s} {status}")
        for d in allowed:
            print(f"      allowed  {d}")
        for d in unexpected:
            print(f"      DIFF     {d}")
    return ok


def _env_fingerprint(cfgs) -> dict[str, Any]:
    from jaxrlworld.rl.runners.base_runner import BaseRunner

    env = BaseRunner._create_env_from_config(cfgs)
    am = env.act_manager
    actuators = [(act.stiffness.detach().cpu().clone(), act.damping.detach().cpu().clone()) for act, _ in am.actuators]
    return {
        "actor_layout": [(name, func.__name__, width) for name, func, width in env.obs_manager.term_layout("actor")],
        "joint_names": list(am.actuated_joint_names),
        "scale": am._scale.detach().cpu().clone(),
        "offset": am._offset[0].detach().cpu().clone(),
        "clip_low": am._clip_low.detach().cpu().clone(),
        "clip_high": am._clip_high.detach().cpu().clone(),
        "actuator_gains": actuators,
    }


def section_b(sim: str, num_envs: int) -> bool:
    import torch

    fp_v = _env_fingerprint(K1VelocityConfig(sim_type=sim, num_envs=num_envs).build())
    fp_g = _env_fingerprint(K1GetupConfig(sim_type=sim, num_envs=num_envs).build())

    def same_value(a: Any, b: Any) -> bool:
        if isinstance(a, torch.Tensor):
            return isinstance(b, torch.Tensor) and a.shape == b.shape and bool(torch.equal(a, b))
        if isinstance(a, list | tuple):
            return len(a) == len(b) and all(same_value(x, y) for x, y in zip(a, b, strict=True))
        return a == b

    ok = True
    for key in fp_v:
        a, b = fp_v[key], fp_g[key]
        same = same_value(a, b)
        ok &= same
        print(f"[B] {key:14s} {'OK' if same else 'FAIL'}")
        if not same:
            print(f"      velocity {a}")
            print(f"      getup    {b}")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sim", choices=("mujoco", "newton", "genesis"), default="mujoco")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--env", action="store_true", help="also build both environments (section B)")
    args = parser.parse_args()

    ok = section_a(args.sim, args.num_envs)
    if args.env:
        ok &= section_b(args.sim, args.num_envs)
    print("ALL OK" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
