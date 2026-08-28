"""K1 G1-recipe variable_posture: is the DEFAULT joint pose it shapes against the
right one (the K1 home keyframe), per joint, and consistent across backends?

``variable_posture`` penalizes ``(q - q_default)^2 / std^2``. If ``q_default`` is
wrong (zeros, a stale pose, or mis-ordered vs the joints) the reward silently
pulls the robot toward the wrong posture. The two wrappers resolve it
differently — newton/genesis take ``env.robot_data.default_joint_pos`` in
``actuated_joint_names`` order; mujoco takes ``robot.data.default_joint_pos``
sliced by the ``find_joints`` ids — so this diag reads the ACTUAL default tensor
the live reward instance holds (``reward_manager._instances['variable_posture']
._impl._default_joint_pos``) and, per sim, checks each joint against the K1
config ``default_joint_angles`` (the home keyframe: shoulder_roll +/-1.4,
elbow_yaw +/-0.4, hip_pitch -0.2, knee 0.4, ankle_pitch -0.2, else 0):

  1. PER-JOINT MATCH — the default value for every joint equals the configured
     home-keyframe angle (regex-resolved by name).
  2. NOT ZEROS — the pose is not silently all-zero (the classic bug).
  3. CROSS-SIM — the {joint: default} map is identical on all three backends.

Run::

    jaxpy -m rlworld.scripts.diag.k1.k1_variable_posture_default_diag --sim mujoco
    jaxpy -m rlworld.scripts.diag.k1.k1_variable_posture_default_diag            # all
"""

from __future__ import annotations

import argparse
import re

_SIMS = ("genesis", "newton", "mujoco")
_SIM_KEY = {"genesis": "Genesis", "newton": "Newton", "mujoco": "MujocoEnv"}
_TOL = 1e-4


def _stage(msg: str) -> None:
    print(f"  · {msg}", flush=True)


def _expected(name: str, angle_dict: dict) -> float:
    """Home-keyframe angle for a joint name (regex fullmatch; 0 if unmatched)."""
    for pat, val in angle_dict.items():
        if re.fullmatch(pat, name):
            return float(val)
    return 0.0


def _names_and_default(sim: str, env, tracker_default):
    """(joint_names, default_values) aligned, in the order the tracker uses."""
    import torch

    d = torch.as_tensor(tracker_default).detach().float().cpu()
    if d.dim() == 2:
        d = d[0]
    if sim in ("newton", "genesis"):
        names = list(env.act_manager.actuated_joint_names)
        return names, d
    # mujoco: the wrapper sliced default_joint_pos by find_joints. Reconstruct
    # the same order to label each value.
    robot = env.scene_manager.get_entity("robot")
    ids, names = robot.find_joints([r".*"])
    full = torch.as_tensor(robot.data.default_joint_pos).detach().float().cpu()
    if full.dim() == 2:
        full = full[0]
    vals = full[list(ids)]
    return list(names), vals


def run_cell(sim: str, num_envs: int, seed: int) -> dict:
    import torch

    torch.manual_seed(seed)
    _stage(f"cell start: {sim}")

    from rlworld.rl.configs.presets.k1_joystick.g1_recipe import K1G1RecipeConfig
    from rlworld.rl.evals.sim_initializers import get_initializer

    preset = K1G1RecipeConfig(sim_type=sim, num_envs=num_envs, seed=seed)
    angle_dict = dict(preset.robot.default_joint_angles)
    cfgs = preset.build()
    env = get_initializer(_SIM_KEY[sim]).init_environment(cfgs)
    env.reset()

    tracker = env.reward_manager._instances["variable_posture"]
    default = tracker._impl._default_joint_pos
    names, vals = _names_and_default(sim, env, default)

    rows = []
    for n, v in zip(names, [float(x) for x in vals]):
        exp = _expected(n, angle_dict)
        rows.append({"name": n, "default": v, "expected": exp, "ok": abs(v - exp) <= _TOL})
    _stage(f"cell done: {sim}")
    return {"sim": sim, "rows": rows, "all_zero": all(abs(r["default"]) <= _TOL for r in rows)}


def _leaf(name: str) -> str:
    return name.split("/")[-1]


def _print_cell(r: dict) -> dict:
    sim = r["sim"]
    print(f"\n===== {sim.upper()} ({len(r['rows'])} joints) =====")
    print(f"    {'joint':24}{'default':>10}{'expected':>10}   ok")
    per_joint = {}
    all_ok = True
    for row in r["rows"]:
        leaf = _leaf(row["name"])
        per_joint[leaf] = row["default"]
        flag = "OK" if row["ok"] else "!! MISMATCH"
        if not row["ok"]:
            all_ok = False
        # only print nonzero-expected joints + any mismatch (keep it readable)
        if abs(row["expected"]) > _TOL or not row["ok"]:
            print(f"    {leaf:24}{row['default']:>10.3f}{row['expected']:>10.3f}   {flag}")
    zeros = r["all_zero"]
    print(f"  → all-zero default (bug): {zeros}   per-joint match: {all_ok}")
    print(f"  VERDICT: {'PASS' if (all_ok and not zeros) else 'CHECK'}")
    return per_joint


def main() -> int:
    ap = argparse.ArgumentParser(description="K1 variable_posture default-pose correctness diag.")
    ap.add_argument("--sim", choices=_SIMS, help="Single backend (default: all).")
    ap.add_argument("--num_envs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sims = [args.sim] if args.sim else list(_SIMS)
    results = []
    for sim in sims:
        try:
            results.append(run_cell(sim, args.num_envs, args.seed))
        except Exception as e:  # noqa: BLE001
            import traceback

            print(f"\n[{sim}] FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()

    maps = {}
    for r in results:
        maps[r["sim"]] = _print_cell(r)

    if len(maps) > 1:
        print("\n===== CROSS-SIM (default per joint) =====")
        sims_done = list(maps)
        all_joints = sorted({j for m in maps.values() for j in m})
        print(f"    {'joint':24}" + "".join(f"{s:>11}" for s in sims_done))
        consistent = True
        for j in all_joints:
            vs = [maps[s].get(j, float("nan")) for s in sims_done]
            row = "".join(f"{v:>11.3f}" for v in vs)
            finite = [v for v in vs if v == v]
            if finite and (max(finite) - min(finite)) > _TOL:
                consistent = False
                row += "  !!"
            # print only nonzero or inconsistent
            if any(abs(v) > _TOL for v in finite) or "!!" in row:
                print(f"    {j:24}{row}")
        print(f"  → identical across backends: {consistent}")

    print()
    return 0 if len(results) == len(sims) else 1


if __name__ == "__main__":
    raise SystemExit(main())
