"""Every preset on every backend: build, step, fingerprint, compare.

Each cell (preset x simulator) runs in its own process: build the
config at a small env count, create the runner and env, reset, and step
``--steps`` times with a fixed, CPU-seeded random action sequence. A
cell passes when nothing raises, every reward and observation stays
finite, and — in ``compiled`` mode — nothing recompiles after the warm
steps. Every step's reward sum, squared sum, done count and actor
observation sums are written as the cell's fingerprint.

Four modes pin down what a change did:

    baseline   run under a copy of the package from BEFORE the change
               (``--tree`` puts it first on PYTHONPATH; the child is
               launched by file path so this script need not exist there)
    eager      the current package with every compiled path switched
               off (reward chain, actuator torque chain, Genesis contact
               substep) — must match ``baseline`` up to run-to-run noise
    compiled   the current package as configured — checks that every
               cell builds, steps, stays finite and never recompiles;
               its trajectory is compared to ``eager`` for information
    shadow     the env runs eager and, at every call, the compiled
               program of each component is evaluated on the SAME inputs
               and compared (state rewound in between). This is the
               numerical verdict on the compiled paths: exact, per
               component, no chaos — a trajectory comparison cannot
               separate a fused kernel's last-bit rounding from a wrong
               computation once physics has amplified it.

The mjwarp-based backends (Newton's SolverMuJoCo, mjlab) are not
bitwise reproducible from one process to the next (parallel reductions,
mujoco_warp#562), so a trajectory fingerprint is judged against the
run-to-run spread of the same code (``--floor``), not against zero:

    jaxpy -m jaxrlworld.scripts.diag.gates.check_all_presets --mode baseline --tree ~/baseline/JaxRLWorld --out runs/baseline
    jaxpy -m jaxrlworld.scripts.diag.gates.check_all_presets --mode eager --out runs/eager
    jaxpy -m jaxrlworld.scripts.diag.gates.check_all_presets --mode eager --out runs/eager2
    jaxpy -m jaxrlworld.scripts.diag.gates.check_all_presets --mode eager --out runs/eager --against runs/baseline --floor runs/eager2 --compare-only
    jaxpy -m jaxrlworld.scripts.diag.gates.check_all_presets --mode compiled --out runs/compiled --against runs/eager --floor runs/eager2
    jaxpy -m jaxrlworld.scripts.diag.gates.check_all_presets --mode shadow --out runs/shadow
    jaxpy -m jaxrlworld.scripts.diag.gates.check_all_presets --mode shadow --out runs/shadow --only k1 --sims mujoco

Presets kept outside this package join the sweep through ``--specs``, a
module exposing ``PRESETS`` in the same entry shape and
``NO_REWARD_SHADOW`` (label -> why its reward stack cannot be evaluated
twice in one step).
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch._dynamo

import jaxrlworld
from jaxrlworld.rl.actuators.actuator_cfg import IdealPDActuatorCfg
from jaxrlworld.rl.runners import BaseRunner

_SIMS = ("genesis", "newton", "mujoco")
_P = "jaxrlworld.rl.configs.presets"

# (label, loader, sims). A loader is ``cls:module:Class`` — built as
# ``Class(sim_type=sim, num_envs=n).build()`` — or ``fn:module:get_config``,
# called as ``get_config(sim=sim)`` (``get_config()`` when it takes no
# backend) with the env count set afterwards.
_PUBLIC = [
    ("go2_flat", f"cls:{_P}.go2.base:Go2FlatConfig", _SIMS),
    ("go2_gait", f"cls:{_P}.go2.genesis.gait_conditioned:Go2GaitConditionedGenesisConfig", ("genesis",)),
    ("go2_gait", f"cls:{_P}.go2.newton.gait_conditioned:Go2GaitConditionedNewtonConfig", ("newton",)),
    ("go2_gait", f"cls:{_P}.go2.mujoco.gait_conditioned:Go2GaitConditionedMujocoConfig", ("mujoco",)),
    ("go2_rough", f"cls:{_P}.go2.genesis.rough:Go2RoughGenesisConfig", ("genesis",)),
    ("go2_rough", f"cls:{_P}.go2.newton.rough:Go2RoughNewtonConfig", ("newton",)),
    ("go2_rough", f"cls:{_P}.go2.mujoco.rough:Go2RoughMujocoConfig", ("mujoco",)),
    ("go2_sac", f"cls:{_P}.go2.newton.sac:Go2SACNewtonConfig", ("newton",)),
    ("g1_flat", f"cls:{_P}.g1_29dof.base:G1FlatConfig", _SIMS),
    ("g1_rough", f"cls:{_P}.g1_29dof.genesis.rough:G1RoughGenesisConfig", ("genesis",)),
    ("g1_rough", f"cls:{_P}.g1_29dof.newton.rough:G1RoughNewtonConfig", ("newton",)),
    ("g1_rough", f"cls:{_P}.g1_29dof.mujoco.rough:G1RoughMujocoConfig", ("mujoco",)),
    ("g1_tracking", f"cls:{_P}.g1_tracking.base:G1TrackingConfig", _SIMS),
    ("g1_tracking_tf", f"cls:{_P}.g1_tracking.transformer:G1TrackingTransformerConfig", _SIMS),
    ("t1_getup", f"cls:{_P}.t1_getup.base:T1GetupConfig", _SIMS),
    ("t1_tracking", f"cls:{_P}.t1_tracking.base:T1TrackingConfig", _SIMS),
    ("t1_tracking_tf", f"cls:{_P}.t1_tracking.transformer:T1TrackingTransformerConfig", _SIMS),
    ("k1_joystick", f"cls:{_P}.k1_joystick.base:K1JoystickConfig", _SIMS),
    ("k1_g1_recipe", f"cls:{_P}.k1_joystick.g1_recipe:K1G1RecipeConfig", _SIMS),
    ("k1_no_heading", f"cls:{_P}.k1_joystick.no_heading:K1NoHeadingConfig", _SIMS),
    ("k1_calib", f"cls:{_P}.k1_joystick.calib:K1CalibConfig", _SIMS),
    ("yam_lift", f"cls:{_P}.yam_lift.base:YamLiftConfig", _SIMS),
    ("yam_lift_vision", f"cls:{_P}.yam_lift.vision:YamLiftVisionConfig", ("mujoco",)),
    ("yam_dual", f"cls:{_P}.yam_dual.base:YamDualArmConfig", _SIMS),
    ("lab_cell", f"cls:{_P}.lab_cell.base:LabCellConfig", _SIMS),
]


# Presets whose reward stack cannot be evaluated twice in one step, so
# the shadow mode reports the reward as unsupported instead of comparing.
_NO_REWARD_SHADOW: dict[str, str] = {}


def _extra_specs(module: str | None):
    """``--specs``: a module exposing ``PRESETS`` (same entry shape as
    ``_PUBLIC``) and ``NO_REWARD_SHADOW``, for presets kept outside this
    package."""
    return importlib.import_module(module) if module else None


def _no_reward_shadow(extra) -> dict[str, str]:
    return {**_NO_REWARD_SHADOW, **(extra.NO_REWARD_SHADOW if extra else {})}


# ---------------------------------------------------------------------
# child: one cell
# ---------------------------------------------------------------------


def _load(loader: str, sim: str, num_envs: int):
    kind, module, name = loader.split(":", 2)
    target = getattr(importlib.import_module(module), name)
    if kind == "cls":
        return target(sim_type=sim, num_envs=num_envs).build()
    cfgs = target(sim=sim) if "sim" in inspect.signature(target).parameters else target()
    cfgs.env.num_envs = num_envs
    cfgs.scene.num_envs = num_envs
    return cfgs


def _apply_mode(cfgs, mode: str, sim: str) -> None:
    """``eager`` / ``shadow``: every compiled path off (shadow compiles on
    the side). ``compiled`` / ``baseline``: untouched."""
    if mode not in ("eager", "shadow"):
        return
    cfgs.reward.compile_terms = False
    if sim == "genesis":
        cfgs.env.compile_contact_kernels = False
    for entity in cfgs.scene.entities.values():
        for act in entity.articulation.actuators:
            if isinstance(act, IdealPDActuatorCfg):
                act.compile_kernel = False


def _finite(value) -> bool:
    if isinstance(value, dict):
        return all(_finite(v) for v in value.values())
    return bool(torch.isfinite(value).all())


def _obs_vector(obs):
    """The actor group's state vector (the dict form carries images too)."""
    actor = obs["actor"]
    return actor["actor"] if isinstance(actor, dict) else actor


_SHADOW_TOL = {"actuator": 1e-5, "reward": 1e-4, "contact": 1e-5}


def run_cell(
    label: str,
    loader: str,
    sim: str,
    mode: str,
    num_envs: int,
    steps: int,
    warm: int,
    seed: int,
    no_reward_shadow: dict[str, str],
) -> dict:
    t_build = time.perf_counter()
    cfgs = _load(loader, sim, num_envs)
    _apply_mode(cfgs, mode, sim)
    runner = BaseRunner.create_with_env(cfgs, use_wandb=False)
    env = runner.env
    obs, _ = env.reset()
    shadows = None
    if mode == "shadow":
        # Loaded here, not at the top: this file also runs as the child of
        # a baseline sweep, under a tree where the shadow module (and the
        # compiled programs it imports) does not exist.
        shadows = importlib.import_module("jaxrlworld.scripts.diag.gates.preset_shadow").install_shadows(
            env, sim, no_reward_shadow.get(label)
        )
    build_s = time.perf_counter() - t_build

    gen = torch.Generator().manual_seed(seed)
    fp = {"rew": [], "rew_sq": [], "done": [], "obs": [], "obs_abs": []}
    times = []
    for k in range(steps):
        # Drawn on the CPU explicitly: a backend may have moved torch's
        # default device to the GPU, and the generator is a CPU one.
        actions = (torch.randn((env.num_envs, env.num_actions), generator=gen, device="cpu") * 0.5).to(env.device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        obs, rewards, terminated, truncated, _infos = env.step(actions)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
        if not _finite(rewards) or not _finite(obs):
            raise RuntimeError(f"non-finite reward or observation at step {k}")
        vec = _obs_vector(obs).double()
        fp["rew"].append(float(rewards.double().sum()))
        fp["rew_sq"].append(float((rewards.double() ** 2).sum()))
        fp["done"].append(int((terminated | truncated).sum()))
        fp["obs"].append(float(vec.sum()))
        fp["obs_abs"].append(float(vec.abs().sum()))
        if mode in ("compiled", "shadow") and k == warm:
            torch._dynamo.config.error_on_recompile = True
    times.sort()
    return {
        "build_s": build_s,
        "ms_per_step": times[len(times) // 2],
        "fingerprint": fp,
        "shadow": importlib.import_module("jaxrlworld.scripts.diag.gates.preset_shadow").shadow_report(shadows)
        if shadows is not None
        else None,
        # Which tree actually ran: a baseline run that resolved to the
        # installed package instead of --tree would compare HEAD to HEAD.
        "package": str(Path(jaxrlworld.__file__).resolve().parent),
    }


# ---------------------------------------------------------------------
# parent
# ---------------------------------------------------------------------


def _max_rel(new: dict, old: dict, steps: int) -> float:
    """Largest relative difference over the first ``steps`` steps of the
    reward / observation fingerprints (dones are compared as counts)."""
    a, b = new["fingerprint"], old["fingerprint"]
    worst = 0.0
    for key in ("rew", "obs", "obs_abs", "done"):
        for x, y in zip(a[key][:steps], b[key][:steps]):
            worst = max(worst, abs(x - y) / max(abs(y), 1e-6))
    return worst


def _floor_verdict(new: dict, ref: dict, rerun: dict, floor_of: str, tol: float, steps: int) -> str:
    """Judge ``new`` against a reference whose engine is not reproducible.

    ``rerun`` is a second run of the same code as ``new`` (``floor_of ==
    "new"``: eager vs baseline sweep) or as ``ref`` (``"reference"``:
    compiled vs eager sweep). The floor is the spread of that same-code
    pair, and the difference is the closest the two codes come — the
    noise (a termination flipping in one run) lands on either side.
    """
    if floor_of == "new":
        floor = _max_rel(new, rerun, steps)
        against = min(_max_rel(new, ref, steps), _max_rel(rerun, ref, steps))
    else:
        floor = _max_rel(ref, rerun, steps)
        against = min(_max_rel(new, ref, steps), _max_rel(new, rerun, steps))
    ok = against <= max(tol, 10.0 * floor)
    return f"{'ok' if ok else 'DIFFERS'} (vs reference {against:.1e}, run-to-run floor {floor:.1e}, {steps} steps)"


def _shadow_verdict(data: dict) -> str:
    parts = []
    differs = False
    for name, tol in _SHADOW_TOL.items():
        value = data["shadow"][name]
        if value is None:
            continue
        if isinstance(value, str):
            parts.append(f"{name} UNSUPPORTED ({value})")
            continue
        bad = value > tol
        differs |= bad
        terms = ""
        if bad and name == "reward":
            terms = (
                " ["
                + ", ".join(
                    f"{t} {v[0]:.1e} (abs {v[1]:.1e} / scale {v[2]:.1e})"
                    for t, v in data["shadow"]["reward_terms"].items()
                    if v[0] > tol
                )
                + "]"
            )
        parts.append(f"{name} {value:.1e}{' > ' + format(tol, '.0e') if bad else ''}{terms}")
    return f"{'DIFFERS' if differs else 'ok'} (" + ", ".join(parts) + ")"


def _print_detail(rows, out: Path, against: str | None, floor: str | None, steps: int) -> None:
    """Per-step relative differences of every cell that differs: which
    step the divergence starts at, and how it grows, next to the
    run-to-run spread — a step-0 difference is a changed computation, a
    late one growing exponentially is chaos on a last-bit change."""
    print()
    print("=" * 100)
    print(f"DETAIL of differing cells (per-step max rel diff, first {steps} steps)")
    print("=" * 100)
    for cell, status, _ms, verdict in rows:
        if "DIFFERS" not in verdict:
            continue
        label, sim = cell.split(":")
        name = f"{label}__{sim}.json"
        data = json.loads((out / name).read_text())
        print(f"  {cell}   package={data.get('package', '?')}")
        ref = json.loads((Path(against) / name).read_text()) if against and (Path(against) / name).exists() else None
        rerun = json.loads((Path(floor) / name).read_text()) if floor and (Path(floor) / name).exists() else None
        pairs = (
            ("new vs reference", data, ref),
            ("new vs rerun", data, rerun),
            ("reference vs rerun", ref, rerun),
        )
        for tag, x, y in pairs:
            if x is None or y is None:
                continue
            a, b = x["fingerprint"], y["fingerprint"]
            print(f"    {tag:<20} ({x.get('package', '?')} vs {y.get('package', '?')})")
            # One row per key: a difference that starts in ``rew`` alone
            # is the reward computation; one in ``obs`` is physics.
            for key in ("rew", "obs", "obs_abs", "done"):
                per_step = [
                    f"{abs(a[key][k] - b[key][k]) / max(abs(b[key][k]), 1e-6):.1e}"
                    for k in range(min(steps, len(a["rew"])))
                ]
                print(f"    {tag:<20} {key:<8}" + " ".join(per_step))
    print("=" * 100)


def _compare(new: dict, old: dict, exact: bool, tol: float, tol_steps: int) -> str:
    """One line: the verdict, the largest relative difference within the
    compared steps, and the first step at which anything differs at all
    — so a last-bit drift and a wrong term read as different things."""
    a, b = new["fingerprint"], old["fingerprint"]
    n = len(a["rew"]) if exact else tol_steps
    worst = 0.0
    first = None
    for key in ("rew", "obs", "obs_abs", "done"):
        for k, (x, y) in enumerate(zip(a[key][:n], b[key][:n])):
            if x != y and (first is None or k < first):
                first = k
            worst = max(worst, abs(x - y) / max(abs(y), 1e-6))
    if exact:
        verdict = "bitwise" if first is None else "DIFFERS"
    else:
        verdict = "ok" if worst <= tol else "DIFFERS"
    where = "" if first is None else f", first at step {first}"
    return f"{verdict} (max rel {worst:.1e} over {n} steps{where})"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("baseline", "eager", "compiled", "shadow"), default="compiled")
    ap.add_argument("--tree", default=None, help="package root put first on PYTHONPATH for the cells (baseline)")
    ap.add_argument(
        "--specs", default=None, help="module exposing PRESETS / NO_REWARD_SHADOW for presets outside this package"
    )
    ap.add_argument("--out", default="preset_sweep", help="directory for the per-cell json files")
    ap.add_argument("--against", default=None, help="a previous --out directory to compare fingerprints with")
    ap.add_argument("--exact", action="store_true", help="comparison must be bit-identical")
    ap.add_argument(
        "--floor",
        default=None,
        help="a second run of the SAME code (its --out directory); a cell passes when its difference from "
        "--against is within 10x its run-to-run difference. mjwarp-based backends are not bitwise reproducible.",
    )
    ap.add_argument(
        "--floor-of",
        choices=("new", "reference"),
        default=None,
        help="which side --floor re-runs; default: 'new' in eager mode, 'reference' in compiled mode",
    )
    ap.add_argument("--detail", action="store_true", help="per-step differences for every differing cell")
    ap.add_argument(
        "--compare-only",
        action="store_true",
        help="do not run cells whose result json already exists in --out; just compare them",
    )
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--tol-steps", type=int, default=10)
    ap.add_argument("--only", default=None, help="substring filter on the cell label")
    ap.add_argument("--sims", default=",".join(_SIMS))
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--warm", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cell", default=None, help="internal: 'label|loader|sim'")
    ap.add_argument("--result-json", default=None, help="internal")
    args = ap.parse_args()
    floor_of = args.floor_of or ("new" if args.mode == "eager" else "reference")

    if args.cell is not None:
        label, loader, sim = args.cell.split("|")
        if args.tree is not None:
            package = Path(jaxrlworld.__file__).resolve().parent
            if not package.is_relative_to(Path(args.tree).resolve()):
                raise RuntimeError(
                    f"--tree {args.tree} did not take precedence: jaxrlworld resolved to {package}. "
                    f"Expected {Path(args.tree).resolve() / 'jaxrlworld' / '__init__.py'} to exist."
                )
        result = run_cell(
            label,
            loader,
            sim,
            args.mode,
            args.num_envs,
            args.steps,
            args.warm,
            args.seed,
            _no_reward_shadow(_extra_specs(args.specs)),
        )
        Path(args.result_json).write_text(json.dumps(result))
        return 0

    specs = list(_PUBLIC)
    extra = _extra_specs(args.specs)
    if extra:
        specs += extra.PRESETS
    sims = tuple(args.sims.split(","))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    child_env = dict(os.environ)
    if args.tree:
        child_env["PYTHONPATH"] = str(Path(args.tree).resolve()) + os.pathsep + child_env.get("PYTHONPATH", "")

    rows = []
    for label, loader, spec_sims in specs:
        if args.only and args.only not in label:
            continue
        for sim in spec_sims:
            if sim not in sims:
                continue
            cell = f"{label}:{sim}"
            result_path = out / f"{label}__{sim}.json"
            log_path = out / f"{label}__{sim}.log"
            print(f"[{args.mode}] {cell} ...", end=" ", flush=True)
            t0 = time.perf_counter()
            rc = 0
            if not (args.compare_only and result_path.exists()):
                if result_path.exists():
                    result_path.unlink()
                with open(log_path, "w") as log:
                    proc = subprocess.run(
                        [
                            sys.executable,
                            str(Path(__file__).resolve()),
                            "--cell",
                            f"{label}|{loader}|{sim}",
                            "--mode",
                            args.mode,
                            "--result-json",
                            str(result_path),
                            "--num-envs",
                            str(args.num_envs),
                            "--steps",
                            str(args.steps),
                            "--warm",
                            str(args.warm),
                            "--seed",
                            str(args.seed),
                            *(["--tree", args.tree] if args.tree else []),
                            *(["--specs", args.specs] if args.specs else []),
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        env=child_env,
                    )
                rc = proc.returncode
            wall = time.perf_counter() - t0
            if result_path.exists():
                data = json.loads(result_path.read_text())
                verdict = ""
                if args.against:
                    ref = Path(args.against) / result_path.name
                    if ref.exists():
                        verdict = _compare(data, json.loads(ref.read_text()), args.exact, args.tol, args.tol_steps)
                        if args.floor:
                            # The same code twice: how far apart two runs
                            # land on their own. A difference from the
                            # reference within that is not a difference.
                            floor_path = Path(args.floor) / result_path.name
                            if floor_path.exists():
                                verdict = _floor_verdict(
                                    data,
                                    json.loads(ref.read_text()),
                                    json.loads(floor_path.read_text()),
                                    floor_of,
                                    args.tol,
                                    args.tol_steps,
                                )
                    else:
                        verdict = "no reference"
                elif args.mode == "shadow":
                    verdict = _shadow_verdict(data)
                rows.append((cell, "PASS", data["ms_per_step"], verdict))
                print(f"PASS  {data['ms_per_step']:6.2f} ms/step  {verdict}  ({wall:.0f}s)")
            else:
                proc = type("_rc", (), {"returncode": rc})()
                tail = ""
                try:
                    tail = log_path.read_text().strip().splitlines()[-1][:120]
                except (OSError, IndexError):
                    pass
                rows.append((cell, f"FAIL rc={proc.returncode}", float("nan"), tail))
                print(f"FAIL (rc={proc.returncode}, {wall:.0f}s)  {tail}")

    if args.detail:
        _print_detail(rows, out, args.against, args.floor, args.tol_steps)

    print()
    print("=" * 100)
    print(
        f"PRESET SWEEP  mode={args.mode}  num_envs={args.num_envs}  steps={args.steps}  specs={args.specs or 'built-in'}"
        + (f"  floor-of={floor_of}" if args.floor else "")
    )
    print("=" * 100)
    failed = 0
    differs = 0
    unsupported = 0
    for cell, status, ms, verdict in rows:
        print(f"  {cell:<34}{status:<12}{ms:8.2f} ms/step   {verdict}")
        failed += status != "PASS"
        differs += "DIFFERS" in verdict
        unsupported += "UNSUPPORTED" in verdict
    print("=" * 100)
    tail = f", {unsupported} reward-compile unsupported" if args.mode == "shadow" else ""
    print(f"  {len(rows)} cells: {len(rows) - failed} pass, {failed} fail, {differs} differ from reference{tail}")
    return 1 if (failed or differs) else 0


if __name__ == "__main__":
    raise SystemExit(main())
