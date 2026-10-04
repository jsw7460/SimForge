"""Run every training entry point for a few iterations and report what breaks.

``check_all_presets`` builds each preset's env and steps it; it never
trains. This runs the training scripts themselves, end to end, each in
its own process exactly as a user launches it: config build and command-line
overrides, runner and algorithm construction, rollouts, updates, the
in-training evaluation, checkpoint saves and logging. It is the check to
run after a simulator pull or any change to the runner / algorithm /
config layers.

Entry points: every ``jaxrlworld/scripts/<task>/<sim>/<name>.py`` (one
run per script) and the scripts that take ``--sim`` (``yam_lift``). Each
run is shortened through the config's own override parser::

    runner.max_iterations=<iters>  env.num_envs=<n>  runner.output_dir=<out>/<run>
    runner.save_interval=<iters/2>  runner.eval_interval=<iters/2>  runner.latest_checkpoint_interval=<iters/2>
    runner.upload_checkpoint=False                 (W&B off: WANDB_MODE=disabled)

A run passes when the process exits 0, its log holds no traceback, and
the final checkpoint (``checkpoint_<iters-1>``) was written -- which
the runner does only after the last iteration finishes.

Run from anywhere; the scripts run from the SimForge root::

    python -m jaxrlworld.scripts.diag.gates.check_train_smoke
    python -m jaxrlworld.scripts.diag.gates.check_train_smoke --sims newton --only go2
    python -m jaxrlworld.scripts.diag.gates.check_train_smoke --list
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import jaxrlworld

SIMS = ("genesis", "newton", "mujoco")
SCRIPTS_DIR = Path(jaxrlworld.__file__).resolve().parent / "scripts"
# Entry points that pick the simulator with a flag instead of a directory.
SIM_FLAG_SCRIPTS = ("yam_lift/train.py", "yam_lift/train_vision.py")
# Flags those scripts take for the env count instead of env.num_envs.
NUM_ENVS_FLAG = {"yam_lift/train.py": "--num-envs", "yam_lift/train_vision.py": "--num-envs"}


@dataclass(frozen=True)
class Run:
    label: str
    sim: str
    script: Path
    args: tuple[str, ...]


def find_simforge_root() -> Path:
    """The directory whose ``JaxRLWorld/jaxrlworld`` is this package: the
    scripts resolve their assets from there."""
    package = SCRIPTS_DIR.parent
    for ancestor in SCRIPTS_DIR.parents:
        if (ancestor / "JaxRLWorld" / "jaxrlworld").resolve() == package:
            return ancestor
    raise FileNotFoundError(f"no ancestor of {SCRIPTS_DIR} holds JaxRLWorld/jaxrlworld")


def discover(num_envs: int) -> list[Run]:
    runs = []
    for sim in SIMS:
        for script in sorted(SCRIPTS_DIR.glob(f"*/{sim}/*.py")):
            if script.name == "__init__.py" or script.parts[-3] == "diag":
                continue
            task = script.parts[-3]
            runs.append(Run(f"{task}/{script.stem}", sim, script, (f"env.num_envs={num_envs}",)))
        for rel in SIM_FLAG_SCRIPTS:
            script = SCRIPTS_DIR / rel
            runs.append(Run(rel.removesuffix(".py"), sim, script, ("--sim", sim, NUM_ENVS_FLAG[rel], str(num_envs))))
    return runs


def final_checkpoint(out: Path, iters: int) -> Path | None:
    hits = sorted(out.rglob(f"checkpoint_{iters - 1}"))
    return hits[0] if hits else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--num-envs", type=int, default=128)
    ap.add_argument("--sims", default=",".join(SIMS))
    ap.add_argument("--only", default=None, help="substring filter on '<task>/<name>'")
    ap.add_argument("--out", default="train_smoke", help="directory for per-run logs and outputs")
    ap.add_argument("--timeout", type=int, default=1800, help="seconds per run")
    ap.add_argument("--list", action="store_true", help="print the runs and stop")
    args = ap.parse_args()

    root = find_simforge_root()
    sims = args.sims.split(",")
    runs = [r for r in discover(args.num_envs) if r.sim in sims and (args.only is None or args.only in r.label)]
    if args.list:
        for r in runs:
            print(f"{r.label}:{r.sim}  {r.script.relative_to(root)} {' '.join(r.args)}")
        print(f"{len(runs)} runs")
        return 0

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    half = max(args.iters // 2, 1)
    env = {
        **os.environ,
        "WANDB_MODE": "disabled",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "PYTHONUNBUFFERED": "1",
    }
    results = []
    print(f"{len(runs)} runs, {args.iters} iterations, {args.num_envs} envs each; logs in {out}\n")
    for r in runs:
        name = f"{r.label.replace('/', '__')}__{r.sim}"
        run_out = out / name
        log = out / f"{name}.log"
        cmd = [
            sys.executable,
            str(r.script),
            *r.args,
            f"runner.max_iterations={args.iters}",
            f"runner.output_dir={run_out}",
            f"runner.save_interval={half}",
            f"runner.eval_interval={half}",
            f"runner.latest_checkpoint_interval={half}",
            "runner.upload_checkpoint=False",
        ]
        t0 = time.perf_counter()
        try:
            with open(log, "w") as fh:
                rc = subprocess.run(
                    cmd, cwd=root, env=env, stdout=fh, stderr=subprocess.STDOUT, timeout=args.timeout
                ).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
        secs = time.perf_counter() - t0
        text = log.read_text(errors="replace")
        ckpt = final_checkpoint(run_out, args.iters)
        problems = []
        if rc != 0:
            problems.append(f"exit {rc}")
        if "Traceback (most recent call last)" in text:
            problems.append("traceback in log")
        if ckpt is None:
            problems.append(f"no checkpoint_{args.iters - 1}")
        status = "PASS" if not problems else "FAIL"
        # The exception line, not the trailer some frameworks print after it
        # (JAX's "internal frames removed" note, a CUDA abort banner).
        lines = [line for line in text.splitlines() if line.strip()]
        errors = [
            line
            for line in lines
            if re.search(r"(Error|Exception|CUDA_ERROR_\w+)\b", line) and not line.startswith(" ")
        ]
        last = (errors or lines or [""])[-1]
        detail = "" if not problems else f"{', '.join(problems)} | {last[:150]}"
        print(f"[{status}] {r.label}:{r.sim}  ({secs:.0f}s)  {detail}", flush=True)
        results.append((r, status, secs, detail))

    failed = [x for x in results if x[1] != "PASS"]
    print("\n" + "=" * 100)
    for r, status, secs, detail in results:
        print(f"  {status}  {r.label + ':' + r.sim:45s} {secs:6.0f}s  {detail}")
    print("=" * 100)
    print(f"  {len(results)} runs: {len(results) - len(failed)} pass, {len(failed)} fail   (logs: {out})")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
