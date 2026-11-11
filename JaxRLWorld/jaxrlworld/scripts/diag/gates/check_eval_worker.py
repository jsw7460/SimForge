"""Gate for the evaluation worker process (``runners/eval_worker.py``).

On a built-in backend, where the in-process evaluation also works, the
same policy is evaluated both ways and compared:

0. the eval config a worker rebuilds from the checkpoint (saved config
   restored onto a fresh preset build, then the in-training eval
   transform) against the in-process one, field by field: exact;
1. a fresh runner's policy, checkpointed, evaluated in process and by a
   worker (first request: the worker builds its env and runner);
2. after a few training iterations, re-checkpointed and evaluated both
   ways again (second request: the worker reloads weights only);
3. the trainer's own path, ``_run_evaluation_in_workers``, as a backend
   with ``one_instance_per_process`` takes it, on the same runner.

Both evaluations run the runner's deterministic policy on the same eval
config, but each owns its env and RNG stream and mjwarp-based backends
are not bitwise reproducible across processes, so the returns are
compared, not required equal: the gate fails on a missing or extra stat,
a non-finite value, a different episode count, or a failed worker, and
prints every stat side by side.

Usage (SimForge root)::

    jaxpy -m jaxrlworld.scripts.diag.gates.check_eval_worker --sim mujoco
    jaxpy -m jaxrlworld.scripts.diag.gates.check_eval_worker --sim newton --iters 3
"""

from __future__ import annotations

import argparse
import math
import os
import sys

from jaxrlworld.rl.configs.base_config import diff_config_dicts, flatten_leaf_paths
from jaxrlworld.rl.configs.eval_mode import in_training_eval_configs
from jaxrlworld.rl.configs.presets.go2.base import Go2FlatConfig
from jaxrlworld.rl.runners import BaseRunner
from jaxrlworld.rl.runners.eval_worker import EvalWorker, eval_worker_dir
from jaxrlworld.rl.utils.checkpoint import load_checkpoint_metadata, load_config_from_checkpoint


def compare(tag: str, inline: dict, worker: dict) -> bool:
    ok = True
    if set(inline) != set(worker):
        print(
            f"  [{tag}] FAIL stat keys differ: only inline {sorted(set(inline) - set(worker))}, "
            f"only worker {sorted(set(worker) - set(inline))}"
        )
        ok = False
    for key in sorted(set(inline) & set(worker)):
        a, b = float(inline[key]), float(worker[key])
        finite = math.isfinite(a) and math.isfinite(b)
        ok &= finite
        print(
            f"  [{tag}] {key:40s} inline {a:12.5f}  worker {b:12.5f}  diff {b - a:+.5f}{'' if finite else '  NON-FINITE'}"
        )
    if inline["eval/num_episodes"] != worker["eval/num_episodes"]:
        print(f"  [{tag}] FAIL episode counts differ")
        ok = False
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", default="mujoco", choices=("genesis", "newton", "mujoco"))
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--iters", type=int, default=2, help="training iterations between the two comparisons")
    ap.add_argument("--out", default="outputs/check_eval_worker")
    args = ap.parse_args()

    cfgs = Go2FlatConfig(sim_type=args.sim, num_envs=args.num_envs).build()
    cfgs.runner.output_dir = os.path.abspath(args.out)
    cfgs.runner.eval_interval = 0  # this gate triggers the evaluations itself
    runner = BaseRunner.create_with_env(cfgs, use_wandb=False)
    stamp = os.path.basename(runner.model_log_dir)
    worker = EvalWorker(None, eval_worker_dir(runner.model_log_dir, "gate"), "gate")
    ok = True
    try:
        # 0. the worker's eval config == the in-process one
        ckpt = os.path.join(runner.model_log_dir, "gate_checkpoint_0")
        runner._save_checkpoint_to(ckpt, 0)
        rebuilt = in_training_eval_configs(load_config_from_checkpoint(load_checkpoint_metadata(ckpt)))
        residual = diff_config_dicts(runner._create_eval_configs().recursive_to_dict(), rebuilt.recursive_to_dict())
        print(f"  [config] worker eval config vs in-process: {'identical' if not residual else 'DIFFERS'}")
        if residual:
            print(f"  [config] differing paths: {flatten_leaf_paths(residual)}")
            ok = False

        # 1. fresh policy: the worker builds its env and runner
        ok &= compare("fresh", runner._run_evaluation(), worker.evaluate(ckpt, runner.env.env_step_counter))

        # 2. trained a little: the worker reloads weights only
        runner.learn(num_learning_iterations=args.iters)
        ckpt = os.path.join(runner.model_log_dir, f"gate_checkpoint_{args.iters}")
        runner._save_checkpoint_to(ckpt, args.iters)
        ok &= compare("trained", runner._run_evaluation(), worker.evaluate(ckpt, runner.env.env_step_counter))

        # 3. the trainer's own worker path
        runner.env.one_instance_per_process = True
        runner._run_evaluation_in_workers(args.iters)
        logged = runner._last_eval_stats
        trainer_ok = logged is not None and logged["eval/iteration"] == args.iters
        print(
            f"  [trainer path] logged eval at iteration {logged and logged['eval/iteration']}: "
            f"{'ok' if trainer_ok else 'FAIL'}"
        )
        ok &= trainer_ok
    finally:
        worker.close()
        runner.close()
    print(f"{'PASS' if ok else 'FAIL'}  check_eval_worker sim={args.sim} run={stamp}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
