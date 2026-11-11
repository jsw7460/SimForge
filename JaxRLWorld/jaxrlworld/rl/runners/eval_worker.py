"""In-training evaluation in a separate process.

A backend whose simulator allows one env per process
(``World.one_instance_per_process``) cannot hold the training env and the
evaluation env side by side, so the runner hands evaluation to worker
processes instead: one per evaluation env (the default one and each
held-out motion set), started on first use and kept for the run.

The hand-off is the checkpoint. At each evaluation the trainer writes a
checkpoint and sends its path with the training env clock; the worker
rebuilds the run's config from it the way the offline evaluator does
(:func:`~jaxrlworld.rl.utils.checkpoint.load_config_from_checkpoint`,
which raises unless the restored config matches the saved one), turns it
into the in-training evaluation config
(:func:`~jaxrlworld.rl.configs.eval_mode.in_training_eval_configs`, the
same transform the in-process evaluation applies), builds the env and the
run's own runner class on it once, and on every request loads the new
weights, mirrors the clock and runs the runner's own evaluation loop.
The trainer waits for the result, so the log lands at the same iteration
as an in-process evaluation.

Workers are started with ``spawn`` (CUDA is already initialized in the
trainer) and inherit its environment, so ``XLA_PYTHON_CLIENT_PREALLOCATE``
and ``CUDA_VISIBLE_DEVICES`` carry over. A worker failure is raised in the
trainer with the worker's traceback.
"""

from __future__ import annotations

import atexit
import multiprocessing as mp
import os
import traceback
from dataclasses import dataclass
from typing import Any

from jaxrlworld.rl.configs.eval_mode import in_training_eval_configs
from jaxrlworld.rl.utils.checkpoint import (
    load_checkpoint_metadata,
    load_config_from_checkpoint,
    runner_class_from_metadata,
)

_JOIN_TIMEOUT_S = 60.0


@dataclass(frozen=True)
class _EvalRequest:
    checkpoint_dir: str
    env_step_counter: int


class EvalWorker:
    """One evaluation env in its own process.

    Args:
        heldout_motion_files: ``None`` for the default evaluation env, or
            the motion files of a held-out set (see
            ``RunnerConfig.eval_extra_motion_files``).
        output_dir: Directory the worker's runner may write its log
            folders into (kept inside the training run's directory).
        name: Label used in the process name and in errors.
    """

    def __init__(self, heldout_motion_files: tuple[str, ...] | None, output_dir: str, name: str):
        self.name = name
        ctx = mp.get_context("spawn")
        self._conn, child_conn = ctx.Pipe()
        # Not daemonic: a daemonic process may not start children, and the
        # worker's own libraries (torch.compile workers, renderers) may.
        self._process = ctx.Process(
            target=_worker_main,
            args=(child_conn, heldout_motion_files, output_dir),
            name=f"eval-worker-{name}",
        )
        self._process.start()
        child_conn.close()
        # Registered after multiprocessing's own exit handler, so it runs
        # first: that handler joins non-daemonic children, which would wait
        # forever on a worker still blocked on its next request.
        atexit.register(self.close)

    def evaluate(self, checkpoint_dir: str, env_step_counter: int) -> dict[str, Any]:
        """Evaluate the policy in ``checkpoint_dir``; returns the eval stats."""
        self._conn.send(_EvalRequest(checkpoint_dir, env_step_counter))
        try:
            status, payload = self._conn.recv()
        except EOFError as exc:
            self._process.join(timeout=_JOIN_TIMEOUT_S)
            raise RuntimeError(
                f"Evaluation worker {self.name!r} died without a reply (exit code {self._process.exitcode})."
            ) from exc
        if status != "ok":
            raise RuntimeError(f"Evaluation worker {self.name!r} failed:\n{payload}")
        return payload

    def close(self) -> None:
        if not self._process.is_alive():
            return
        self._conn.send(None)
        self._process.join(timeout=_JOIN_TIMEOUT_S)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join()


def _build_runner(checkpoint_dir: str, heldout_motion_files: tuple[str, ...] | None, output_dir: str):
    metadata = load_checkpoint_metadata(checkpoint_dir)
    cfgs = load_config_from_checkpoint(metadata)
    eval_cfgs = in_training_eval_configs(cfgs, heldout_motion_files=heldout_motion_files)
    if eval_cfgs is None:
        raise ValueError("Held-out motion files were given but the preset has no 'motion' command term.")
    eval_cfgs.runner.output_dir = output_dir
    runner_class = runner_class_from_metadata(metadata)
    env = runner_class._create_env_from_config(eval_cfgs)
    runner = runner_class(env=env, cfgs=eval_cfgs, use_wandb=False)
    # The runner's evaluation entry point looks its evaluation env up
    # lazily; here it is the runner's own env.
    runner._eval_env = env
    return runner, metadata


def _worker_main(conn, heldout_motion_files: tuple[str, ...] | None, output_dir: str) -> None:
    runner = None
    try:
        while True:
            request = conn.recv()
            if request is None:
                return
            try:
                if runner is None:
                    runner, metadata = _build_runner(request.checkpoint_dir, heldout_motion_files, output_dir)
                else:
                    metadata = load_checkpoint_metadata(request.checkpoint_dir)
                runner.alg.load_train_state(request.checkpoint_dir, metadata)
                runner.env.env_step_counter = request.env_step_counter
                if heldout_motion_files is None:
                    stats = runner._run_evaluation()
                else:
                    stats = runner._evaluate_env(runner.env)
            except Exception:
                conn.send(("error", traceback.format_exc()))
                return
            conn.send(("ok", stats))
    finally:
        conn.close()


def eval_worker_dir(model_log_dir: str, name: str) -> str:
    """Where worker ``name`` keeps its runner's log folders."""
    return os.path.join(model_log_dir, "eval_workers", name)
