"""Runtime guard for the lazy-import contract.

Each simulator backend (Genesis, Newton, MuJoCo via mjlab) takes substantial
GPU memory at import time (Taichi/Warp/MuJoCo compilation, kernel caches,
PyTorch CUDA contexts). Only ONE backend should be loaded per training /
evaluation process. The lazy-import design in ``rlworld/rl/envs/`` enforces
this at the source level — but refactors can silently break it (a stray
top-level import in a shared module pulls in a second backend).

This module provides a runtime check: scan ``sys.modules`` for the heavy
backend packages; if 2+ are present, raise. Call it once at the entry point
(``BaseRunner.create_with_env``) so any leakage surfaces immediately.

Set ``JAXRLWORLD_ALLOW_MULTI_SIM=1`` to bypass — required for diag /
cross-sim scripts that intentionally build multiple sims sequentially.
"""

from __future__ import annotations

import os
import sys

# Top-level package names that pull in heavy backend state.
# ``mjlab`` is the canonical MuJoCo entry (bare ``mujoco`` is a small
# dependency used universally and not worth gating on).
_HEAVY_SIM_PKGS: tuple[str, ...] = ("genesis", "newton", "mjlab")

_BYPASS_ENV_VAR = "JAXRLWORLD_ALLOW_MULTI_SIM"


def assert_single_sim_loaded() -> None:
    """Raise if more than one simulator backend is loaded in ``sys.modules``.

    Bypassed when ``JAXRLWORLD_ALLOW_MULTI_SIM`` is set to a truthy value.
    """
    if os.environ.get(_BYPASS_ENV_VAR):
        return
    loaded = [pkg for pkg in _HEAVY_SIM_PKGS if pkg in sys.modules]
    if len(loaded) >= 2:
        raise RuntimeError(
            f"Lazy-import violation: {len(loaded)} simulator backends are "
            f"loaded simultaneously ({loaded}). Each takes substantial GPU "
            f"memory; only one should be loaded per training/eval process. "
            f"This usually means a shared module imported one backend at "
            f"module top-level when another was already in use — grep "
            f"top-level `import {loaded[0]}` / `import {loaded[-1]}` in "
            f"shared (cross-sim) code. "
            f"Set {_BYPASS_ENV_VAR}=1 to bypass (diag/cross-sim scripts only)."
        )
