"""Warp console verbosity for the warp-backed backends (Newton, mjlab).

Warp logs one line per kernel module it loads — roughly 45 of them for a
locomotion scene, all at ``LOG_INFO`` — plus a device banner. That is build
noise, so it follows the same switch as the manager tables and the policy
parameter dump: see :mod:`jaxrlworld.rl.utils.verbosity`.

Importing this module pulls in ``warp``, so only the Newton and MuJoCo
backends (which import it anyway) may import it. A Genesis-only process must
stay warp-free — see ``lazy_import_check``.
"""

from __future__ import annotations

import warp as wp

from jaxrlworld.rl.utils.verbosity import build_summary_enabled


def configure_warp_logging() -> None:
    """Quiet Warp's per-module load timings unless build summaries are on.

    Called from the warp-backed envs' ``__init__`` — early enough to cover
    scene construction and kernel compilation, which is where the bulk of the
    lines come from. The banner Warp prints when it first initializes a device
    can precede this (it fires on ``import newton``), so it is left alone.
    """
    if build_summary_enabled():
        return
    wp.config.log_level = wp.LOG_WARNING
