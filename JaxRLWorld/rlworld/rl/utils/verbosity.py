"""Console verbosity for the summaries printed while an env/runner is built.

Building an environment used to print several hundred lines before a run said
anything of its own: a Rich table per manager (observation groups, actions,
rewards, terminations, commands, events, contact sensors), the policy's
parameter tree, and one line per Warp kernel module loaded. Useful once while
wiring a preset up; noise on every training run, eval and diagnostic after
that, and enough of it to push the actual output out of a terminal scrollback.

They are therefore **off by default** and restored with::

    JAXRLWORLD_BUILD_SUMMARY=1 python -m rlworld.scripts...

One switch covers all of them, so a preset-wiring session gets the full picture
back with a single variable rather than three.
"""

from __future__ import annotations

import os

_TRUTHY = ("1", "true", "on", "yes")


def build_summary_enabled() -> bool:
    """Whether build-time console summaries should be printed.

    Returns:
        True only if ``JAXRLWORLD_BUILD_SUMMARY`` is set to a truthy value.
    """
    return os.environ.get("JAXRLWORLD_BUILD_SUMMARY", "0").strip().lower() in _TRUTHY
