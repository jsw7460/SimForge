"""Stable-Baselines3 on any (algorithm, task) cell of the comparison.

The per-cell launchers next to this file exist so a single comparison
can be started by name and read at a glance. A sweep over every cell
would need one file per combination, so it uses this instead; the
parity contract in ``_common.py`` is the same either way.

Run:  python rlworld/scripts/benchmark/sb3_compare/sb3_run.py --algo sac --task hopper
"""

from __future__ import annotations

import argparse

from rlworld.scripts.benchmark.sb3_compare._common import ALGOS, TASKS, run_sb3


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--algo", choices=ALGOS, required=True)
    ap.add_argument("--task", choices=sorted(TASKS), required=True)
    args = ap.parse_args()
    run_sb3(args.algo, args.task)


if __name__ == "__main__":
    main()
