"""JaxRLWorld on any (algorithm, task) cell of the comparison.

Counterpart to ``sb3_run.py``; see it for why the sweep uses these
rather than the per-cell launchers.

Run:  jaxpy rlworld/scripts/benchmark/sb3_compare/jrw_run.py --algo sac --task hopper
"""

from __future__ import annotations

import argparse

from rlworld.scripts.benchmark.sb3_compare._common import ALGOS, TASKS, run_jrw


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--algo", choices=ALGOS, required=True)
    ap.add_argument("--task", choices=sorted(TASKS), required=True)
    args = ap.parse_args()
    run_jrw(args.algo, args.task)


if __name__ == "__main__":
    main()
