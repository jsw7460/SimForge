"""Train the YAM arm to lift the cube to a commanded point.

    python -m jaxrlworld.scripts.yam_lift.train --sim mujoco
    python -m jaxrlworld.scripts.yam_lift.train --sim mujoco --difficulty dynamic

``--difficulty fixed`` (the default) aims at one point every episode.
Worth starting there: a policy that cannot solve a single goal will not
solve a distribution of them, and the failure is much easier to read —
a flat curve on one goal is a task or reward problem, whereas a flat
curve on a distribution could be either that or the spread being too
wide.

Anything else is a config path, written as ``path=value`` with no
leading dashes — the config's own override parser reads them::

    env.num_envs=8192 runner.max_iterations=3000
"""

from __future__ import annotations

import argparse
import sys

from jaxrlworld.rl.configs.presets.yam_lift.base import YamLiftConfig
from jaxrlworld.rl.runners import BaseRunner


def main() -> int:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--sim", default="mujoco", choices=("genesis", "newton", "mujoco"))
    ap.add_argument("--difficulty", default="fixed", choices=("fixed", "dynamic"))
    ap.add_argument("--num-envs", type=int, default=16_384)
    ap.add_argument(
        "--run-past-success",
        action="store_true",
        help="Keep the episode going after the cube reaches the goal, instead of ending "
        "there. Ending is the default and makes an episode one attempt. Comparing the two "
        "settings, use Success Rate and Mean Episode Length: the per-term reward means are "
        "per STEP, so running past success inflates them by adding held-at-the-goal steps.",
    )
    args, rest = ap.parse_known_args()
    # Hand the remainder to the config's own override parser.
    sys.argv = [sys.argv[0], *rest]

    suffix = {"newton": "Newton", "genesis": "Genesis", "mujoco": "Mujoco"}[args.sim]
    cfg = YamLiftConfig(
        sim_type=args.sim,
        num_envs=args.num_envs,
        difficulty=args.difficulty,
        end_on_success=not args.run_past_success,
        run_name=f"YamLift_{suffix}_{args.difficulty}" + ("_runpast" if args.run_past_success else ""),
    )
    cfgs_for_run = cfg.build().with_cli_overrides()

    runner = BaseRunner.create_with_env(cfgs_for_run)
    runner.learn(
        num_learning_iterations=cfgs_for_run.runner.max_iterations,
        init_at_random_ep_len=cfgs_for_run.runner.init_at_random_ep_len,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
