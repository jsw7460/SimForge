"""Train the YAM arm to lift the cube from a wrist depth camera.

    jaxpy -m rlworld.scripts.yam_lift.train_vision
    jaxpy -m rlworld.scripts.yam_lift.train_vision --difficulty dynamic

The actor is not told where the cube is — it sees a 32x32 depth image
from the D405 on the wrist, and the two terms that used to hand it the
cube's position are gone. The critic keeps them.

mjlab only for now: Newton and Genesis have camera sensors, but nothing
in this repo renders them yet.

Anything else is a config path, e.g.::

    --env.num_envs 4096 --runner.max_iterations 3000
"""

from __future__ import annotations

import argparse
import sys

from rlworld.rl.configs.presets.yam_lift.vision import YamLiftVisionConfig
from rlworld.rl.runners import BaseRunner


def main() -> int:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--difficulty", default="fixed", choices=("fixed", "dynamic"))
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--resolution", type=int, default=32, help="Square camera resolution, pixels.")
    args, rest = ap.parse_known_args()
    # Hand the remainder to the config's own override parser.
    sys.argv = [sys.argv[0], *rest]

    cfg = YamLiftVisionConfig(
        sim_type="mujoco",
        num_envs=args.num_envs,
        difficulty=args.difficulty,
        camera_width=args.resolution,
        camera_height=args.resolution,
        run_name=f"YamLiftVision_Mujoco_{args.difficulty}",
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
