"""Train the YAM arm to lift the cube from a wrist depth camera.

    jaxpy -m rlworld.scripts.yam_lift.train_vision
    jaxpy -m rlworld.scripts.yam_lift.train_vision --sim newton

The actor is not told where the cube is — it sees a 32x32 depth image
from the D405 on the wrist, and the two terms that used to hand it the
cube's position are gone. The critic keeps them.

All three backends. Genesis draws through Madrona's batch renderer, so
it needs ``gs_madrona`` installed.

Anything else is a config path, written as ``path=value`` with no
leading dashes — the config's own override parser reads them::

    env.num_envs=2048 algorithm.actor_lr=5e-4
"""

from __future__ import annotations

import argparse
import sys

from rlworld.rl.configs.presets.yam_lift.vision import YamLiftVisionConfig
from rlworld.rl.runners import BaseRunner


def main() -> int:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--sim", default="mujoco", choices=("mujoco", "newton", "genesis"))
    ap.add_argument("--difficulty", default="fixed", choices=("fixed", "dynamic"))
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--resolution", type=int, default=32, help="Square camera resolution, pixels.")
    ap.add_argument("--iterations", type=int, default=30000)
    args, rest = ap.parse_known_args()
    # Hand the remainder to the config's own override parser.
    sys.argv = [sys.argv[0], *rest]

    cfg = YamLiftVisionConfig(
        sim_type=args.sim,
        num_envs=args.num_envs,
        difficulty=args.difficulty,
        camera_width=args.resolution,
        camera_height=args.resolution,
        max_iterations=args.iterations,
        run_name=f"YamLiftVision_{args.sim.capitalize()}_{args.difficulty}",
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
