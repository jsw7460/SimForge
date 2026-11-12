"""Does a change to the env leave every step's observation, reward and done bit for bit the same?

A rewrite of how the managers address the simulator (which index form a
setter takes, how many calls a reset makes) must leave the numbers that
reach the policy untouched. This records the whole stream one env
produces under a fixed seed and a fixed action sequence, and compares a
second recording against it.

The run exercises every write path: the robot starts from staggered
episode ages so resets happen throughout the window, the actions are a
seeded uniform sequence so some episodes end by falling, and the preset's
interval events fire at their own rate. Each step records the observation
groups, the reward and each of its terms, the termination flags and the
root pose read back from the simulator (so a write that lands in the wrong place shows up
even when no observation term reads it).

Record on the code before the change, then compare on the code after it:

    jaxpy -m jaxrlworld.scripts.diag.parity.check_env_step_stream --sim genesis --record /tmp/g1_stream.pt
    jaxpy -m jaxrlworld.scripts.diag.parity.check_env_step_stream --sim genesis --compare /tmp/g1_stream.pt

The comparison prints, per stream, the largest absolute difference and the
first step it appears at, and exits non-zero when any stream differs by
more than ``--atol`` (zero by default: the change is meant to be exact).
Record twice on the same code first to see that the backend itself is
deterministic at this size before reading anything into a difference.
"""

from __future__ import annotations

import argparse
import importlib
import sys

import torch

from jaxrlworld.rl.evals.sim_initializers import get_initializer

_SIM_NAMES = {"genesis": "Genesis", "newton": "Newton", "mujoco": "MujocoEnv"}


def build_env(config: str, sim: str, num_envs: int):
    mod_path, cls_name = config.split(":", 1)
    cfg_cls = getattr(importlib.import_module(mod_path), cls_name)
    cfgs = cfg_cls(sim_type=sim, num_envs=num_envs, seed=0).build()
    return get_initializer(_SIM_NAMES[sim]).init_environment(cfgs)


def record_stream(env, steps: int, seed: int, action_scale: float) -> dict[str, torch.Tensor]:
    gen = torch.Generator(device=env.device)
    gen.manual_seed(seed)
    env.reset()
    env.termination_manager.episode_length_buf = torch.randint(
        0, int(env.max_episode_length), (env.num_envs,), device=env.device, generator=gen
    )
    rd = env.get_entity_data("robot")
    streams: dict[str, list[torch.Tensor]] = {}
    for _ in range(steps):
        actions = (torch.rand((env.num_envs, env.num_actions), device=env.device, generator=gen) * 2 - 1) * action_scale
        obs, rew, terminated, truncated, extras = env.step(actions)
        frame = {f"obs/{key}": value for key, value in obs.items()}
        frame["reward"] = rew
        for name, value in extras["rewards_per_type"].items():
            frame[f"reward/{name}"] = value
        frame["terminated"] = terminated
        frame["truncated"] = truncated
        frame["root_pos_w"] = rd.root_link_pos_w
        frame["root_quat_w"] = rd.root_link_quat_w
        frame["joint_pos"] = rd.joint_pos
        frame["joint_vel"] = rd.joint_vel
        for key, value in frame.items():
            streams.setdefault(key, []).append(value.detach().clone())
    return {key: torch.stack(values).cpu() for key, values in streams.items()}


def compare_streams(reference: dict[str, torch.Tensor], current: dict[str, torch.Tensor], atol: float) -> bool:
    ok = True
    for key in sorted(reference.keys() ^ current.keys()):
        print(f"stream {key!r} is only in the {'reference' if key in reference else 'current'} recording")
    print(f"{'max |diff|':>12}  {'first step':>10}  {'differing envs':>14}  stream")
    for key, ref in reference.items():
        if key not in current:
            continue
        cur = current[key]
        if ref.shape != cur.shape:
            print(f"{'shape':>12}  {'':>10}  {'':>14}  {key}: {tuple(ref.shape)} vs {tuple(cur.shape)}")
            ok = False
            continue
        diff = (ref.float() - cur.float()).abs()
        max_diff = float(diff.max()) if diff.numel() else 0.0
        if max_diff > atol:
            ok = False
            per_step = diff.flatten(1).amax(dim=1) if diff.ndim > 1 else diff
            first = int((per_step > atol).nonzero()[0])
            per_env = diff.flatten(2).amax(dim=2) if diff.ndim > 2 else diff
            n_envs = int((per_env > atol).any(dim=0).sum()) if per_env.ndim == 2 else -1
            print(f"{max_diff:12.3e}  {first:10d}  {n_envs:14d}  {key}")
        else:
            print(f"{max_diff:12.3e}  {'-':>10}  {'-':>14}  {key}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="jaxrlworld.rl.configs.presets.g1_29dof.base:G1FlatConfig")
    ap.add_argument("--sim", default="genesis", choices=tuple(_SIM_NAMES))
    ap.add_argument("--num-envs", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0, help="seed of the episode ages and the action sequence")
    ap.add_argument("--action-scale", type=float, default=0.5, help="actions are uniform in +-this")
    ap.add_argument("--atol", type=float, default=0.0)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--record", metavar="PATH", help="run and save the stream here")
    mode.add_argument("--compare", metavar="PATH", help="run and compare against the stream saved here")
    args = ap.parse_args()

    env = build_env(args.config, args.sim, args.num_envs)
    streams = record_stream(env, args.steps, args.seed, args.action_scale)
    print(f"[stream] {args.config} sim={args.sim} num_envs={args.num_envs} steps={args.steps}")
    resets = int(streams["terminated"].sum() + streams["truncated"].sum())
    print(f"[stream] episodes ended in the window: {resets}")

    if args.record:
        torch.save(streams, args.record)
        print(f"[stream] recorded {len(streams)} streams to {args.record}")
        return
    reference = torch.load(args.compare)
    if compare_streams(reference, streams, args.atol):
        print(f"[stream] IDENTICAL to {args.compare} (atol {args.atol})")
        return
    print(f"[stream] DIFFERS from {args.compare}")
    sys.exit(1)


if __name__ == "__main__":
    main()
