"""Why a Go2 SAC checkpoint trained on Newton collapses on Genesis and MuJoCo.

``Go2SACNewtonConfig`` sets ``use_joint_limit_action=True``. Only the Newton
action builder reads that flag (``scale="joint_limit"``, ``clip=(-1, 1)``);
the Genesis and MuJoCo builders return the PPO mapping unconditionally
(``GO2_ACTION_SCALE``, ``clip=(-100, 100)``). Cross-sim eval rebuilds the
preset with ``sim_type`` swapped, so the policy that learned

    target = default + a * soft_half          (a in [-1, 1], soft_half ~0.9-2.3 rad)

is driven through

    raw    = a * 100                           (OffPolicyRunner rescales the tanh
                                                output to env.action_low/high,
                                                which are the clip bounds)
    target = default + clip(raw, +-100) * 0.33  (= default + 33 * a)

Every joint is commanded tens of radians past its hard limit, and the
``prev_actions`` observation reads ``raw`` (100x the training range), so
the obs normalizer is fed out-of-distribution input as well. PPO is immune:
its three builders agree (same scale, same clip), and the non-squashed
runner path only clips.

This diag proves it in three stages, each a stronger claim than the last:

  config  Reproduce the exact eval-time config resolution for ``--sim`` from
          a Newton SAC checkpoint's metadata and show the action config it
          yields is the PPO one; show the eval guard (observation parity)
          passes, so nothing in eval catches it.
  env     Build the env + runner for that config and push one probe action
          through the real post-processing chain; print per joint the
          target the policy meant versus the target the sim receives, with
          the hard limits next to them.
  eval    (``--policy_path``) Evaluate the checkpoint four ways and show the
          outcome follows the action mapping, not the simulator:
            newton  / resolved     reference (what ``--eval_sim newton`` runs)
            newton  / ppo          SAME simulator, the Genesis/MuJoCo mapping
            target  / resolved     what ``--eval_sim <target>`` runs
            target  / joint_limit  the target sim with the training mapping

The verdict reads each eval cell against its env cell: a mapping the env
stage shows to be the training one must walk, any other must collapse or
be refused by the evaluator's action-parity guard (``_check_action_parity``,
which exists because of this bug). Before the builder fix the ``resolved``
target cells collapse; after it they walk and the ``ppo`` control cell is
refused by the guard. Both runs end in CONFIRMED.

One backend per process (``assert_single_sim_loaded``); ``--all`` spawns a
subprocess per cell and prints the verdict table.

    # config + env stages, CPU is enough
    python -m jaxrlworld.scripts.diag.gates.check_sac_cross_sim_action_mapping --sim mujoco
    # the checkpoint A/B on the GPU box
    python -m jaxrlworld.scripts.diag.gates.check_sac_cross_sim_action_mapping --all --policy_path <ckpt_dir>
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import jax.numpy as jnp
import torch

from jaxrlworld.rl.configs.presets.go2.base import Go2FlatConfig
from jaxrlworld.rl.configs.presets.go2.newton.sac import Go2SACNewtonConfig
from jaxrlworld.rl.configs.robots.go2 import GO2_ACTION_SCALE
from jaxrlworld.rl.evals.evaluator import PolicyEvaluator, _check_observation_parity
from jaxrlworld.rl.evals.sim_initializers import resolve_cross_sim_config
from jaxrlworld.rl.runners import BaseRunner
from jaxrlworld.rl.utils.checkpoint import load_checkpoint_metadata
from jaxrlworld.rl.utils.jax_utils import jax_to_torch
from jaxrlworld.scripts.diag.gates.check_all_presets import _apply_mode

SIMS = ("newton", "genesis", "mujoco")
MAPPINGS = ("resolved", "joint_limit", "ppo")
PROBE = 0.5  # the policy output pushed through the chain, every joint
# A cell "walks" when it keeps at least this fraction of the reference
# return and "collapses" below it; the expected gap is 10-100x, so the
# threshold is not delicate.
WALK_FRACTION = 0.5
# An env cell whose applied/meant ratio is within this band carries the
# training mapping.
TRAINING_RATIO = (0.67, 1.5)
GUARD_MESSAGE = "Cross-sim eval action config differs from the checkpoint's"


def checkpoint_metadata(policy_path: str | None) -> dict:
    """The checkpoint's metadata, or the three fields the resolver reads
    from it, synthesized the way ``Go2SACNewtonConfig.build`` writes them.

    Synthesizing avoids building the Newton config here: that imports the
    Newton backend, and the env stage must hold a single backend.
    """
    if policy_path is not None:
        return load_checkpoint_metadata(policy_path)
    preset = Go2SACNewtonConfig()
    return {
        "config": {
            "preset_module": type(preset).__module__,
            "preset_class_name": type(preset).__name__,
            "preset_kwargs": preset._get_preset_kwargs(),
        }
    }


def resolve(sim: str, metadata: dict):
    """The config cross-sim eval builds for ``sim`` from that checkpoint.

    Newton is the checkpoint's own backend: same-sim eval reloads the
    saved config, which is the preset built as it was built for training.
    """
    if sim == "newton":
        return Go2SACNewtonConfig().build()
    return resolve_cross_sim_config(metadata, sim)


def apply_mapping(cfgs, mapping: str) -> None:
    """Rewrite the resolved action config to one of the two mappings."""
    if mapping == "resolved":
        return
    if mapping == "joint_limit":
        cfgs.action.action_scale = "joint_limit"
        cfgs.action.clip_actions = (-1.0, 1.0)
        return
    if mapping == "ppo":
        cfgs.action.action_scale = GO2_ACTION_SCALE
        cfgs.action.clip_actions = (-100.0, 100.0)
        return
    raise ValueError(f"unknown mapping {mapping!r}")


def describe_action(cfg) -> str:
    scale = cfg.action_scale if isinstance(cfg.action_scale, str) else f"dict({len(cfg.action_scale)} patterns)"
    return f"scale={scale} clip={cfg.clip_actions} offset={'default pose' if cfg.offset else cfg.offset}"


# ---------------------------------------------------------------- config stage


def stage_config(sim: str, metadata: dict) -> None:
    print(f"\n=== config: what cross-sim eval resolves for {sim} from a Go2SACNewtonConfig checkpoint ===")
    train = metadata["config"]
    if "action" in train:
        print(
            f"  checkpoint (newton)  : scale={train['action']['action_scale']} clip={tuple(train['action']['clip_actions'])}"
        )
    else:
        print("  checkpoint (newton)  : (synthesized metadata; the training mapping is printed by --sim newton)")
    cfgs = resolve(sim, metadata)
    print(f"  resolved ({sim:7s})  : {describe_action(cfgs.action)}")
    ppo = Go2FlatConfig(sim_type=sim).build()
    print(f"  PPO Go2Flat ({sim:7s}): {describe_action(ppo.action)}")
    same_as_ppo = (
        cfgs.action.action_scale == ppo.action.action_scale and cfgs.action.clip_actions == ppo.action.clip_actions
    )
    print(f"  resolved SAC action config == PPO action config: {same_as_ppo}")
    if "observation" in train:
        _check_observation_parity(train["observation"], cfgs.observation.recursive_to_dict())
        print("  eval's observation-parity guard: PASSES (it never looks at the action config)")
    if sim == "newton":
        print("  -> newton keeps the training mapping (joint_limit, clip +-1)")
    elif same_as_ppo:
        print(f"  -> {sim} builder ignores use_joint_limit_action: the SAC policy gets the PPO mapping")
    else:
        print(f"  -> {sim} builder honours use_joint_limit_action; the mapping is NOT the cause here")


# ---------------------------------------------------------------- env stage


def build_runner(sim: str, mapping: str, num_envs: int, metadata: dict) -> BaseRunner:
    cfgs = resolve(sim, metadata)
    apply_mapping(cfgs, mapping)
    cfgs.env.num_envs = num_envs
    cfgs.scene.num_envs = num_envs
    # The preset's 5M-transition replay buffer is sized for training.
    cfgs.algorithm.buffer_size = num_envs * 16
    _apply_mode(cfgs, "eager", sim)
    return BaseRunner.create_with_env(cfgs, use_wandb=False)


def stage_env(sim: str, mapping: str, num_envs: int, metadata: dict) -> dict:
    print(f"\n=== env: probe action a={PROBE:+.2f} on every joint through {sim} / {mapping} ===")
    runner = build_runner(sim, mapping, num_envs, metadata)
    env = runner.env
    env.reset()
    am = env.act_manager
    names = list(am.actuated_joint_names)
    lower, upper = am._get_joint_limits()
    mid, soft_half = am.soft_joint_limits_of(env.robot_entity_name)
    offset = am._offset

    a = torch.full((env.num_envs, env.num_actions), PROBE, dtype=torch.float32)
    raw = jax_to_torch(runner._process_action_for_env(jnp.asarray(a.numpy())), env.device)
    target = am.process_actions(raw)
    meant = offset + PROBE * soft_half  # what the Newton-trained policy meant by ``a``

    print(
        f"  runner.squash_output={runner.squash_output}  env.action_low/high = {env.action_low[0].item():+.0f} / "
        f"{env.action_high[0].item():+.0f}  -> runner multiplies the tanh output by {runner.action_scale[0]:.0f}"
    )
    print(f"  prev_actions observation reads the raw action: {raw[0, 0].item():+.3f} (training range: +-1)")
    print(
        f"  {'joint':16s} {'lower':>8s} {'upper':>8s} {'default':>8s} {'meant':>8s} {'applied':>9s} {'ratio':>7s}  limit"
    )
    worst_ratio = 0.0
    n_outside = 0
    for j, name in enumerate(names):
        lo, hi, off, m, t = (
            x[j].item() if x.dim() == 1 else x[0, j].item() for x in (lower, upper, offset, meant, target)
        )
        ratio = (t - off) / (m - off)
        outside = t < lo or t > hi
        n_outside += int(outside)
        worst_ratio = max(worst_ratio, abs(ratio))
        print(
            f"  {name:16s} {lo:+8.3f} {hi:+8.3f} {off:+8.3f} {m:+8.3f} {t:+9.3f} {ratio:7.1f}x  {'OUTSIDE' if outside else 'inside'}"
        )
    summary = {
        "sim": sim,
        "mapping": mapping,
        "raw_scale": float(raw[0, 0].item() / PROBE),
        "worst_ratio": worst_ratio,
        "joints_outside_hard_limits": n_outside,
        "num_joints": len(names),
    }
    if worst_ratio > 1.5 or worst_ratio < 0.67:
        print(
            f"  -> the sim applies {worst_ratio:.0f}x the joint excursion the policy meant; "
            f"{n_outside}/{len(names)} targets outside the hard limits"
        )
    else:
        print("  -> applied == meant: the training mapping")
    return summary


# ---------------------------------------------------------------- eval stage


def stage_eval(sim: str, mapping: str, policy_path: str, num_envs: int) -> dict:
    print(f"\n=== eval: {policy_path} on {sim} / {mapping} ===")
    overrides = {"env": {"num_envs": num_envs}, "algorithm": {"buffer_size": num_envs * 16}}
    if sim == "newton" and mapping == "resolved":
        eval_target = None  # same-sim eval: the saved config, as ``--eval_sim newton`` runs it
    elif mapping == "resolved":
        eval_target = sim  # the resolver, as ``--eval_sim <sim>`` runs it
    else:
        eval_target = resolve(sim, load_checkpoint_metadata(policy_path))
        apply_mapping(eval_target, mapping)
    try:
        evaluator = PolicyEvaluator(
            policy_path=policy_path,
            eval_target=eval_target,
            num_evals=1,
            record_video=False,
            save_data=False,
            use_rich_display=False,
            extra_overrides=overrides,
        )
    except ValueError as exc:
        # The evaluator's action-parity guard refusing a mapping that is not
        # the checkpoint's is an outcome of this experiment, not a failure.
        if not str(exc).startswith(GUARD_MESSAGE):
            raise
        print(f"  refused by the action-parity guard:\n{exc}")
        return {"sim": sim, "mapping": mapping, "refused": True}
    stats = evaluator.evaluate()
    return {
        "sim": sim,
        "mapping": mapping,
        "refused": False,
        "mean_return": stats["mean_return"],
        "mean_length": stats["mean_length"],
        "num_episodes": stats["num_episodes"],
    }


# ---------------------------------------------------------------- orchestration


def run_cell(args: list[str], out: Path) -> dict:
    cmd = [
        sys.executable,
        "-m",
        "jaxrlworld.scripts.diag.gates.check_sac_cross_sim_action_mapping",
        *args,
        "--out",
        str(out),
    ]
    print("\n$ " + " ".join(cmd[2:]), flush=True)
    subprocess.run(cmd, check=True)
    return json.loads(out.read_text())


def verdict(env_rows: list[dict], eval_rows: list[dict]) -> None:
    print("\n=== env stage ===")
    print(f"  {'sim':8s} {'mapping':12s} {'raw/a':>7s} {'applied/meant':>14s} {'outside limits':>15s}")
    for r in env_rows:
        print(
            f"  {r['sim']:8s} {r['mapping']:12s} {r['raw_scale']:7.0f} {r['worst_ratio']:14.1f} "
            f"{r['joints_outside_hard_limits']:>7d}/{r['num_joints']}"
        )
    if not eval_rows:
        return
    print("\n=== eval stage ===")
    training_mapping = {
        (r["sim"], r["mapping"]): TRAINING_RATIO[0] <= r["worst_ratio"] <= TRAINING_RATIO[1] for r in env_rows
    }
    ref = next(r for r in eval_rows if (r["sim"], r["mapping"]) == ("newton", "resolved"))
    print(f"  {'sim':8s} {'mapping':12s} {'mean return':>12s} {'mean length':>12s}  outcome")
    ok = True
    for r in eval_rows:
        expect_walk = training_mapping[(r["sim"], r["mapping"])]
        expected = "walks" if expect_walk else "collapses or refused"
        if r["refused"]:
            ok &= not expect_walk
            print(f"  {r['sim']:8s} {r['mapping']:12s} {'-':>12s} {'-':>12s}  refused by guard (expected {expected})")
            continue
        walks = r["mean_return"] >= WALK_FRACTION * ref["mean_return"]
        ok &= walks == expect_walk
        print(
            f"  {r['sim']:8s} {r['mapping']:12s} {r['mean_return']:12.2f} {r['mean_length']:12.1f}  "
            f"{'walks' if walks else 'collapses'} (expected {expected})"
        )
    print()
    if ok:
        print(
            "CONFIRMED: the outcome follows the action mapping, not the simulator. Every cell that carries the "
            "training mapping walks; every other mapping collapses or is refused by the action-parity guard."
        )
    else:
        print("NOT CONFIRMED: at least one cell contradicts the mapping explanation; read the rows above.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--sim", choices=SIMS, help="run the config and env stages (and eval with --policy_path) in-process"
    )
    ap.add_argument("--mapping", choices=MAPPINGS, default="resolved")
    ap.add_argument(
        "--stage",
        choices=("config", "env", "eval"),
        default=None,
        help="one stage only (default: config + env, or eval when --policy_path is given)",
    )
    ap.add_argument("--all", action="store_true", help="orchestrate every cell in subprocesses and print the verdict")
    ap.add_argument("--sims", nargs="+", choices=SIMS, default=list(SIMS), help="backends for --all")
    ap.add_argument("--policy_path", default=None, help="Go2SACNewtonConfig checkpoint dir; enables the eval stage")
    ap.add_argument("--num-envs", type=int, default=4, help="env stage")
    ap.add_argument("--eval-envs", type=int, default=256, help="eval stage")
    ap.add_argument("--out", default=None, help="write the stage's summary json here (used by --all)")
    args = ap.parse_args()

    if args.all:
        tmp = Path(tempfile.mkdtemp(prefix="sac_mapping_"))
        env_rows, eval_rows = [], []
        for sim in args.sims:
            mappings = ("resolved", "ppo") if sim == "newton" else ("resolved", "joint_limit")
            for mapping in mappings:
                cell = ["--sim", sim, "--mapping", mapping, "--stage", "env", "--num-envs", str(args.num_envs)]
                if args.policy_path is not None:
                    cell += ["--policy_path", args.policy_path]
                env_rows.append(run_cell(cell, tmp / f"env_{sim}_{mapping}.json"))
        if args.policy_path is not None:
            for sim in args.sims:
                mappings = ("resolved", "ppo") if sim == "newton" else ("resolved", "joint_limit")
                for mapping in mappings:
                    eval_rows.append(
                        run_cell(
                            [
                                "--sim",
                                sim,
                                "--mapping",
                                mapping,
                                "--stage",
                                "eval",
                                "--policy_path",
                                args.policy_path,
                                "--eval-envs",
                                str(args.eval_envs),
                            ],
                            tmp / f"eval_{sim}_{mapping}.json",
                        )
                    )
        verdict(env_rows, eval_rows)
        return 0

    if args.sim is None:
        ap.error("pass --sim <backend> or --all")
    stage = args.stage
    if stage is None:
        stage = "eval" if args.policy_path is not None else "both"
    metadata = checkpoint_metadata(args.policy_path)
    summary: dict = {}
    if stage in ("config", "both"):
        stage_config(args.sim, metadata)
    if stage in ("env", "both"):
        summary = stage_env(args.sim, args.mapping, args.num_envs, metadata)
    if stage == "eval":
        if args.policy_path is None:
            ap.error("--stage eval needs --policy_path")
        summary = stage_eval(args.sim, args.mapping, args.policy_path, args.eval_envs)
    if args.out is not None:
        Path(args.out).write_text(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
