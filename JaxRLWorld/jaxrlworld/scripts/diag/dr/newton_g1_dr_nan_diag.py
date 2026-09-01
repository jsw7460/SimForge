"""Newton g1_29dof training-NaN bisect: which event term poisons the state.

Context: g1_29dof Newton training NaNs from iteration 0, while a
mujoco-trained policy EVALUATES cleanly on Newton.  The evaluator disables
``reset_dr`` and ``interval`` events (``PolicyEvaluator._apply_eval_defaults``),
so the suspects are exactly the terms training runs and eval does not:

    reset_dr: randomize_encoder_bias / randomize_body_com / randomize_joint_friction
    interval: push_robot

Each cell builds the g1 flat Newton env with a chosen subset of event terms
DISABLED (the same ``setattr(event_cfg, name, None)`` mechanism the
evaluator uses), then:

    1. reset() and — when the body-com term is active — dumps the actual
       ``model.body_com`` deviation from the frozen baseline (is the write
       sane, i.e. within the configured +/-0.03 range, and finite?)
    2. rolls N random-action control steps, checking joint_vel / root vel /
       actor obs / reward for finiteness after every step
    3. reports the first non-finite step and which quantity broke

Cells run in subprocesses for isolation; raw logs go to a sibling
``*_logs/`` dir and the aggregated report to a .txt file.

Usage (GPU box):
    python -m jaxrlworld.scripts.diag.dr.newton_g1_dr_nan_diag
    python -m jaxrlworld.scripts.diag.dr.newton_g1_dr_nan_diag --num-envs 4096
    python -m jaxrlworld.scripts.diag.dr.newton_g1_dr_nan_diag --cells full,no_body_com

Rough-terrain mode (--rough): builds the env with use_rough_terrain=True.
On rough the contact path differs from flat — use_mujoco_contacts=False,
so contacts come from Newton's own ``model.collide()`` (CollisionPipeline)
and are then consumed by the mjwarp solver. That adds TWO more silently-
overflowing buffers on top of mjwarp's nacon pool: the Newton rigid-
contact buffer (``rigid_contact_max``) and the mjwarp per-world
constraint rows (``nefc`` vs ``njmax``). All three are measured per step.

    python -m jaxrlworld.scripts.diag.dr.newton_g1_dr_nan_diag --rough \
        --num-envs 4096 --num-steps 300 --cells act_050,full_100
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_MODULE = "jaxrlworld.scripts.diag.dr.newton_g1_dr_nan_diag"

# Round 1 (event-term bisect) showed EVERY cell non-finite, including
# eval_like: the DR/interval terms are innocent — the Newton g1 physics
# itself diverges under random actions.  Round 2 characterizes that
# divergence: all cells run event-free (eval_like) and sweep the action
# magnitude; "full_100" keeps every event on as the training reference.
# cell name -> (event terms to disable, action scale)
_CELLS: dict[str, tuple[tuple[str, ...], float]] = {
    "zero_action": (("@reset_dr", "@interval"), 0.0),
    "act_025": (("@reset_dr", "@interval"), 0.25),
    "act_050": (("@reset_dr", "@interval"), 0.5),
    "act_100": (("@reset_dr", "@interval"), 1.0),
    "full_100": ((), 1.0),
}


def _stage(msg: str) -> None:
    print(f"[STAGE] {msg}", flush=True)


# ── Child ───────────────────────────────────────────────────────────


def _finite(t) -> bool:
    import torch

    return bool(torch.isfinite(t).all().item())


def _check_ccd_scratch(num_envs: int, nconmax: int, ccd_iterations: int) -> None:
    """Refuse an nconmax whose GJK/EPA scratch will not fit.

    mjwarp sizes its EPA workspace off the contact pool rather than off the
    contacts that occur: ``naccdmax`` defaults to ``num_envs * nconmax`` and
    ``convex_narrowphase`` allocates ``naccdmax x (10 + 2 * ccd_iterations)``
    vec3s from it. At 50 CCD iterations that is 1320 bytes per contact slot
    per environment, so raising nconmax to buy measurement headroom costs
    gigabytes: 4096 envs at nconmax 2000 asks for 10.8 GB and dies inside
    the collision driver, several frames deep, as a bare allocation failure.

    Predicted here instead, before the scene is built, with the arithmetic
    shown — so the fix (fewer envs, or a smaller ceiling) is obvious rather
    than something to bisect.
    """
    import torch

    row_vec3 = 10 + 2 * ccd_iterations
    scratch = num_envs * nconmax * row_vec3 * 12
    free, total = torch.cuda.mem_get_info()
    print(
        f"[INFO] EPA scratch: num_envs {num_envs} x nconmax {nconmax} x {row_vec3} vec3 x 12 B "
        f"= {scratch / 2**30:.2f} GiB (device free {free / 2**30:.2f} / {total / 2**30:.2f} GiB)",
        flush=True,
    )
    # Two thirds, not all of it: the model, the policy and every other
    # solver buffer still have to fit alongside this one allocation.
    if scratch > 0.66 * free:
        raise RuntimeError(
            f"nconmax={nconmax} at num_envs={num_envs} needs {scratch / 2**30:.2f} GiB of EPA "
            f"scratch, against {free / 2**30:.2f} GiB free. Lower --num-envs or --nconmax: the "
            f"cost is linear in both. num_envs={num_envs} tops out near "
            f"nconmax={int(0.66 * free / (num_envs * row_vec3 * 12))}."
        )


def _load_policy(checkpoint: str, cfgs, env):
    """Deterministic action from a trained checkpoint, as torch.

    Deterministic on purpose: the question is what demand the learned
    gait places on the contact buffers, and exploration noise would blur
    the peak this is trying to find.
    """
    from jaxrlworld.rl.algorithms import get_runner_class
    from jaxrlworld.rl.utils.jax_utils import jax_to_torch, torch_to_jax
    from jaxrlworld.rl.utils.wandb_checkpoint import resolve_checkpoint_path

    # Through the concrete runner class, as ``BaseRunner.create_with_env``
    # does. ``BaseRunner.load_checkpoint`` is the abstract declaration and
    # its body is ``pass``, so calling it on the base returns None and the
    # first attribute access on the result is what fails.
    runner_cls = get_runner_class(cfgs.algorithm.algorithm_name)
    runner = runner_cls.load_checkpoint(
        checkpoint_path=resolve_checkpoint_path(checkpoint),
        cfgs=cfgs,
        env=env,
        use_wandb=False,
    )
    alg = runner.alg

    def act(obs_dict):
        action = alg.act(
            alg.ActInput(torch_to_jax(obs_dict["actor"]), torch_to_jax(obs_dict["critic"])),
            deterministic=True,
        )
        return jax_to_torch(runner._process_action_for_env(action), runner.device)

    return act


def run_cell(
    cell: str,
    num_envs: int,
    num_steps: int,
    seed: int,
    nconmax: int,
    njmax: int,
    rough: bool,
    checkpoint: str | None = None,
) -> dict:
    import torch
    import warp as wp

    torch.manual_seed(seed)
    disable, action_scale = _CELLS[cell]
    _stage(
        f"cell start: {cell} (disable={list(disable)}, action_scale={action_scale}) "
        f"num_envs={num_envs} nconmax={nconmax} njmax={njmax} rough={rough}"
    )

    from jaxrlworld.rl.configs.base_config import iter_terms
    from jaxrlworld.rl.configs.events.event_term_config import EventTermConfig
    from jaxrlworld.rl.configs.presets.g1_29dof.base import G1FlatConfig

    cfgs = G1FlatConfig(sim_type="newton", num_envs=num_envs, seed=seed, use_rough_terrain=rough).build()
    # Budget overrides for overflow-free demand measurement (overflowed
    # contacts are silently skipped and the resulting NaN cuts the
    # rollout short). The live fields are solver_cfg.nconmax/njmax —
    # the scene manager builds SolverMuJoCo from those.
    if nconmax > 0:
        cfgs.scene.solver_cfg.nconmax = nconmax
    if njmax > 0:
        cfgs.scene.solver_cfg.njmax = njmax
    _check_ccd_scratch(num_envs, cfgs.scene.solver_cfg.nconmax, cfgs.scene.solver_cfg.ccd_iterations)

    disabled_names: list[str] = []
    for name, term in list(iter_terms(cfgs.event, EventTermConfig).items()):
        if name in disable or f"@{term.mode}" in disable:
            setattr(cfgs.event, name, None)
            disabled_names.append(name)
    _stage(f"config built (disabled terms: {disabled_names})")

    from jaxrlworld.rl.evals.sim_initializers.newton import NewtonInitializer

    env = NewtonInitializer().init_environment(cfgs)
    _stage("env built")

    # A trained policy, when one is given. The budgets in the preset were
    # sized from RANDOM-action rollouts, and random flailing is not an upper
    # bound on the contact demand of a gait: a policy that has learned to
    # walk rough terrain lands its feet hard and puts both of them, plus
    # knees and hands, on the heightfield at once. That is the state a
    # late-training NaN comes from, and no random rollout visits it.
    policy = _load_policy(checkpoint, cfgs, env) if checkpoint else None
    _stage(f"policy: {'checkpoint ' + checkpoint if policy else 'random actions'}")

    env.reset()
    torch.cuda.synchronize()
    _stage("reset done")

    # ── body_com write sanity (always dumped for visibility) ─────────
    from jaxrlworld.rl.envs.utils.newton.body_cache import get_cache

    cache = get_cache(env)
    model = env.scene_manager.model
    body_com = wp.to_torch(model.body_com).reshape(env.num_envs, cache.bodies_per_env, 3)
    baseline = env._dr_baselines.body_com
    dev = (body_com - baseline).abs()
    body_com_finite = _finite(body_com)
    max_dev = float(dev.max().item())
    n_perturbed_bodies = int((dev.amax(dim=(0, 2)) > 0).sum().item())
    print(
        f"[INFO] body_com: finite={body_com_finite}  max|dev from baseline|={max_dev:.4e}  "
        f"bodies perturbed={n_perturbed_bodies}/{cache.bodies_per_env}",
        flush=True,
    )

    rd = env.get_robot_data()
    joint_names = list(env.act_manager.actuated_joint_names)
    mjw_data = env.scene_manager.solver.mjw_data
    naconmax = int(mjw_data.naconmax)
    njmax_cap = int(mjw_data.njmax)
    # Newton-side rigid contact buffer (the collide() output the mjwarp
    # solver consumes on rough where use_mujoco_contacts=False). Sized
    # by CollisionPipeline rigid_contact_max; overflow is silent.
    nt_contacts = env.scene_manager.contacts
    rigid_cap = int(nt_contacts.rigid_contact_point_id.shape[0])
    peak_rigid_count = 0
    peak_nefc_world = 0
    first_bad_step = -1
    bad_quantity = ""
    peak_nacon = 0
    peak_ncon_world = 0
    trace: list[tuple[int, float, str]] = []  # (step, max|joint_vel|, argmax joint)
    for k in range(num_steps):
        if policy is None:
            a = torch.empty((num_envs, env.num_actions), device=env.device).uniform_(-1.0, 1.0) * action_scale
        else:
            a = policy(env.get_observation())
        obs, rew, _term, _trunc, _infos = env.step(a)
        # Contact-slot demand: nacon records the TRUE demand even when the
        # pool overflows (writes beyond naconmax are skipped, the counter
        # is not clamped).  Per-world peak from the written contacts.
        nacon = int(wp.to_torch(mjw_data.nacon)[0].item())
        peak_nacon = max(peak_nacon, nacon)
        n_written = min(nacon, naconmax)
        if n_written > 0:
            worldid = wp.to_torch(mjw_data.contact.worldid)[:n_written]
            peak_ncon_world = max(peak_ncon_world, int(torch.bincount(worldid, minlength=num_envs).max().item()))
        peak_rigid_count = max(peak_rigid_count, int(wp.to_torch(nt_contacts.rigid_contact_count)[0].item()))
        peak_nefc_world = max(peak_nefc_world, int(wp.to_torch(mjw_data.nefc).max().item()))
        jv_abs = rd.joint_vel.abs()
        jv_flat = torch.nan_to_num(jv_abs, nan=float("inf"))
        flat_idx = int(jv_flat.argmax().item())
        peak_joint = joint_names[flat_idx % jv_abs.shape[1]]
        finite_mask = torch.isfinite(jv_abs)
        jv_max = float(jv_abs[finite_mask].max().item()) if finite_mask.any() else float("nan")
        trace.append((k, jv_max, peak_joint))
        checks = (
            ("joint_vel", rd.joint_vel),
            ("root_lin_vel", rd.root_link_lin_vel_w),
            ("actor_obs", obs["actor"]),
            ("reward", rew),
        )
        for qname, tensor in checks:
            if not _finite(tensor):
                first_bad_step = k
                bad_quantity = qname
                n_bad_envs = int((~torch.isfinite(tensor).reshape(num_envs, -1).all(dim=1)).sum().item())
                print(f"[INFO] first non-finite at step {k}: {qname}  (bad envs: {n_bad_envs}/{num_envs})", flush=True)
                print("[INFO] max|joint_vel| trajectory over the last 15 steps (growth pattern):", flush=True)
                for st, vmax, jname in trace[-15:]:
                    print(f"[INFO]   step {st:>3}: max|joint_vel| = {vmax:>12.4e}  (peak joint: {jname})", flush=True)
                break
        if first_bad_step >= 0:
            break
        if (k + 1) % 25 == 0:
            _stage(f"step {k + 1}/{num_steps} clean (max|joint_vel| so far {max(v for _s, v, _j in trace):.3e})")

    ok = first_bad_step < 0 and body_com_finite
    _stage(f"cell done: {'CLEAN' if ok else f'NON-FINITE ({bad_quantity} @ step {first_bad_step})'}")
    peak_v = max((v for _s, v, _j in trace if v == v), default=float("nan"))
    overflowed = peak_nacon > naconmax
    rigid_overflowed = peak_rigid_count >= rigid_cap
    nefc_overflowed = peak_nefc_world >= njmax_cap
    print(
        f"[INFO] contact demand: peak nacon={peak_nacon} (pool naconmax={naconmax}, "
        f"overflow={overflowed})  peak per-world ncon={peak_ncon_world}",
        flush=True,
    )
    print(
        f"[INFO] newton rigid contacts: peak count={peak_rigid_count} (cap rigid_contact_max={rigid_cap}, "
        f"saturated={rigid_overflowed})",
        flush=True,
    )
    print(
        f"[INFO] mjwarp constraint rows: peak per-world nefc={peak_nefc_world} (cap njmax={njmax_cap}, "
        f"saturated={nefc_overflowed})",
        flush=True,
    )
    return {
        "cell": cell,
        "num_envs": num_envs,
        "action_scale": action_scale,
        # What drove the rollout belongs in the result, not only in the log.
        # A contact-demand number means something different depending on it:
        # random flailing and a trained gait visit different states, and the
        # whole point of the checkpoint path is that the preset's budgets
        # were sized from the former.
        "driver": "checkpoint" if policy is not None else "random",
        "peak_joint_vel": peak_v,
        "nconmax_used": nconmax,
        "naconmax": naconmax,
        "peak_nacon": peak_nacon,
        "peak_ncon_world": peak_ncon_world,
        "overflowed": overflowed,
        "rough": rough,
        "rigid_contact_cap": rigid_cap,
        "peak_rigid_count": peak_rigid_count,
        "rigid_overflowed": rigid_overflowed,
        "njmax_cap": njmax_cap,
        "peak_nefc_world": peak_nefc_world,
        "nefc_overflowed": nefc_overflowed,
        "disabled": disabled_names,
        "body_com_finite": body_com_finite,
        "body_com_max_dev": max_dev,
        "bodies_perturbed": n_perturbed_bodies,
        "steps_clean": first_bad_step if first_bad_step >= 0 else num_steps,
        "first_bad_step": first_bad_step,
        "bad_quantity": bad_quantity,
        "ok": ok,
    }


# ── Parent ──────────────────────────────────────────────────────────


def run_parent(args) -> int:
    out_path = Path(args.out).resolve()
    log_dir = out_path.parent / (out_path.stem + "_logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    cells = [c.strip() for c in args.cells.split(",")] if args.cells else list(_CELLS)
    results: list[dict] = []
    for cell in cells:
        log_path = log_dir / f"{cell}.log"
        result_path = log_dir / f"{cell}.json"
        if result_path.exists():
            result_path.unlink()
        print(f"[bisect] running cell {cell} ...", flush=True)
        t0 = time.time()
        with open(log_path, "w") as lf:
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    _MODULE,
                    "--cell",
                    cell,
                    "--result-json",
                    str(result_path),
                    "--num-envs",
                    str(args.num_envs),
                    "--num-steps",
                    str(args.num_steps),
                    "--seed",
                    str(args.seed),
                    "--nconmax",
                    str(args.nconmax),
                    "--njmax",
                    str(args.njmax),
                    *(["--rough"] if args.rough else []),
                    *(["--checkpoint", args.checkpoint] if args.checkpoint else []),
                ],
                stdout=lf,
                stderr=subprocess.STDOUT,
                env={**os.environ, "CUDA_LAUNCH_BLOCKING": "1"},
            )
        elapsed = time.time() - t0
        if result_path.exists():
            data = json.loads(result_path.read_text())
        else:
            data = {"cell": cell, "ok": False, "bad_quantity": "CRASH (see log)", "first_bad_step": -2}
        results.append(data)
        verdict = "CLEAN" if data["ok"] else f"BAD ({data['bad_quantity']} @ step {data.get('first_bad_step')})"
        print(f"[bisect]   -> {verdict} ({elapsed:.0f}s)", flush=True)

    lines: list[str] = []
    lines.append("=" * 100)
    lines.append("Newton g1_29dof DR NaN bisect")
    lines.append("=" * 100)
    lines.append(f"num_envs: {args.num_envs}   steps/cell: {args.num_steps}   seed: {args.seed}")
    lines.append(f"cell logs: {log_dir}")
    lines.append("")
    lines.append(f"{'cell':<20}{'verdict':<12}{'first bad step':<16}{'quantity':<14}{'body_com max dev':<18}detail")
    lines.append("-" * 100)
    for d in results:
        verdict = "CLEAN" if d["ok"] else "BAD"
        dev = f"{d.get('body_com_max_dev', float('nan')):.3e}" if "body_com_max_dev" in d else "-"
        lines.append(
            f"{d['cell']:<20}{verdict:<12}{str(d.get('first_bad_step', '?')):<16}"
            f"{d.get('bad_quantity', '') or '-':<14}{dev:<18}"
            f"driver={d.get('driver', '?')} scale={d.get('action_scale', '?')} "
            f"peak|jv|={d.get('peak_joint_vel', float('nan')):.2e} "
            f"peak_nacon={d.get('peak_nacon', '?')} peak_ncon/world={d.get('peak_ncon_world', '?')} "
            f"overflow={d.get('overflowed', '?')} "
            f"rigid={d.get('peak_rigid_count', '?')}/{d.get('rigid_contact_cap', '?')} "
            f"(sat={d.get('rigid_overflowed', '?')}) "
            f"nefc/world={d.get('peak_nefc_world', '?')}/{d.get('njmax_cap', '?')} "
            f"(sat={d.get('nefc_overflowed', '?')}) disabled={d.get('disabled', '?')}"
        )
    lines.append("")
    lines.append("[Reading] zero_action BAD -> gross instability (standing still explodes);")
    lines.append("          clean below some scale -> divergence needs violent inputs (limit/contact blowup);")
    lines.append("          the velocity trace shows gradual growth vs single-step jump to inf.")
    worst_world = max((d.get("peak_ncon_world", 0) for d in results), default=0)
    worst_total = max((d.get("peak_nacon", 0) for d in results), default=0)
    per_world_avg = worst_total / max(args.num_envs, 1)
    rec = max(worst_world, int(per_world_avg + 0.999))
    lines.append("")
    lines.append(
        f"[nconmax recommendation] worst per-world ncon={worst_world}, "
        f"worst total nacon={worst_total} ({per_world_avg:.1f}/env avg) "
        f"-> suggest nconmax >= {int(rec * 1.5 + 0.999)} (peak x1.5 headroom)"
    )
    worst_rigid = max((d.get("peak_rigid_count", 0) for d in results), default=0)
    rigid_cap = max((d.get("rigid_contact_cap", 0) for d in results), default=0)
    worst_nefc = max((d.get("peak_nefc_world", 0) for d in results), default=0)
    njcap = max((d.get("njmax_cap", 0) for d in results), default=0)
    lines.append(
        f"[rough buffers] newton rigid contacts peak={worst_rigid} / cap={rigid_cap}; "
        f"mjwarp per-world nefc peak={worst_nefc} / njmax={njcap}. "
        f"Any sat=True above = silent-drop overflow: raise that budget (peak x1.5 headroom)."
    )
    report = "\n".join(lines)
    out_path.write_text(report + "\n")
    print()
    print(report)
    print(f"\nReport written to: {out_path}")
    return 0 if all(d["ok"] for d in results) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", default=None, help="internal: run one cell")
    ap.add_argument("--result-json", default=None, help="internal: child result path")
    ap.add_argument("--cells", default=None, help="comma-separated subset of cells")
    ap.add_argument("--num-envs", type=int, default=1024)
    ap.add_argument("--num-steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--nconmax", type=int, default=512, help="solver_cfg nconmax override for overflow-free measurement"
    )
    ap.add_argument("--njmax", type=int, default=0, help="solver_cfg njmax override (0 = preset value)")
    ap.add_argument("--rough", action="store_true", help="build with use_rough_terrain=True")
    ap.add_argument(
        "--checkpoint",
        default=None,
        help="Drive with this trained policy instead of random actions (local dir or wandb run path).",
    )
    ap.add_argument("--out", default="newton_g1_dr_nan_diag.txt")
    args = ap.parse_args()

    if args.cell is not None:
        result = run_cell(
            args.cell,
            args.num_envs,
            args.num_steps,
            args.seed,
            args.nconmax,
            args.njmax,
            args.rough,
            args.checkpoint,
        )
        Path(args.result_json).write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return 0

    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
