"""K1 domain-randomization landing diag — does the sim2real DR actually
take effect, per backend, and does ``reset_dr`` re-sample every reset?

Motivation: DR has a history of silently NOT landing (friction never
reaching the solver; mjlab per-env expand bug; small-env diags passing on
OOB). Before spending a long training run on the expanded K1 DR (kp/kd/
armature/friction re-randomized per episode + command delay + sensor
delay), verify at scale that every knob varies per env AND changes across
resets on all three simulators.

Checks per sim (verbose — dumps every measurement on the first run):

  A. ACTUATOR DR (sim-agnostic, explicit IdealPD/DelayedPD path):
     - kp / kd per-env spread (std > 0 ⇒ ``dr_kp`` / ``dr_kd`` landed)
     - kd on ankle joints wider than others (``dr_ankle_kd`` overlay)
  B. ACTION LATENCY (DelayedPDActuator):
     - actuator is DelayedPD, per-env ``_delay`` covers [0, max_delay]
     - resampled on reset
  C. OBS DELAY (observation manager DelayBuffer):
     - delay buffer registered on the actor proprio terms
     - per-env current lag covers [0, max_lag]
  D. reset_dr RE-SAMPLE:
     - snapshot kp/kd/delay, call env.reset(), confirm the values CHANGED
       (reset_dr fired) — the whole point of reset_dr over startup.
  E. MODEL-LEVEL DR (best-effort, per-sim; non-fatal if the read path
     differs): armature per-env spread (startup DR).

Run per cell on the training box::

    jaxpy -m rlworld.scripts.diag.k1.k1_dr_landing_diag --sim newton
    jaxpy -m rlworld.scripts.diag.k1.k1_dr_landing_diag            # all three
"""

from __future__ import annotations

import argparse

_SIMS = ("genesis", "newton", "mujoco")
_SIM_KEY = {"genesis": "Genesis", "newton": "Newton", "mujoco": "MujocoEnv"}
_SETTLE = 10


def _stage(msg: str) -> None:
    print(f"  · {msg}", flush=True)


def _stats(t) -> dict:
    t = t.detach().float().reshape(-1)
    return {
        "min": float(t.min()),
        "max": float(t.max()),
        "mean": float(t.mean()),
        "std": float(t.std()),
    }


def _fmt(s: dict) -> str:
    return f"min={s['min']:.4f} max={s['max']:.4f} mean={s['mean']:.4f} std={s['std']:.5f}"


def run_cell(sim: str, num_envs: int, seed: int) -> dict:
    import torch

    torch.manual_seed(seed)
    _stage(f"cell start: {sim} num_envs={num_envs} seed={seed}")

    from rlworld.rl.configs.presets.k1_joystick.base import K1JoystickConfig
    from rlworld.rl.evals.sim_initializers import get_initializer

    preset = K1JoystickConfig(sim_type=sim, num_envs=num_envs, seed=seed)
    ankle_pats = preset.robot.ankle_joint_patterns
    cfgs = preset.build()
    env = get_initializer(_SIM_KEY[sim]).init_environment(cfgs)
    env.reset()
    _stage("env built + first reset (startup + reset_dr applied)")

    am = env.act_manager
    names = list(am.actuated_joint_names)
    ankle_cols = [i for i, n in enumerate(names) if any(_re_full(p, n) for p in ankle_pats)]

    out: dict = {"sim": sim, "num_envs": num_envs, "n_joints": len(names)}

    # ── A/B: actuator kp/kd/delay ─────────────────────────────────────
    def read_gains():
        kp = torch.zeros(num_envs, len(names), device=env.device)
        kd = torch.zeros(num_envs, len(names), device=env.device)
        delays = None
        act_type = None
        for actuator, joint_idx in am.actuators:
            kp[:, joint_idx] = actuator.stiffness.to(kp.device)
            kd[:, joint_idx] = actuator.damping.to(kd.device)
            act_type = type(actuator).__name__
            d = getattr(actuator, "_delay", None)
            if d is not None:
                delays = d.detach().clone()
        return kp, kd, delays, act_type

    kp0, kd0, delay0, act_type = read_gains()
    out["actuator_type"] = act_type
    out["kp"] = _stats(kp0)
    out["kd_all"] = _stats(kd0)
    if ankle_cols:
        out["kd_ankle"] = _stats(kd0[:, ankle_cols])
        non_ankle = [i for i in range(len(names)) if i not in ankle_cols]
        out["kd_non_ankle"] = _stats(kd0[:, non_ankle])
    if delay0 is not None:
        out["action_delay"] = _stats(delay0)
        out["action_delay_hist"] = torch.bincount(delay0.long().reshape(-1)).tolist()

    # ── C: obs delay buffers ──────────────────────────────────────────
    obs_delay = {}
    delay_bufs = getattr(env.obs_manager, "_group_obs_term_delay_buffer", {})
    for term_name, buf in delay_bufs.get("actor", {}).items():
        lags = buf.current_lags
        obs_delay[term_name] = {"lag": _stats(lags), "hist": torch.bincount(lags.long().reshape(-1)).tolist()}
    out["obs_delay_terms"] = obs_delay

    # ── D: reset_dr re-sample ─────────────────────────────────────────
    env.reset()
    kp1, kd1, delay1, _ = read_gains()
    out["resample_kp_changed_frac"] = float((kp0 != kp1).float().mean())
    out["resample_kd_changed_frac"] = float((kd0 != kd1).float().mean())
    if delay0 is not None and delay1 is not None:
        out["resample_delay_changed_frac"] = float((delay0 != delay1).float().mean())

    # ── E: model armature (best-effort, per-sim) ──────────────────────
    try:
        out["armature_model"] = _read_model_armature(sim, env)
    except Exception as e:  # noqa: BLE001 — diagnostic best-effort
        out["armature_model"] = f"<unavailable: {type(e).__name__}: {e}>"

    # A few steps to confirm nothing NaNs with the new DR active.
    zero = torch.zeros((num_envs, env.num_actions), device=env.device)
    nan_seen = False
    for _ in range(_SETTLE):
        obs, *_ = env.step(zero)
        actor = obs["actor"] if isinstance(obs, dict) else obs
        if torch.isnan(actor).any():
            nan_seen = True
            break
    out["nan_after_steps"] = nan_seen
    _stage(f"cell done: {sim}")
    return out


def _re_full(pattern: str, name: str) -> bool:
    import re

    return re.fullmatch(pattern, name) is not None


def _read_model_armature(sim: str, env) -> dict:
    """Per-env armature spread from the live sim model (startup DR)."""
    import torch

    sm = env.scene_manager
    if sim == "newton":
        import warp as wp

        arm = wp.to_torch(sm.robot_view.get_attribute("joint_armature", sm.model))
        return _stats(arm)
    if sim == "mujoco":
        # mjlab batched model: dof_armature shape (num_envs, nv) or (nv,)
        arm = torch.as_tensor(sm.model.dof_armature)
        return _stats(arm)
    if sim == "genesis":
        arm = torch.as_tensor(sm["robot"].get_dofs_armature())
        return _stats(arm)
    return {}


def _print_cell(r: dict) -> None:
    print(f"\n===== {r['sim'].upper()} (num_envs={r['num_envs']}, {r['n_joints']} joints) =====")
    print(f"  actuator type      : {r['actuator_type']}")
    print(f"  A. kp   (all)      : {_fmt(r['kp'])}   [std>0 ⇒ dr_kp landed]")
    print(f"     kd   (all)      : {_fmt(r['kd_all'])}   [std>0 ⇒ dr_kd landed]")
    if "kd_ankle" in r:
        print(f"     kd   (ankle)    : {_fmt(r['kd_ankle'])}   [wider ⇒ dr_ankle_kd]")
        print(f"     kd   (non-ankle): {_fmt(r['kd_non_ankle'])}")
    if "action_delay" in r:
        print(f"  B. action delay    : {_fmt(r['action_delay'])}  hist(ticks)={r['action_delay_hist']}")
    else:
        print("  B. action delay    : <no _delay tensor — actuator is not DelayedPD>")
    if r["obs_delay_terms"]:
        print("  C. obs delay (actor terms):")
        for term, d in r["obs_delay_terms"].items():
            print(f"       {term:12s} lag {_fmt(d['lag'])}  hist={d['hist']}")
    else:
        print("  C. obs delay       : <no delay buffers registered>")
    print(
        f"  D. reset_dr resample: kp {r['resample_kp_changed_frac']*100:.1f}% changed, "
        f"kd {r['resample_kd_changed_frac']*100:.1f}% changed"
        + (f", delay {r['resample_delay_changed_frac']*100:.1f}% changed" if "resample_delay_changed_frac" in r else "")
    )
    print(
        f"  E. armature (model) : {r['armature_model'] if isinstance(r['armature_model'], str) else _fmt(r['armature_model'])}"
    )
    print(f"  NaN after {_SETTLE} steps : {r['nan_after_steps']}")

    # Verdicts
    ok = (
        r["kp"]["std"] > 1e-6
        and r["kd_all"]["std"] > 1e-6
        and r["resample_kp_changed_frac"] > 0.5
        and not r["nan_after_steps"]
    )
    print(f"  VERDICT            : {'PASS' if ok else 'CHECK'}")


def main() -> int:
    ap = argparse.ArgumentParser(description="K1 DR landing / reset_dr resample diag.")
    ap.add_argument("--sim", choices=_SIMS, help="Single backend (default: all).")
    ap.add_argument("--num_envs", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sims = [args.sim] if args.sim else list(_SIMS)
    results = []
    for sim in sims:
        try:
            results.append(run_cell(sim, args.num_envs, args.seed))
        except Exception as e:  # noqa: BLE001
            import traceback

            print(f"\n[{sim}] FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()

    for r in results:
        _print_cell(r)
    print()
    return 0 if len(results) == len(sims) else 1


if __name__ == "__main__":
    raise SystemExit(main())
