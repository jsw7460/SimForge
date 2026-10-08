"""Where does a getup policy's head bobbing come from?

A trained K1 getup policy nods its head side to side after standing up in the
interactive viewer, on MuJoCo and on Genesis alike. A rollout under TRAINING
conditions (randomization on, observation noise on, 6 s episodes) does not
show it, so the conditions the viewer runs under are what this script
reproduces and then puts back one at a time. The viewer (``PolicyEvaluator``)
differs from training in three ways: every ``interval_dr`` / ``reset_dr`` /
``interval`` event is removed, so the joints run at the asset's zero passive
damping and the nominal gains; observation noise is off; the episode never
times out. The policy is the deterministic ``act_inference`` mean.

Conditions, each a fresh environment with the same checkpoint:

  viewer        the viewer's conditions exactly
  +damping      viewer conditions with the passive joint-damping draw kept
  +noise        viewer conditions with observation noise kept
  training      everything the training run had

For each condition the policy runs ``--seconds`` of deterministic control on
``--num_envs`` fallen-or-standing resets and the head joints are recorded at
every step. Reported per condition:

  A. Over the stood-up part of the run, in 2 s windows: the fraction of
     environments whose head yaw swings more than the threshold in any
     window, the swing onset time, and the head yaw / head pitch / hip pitch
     position p2p, target std, raw-action sign-flip rate, dominant frequency
     and target-to-position lag (p50 / p90 over environments).
  B. For the swinging environments, which way the robot was lying at reset.
  C. A step trace of the worst environment around its worst window.

Run::

    jaxpy -m jaxrlworld.scripts.diag.k1.getup_head_oscillation_diag --checkpoint outputs/models/<date>/<time>/checkpoint_<it>
    jaxpy -m jaxrlworld.scripts.diag.k1.getup_head_oscillation_diag --checkpoint ... --sim genesis --conditions viewer
"""

from __future__ import annotations

import argparse
import re

import numpy as np
import torch

from jaxrlworld.rl.algorithms.ppo import PPO
from jaxrlworld.rl.configs.base_config import iter_terms
from jaxrlworld.rl.configs.common_config_classes import disable_corruption
from jaxrlworld.rl.configs.events.event_term_config import EventTermConfig
from jaxrlworld.rl.configs.presets.k1_getup.base import K1GetupConfig
from jaxrlworld.rl.runners.on_policy_runner import OnPolicyRunner
from jaxrlworld.rl.utils.jax_utils import jax_to_torch

HEAD_YAW = r".*Head_Yaw"
HEAD_PITCH = r".*Head_Pitch"
REFERENCE = r".*Left_Hip_Pitch"
# ``sum((UP - g_b)^2)`` below this counts as upright (the success tracker's
# threshold, about 18 degrees).
UPRIGHT_ERR = 0.05
# Head yaw peak-to-peak inside a window above this is a swing.
SWING_P2P = 0.15
WINDOW_S = 2.0
# Seconds after standing up before the windows start (the settling transient).
STOOD_MARGIN_S = 1.0

# Condition name -> event names kept among the ones the viewer strips
# (``None`` keeps everything, i.e. the training configuration).
CONDITIONS = {
    "viewer": (),
    "+damping": ("dr_joint_damping",),
    "+noise": (),
    "training": None,
}


def _joint_index(names: list[str], pattern: str) -> int:
    hits = [i for i, n in enumerate(names) if re.fullmatch(pattern, n)]
    if len(hits) != 1:
        raise SystemExit(f"{pattern!r} matched {len(hits)} joints in {names}")
    return hits[0]


def build_runner(checkpoint: str, sim: str, num_envs: int, condition: str) -> OnPolicyRunner:
    cfgs = K1GetupConfig(sim_type=sim, num_envs=num_envs).build()
    keep = CONDITIONS[condition]
    if keep is not None:
        # The evaluator's eval-mode defaults (``PolicyEvaluator._apply_eval_defaults``).
        if condition != "+noise":
            disable_corruption(cfgs.observation)
        for name, term in iter_terms(cfgs.event, EventTermConfig).items():
            if term.mode in ("interval", "interval_dr", "reset_dr") and name not in keep:
                setattr(cfgs.event, name, None)
        cfgs.env.episode_length_s = 10e9
    runner = OnPolicyRunner.load_checkpoint(checkpoint, cfgs=cfgs, use_wandb=False)
    runner.set_eval_mode()
    return runner


def rollout(runner: OnPolicyRunner, steps: int, joints: list[int]) -> dict[str, np.ndarray]:
    env = runner.env
    rd = env.get_entity_data("robot")
    am = env.act_manager
    obs, _ = env.reset()
    up = torch.tensor([0.0, 0.0, -1.0], device=env.device)
    initial_gravity = rd.projected_gravity_b.clone()
    rec = {k: [] for k in ("raw", "target", "pos", "vel", "orient_err")}
    for _ in range(steps):
        act_in = PPO.ActInput(runner._pack_obs(obs, "actor"), runner._pack_obs(obs, "critic"))
        actions = runner.alg.act(act_in, deterministic=True)
        actions_torch = jax_to_torch(runner._process_action_for_env(actions), runner.device)
        obs, _, _, _, _ = env.step(actions_torch)
        rec["raw"].append(am.raw_actions[:, joints].clone())
        rec["target"].append(am.processed_actions[:, joints].clone())
        rec["pos"].append(rd.joint_pos[:, joints].clone())
        rec["vel"].append(rd.joint_vel[:, joints].clone())
        rec["orient_err"].append(torch.sum(torch.square(up - rd.projected_gravity_b), dim=-1))
    out = {k: torch.stack(v, dim=0).cpu().numpy() for k, v in rec.items()}  # (T, N, J) / (T, N)
    out["initial_gravity"] = initial_gravity.cpu().numpy()
    return out


def _dominant_freq(x: np.ndarray, dt: float) -> np.ndarray:
    """Per-column dominant frequency of a (T, N) signal, excluding DC."""
    x = x - x.mean(axis=0, keepdims=True)
    spec = np.abs(np.fft.rfft(x, axis=0))
    freqs = np.fft.rfftfreq(x.shape[0], d=dt)
    spec[0] = 0.0
    return freqs[np.argmax(spec, axis=0)]


def _best_lag(target: np.ndarray, pos: np.ndarray, max_lag: int = 10) -> np.ndarray:
    """Per-env lag (steps) maximizing corr(target[t], pos[t + lag]); positive = target leads."""
    t = target - target.mean(axis=0, keepdims=True)
    p = pos - pos.mean(axis=0, keepdims=True)
    lags = np.arange(-max_lag, max_lag + 1)
    corr = np.zeros((len(lags), t.shape[1]))
    for i, lag in enumerate(lags):
        if lag >= 0:
            a, b = t[: t.shape[0] - lag], p[lag:]
        else:
            a, b = t[-lag:], p[: p.shape[0] + lag]
        corr[i] = (a * b).sum(axis=0) / (np.sqrt((a * a).sum(axis=0) * (b * b).sum(axis=0)) + 1e-9)
    return lags[np.argmax(corr, axis=0)]


def _flip_rate(raw: np.ndarray) -> np.ndarray:
    """Fraction of steps at which the raw action increment changes sign, per env."""
    d = np.diff(raw, axis=0)
    s = np.sign(d)
    flips = (s[1:] * s[:-1] < 0).sum(axis=0)
    return flips / max(1, s.shape[0] - 1)


def _stood_step(orient_err: np.ndarray, hold: int) -> np.ndarray:
    """First step of the upright stretch that reaches the end of the run, if
    that stretch is at least ``hold`` steps long; else -1."""
    upright = orient_err < UPRIGHT_ERR
    T, N = upright.shape
    final_run = np.zeros(N, dtype=int)
    alive = np.ones(N, dtype=bool)
    for t in range(T - 1, -1, -1):
        alive &= upright[t]
        final_run += alive
    return np.where(final_run >= hold, T - final_run, -1)


def analyze(condition: str, r: dict[str, np.ndarray], labels: list[str], dt: float) -> tuple[np.ndarray, np.ndarray]:
    T, N, _ = r["pos"].shape
    win = int(WINDOW_S / dt)
    margin = int(STOOD_MARGIN_S / dt)
    stood = _stood_step(r["orient_err"], hold=win + margin)
    valid = stood >= 0
    print(
        f"\n=== A. {condition}: {int(valid.sum())}/{N} environments stood up and stayed up "
        f"for >= {WINDOW_S + STOOD_MARGIN_S:.0f} s ==="
    )
    if valid.sum() == 0:
        return valid, np.zeros(N, dtype=bool)
    yaw = r["pos"][:, :, 0]
    worst_p2p = np.full(N, -1.0)
    worst_start = np.full(N, -1)
    for n in np.flatnonzero(valid):
        for start in range(stood[n] + margin, T - win + 1, win // 2):
            seg = yaw[start : start + win, n]
            p2p = seg.max() - seg.min()
            if p2p > worst_p2p[n]:
                worst_p2p[n], worst_start[n] = p2p, start
    swinging = valid & (worst_p2p > SWING_P2P)
    print(
        f"    head yaw swing > {SWING_P2P} rad in some {WINDOW_S:.0f} s window: {int(swinging.sum())}/{int(valid.sum())}; "
        f"worst-window p2p p50 {np.percentile(worst_p2p[valid], 50):.3f} "
        f"p90 {np.percentile(worst_p2p[valid], 90):.3f} max {worst_p2p[valid].max():.3f} rad"
    )
    if swinging.any():
        onset = (worst_start[swinging] - stood[swinging]) * dt
        print(f"    swing onset after standing: p50 {np.percentile(onset, 50):.1f} s, max {onset.max():.1f} s")
    print(
        f"    {'joint':16s} {'pos p2p':>13s} {'tgt std':>13s} {'vel rms':>13s} {'flip rate':>13s} "
        f"{'freq Hz':>13s} {'lag':>13s}   (p50 / p90 over each env's worst window)"
    )
    idx = np.flatnonzero(valid)
    for j, label in enumerate(labels):
        pos = np.stack([r["pos"][worst_start[n] : worst_start[n] + win, n, j] for n in idx], axis=1)
        tgt = np.stack([r["target"][worst_start[n] : worst_start[n] + win, n, j] for n in idx], axis=1)
        raw = np.stack([r["raw"][worst_start[n] : worst_start[n] + win, n, j] for n in idx], axis=1)
        vel = np.stack([r["vel"][worst_start[n] : worst_start[n] + win, n, j] for n in idx], axis=1)
        stats = [
            pos.max(axis=0) - pos.min(axis=0),
            tgt.std(axis=0),
            np.sqrt((vel * vel).mean(axis=0)),
            _flip_rate(raw),
            _dominant_freq(pos, dt),
            _best_lag(tgt, pos).astype(float),
        ]
        cells = [f"{np.percentile(v, 50):.3f}/{np.percentile(v, 90):.3f}" for v in stats]
        print(f"    {label:16s} " + " ".join(f"{c:>13s}" for c in cells))
    if swinging.any():
        g = r["initial_gravity"]
        print(f"\n=== B. {condition}: where the swinging environments were lying at reset ===")
        for label, mask in (("swinging", swinging), ("quiet", valid & ~swinging)):
            if mask.sum() == 0:
                continue
            gm = g[mask]
            faces = {
                "front": (gm[:, 0] > 0.5).mean(),
                "back": (gm[:, 0] < -0.5).mean(),
                "left": (gm[:, 1] > 0.5).mean(),
                "right": (gm[:, 1] < -0.5).mean(),
                "upright": (gm[:, 2] < -0.5).mean(),
                "inverted": (gm[:, 2] > 0.5).mean(),
            }
            print(f"    {label:9s} n={int(mask.sum()):5d}  " + "  ".join(f"{k} {v:.2f}" for k, v in faces.items()))
        worst = int(np.argmax(np.where(swinging, worst_p2p, -1.0)))
        start = max(0, worst_start[worst] - 10)
        print(
            f"\n=== C. {condition}: environment {worst} (stood at {stood[worst] * dt:.1f} s), "
            f"head yaw from {start * dt:.1f} s ==="
        )
        print(f"    {'t [s]':>6s} {'raw':>8s} {'target':>8s} {'pos':>8s} {'vel':>8s}")
        for t in range(start, min(T, start + 60)):
            print(
                f"    {t * dt:6.2f} {r['raw'][t, worst, 0]:8.3f} {r['target'][t, worst, 0]:8.3f} "
                f"{r['pos'][t, worst, 0]:8.3f} {r['vel'][t, worst, 0]:8.3f}"
            )
    return valid, swinging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sim", choices=("mujoco", "newton", "genesis"), default="mujoco")
    parser.add_argument("--num_envs", type=int, default=1024)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument(
        "--conditions", default=",".join(CONDITIONS), help="comma-separated subset of " + ",".join(CONDITIONS)
    )
    args = parser.parse_args()
    conditions = args.conditions.split(",")
    unknown = [c for c in conditions if c not in CONDITIONS]
    if unknown:
        raise SystemExit(f"unknown conditions {unknown}; choose from {list(CONDITIONS)}")

    summary = {}
    for condition in conditions:
        runner = build_runner(args.checkpoint, args.sim, args.num_envs, condition)
        env = runner.env
        names = list(env.act_manager.actuated_joint_names)
        joints = [_joint_index(names, p) for p in (HEAD_YAW, HEAD_PITCH, REFERENCE)]
        labels = [names[j].rsplit("/", 1)[-1] for j in joints]
        dt = env.control_dt
        steps = int(args.seconds / dt)
        if condition == "training":
            # Training episodes time out at 6 s; stay inside one episode.
            steps = min(steps, env.max_episode_length - 1)
        print(f"\n##### condition {condition!r}: {args.num_envs} envs, {steps * dt:.1f} s, deterministic policy")
        r = rollout(runner, steps, joints)
        valid, swinging = analyze(condition, r, labels, dt)
        summary[condition] = (int(swinging.sum()), int(valid.sum()))

    print("\n=== summary: environments with a head yaw swing / environments that stood up ===")
    for condition, (n_swing, n_valid) in summary.items():
        print(f"    {condition:10s} {n_swing:5d} / {n_valid:5d}")


if __name__ == "__main__":
    main()
