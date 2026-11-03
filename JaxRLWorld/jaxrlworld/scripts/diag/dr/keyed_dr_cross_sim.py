"""Do the three simulators draw the same domain randomization?

With the keyed sampler every event-term draw is a function of
(seed, term, stage, env, episode, coordinates), so a preset reset on
Genesis, Newton and MuJoCo should produce the same raw draws for the
same term wherever the coordinates mean the same thing (per-env draws,
per-canonical-joint draws), and a 1024-env run should draw for envs
0..1023 what an 8192-env run draws for them. This diag records the
draws a real reset and the first step(s) make and checks that, on the GPU,
through the real presets. The recorded values are the raw uniforms in
``[0, 1)`` each draw started from. One simulator per process (the backends do not share one):

    jaxpy -m jaxrlworld.scripts.diag.dr.keyed_dr_cross_sim --preset go2 --sim newton --num-envs 1024 --out parity_out/keyed_dr
    jaxpy -m jaxrlworld.scripts.diag.dr.keyed_dr_cross_sim --preset go2 --sim mujoco --num-envs 1024 --out parity_out/keyed_dr
    jaxpy -m jaxrlworld.scripts.diag.dr.keyed_dr_cross_sim --preset go2 --sim genesis --num-envs 1024 --out parity_out/keyed_dr
    jaxpy -m jaxrlworld.scripts.diag.dr.keyed_dr_cross_sim --preset go2 --sim newton --num-envs 4096 --out parity_out/keyed_dr
    jaxpy -m jaxrlworld.scripts.diag.dr.keyed_dr_cross_sim --compare parity_out/keyed_dr

Two dumps of one simulator (``--tag a`` / ``--tag b``) check that two
processes draw the same values. ``--compare`` reads every dump in the
directory and, per term and draw,
reports whether the simulators agree (bitwise) and whether the smaller
run is a prefix of the larger. A draw whose shape differs between
backends (a backend's own shape or dof order, e.g. Newton's per-shape
friction) is reported as INFO, not a failure: those coordinates are
backend-specific by design. ``--bench`` times a keyed draw against
``torch.rand`` at training size on the current device.
"""

from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from jaxrlworld.rl.envs.mdp.events.keyed_sampler import STAGE_IDS, TermSampler, term_id

_PRESETS: dict[str, tuple[str, str]] = {
    "go2": ("jaxrlworld.rl.configs.presets.go2.base", "Go2FlatConfig"),
    "g1_29dof": ("jaxrlworld.rl.configs.presets.g1_29dof.base", "G1FlatConfig"),
    "k1": ("jaxrlworld.rl.configs.presets.k1_velocity.base", "K1VelocityConfig"),
}
_STAGE_NAMES = {v: k for k, v in STAGE_IDS.items()}


def _build_env(preset: str, sim: str, num_envs: int):
    from jaxrlworld.rl.runners import BaseRunner

    if ":" in preset:
        mod_path, cls_name = preset.split(":", 1)
    else:
        mod_path, cls_name = _PRESETS[preset]
    cfg_cls = getattr(importlib.import_module(mod_path), cls_name)
    cfgs = cfg_cls(sim_type=sim, num_envs=num_envs).build()
    return BaseRunner.create_with_env(cfgs, use_wandb=False).env


def record(args) -> int:
    env = _build_env(args.preset, args.sim, args.num_envs)
    mgr = env.event_manager
    names_by_id = {term_id(n): n for n in mgr._all_terms}
    draws: list[tuple[str, str, int, torch.Tensor]] = []
    original = TermSampler.uniforms

    # Every draw (draw, draw_shaped, draw_int, bernoulli, the gaussian pair,
    # the direct ``uniforms`` of the pose / push terms) funnels through
    # ``uniforms``, so this sees them all, as raw [0, 1) values.
    def recording_uniforms(self, env_ids, coords=(), stream=0):
        out = original(self, env_ids, coords, stream)
        name = names_by_id.get(self.term_id, f"id{self.term_id}")
        draws.append((name, _STAGE_NAMES.get(self.stage, str(self.stage)), int(stream), out.detach().clone()))
        return out

    TermSampler.uniforms = recording_uniforms
    try:
        env.reset()
        # Interval and interval_dr terms draw on a step, not on the reset;
        # the global interval_dr timer starts at zero, so one zero-action
        # step fires every DR term the preset routes through it.
        zero = torch.zeros(env.num_envs, env.act_manager.num_actions, device=env.device)
        for _ in range(args.steps):
            env.step(zero)
    finally:
        TermSampler.uniforms = original

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.sim}_n{args.num_envs}" + (f"_{args.tag}" if args.tag else "")
    arrays = {}
    index = []
    per_term_count: dict[str, int] = {}
    for name, stage, stream, value in draws:
        k = per_term_count.get(name, 0)
        per_term_count[name] = k + 1
        key = f"{name}|{stage}|{k}"
        arrays[key] = value.float().cpu().numpy()
        index.append({"key": key, "term": name, "stage": stage, "stream": stream, "shape": list(value.shape)})
    np.savez(out_dir / f"{stem}.npz", **arrays)
    meta = {
        "preset": args.preset,
        "sim": args.sim,
        "num_envs": args.num_envs,
        "seed": int(env.seed),
        "episode_count_after_reset": int(mgr.episode_count.min()),
        "draws": index,
    }
    (out_dir / f"{stem}.json").write_text(json.dumps(meta, indent=1))
    print(f"recorded {len(draws)} draws of {len(per_term_count)} terms -> {out_dir / stem}.npz")
    for name, count in per_term_count.items():
        rows = [d for d in index if d["term"] == name]
        shapes = [d["shape"] for d in rows]
        stages = sorted({d["stage"] for d in rows})
        print(f"  {name:<32} {count} draw(s)  stages {stages}  shapes {shapes}")
    return 0


def compare(args) -> int:
    out_dir = Path(args.compare)
    runs = {}
    for meta_path in sorted(out_dir.glob("*.json")):
        meta = json.loads(meta_path.read_text())
        runs[meta_path.stem] = (meta, np.load(out_dir / f"{meta_path.stem}.npz"))
    if len(runs) < 2:
        print(f"need at least two dumps in {out_dir}, found {len(runs)}")
        return 1
    presets = {m["preset"] for m, _ in runs.values()}
    seeds = {m["seed"] for m, _ in runs.values()}
    if len(presets) != 1 or len(seeds) != 1:
        print(f"dumps mix presets {presets} / seeds {seeds}; compare one preset and seed at a time")
        return 1
    failures = 0
    infos = 0
    keys = sorted(set().union(*(set(a.files) for _, a in runs.values())))
    print(f"{'draw':<48} {'runs':>5}  verdict")
    for key in keys:
        present = {stem: a[key] for stem, (_, a) in runs.items() if key in a.files}
        if len(present) < 2:
            print(f"{key:<48} {len(present):>5}  INFO only in {list(present)}")
            infos += 1
            continue
        # Compare every pair on the envs both have; shapes beyond the env axis must match.
        stems = sorted(present)
        verdicts = []
        for i in range(len(stems)):
            for j in range(i + 1, len(stems)):
                a, b = present[stems[i]], present[stems[j]]
                if a.shape[1:] != b.shape[1:]:
                    verdicts.append(f"INFO {stems[i]} vs {stems[j]}: backend shapes {a.shape[1:]} vs {b.shape[1:]}")
                    continue
                n = min(a.shape[0], b.shape[0])
                same = np.array_equal(a[:n], b[:n])
                if same:
                    verdicts.append(f"OK {stems[i]} == {stems[j]} ({n} envs)")
                else:
                    diff = np.abs(a[:n] - b[:n]).max()
                    verdicts.append(f"FAIL {stems[i]} != {stems[j]} (max |diff| {diff:.3g})")
                    failures += 1
        infos += sum(v.startswith("INFO") for v in verdicts)
        print(f"{key:<48} {len(present):>5}  " + "; ".join(verdicts))
    print("=" * 78)
    print(
        f"=== RESULT: {'ALL OK' if failures == 0 else f'{failures} FAILED'} ({infos} backend-specific draws noted) ==="
    )
    return 1 if failures else 0


def bench(args) -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n, cols = args.num_envs, 32
    ids = torch.arange(n, device=device)
    sampler = TermSampler.bind(
        base_seed=1, term_name="bench", stage=3, counter=torch.zeros(n, dtype=torch.long, device=device)
    )
    coords = (torch.arange(cols, device=device)[None, :],)

    def timed(fn, reps=200):
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / reps * 1e3

    keyed = timed(lambda: sampler.draw(0.5, 1.5, ids, coords))
    keyed_shared = timed(lambda: sampler.draw(0.5, 1.5, ids))
    keyed_gauss = timed(lambda: sampler.draw(0.5, 1.5, ids, coords, "gaussian"))
    torch_rand = timed(lambda: torch.empty((n, cols), device=device).uniform_(0.5, 1.5))
    print(f"device {device}, {n} envs x {cols} columns, ms per draw:")
    print(f"  keyed uniform [E, {cols}]   {keyed:8.4f}")
    print(f"  keyed uniform [E]        {keyed_shared:8.4f}")
    print(f"  keyed gaussian [E, {cols}]  {keyed_gauss:8.4f}")
    print(f"  torch uniform_ [E, {cols}]  {torch_rand:8.4f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="go2", help="table key or module:Class")
    ap.add_argument("--sim", default="newton", choices=("genesis", "newton", "mujoco"))
    ap.add_argument("--num-envs", type=int, default=1024)
    ap.add_argument("--out", default="parity_out/keyed_dr", help="record: directory for the dump")
    ap.add_argument(
        "--tag", default="", help="record: suffix for the dump name (two runs of one sim: --tag a / --tag b)"
    )
    ap.add_argument(
        "--steps", type=int, default=1, help="record: zero-action steps after the reset (fires interval DR)"
    )
    ap.add_argument("--compare", default=None, help="compare the dumps in this directory instead of recording")
    ap.add_argument("--bench", action="store_true", help="time a keyed draw against torch.rand and exit")
    args = ap.parse_args()
    if args.bench:
        return bench(args)
    if args.compare:
        return compare(args)
    return record(args)


if __name__ == "__main__":
    raise SystemExit(main())
