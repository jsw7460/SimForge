"""Decompose the go2 feet_slip cross-sim gap into its actual causes.

``check_reward_parity`` shows a stable ordering at rest — genesis
-0.00137, newton -0.00118 (0.86x), mujoco -0.00098 (0.71x), with
intermittent newton spikes — after every static quantity (geometry,
contact params, DOF params, solver options) was proven equal. This diag
stops guessing and measures the three remaining mechanisms separately:

 1. READ TIME.  Each ``feet_slip`` raw value is captured twice per
    control step: once at reward time (inside ``env.step``, exactly as
    training computes it) and once re-invoked after the step returns.
    On genesis/newton both reads see the same buffers; on mujoco the
    reward-time read sees the one-substep-stale ``robot.data`` and the
    post-step read sees the refreshed one, so the difference IS the
    staleness contribution, in reward units, with nothing inferred.

 2. MOTION vs SAMPLING.  Foot positions are recorded every control step
    and differentiated (``v_fd``). Finite-differenced displacement is
    convention-free — the same quantity on every backend — while the
    reward reads an instantaneous velocity (``v_inst``). If the feet
    truly drift, ``v_inst ~= v_fd``; if they oscillate in place at
    substep frequency, ``v_inst >> v_fd`` and each engine's ratio
    exposes its own jitter floor.

 3. SPIKES.  Per step, the env with the most negative ``feet_slip`` is
    logged; when it exceeds 5x the step median it is dumped per foot
    (displacement speed, height, contact-force magnitude) so the newton
    transients get an anatomy instead of a shrug.

Reads mirror the reward implementations exactly (genesis
``get_links_vel(ref=link_COM)``, newton ``body_qd`` linear rows, mujoco
site positions from the live warp data by site id).

Usage:
    python -m jaxrlworld.scripts.diag.parity.go2_feet_slip_decomposition_diag
    python -m jaxrlworld.scripts.diag.parity.go2_feet_slip_decomposition_diag --sims newton mujoco --steps 20
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("JAXRLWORLD_ALLOW_MULTI_SIM", "1")

import numpy as np  # noqa: E402

from jaxrlworld.scripts.diag.parity.check_reward_parity import _build_env  # noqa: E402

SIMS = ("genesis", "newton", "mujoco")
NUM_ENVS = 16
STEPS = 14
WARM = 4  # steady-state window starts here (settling transient before)
TERM = "feet_slip"
GROUP = "feet_ground_contact"


def leaf(name: str) -> str:
    return name.split("/")[-1]


def capture_step(env, action):
    """Step once; return (reward-time raw, post-step raw) for ``TERM``.

    Same monkey-patch as ``check_reward_parity`` — every term computes
    exactly once at its normal place in the pipeline — plus the term's
    resolved callable is kept so the SAME function can be re-invoked
    after the step returns, against whatever the buffers hold then.
    """
    from jaxrlworld.rl.envs.managers.common.reward import get_weight_value

    mgr = env.reward_manager
    raw_rt: dict[str, np.ndarray] = {}
    recall: dict[str, object] = {}
    orig = mgr._compute_weighted_reward

    def wrapper(name, term):
        if name in mgr._instances:
            fn = mgr._instances[name]
            raw = fn(mgr.env)
            recall[name] = lambda fn=fn: fn(mgr.env)
        else:
            fn = mgr._resolved_fns[name]
            raw = fn(mgr.env, **term.params)
            recall[name] = lambda fn=fn, p=term.params: fn(mgr.env, **p)
        raw_rt[name] = raw.detach().cpu().float().numpy()
        w = get_weight_value(term.weight, mgr.env_step_calls)
        return raw * w * mgr.env.control_dt

    mgr._compute_weighted_reward = wrapper
    try:
        env.step(action)
    finally:
        mgr._compute_weighted_reward = orig

    raw_post = recall[TERM]().detach().cpu().float().numpy()
    return raw_rt[TERM], raw_post


def foot_readers(env, sim: str):
    """Return (positions_fn, inst_vel_fn) for the gait feet, in gait order.

    ``inst_vel_fn`` is None on mujoco — its instantaneous read lives
    behind the entity-data staleness this diag measures at the raw
    level instead.
    """
    feet = list(env.gait_manager.foot_names)

    if sim == "genesis":
        import genesis as gs

        from jaxrlworld.rl.utils import entity_utils as eu

        entity = env.scene_manager["robot"]
        idx, _ = eu.find_links(entity, feet, global_ids=False, preserve_order=True)

        def pos():
            return entity.get_links_pos(links_idx_local=idx).detach().cpu().numpy()

        def vel():
            return entity.get_links_vel(links_idx_local=idx, ref=gs.link_ref_frame.link_COM).detach().cpu().numpy()

        return pos, vel

    if sim == "newton":
        import warp as wp

        from jaxrlworld.rl.envs.mdp.observations.newton.body_utils import get_bodies_pos
        from jaxrlworld.rl.envs.utils.newton.body_cache import get_cache

        cache = get_cache(env)
        body_indices = cache.get_body_indices(feet)

        def pos():
            return get_bodies_pos(env, feet).data.detach().cpu().numpy()

        def vel():
            qd = wp.to_torch(env.scene_manager.state.body_qd).reshape(env.num_envs, cache.bodies_per_env, 6)
            return qd[:, body_indices, :3].detach().cpu().numpy()

        return pos, vel

    # mujoco: site positions straight off the live warp data, by site id.
    import warp as wp

    m = env.scene_manager.mj_model
    names = [m.site(i).name for i in range(m.nsite)]
    ids = []
    for foot in feet:
        hits = [i for i, n in enumerate(names) if leaf(n) in foot or foot in leaf(n)]
        if len(hits) != 1:
            raise ValueError(f"Cannot map foot {foot!r} to sites {names}: {hits}")
        ids.append(hits[0])

    def pos():
        return wp.to_torch(env.scene_manager.sim.wp_data.site_xpos)[:, ids].detach().cpu().numpy()

    return pos, None


def contact_force_norms(env) -> np.ndarray:
    f = env.contact_manager.contact_force(GROUP)
    return np.linalg.norm(f.detach().cpu().numpy(), axis=-1)


def run(sim: str, steps: int) -> dict[str, object]:
    import torch

    env = _build_env("go2_gait", sim, NUM_ENVS)
    pos_fn, vel_fn = foot_readers(env, sim)
    zero = torch.zeros(NUM_ENVS, env.act_manager.num_actions, device=env.device)
    control_dt = env.control_dt

    prev_p = pos_fn()
    steady = {"v_fd": [], "v_inst": [], "raw_rt": [], "raw_post": [], "in_contact": []}
    spikes: list[str] = []

    for step in range(steps):
        raw_rt, raw_post = capture_step(env, zero)
        p = pos_fn()
        v_fd = (p - prev_p) / control_dt
        prev_p = p
        v_inst = vel_fn() if vel_fn is not None else None
        forces = contact_force_norms(env)

        if step >= WARM:
            steady["v_fd"].append(np.linalg.norm(v_fd[..., :2], axis=-1))
            if v_inst is not None:
                steady["v_inst"].append(np.linalg.norm(v_inst[..., :2], axis=-1))
            steady["raw_rt"].append(raw_rt)
            steady["raw_post"].append(raw_post)
            steady["in_contact"].append((forces > 1e-3).astype(np.float64))

        worst = int(np.argmin(raw_rt))
        med = float(np.median(raw_rt))
        if raw_rt[worst] < 5.0 * med and raw_rt[worst] < -1e-4:
            fd_mag = np.linalg.norm(v_fd[worst, :, :2], axis=-1)
            spikes.append(
                f"      step {step:>2} env {worst:>2}: raw {raw_rt[worst]:+.5f} (median {med:+.5f})  "
                f"|v_fd_xy| {np.array2string(fd_mag, precision=3)}  "
                f"foot z {np.array2string(p[worst, :, 2], precision=3)}  "
                f"|F| {np.array2string(forces[worst], precision=1)}"
            )

    out = {
        "v_fd": float(np.mean(np.concatenate(steady["v_fd"]))),
        "v_inst": float(np.mean(np.concatenate(steady["v_inst"]))) if steady["v_inst"] else None,
        "raw_rt": float(np.mean(np.concatenate(steady["raw_rt"]))),
        "raw_post": float(np.mean(np.concatenate(steady["raw_post"]))),
        "contact_frac": float(np.mean(np.concatenate(steady["in_contact"]))),
        "spikes": spikes,
    }
    del env
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sims", nargs="+", default=list(SIMS), choices=list(SIMS))
    ap.add_argument("--steps", type=int, default=STEPS)
    args = ap.parse_args()

    print("=" * 100)
    print("  GO2 FEET_SLIP — WHERE DOES THE CROSS-SIM GAP ACTUALLY COME FROM")
    print("=" * 100)

    results = {sim: run(sim, args.steps) for sim in args.sims}

    print(f"\n    STEADY STATE (steps {WARM}..{args.steps - 1}, {NUM_ENVS} envs, per-foot means)")
    print(f"      {'quantity':<34}" + "".join(f"{sim:>16}" for sim in args.sims))

    def row(label: str, key: str, fmt: str = "{:+.5f}") -> None:
        cells = []
        for sim in args.sims:
            v = results[sim][key]
            cells.append(f"{fmt.format(v):>16}" if v is not None else f"{'—':>16}")
        print(f"      {label:<34}" + "".join(cells))

    row("feet_slip raw @ reward time", "raw_rt")
    row("feet_slip raw @ post-step", "raw_post")
    print("        -> reward-time vs post-step gap = the read-TIMING contribution (mujoco's staleness)")
    row("|v_fd_xy|  (net motion / dt)  m/s", "v_fd", "{:.4f}")
    row("|v_inst_xy| (reward's read)   m/s", "v_inst", "{:.4f}")
    print("        -> v_inst >> v_fd = in-place substep jitter, sampled; v_inst ~= v_fd = real drift")
    row("feet in contact (fraction)", "contact_frac", "{:.3f}")

    print("\n    SPIKE ANATOMY (env with most negative feet_slip, when >5x the step median)")
    for sim in args.sims:
        spikes = results[sim]["spikes"]
        print(f"      [{sim}] {len(spikes)} spike step(s)")
        for line in spikes:
            print(line)

    print("\n" + "=" * 100)
    print("  interpretation: gap components are (a) raw_rt vs raw_post per sim, (b) v_fd across sims,")
    print("  (c) v_inst/v_fd per sim; whichever is non-flat across the table is the cause.")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
