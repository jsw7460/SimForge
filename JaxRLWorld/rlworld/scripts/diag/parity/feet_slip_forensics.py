"""feet_slip cross-sim forensics: where exactly does mujoco's number diverge?

Observed on g1 / g1 rough / go2 / k1: the mujoco cell's feet_slip
penalty is 10-50% smaller in magnitude than genesis/newton. The go2
audit (2026-08-25) proved all parameters equal there and attributed the
residual to engine-level tangential creep (finite-difference foot speed
gs 0.0189 / nt 0.0172 / mj 0.0162 m/s, whose squared ratios matched the
reward ratios). This diag generalizes that proof to ANY preset, and
re-verifies per preset the hypotheses that were only checked on go2:

  A. Source dump      — the term's actual function, params, resolved
                        selector (site vs body ids/names), the contact
                        group's tracked names, and each backend's
                        is_contact semantics. "Are the three sims even
                        reading the same things?"
  B. Geometry         — settled foot positions of the term's own
                        position source, cross-sim (a site-vs-link
                        measurement-point mismatch shows up here as an
                        offset), plus per-sim velocity-source integrity:
                        the term's velocity against a finite difference
                        of the term's own positions.
  C. Decomposition    — captured INSIDE the reward manager's own call
                        (same cache generation as training): raw term
                        value, contact fraction, mean squared planar
                        foot speed over contacted feet, and the command
                        gate. reward ≈ E[v² · contact] · gate, so the
                        cross-sim ratio factors into "contacts differ"
                        vs "slip speed differs".
  E. Source ablation  — the decisive probe for a raw-value divergence:
                        inside the same reward-time capture, recompute
                        the term's sum three ways that differ ONLY in
                        the velocity source: (A) the term's own exact
                        read (newton: raw ``state.body_qd`` linear part;
                        genesis: ``get_links_vel(ref=link_COM)``;
                        mujoco: ``site_lin_vel_w``), (B) the RobotData
                        link-origin velocity for the same feet, and
                        (FD) a finite difference of the term's own raw
                        positions across control steps. FD is
                        convention-free ground truth: whichever of A/B
                        matches FD is reading the physical foot
                        velocity; the other one is reading a different
                        point (or a stale buffer) — and the A/raw ratio
                        checks that the replication really is the term.
  D. Creep fingerprint— zero action, commands untouched, no reward code:
                        finite-difference planar foot speed of contacted
                        feet at steady standing. This is the pure
                        constraint-solver residual. The verdict compares
                        creep² ratios against the measured reward ratios
                        — a match proves the divergence is the engines'
                        friction creep, not our plumbing.

Dead hypotheses from the go2 audit (do not re-chase without new
evidence): mjlab one-substep staleness (measured zero), go2 site-vs-link
measurement point (coincident there — phase B re-checks it per preset),
"the sims run different robots" (pair/param/dof parity diags).

Usage:
    jaxpy -m rlworld.scripts.diag.parity.feet_slip_forensics --preset go2_gait
    jaxpy -m rlworld.scripts.diag.parity.feet_slip_forensics --preset g1_29dof
    jaxpy -m rlworld.scripts.diag.parity.feet_slip_forensics --preset g1_29dof_rough
    jaxpy -m rlworld.scripts.diag.parity.feet_slip_forensics --preset k1_joystick
"""

from __future__ import annotations

import argparse
import importlib
import os

os.environ.setdefault("JAXRLWORLD_ALLOW_MULTI_SIM", "1")
os.environ.setdefault("JAXRLWORLD_PLAIN_LOG", "1")

import numpy as np
import torch

_SIMS = ("genesis", "newton", "mujoco")

# name -> (module, class, ctor kwargs)
_PRESETS: dict[str, tuple[str, str, dict]] = {
    "go2": ("rlworld.rl.configs.presets.go2.base", "Go2FlatConfig", {}),
    "g1_29dof": ("rlworld.rl.configs.presets.g1_29dof.base", "G1FlatConfig", {}),
    "g1_29dof_rough": ("rlworld.rl.configs.presets.g1_29dof.base", "G1FlatConfig", {"use_rough_terrain": True}),
    "k1_joystick": ("rlworld.rl.configs.presets.k1_joystick.base", "K1JoystickConfig", {}),
    "k1_g1_recipe": ("rlworld.rl.configs.presets.k1_joystick.g1_recipe", "K1G1RecipeConfig", {}),
}
_PER_SIM_PRESETS: dict[str, dict[str, tuple[str, str, dict]]] = {
    "go2_gait": {
        "genesis": ("rlworld.rl.configs.presets.go2.genesis.gait_conditioned", "Go2GaitConditionedGenesisConfig", {}),
        "newton": ("rlworld.rl.configs.presets.go2.newton.gait_conditioned", "Go2GaitConditionedNewtonConfig", {}),
        "mujoco": ("rlworld.rl.configs.presets.go2.mujoco.gait_conditioned", "Go2GaitConditionedMujocoConfig", {}),
    },
}


def _build_env(preset: str, sim: str, num_envs: int):
    from rlworld.rl.runners import BaseRunner

    if ":" in preset:
        mod_path, cls_name, kwargs = *preset.split(":", 1), {}
    elif preset in _PER_SIM_PRESETS:
        mod_path, cls_name, kwargs = _PER_SIM_PRESETS[preset][sim]
    else:
        mod_path, cls_name, kwargs = _PRESETS[preset]
    cfg_cls = getattr(importlib.import_module(mod_path), cls_name)
    cfgs = cfg_cls(sim_type=sim, num_envs=num_envs, **kwargs).build()
    runner = BaseRunner.create_with_env(cfgs)
    return runner.env


def _find_slip_term(env):
    """The feet_slip reward term registered on this env."""
    for name, term in env.reward_manager.reward_terms.items():
        if "feet_slip" in name or "foot_slip" in name:
            return name, term
    raise ValueError(f"No feet_slip term found. Terms: {list(env.reward_manager.reward_terms)}")


def _slip_sources(env, term):
    """(kind, asset_cfg, contact_group, contact_order, command_threshold).

    ``kind`` is "foot" for the shared penalize_feet_slip family (foot
    planar speed squared x contact) and "base" for K1's verbatim
    feet_slip_base_vel (base planar speed x contact count).
    """
    fn = env.reward_manager._resolved_fns[_find_slip_term(env)[0]]
    kind = "base" if "base_vel" in getattr(fn, "__name__", "") else "foot"
    params = term.params
    return (
        kind,
        params.get("asset_cfg"),
        params.get("contact_group", "feet_ground_contact"),
        params.get("contact_order"),
        params.get("command_threshold", 0.05 if kind == "base" else 0.01),
    )


def _make_foot_reader(env, asset_cfg, contact_group):
    """A zero-arg () -> (pos, vel) reader for the term's feet.

    Uses the term's own selector when it is resolved (sites on mujoco,
    bodies on newton/genesis — exactly what the reward reads). Terms
    that rely on the unresolved default selector (the WTW family
    resolves feet internally) fall back to the contact group's tracked
    body names, which name the same feet.
    """
    if asset_cfg is not None and ((asset_cfg.body_ids is not None) != (asset_cfg.site_ids is not None)):
        from rlworld.rl.envs.mdp.rewards.common.reward_terms import _foot_pos_vel as fpv

        return lambda: fpv(env, asset_cfg), "selector"

    tracked = list(env.contact_manager._groups[contact_group].tracked_names)
    rd = env.get_robot_data(env.robot_entity_name)
    ids = torch.tensor([rd.find_body_index(n) for n in tracked], device=env.device, dtype=torch.long)

    def read():
        return rd.body_pos_w_all[:, ids], rd.body_lin_vel_w_all[:, ids]

    return read, f"contact-group bodies {tracked}"


def _term_contact_order(asset_cfg, contact_order):
    from rlworld.rl.envs.mdp.rewards.common.reward_terms import _feet_contact_order

    return _feet_contact_order(asset_cfg, contact_order) if asset_cfg is not None else contact_order


def _contact(env, contact_group, asset_cfg, contact_order):
    return env.contact_manager.is_contact(contact_group, order=_term_contact_order(asset_cfg, contact_order))


def _make_term_probe(env, sim, kind, is_wtw, asset_cfg, group):
    """Velocity-source ablation readers in the TERM's own foot order.

    Returns ``(read_a, read_b, contacts, names, swap)`` or ``None``
    (non-foot formulas). ``read_a() -> (pos, vel)`` replicates the
    term's exact position/velocity source; ``read_b() -> vel`` is the
    RobotData link-origin velocity for the same feet (``None`` where it
    would duplicate ``read_a``); ``contacts() -> (contact, prev)`` are
    the term's exact contact reads as float tensors. ``swap`` is the
    read-TIMING ablation for the two mjwarp-backed cells, or ``None``:
    mjlab computes velocities from ``data.cvel``, which the last
    substep's ``forward`` filled BEFORE the final integration (one
    substep old), while newton fills ``state.body_qd`` from the
    post-integration ``qvel`` (true end-of-step). ``swap`` reads each
    cell the OTHER cell's way: on mujoco it reruns
    ``mujoco_warp.forward`` on the post-integration state and re-reads
    (true end-of-step); on newton it reads the stale ``cvel`` buffer
    exactly as mjlab would. All tensors are column-aligned in the
    term's own foot order.

    Sim-specific modules are imported here on purpose: this diag builds
    one simulator per process pass, and importing every backend at the
    top would initialize simulators the pass never uses.
    """
    if kind != "foot":
        return None
    if not is_wtw:
        from rlworld.rl.envs.mdp.rewards.common.reward_terms import _foot_pos_vel

        order = _term_contact_order(asset_cfg, None)

        def contacts():
            # The shared penalize_feet_slip family has no prev-filter;
            # prev is still returned for edge accounting.
            c = env.contact_manager.is_contact(group, order=order).float()
            p = env.contact_manager.prev_is_contact(group, order=order).float()
            return c, p

        return (lambda: _foot_pos_vel(env, asset_cfg)), None, contacts, list(order) if order is not None else None, None

    feet = tuple(env.gait_manager.foot_names)
    rd = env.get_robot_data(env.robot_entity_name)

    if sim == "newton":
        import warp as wp

        from rlworld.rl.envs.mdp.observations.newton.body_utils import get_bodies_height_with_contact
        from rlworld.rl.envs.utils.newton.body_cache import get_cache

        cache = get_cache(env)
        result = get_bodies_height_with_contact(env, list(feet))
        idx = list(result.body_indices)
        names = list(result.body_names)
        rd_ids = torch.tensor([rd.find_body_index(n) for n in names], device=env.device, dtype=torch.long)

        def read_a():
            st = env.scene_manager.state
            q = wp.to_torch(st.body_q).view(env.num_envs, cache.bodies_per_env, 7)
            qd = wp.to_torch(st.body_qd).view(env.num_envs, cache.bodies_per_env, 6)
            return q[:, idx, :3].clone(), qd[:, idx, :3].clone()

        def read_b():
            return rd.body_lin_vel_w_all[:, rd_ids]

        def contacts():
            c = env.contact_manager.is_contact(group, order=names).float()
            p = env.contact_manager.prev_is_contact(group, order=names).float()
            return c, p

        # Read-timing swap: the mjlab-style read of the SAME mjwarp
        # engine — data.cvel, filled by the last substep's forward
        # BEFORE the final integration (one substep older than the
        # body_qd the term reads).
        solver = env.scene_manager.solver
        m2n = wp.to_torch(solver.mjc_body_to_newton)[0].long()
        mjc_ids = []
        for nt_idx in idx:
            hits = (m2n == nt_idx).nonzero(as_tuple=False).flatten()
            if hits.numel() != 1:
                raise ValueError(f"newton body {nt_idx} maps to {hits.numel()} mjc bodies (expected 1)")
            mjc_ids.append(int(hits.item()))
        mjc_ids_t = torch.tensor(mjc_ids, device=env.device, dtype=torch.long)
        rootid_t = wp.to_torch(solver.mjw_model.body_rootid).long()
        if rootid_t.dim() == 2:
            rootid_t = rootid_t[0]
        root_of_feet = rootid_t[mjc_ids_t]

        def swap_read():
            d = solver.mjw_data
            cvel = wp.to_torch(d.cvel)[:, mjc_ids_t]  # (W, F, 6) [ang, lin] about subtree com
            xpos = wp.to_torch(d.xpos)[:, mjc_ids_t]  # (W, F, 3)
            scom = wp.to_torch(d.subtree_com)[:, root_of_feet]  # (W, F, 3)
            ang, lin_c = cvel[..., 0:3], cvel[..., 3:6]
            return lin_c - torch.cross(ang, scom - xpos, dim=-1)

        return read_a, read_b, contacts, names, ("cvel (mjlab-style, pre-integration)", swap_read)

    if sim == "genesis":
        import genesis as gs

        from rlworld.rl.utils import entity_utils as eu

        entity = env.scene_manager[asset_cfg.name if asset_cfg is not None else "robot"]
        links_idx_local, _ = eu.find_links(entity, list(feet), global_ids=False, preserve_order=True)
        rd_ids = torch.tensor([rd.find_body_index(n) for n in feet], device=env.device, dtype=torch.long)

        def read_a():
            pos = entity.get_links_pos(links_idx_local=links_idx_local)
            vel = entity.get_links_vel(links_idx_local=links_idx_local, ref=gs.link_ref_frame.link_COM)
            return pos.clone(), vel.clone()

        def read_b():
            # Genesis RobotData reads get_links_vel() at its default
            # reference (link origin) — the ablation against ref=link_COM.
            return rd.body_lin_vel_w_all[:, rd_ids]

        def contacts():
            c = env.contact_manager.is_contact(group, order=list(feet)).float()
            p = env.contact_manager.prev_is_contact(group, order=list(feet)).float()
            return c, p

        return read_a, read_b, contacts, list(feet), None

    from rlworld.rl.envs.mdp.rewards.mujoco.reward_terms import (
        _contact_order_matching_gait,
        _site_ids_matching_gait,
    )

    robot = env.scene_manager.get_entity(asset_cfg.name if asset_cfg is not None else "robot")
    contact_order = _contact_order_matching_gait(env, group)
    site_ids = _site_ids_matching_gait(env, asset_cfg)

    def read_a():
        return robot.data.site_pos_w[:, site_ids].clone(), robot.data.site_lin_vel_w[:, site_ids].clone()

    def contacts():
        c = env.contact_manager.is_contact(group, order=contact_order).float()
        p = env.contact_manager.prev_is_contact(group, order=contact_order).float()
        return c, p

    # Read-timing swap: recompute the derived buffers (cvel, xpos, …)
    # from the POST-integration qpos/qvel — the true end-of-step
    # instantaneous velocity, i.e. what newton/genesis read. qpos/qvel
    # are untouched, so the trajectory is preserved; only derived
    # fields and the solver warm-start are perturbed, which is
    # acceptable in a measurement run driven by open-loop actions.
    import mujoco_warp
    import warp as wp

    sim = env.scene_manager.sim

    def swap_read():
        with wp.ScopedDevice(sim.wp_device):
            mujoco_warp.forward(sim.wp_model, sim.wp_data)
        return read_a()[1]

    return read_a, None, contacts, list(contact_order), ("post-forward (true end-of-step)", swap_read)


def _phase_a(env, sim, preset):
    name, term = _find_slip_term(env)
    fn = env.reward_manager._resolved_fns[name]
    kind, asset_cfg, group, order, cmd_thr = _slip_sources(env, term)
    print(f"\n  [{sim}] A. sources")
    print(f"    term            : {name}  ({fn.__module__}.{fn.__name__})  weight={term.weight}")
    print(f"    formula kind    : {kind}")
    print(f"    contact_group   : {group}  order={order}")
    print(f"    command_thresh  : {cmd_thr}")
    if asset_cfg is not None:
        src = "SITES" if asset_cfg.site_ids is not None else "BODIES"
        names = asset_cfg.site_names if asset_cfg.site_ids is not None else asset_cfg.body_names
        print(f"    velocity source : {src}  {list(names or [])}")
    tracked = env.contact_manager._groups[group].tracked_names
    print(f"    contact tracked : {list(tracked)}")
    print(f"    contact backend : {type(env.contact_manager).__name__}")
    return kind, asset_cfg, group, order, cmd_thr


def _phase_b(env, sim, kind, read_feet, settle_steps, fd_steps):
    """Settled foot geometry + velocity-source integrity (per sim)."""
    zero = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    for _ in range(settle_steps):
        env.step(zero)

    if kind == "base":
        print(f"  [{sim}] B. (base-velocity formula — foot geometry n/a)")
        return None

    pos, _ = read_feet()
    mean_pos = pos.mean(dim=0).cpu().numpy()  # (n_feet, 3)

    # Velocity integrity: the term's instantaneous velocity against the
    # finite difference of the term's own positions over control steps.
    diffs = []
    prev_pos, _ = read_feet()
    prev_pos = prev_pos.clone()
    for _ in range(fd_steps):
        env.step(zero)
        p, v = read_feet()
        v_fd = (p - prev_pos) / env.control_dt
        prev_pos = p.clone()
        num = (v[..., :2] - v_fd[..., :2]).norm(dim=-1)
        den = v_fd[..., :2].norm(dim=-1).clamp_min(1e-4)
        diffs.append((num / den).mean().item())
    print(f"  [{sim}] B. settled foot positions (mean over envs, per foot):")
    for i, p in enumerate(mean_pos):
        print(f"      foot[{i}]  x={p[0]:+.4f}  y={p[1]:+.4f}  z={p[2]:+.4f}")
    print(f"      v_term vs v_fd mean rel dev over {fd_steps} steps: {np.mean(diffs):.3f}")
    return mean_pos


def _phase_cd(
    env,
    sim,
    kind,
    read_feet,
    reader_natural,
    asset_cfg,
    group,
    order,
    cmd_thr,
    steps,
    creep_steps,
    action_mode,
    action_scale,
    debug_pairing,
):
    """Decomposition at reward time + standing creep fingerprint."""
    from rlworld.rl.envs.mdp.rewards.common.reward_terms import _command_active

    mgr = env.reward_manager
    slip_name, _ = _find_slip_term(env)
    fn_name = getattr(mgr._resolved_fns[slip_name], "__name__", "")
    is_wtw = fn_name.startswith("wtw")
    rec = {
        "raw": [],
        "contact_frac": [],
        "v2_contact": [],
        "gate": [],
        "v2_all": [],
        "edge_frac": [],
        "v2_edge_share": [],
        "sum_A": [],
        "sum_B": [],
        "sum_FD": [],
        "e2_A": [],
        "e2_B": [],
        "e2_FD": [],
        "pool_num": [],
        "pool_den": [],
        "sum_C": [],
        "sum_swap": [],
    }

    probe = _make_term_probe(env, sim, kind, is_wtw, asset_cfg, group)
    orig = mgr._compute_weighted_reward
    prev_contact_buf = {"c": None}
    probe_buf = {"pos": None, "dump": 0, "div": 0}

    def patched(name, term_cfg):
        value = orig(name, term_cfg)
        if name != slip_name:
            return value
        rec["gate"].append(_command_active(env, cmd_thr).mean().item())
        rec["raw"].append((value / (term_cfg.weight * env.control_dt)).mean().item())

        if probe is None:
            # Base-velocity formula: natural-order contact stats + base speed.
            contact = _contact(env, group, asset_cfg, order).float()
            if is_wtw:
                prev = env.contact_manager.prev_is_contact(group, order=_term_contact_order(asset_cfg, order)).float()
                filt = torch.maximum(contact, prev)
            else:
                filt = contact
            prev_state = prev_contact_buf["c"]
            edge = (contact != prev_state).float() * filt if prev_state is not None else torch.zeros_like(filt)
            prev_contact_buf["c"] = contact.clone()
            rec["contact_frac"].append(filt.mean().item())
            rec["edge_frac"].append(edge.mean().item())
            rd = env.get_entity_data(asset_cfg.name if asset_cfg else "robot")
            rec["v2_contact"].append(rd.root_link_lin_vel_w[:, :2].norm(dim=1).mean().item())
            rec["v2_all"].append(rec["v2_contact"][-1])
            rec["v2_edge_share"].append(0.0)
            return value

        # Term-aligned decomposition: every tensor below shares the term's
        # own foot order, so contact and velocity columns pair exactly the
        # way the reward multiplies them.
        read_a, read_b, pcontacts, term_names, swap = probe
        contact, prev = pcontacts()
        filt = torch.maximum(contact, prev) if is_wtw else contact
        prev_state = prev_contact_buf["c"]
        edge = (contact != prev_state).float() * filt if prev_state is not None else torch.zeros_like(filt)
        prev_contact_buf["c"] = contact.clone()

        pos_a, vel_a = read_a()
        v2a = vel_a[..., :2].square().sum(dim=-1)

        rec["contact_frac"].append(filt.mean().item())
        rec["edge_frac"].append(edge.mean().item())
        m = filt.sum().clamp_min(1.0)
        rec["v2_contact"].append(((v2a * filt).sum() / m).item())
        rec["pool_num"].append((v2a * filt).sum().item())
        rec["pool_den"].append(filt.sum().item())
        rec["v2_all"].append(v2a.mean().item())
        total = (v2a * filt).sum().clamp_min(1e-12)
        rec["v2_edge_share"].append(((v2a * edge).sum() / total).item())

        rec["sum_A"].append((v2a * filt).sum(dim=1).mean().item())
        rec["e2_A"].append(v2a.mean().item())
        if read_b is not None:
            v2b = read_b()[..., :2].square().sum(dim=-1)
            rec["sum_B"].append((v2b * filt).sum(dim=1).mean().item())
            rec["e2_B"].append(v2b.mean().item())
        if probe_buf["pos"] is not None:
            v_fd = (pos_a - probe_buf["pos"]) / env.control_dt
            # Freshly reset envs teleported to spawn: their position
            # difference is the respawn jump, not travel. Rare (~0.01
            # resets/step) but each contributes v² ~ 10³, enough to
            # visibly inflate the mean. The FD reward term masks the
            # same frames.
            v_fd[env.episode_length_buf <= 1] = 0.0
            v2f = v_fd[..., :2].square().sum(dim=-1)
            rec["sum_FD"].append((v2f * filt).sum(dim=1).mean().item())
            rec["e2_FD"].append(v2f.mean().item())
        probe_buf["pos"] = pos_a
        # Handed to the outer loop, which knows this step's termination
        # outcome and buckets the slip mass by episode phase.
        probe_buf["per_env"] = (v2a * filt).sum(dim=1).detach()

        # F. read-timing swap (mjwarp cells only) — same engine, same
        # instant, same filter; only WHEN the velocity buffer was
        # computed changes. Runs last: the mujoco variant reruns
        # forward, which refreshes derived buffers for anything reading
        # them later in this same step.
        if swap is not None:
            vel_s = swap[1]()
            v2s = vel_s[..., :2].square().sum(dim=-1)
            rec["sum_swap"].append((v2s * filt).sum(dim=1).mean().item())

        # Independent cross-check, only meaningful when the diag reader's
        # column order IS the contact group's natural order (the fallback
        # tracked-names reader): the same sum computed from the natural
        # contact tensor and the RobotData reader. It is algebraically
        # identical to the term-aligned sum when the backend's labels are
        # truthful and nothing has corrupted the reordered read — the
        # multi-sim-process contamination this diag caught showed up
        # exactly as a drift between these two sums.
        if reader_natural:
            c_nat = env.contact_manager.is_contact(group).float()
            p_nat = env.contact_manager.prev_is_contact(group).float()
            filt_nat = torch.maximum(c_nat, p_nat) if is_wtw else c_nat
            _, v_diag = read_feet()
            v2_diag = v_diag[..., :2].square().sum(dim=-1)
            sum_c = (v2_diag * filt_nat).sum(dim=1).mean().item()
            sum_e = rec["sum_A"][-1]
            rec["sum_C"].append(sum_c)
            diverged = sum_e > 2.0 * sum_c + 0.02 or sum_c > 2.0 * sum_e + 0.02
            want_dump = probe_buf["dump"] < debug_pairing or (diverged and probe_buf["div"] < debug_pairing)
            if want_dump and debug_pairing > 0:
                if diverged and probe_buf["dump"] >= debug_pairing:
                    probe_buf["div"] += 1
                    tag = f"DIVERGED #{probe_buf['div']}"
                else:
                    probe_buf["dump"] += 1
                    tag = f"#{probe_buf['dump']}"
                tracked = list(env.contact_manager._groups[group].tracked_names)
                e_env = int((v2a * filt).sum(dim=1).argmax().item())

                def row(t):
                    return "[" + " ".join(f"{x:8.4f}" for x in t[e_env].tolist()) + "]"

                print(f"    [{sim}] pairing dump {tag} (env {e_env}):")
                print(f"      tracked order   : {tracked}")
                print(f"      term order      : {term_names}")
                print(f"      filt nat        : {row(filt_nat)}")
                print(f"      filt term       : {row(filt)}")
                print(f"      v2 diag         : {row(v2_diag)}   (tracked order)")
                print(f"      v2 term A       : {row(v2a)}   (term order)")
                print(f"      foot z(A)       : {row(pos_a[..., 2])}")
                print(f"      per-step sums   : C-pairing={sum_c:.6f}  E-pairing={sum_e:.6f}")
        return value

    mgr._compute_weighted_reward = patched
    zero = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    action_gen = torch.Generator(device="cpu").manual_seed(12345)

    def next_action():
        if action_mode == "zero":
            return zero
        # Identical N(0,1)*scale sequence in every sim: the generator is
        # CPU-seeded so the draw stream does not depend on the backend.
        a = torch.randn(env.num_envs, env.num_actions, generator=action_gen, device="cpu")
        return (action_scale * a).to(env.device)

    # Slip mass by episode phase. The reward is computed BEFORE
    # _reset_idx (world.step ordering), so a "terminal" frame's slip is
    # the falling robot's real physics, not a teleport artifact.
    since_reset = torch.full((env.num_envs,), 10**6, dtype=torch.long, device=env.device)
    buckets = {name: [0.0, 0] for name in ("terminal", "fresh", "steady")}
    resets_total = 0

    try:
        for _ in range(steps):
            ret = env.step(next_action())
            done = (ret[2] | ret[3]).bool()
            per_env = probe_buf.pop("per_env", None)
            if per_env is not None:
                fresh = (since_reset <= 2) & ~done
                steady = ~done & ~fresh
                for name, mask in (("terminal", done), ("fresh", fresh), ("steady", steady)):
                    buckets[name][0] += per_env[mask].sum().item()
                    buckets[name][1] += int(mask.sum().item())
                resets_total += int(done.sum().item())
            since_reset += 1
            since_reset[done] = 0
    finally:
        mgr._compute_weighted_reward = orig

    # D. standing creep: convention-free finite difference of the foot
    # positions, only over feet in contact, no reward code involved.
    creep = None
    if kind == "foot":
        speeds = []
        prev, _ = read_feet()
        prev = prev.clone()
        for _ in range(creep_steps):
            env.step(zero)
            p, _ = read_feet()
            v_fd = (p - prev)[..., :2].norm(dim=-1) / env.control_dt
            prev = p.clone()
            contact = _contact(env, group, asset_cfg, order).float()
            m = contact.sum().clamp_min(1.0)
            speeds.append(((v_fd * contact).sum() / m).item())
        creep = float(np.mean(speeds))

    summary = {
        "raw": float(np.mean(rec["raw"])),
        "contact_frac": float(np.mean(rec["contact_frac"])),
        "v2_contact": float(np.mean(rec["v2_contact"])),
        "v2_all": float(np.mean(rec["v2_all"])),
        "edge_frac": float(np.mean(rec["edge_frac"])),
        "v2_edge_share": float(np.mean(rec["v2_edge_share"])),
        "gate": float(np.mean(rec["gate"])),
        "creep": creep,
    }
    for key in ("sum_A", "sum_B", "sum_FD", "sum_C", "sum_swap", "e2_A", "e2_B", "e2_FD"):
        summary[key] = float(np.mean(rec[key])) if rec[key] else None
    summary["v2_contact_pooled"] = (
        float(np.sum(rec["pool_num"]) / max(np.sum(rec["pool_den"]), 1.0)) if rec["pool_num"] else None
    )
    summary["resets_per_step"] = resets_total / max(steps, 1)
    print(f"  [{sim}] C. decomposition over {steps} steps (means, actions={action_mode}):")
    raw_arr = np.abs(np.array(rec["raw"]))
    print(
        f"      raw slip value      : {summary['raw']:+.6f}   "
        f"(per-step |raw| p50={np.percentile(raw_arr, 50):.4f} p90={np.percentile(raw_arr, 90):.4f} "
        f"max={raw_arr.max():.3f})"
    )
    print(f"      P(filter)           : {summary['contact_frac']:.4f}   (WTW: contact|prev)")
    print(f"      P(edge frame)       : {summary['edge_frac']:.4f}")
    key = "E[v_xy^2 | filter]" if kind == "foot" else "E[base |v_xy|]"
    print(f"      {key:<20}: {summary['v2_contact']:.6f}")
    if summary["v2_contact_pooled"] is not None:
        print(f"      …pooled over frames : {summary['v2_contact_pooled']:.6f}   (heavy-tail-aware aggregate)")
    print(f"      E[v_xy^2] uncond    : {summary['v2_all']:.6f}")
    print(f"      edge share of slip  : {summary['v2_edge_share']:.3f}")
    print(f"      command gate        : {summary['gate']:.4f}")
    if summary["sum_A"] is not None:
        raw_mag = abs(summary["raw"]) if summary["raw"] else float("nan")

        def _fmt(v):
            return f"{v:.6f}" if v is not None else "     n/a"

        print(f"  [{sim}] E. velocity-source ablation (term order+filter, means):")
        print(f"      |term raw|          : {raw_mag:.6f}")
        print(f"      Σ filt·v²  A(term)  : {_fmt(summary['sum_A'])}   (A/raw = {summary['sum_A'] / raw_mag:.3f})")
        if summary["sum_C"] is not None:
            print(
                f"      Σ filt·v²  C(diag)  : {_fmt(summary['sum_C'])}   (C/A = {summary['sum_C'] / summary['sum_A']:.3f})"
            )
        if summary["sum_B"] is not None:
            print(
                f"      Σ filt·v²  B(rd)    : {_fmt(summary['sum_B'])}   (B/A = {summary['sum_B'] / summary['sum_A']:.3f})"
            )
        if summary["sum_FD"] is not None:
            print(
                f"      Σ filt·v²  FD(pos)  : {_fmt(summary['sum_FD'])}   (FD/A = {summary['sum_FD'] / summary['sum_A']:.3f})"
            )
        if summary["sum_swap"] is not None:
            label = probe[4][0]
            print(
                f"      Σ filt·v²  F(swap)  : {_fmt(summary['sum_swap'])}   "
                f"(swap/A = {summary['sum_swap'] / summary['sum_A']:.3f})   [{label}]"
            )
        print(
            f"      E[v²] uncond A/B/FD : {_fmt(summary['e2_A'])} / {_fmt(summary['e2_B'])} / {_fmt(summary['e2_FD'])}"
        )
        print(
            "      (A replicates the LEGACY instantaneous read. For terms already\n"
            "       converted to the finite-difference definition, raw matches the\n"
            "       FD row instead and A/raw < 1 is expected — A then shows how\n"
            "       much of the real travel the old instantaneous read captured.)"
        )
        total_mass = sum(b[0] for b in buckets.values())
        if total_mass > 0:
            print(f"  [{sim}] E2. slip mass by episode phase (A source, resets/step={summary['resets_per_step']:.2f}):")
            for name, (mass, count) in buckets.items():
                frac_frames = count / max(sum(b[1] for b in buckets.values()), 1)
                print(
                    f"      {name:<9}: share={mass / total_mass:6.1%}  frames={frac_frames:6.1%}  "
                    f"mean={mass / max(count, 1):.4f}"
                )
    if creep is not None:
        print(f"  [{sim}] D. standing creep (FD, contacted feet): {creep:.5f} m/s")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="go2_gait")
    ap.add_argument("--sims", default="genesis,newton,mujoco")
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--settle-steps", type=int, default=50)
    ap.add_argument("--fd-steps", type=int, default=20)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument(
        "--actions",
        default="zero",
        choices=("zero", "random"),
        help="zero = standing creep regime; random = identical N(0,1)*scale action "
        "sequence in every sim (the iteration-1 rollout regime).",
    )
    ap.add_argument("--action-scale", type=float, default=1.0)
    ap.add_argument("--creep-steps", type=int, default=200)
    ap.add_argument(
        "--debug-pairing",
        type=int,
        default=0,
        help="Print N per-step column-pairing dumps (contact/velocity rows in both the natural and the term order).",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sims = [s.strip() for s in args.sims.split(",")]
    if len(sims) > 1:
        print(
            "WARNING: building multiple simulators in one process has been\n"
            "observed to contaminate the reward-time reads of sims built\n"
            "after the first (newton-after-genesis inflated the reordered\n"
            "contact read ~5x while all natural-order statistics stayed\n"
            "clean). Trust single-sim invocations; the C/A cross-check\n"
            "below flags the contamination when it occurs."
        )
    results: dict[str, dict] = {}
    foot_geo: dict[str, np.ndarray] = {}

    for sim in sims:
        print(f"\n{'=' * 72}\nBuilding [{sim}] {args.preset!r} (num_envs={args.num_envs}) ...")
        torch.manual_seed(args.seed)
        env = _build_env(args.preset, sim, args.num_envs)
        env.reset()
        kind, asset_cfg, group, order, cmd_thr = _phase_a(env, sim, args.preset)
        read_feet, src_desc = (None, "n/a") if kind == "base" else _make_foot_reader(env, asset_cfg, group)
        reader_natural = src_desc.startswith("contact-group bodies")
        print(f"    diag foot reader: {src_desc}")
        geo = _phase_b(env, sim, kind, read_feet, args.settle_steps, args.fd_steps)
        if geo is not None:
            foot_geo[sim] = geo
        results[sim] = _phase_cd(
            env,
            sim,
            kind,
            read_feet,
            reader_natural,
            asset_cfg,
            group,
            order,
            cmd_thr,
            args.steps,
            args.creep_steps,
            args.actions,
            args.action_scale,
            args.debug_pairing,
        )
        del env
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── verdict ──────────────────────────────────────────────────────
    print(f"\n{'=' * 72}\nVERDICT [{args.preset}]")
    if len(foot_geo) >= 2:
        sims_g = list(foot_geo)
        ref = foot_geo[sims_g[0]]
        print("  measurement-point offsets vs " + sims_g[0] + " (max |Δ| per axis, m):")
        for s in sims_g[1:]:
            d = np.abs(foot_geo[s] - ref).max(axis=0)
            print(f"    {s:<8} dx={d[0]:.4f}  dy={d[1]:.4f}  dz={d[2]:.4f}")

    ref_sim = sims[0]
    print(f"\n  ratios vs {ref_sim}:")
    print(
        f"  {'sim':<8} {'raw slip':>10} {'P(filter)':>11} {'slip pooled':>12} {'v2 uncond':>10} "
        f"{'edge':>6} {'resets/st':>10} {'creep':>8} {'creep^2':>9}"
    )
    for s in sims:
        r = results[s]
        ref = results[ref_sim]

        def ratio(a, b):
            return a / b if b not in (0.0, None) and a is not None else float("nan")

        creep_r = ratio(r["creep"], ref["creep"]) if r["creep"] is not None else float("nan")
        pooled = r["v2_contact_pooled"] if r["v2_contact_pooled"] is not None else r["v2_contact"]
        pooled_ref = ref["v2_contact_pooled"] if ref["v2_contact_pooled"] is not None else ref["v2_contact"]
        print(
            f"  {s:<8} {ratio(r['raw'], ref['raw']):>10.3f} {ratio(r['contact_frac'], ref['contact_frac']):>11.3f} "
            f"{ratio(pooled, pooled_ref):>12.3f} {ratio(r['v2_all'], ref['v2_all']):>10.3f} "
            f"{ratio(r['edge_frac'], ref['edge_frac']):>6.2f} {r['resets_per_step']:>10.2f} "
            f"{creep_r:>8.3f} {creep_r**2:>9.3f}"
        )
    print(
        "\n  Reading: if 'raw slip' ratio ≈ 'creep^2' ratio while P(contact),\n"
        "  gate, and the measurement points match, the divergence is the\n"
        "  engines' friction creep (constraint-solver residual), reproducing\n"
        "  the go2 audit on this preset. If instead P(contact) or the\n"
        "  geometry differs, THAT sim's contact/measurement plumbing is the\n"
        "  cause and the offending input is printed above."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
