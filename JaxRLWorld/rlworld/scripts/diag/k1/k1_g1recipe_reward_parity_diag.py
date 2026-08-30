"""K1 G1-recipe first-step reward parity: genesis-high feet terms + mujoco-low joint_pos_limits.

Symptoms (training, first steps): genesis's feet_clearance / feet_slip /
feet_swing_height / track_ang_vel all sit far above newton+mujoco, while
mujoco's joint_pos_limits is abnormally low (more negative) than both.

The diag captures, per sim, the EXACT tensors the reward manager consumed:

1. Reward shims installed BEFORE the config is built, on the per-backend
   function objects the preset dispatches (genesis/newton →
   ``rewards.{sim}.mjlab_rewards.*_mjlab``; mujoco →
   ``rewards.mujoco.reward_terms.*``; tracking terms → common). Each call
   records output stats; the joint_pos_limits shim ALSO recomputes both
   candidate formulas from live state at call time:
       cand_hard = -Σ max(lo_hard - q, 0) + max(q - hi_hard, 0)
       cand_soft = same vs RobotData.soft_joint_pos_limits
   Known source-level suspect: mujoco's ``joint_pos_limits`` reads
   soft_joint_pos_limits (mid±half·0.95 on K1) while
   ``joint_pos_limits_mjlab`` delegates to penalize_joint_pos_limits_l1
   = HARD limits × soft_limit_factor(default 1.0). The attribution table
   proves/refutes numerically.

2. Command distribution at t=0: every feet term is gated by
   ``command_threshold=0.05``; the per-sim command RNG streams are NOT
   aligned (known: gait-phase counters differ), so a different fraction
   of envs passing the gate inflates feet terms mechanically. Dumped:
   per-axis command stats + gate-pass fraction + first 8 raw commands.

3. Canonical state around the first step (post-reset and post-step):
   resolved feet ids/names (ORDER!), per-foot z, contact found/force,
   foot xy speed (slip proxy), base ang vel vs command, joint deviation
   and L1 violation vs hard AND soft limits.

4. 50-step zero-action settle time series for all of the above.

Run (server):
    python -m rlworld.scripts.diag.k1.k1_g1recipe_reward_parity_diag
"""

from __future__ import annotations

import argparse
import functools
import importlib
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

_MODULE = "rlworld.scripts.diag.k1.k1_g1recipe_reward_parity_diag"
# The *_fix cells are retired: both in-diag candidate fixes were applied
# to the framework (writer zero_velocity=False, joint_pos_limits on soft
# limits) and verified plain ≡ fix, so the plain cells ARE the fixed
# behavior now.
# Plain genesis now runs integrator=implicitfast via the K1 preset fix;
# the genesis_* variant cells remain invokable via --cell for A/B checks.
_SIMS = ("genesis", "newton", "mujoco")
_SIM_KEY = {
    "genesis": "Genesis",
    "newton": "Newton",
    "mujoco": "MujocoEnv",
    "genesis_fix": "Genesis",
    "newton_fix": "Newton",
}

# Terms shown in the FIRST-call table, per task.
_TASK_TERMS = {
    "g1recipe": (
        "feet_clearance",
        "feet_swing_height",
        "feet_slip",
        "feet_pair_collision",
        "track_ang_vel",
        "track_lin_vel",
        "joint_pos_limits",
    ),
    "joystick": (
        "feet_phase",
        "feet_air_time",
        "feet_slip",
        "collision",
        "track_lin_vel",
        "track_ang_vel",
    ),
}


# term name -> (module, function) per sim family
def _term_map(sim: str, task: str) -> dict[str, tuple[str, str]]:
    common = "rlworld.rl.envs.mdp.rewards.common.reward_terms"
    k1mod = "rlworld.rl.envs.mdp.rewards.k1_locomotion"
    if task == "joystick":
        # The joystick task's contested terms are ALL sim-agnostic
        # private functions (single implementation shared by the three
        # backends) — the shim verifies their inputs, not per-sim code.
        return {
            "feet_phase": (k1mod, "feet_phase_bezier"),
            "feet_air_time": (k1mod, "K1FeetAirTime"),
            "feet_slip": (k1mod, "feet_slip_base_vel"),
            "collision": (k1mod, "contact_pair_penalty"),
            "track_lin_vel": (k1mod, "track_lin_vel_xy_exp"),
            "track_ang_vel": (k1mod, "track_ang_vel_z_exp"),
        }
    if sim == "mujoco":
        mod = "rlworld.rl.envs.mdp.rewards.mujoco.reward_terms"
        return {
            "feet_clearance": (mod, "feet_clearance"),
            "feet_swing_height": (mod, "feet_swing_height"),
            "feet_slip": (mod, "feet_slip"),
            "joint_pos_limits": (mod, "joint_pos_limits"),
            "track_ang_vel": (common, "track_ang_vel"),
            "track_lin_vel": (common, "track_lin_vel"),
            "feet_pair_collision": (k1mod, "contact_pair_penalty"),
        }
    mod = f"rlworld.rl.envs.mdp.rewards.{sim}.mjlab_rewards"
    return {
        "feet_clearance": (mod, "feet_clearance_mjlab"),
        "feet_swing_height": (mod, "feet_swing_height_mjlab"),
        "feet_slip": (mod, "feet_slip_mjlab"),
        "joint_pos_limits": (mod, "joint_pos_limits_mjlab"),
        "track_ang_vel": (common, "track_ang_vel"),
        "track_lin_vel": (common, "track_lin_vel"),
        "feet_pair_collision": (k1mod, "contact_pair_penalty"),
    }


def _stage(msg: str) -> None:
    print(f"[STAGE] {msg}", flush=True)


# ── Child ───────────────────────────────────────────────────────────


def run_cell(sim: str, num_envs: int, settle_steps: int, seed: int, task: str) -> dict:
    import torch

    torch.manual_seed(seed)
    fix = sim.endswith("_fix")
    if fix:
        sim = sim[: -len("_fix")]
    # Genesis variants (stock genesis with ONE knob changed):
    #   _noprune      — contact_pruning_tolerance=None (REFUTED: bit-identical)
    #   _implicitfast — integrator=implicitfast, the MuJoCo-consistent one
    #                   (preset default approximate_implicitfast folds joint
    #                   damping into M BEFORE the constraint solve — extra
    #                   dissipation; candidate for the −4% leg-qvel gap)
    #   _ifast_ellip  — implicitfast + elliptic cone (full mjwarp-recipe align)
    variant = sim if sim.startswith("genesis_") else None
    if variant:
        sim = "genesis"
    _stage(f"cell start: {sim} num_envs={num_envs} fix={fix} variant={variant}")

    if fix and sim == "genesis":
        # CANDIDATE FIX A (in-diag only): Genesis's set_pos / set_quat /
        # set_dofs_position default to zero_velocity=True, which zeroes
        # ALL entity dof velocities — so the reset_joints event silently
        # wipes the root velocity that reset_root just randomized.
        # Re-declare the writer methods with zero_velocity=False (the
        # writer must only write what was asked; velocities are written
        # explicitly by the event protocol).
        from rlworld.rl.envs.genesis.robot_state_writer import GenesisRobotStateWriter as _W

        def _set_dof_positions(self, values, env_ids=None):
            self._entity.set_dofs_position(
                position=values,
                dofs_idx_local=self._actuated_dof_ids,
                envs_idx=env_ids,
                zero_velocity=False,
            )

        def _set_root_pose(self, pos, quat_wxyz, env_ids=None):
            self._entity.set_pos(pos, envs_idx=env_ids, zero_velocity=False)
            self._entity.set_quat(quat_wxyz, envs_idx=env_ids, zero_velocity=False)

        _W.set_dof_positions = _set_dof_positions
        _W.set_root_pose = _set_root_pose
        _stage("fix A installed: genesis writer zero_velocity=False")

    if fix:
        # CANDIDATE FIX B (in-diag only): unify joint_pos_limits on the
        # SOFT limits (mid ± half·factor), the mujoco/IsaacLab semantic.
        # Replaces the *_mjlab implementation (hard × factor) before the
        # recording shim wraps it, so the shim captures the fixed fn.
        mj_mod = importlib.import_module(f"rlworld.rl.envs.mdp.rewards.{sim}.mjlab_rewards")

        def _joint_pos_limits_soft(env, *a, **kw):
            rd = env.get_robot_data()
            q_now = rd.joint_pos
            lim = rd.soft_joint_pos_limits
            lo, hi = (lim[0], lim[1]) if isinstance(lim, tuple | list) else (lim[..., 0], lim[..., 1])
            viol = (lo - q_now).clamp(min=0.0) + (q_now - hi).clamp(min=0.0)
            return -viol.sum(dim=-1)

        mj_mod.joint_pos_limits_mjlab = _joint_pos_limits_soft
        _stage("fix B installed: joint_pos_limits_mjlab -> soft limits")

    records: dict[str, list[dict]] = defaultdict(list)
    term_meta: dict[str, dict] = {}
    env_holder: dict = {}

    def _q(t: torch.Tensor) -> list[float]:
        qs = torch.tensor([0.1, 0.5, 0.9], device=t.device, dtype=torch.float32)
        return [round(float(v), 6) for v in torch.quantile(t.float(), qs).tolist()]

    def _lims(lim):
        """Normalize the two RobotData limit conventions: genesis/newton
        return (lower, upper) tuples of (J,) tensors, mjlab returns an
        (N, J, 2) tensor."""
        if isinstance(lim, tuple | list):
            return lim[0], lim[1]
        return lim[..., 0], lim[..., 1]

    def _hard_lims(rd):
        """Hard limits. MujocoRobotData deliberately does NOT expose them
        (mjlab only stores soft limits), so on mjlab we read jnt_range
        from the host mj_model (precomputed in canonical actuated order
        and stashed in env_holder after env build)."""
        if "hard_lims" in env_holder:
            return env_holder["hard_lims"]
        return _lims(rd.joint_pos_limits)

    def _canonical_snapshot() -> dict:
        """The quantities every diverging term depends on, read via the
        canonical accessors at reward-call time (captures mjlab staleness
        relative to the post-step read in the rollout loop)."""
        env = env_holder["env"]
        rd = env.get_robot_data()
        snap: dict = {}
        if "feet_ids" in env_holder:
            fz = rd.body_pos_w_by_ids(env_holder["feet_ids"])[..., 2]
            fv = rd.body_lin_vel_w_by_ids(env_holder["feet_ids"])[..., :2].norm(dim=-1)
            snap["foot_z_mean"] = [round(float(v), 5) for v in fz.mean(dim=0).tolist()]
            snap["foot_xyspeed_mean"] = [round(float(v), 5) for v in fv.mean(dim=0).tolist()]
        contact = env.contact_manager.is_contact("feet_ground_contact")
        snap["contact_frac"] = [round(float(v), 4) for v in contact.float().mean(dim=0).tolist()]
        cmd = env.command_manager.get_term("velocity").command
        snap["cmd_abs_mean"] = [round(float(v), 4) for v in cmd.abs().mean(dim=0).tolist()]
        snap["cmd_gate_frac_005"] = round(float((cmd.norm(dim=-1) > 0.05).float().mean()), 4)
        snap["base_ang_vel_z_absmean"] = round(float(rd.root_link_ang_vel_w[:, 2].abs().mean()), 4)
        return snap

    def _record(term: str, out, kwargs_repr: dict) -> None:
        rec = {
            "out_mean": float(out.mean()),
            "out_q10_50_90": _q(out),
        }
        if term not in term_meta:
            term_meta[term] = {"kwargs": kwargs_repr}
        rec.update(_canonical_snapshot())
        if term == "joint_pos_limits":
            rd = env_holder["env"].get_robot_data()
            q_now = rd.joint_pos
            lo_h, hi_h = _hard_lims(rd)
            lo_s, hi_s = _lims(rd.soft_joint_pos_limits)
            for tag, (lo, hi) in (("hard", (lo_h, hi_h)), ("soft", (lo_s, hi_s))):
                viol = (lo - q_now).clamp(min=0.0) + (q_now - hi).clamp(min=0.0)
                rec[f"cand_{tag}_mean"] = float((-viol.sum(dim=-1)).mean())
            rec["limits_first4_hard"] = [
                [round(float(a), 4), round(float(b), 4)]
                for a, b in zip(
                    lo_h.reshape(-1, lo_h.shape[-1])[0, :4].tolist(), hi_h.reshape(-1, hi_h.shape[-1])[0, :4].tolist()
                )
            ]
            rec["limits_first4_soft"] = [
                [round(float(a), 4), round(float(b), 4)]
                for a, b in zip(
                    lo_s.reshape(-1, lo_s.shape[-1])[0, :4].tolist(), hi_s.reshape(-1, hi_s.shape[-1])[0, :4].tolist()
                )
            ]
        records[term].append(rec)

    def _install(term: str, mod_name: str, fn_name: str) -> None:
        import inspect

        mod = importlib.import_module(mod_name)
        orig = getattr(mod, fn_name)

        if inspect.isclass(orig):
            # Stateful class term (feet_swing_height): the manager
            # instantiates it and calls instance(env) — patch __call__
            # on the class so the manager's instance is captured.
            orig_call = orig.__call__

            def call_shim(self, env, *a, __orig_call=orig_call, __term=term, **kw):
                out = __orig_call(self, env, *a, **kw)
                kwargs_repr = {k: repr(v)[:160] for k, v in vars(self).items() if not hasattr(v, "shape")}
                kwargs_repr["__class__"] = f"{mod_name}.{fn_name}"
                _record(__term, out, kwargs_repr)
                return out

            orig.__call__ = call_shim
            return

        def shim(env, *a, __orig=orig, __term=term, **kw):
            out = __orig(env, *a, **kw)
            kwargs_repr = {k: repr(v)[:160] for k, v in kw.items()}
            kwargs_repr["__func__"] = f"{mod_name}.{fn_name}"
            acfg = kw.get("asset_cfg")
            if acfg is not None and getattr(acfg, "body_ids", None) is not None:
                env_holder.setdefault("feet_ids", acfg.body_ids)
                kwargs_repr["resolved_body_names"] = repr(getattr(acfg, "body_names", None))
            _record(__term, out, kwargs_repr)
            return out

        # Expose the ORIGINAL signature through the shim: the reward
        # manager inspect.signature()s the term function to discover and
        # inject resolved defaults for selector-valued parameters (e.g.
        # mujoco joint_pos_limits' asset_cfg=_DEFAULT_SELECTOR). A bare
        # (*a, **kw) shim hides that default and the unresolved selector
        # reaches the term (AttributeError: joint_ids_native).
        functools.update_wrapper(shim, orig)
        setattr(mod, fn_name, shim)

    for term, (m, f) in _term_map(sim, task).items():
        _install(term, m, f)
    _stage("shims installed")

    from rlworld.rl.configs.base_config import iter_terms
    from rlworld.rl.configs.rewards import RewardTermConfig

    if task == "joystick":
        from rlworld.rl.configs.presets.k1_joystick.base import K1JoystickConfig as _Cfg
    else:
        from rlworld.rl.configs.presets.k1_joystick.g1_recipe import K1G1RecipeConfig as _Cfg

    cfgs = _Cfg(sim_type=sim, num_envs=num_envs, seed=seed).build()
    if variant == "genesis_noprune":
        cfgs.scene.rigid_options.contact_pruning_tolerance = None
    elif variant == "genesis_implicitfast":
        import genesis as gs

        cfgs.scene.rigid_options.integrator = gs.integrator.implicitfast
    elif variant == "genesis_ifast_ellip":
        import genesis as gs

        cfgs.scene.rigid_options.integrator = gs.integrator.implicitfast
        cfgs.scene.rigid_options.friction_cone = gs.friction_cone.elliptic
    elif variant == "genesis_nomasscom":
        # Disable ONLY the body mass / COM domain-randomization terms (the
        # ones migrated to Genesis's absolute set_links_mass / set_links_COM
        # under the #3237 engine bump). Everything else — armature/friction/
        # kp/kd DR, batch_links_info, the current engine — is untouched.
        # Setting a term attribute to None is the documented disable in
        # iter_terms. If this cell STANDS while plain genesis collapses, the
        # mass/COM DR path is the trigger; if it still collapses, the #3237
        # engine physics is, independent of DR.
        for _t in ("dr_body_com", "dr_trunk_mass", "dr_link_mass"):
            if hasattr(cfgs.event, _t):
                setattr(cfgs.event, _t, None)
    elif variant is not None:
        raise ValueError(f"unknown genesis variant: {variant}")

    wiring: dict = {"reward_terms": {}}
    for name, term in iter_terms(cfgs.reward, RewardTermConfig).items():
        wiring["reward_terms"][name] = {
            "func": f"{term.func.__module__}.{getattr(term.func, '__name__', type(term.func).__name__)}",
            "weight": term.weight,
            "params": {k: repr(v)[:200] for k, v in (term.params or {}).items()},
        }
    if sim == "mujoco":
        mj_cfg = cfgs.scene.mjlab_sim_cfg.mujoco
        wiring["solver"] = (
            f"iterations={mj_cfg.iterations} ls_iterations={mj_cfg.ls_iterations} impratio={mj_cfg.impratio} "
            f"cone={mj_cfg.cone} integrator={mj_cfg.integrator} disableflags={mj_cfg.disableflags}"
        )
    elif sim == "newton":
        nt = cfgs.scene.solver_cfg
        wiring["solver"] = (
            f"iterations={nt.iterations} ls_iterations={nt.ls_iterations} impratio={nt.impratio} "
            f"cone={nt.cone} integrator={nt.integrator} use_mujoco_contacts={nt.use_mujoco_contacts}"
        )
    else:
        ro = cfgs.scene.rigid_options
        wiring["solver"] = (
            f"integrator={ro.integrator} constraint_solver={ro.constraint_solver} iterations={ro.iterations} "
            f"constraint_timeconst={ro.constraint_timeconst} "
            f"contact_pruning_tolerance={ro.contact_pruning_tolerance} enable_multi_contact={ro.enable_multi_contact} "
            f"friction_cone={ro.friction_cone} impratio={ro.impratio}"
        )
    _stage("config built")

    from rlworld.rl.evals.sim_initializers import get_initializer

    env = get_initializer(_SIM_KEY[sim]).init_environment(cfgs)
    env_holder["env"] = env
    wiring["decimation"] = env.decimation
    wiring["physics_dt"] = env.physics_dt
    _stage("env built")

    # ---- MASS / ARMATURE / DOF-PARAM AUDIT ----------------------------
    # Do the three backends end up with the same rigid bodies (link
    # fusion / merging!), masses, armature, damping, frictionloss?
    # genesis reports per-env min/mean/max because startup DR writes
    # per-env dofs_info there, while the mjlab/newton HOST model keeps
    # raw values (their DR lives on the device copy) — the genesis MIN
    # should sit on the raw base value if the injection paths agree.
    audit: dict = {}
    act_leafs = [n.rsplit("/", 1)[-1] for n in env.act_manager.actuated_joint_names]
    if sim == "genesis":
        entity = env.get_robot_state_writer()._entity
        dof_ids = env.get_robot_state_writer()._actuated_dof_ids
        audit["n_links"] = len(entity.links)

        # batch_links_info=True makes get_mass() / link.get_mass() return a
        # per-env (n_envs,) tensor (each env carries its own DR'd mass), so
        # reduce to the cross-env mean for the scalar comparison table. A
        # scalar (no per-env DR) reshapes to numel 1 and means to itself.
        def _mass_mean(t) -> float:
            return round(float(torch.as_tensor(t, device=env.device).float().reshape(-1).mean()), 6)

        audit["total_mass"] = _mass_mean(entity.get_mass())
        audit["link_masses"] = {l.name.rsplit("/", 1)[-1]: _mass_mean(l.get_mass()) for l in entity.links}

        def _dofstat(t):
            t = torch.as_tensor(t, device=env.device).float()
            if t.dim() == 1:
                t = t.unsqueeze(0)
            return {
                "min": [round(float(v), 6) for v in t.min(dim=0).values.tolist()],
                "mean": [round(float(v), 6) for v in t.mean(dim=0).tolist()],
                "max": [round(float(v), 6) for v in t.max(dim=0).values.tolist()],
            }

        audit["armature"] = _dofstat(entity.get_dofs_armature(dofs_idx_local=dof_ids))
        audit["damping"] = _dofstat(entity.get_dofs_damping(dofs_idx_local=dof_ids))
        audit["frictionloss"] = _dofstat(entity.get_dofs_frictionloss(dofs_idx_local=dof_ids))
    else:
        import mujoco as _mj2

        mjm_a = env.scene_manager.solver.mj_model if sim == "newton" else env.scene_manager.mj_model
        body_masses = {}
        for b in range(mjm_a.nbody):
            nm = _mj2.mj_id2name(mjm_a, _mj2.mjtObj.mjOBJ_BODY, b) or f"body{b}"
            leaf = nm.rsplit("/", 1)[-1]
            if leaf in ("world", "worldbody", ""):
                continue
            body_masses[leaf] = round(float(mjm_a.body_mass[b]), 6)
        audit["n_links"] = len(body_masses)
        audit["total_mass"] = round(float(sum(body_masses.values())), 6)
        audit["link_masses"] = body_masses
        jnt_names = [(_mj2.mj_id2name(mjm_a, _mj2.mjtObj.mjOBJ_JOINT, j) or f"jnt{j}") for j in range(mjm_a.njnt)]

        def _find_jnt(leaf: str) -> int:
            # mujoco host model: "robot/Left_Hip_Pitch" → slash-leaf match.
            # newton host model: underscore-joined body path + joint name
            # ("K1_worldbody_Trunk_Left_Hip_Pitch_Left_Hip_Pitch") → the
            # name ENDS with "_<joint>" (a bare contains-match is ambiguous
            # because body names repeat the joint names in the path).
            exact = [j for j, nm in enumerate(jnt_names) if nm.rsplit("/", 1)[-1] == leaf]
            if len(exact) == 1:
                return exact[0]
            suffix = [j for j, nm in enumerate(jnt_names) if nm.endswith("_" + leaf)]
            if len(suffix) == 1:
                return suffix[0]
            raise KeyError(f"joint leaf {leaf!r}: exact={exact} suffix={suffix}; host joints={jnt_names}")

        dofadr = [int(mjm_a.jnt_dofadr[_find_jnt(n)]) for n in act_leafs]
        audit["armature"] = {"raw_host": [round(float(mjm_a.dof_armature[d]), 6) for d in dofadr]}
        audit["damping"] = {"raw_host": [round(float(mjm_a.dof_damping[d]), 6) for d in dofadr]}
        audit["frictionloss"] = {"raw_host": [round(float(mjm_a.dof_frictionloss[d]), 6) for d in dofadr]}
    audit["actuated_leaf_order"] = act_leafs
    wiring["mass_audit"] = audit
    _stage("mass/armature audit done")

    if sim == "mujoco":
        # MujocoRobotData raises on .joint_pos_limits (mjlab stores soft
        # only) — precompute HARD limits from the host mj_model jnt_range
        # in canonical actuated order for the attribution table.
        import mujoco as _mj

        mjm = env.scene_manager.mj_model
        name_to_range = {}
        for j in range(mjm.njnt):
            if int(mjm.jnt_type[j]) == 0:  # free joint
                continue
            nm = (_mj.mj_id2name(mjm, _mj.mjtObj.mjOBJ_JOINT, j) or f"jnt{j}").rsplit("/", 1)[-1]
            name_to_range[nm] = (float(mjm.jnt_range[j, 0]), float(mjm.jnt_range[j, 1]))
        leafs = [n.rsplit("/", 1)[-1] for n in env.act_manager.actuated_joint_names]
        env_holder["hard_lims"] = (
            torch.tensor([name_to_range[n][0] for n in leafs], device=env.device),
            torch.tensor([name_to_range[n][1] for n in leafs], device=env.device),
        )

    # Resolve feet independently of the shims so state capture works from
    # step 0 even before the first reward call.
    from rlworld.rl.configs.scene import SceneEntitySelector

    feet_sel = env.resolve_selector(
        SceneEntitySelector(name="robot", body_names=tuple(cfgs_robot_foot_names(cfgs)), preserve_order=True)
    )
    env_holder["feet_ids"] = feet_sel.body_ids
    env_holder["knee_ids"] = [
        i for i, n in enumerate(n.rsplit("/", 1)[-1] for n in env.act_manager.actuated_joint_names) if "Knee_Pitch" in n
    ]
    wiring["feet_resolved"] = {
        "names_in_order": list(feet_sel.body_names or []),
        "ids": [int(i) for i in feet_sel.body_ids.tolist()],
    }

    env.reset()
    torch.cuda.synchronize()
    _stage("reset done")

    def _gait_capture(rd) -> dict:
        """Joystick-only: the feet_phase reward's exact inputs and value.

        The gait clock (per-episode freq ~ U(1.25, 1.75), phase advance,
        freeze gate) and the bezier profile are ONE shared implementation
        across the sims — this dump proves the clock streams are aligned
        (freq/phase identical) and recomputes the exact term value
        exp(-sum((foot_z - rz)^2) / 0.01) outside the reward manager.
        """
        from rlworld.rl.envs.mdp.rewards.k1_locomotion import _bezier_rz

        gp = env.command_manager.get_term("gait_phase")
        fz = rd.body_pos_w_by_ids(env_holder["feet_ids"])[..., 2]
        rz = _bezier_rz(gp.command, 0.12)
        val = torch.exp(-torch.sum(torch.square(fz - rz), dim=1) / 0.01)
        return {
            "gait_freq_first8": [round(float(v), 5) for v in gp.freq[:8].tolist()],
            "gait_freq_mean": round(float(gp.freq.mean()), 6),
            "gait_phase_first4": [[round(float(x), 5) for x in row] for row in gp.command[:4].tolist()],
            "bezier_rz_first4": [[round(float(x), 5) for x in row] for row in rz[:4].tolist()],
            "feet_phase_direct_mean": round(float(val.mean()), 6),
        }

    def _obs_capture() -> dict:
        """Per-term observation stats — the ONE layer the reward-parity
        shims never covered. At synchronized deterministic states the
        per-term means must match across sims; a broken term (frame
        flip, stale read, garbage after a simulator bump) shows up as a
        mismatched row, and NaN/Inf are counted explicitly because a
        single non-finite obs silently kills policy learning.
        """
        out: dict = {}
        om = env.obs_manager
        if not om._is_term_indices_built:
            om._build_term_indices()
        obs = om.get_observation()
        for gname, tensor in obs.items():
            out[f"__{gname}__nonfinite"] = int((~torch.isfinite(tensor)).sum())
            term_slices = om._group_term_indices.get(gname)
            if not term_slices:
                sl = tensor.float()
                out[f"{gname}/<whole>"] = [
                    round(float(sl.mean()), 5),
                    round(float(sl.std()), 5),
                    round(float(sl.abs().max()), 3),
                ]
                continue
            for tname, (a, b) in term_slices.items():
                sl = tensor[:, a:b].float()
                out[f"{gname}/{tname}"] = [
                    round(float(sl.mean()), 5),
                    round(float(sl.std()), 5),
                    round(float(sl.abs().max()), 3),
                ]
        return out

    def state_capture(tag: str) -> dict:
        rd = env.get_robot_data()
        fz = rd.body_pos_w_by_ids(env_holder["feet_ids"])[..., 2]
        fv = rd.body_lin_vel_w_by_ids(env_holder["feet_ids"])[..., :2].norm(dim=-1)
        contact = env.contact_manager.is_contact("feet_ground_contact")
        force = env.contact_manager.contact_force("feet_ground_contact").norm(dim=-1)
        cmd = env.command_manager.get_term("velocity").command
        q_now = rd.joint_pos
        lo_h, hi_h = _hard_lims(rd)
        lo_s, hi_s = _lims(rd.soft_joint_pos_limits)
        viol_h = (lo_h - q_now).clamp(min=0.0) + (q_now - hi_h).clamp(min=0.0)
        viol_s = (lo_s - q_now).clamp(min=0.0) + (q_now - hi_s).clamp(min=0.0)
        return {
            "tag": tag,
            "foot_z_mean": [round(float(v), 5) for v in fz.mean(dim=0).tolist()],
            "foot_z_q10_50_90": _q(fz.flatten()),
            "foot_xyspeed_mean": [round(float(v), 5) for v in fv.mean(dim=0).tolist()],
            "contact_frac": [round(float(v), 4) for v in contact.float().mean(dim=0).tolist()],
            "force_mean": [round(float(v), 3) for v in force.mean(dim=0).tolist()],
            "cmd_abs_mean_per_axis": [round(float(v), 4) for v in cmd.abs().mean(dim=0).tolist()],
            "cmd_gate_frac_005": round(float((cmd.norm(dim=-1) > 0.05).float().mean()), 4),
            "cmd_first8": [[round(float(x), 3) for x in row] for row in cmd[:8].tolist()],
            "base_ang_vel_absmean_xyz": [round(float(v), 4) for v in rd.root_link_ang_vel_w.abs().mean(dim=0).tolist()],
            "viol_hard_L1_mean": round(float(viol_h.sum(dim=-1).mean()), 6),
            "viol_soft_L1_mean": round(float(viol_s.sum(dim=-1).mean()), 6),
            "top_soft_violators": _top_violators(viol_s, env),
            "pair_contact_frac": round(
                float(env.contact_manager.is_contact("feet_pair_contact").any(dim=1).float().mean()), 4
            ),
            # Exact joystick-task joint_deviation_knee formula:
            # sum |q_knee - q0_knee| (ungated). Pure joint_pos readout —
            # verifies the term the user sees diverging late in training.
            "knee_dev_L1_mean": round(
                float(
                    (q_now[:, env_holder["knee_ids"]] - env.act_manager.offset[:, env_holder["knee_ids"]])
                    .abs()
                    .sum(dim=1)
                    .mean()
                ),
                6,
            ),
            "knee_q_mean": [round(float(v), 5) for v in q_now[:, env_holder["knee_ids"]].mean(dim=0).tolist()],
            **(_gait_capture(rd) if task == "joystick" else {}),
            "obs_terms": _obs_capture(),
            "interfoot_dist_q10_50_90": _q(
                (
                    rd.body_pos_w_by_ids(env_holder["feet_ids"])[:, 0]
                    - rd.body_pos_w_by_ids(env_holder["feet_ids"])[:, 1]
                ).norm(dim=-1)
            ),
        }

    def _top_violators(viol: torch.Tensor, env) -> list:
        per_joint = viol.mean(dim=0)
        names = [n.rsplit("/", 1)[-1] for n in env.act_manager.actuated_joint_names]
        top = torch.topk(per_joint, k=min(5, per_joint.numel()))
        return [(names[int(i)], round(float(v), 5)) for v, i in zip(top.values.tolist(), top.indices.tolist())]

    post_reset = state_capture("post_reset")

    zero = torch.zeros((num_envs, env.num_actions), device=env.device)
    env.step(zero)
    torch.cuda.synchronize()
    first_step = state_capture("after_step1")
    _stage("first step done (first reward call captured)")

    for _k in range(settle_steps - 1):
        env.step(zero)
    settle_end = state_capture(f"after_step{settle_steps}")
    _stage("settle done")

    # ---- RANDOM-action window (early-training proxy) ------------------
    # An untrained policy emits ~N(0,1)-scale actions. CPU generator with
    # a fixed seed → bit-identical action sequences across sims. 300
    # steps so robots actually FALL and reset like the first training
    # iterations — the regime where the wandb curves are claimed to
    # diverge. Per-50-step blocks separate "just after synchronized
    # reset" from "mostly-fallen chaotic distribution".
    rand_start = {t: len(rs) for t, rs in records.items()}
    gen = torch.Generator().manual_seed(20260719)
    rand_steps = 300
    block = 50
    upright_frac_blocks: list = []
    pair_frac_blocks: list = []
    reset_count_blocks: list = []
    # Kinematic attribution: if genesis's collision/swing/clearance run
    # low because its limbs MOVE less under identical targets, these
    # show it; if motion matches but pair contact still differs, the
    # box-box detection in crossed configurations is the suspect.
    kin_blocks: dict[str, list] = {
        k: []
        for k in (
            "foot_xyspeed",
            "foot_z",
            "leg_qvel_abs",
            "ground_contact_frac",
            "interfoot_q10",
            "knee_dev_L1",
        )
    }
    up_acc = pair_acc = rst_acc = 0.0
    kin_acc = {k: 0.0 for k in kin_blocks}
    leg_ids = [
        i
        for i, n in enumerate(n.rsplit("/", 1)[-1] for n in env.act_manager.actuated_joint_names)
        if ("Hip" in n or "Knee" in n or "Ankle" in n)
    ]
    for _k in range(rand_steps):
        # device="cpu" is load-bearing: genesis installs a global default
        # device of cuda, which would reject the CPU generator.
        a = torch.randn((num_envs, env.num_actions), generator=gen, device="cpu").to(env.device)
        _o, _r, term_b, trunc_b, _e = env.step(a)
        rd_now = env.get_robot_data()
        up_acc += float((rd_now.root_link_pos_w[:, 2] > 0.4).float().mean())
        pair_acc += float(env.contact_manager.is_contact("feet_pair_contact").any(dim=1).float().mean())
        rst_acc += float((term_b | trunc_b).sum())
        fpos = rd_now.body_pos_w_by_ids(env_holder["feet_ids"])
        kin_acc["foot_xyspeed"] += float(
            rd_now.body_lin_vel_w_by_ids(env_holder["feet_ids"])[..., :2].norm(dim=-1).mean()
        )
        kin_acc["foot_z"] += float(fpos[..., 2].mean())
        kin_acc["leg_qvel_abs"] += float(rd_now.joint_vel[:, leg_ids].abs().mean())
        kin_acc["ground_contact_frac"] += float(env.contact_manager.is_contact("feet_ground_contact").float().mean())
        kin_acc["interfoot_q10"] += float(torch.quantile((fpos[:, 0] - fpos[:, 1]).norm(dim=-1), 0.1))
        kin_acc["knee_dev_L1"] += float(
            (rd_now.joint_pos[:, env_holder["knee_ids"]] - env.act_manager.offset[:, env_holder["knee_ids"]])
            .abs()
            .sum(dim=1)
            .mean()
        )
        if (_k + 1) % block == 0:
            upright_frac_blocks.append(round(up_acc / block, 4))
            pair_frac_blocks.append(round(pair_acc / block, 5))
            reset_count_blocks.append(int(rst_acc))
            for k in kin_blocks:
                kin_blocks[k].append(round(kin_acc[k] / block, 5))
                kin_acc[k] = 0.0
            up_acc = pair_acc = rst_acc = 0.0
    after_random = state_capture(f"after_random{rand_steps}")

    # ---- LIMIT-PUSH: joint behavior AT and BEYOND the range ----------
    # Saturating actions (clip is ±100, per-joint scale ~0.1-0.6 → PD
    # targets several rad beyond the hard limits) slam every joint into
    # its stop; a bang-bang phase whips through the range at max speed.
    # Per phase we record, for knees and for ALL actuated joints, where
    # q actually settles/overshoots per sim — this is where a hard-clamp
    # vs constraint-based limit implementation would diverge.
    def _limit_phase(name: str, action_fn, steps: int) -> dict:
        knee_min = None
        knee_max = None
        dev_acc = 0.0
        beyond_soft = 0.0
        beyond_hard = 0.0
        overshoot_max = 0.0
        for t in range(steps):
            env.step(action_fn(t))
            rd_lp = env.get_robot_data()
            qk = rd_lp.joint_pos[:, env_holder["knee_ids"]]
            knee_min = qk.min() if knee_min is None else torch.minimum(knee_min, qk.min())
            knee_max = qk.max() if knee_max is None else torch.maximum(knee_max, qk.max())
            dev_acc += float((qk - env.act_manager.offset[:, env_holder["knee_ids"]]).abs().sum(dim=1).mean())
            q_all = rd_lp.joint_pos
            lo_h, hi_h = _hard_lims(rd_lp)
            lo_s, hi_s = _lims(rd_lp.soft_joint_pos_limits)
            beyond_soft += float(((q_all < lo_s) | (q_all > hi_s)).float().mean())
            beyond_hard += float(((q_all < lo_h) | (q_all > hi_h)).float().mean())
            over = torch.maximum(lo_h - q_all, q_all - hi_h).clamp(min=0.0)
            overshoot_max = max(overshoot_max, float(over.max()))
        return {
            "knee_q_min": round(float(knee_min), 5),
            "knee_q_max": round(float(knee_max), 5),
            "knee_dev_mean": round(dev_acc / steps, 6),
            "beyond_soft_frac": round(beyond_soft / steps, 5),
            "beyond_hard_frac": round(beyond_hard / steps, 5),
            "max_overshoot_beyond_hard_rad": round(overshoot_max, 6),
        }

    big = 10.0
    limit_push = {
        "push_pos": _limit_phase(
            "push_pos", lambda t: torch.full((num_envs, env.num_actions), big, device=env.device), 20
        ),
        "push_neg": _limit_phase(
            "push_neg", lambda t: torch.full((num_envs, env.num_actions), -big, device=env.device), 20
        ),
        "bang_bang": _limit_phase(
            "bang_bang",
            lambda t: torch.full((num_envs, env.num_actions), big if (t // 2) % 2 == 0 else -big, device=env.device),
            24,
        ),
    }
    _stage("limit-push done")

    rand_blocks_meta = {
        "block": block,
        "upright_frac": upright_frac_blocks,
        "pair_contact_frac": pair_frac_blocks,
        "resets": reset_count_blocks,
        **{f"kin_{k}": v for k, v in kin_blocks.items()},
    }
    _stage("random-action window done")

    # ---- FOOT-PAIR proximity sweep ------------------------------------
    # Behavioral threshold for the foot-foot (box-box) contact used by
    # feet_pair_collision: hold the robot in the air, sweep symmetric
    # hip-roll offsets so lateral foot separation spans a grid, one zero
    # step, then record (measured inter-foot distance, pair found).
    writer = env.get_robot_state_writer()
    all_env_ids = torch.arange(num_envs, device=env.device)
    default_qpos = env.act_manager.offset.clone()
    leaf_names = [n.rsplit("/", 1)[-1] for n in env.act_manager.actuated_joint_names]
    l_roll, r_roll = leaf_names.index("Left_Hip_Roll"), leaf_names.index("Right_Hip_Roll")
    d_grid = torch.linspace(-0.35, 0.35, num_envs, device=env.device)
    qpos = default_qpos.clone()
    qpos[:, l_roll] += d_grid
    qpos[:, r_roll] -= d_grid
    writer.set_dof_positions(qpos, env_ids=all_env_ids)
    writer.set_dof_velocities(torch.zeros_like(qpos), env_ids=all_env_ids)
    rd0 = env.get_robot_data()
    pos = rd0.root_link_pos_w.clone()
    pos[:, 2] = 0.8  # feet well off the ground: only foot-foot can touch
    quat = torch.zeros((num_envs, 4), device=env.device)
    quat[:, 0] = 1.0
    writer.set_root_pose(pos, quat, env_ids=all_env_ids)
    writer.set_root_velocity(
        torch.zeros((num_envs, 3), device=env.device),
        torch.zeros((num_envs, 3), device=env.device),
        env_ids=all_env_ids,
    )
    writer.eval_fk(env_ids=all_env_ids)
    env.step(zero)
    rd1 = env.get_robot_data()
    fp = rd1.body_pos_w_by_ids(env_holder["feet_ids"])
    d_ff = (fp[:, 0] - fp[:, 1]).norm(dim=-1)
    pair_found = env.contact_manager.is_contact("feet_pair_contact").any(dim=1)
    bins = []
    edges_m = torch.arange(0.0, 0.205, 0.01, device=env.device)
    for i in range(len(edges_m) - 1):
        msk = (d_ff >= edges_m[i]) & (d_ff < edges_m[i + 1])
        n = int(msk.sum())
        if n == 0:
            continue
        bins.append(
            {
                "dist_cm": [round(float(edges_m[i]) * 100, 1), round(float(edges_m[i + 1]) * 100, 1)],
                "n": n,
                "pair_found_frac": round(float(pair_found[msk].float().mean()), 3),
            }
        )
    on_d = d_ff[pair_found]
    off_d = d_ff[~pair_found]
    pair_sweep = {
        "bins": bins,
        "max_dist_WITH_pair_contact_cm": round(float(on_d.max()) * 100, 3) if on_d.numel() else None,
        "min_dist_WITHOUT_cm": round(float(off_d.min()) * 100, 3) if off_d.numel() else None,
        "n_pair_contact": int(pair_found.sum()),
    }
    _stage("foot-pair sweep done")

    # Per-term series: first call + last-10-of-settle mean + random-window
    # means (overall and per 50-step block).
    term_summary = {}
    for term, recs in records.items():
        outs = [r["out_mean"] for r in recs]
        n0 = rand_start.get(term, len(outs))
        settle_outs = outs[:n0]
        rand_outs = outs[n0 : n0 + rand_steps]
        blocks = [
            round(sum(rand_outs[i : i + block]) / max(len(rand_outs[i : i + block]), 1), 6)
            for i in range(0, len(rand_outs), block)
        ]
        term_summary[term] = {
            "n_calls": len(recs),
            "first_out": outs[0] if outs else None,
            "last10_out_mean": (sum(settle_outs[-10:]) / max(len(settle_outs[-10:]), 1) if settle_outs else None),
            "rand24_out_mean": sum(rand_outs[:24]) / max(len(rand_outs[:24]), 1) if rand_outs else None,
            "rand_all_out_mean": sum(rand_outs) / max(len(rand_outs), 1) if rand_outs else None,
            "rand_block_means": blocks,
            "first_rec": recs[0] if recs else None,
        }

    return {
        "sim": sim,
        "wiring": wiring,
        "term_meta": term_meta,
        "term_summary": term_summary,
        "post_reset": post_reset,
        "first_step": first_step,
        "settle_end": settle_end,
        "after_random": after_random,
        "rand_blocks_meta": rand_blocks_meta,
        "limit_push": limit_push,
        "pair_sweep": pair_sweep,
    }


def cfgs_robot_foot_names(cfgs) -> tuple:
    # The preset stores robot cfg on the preset object; after build the
    # foot names also live in the contact cfg — read from the robot cfg
    # module to stay preset-agnostic.
    from rlworld.rl.configs.robots.k1 import K1Config

    return tuple(K1Config().foot_names)


# ── Parent ──────────────────────────────────────────────────────────


def run_parent(args) -> int:
    out_path = Path(args.out).resolve()
    log_dir = out_path.parent / (out_path.stem + "_logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Default sweep is the three shipping sims; --sims lets an A/B include
    # genesis integrator variants (genesis_implicitfast / genesis_ifast_ellip
    # / genesis_noprune) in the SAME comparison table.
    sim_list = [s.strip() for s in args.sims.split(",")] if args.sims else list(_SIMS)

    results: dict[str, dict] = {}
    for sim in sim_list:
        log_path = log_dir / f"{sim}.log"
        result_path = log_dir / f"{sim}.json"
        if result_path.exists():
            result_path.unlink()
        print(f"[diag] running {sim} ...", flush=True)
        t0 = time.time()
        with open(log_path, "w") as lf:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    _MODULE,
                    "--cell",
                    sim,
                    "--result-json",
                    str(result_path),
                    "--num-envs",
                    str(args.num_envs),
                    "--settle-steps",
                    str(args.settle_steps),
                    "--seed",
                    str(args.seed),
                    "--task",
                    args.task,
                ],
                stdout=lf,
                stderr=subprocess.STDOUT,
                env={**os.environ},
            )
        ok = result_path.exists()
        print(f"[diag]   -> {'ok' if ok else 'CRASH (see log)'} ({time.time() - t0:.0f}s)", flush=True)
        if ok:
            results[sim] = json.loads(result_path.read_text())

    sims = [s for s in sim_list if s in results]
    L: list[str] = []
    L.append("=" * 118)
    L.append("K1 G1-recipe reward parity — first-step + settle dump")
    L.append("=" * 118)
    L.append(f"num_envs: {args.num_envs}  settle: {args.settle_steps}  seed: {args.seed}  logs: {log_dir}")
    L.append("")

    for s in sims:
        w = results[s]["wiring"]
        L.append(f"── [{s}] wiring ──")
        L.append(f"  solver: {w['solver']}")
        L.append(f"  decimation: {w['decimation']}  physics_dt: {w['physics_dt']}")
        L.append(f"  feet resolved (ORDER): {w['feet_resolved']}")
        for name, t in sorted(w["reward_terms"].items()):
            L.append(f"  term {name}: {t['func']}  w={t['weight']}  params={t['params']}")
        L.append("")

    def row(label, getter, fmt="{:+.6f}"):
        cells = []
        for s in sims:
            try:
                v = getter(results[s])
                cells.append(fmt.format(v) if isinstance(v, int | float) and v is not None else str(v))
            except Exception as e:  # noqa: BLE001 — report the hole, don't hide it
                cells.append(f"ERR:{type(e).__name__}")
        L.append(f"  {label:<44s}" + "".join(f"{c:<26s}" for c in cells))

    L.append("── MASS / ARMATURE / DOF-PARAM AUDIT (link fusion + parameter injection parity) ──")
    for s in sims:
        a = results[s]["wiring"]["mass_audit"]
        L.append(f"  [{s}] n_links={a['n_links']}  total_mass={a['total_mass']}")
    all_leaves = sorted({leaf for s in sims for leaf in results[s]["wiring"]["mass_audit"]["link_masses"]})
    L.append("  link masses (only rows that differ or are missing somewhere):")
    for leaf in all_leaves:
        vals = [results[s]["wiring"]["mass_audit"]["link_masses"].get(leaf) for s in sims]
        present = [v for v in vals if v is not None]
        if len(present) == len(sims) and max(present) - min(present) < 1e-6:
            continue
        L.append(f"    {leaf:<22s}" + "".join(f"{str(v):<16s}" for v in vals))
    L.append("  per-dof params (canonical actuated order; genesis=min/mean/max over envs b/c per-env DR):")
    for key in ("armature", "damping", "frictionloss"):
        for s in sims:
            L.append(f"    [{s}] {key}: {results[s]['wiring']['mass_audit'][key]}")
        L.append("")
    L.append(f"  actuated order: {results[sims[0]]['wiring']['mass_audit']['actuated_leaf_order']}")
    L.append("")

    L.append("── FIRST reward call (raw term outputs, weight NOT applied) ──")
    L.append(f"  {'quantity':<44s}" + "".join(f"{s:<26s}" for s in sims))
    for term in _TASK_TERMS[args.task]:
        row(f"first {term}", lambda r, t=term: r["term_summary"][t]["first_out"])
        row(f"settle-mean {term}", lambda r, t=term: r["term_summary"][t]["last10_out_mean"])
        row(f"rand24-mean {term}", lambda r, t=term: r["term_summary"][t]["rand24_out_mean"])
    L.append("")

    L.append("── EARLY-TRAINING PROXY: 300 random-action steps, per-50-step-block means ──")
    L.append("  Block 1 = fresh synchronized resets; later blocks = fallen/chaotic")
    L.append("  distribution like the first training iterations. If a term aligns in")
    L.append("  block 1 but diverges later, the divergence is state-distribution drift")
    L.append("  (chaos), not the measurement.")
    for s in sims:
        m = results[s]["rand_blocks_meta"]
        L.append(f"  [{s}] upright_frac/block: {m['upright_frac']}  resets/block: {m['resets']}")
        L.append(f"  [{s}] pair_contact_frac/block: {m['pair_contact_frac']}")
    L.append("")
    L.append("  -- kinematics per block (motion-amplitude attribution) --")
    for key in (
        "kin_foot_xyspeed",
        "kin_leg_qvel_abs",
        "kin_foot_z",
        "kin_ground_contact_frac",
        "kin_interfoot_q10",
        "kin_knee_dev_L1",
    ):
        for s in sims:
            L.append(f"  [{s}] {key[4:]}: {results[s]['rand_blocks_meta'][key]}")
        L.append("")
    for term in [t for t in _TASK_TERMS[args.task] if t != "joint_pos_limits"]:
        for s in sims:
            bl = results[s]["term_summary"][term]["rand_block_means"]
            L.append(f"  [{s}] {term} blocks: {bl}")
        L.append("")

    if args.task == "joystick":
        L.append("── GAIT CLOCK / feet_phase inputs (single shared implementation; streams must align) ──")
        for tag in ("post_reset", "first_step", "settle_end", "after_random"):
            for s in sims:
                st = results[s][tag]
                L.append(
                    f"  [{s}][{tag}] feet_phase_direct={st['feet_phase_direct_mean']}  "
                    f"freq_mean={st['gait_freq_mean']}  freq_first8={st['gait_freq_first8']}"
                )
                L.append(f"  [{s}][{tag}]   phase_first4={st['gait_phase_first4']}  rz_first4={st['bezier_rz_first4']}")
            L.append("")

    L.append("── OBSERVATION PARITY (per-term mean/std/absmax; states are synchronized so rows must match) ──")
    for tag in ("post_reset", "first_step"):
        ref_terms = sorted(results[sims[0]][tag]["obs_terms"].keys())
        mismatches = 0
        for term in ref_terms:
            vals = [results[s][tag]["obs_terms"].get(term) for s in sims]
            if term.endswith("__nonfinite"):
                if any(v != 0 for v in vals):
                    L.append(f"  [{tag}] !!! NONFINITE {term}: " + " ".join(str(v) for v in vals))
                    mismatches += 1
                continue
            means = [v[0] for v in vals if v is not None]
            if len(means) < len(sims) or (max(means) - min(means)) > max(1e-3, 0.01 * max(abs(m) for m in means)):
                L.append(f"  [{tag}] MISMATCH {term}: " + "  ".join(f"{s}={v}" for s, v in zip(sims, vals)))
                mismatches += 1
        L.append(
            f"  [{tag}] {len(ref_terms)} obs entries checked, "
            f"{mismatches} mismatch/nonfinite rows (0 = observation pipeline aligned)"
        )
    L.append("  full per-term dump lives in the per-cell JSON files")
    L.append("")

    L.append("── FOOT-PAIR contact (feet_pair_collision inputs) ──")
    for tag in ("post_reset", "first_step", "settle_end", "after_random"):
        for s in sims:
            st = results[s][tag]
            L.append(
                f"  [{s}][{tag}] pair_contact_frac={st['pair_contact_frac']}  "
                f"interfoot_dist_q10_50_90={st['interfoot_dist_q10_50_90']}"
            )
        L.append("")
    L.append("── FOOT-PAIR proximity sweep (feet in air; pair-contact ON vs inter-foot dist) ──")
    for s in sims:
        sw = results[s]["pair_sweep"]
        L.append(
            f"  [{s}] pair contact {sw['n_pair_contact']}/{args.num_envs}  "
            f"max dist WITH pair contact = {sw['max_dist_WITH_pair_contact_cm']} cm  "
            f"min dist WITHOUT = {sw['min_dist_WITHOUT_cm']} cm"
        )
        for b in sw["bins"]:
            L.append(f"      dist {b['dist_cm']} cm  n={b['n']:<4d} pair_found={b['pair_found_frac']}")
        L.append("")

    if args.task == "g1recipe":
        L.append("── joint_pos_limits ATTRIBUTION (at first call; out should match ONE candidate) ──")
        for s in sims:
            fr = results[s]["term_summary"]["joint_pos_limits"]["first_rec"] or {}
            L.append(
                f"  [{s}] out={fr.get('out_mean'):.6f}  cand_HARD={fr.get('cand_hard_mean'):.6f}  "
                f"cand_SOFT={fr.get('cand_soft_mean'):.6f}"
            )
        L.append("  (mujoco-low expected iff mujoco matches cand_SOFT while genesis/newton match cand_HARD)")
        L.append("")

    L.append("── command / gate at t=0 (feet terms are gated by |cmd|>0.05) ──")
    for tag in ("post_reset", "first_step", "settle_end"):
        for s in sims:
            st = results[s][tag]
            L.append(
                f"  [{s}][{tag}] cmd_abs_mean={st['cmd_abs_mean_per_axis']}  gate_frac={st['cmd_gate_frac_005']}  "
                f"base_angvel_absmean={st['base_ang_vel_absmean_xyz']}"
            )
        L.append("")

    L.append("── LIMIT-PUSH: joints slammed to/through the range (a=±10, PD targets beyond limits) ──")
    L.append("  Where a hard-clamp vs constraint-based joint-limit implementation would diverge.")
    for phase in ("push_pos", "push_neg", "bang_bang"):
        for s in sims:
            lp = results[s]["limit_push"][phase]
            L.append(
                f"  [{s}][{phase}] knee_q=[{lp['knee_q_min']}, {lp['knee_q_max']}]  "
                f"knee_dev={lp['knee_dev_mean']}  beyond_soft={lp['beyond_soft_frac']}  "
                f"beyond_hard={lp['beyond_hard_frac']}  max_overshoot={lp['max_overshoot_beyond_hard_rad']} rad"
            )
        L.append("")

    L.append("── joint_deviation_knee (exact joystick formula: sum |q_knee − q0_knee|, ungated) ──")
    for tag in ("post_reset", "first_step", "settle_end", "after_random"):
        for s in sims:
            st = results[s][tag]
            L.append(f"  [{s}][{tag}] knee_dev_L1={st['knee_dev_L1_mean']}  knee_q_mean={st['knee_q_mean']}")
        L.append("")

    L.append("── canonical feet/limits state ──")
    for tag in ("post_reset", "first_step", "settle_end"):
        for s in sims:
            st = results[s][tag]
            L.append(
                f"  [{s}][{tag}] foot_z={st['foot_z_mean']} (q={st['foot_z_q10_50_90']})  "
                f"contact={st['contact_frac']}  force={st['force_mean']}  xyspeed={st['foot_xyspeed_mean']}"
            )
        for s in sims:
            st = results[s][tag]
            L.append(
                f"  [{s}][{tag}] violL1 hard={st['viol_hard_L1_mean']} soft={st['viol_soft_L1_mean']}  "
                f"top_soft={st['top_soft_violators']}"
            )
        L.append("")

    L.append("── first 8 raw commands per sim (RNG stream alignment check) ──")
    for s in sims:
        L.append(f"  [{s}] {results[s]['post_reset']['cmd_first8']}")
    L.append("")

    L.append("── term meta (exact function + captured kwargs at first call) ──")
    for s in sims:
        for term, m in sorted(results[s]["term_meta"].items()):
            L.append(f"  [{s}] {term}: kwargs={m.get('kwargs')}")
        L.append("")

    report = "\n".join(L)
    out_path.write_text(report + "\n")
    print()
    print(report)
    print(f"Report written to: {out_path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", default=None)
    ap.add_argument("--result-json", default=None)
    ap.add_argument("--num-envs", type=int, default=512)
    ap.add_argument("--settle-steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--task", default="g1recipe", choices=("g1recipe", "joystick"))
    ap.add_argument(
        "--sims",
        default=None,
        help="comma list of cells to sweep into one report "
        "(default: genesis,newton,mujoco). Add genesis integrator "
        "variants for an A/B, e.g. "
        "--sims genesis,genesis_implicitfast,newton,mujoco",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.out is None:
        args.out = f"k1_{args.task}_reward_parity_diag.txt"

    if args.cell is not None:
        result = run_cell(args.cell, args.num_envs, args.settle_steps, args.seed, args.task)
        Path(args.result_json).write_text(json.dumps(result, indent=2))
        return 0
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
