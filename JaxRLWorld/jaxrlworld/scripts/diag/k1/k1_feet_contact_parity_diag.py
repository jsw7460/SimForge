"""K1 feet_air_time / feet_phase — full 3-sim divergence dump.

Mid-training, Genesis's ``feet_air_time`` and ``feet_phase`` rewards drift
away from Newton/mjlab (which stay close to each other). Those two terms
consume exactly TWO sim-dependent quantities:

    1. per-foot ``is_contact`` of the ``feet_ground_contact`` group
       (K1FeetAirTime bookkeeping runs entirely on this bool), and
    2. the FOOT LINK world z (``body_pos_w_by_ids``) that
       ``feet_phase_bezier`` compares against the bezier swing profile,

plus the sim-agnostic gait-phase counter. This diag dumps everything that
produces those quantities, per simulator, into one combined report:

  * term wiring verbatim (thresholds, swing_height/sigma, command gates,
    feet selector resolution, contact-group tracked-name ORDER — a
    left/right order swap between the phase columns and the contact
    columns would silently invert the gait)
  * FOOT GEOMETRY from each backend's raw model: every collision geom of
    each foot link — type, size/params, local offset, friction — plus
    the world-frame bottom (Genesis: exact per-geom AABB; Newton/mjlab:
    from the compiled MjModel with an approximate bottom under the
    flat-stance assumption). Answers "is the foot geom planted at a
    different height on Genesis?" directly.
  * SETTLE phase (zero action, PD stand): per-step per-foot contact
    fraction, |force|, foot link z, manager air time — stance
    penetration and contact stability at rest.
  * SCRIPTED GAIT probe: an open-loop alternating leg sinusoid (BIT-
    IDENTICAL action sequence on all three sims, no policy involved)
    forces swing/stance cycles; per step we record, THROUGH THE REWARD
    SHIMS (the exact tensors the reward manager consumed): gait phase,
    per-foot foot_z, bezier target rz, feet_phase reward, K1FeetAirTime
    air-time buffer / first-contact / reward, manager air time, contact
    fraction and per-env contact TOGGLE counts (chatter — a flickering
    contact resets air time and is invisible in step-level bools).
  * RANDOM-action phase: same recording under identical random actions.
  * Cross-sim tables with |genesis - mjlab| vs |newton - mjlab| drift
    aggregates and FLAGS on every quantity whose spread exceeds
    tolerance.

Usage (GPU box):
    python -m jaxrlworld.scripts.diag.k1.k1_feet_contact_parity_diag
    python -m jaxrlworld.scripts.diag.k1.k1_feet_contact_parity_diag --num-envs 1024
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

_MODULE = "jaxrlworld.scripts.diag.k1.k1_feet_contact_parity_diag"
# Variant cells (config files untouched, in-diag overrides only):
#   genesis_stiff: contact-stiffness override — TESTED, refuted (no change).
#   mujoco_stiff:  impratio raised to 100 — tests whether mjlab's leg-splay
#                  (hip/ankle ROLL drift + lateral foot creep) is friction
#                  regularization creep that impratio suppresses.
_SIMS = ("genesis", "newton", "mujoco")
_SIM_KEY = {
    "genesis": "Genesis",
    "genesis_stiff": "Genesis",
    "newton": "Newton",
    "mujoco": "MujocoEnv",
    "mujoco_stiff": "MujocoEnv",
    "mujoco_solver": "MujocoEnv",
    "mujoco_elliptic": "MujocoEnv",
    "mujoco_eulerdamp": "MujocoEnv",
    "mujoco_solref": "MujocoEnv",
    "mujoco_ei10": "MujocoEnv",
    "mujoco_ei100": "MujocoEnv",
    "newton_ei10": "Newton",
    "mujoco_prio": "MujocoEnv",
    "mujoco_solimp": "MujocoEnv",
    "mujoco_implicitfast": "MujocoEnv",
    "genesis_euler": "Genesis",
    "genesis_fl0": "Genesis",
    "mujoco_fl0": "MujocoEnv",
    "mujoco_canonical": "MujocoEnv",
    "newton_canonical": "Newton",
}
_GAIT_HZ = 1.4  # scripted-probe stepping frequency (identical on all sims)


def _stage(msg: str) -> None:
    print(f"[STAGE] {msg}", flush=True)


# ── Child ───────────────────────────────────────────────────────────


def run_cell(sim: str, num_envs: int, settle_steps: int, probe_steps: int, random_steps: int, seed: int) -> dict:
    import torch

    torch.manual_seed(seed)
    _stage(f"cell start: {sim} (num_envs={num_envs})")

    # ---- reward shims (must be installed BEFORE the config is built) --
    from jaxrlworld.rl.envs.mdp.rewards import k1_locomotion as k1_rf

    phase_calls: list[dict] = []
    orig_phase_fn = k1_rf.feet_phase_bezier
    phase_meta: dict = {}
    feet_ids_holder: dict = {}

    def rec_phase(env, *a, **kw):
        from jaxrlworld.rl.envs.mdp.rewards.k1_locomotion import _bezier_rz

        out = orig_phase_fn(env, *a, **kw)
        asset_cfg = kw.get("asset_cfg")
        if not phase_meta:
            phase_meta["asset_cfg_repr"] = repr(asset_cfg)
            phase_meta["kwargs"] = {k: repr(v) for k, v in kw.items() if k != "asset_cfg"}
            feet_ids_holder["ids"] = asset_cfg.body_ids
        phase = env.command_manager.get_term(kw.get("command_term", "gait_phase")).command
        foot_z = env.get_robot_data(asset_cfg.name).body_pos_w_by_ids(asset_cfg.body_ids)[..., 2]
        rz = _bezier_rz(phase, kw.get("swing_height", 0.12))
        phase_calls.append(
            {
                "phase": phase.mean(dim=0).tolist(),
                "foot_z": foot_z.mean(dim=0).tolist(),
                "foot_z_max": foot_z.max(dim=0).values.tolist(),
                "rz": rz.mean(dim=0).tolist(),
                "reward_mean": float(out.mean()),
            }
        )
        return out

    k1_rf.feet_phase_bezier = rec_phase

    air_calls: list[dict] = []
    orig_air_call = k1_rf.K1FeetAirTime.__call__

    def rec_air_call(self, env):
        contact = env.contact_manager.is_contact(self._contact_group)
        pre_air = self.air_time.clone()
        first_contact = (pre_air > 0.0) & (contact | self._last_contact)
        out = orig_air_call(self, env)
        air_calls.append(
            {
                "contact_frac": contact.float().mean(dim=0).tolist(),
                "air_time_pre": pre_air.mean(dim=0).tolist(),
                "first_contact_frac": first_contact.float().mean(dim=0).tolist(),
                "mgr_air_time": env.contact_manager.current_air_time(self._contact_group).mean(dim=0).tolist(),
                "reward_mean": float(out.mean()),
            }
        )
        return out

    k1_rf.K1FeetAirTime.__call__ = rec_air_call

    from jaxrlworld.rl.configs.presets.k1_joystick.base import K1JoystickConfig

    stiff = sim == "genesis_stiff"
    mj_stiff = sim == "mujoco_stiff"
    mj_solver = sim == "mujoco_solver"
    mj_elliptic = sim == "mujoco_elliptic"
    mj_eulerdamp = sim == "mujoco_eulerdamp"
    mj_solref = sim == "mujoco_solref"
    mj_ei10 = sim == "mujoco_ei10"
    mj_ei100 = sim == "mujoco_ei100"
    nt_ei10 = sim == "newton_ei10"
    mj_prio = sim == "mujoco_prio"
    mj_solimp = sim == "mujoco_solimp"
    mj_implicit = sim == "mujoco_implicitfast"
    gen_euler = sim == "genesis_euler"
    gen_fl0 = sim == "genesis_fl0"
    mj_fl0 = sim == "mujoco_fl0"
    mj_canon = sim == "mujoco_canonical"
    nt_canon = sim == "newton_canonical"
    if stiff or gen_euler or gen_fl0:
        sim = "genesis"
    if nt_ei10 or nt_canon:
        sim = "newton"
    if (
        mj_stiff
        or mj_solver
        or mj_elliptic
        or mj_eulerdamp
        or mj_solref
        or mj_ei10
        or mj_ei100
        or mj_prio
        or mj_solimp
        or mj_implicit
        or mj_fl0
        or mj_canon
    ):
        sim = "mujoco"
    cfgs = K1JoystickConfig(sim_type=sim, num_envs=num_envs, seed=seed).build()
    # NOTE: the K1 mjlab preset supplies a FULL mjlab_sim_cfg, which takes
    # precedence over the wrapper-level scene fields — overrides must go
    # through mjlab_sim_cfg.mujoco, not cfgs.scene.impratio (an earlier
    # cell wrote the dead field and tested nothing).
    if mj_stiff:
        # Friction constraints enforced much harder relative to normal
        # ones: if the leg-splay stance is friction-regularization creep,
        # this must converge toward genesis (roll dev -> 0, slip -> 0).
        cfgs.scene.mjlab_sim_cfg.mujoco.impratio = 100.0
    if mj_solver:
        # Upstream mirrors iterations=3 / ls_iterations=5 — a very low
        # solver budget. Poor convergence leaves friction-constraint
        # violation = slip/jitter; raise the budget with everything else
        # (pyramidal, impratio=1) unchanged to isolate it.
        cfgs.scene.mjlab_sim_cfg.mujoco.iterations = 50
        cfgs.scene.mjlab_sim_cfg.mujoco.ls_iterations = 20
    if mj_elliptic:
        # Pyramidal friction cones are anisotropic and known to produce
        # tangential creep under regularization; elliptic is isotropic.
        cfgs.scene.mjlab_sim_cfg.mujoco.cone = "elliptic"
    if mj_eulerdamp:
        # Upstream disables eulerdamp (implicit joint-damping in the
        # Euler integrator). Without it, joint damping integrates
        # explicitly — a classic source of high-frequency jitter that
        # would keep stance feet buzzing and creeping.
        cfgs.scene.mjlab_sim_cfg.mujoco.disableflags = ("nativeccd",)
    if mj_solref:
        # Last remaining knob: contact impedance. MuJoCo friction is a
        # regularized soft constraint whose steady-state creep is set by
        # solref/solimp, not by impratio/cone/iterations (all refuted).
        # Tighten every robot geom's solref from the (0.02, 1) default
        # to (0.005, 1) by wrapping the entity spec_fn — file untouched.
        orig_spec_fn = cfgs.scene.entities["robot"].spec_fn

        class _StiffSpecFn:
            def __call__(self):
                spec = orig_spec_fn()
                for g in spec.geoms:
                    g.solref = [0.005, 1.0]
                return spec

        cfgs.scene.entities["robot"].spec_fn = _StiffSpecFn()
    if mj_ei10 or mj_ei100:
        # The MuJoCo-documented anti-slip recipe is the COMBINATION of an
        # elliptic cone and impratio > 1 (impratio makes friction
        # impedance harder relative to normal). Earlier cells tested each
        # alone: impratio=100 with the pyramidal cone (force blowup — a
        # combination the docs warn against) and elliptic with impratio=1
        # (friction impedance unchanged, so no effect was expected).
        # mjwarp does not implement the noslip solver (io.py raises
        # NotImplementedError), so this combo is the last in-solver knob.
        cfgs.scene.mjlab_sim_cfg.mujoco.cone = "elliptic"
        cfgs.scene.mjlab_sim_cfg.mujoco.impratio = 100.0 if mj_ei100 else 10.0
    if nt_ei10:
        # Same elliptic + impratio recipe on the Newton/mjwarp backend.
        cfgs.scene.solver_cfg.cone = "elliptic"
        cfgs.scene.solver_cfg.impratio = 10.0
    if mj_prio or mj_solimp:
        # Steady-state friction creep is proportional to the constraint
        # regularization R = (1-imp)/imp, and imp is CAPPED by solimp
        # dmax (0.95 default -> R >= 0.053). The earlier solref cell only
        # stiffened K/B (dynamics), never this cap — dmax is the one
        # regularization knob not yet tested. MuJoCo mixes the two geoms'
        # sol params (solmix mean), so raising the robot side alone gets
        # diluted by the terrain geom: priority=1 makes the robot geom's
        # params (incl. its own friction 0.6/1.0 — no max-combine with
        # the ground) apply exclusively. mujoco_prio isolates that
        # priority/friction side effect; mujoco_solimp adds dmax=0.999
        # (R <= 0.001, ~50x stiffer friction) on top.
        orig_robot_spec_fn = cfgs.scene.entities["robot"].spec_fn

        class _SolSpecFn:
            def __init__(self, orig, set_solimp):
                self._orig = orig
                self._set_solimp = set_solimp

            def __call__(self):
                spec = self._orig()
                for g in spec.geoms:
                    if g.contype or g.conaffinity:
                        g.priority = 1
                        if self._set_solimp:
                            g.solimp = [0.9, 0.999, 0.001, 0.5, 2.0]
                return spec

        cfgs.scene.entities["robot"].spec_fn = _SolSpecFn(orig_robot_spec_fn, mj_solimp)
    if mj_implicit:
        # INTEGRATOR axis. Every friction/regularization knob is refuted
        # (impratio, iterations, cone, eulerdamp, solref, elliptic+
        # impratio, solimp dmax) and the solimp cell proved the measured
        # slip is NOT constraint creep (50x stiffer regularization, slip
        # unchanged): the mjwarp cells never reach static rest (settle
        # force/mg 1.21 mjlab / 0.79 newton vs genesis 0.96) — feet
        # micro-bounce, drifting laterally while momentarily airborne.
        # K1 mirrors upstream with euler + eulerdamp DISABLED (explicit
        # joint damping — classically unstable on low-inertia ankle
        # dofs), while genesis runs its default approximate_implicitfast
        # integrator. Two-sided test: this cell moves mjlab to
        # implicitfast (Newton's own canonical humanoid recipe), and
        # genesis_euler moves genesis to Euler.
        cfgs.scene.mjlab_sim_cfg.mujoco.integrator = "implicitfast"
        cfgs.scene.mjlab_sim_cfg.mujoco.disableflags = ("nativeccd",)
    if gen_euler:
        # Reverse side of the integrator test: if genesis on Euler starts
        # bouncing/slipping like the mjwarp cells, the integrator is the
        # divergence axis; if it stays flat, the axis is elsewhere.
        import genesis as gs

        cfgs.scene.rigid_options.integrator = gs.integrator.Euler
    if mj_fl0:
        # JOINT DRY FRICTION axis. Settle torques show genesis hip-roll
        # PD torque (0.06-0.09 Nm) sits INSIDE the 0.1 Nm frictionloss
        # dead zone while mjwarp needs 0.7-0.9 Nm (= full gravity moment
        # of the crept/splayed stance; dev 0.028 rad = 0.7/kp25). The
        # hypothesis: genesis's stance is frozen by joint dry friction,
        # while on mjwarp frictionloss is ineffective in practice. If
        # this cell (frictionloss=0) is IDENTICAL to baseline mujoco,
        # frictionloss contributes nothing on mjwarp — one half of the
        # proof; genesis_fl0 is the other half.
        orig_fl_spec_fn = cfgs.scene.entities["robot"].spec_fn

        class _NoFrictionlossSpecFn:
            def __init__(self, orig):
                self._orig = orig

            def __call__(self):
                spec = self._orig()
                for j in spec.joints:
                    j.frictionloss = 0.0
                return spec

        cfgs.scene.entities["robot"].spec_fn = _NoFrictionlossSpecFn(orig_fl_spec_fn)
    if mj_canon:
        # THE CANONICAL COMBINATION. Every single-axis cell so far kept
        # the other axes at the upstream-mirror values (euler + 3/5
        # iterations + pyramidal + impratio 1) and none moved the
        # micro-bounce/slip. But every known-good mjwarp humanoid recipe
        # (Newton's example_robot_g1, our SolverMuJoCoCfg defaults)
        # combines implicitfast + 100/50 iterations + elliptic +
        # impratio 100 SIMULTANEOUSLY — the bounce can be sustained by
        # whichever failure mode is left unfixed, so the axes must be
        # flipped together.
        mjm = cfgs.scene.mjlab_sim_cfg.mujoco
        mjm.integrator = "implicitfast"
        mjm.iterations = 100
        mjm.ls_iterations = 50
        mjm.cone = "elliptic"
        mjm.impratio = 100.0
        mjm.disableflags = ("nativeccd",)
    if nt_canon:
        # Same canonical recipe on the Newton backend: drop the
        # upstream-mirror overrides and take the framework defaults
        # (newton solver, implicitfast, elliptic, 100/50, impratio 100).
        from jaxrlworld.rl.configs.newton_config_classes import SolverMuJoCoCfg as _SolverMuJoCoCfg

        cfgs.scene.solver_cfg = _SolverMuJoCoCfg(ccd_iterations=50)
    if stiff:
        # In-diag override only — preset files untouched. Aligns genesis
        # contact stiffness with the mjlab-side conventions: solver
        # pinned to Newton with the g1-family iteration budget, and
        # constraint_timeconst at the MuJoCo 2x-timestep guideline
        # (physics_dt 0.005, substeps 1 -> 0.01; the preset ships 0.02).
        import genesis as gs

        ro = cfgs.scene.rigid_options
        cfgs.scene.rigid_options = gs.options.RigidOptions(
            dt=ro.dt,
            constraint_solver=gs.constraint_solver.Newton,
            iterations=10,
            ls_iterations=20,
            tolerance=1e-5,
            constraint_timeconst=0.01,
            enable_collision=True,
            enable_self_collision=True,
            enable_joint_limit=True,
            batch_dofs_info=True,
            box_box_detection=True,
        )

    wiring = {
        "stiff_override": stiff,
        "mj_stiff_override": mj_stiff,
    }
    if sim == "genesis":
        ro = cfgs.scene.rigid_options
        wiring["genesis_rigid_options"] = (
            f"integrator={ro.integrator} constraint_solver={ro.constraint_solver} "
            f"iterations={ro.iterations} ls_iterations={ro.ls_iterations} tolerance={ro.tolerance} "
            f"constraint_timeconst={ro.constraint_timeconst}"
        )
    if sim == "newton":
        nt_cfg = cfgs.scene.solver_cfg
        wiring["newton_solver_cfg"] = (
            f"iterations={nt_cfg.iterations} ls_iterations={nt_cfg.ls_iterations} "
            f"impratio={nt_cfg.impratio} cone={nt_cfg.cone}"
        )
    if sim == "mujoco":
        mj_cfg = cfgs.scene.mjlab_sim_cfg.mujoco
        wiring["mjlab_sim_cfg.mujoco"] = (
            f"iterations={mj_cfg.iterations} ls_iterations={mj_cfg.ls_iterations} "
            f"impratio={mj_cfg.impratio} cone={mj_cfg.cone} integrator={mj_cfg.integrator} "
            f"disableflags={mj_cfg.disableflags}"
        )
    wiring |= {
        "feet_air_time": {
            "func": f"{cfgs.reward.feet_air_time.func.__module__}.{getattr(cfgs.reward.feet_air_time.func, '__name__', '?')}",
            "weight": cfgs.reward.feet_air_time.weight,
            "params": {k: repr(v) for k, v in (cfgs.reward.feet_air_time.params or {}).items()},
        },
        "feet_phase": {
            "weight": cfgs.reward.feet_phase.weight,
            "params": {k: repr(v) for k, v in (cfgs.reward.feet_phase.params or {}).items()},
        },
        "gait_phase_cmd": repr(cfgs.command.terms.get("gait_phase")),
        "contact_group_cfgs": [repr(c) for c in cfgs.scene.contact_sensors]
        if hasattr(cfgs.scene, "contact_sensors")
        else [repr(c) for c in cfgs.scene.sensors],
    }

    from jaxrlworld.rl.evals.sim_initializers import get_initializer

    env = get_initializer(_SIM_KEY[sim]).init_environment(cfgs)
    if gen_fl0:
        # Other half of the dry-friction proof: strip the 0.1 Nm joint
        # frictionloss from genesis AFTER build (post-reset, so the DR
        # startup event has already run). If the genesis stance then
        # splays/creeps like mjwarp, joint dry friction is what was
        # holding it symmetric and flat.
        ent_fl = env.scene_manager.robot
        fl_cur = ent_fl.get_dofs_frictionloss()
        ent_fl.set_dofs_frictionloss(torch.zeros_like(fl_cur))
    env.reset()
    _stage("env built + reset")

    group = env.contact_manager._groups["feet_ground_contact"]
    feet_names = list(group.tracked_names)
    wiring["contact_tracked_names"] = feet_names
    wiring["decimation"] = env.decimation
    wiring["physics_dt"] = env.physics_dt

    # ---- geometry dump ------------------------------------------------
    geometry: dict[str, list[dict]] = {}
    foot_link_z0: dict[str, float] = {}
    if sim == "genesis":
        entity = env.scene_manager.robot
        for fname in feet_names:
            link = entity.get_link(fname)
            foot_link_z0[fname] = float(entity.get_links_pos(links_idx_local=[link.idx_local])[0, 0, 2])
            geoms = []
            for g in link.geoms:
                aabb = g.get_AABB()  # (n_envs, 2, 3)
                geoms.append(
                    {
                        "type": str(g.type),
                        "friction": float(g.friction),
                        "contype": int(g.contype),
                        "conaffinity": int(g.conaffinity),
                        "init_pos_local": [float(v) for v in g.init_pos],
                        "world_aabb_z": [float(aabb[0, 0, 2]), float(aabb[0, 1, 2])],
                        "world_aabb_x": [float(aabb[0, 0, 0]), float(aabb[0, 1, 0])],
                        "world_aabb_y": [float(aabb[0, 0, 1]), float(aabb[0, 1, 1])],
                    }
                )
            geometry[fname] = geoms
        # Ground/terrain geoms on the genesis side, same rationale as the
        # mj-backend dump below: ground friction/sol params were never
        # verified cross-sim.
        gnd = []
        for entity in env.scene_manager.scene.entities:
            if entity is env.scene_manager.robot:
                continue
            for g in entity.geoms:
                gnd.append(
                    {
                        "type": str(g.type),
                        "friction": float(g.friction),
                        "contype": int(g.contype),
                        "conaffinity": int(g.conaffinity),
                        "sol_params": [float(v) for v in g.sol_params],
                    }
                )
        geometry["__ground__"] = gnd
    else:
        import mujoco

        mj_model = env.scene_manager.solver.mj_model if sim == "newton" else env.scene_manager.mj_model
        body_names = {
            i: (mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, i) or "") for i in range(mj_model.nbody)
        }
        for fname in feet_names:
            body_ids = [
                i
                for i, n in body_names.items()
                if n == fname or n.endswith("/" + fname) or n.rsplit("/", 1)[-1] == fname or fname in n
            ]
            if not body_ids:
                # Dump the full body-name table so the mismatch is diagnosable
                # from the report alone.
                geometry[fname] = [{"NO_BODY_MATCH": True, "all_body_names": sorted(body_names.values())}]
                continue
            geoms = []
            for gid in range(mj_model.ngeom):
                if int(mj_model.geom_bodyid[gid]) in body_ids:
                    gname = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, gid) or f"geom{gid}"
                    geoms.append(
                        {
                            "name": gname,
                            "type": int(mj_model.geom_type[gid]),
                            "size": [float(v) for v in mj_model.geom_size[gid]],
                            "pos_local": [float(v) for v in mj_model.geom_pos[gid]],
                            "friction": [float(v) for v in mj_model.geom_friction[gid]],
                            "margin": float(mj_model.geom_margin[gid]),
                            "condim": int(mj_model.geom_condim[gid]),
                            "contype": int(mj_model.geom_contype[gid]),
                            "conaffinity": int(mj_model.geom_conaffinity[gid]),
                            "solref": [float(v) for v in mj_model.geom_solref[gid]],
                            "solimp": [float(v) for v in mj_model.geom_solimp[gid]],
                            "priority": int(mj_model.geom_priority[gid]),
                            "solmix": float(mj_model.geom_solmix[gid]),
                        }
                    )
            geometry[fname] = geoms
        # Ground/terrain geoms: their friction and sol params co-determine
        # every foot contact (solmix combine unless priorities differ) —
        # never verified across sims until now.
        gnd = []
        for gid in range(mj_model.ngeom):
            bname = body_names[int(mj_model.geom_bodyid[gid])]
            gname = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, gid) or f"geom{gid}"
            is_plane = int(mj_model.geom_type[gid]) == 0
            if is_plane or "terrain" in bname.lower() or "ground" in gname.lower():
                gnd.append(
                    {
                        "name": gname,
                        "body": bname,
                        "type": int(mj_model.geom_type[gid]),
                        "friction": [float(v) for v in mj_model.geom_friction[gid]],
                        "condim": int(mj_model.geom_condim[gid]),
                        "contype": int(mj_model.geom_contype[gid]),
                        "conaffinity": int(mj_model.geom_conaffinity[gid]),
                        "solref": [float(v) for v in mj_model.geom_solref[gid]],
                        "solimp": [float(v) for v in mj_model.geom_solimp[gid]],
                        "priority": int(mj_model.geom_priority[gid]),
                        "solmix": float(mj_model.geom_solmix[gid]),
                    }
                )
        geometry["__ground__"] = gnd
    # Foot link world z for newton/mujoco comes from the phase shim at
    # runtime (the exact reward accessor), recorded every step below.

    # ---- helpers ------------------------------------------------------
    def contact_state():
        return env.contact_manager.is_contact("feet_ground_contact")

    def foot_force():
        f = env.contact_manager.contact_force("feet_ground_contact")
        return f.norm(dim=-1)

    joint_leafs = [n.rsplit("/", 1)[-1] for n in env.act_manager.actuated_joint_names]
    ankle_ids = [joint_leafs.index("Left_Ankle_Pitch"), joint_leafs.index("Right_Ankle_Pitch")]

    # ---- total robot mass (a mass mismatch would explain everything) --
    if sim == "genesis":
        total_mass = float(env.scene_manager.robot.get_mass())
    else:
        import mujoco as _mj

        _model = env.scene_manager.solver.mj_model if sim == "newton" else env.scene_manager.mj_model
        robot_body_ids = [
            i
            for i in range(_model.nbody)
            if "foot" in (_mj.mj_id2name(_model, _mj.mjtObj.mjOBJ_BODY, i) or "")
            or "Trunk" in (_mj.mj_id2name(_model, _mj.mjtObj.mjOBJ_BODY, i) or "")
        ]
        # Sum EVERY body mass except the world; terrain bodies are massless.
        total_mass = float(_model.body_mass[1:].sum())
        _ = robot_body_ids
    wiring["total_mass_kg"] = total_mass

    # Per-dof damping / armature / frictionloss: the last physical
    # parameter class never compared across sims. Each backend reads
    # these from the asset independently, and they gate both joint
    # stiction and the explicit-damping stability of the integrator.
    if sim == "genesis":
        ent = env.scene_manager.robot
        damping = ent.get_dofs_damping()
        armature = ent.get_dofs_armature()
        frictionloss = ent.get_dofs_frictionloss()
        if frictionloss.ndim == 2:
            wiring["live_dof_frictionloss_env01"] = [
                [round(float(v), 4) for v in frictionloss[0].tolist()],
                [round(float(v), 4) for v in frictionloss[1].tolist()],
            ]
        else:
            wiring["live_dof_frictionloss_env01"] = [[round(float(v), 4) for v in frictionloss.tolist()]]
        if damping.ndim == 2:
            damping, armature, frictionloss = damping[0], armature[0], frictionloss[0]
        per_joint = {}
        for joint in ent.joints:
            if joint.n_dofs != 1:
                continue
            i = int(joint.dofs_idx_local[0])
            per_joint[joint.name.rsplit("/", 1)[-1]] = [
                round(float(damping[i]), 6),
                round(float(armature[i]), 6),
                round(float(frictionloss[i]), 6),
            ]
        wiring["dof_params"] = per_joint
    else:
        import mujoco as _mj2

        _model2 = env.scene_manager.solver.mj_model if sim == "newton" else env.scene_manager.mj_model
        per_joint = {}
        for j in range(_model2.njnt):
            if int(_model2.jnt_type[j]) == 0:  # free joint
                continue
            jname = _mj2.mj_id2name(_model2, _mj2.mjtObj.mjOBJ_JOINT, j) or f"jnt{j}"
            adr = int(_model2.jnt_dofadr[j])
            per_joint[jname.rsplit("/", 1)[-1]] = [
                round(float(_model2.dof_damping[adr]), 6),
                round(float(_model2.dof_armature[adr]), 6),
                round(float(_model2.dof_frictionloss[adr]), 6),
            ]
        wiring["dof_params"] = per_joint
        # The host mj_model above shows PRE-DR nominals; dump the LIVE
        # warp-side field the solver actually reads (envs 0 and 1) — this
        # is where an ineffective/never-applied frictionloss would show.
        if sim == "newton":
            import warp as _wp2

            live_fl = env.scene_manager.solver.mjw_model.dof_frictionloss
            live_fl = live_fl if torch.is_tensor(live_fl) else _wp2.to_torch(live_fl)
        else:
            live_fl = env.scene_manager._sim.model.dof_frictionloss[:]
        if live_fl.ndim == 1:
            live_rows = [[round(float(v), 4) for v in live_fl.tolist()]]
        else:
            live_rows = [
                [round(float(v), 4) for v in live_fl[0].tolist()],
                [round(float(v), 4) for v in live_fl[1].tolist()],
            ]
        wiring["live_dof_frictionloss_env01"] = live_rows

    # Genesis-only: the simulator-INTERNAL PD gains. Our K1 actuator is
    # explicit (IdealPD via control_dofs_force, force mode), so these must
    # be irrelevant — unless the MJCF import left nonzero gains AND the
    # force-mode override assumption fails somewhere, which would act as a
    # hidden second PD holding the robot stiff.
    if sim == "genesis":
        kp = env.scene_manager.robot.get_dofs_kp()
        kv = env.scene_manager.robot.get_dofs_kv()
        wiring["genesis_internal_kp"] = kp[0].tolist() if kp.ndim == 2 else kp.tolist()
        wiring["genesis_internal_kv"] = kv[0].tolist() if kv.ndim == 2 else kv.tolist()

    def base_state():
        rd_now = env.get_robot_data()
        q = rd_now.root_link_quat_w  # wxyz
        w, x, y, z = q.unbind(dim=1)
        pitch = torch.asin(torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0))
        return float(rd_now.root_link_pos_w[:, 2].mean()), float(pitch.mean())

    # Genesis-only: classify feet-ground contacts by which foot geom
    # (box vs corner sphere) is touching, straight from the collider list.
    def genesis_geom_attribution():
        if sim != "genesis":
            return None
        from genesis.utils.misc import qd_to_torch

        solver = env.scene_manager.scene.sim.rigid_solver
        cd = solver.collider.get_contacts(as_tensor=True, to_torch=True)
        n_live = qd_to_torch(solver.collider._collider_state.n_contacts, copy=False)
        row_valid = torch.arange(cd["link_a"].shape[1], device=cd["link_a"].device)[None, :] < n_live[:, None]
        entity = env.scene_manager.robot
        out = {}
        for fname in feet_names:
            link = entity.get_link(fname)
            box_ids = torch.tensor([g.idx for g in link.geoms if "1" not in str(g.type)], device=cd["geom_a"].device)
            sphere_ids = torch.tensor([g.idx for g in link.geoms if "1" in str(g.type)], device=cd["geom_a"].device)

            def count(ids):
                hit = (cd["geom_a"].unsqueeze(-1) == ids).any(-1) | (cd["geom_b"].unsqueeze(-1) == ids).any(-1)
                return float((hit & row_valid).sum(dim=1).float().mean())

            out[fname] = {"box_contacts_per_env": count(box_ids), "sphere_contacts_per_env": count(sphere_ids)}
        return out

    def record_phase_steps(n_steps: int, action_fn, tag: str) -> dict:
        toggles = torch.zeros(num_envs, len(feet_names), device=env.device)
        # Stance slip: xy displacement of the foot frame between control
        # steps while the foot stayed in contact — direct measurement of
        # friction creep (a perfectly anchored stance foot slips 0).
        slip_sum = torch.zeros(num_envs, len(feet_names), device=env.device)
        slip_cnt = torch.zeros(num_envs, len(feet_names), device=env.device)
        # Time-windowed slip: separates the reset/landing transient
        # (windows 0-9, 10-29) from the late steady state (30+). If the
        # asymmetric-stance creep is a landing phenomenon it lives in
        # window 0 and the late window matches genesis.
        win_edges = (10, 30)
        slip_sum_w = torch.zeros(3, num_envs, len(feet_names), device=env.device)
        slip_cnt_w = torch.zeros(3, num_envs, len(feet_names), device=env.device)
        prev_xy = None
        prev = contact_state().clone()
        rows = []
        base_calls_phase = len(phase_calls)
        base_calls_air = len(air_calls)
        for k in range(n_steps):
            env.step(action_fn(k))
            cur = contact_state()
            toggles += (cur ^ prev).float()
            pos3 = env.get_robot_data().body_pos_w_by_ids(feet_ids_holder["ids"]).clone()
            xy = pos3[..., :2]
            if prev_xy is not None:
                both = (cur & prev).float()
                d = (xy - prev_xy).norm(dim=-1)
                slip_sum += d * both
                slip_cnt += both
                w = 0 if k < win_edges[0] else (1 if k < win_edges[1] else 2)
                slip_sum_w[w] += d * both
                slip_cnt_w[w] += both
            prev_xy = xy
            prev = cur.clone()
            rd_now = env.get_robot_data()
            base_z, base_pitch = base_state()
            row = {
                "contact_frac": cur.float().mean(dim=0).tolist(),
                "force": foot_force().mean(dim=0).tolist(),
                "mgr_air": env.contact_manager.current_air_time("feet_ground_contact").mean(dim=0).tolist(),
                "ankle_pitch": rd_now.joint_pos[:, ankle_ids].mean(dim=0).tolist(),
                "ankle_torque": rd_now.applied_torque[:, ankle_ids].mean(dim=0).tolist(),
                "base_z": base_z,
                "base_pitch": base_pitch,
                "feet_lateral_dist": float((pos3[:, 0, 1] - pos3[:, 1, 1]).abs().mean()),
            }
            if k == n_steps - 1:
                # Stance-asymmetry population stats: a flat symmetric
                # stance has both foot origins at ~0.048 (sphere bottom
                # offset); a rolled-onto-edge foot rides 5-30 mm higher.
                # Means hide the bimodal (random-side) split — count it.
                zlr = pos3[..., 2]
                dz = (zlr[:, 0] - zlr[:, 1]).abs()
                row["stance_z_asym"] = {
                    "mean_abs_LR_diff_mm": float(dz.mean() * 1000.0),
                    "frac_envs_LR_diff_gt_5mm": float((dz > 0.005).float().mean()),
                    "frac_left_above_55mm": float((zlr[:, 0] > 0.055).float().mean()),
                    "frac_right_above_55mm": float((zlr[:, 1] > 0.055).float().mean()),
                }
                qs = torch.tensor([0.1, 0.5, 0.9], device=env.device)
                row["z_percentiles"] = {
                    "base_z_p10_p50_p90": [
                        round(float(v), 4) for v in torch.quantile(rd_now.root_link_pos_w[:, 2], qs).tolist()
                    ],
                    "foot_z_p10_p50_p90": [round(float(v), 4) for v in torch.quantile(zlr.flatten(), qs).tolist()],
                }
                row["geom_attribution"] = genesis_geom_attribution()
                offset = env.act_manager.offset
                row["joint_dev_all"] = (rd_now.joint_pos - offset).mean(dim=0).tolist()
                row["joint_torque_all"] = rd_now.applied_torque.mean(dim=0).tolist()
                row["joint_names_all"] = joint_leafs
                row["frame_truth"] = frame_truth()
            rows.append(row)
        return {
            "steps": rows,
            "toggles_per_step": (toggles / n_steps).mean(dim=0).tolist(),
            "slip_mm_per_stance_step": (slip_sum / slip_cnt.clamp(min=1.0) * 1000.0).mean(dim=0).tolist(),
            "slip_windows_mm": [
                (slip_sum_w[w] / slip_cnt_w[w].clamp(min=1.0) * 1000.0).mean(dim=0).tolist() for w in range(3)
            ],
            "phase_calls": phase_calls[base_calls_phase:],
            "air_calls": air_calls[base_calls_air:],
            "tag": tag,
        }

    def frame_truth():
        """Frame-free ground truth for env 0: the reward-path accessors
        vs the raw body frames (xpos), body COMs (xipos) and the foot
        sphere geom world centers, plus the ankle-cross bodies — decides
        whether cross-sim z offsets are physics or accessor/frame
        conventions."""
        rd_now = env.get_robot_data()
        out = {
            "accessor_root_z": float(rd_now.root_link_pos_w[0, 2]),
        }
        if sim == "genesis":
            entity = env.scene_manager.robot
            for nm in feet_names + ["Left_Ankle_Cross", "Right_Ankle_Cross", "Trunk"]:
                link = entity.get_link(nm)
                out[f"link_xpos_z[{nm}]"] = float(entity.get_links_pos(links_idx_local=[link.idx_local])[0, 0, 2])
            for fname in feet_names:
                link = entity.get_link(fname)
                sphere_centers = []
                for g in link.geoms:
                    if str(g.type) == "1":
                        aabb = g.get_AABB()
                        sphere_centers.append(float((aabb[0, 0, 2] + aabb[0, 1, 2]) / 2.0))
                out[f"sphere_centers_z[{fname}]"] = sphere_centers
        else:
            import mujoco as _mj
            import warp as wp

            def as_torch(x):
                import torch as _t

                return x if _t.is_tensor(x) else wp.to_torch(x)

            if sim == "newton":
                mj_model = env.scene_manager.solver.mj_model
                mjw_data = env.scene_manager.solver.mjw_data
                xpos = as_torch(mjw_data.xpos)
                xipos = as_torch(mjw_data.xipos)
                geom_xpos = as_torch(mjw_data.geom_xpos)
            else:
                mj_model = env.scene_manager.mj_model
                data = env.scene_manager._sim.data
                # mjlab exposes warp state through TorchArray wrappers:
                # indexing yields zero-copy torch views.
                xpos = data.xpos[:]
                xipos = data.xipos[:]
                geom_xpos = data.geom_xpos[:]
            out["shapes"] = {
                "xpos": list(xpos.shape),
                "geom_xpos": list(geom_xpos.shape),
            }
            if xpos.ndim == 3:
                xpos, xipos, geom_xpos = xpos[0], xipos[0], geom_xpos[0]
            body_names = {i: (_mj.mj_id2name(mj_model, _mj.mjtObj.mjOBJ_BODY, i) or "") for i in range(mj_model.nbody)}
            # Dump EVERY body: name matching against namespaced/suffixed
            # names proved unreliable, and the robot only has ~24 bodies.
            out["all_bodies_xpos_xipos_z"] = {
                body_names[i]: [float(xpos[i, 2]), float(xipos[i, 2])] for i in range(mj_model.nbody)
            }
            for fname in feet_names:
                centers = []
                for gid in range(mj_model.ngeom):
                    gname = _mj.mj_id2name(mj_model, _mj.mjtObj.mjOBJ_GEOM, gid) or ""
                    if int(mj_model.geom_type[gid]) == 2 and fname.split("_")[0].lower() in gname.lower():
                        bname = body_names[int(mj_model.geom_bodyid[gid])]
                        if fname in bname or bname.endswith(fname):
                            centers.append(float(geom_xpos[gid, 2]))
                out[f"sphere_centers_z[{fname}]"] = centers
        # The exact tensor the reward consumes, env 0.
        if phase_calls:
            out["accessor_foot_z_last_call"] = phase_calls[-1]["foot_z"]
        return out

    zero = torch.zeros((num_envs, env.num_actions), device=env.device)

    # ---- settle (zero action) ----------------------------------------
    settle = record_phase_steps(settle_steps, lambda k: zero, "settle")
    _stage("settle done")

    # ---- scripted gait probe (identical open-loop actions) -----------
    names = [n.rsplit("/", 1)[-1] for n in env.act_manager.actuated_joint_names]

    def idx(sub: str) -> int:
        return names.index(sub)

    legs = {
        "L": (idx("Left_Hip_Pitch"), idx("Left_Knee_Pitch"), idx("Left_Ankle_Pitch")),
        "R": (idx("Right_Hip_Pitch"), idx("Right_Knee_Pitch"), idx("Right_Ankle_Pitch")),
    }
    ctrl_dt = env.control_dt

    def gait_action(k: int):
        a = torch.zeros((num_envs, env.num_actions), device=env.device)
        s = math.sin(2.0 * math.pi * _GAIT_HZ * k * ctrl_dt)
        swings = {"L": max(s, 0.0), "R": max(-s, 0.0)}
        for leg, (hip, knee, ankle) in legs.items():
            w = swings[leg]
            a[:, hip] = -1.0 * w
            a[:, knee] = 1.6 * w
            a[:, ankle] = -0.5 * w
        return a

    probe = record_phase_steps(probe_steps, gait_action, "gait_probe")
    _stage("gait probe done")

    # ---- random actions ----------------------------------------------
    gen = torch.Generator(device="cpu").manual_seed(seed + 1)
    # device="cpu" is explicit: Genesis installs a cuda default-device hook,
    # which otherwise rejects the cpu generator.
    rand_actions = torch.rand((random_steps, env.num_actions), generator=gen, device="cpu").mul_(2.0).sub_(1.0)

    def random_action(k: int):
        return rand_actions[k].to(env.device).unsqueeze(0).expand(num_envs, -1).contiguous()

    rand = record_phase_steps(random_steps, random_action, "random")
    _stage("random done")

    k1_rf.feet_phase_bezier = orig_phase_fn
    k1_rf.K1FeetAirTime.__call__ = orig_air_call

    return {
        "sim": sim,
        "num_envs": num_envs,
        "wiring": wiring,
        "phase_meta": phase_meta,
        "feet_names": feet_names,
        "geometry": geometry,
        "foot_link_z0_genesis_only": foot_link_z0,
        "settle": settle,
        "probe": probe,
        "random": rand,
    }


# ── Parent ──────────────────────────────────────────────────────────


def _fmt(cells, widths):
    return "".join(str(c)[: w - 1].ljust(w) for c, w in zip(cells, widths))


def run_parent(args) -> int:
    out_path = Path(args.out).resolve()
    log_dir = out_path.parent / (out_path.stem + "_logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    for sim in _SIMS:
        log_path = log_dir / f"{sim}.log"
        result_path = log_dir / f"{sim}.json"
        if result_path.exists():
            result_path.unlink()
        print(f"[diag] running {sim} ...", flush=True)
        t0 = time.perf_counter()
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
                    "--probe-steps",
                    str(args.probe_steps),
                    "--random-steps",
                    str(args.random_steps),
                    "--seed",
                    str(args.seed),
                ],
                stdout=lf,
                stderr=subprocess.STDOUT,
                env=os.environ.copy(),
            )
        wall = time.perf_counter() - t0
        if result_path.exists():
            results[sim] = json.loads(result_path.read_text())
            print(f"[diag]   -> ok ({wall:.0f}s)", flush=True)
        else:
            print(f"[diag]   -> CRASH (see {log_path})", flush=True)

    L: list[str] = []
    L.append("=" * 120)
    L.append("K1 feet_air_time / feet_phase — 3-sim divergence dump")
    L.append("=" * 120)
    L.append(
        f"num_envs: {args.num_envs}  settle: {args.settle_steps}  gait probe: {args.probe_steps} "
        f"({_GAIT_HZ} Hz open loop)  random: {args.random_steps}  seed: {args.seed}"
    )
    L.append(f"cell logs/JSON: {log_dir}")
    L.append("")

    sims = [s for s in _SIMS if s in results]
    if not sims:
        out_path.write_text("\n".join(L) + "\nALL CELLS CRASHED\n")
        print("\n".join(L))
        return 1

    # ---- wiring -------------------------------------------------------
    for s in sims:
        w = results[s]["wiring"]
        L.append(f"── [{s}] wiring ──")
        L.append(f"  contact tracked names (ORDER): {w['contact_tracked_names']}")
        L.append(f"  feet_air_time: {w['feet_air_time']}")
        L.append(f"  feet_phase: {w['feet_phase']}")
        L.append(f"  phase term runtime meta: {results[s].get('phase_meta')}")
        L.append(f"  gait cmd: {w['gait_phase_cmd']}")
        L.append(f"  decimation: {w['decimation']}  physics_dt: {w['physics_dt']}")
        L.append(
            f"  TOTAL ROBOT MASS: {w.get('total_mass_kg'):.4f} kg (m*g = {w.get('total_mass_kg', 0) * 9.81:.1f} N)"
        )
        if "mjlab_sim_cfg.mujoco" in w:
            L.append(f"  mjlab_sim_cfg.mujoco: {w['mjlab_sim_cfg.mujoco']}")
        if "genesis_rigid_options" in w:
            L.append(f"  genesis_rigid_options: {w['genesis_rigid_options']}")
        if "newton_solver_cfg" in w:
            L.append(f"  newton_solver_cfg: {w['newton_solver_cfg']}")
        if "dof_params" in w:
            L.append("  dof params [damping, armature, frictionloss]:")
            for jn, vals in w["dof_params"].items():
                L.append(f"    {jn}: {vals}")
        if "live_dof_frictionloss_env01" in w:
            for i, r in enumerate(w["live_dof_frictionloss_env01"]):
                L.append(f"  LIVE dof_frictionloss env{i}: {r}")
        for c in w["contact_group_cfgs"]:
            L.append(f"  contact cfg: {c}")
        L.append("")

    order_ok = len({tuple(results[s]["wiring"]["contact_tracked_names"]) for s in sims}) == 1
    if not order_ok:
        L.append("  !! CONTACT TRACKED-NAME ORDER DIFFERS BETWEEN SIMS !!")
        L.append("")

    # ---- geometry -----------------------------------------------------
    L.append("── foot geometry (raw per-backend model) ──")
    for s in sims:
        L.append(f"  [{s}]")
        for fname, geoms in results[s]["geometry"].items():
            L.append(f"    {fname}: {len(geoms)} geoms")
            for g in geoms:
                L.append(f"      {g}")
        z0 = results[s].get("foot_link_z0_genesis_only") or {}
        if z0:
            L.append(f"    foot link world z at reset: {z0}")
    L.append("")

    # ---- settle summary ----------------------------------------------
    feet = results[sims[0]]["feet_names"]

    def phase_series(s, section, key, col):
        return [c[key][col] for c in results[s][section]["phase_calls"]]

    def steps_series(s, section, key, col):
        return [row[key][col] for row in results[s][section]["steps"]]

    L.append("── settle (zero action): last-10-step means ──")
    hdr = ["quantity"] + sims + ["gen-mj", "newt-mj"]
    wds = [34] + [14] * (len(sims) + 2)
    L.append(_fmt(hdr, wds))

    def tail_mean(series):
        t = series[-10:]
        return sum(t) / len(t)

    flags: list[str] = []

    def cross_row(label, per_sim_vals, tol):
        row = [label] + [f"{v:.5f}" for v in per_sim_vals]
        if "mujoco" in sims and len(per_sim_vals) == len(sims):
            mj = per_sim_vals[sims.index("mujoco")]
            gen = per_sim_vals[sims.index("genesis")] if "genesis" in sims else float("nan")
            newt = per_sim_vals[sims.index("newton")] if "newton" in sims else float("nan")
            dg, dn = gen - mj, newt - mj
            row += [f"{dg:+.5f}", f"{dn:+.5f}"]
            if abs(dg) > tol and abs(dg) > 3 * max(abs(dn), 1e-9):
                flags.append(f"{label}: genesis dev {dg:+.5f} vs newton dev {dn:+.5f}")
        L.append(_fmt(row, wds))

    for fi, fname in enumerate(feet):
        cross_row(
            f"settle foot_z[{fname}]",
            [tail_mean(phase_series(s, "settle", "foot_z", fi)) for s in sims],
            5e-4,
        )
        cross_row(
            f"settle contact_frac[{fname}]",
            [tail_mean(steps_series(s, "settle", "contact_frac", fi)) for s in sims],
            0.01,
        )
        cross_row(
            f"settle force[{fname}]",
            [tail_mean(steps_series(s, "settle", "force", fi)) for s in sims],
            5.0,
        )
    for side, si in (("L", 0), ("R", 1)):
        cross_row(
            f"settle ankle_pitch[{side}] (target -0.2)",
            [tail_mean(steps_series(s, "settle", "ankle_pitch", si)) for s in sims],
            0.01,
        )
        cross_row(
            f"settle ankle_torque[{side}]",
            [tail_mean(steps_series(s, "settle", "ankle_torque", si)) for s in sims],
            0.5,
        )
    cross_row(
        "settle toggles/step (mean feet)",
        [sum(results[s]["settle"]["toggles_per_step"]) / len(feet) for s in sims],
        0.01,
    )
    cross_row(
        "settle base_z",
        [tail_mean([row["base_z"] for row in results[s]["settle"]["steps"]]) for s in sims],
        0.002,
    )
    cross_row(
        "settle base_pitch",
        [tail_mean([row["base_pitch"] for row in results[s]["settle"]["steps"]]) for s in sims],
        0.01,
    )
    for fi, fname in enumerate(feet):
        cross_row(
            f"settle SLIP mm/stance-step[{fname}]",
            [results[s]["settle"]["slip_mm_per_stance_step"][fi] for s in sims],
            0.05,
        )
    cross_row(
        "settle feet lateral dist START",
        [results[s]["settle"]["steps"][0]["feet_lateral_dist"] for s in sims],
        0.003,
    )
    cross_row(
        "settle feet lateral dist END",
        [results[s]["settle"]["steps"][-1]["feet_lateral_dist"] for s in sims],
        0.003,
    )
    for kk in (5, 10, 20, 30):
        cross_row(
            f"settle feet lateral dist @step{kk}",
            [results[s]["settle"]["steps"][kk]["feet_lateral_dist"] for s in sims],
            0.003,
        )
    for w, label in enumerate(("steps 0-9", "steps 10-29", "steps 30-49")):
        cross_row(
            f"settle SLIP window {label} [L]",
            [results[s]["settle"]["slip_windows_mm"][w][0] for s in sims],
            0.5,
        )
        cross_row(
            f"settle SLIP window {label} [R]",
            [results[s]["settle"]["slip_windows_mm"][w][1] for s in sims],
            0.5,
        )
    cross_row(
        "settle total feet force vs m*g",
        [
            (tail_mean(steps_series(s, "settle", "force", 0)) + tail_mean(steps_series(s, "settle", "force", 1)))
            / (results[s]["wiring"]["total_mass_kg"] * 9.81)
            for s in sims
        ],
        0.05,
    )

    # Micro-bounce quantification: a robot at static rest has a nearly
    # constant contact force; persistent step-to-step force variance =
    # feet bouncing, which fakes "slip" via airborne lateral drift.
    def tail_std(xs):
        t = xs[-10:]
        m = sum(t) / len(t)
        return (sum((v - m) ** 2 for v in t) / len(t)) ** 0.5

    cross_row(
        "settle force STD last10 [L] (bounce)",
        [tail_std(steps_series(s, "settle", "force", 0)) for s in sims],
        2.0,
    )
    cross_row(
        "settle force STD last10 [R] (bounce)",
        [tail_std(steps_series(s, "settle", "force", 1)) for s in sims],
        2.0,
    )
    for s in sims:
        asym = results[s]["settle"]["steps"][-1].get("stance_z_asym")
        if asym:
            L.append(f"  [{s}] settle stance z-asymmetry (flat symmetric => diff~0, frac~0): {asym}")
    for s in sims:
        zp = results[s]["settle"]["steps"][-1].get("z_percentiles")
        if zp:
            L.append(f"  [{s}] settle z percentiles (mean-contamination check): {zp}")
    for s in sims:
        attribution = results[s]["settle"]["steps"][-1].get("geom_attribution")
        if attribution:
            L.append(f"  [{s}] settle contact geom attribution (contacts/env): {attribution}")
    L.append("")

    # Per-joint deviation-from-default + applied torque at settle end.
    L.append("── settle end: per-joint deviation from default (rad) | applied torque (Nm) ──")
    jnames = results[sims[0]]["settle"]["steps"][-1]["joint_names_all"]
    hdrj = ["joint"] + sims
    wdsj = [26] + [26] * len(sims)
    L.append(_fmt(hdrj, wdsj))
    for ji, jn in enumerate(jnames):
        row = [jn]
        for s in sims:
            last = results[s]["settle"]["steps"][-1]
            row.append(f"{last['joint_dev_all'][ji]:+.4f} | {last['joint_torque_all'][ji]:+.3f}")
        L.append(_fmt(row, wdsj))
    for s in sims:
        w = results[s]["wiring"]
        if "genesis_internal_kp" in w:
            L.append(f"  [{s}] INTERNAL sim PD gains (must be irrelevant under force mode):")
            L.append(f"    kp: {[f"{v:.1f}" for v in w["genesis_internal_kp"]]}")
            L.append(f"    kv: {[f"{v:.2f}" for v in w["genesis_internal_kv"]]}")
    L.append("")

    L.append("── settle end FRAME TRUTH (env 0): accessors vs raw frames vs geom world centers ──")
    for s in sims:
        ft = results[s]["settle"]["steps"][-1].get("frame_truth") or {}
        L.append(f"  [{s}]")
        for k2, v2 in ft.items():
            L.append(f"    {k2}: {v2}")
    L.append("")

    # ---- gait probe ---------------------------------------------------
    L.append(f"── scripted gait probe ({_GAIT_HZ} Hz, identical open-loop actions) ──")
    for fi, fname in enumerate(feet):
        cross_row(
            f"probe foot_z mean[{fname}]",
            [sum(phase_series(s, "probe", "foot_z", fi)) / len(phase_series(s, "probe", "foot_z", fi)) for s in sims],
            1e-3,
        )
        cross_row(
            f"probe foot_z_max mean[{fname}]",
            [
                sum(phase_series(s, "probe", "foot_z_max", fi)) / len(phase_series(s, "probe", "foot_z_max", fi))
                for s in sims
            ],
            2e-3,
        )
        cross_row(
            f"probe contact_frac[{fname}]",
            [
                sum(steps_series(s, "probe", "contact_frac", fi)) / len(steps_series(s, "probe", "contact_frac", fi))
                for s in sims
            ],
            0.02,
        )
        cross_row(
            f"probe mgr_air mean[{fname}]",
            [sum(steps_series(s, "probe", "mgr_air", fi)) / len(steps_series(s, "probe", "mgr_air", fi)) for s in sims],
            0.01,
        )
        cross_row(
            f"probe K1 air_time pre[{fname}]",
            [
                sum(c["air_time_pre"][fi] for c in results[s]["probe"]["air_calls"])
                / max(len(results[s]["probe"]["air_calls"]), 1)
                for s in sims
            ],
            0.01,
        )
    cross_row(
        "probe feet_phase reward mean",
        [
            sum(c["reward_mean"] for c in results[s]["probe"]["phase_calls"])
            / max(len(results[s]["probe"]["phase_calls"]), 1)
            for s in sims
        ],
        0.01,
    )
    cross_row(
        "probe feet_air reward mean",
        [
            sum(c["reward_mean"] for c in results[s]["probe"]["air_calls"])
            / max(len(results[s]["probe"]["air_calls"]), 1)
            for s in sims
        ],
        0.005,
    )
    for side, si in (("L", 0), ("R", 1)):
        cross_row(
            f"probe ankle_pitch mean[{side}]",
            [
                sum(steps_series(s, "probe", "ankle_pitch", si)) / len(steps_series(s, "probe", "ankle_pitch", si))
                for s in sims
            ],
            0.01,
        )
        cross_row(
            f"probe ankle_torque mean[{side}]",
            [
                sum(steps_series(s, "probe", "ankle_torque", si)) / len(steps_series(s, "probe", "ankle_torque", si))
                for s in sims
            ],
            0.5,
        )
    for fi, fname in enumerate(feet):
        cross_row(
            f"probe SLIP mm/stance-step[{fname}]",
            [results[s]["probe"]["slip_mm_per_stance_step"][fi] for s in sims],
            0.1,
        )
    cross_row(
        "probe toggles/step (mean feet)",
        [sum(results[s]["probe"]["toggles_per_step"]) / len(feet) for s in sims],
        0.02,
    )
    L.append("")

    L.append("── gait probe time series (every 10th step; per sim: footL_z | contactL | phase_rew | air_rew) ──")
    n_probe = min(len(results[s]["probe"]["phase_calls"]) for s in sims)
    hdr2 = ["step"] + sims
    wds2 = [7] + [40] * len(sims)
    L.append(_fmt(hdr2, wds2))
    for k in range(0, n_probe, 10):
        row = [str(k)]
        for s in sims:
            pc = results[s]["probe"]["phase_calls"][k]
            ac = results[s]["probe"]["air_calls"][k]
            st = results[s]["probe"]["steps"][k]
            row.append(
                f"{pc['foot_z'][0]:.4f} | {st['contact_frac'][0]:.2f} | {pc['reward_mean']:.4f} | {ac['reward_mean']:.4f}"
            )
        L.append(_fmt(row, wds2))
    L.append("")

    # ---- random phase aggregate --------------------------------------
    L.append("── random actions: aggregates ──")
    cross_row(
        "random feet_phase reward mean",
        [
            sum(c["reward_mean"] for c in results[s]["random"]["phase_calls"])
            / max(len(results[s]["random"]["phase_calls"]), 1)
            for s in sims
        ],
        0.01,
    )
    cross_row(
        "random toggles/step (mean feet)",
        [sum(results[s]["random"]["toggles_per_step"]) / len(feet) for s in sims],
        0.02,
    )
    L.append("")

    # ---- phase counter sanity ----------------------------------------
    L.append("── gait phase counter (probe step 0/50/last; must be identical across sims) ──")
    for k in [0, min(50, n_probe - 1), n_probe - 1]:
        vals = {s: results[s]["probe"]["phase_calls"][k]["phase"] for s in sims}
        L.append(f"  step {k}: {vals}")
    L.append("")

    L.append("── FLAGS (genesis deviates from mjlab by >tol AND >3x newton's deviation) ──")
    if flags:
        for f in flags:
            L.append(f"  !! {f}")
    else:
        L.append("  (none — genesis tracks mjlab as closely as newton does)")

    report = "\n".join(L)
    out_path.write_text(report + "\n")
    print()
    print(report)
    print(f"\nReport written to: {out_path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", default=None, help="internal: run one sim")
    ap.add_argument("--result-json", default=None, help="internal: child result path")
    ap.add_argument("--num-envs", type=int, default=512)
    ap.add_argument("--settle-steps", type=int, default=50)
    ap.add_argument("--probe-steps", type=int, default=200)
    ap.add_argument("--random-steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="k1_feet_contact_parity_diag.txt")
    args = ap.parse_args()

    if args.cell is not None:
        result = run_cell(args.cell, args.num_envs, args.settle_steps, args.probe_steps, args.random_steps, args.seed)
        Path(args.result_json).write_text(json.dumps(result))
        return 0

    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
