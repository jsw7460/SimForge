"""Verify joint-subset (asset_cfg.joint_names) DR on every backend/path.

randomize_pd_gains / randomize_joint_armature / randomize_joint_friction accept
``asset_cfg.joint_names`` to randomize a canonical joint SUBSET (e.g. arm-only
on a mobile manipulator). Previously only mjlab honored it — Genesis/Newton and
the explicit-actuator path silently randomized ALL DOFs, so the same config ran
a different experiment per simulator.

This diag builds Go2 (2 envs), applies each DR term to the 4 hip joints of
env 0 ONLY, and checks three invariants against a before-snapshot (values read
back in canonical joint order from the store the term actually mutates):

  hip_changed      — env 0 hip columns changed
  others_untouched — env 0 non-hip columns identical
  env1_untouched   — env 1 fully identical

Paths covered per invocation:
  --sim genesis|newton|mujoco       explicit path for pd-gains (Go2 default) +
                                    armature/friction (per-sim sim store; on
                                    mjlab read back from the mujoco-warp model)
  --sim genesis|newton --implicit   actuators swapped to ImplicitActuatorCfg ->
                                    the per-sim implicit pd-gains backends
  --sim mujoco --implicit           pd-gain joint subsets are inexpressible via
                                    mjlab (whole-actuator-group granularity):
                                    verifies the call REJECTS the subset loudly

Run (GPU box):
    jaxpy rlworld/scripts/diag/dr_joint_subset_diag.py --sim genesis
    jaxpy rlworld/scripts/diag/dr_joint_subset_diag.py --sim genesis --implicit
    jaxpy rlworld/scripts/diag/dr_joint_subset_diag.py --sim newton
    jaxpy rlworld/scripts/diag/dr_joint_subset_diag.py --sim newton --implicit
    jaxpy rlworld/scripts/diag/dr_joint_subset_diag.py --sim mujoco
    jaxpy rlworld/scripts/diag/dr_joint_subset_diag.py --sim mujoco --implicit
"""

from __future__ import annotations

import argparse

import torch

from rlworld.rl.actuators.actuator_cfg import ImplicitActuatorCfg
from rlworld.rl.configs.presets.go2.base import Go2FlatConfig
from rlworld.rl.configs.scene.entity_selector import SceneEntitySelector
from rlworld.rl.envs.mdp.events.dr.unified import (
    randomize_joint_armature,
    randomize_joint_friction,
    randomize_pd_gains,
)
from rlworld.rl.runners import BaseRunner

SUBSET = [".*_hip_joint"]
DR_ENV = 0  # randomize env 0 only; env 1 must stay untouched


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", choices=["genesis", "newton", "mujoco"], default="genesis")
    ap.add_argument("--implicit", action="store_true", help="Swap Go2 to ImplicitActuatorCfg (sim-PD paths).")
    args = ap.parse_args()

    cfg = Go2FlatConfig(sim_type=args.sim, num_envs=2)
    cfgs = cfg.build()
    if args.implicit:
        ent = cfgs.scene.entities["robot"]
        ent.articulation.actuators = tuple(
            ImplicitActuatorCfg(
                target_names_expr=a.target_names_expr,
                stiffness=a.stiffness,
                damping=a.damping,
                effort_limit=a.effort_limit,
                velocity_limit=a.velocity_limit,
                armature=a.armature,
                frictionloss=a.frictionloss,
            )
            for a in ent.articulation.actuators
        )

    env = BaseRunner.create_with_env(cfgs).env
    env.reset()

    mode = "implicit" if args.implicit else "explicit"
    print("=" * 76)
    print(f"DR JOINT-SUBSET DIAG  [sim={args.sim} mode={mode}]  has_explicit={env.act_manager.has_explicit_actuators}")
    print("=" * 76)
    if args.implicit == env.act_manager.has_explicit_actuators:
        print("FAIL: actuator mode does not match the requested test mode")
        return 1

    resolved = env.resolve_selector(SceneEntitySelector(name="robot", joint_names=SUBSET))
    sel = resolved.joint_ids
    n_act = len(env.act_manager.indexing.joint_names)
    sel_mask = torch.zeros(n_act, dtype=torch.bool, device=env.device)
    sel_mask[sel] = True
    print(f"[subset] {SUBSET} -> canonical ids {sel.tolist()} ({resolved.joint_names})")

    # ── Canonical-order readers for the store each term actually mutates ──
    def read_explicit_kp():
        out = torch.zeros((env.num_envs, n_act), device=env.device)
        for act, jidx in env.act_manager.actuators:
            out[:, jidx] = act.stiffness
        return out.clone()

    def read_genesis(getter):
        vals = torch.as_tensor(getter()).float()
        sim_idx = env.act_manager.indexing.sim_indices
        return (vals[sim_idx] if vals.dim() == 1 else vals[:, sim_idx]).clone()

    def read_newton(attr):
        import warp as wp

        from rlworld.rl.envs.mdp.events.dr.unified import _newton_dof_view

        view = env.scene_manager.robot_view
        vals = _newton_dof_view(wp.to_torch(view.get_attribute(attr, env.scene_manager.model)))
        return vals[:, env.act_manager.indexing.newton_qd_indices].clone()

    # ── mjlab readback: canonical joint -> mj actuator / dof, then read the
    #    per-world mujoco-warp model fields the mjlab dr terms mutate. Verifies
    #    the FULL chain including our selector->mjlab SceneEntityCfg conversion.
    def _mj_maps():
        import mujoco

        mjm = env.scene_manager.mj_model
        name2jid = {}
        for j in range(mjm.njnt):
            nm = mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
            name2jid[nm.rsplit("/", 1)[-1]] = j  # attached names are "robot/<name>"
        jid2act = {}
        for a in range(mjm.nu):
            if mjm.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT:
                jid2act[int(mjm.actuator_trnid[a, 0])] = a
        return mjm, name2jid, jid2act

    def read_mujoco(field: str):
        mjm, name2jid, jid2act = _mj_maps()
        sim = env.scene_manager.sim
        model = env.scene_manager.model
        raw = getattr(model, field)
        wp_shape = tuple(getattr(sim.wp_model, field).shape)
        t = raw._tensor if hasattr(raw, "_tensor") else raw
        stride0 = t.stride(0) if hasattr(t, "stride") and t.dim() > 0 else "?"
        print(
            f"    [read_mujoco] {field}: type={type(raw).__name__} wp_shape={wp_shape} "
            f"bridge_shape={tuple(t.shape)} stride0={stride0} "
            f"in_expanded={field in sim._expanded_fields}"
        )
        if isinstance(raw, torch.Tensor):
            vals = raw
        elif hasattr(raw, "_tensor"):  # mjlab TorchArray proxy
            vals = raw._tensor
        elif hasattr(raw, "__dlpack__"):
            # DLPack works for warp arrays regardless of warp-version quirks
            # (wp.to_torch chokes on this build's device object).
            vals = torch.from_dlpack(raw)
        else:
            raise RuntimeError(f"{field}: unsupported model field type {type(raw)}")
        # Un-expanded mjwarp fields may lack the per-world dim entirely
        # (mjlab's @requires_model_fields expands them on the term's 1st call).
        core_dims = 2 if field == "actuator_gainprm" else 1
        if vals.dim() == core_dims:
            vals = vals[None]
        if vals.shape[0] == 1:
            vals = vals.expand(env.num_envs, *vals.shape[1:])
        cols = []
        for n in env.act_manager.indexing.joint_names:
            jid = name2jid[n]
            if field == "actuator_gainprm":
                cols.append(vals[:, jid2act[jid], 0])  # gainprm[0] = kp for <position>
            else:  # dof_armature / dof_frictionloss, indexed by the joint's dof
                cols.append(vals[:, int(mjm.jnt_dofadr[jid])])
        return torch.stack(cols, dim=1).clone().float()

    entity = env.scene_manager["robot"] if args.sim == "genesis" else None
    checks: list[tuple[str, object, object]] = []  # (label, reader, dr_call)
    results: dict[str, bool] = {}
    asset = SceneEntitySelector(name="robot", joint_names=SUBSET)
    env_ids = torch.tensor([DR_ENV], device=env.device)

    if not args.implicit:
        checks.append(
            (
                "pd_kp(explicit)",
                read_explicit_kp,
                lambda: randomize_pd_gains(env, env_ids, asset_cfg=asset, kp_range=(0.5, 1.5), operation="scale"),
            )
        )
    elif args.sim == "genesis":
        checks.append(
            (
                "pd_kp(genesis-implicit)",
                lambda: read_genesis(entity.get_dofs_kp),
                lambda: randomize_pd_gains(env, env_ids, asset_cfg=asset, kp_range=(0.5, 1.5), operation="scale"),
            )
        )
    elif args.sim == "newton":
        checks.append(
            (
                "pd_kp(newton-implicit)",
                lambda: read_newton("joint_target_ke"),
                lambda: randomize_pd_gains(env, env_ids, asset_cfg=asset, kp_range=(0.5, 1.5), operation="scale"),
            )
        )
    else:
        # mjlab's dr.pd_gains randomizes whole actuator groups; joint-level
        # subsets are unsupported on the mujoco implicit path and must FAIL
        # LOUDLY (previously they silently randomized all joints).
        try:
            randomize_pd_gains(
                env, env_ids, asset_cfg=asset, kp_range=(0.5, 1.5), kd_range=(0.5, 1.5), operation="scale"
            )
            print("[pd_kp(mjlab-implicit)] no error raised — silent full-set randomization!")
            results["pd_kp(mjlab-implicit rejects subset)"] = False
        except NotImplementedError as e:
            print(f"[pd_kp(mjlab-implicit)] raised as expected: {e}")
            results["pd_kp(mjlab-implicit rejects subset)"] = True

    if args.sim == "genesis":
        checks.append(
            (
                "armature(sim)",
                lambda: read_genesis(entity.get_dofs_armature),
                lambda: randomize_joint_armature(
                    env, env_ids, asset_cfg=asset, armature_range=(0.5, 1.5), operation="scale"
                ),
            )
        )
        checks.append(
            (
                "joint_friction(sim)",
                lambda: read_genesis(entity.get_dofs_frictionloss),
                lambda: randomize_joint_friction(
                    env, env_ids, asset_cfg=asset, friction_range=(0.02, 0.05), operation="abs"
                ),
            )
        )
    elif args.sim == "newton":
        checks.append(
            (
                "armature(sim)",
                lambda: read_newton("joint_armature"),
                lambda: randomize_joint_armature(
                    env, env_ids, asset_cfg=asset, armature_range=(0.5, 1.5), operation="scale"
                ),
            )
        )
        checks.append(
            (
                "joint_friction(sim)",
                lambda: read_newton("joint_friction"),
                lambda: randomize_joint_friction(
                    env, env_ids, asset_cfg=asset, friction_range=(0.02, 0.05), operation="abs"
                ),
            )
        )
    else:
        checks.append(
            (
                "armature(mjlab)",
                lambda: read_mujoco("dof_armature"),
                lambda: randomize_joint_armature(
                    env, env_ids, asset_cfg=asset, armature_range=(0.5, 1.5), operation="scale"
                ),
            )
        )
        checks.append(
            (
                "joint_friction(mjlab)",
                lambda: read_mujoco("dof_frictionloss"),
                lambda: randomize_joint_friction(
                    env, env_ids, asset_cfg=asset, friction_range=(0.02, 0.05), operation="abs"
                ),
            )
        )

    for label, reader, dr_call in checks:
        before = reader()
        dr_call()
        after = reader()
        d = (after - before).abs()
        hip_changed = bool((d[DR_ENV, sel_mask] > 1e-9).any())
        others_ok = bool((d[DR_ENV, ~sel_mask] == 0).all())
        env1_ok = bool((d[1 - DR_ENV] == 0).all())
        print(f"[{label}]")
        print(f"    before[env0]={[round(v, 4) for v in before[DR_ENV].tolist()]}")
        print(f"    after [env0]={[round(v, 4) for v in after[DR_ENV].tolist()]}")
        print(f"    hip_changed={hip_changed}  others_untouched={others_ok}  env1_untouched={env1_ok}")
        results[label] = hip_changed and others_ok and env1_ok

    print("-" * 76)
    for k, v in results.items():
        print(f"  {k:28s}: {'PASS' if v else 'FAIL'}")
    ok = bool(results) and all(results.values())
    print(f"  OVERALL                     : {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
