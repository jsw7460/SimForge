"""Regression test: Genesis IMPLICIT pd-gains / armature DR must not compound.

Confirmed bug (measured 2026-07-02 with this diag; band escaped at call 2,
ratios reaching x12 / x0.06 by call 20): the Genesis backends multiplied the
CURRENT sim value (``get_dofs_kp()`` / ``get_dofs_armature()`` — Genesis
setters are absolute-set APIs on the persistent ``dofs_info`` store, and
nothing restores it on reset) by a fresh ratio each call, so per-reset DR
multiplied ratios into a log random-walk. Fixed by scaling a baseline
captured on the first call (``_genesis_dr_baseline``), like the Newton
implicit path (``env._dr_baselines``) and the explicit-actuator path.

Method: build Go2 on Genesis with its actuators swapped to
``ImplicitActuatorCfg`` (same gains — the swap routes ``randomize_pd_gains``
to the Genesis implicit backend, exactly the code under test), then call
``randomize_pd_gains(scale, kp_range=(0.5, 1.5))`` N times, tracking the sim
kp after every call against the build-time base:

  * compounding      -> ratios multiply; values escape ``base*[0.5,1.5]``
                        quickly (x0.05 .. x5 scale after ~20 calls)
  * no compounding   -> every call stays within ``base*[0.5,1.5]``

Prints the min/max kp-to-base ratio after every call and a final verdict.

Run (GPU box):
    jaxpy jaxrlworld/scripts/diag/pd_gains_compounding_diag.py
"""

from __future__ import annotations

import torch

from jaxrlworld.rl.actuators.actuator_cfg import ImplicitActuatorCfg
from jaxrlworld.rl.configs.presets.go2.base import Go2FlatConfig
from jaxrlworld.rl.envs.mdp.events.dr.unified import randomize_joint_armature, randomize_pd_gains
from jaxrlworld.rl.runners import BaseRunner

N_CALLS = 20
KP_RANGE = (0.5, 1.5)
BAND_TOL = 1.05  # values outside base*[0.5/1.05, 1.5*1.05] count as escaped


def main() -> int:
    cfg = Go2FlatConfig(sim_type="genesis", num_envs=2)
    cfgs = cfg.build()

    # Swap every actuator group to ImplicitActuatorCfg with identical gains so
    # the robot runs on Genesis' internal PD and the DR term routes to the
    # Genesis implicit backend (the code under test).
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

    print("=" * 74)
    print(f"GENESIS IMPLICIT PD-GAINS COMPOUNDING DIAG  has_explicit={env.act_manager.has_explicit_actuators}")
    print("=" * 74)
    if env.act_manager.has_explicit_actuators:
        print("FAIL: actuator swap did not take effect — test would exercise the wrong path")
        return 1

    entity = env.scene_manager["robot"]
    env_ids = torch.arange(env.num_envs, device=env.device)
    lo_band, hi_band = KP_RANGE[0] / BAND_TOL, KP_RANGE[1] * BAND_TOL

    def track(label: str, getter, dr_call) -> int | None:
        """Call ``dr_call`` N times; report the value/base ratio band each call.

        Only DOFs with a positive build-time value are compared (the 6
        floating-base DOFs have kp/armature = 0, and 0/0 would poison the
        ratio with NaN). Returns the first call index that escaped the band,
        or None if all calls stayed inside (= no compounding).
        """
        base = torch.as_tensor(getter()).clone().float()
        mask = (base > 0.0) if base.dim() == 1 else (base > 0.0).all(dim=0)
        if not bool(mask.any()):
            print(f"[{label}] all build-time values are zero — cannot test, skipping")
            return None
        base_j = base[mask] if base.dim() == 1 else base[:, mask]
        print(
            f"[{label}] base: min={base_j.min():.4f} max={base_j.max():.4f}  (excluding {int((~mask).sum())} zero DOFs)"
        )
        escaped: int | None = None
        for i in range(1, N_CALLS + 1):
            dr_call()
            val = torch.as_tensor(getter()).float()
            val_j = val[mask] if val.dim() == 1 else val[:, mask]
            ratio = val_j / base_j if val_j.dim() == base_j.dim() else val_j / base_j.unsqueeze(0)
            if torch.isnan(ratio).any():
                raise RuntimeError(f"[{label}] NaN in ratio at call {i} — diag assumptions broken")
            rmin, rmax = float(ratio.min()), float(ratio.max())
            outside = bool(rmin < lo_band or rmax > hi_band)
            if outside and escaped is None:
                escaped = i
            print(
                f"[{label} call {i:2d}] ratio: min={rmin:.4f}  max={rmax:.4f}  {'<<< OUTSIDE band' if outside else ''}"
            )
        return escaped

    kp_escaped = track(
        "kp",
        entity.get_dofs_kp,
        lambda: randomize_pd_gains(env, env_ids, kp_range=KP_RANGE, kd_range=None, operation="scale"),
    )
    arm_escaped = track(
        "armature",
        entity.get_dofs_armature,
        lambda: randomize_joint_armature(env, env_ids, armature_range=KP_RANGE, operation="scale"),
    )

    print("-" * 74)
    ok = True
    for label, escaped in (("kp", kp_escaped), ("armature", arm_escaped)):
        if escaped is not None:
            print(
                f"VERDICT[{label}]: COMPOUNDING — escaped base*[{KP_RANGE[0]}, {KP_RANGE[1]}] at call {escaped} "
                f"(each call multiplies the PREVIOUS value by a fresh ratio)."
            )
            ok = False
        else:
            print(
                f"VERDICT[{label}]: NO COMPOUNDING — all {N_CALLS} calls within base*[{KP_RANGE[0]}, {KP_RANGE[1]}] "
                f"(scale is applied to the build-time baseline)."
            )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
