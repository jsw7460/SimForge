"""Pure-mjlab minimal reproduction: expand_model_fields at scale.

Zero JaxRLWorld code — imports only mujoco / mjlab / warp / torch.  Builds
an inline 3-hinge pendulum model, constructs ``mjlab.sim.Sim`` directly,
expands ``dof_frictionloss`` + ``body_ipos`` to per-env storage, writes
per-env values through the same ModelBridge/TorchArray path mjlab's own
DR core uses, then steps and resets with graph replay.

Purpose: decide whether the ``CUDA_ERROR_ILLEGAL_ADDRESS`` seen at
num_envs>=256 in the JaxRLWorld matrix diagnostic
(``mjlab_dr_training_matrix_diag``) reproduces with mjlab APIs alone.

    - CRASHES here too  -> mjlab / mujoco-warp / warp version-combo bug.
      This file IS the upstream issue reproduction.
    - PASSES here       -> the fault is in how JaxRLWorld drives mjlab;
      bisect our adapter/write path next.

Usage (GPU box, run with CUDA_LAUNCH_BLOCKING=1 for exact fault points):
    CUDA_LAUNCH_BLOCKING=1 python -m rlworld.scripts.diag.dr.mjlab_expand_minimal_repro
    CUDA_LAUNCH_BLOCKING=1 python -m rlworld.scripts.diag.dr.mjlab_expand_minimal_repro --num-envs 8
"""

from __future__ import annotations

import argparse

import mujoco
import torch
import warp as wp
from mjlab.managers.event_manager import _DERIVED_FIELDS, RecomputeLevel
from mjlab.sim.sim import Simulation, SimulationCfg

_XML = """
<mujoco model="expand_repro">
  <option timestep="0.005"/>
  <worldbody>
    <body name="link1" pos="0 0 1">
      <joint name="j1" type="hinge" axis="0 1 0" frictionloss="0.1" armature="0.01"/>
      <geom name="g1" type="capsule" fromto="0 0 0 0 0 -0.3" size="0.04" mass="1.0"/>
      <body name="link2" pos="0 0 -0.3">
        <joint name="j2" type="hinge" axis="0 1 0" frictionloss="0.1" armature="0.01"/>
        <geom name="g2" type="capsule" fromto="0 0 0 0 0 -0.3" size="0.04" mass="1.0"/>
        <body name="link3" pos="0 0 -0.3">
          <joint name="j3" type="hinge" axis="0 1 0" frictionloss="0.1" armature="0.01"/>
          <geom name="g3" type="capsule" fromto="0 0 0 0 0 -0.3" size="0.04" mass="1.0"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

_FIELDS = ("dof_frictionloss", "body_ipos")


def _stage(msg: str) -> None:
    print(f"[STAGE] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num-envs", type=int, default=256)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    n = args.num_envs
    torch.manual_seed(args.seed)

    import mjlab

    print(f"[INFO] mjlab {getattr(mjlab, '__version__', '?')}  warp {wp.__version__}  mujoco {mujoco.__version__}")
    import mujoco_warp

    print(f"[INFO] mujoco_warp {getattr(mujoco_warp, '__version__', '?')}")
    print(f"[INFO] num_envs={n} steps={args.steps}")

    mj_model = mujoco.MjModel.from_xml_string(_XML)
    sim = Simulation(num_envs=n, cfg=SimulationCfg(), model=mj_model, device="cuda:0")
    torch.cuda.synchronize()
    _stage(f"Sim built (use_cuda_graph={sim.use_cuda_graph})")

    # Expand the DR-written fields PLUS the derived fields that
    # recompute_constants(set_const) writes — exactly what mjlab's own
    # ``requires_model_fields(..., recompute=set_const)`` nominates.
    expand_fields = tuple(dict.fromkeys(_FIELDS + _DERIVED_FIELDS[RecomputeLevel.set_const]))
    print(f"[INFO] expanding: {expand_fields}")
    sim.expand_model_fields(expand_fields)
    torch.cuda.synchronize()
    for f in _FIELDS:
        shape = tuple(getattr(sim.wp_model, f).shape)
        print(f"[INFO] {f}: wp_model shape = {shape}")
        assert shape[0] == n, f"{f} not expanded: {shape}"
    _stage("expand_model_fields done")

    # Per-env write through the ModelBridge (TorchArray) — the same path
    # mjlab's DR core (_randomize_model_field) uses.
    fl = sim.model.dof_frictionloss
    fl_vals = torch.rand((n, fl.shape[1]), device="cuda:0") * 0.3
    fl[:] = fl_vals
    ipos = sim.model.body_ipos
    ipos_vals = torch.rand((n, ipos.shape[1], 3), device="cuda:0") * 0.01
    ipos[:] = ipos_vals
    torch.cuda.synchronize()
    _stage("per-env DR writes done")

    # The exact call the JaxRLWorld mass/com DR backends make afterwards —
    # and the launch (mujoco_warp set_const -> smooth.kinematics) where the
    # matrix diagnostic crashes at num_envs>=256.
    sim.recompute_constants(RecomputeLevel.set_const)
    torch.cuda.synchronize()
    _stage("recompute_constants(set_const) done")

    # Read back and verify the writes actually landed per-env.
    fl_back = wp.to_torch(sim.wp_model.dof_frictionloss).detach()
    assert torch.allclose(fl_back, fl_vals), "dof_frictionloss readback mismatch"
    env_std = float(fl_back.std(dim=0).mean().item())
    print(f"[INFO] dof_frictionloss env-axis std = {env_std:.3e}")
    assert env_std > 0.0, "dof_frictionloss is env-shared after write"
    _stage("readback verified")

    # Step with graph replay, then reset, then step again.
    for k in range(args.steps):
        sim.step()
    torch.cuda.synchronize()
    _stage(f"{args.steps} sim.step() done")

    sim.reset()
    torch.cuda.synchronize()
    _stage("sim.reset() done")

    for k in range(args.steps):
        sim.step()
    torch.cuda.synchronize()
    _stage(f"{args.steps} more sim.step() done")

    qpos = wp.to_torch(sim.wp_data.qpos).detach()
    assert bool(torch.isfinite(qpos).all().item()), "non-finite qpos after stepping"
    print(f"\nOVERALL: PASS (num_envs={n})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
