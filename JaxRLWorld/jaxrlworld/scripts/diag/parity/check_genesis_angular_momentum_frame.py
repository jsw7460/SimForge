"""Which frame does Genesis store a link's inertia tensor in, and does our angular momentum use it?

``GenesisRigidObjectData.angular_momentum_w`` rotates ``dyn_info.links.inertial_i``
by the LINK quaternion. Genesis's MJCF loader stores ``inertial_i`` as the
diagonal inertia in the body's INERTIAL frame and keeps the inertial frame's
orientation separately in ``inertial_quat``; forward kinematics composes the
two into ``dyn_state.links.i_quat`` (the inertial frame in world). For every
link whose ``<inertial quat>`` is not the identity the spin term is therefore
rotated by the wrong frame.

This diag builds the preset on Genesis, drives it with random actions so the
links move, and compares three spin sums on the live state:

    link   - the current implementation: R(link quat) I R(link quat)^T w
    iquat  - R(i_quat) I R(i_quat)^T w, i_quat read from the solver
    manual - R(link quat * inertial_quat) I R(...)^T w, composed by hand

``iquat`` and ``manual`` agreeing confirms the frame convention; their gap to
``link`` is the error in the reward term ``angular_momentum_penalty`` reads.

    jaxpy -m jaxrlworld.scripts.diag.parity.check_genesis_angular_momentum_frame --preset g1_flat
    jaxpy -m jaxrlworld.scripts.diag.parity.check_genesis_angular_momentum_frame --preset k1_velocity
"""

from __future__ import annotations

import argparse

import torch
from genesis.utils.misc import qd_to_torch

from jaxrlworld.rl.runners import BaseRunner
from jaxrlworld.rl.utils.quat_utils import quat_mul_wxyz, quat_rotate_inverse_wxyz, quat_rotate_wxyz
from jaxrlworld.scripts.diag.gates.check_all_presets import _PUBLIC, _load


def _spin(q_wxyz: torch.Tensor, inertia: torch.Tensor, omega_w: torch.Tensor) -> torch.Tensor:
    omega_f = quat_rotate_inverse_wxyz(q_wxyz, omega_w)
    return quat_rotate_wxyz(q_wxyz, torch.einsum("nbij,nbj->nbi", inertia, omega_f)).sum(dim=1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--preset", required=True, help="label from check_all_presets' table")
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--steps", type=int, default=20, help="random-action steps before the comparison")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    loader = next(loader for label, loader, sims in _PUBLIC if label == args.preset and "genesis" in sims)
    cfgs = _load(loader, "genesis", args.num_envs)
    env = BaseRunner.create_with_env(cfgs, use_wandb=False).env
    env.reset()
    gen = torch.Generator().manual_seed(args.seed)
    for _ in range(args.steps):
        actions = (torch.randn((env.num_envs, env.num_actions), generator=gen) * 0.5).to(env.device)
        env.step(actions)
    env._invalidate_cache()

    rd = env.get_entity_data("robot")
    solver = rd._entity._solver
    link_ids = rd._global_link_ids
    n = env.num_envs

    inertia = qd_to_torch(solver.dyn_info.links.inertial_i, None, link_ids, transpose=True, copy=True)
    inertial_quat = qd_to_torch(solver.dyn_info.links.inertial_quat, None, link_ids, transpose=True, copy=True)
    if inertia.dim() == 3:
        inertia = inertia.unsqueeze(0).expand(n, -1, -1, -1)
    if inertial_quat.dim() == 2:
        inertial_quat = inertial_quat.unsqueeze(0).expand(n, -1, -1)
    i_quat = qd_to_torch(solver.dyn_state.links.i_quat, None, link_ids, transpose=True, copy=True)
    link_quat = rd.body_quat_w_all
    omega = rd.body_ang_vel_w_all

    off_diag = inertia - torch.diag_embed(torch.diagonal(inertia, dim1=-2, dim2=-1))
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0], device=inertial_quat.device)
    non_identity = (inertial_quat[0] - identity).abs().amax(dim=-1) > 1e-6

    spin_link = _spin(link_quat, inertia, omega)
    spin_iquat = _spin(i_quat, inertia, omega)
    spin_manual = _spin(quat_mul_wxyz(link_quat, inertial_quat), inertia, omega)
    total = rd.angular_momentum_w()
    orbital = total - spin_link
    corrected = orbital + spin_iquat

    rel = (spin_link - spin_iquat).norm(dim=-1) / spin_iquat.norm(dim=-1).clamp(min=1e-9)
    rel_total = (total - corrected).norm(dim=-1) / corrected.norm(dim=-1).clamp(min=1e-9)
    print(f"{args.preset}:genesis  num_envs={n}  links={inertia.shape[1]}  after {args.steps} random steps")
    print(
        f"  inertial_i off-diagonal max |.|          : {float(off_diag.abs().max()):.3e}  (0 -> stored diagonal, inertial frame)"
    )
    print(f"  links with non-identity <inertial quat>  : {int(non_identity.sum())}/{inertia.shape[1]}")
    print(
        f"  |spin(iquat) - spin(manual)| max         : {float((spin_iquat - spin_manual).abs().max()):.3e}  (frame convention check)"
    )
    print(f"  |spin(link) - spin(iquat)| / |spin(iquat)|: median {float(rel.median()):.3f}  max {float(rel.max()):.3f}")
    print(
        f"  |L(current) - L(corrected)| / |L(corrected)|: median {float(rel_total.median()):.3f}  max {float(rel_total.max()):.3f}"
    )
    print(
        f"  |L| median: current {float(total.norm(dim=-1).median()):.4f}  corrected {float(corrected.norm(dim=-1).median()):.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
