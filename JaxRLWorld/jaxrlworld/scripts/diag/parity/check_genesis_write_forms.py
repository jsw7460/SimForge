"""Do the Genesis managers' masked, range-addressed writes land the same bits as the index-tensor calls?

The Genesis state writer and action manager address the simulator with
env masks and DOF ranges (``GenesisDofSelection``). This checks, inside
one process and from one state, that each of those writes leaves the
solver exactly where the index-tensor form of the same write leaves it.
Every write here is absolute (a pose, a velocity, a joint state, a
control target), so applying the two forms in turn from any state must
produce identical solver state after each, and the reads behind the
managers must return identical values.

For each write: the index-tensor form (the calls the writer made before)
is applied and the solver state snapshotted, then the manager's form is
applied with the same values and snapshotted again, then the index form
once more. The snapshots cover the configuration, DOF positions and
velocities, every link's pose and velocity, the collider's live contact
counts and the control buffers.

    jaxpy -m jaxrlworld.scripts.diag.parity.check_genesis_write_forms --num-envs 64
"""

from __future__ import annotations

import argparse
import importlib
import sys

import torch
from genesis.utils.misc import qd_to_torch

from jaxrlworld.rl.evals.sim_initializers import get_initializer
from jaxrlworld.rl.utils.quat_utils import quat_rotate_inverse_wxyz


def build_env(config: str, num_envs: int):
    mod_path, cls_name = config.split(":", 1)
    cfg_cls = getattr(importlib.import_module(mod_path), cls_name)
    cfgs = cfg_cls(sim_type="genesis", num_envs=num_envs, seed=0).build()
    return get_initializer("Genesis").init_environment(cfgs)


def snapshot(env) -> dict[str, torch.Tensor]:
    entity = env.scene_manager.entities["robot"]
    solver = entity.solver
    dofs = solver.dyn_state.dofs
    return {
        "qpos": entity.get_qpos().clone(),
        "dofs_pos": entity.get_dofs_position().clone(),
        "dofs_vel": entity.get_dofs_velocity().clone(),
        "links_pos": entity.get_links_pos().clone(),
        "links_quat": entity.get_links_quat().clone(),
        "links_vel": entity.get_links_vel().clone(),
        "links_ang": entity.get_links_ang().clone(),
        "n_contacts": qd_to_torch(solver.collider.collider_state.n_contacts).clone(),
        "ctrl_mode": qd_to_torch(dofs.ctrl_mode, transpose=True).clone(),
        "ctrl_pos": qd_to_torch(dofs.ctrl_pos, transpose=True).clone(),
        "ctrl_vel": qd_to_torch(dofs.ctrl_vel, transpose=True).clone(),
    }


def compare(label: str, a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> bool:
    bad = [key for key in a if not torch.equal(a[key], b[key])]
    for key in bad:
        print(f"  {label}: {key} differs, max |diff| {(a[key].float() - b[key].float()).abs().max():.3e}")
    print(f"  {label}: {'IDENTICAL' if not bad else 'DIFFERS'}")
    return not bad


class IndexTensorWriter:
    """The writer's calls in their index-tensor form (env id tensor, DOF index tensor)."""

    def __init__(self, entity, dof_ids: torch.Tensor):
        self._entity = entity
        self._dof_ids = dof_ids

    def set_root_pose(self, pos, quat_wxyz, env_ids):
        self._entity.set_pos(pos, envs_idx=env_ids, zero_velocity=False, skip_forward=True)
        self._entity.set_quat(quat_wxyz, envs_idx=env_ids, zero_velocity=False)

    def set_root_velocity(self, lin_vel, ang_vel, env_ids):
        quat_wxyz = self._entity.get_quat(envs_idx=env_ids)
        combined = torch.cat([lin_vel, quat_rotate_inverse_wxyz(quat_wxyz, ang_vel)], dim=-1)
        self._entity.set_dofs_velocity(velocity=combined, dofs_idx_local=list(range(6)), envs_idx=env_ids)

    def set_dof_state(self, positions, velocities, env_ids):
        self._entity.set_dofs_velocity(
            velocity=velocities, dofs_idx_local=self._dof_ids, envs_idx=env_ids, skip_forward=True
        )
        self._entity.set_dofs_position(
            position=positions, dofs_idx_local=self._dof_ids, envs_idx=env_ids, zero_velocity=False
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="jaxrlworld.rl.configs.presets.g1_29dof.base:G1FlatConfig")
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--reset-envs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    env = build_env(args.config, args.num_envs)
    env.reset()
    actions = torch.zeros((env.num_envs, env.num_actions), device=env.device)
    for _ in range(10):
        env.step(actions)

    entity = env.scene_manager.entities["robot"]
    rd = env.get_entity_data("robot")
    writer = env.get_robot_state_writer("robot")
    dof_ids = env.entity_indexing("robot").sim_indices
    old = IndexTensorWriter(entity, dof_ids)
    gen = torch.Generator(device=env.device)
    gen.manual_seed(args.seed)
    n = args.reset_envs
    env_ids = torch.randperm(env.num_envs, device=env.device, generator=gen)[:n].sort().values
    num_joints = len(dof_ids)

    def uniform(shape, lo, hi):
        return torch.rand(shape, device=env.device, generator=gen) * (hi - lo) + lo

    pos = rd.root_link_pos_w[env_ids] + uniform((n, 3), -0.3, 0.3)
    quat = torch.nn.functional.normalize(uniform((n, 4), -1.0, 1.0), dim=-1)
    lin_vel = uniform((n, 3), -0.5, 0.5)
    ang_vel = uniform((n, 3), -0.5, 0.5)
    joint_pos = rd.joint_pos[env_ids] + uniform((n, num_joints), -0.2, 0.2)
    joint_vel = uniform((n, num_joints), -1.0, 1.0)
    targets = rd.joint_pos + uniform((env.num_envs, num_joints), -0.3, 0.3)

    ok = True
    print("reset (root pose + root velocity + joint state)")
    old.set_root_pose(pos, quat, env_ids)
    old.set_root_velocity(lin_vel, ang_vel, env_ids)
    old.set_dof_state(joint_pos, joint_vel, env_ids)
    s_old = snapshot(env)
    writer.set_root_pose(pos, quat, env_ids=env_ids)
    writer.set_root_velocity(lin_vel, ang_vel, env_ids=env_ids)
    writer.set_dof_state(joint_pos, joint_vel, env_ids=env_ids)
    s_new = snapshot(env)
    ok &= compare("index form -> masked form", s_old, s_new)
    old.set_root_pose(pos, quat, env_ids)
    old.set_root_velocity(lin_vel, ang_vel, env_ids)
    old.set_dof_state(joint_pos, joint_vel, env_ids)
    ok &= compare("masked form -> index form", s_new, snapshot(env))

    print("push (root velocity alone)")
    old.set_root_velocity(lin_vel * 2, ang_vel * 2, env_ids)
    s_old = snapshot(env)
    writer.set_root_velocity(lin_vel * 2, ang_vel * 2, env_ids=env_ids)
    ok &= compare("index form -> masked form", s_old, snapshot(env))

    print("joint state on every env (env_ids=None)")
    all_pos = rd.joint_pos + uniform((env.num_envs, num_joints), -0.2, 0.2)
    all_vel = uniform((env.num_envs, num_joints), -1.0, 1.0)
    old.set_dof_state(all_pos, all_vel, torch.arange(env.num_envs, device=env.device))
    s_old = snapshot(env)
    writer.set_dof_state(all_pos, all_vel, env_ids=None)
    ok &= compare("index form -> masked form", s_old, snapshot(env))

    print("control target (control_dofs_position)")
    entity.control_dofs_position(targets, dof_ids)
    s_old = snapshot(env)
    env.act_manager._apply_position(targets, "robot")
    ok &= compare("index form -> range form", s_old, snapshot(env))

    print("reads")
    env._invalidate_cache()
    for name, index_read, manager_read in (
        ("joint_pos", entity.get_dofs_position(dof_ids), rd.joint_pos),
        ("joint_vel", entity.get_dofs_velocity(dof_ids), rd.joint_vel),
        ("applied_torque", entity.get_dofs_control_force(dofs_idx_local=dof_ids), rd.applied_torque),
    ):
        same = torch.equal(index_read, manager_read)
        ok &= same
        print(
            f"  {name}: {'IDENTICAL' if same else 'DIFFERS'}; strides index {index_read.stride()} "
            f"manager {manager_read.stride()}, contiguous {manager_read.is_contiguous()}"
        )

    print(f"[write-forms] {'ALL IDENTICAL' if ok else 'DIFFERENCES FOUND'}")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
