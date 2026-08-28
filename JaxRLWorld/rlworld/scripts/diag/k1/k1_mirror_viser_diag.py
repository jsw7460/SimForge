"""Viser: does the policy actually behave left/right symmetrically?

Renders TWO robots in one viser scene:
  - robot A (env 0): the pretrained policy walking normally.
  - robot B (env 1, blue ghost): env 1 is INITIALIZED as the left-right mirror of
    env 0 (mirror_qpos/qvel), then rolled out by the SAME policy (envs step
    together, batched).

If the policy is symmetric, B walks as A's mirror image and stays a clean
mirror. Asymmetry (today's right-side bias) shows as B drifting out of mirror —
visible to the eye. This validates both the mirror spec (physical) and the
policy's symmetry, before we add the mirror loss to training.

Run (server; jaxpy for JAX policy; open the printed viser URL):
    jaxpy -m rlworld.scripts.diag.k1.k1_mirror_viser_diag \\
        --wandb-run-path jsw7460/K1_Joystick/wdx6erdb --sim genesis --port 8080
"""

from __future__ import annotations

import argparse


def _init_mirror_env1(env, jperm, jsign) -> None:
    """Write env index 1 = left-right mirror of env index 0 (root + joints)."""
    import torch

    dev = env.device
    rd = env.get_robot_data("robot")
    writer = env.get_robot_state_writer("robot")
    ids = torch.tensor([1], device=dev)
    fy = torch.tensor([1.0, -1.0, 1.0], device=dev)
    fang = torch.tensor([-1.0, 1.0, -1.0], device=dev)

    pos = rd.root_link_pos_w[0:1] * fy
    q = rd.root_link_quat_w[0:1]  # wxyz
    quat = torch.stack([q[:, 0], -q[:, 1], q[:, 2], -q[:, 3]], dim=-1)
    lin = rd.root_link_lin_vel_w[0:1] * fy
    ang = rd.root_link_ang_vel_w[0:1] * fang
    jp = rd.joint_pos[0:1][:, jperm] * jsign
    jv = rd.joint_vel[0:1][:, jperm] * jsign

    writer.set_root_pose(pos, quat, env_ids=ids)
    writer.set_root_velocity(lin, ang, env_ids=ids)
    writer.set_dof_positions(jp, env_ids=ids)
    writer.set_dof_velocities(jv, env_ids=ids)
    writer.eval_fk(env_ids=ids)


class _RobotGhost:
    """Full-robot ghost meshes driven by one env index's world body poses."""

    def __init__(self, server, env, env_idx: int, prefix: str, color, opacity: float = 0.55) -> None:
        import numpy as np

        self._env = env
        self._env_idx = env_idx
        self._names = list(env.scene_manager.find_body_names([".*"], entity_name="robot"))
        meshes = env.scene_manager.get_visual_meshes(tuple(self._names))
        self._handles = {}
        for name in self._names:
            m = meshes.get(name)
            if m is None or len(m.vertices) == 0:
                continue
            self._handles[name] = server.scene.add_mesh_simple(
                f"{prefix}/{name}",
                vertices=np.asarray(m.vertices, dtype=np.float32),
                faces=np.asarray(m.faces, dtype=np.int32),
                color=color,
                opacity=opacity,
                cast_shadow=False,
                receive_shadow=False,
            )

    def update(self, _env_idx_ignored=None) -> None:
        rd = self._env.get_robot_data("robot")
        pos = rd.body_pos_w_all[self._env_idx].detach().cpu().numpy()
        quat = rd.body_quat_w_all[self._env_idx].detach().cpu().numpy()
        for i, name in enumerate(self._names):
            h = self._handles.get(name)
            if h is not None:
                h.position = tuple(pos[i].tolist())
                h.wxyz = tuple(quat[i].tolist())

    def set_visible(self, visible: bool) -> None:
        for h in self._handles.values():
            h.visible = visible

    def set_opacity(self, opacity: float) -> None:
        for h in self._handles.values():
            h.opacity = opacity


class _GhostGroup:
    """Composite exposing update/set_visible/set_opacity over several ghosts."""

    def __init__(self, ghosts) -> None:
        self._ghosts = [g for g in ghosts if g is not None]

    def update(self, env_idx) -> None:
        for g in self._ghosts:
            g.update(env_idx)

    def set_visible(self, visible: bool) -> None:
        for g in self._ghosts:
            if hasattr(g, "set_visible"):
                g.set_visible(visible)

    def set_opacity(self, opacity: float) -> None:
        for g in self._ghosts:
            if hasattr(g, "set_opacity"):
                g.set_opacity(opacity)


def main() -> int:
    import numpy as np
    import torch

    from rlworld.rl.algorithms.ppo.symmetry import build_mirror_spec
    from rlworld.rl.evals import PolicyEvaluator
    from rlworld.rl.vis.viser.play_viewer import ViserPlayViewer

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wandb-run-path", required=True)
    ap.add_argument("--sim", default="genesis", choices=("genesis", "newton"))
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument(
        "--forced-mirror",
        action="store_true",
        help="every step force env1 = mirror(env0) so the ghost is a CLEAN geometric "
        "mirror of A (validates the transform). Without it, env1 rolls out "
        "independently and drifting/falling = policy asymmetry.",
    )
    args = ap.parse_args()
    forced = args.forced_mirror

    ev = PolicyEvaluator(
        wandb_run_path=args.wandb_run_path,
        eval_target=args.sim,
        num_evals=1,
        seed=0,
        record_video=False,
        save_data=False,
        use_rich_display=False,
        extra_overrides={"env": {"num_envs": 2}},
    )
    env, policy = ev.env, ev.policy
    spec = build_mirror_spec(env.obs_manager, list(env.act_manager.actuated_joint_names))
    jperm = torch.as_tensor(np.asarray(spec.action_perm), device=env.device, dtype=torch.long)
    jsign = torch.as_tensor(np.asarray(spec.action_sign), device=env.device, dtype=torch.float32)

    class MirrorViewer(ViserPlayViewer):
        def setup(self) -> None:
            super().setup()
            # Never let a resample pick env1 as a "standing" (zero-command) env,
            # which would make the ghost freeze for seconds at a time.
            self.env.command_manager._terms["velocity"].cfg.rel_standing_envs = 0.0
            _init_mirror_env1(self.env, jperm, jsign)
            mirror_ghost = _RobotGhost(self._server, self.env, env_idx=1, prefix="/mirror_ghost", color=(80, 140, 255))
            self._motion_ghost = _GhostGroup([self._motion_ghost, mirror_ghost])

        def _execute_step(self):
            if not forced:
                # Independent mode: keep env1's command the MIRROR of env0's (GUI
                # drives env0): [vx,vy,wz] -> [vx,-vy,-wz], written into the
                # velocity term buffer directly (not external-control, which would
                # break the gait_phase clock). env1 then walks on its own — drift
                # or falling = policy asymmetry.
                vt = self.env.command_manager._terms["velocity"]
                c0 = vt._command[0]
                vt._command[1, 0] = c0[0]
                vt._command[1, 1] = -c0[1]
                vt._command[1, 2] = -c0[2]
            ret = super()._execute_step()
            if forced:
                # Forced mode: overwrite env1 with the exact mirror of env0 every
                # step -> the ghost is a clean geometric mirror (transform check).
                _init_mirror_env1(self.env, jperm, jsign)
            return ret

    play_scene = ev._create_play_scene()
    print(f"[viser] robot A = env0 (normal), robot B = env1 blue ghost (mirror-init). port {args.port}")
    viewer = MirrorViewer(env=env, play_scene=play_scene, policy=policy, port=args.port)
    viewer.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
