"""Look at a preset's scene in Viser, without a trained policy.

Every existing Viser entry point goes through ``PolicyEvaluator`` and
therefore needs a checkpoint. That is the wrong requirement for the
question "is this scene actually assembled the way I described it" —
which is the question a new multi-entity preset raises, and the one
worth answering before spending a training run on it.

So this drives the environment with a fixed action instead of a policy.
The default is all zeros, which on a preset whose action terms use
``use_default_offset`` (or whose action config declares an ``offset``)
commands every robot to its home pose: robots hold themselves up,
props settle, and anything mis-placed is visible immediately.

Per-term sliders in the "Manual actions" panel let you push individual
joints around to check reach and clearance by hand.

    python -m rlworld.scripts.view_scene --preset yam_dual --sim newton

``--preset`` takes either a registered short name (see ``--list``) or a
``module:ClassName`` path to any preset config class.
"""

from __future__ import annotations

import argparse
import importlib

import numpy as np
import torch

# Short names for the presets worth eyeballing. Anything not here is
# still reachable as ``module:ClassName``.
_PRESETS: dict[str, str] = {
    "yam_arm": "rlworld.rl.configs.presets.yam_arm.base:YamArmConfig",
    "yam_dual": "rlworld.rl.configs.presets.yam_dual.base:YamDualArmConfig",
    "lab_cell": "rlworld.rl.configs.presets.lab_cell.base:LabCellConfig",
    "yam_lift": "rlworld.rl.configs.presets.yam_lift.base:YamLiftConfig",
    "go2": "rlworld.rl.configs.presets.go2.base:Go2FlatConfig",
    "g1_29dof": "rlworld.rl.configs.presets.g1_29dof.base:G1FlatConfig",
}

_SIMS = ("genesis", "newton", "mujoco")

# The evaluation initializers are keyed on the env class name the preset
# builds, not on the preset's own ``sim_type`` string.
_INITIALIZER_KEY = {"genesis": "Genesis", "newton": "Newton", "mujoco": "MujocoEnv"}


class FixedAction:
    """The policy interface the play viewer needs, driven by a buffer.

    Duck-typed rather than a :class:`PolicyWrapper` subclass: that base
    takes a runner, and there is no runner here. The viewer only calls
    ``get_action``, ``notify_reset`` and — when it exists — ``reset``.
    """

    def __init__(self, num_envs: int, num_actions: int, device: torch.device) -> None:
        self.action = torch.zeros(num_envs, num_actions, device=device)
        self.sliders: list = []

    def get_action(self, obs, robot_states) -> torch.Tensor:  # noqa: ARG002 — viewer's call signature
        return self.action

    def notify_reset(self, reset_idx) -> None:
        pass

    def reset(self) -> None:
        """Drop the standing command when the environment is reset.

        The viewer's Reset button puts the robots back at their home
        pose, but the command is not part of the environment and would
        survive it: the arm would return home and then walk straight
        back into whatever the sliders were last left at, which reads as
        the reset having failed. Sliders go back to the home angle, which
        is the angle a zero action commands, so the panel and the robot
        agree about what has been asked for.
        """
        self.action.zero_()
        for slider, home in self.sliders:
            slider.value = home


def _resolve_preset(name: str):
    target = _PRESETS.get(name, name)
    if ":" not in target:
        raise SystemExit(f"Unknown preset {name!r}. Use one of {sorted(_PRESETS)} or a 'module:ClassName' path.")
    module_path, class_name = target.split(":", 1)
    return getattr(importlib.import_module(module_path), class_name)


def _add_action_sliders(viewer, policy: FixedAction, env) -> None:
    """A slider per actuated joint, in RADIANS, spanning that joint's travel.

    Not in raw action units. An action is ``raw * scale + offset`` and the
    scale is per joint and small — this arm moves 0.16 rad per unit on its
    second joint against 3.67 rad of travel — so a slider over a plausible
    raw range covers a few degrees and the arm looks unable to reach
    anything. The slider therefore commands a joint ANGLE, bounded by that
    joint's own soft limits, and the raw action is solved back out of it.

    Grouped by action term, because a scene with several robots has an
    action vector whose parts belong to different machines.
    """
    server = viewer._server
    manager = env.act_manager

    def _group(label: str, entity_name: str, joint_names: list[str], columns, scale, offset) -> None:
        mid, half = manager.soft_joint_limits_of(entity_name)
        entity_joints = list(env.entity_indexing(entity_name).joint_names)
        with server.gui.add_folder(label):
            for local_idx, (column, joint) in enumerate(zip(columns, joint_names, strict=True)):
                joint_idx = entity_joints.index(joint)
                low = float(mid[joint_idx] - half[joint_idx])
                high = float(mid[joint_idx] + half[joint_idx])
                gain = float(scale[local_idx])
                home = float(offset[local_idx])
                if gain == 0.0:
                    print(f"[view_scene] {joint!r} has action scale 0 — no slider (it cannot be commanded)")
                    continue
                # A joint with no declared range reports an infinite limit
                # -- or, on Newton, a finite sentinel around 1e8 -- neither
                # of which is a number a slider can span. Say so and give
                # it a workable window around home rather than emitting a
                # control whose ends are unreachable.
                if not (np.isfinite(low) and np.isfinite(high) and high > low) or (high - low) > 1.0e3:
                    print(
                        f"[view_scene] {joint!r} has no usable joint range "
                        f"(low={low}, high={high}) — slider spans home +-pi instead"
                    )
                    low, high = home - np.pi, home + np.pi
                print(
                    f"[view_scene]   slider {joint:<14} range=[{low:+.3f}, {high:+.3f}] home={home:+.3f} scale={gain:.4f}"
                )
                slider = server.gui.add_slider(
                    joint,
                    min=round(low, 4),
                    max=round(high, 4),
                    step=round((high - low) / 200.0, 5),
                    initial_value=round(min(max(home, low), high), 4),
                )
                policy.sliders.append((slider, home))

                @slider.on_update
                def _(_event, column=column, slider=slider, gain=gain, home=home) -> None:
                    policy.action[:, column] = (float(slider.value) - home) / gain

    if manager.terms:
        for term_name, term in manager.terms.items():
            term_slice = manager.term_action_slices[term_name]
            _group(
                f"{term_name} ({term.entity_name})",
                term.entity_name,
                list(term.joint_names),
                range(term_slice.start, term_slice.stop),
                term._scale,
                term._offset,
            )
    else:
        # Legacy path: the whole action vector is the driven robot's joints,
        # and the manager owns the scale/offset directly.
        names = list(manager.actuated_joint_names)
        _group(
            f"actions ({env.robot_entity_name})",
            env.robot_entity_name,
            names,
            range(len(names)),
            manager._scale,
            manager.offset[0],
        )


def _report_geometry(play_scene) -> None:
    """Say what the simulator bridge actually handed the renderer.

    An empty viewer looks the same whether the geometry never made it out
    of the bridge or the renderer dropped it, and the two have nothing to
    do with each other. This settles which half to look at before anyone
    starts guessing.
    """
    bridge = getattr(play_scene, "_bridge", None)
    if bridge is None:
        print("[view_scene] this backend does not expose a bridge; geometry not inspected")
        return
    geometry = bridge.extract_geometry()
    total = sum(len(group.meshes) for group in geometry.mesh_groups)
    print(f"[view_scene] bridge extracted {len(geometry.mesh_groups)} body groups, {total} meshes")
    if not geometry.mesh_groups:
        print("[view_scene] NOTHING TO RENDER — the bridge found no visual geometry.")
    # Where the renderer will actually put each group. An empty viewer with
    # geometry in hand means the transforms are the problem — NaN, all-zero,
    # or somewhere the camera is not.
    bridge.begin_frame()
    positions = bridge.get_body_positions(0)
    quaternions = bridge.get_body_quaternions(0)
    print(f"[view_scene] transforms: pos{positions.shape} quat{quaternions.shape} for env 0")
    for group in geometry.mesh_groups:
        faces = sum(len(m.faces) for m in group.meshes)
        if 0 <= group.body_id < len(positions):
            pos = ", ".join(f"{v:+.3f}" for v in positions[group.body_id])
            quat = ", ".join(f"{v:+.3f}" for v in quaternions[group.body_id])
            where = f"pos=[{pos}] quat=[{quat}]"
        else:
            where = f"NO TRANSFORM (body_id outside 0..{len(positions) - 1})"
        print(
            f"[view_scene]   body {group.body_id:>4}  {group.body_name:<24} "
            f"meshes={len(group.meshes):<3} faces={faces:<6} {where}"
        )
    finite = bool(np.isfinite(positions).all() and np.isfinite(quaternions).all())
    print(f"[view_scene] all transforms finite: {finite}")

    # The vertex buffers themselves. A single non-finite vertex gives the
    # renderer a NaN bounding volume, which kills the client during its
    # first frame — a blank page with no GUI at all, rather than one
    # missing mesh. Degenerate or absurd extents do the same to the
    # camera fit, so the bounds are reported too.
    bad: list[str] = []
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for group in geometry.mesh_groups:
        for mesh_idx, mesh in enumerate(group.meshes):
            verts = np.asarray(mesh.vertices, dtype=np.float64)
            faces_arr = np.asarray(mesh.faces)
            where = f"{group.body_name}[{mesh_idx}]"
            if verts.size == 0 or faces_arr.size == 0:
                bad.append(f"{where}: empty ({verts.shape} verts, {faces_arr.shape} faces)")
                continue
            if not np.isfinite(verts).all():
                n_bad = int((~np.isfinite(verts)).any(axis=1).sum())
                bad.append(f"{where}: {n_bad} non-finite vertices")
                continue
            if faces_arr.max() >= len(verts):
                bad.append(f"{where}: face index {int(faces_arr.max())} >= {len(verts)} vertices")
                continue
            lo = np.minimum(lo, verts.min(axis=0))
            hi = np.maximum(hi, verts.max(axis=0))
            visual_type = type(mesh.visual).__name__
            if visual_type not in ("ColorVisuals",):
                bad.append(f"{where}: visual is {visual_type}")
    print(f"[view_scene] mesh vertex bounds: min={np.round(lo, 3).tolist()} max={np.round(hi, 3).tolist()}")
    if bad:
        print(f"[view_scene] {len(bad)} SUSPECT MESHES:")
        for line in bad[:20]:
            print(f"[view_scene]   {line}")
    else:
        print("[view_scene] every mesh has finite vertices, valid faces and plain colour visuals")
    print(f"[view_scene] num_envs reported by the bridge: {bridge.num_envs}")
    print(f"[view_scene] tracked body id: {bridge.tracked_body_id}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="yam_dual", help="Short name or 'module:ClassName'.")
    ap.add_argument("--sim", default="newton", choices=list(_SIMS))
    ap.add_argument("--num-envs", type=int, default=1)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--no-share", action="store_true", help="Do not open a public share URL.")
    ap.add_argument("--list", action="store_true", help="List the registered preset names and exit.")
    args = ap.parse_args()

    if args.list:
        for name, target in sorted(_PRESETS.items()):
            print(f"  {name:<12} {target}")
        return 0

    from rlworld.rl.evals.sim_initializers import get_initializer
    from rlworld.rl.runners import BaseRunner
    from rlworld.rl.vis.viser.play_viewer import ViserPlayViewer

    preset_cls = _resolve_preset(args.preset)
    cfgs = preset_cls(sim_type=args.sim, num_envs=args.num_envs).build()
    env = BaseRunner._create_env_from_config(cfgs)
    env.reset()

    policy = FixedAction(env.num_envs, env.act_manager.num_actions, env.device)
    play_scene = get_initializer(_INITIALIZER_KEY[args.sim]).create_play_scene(env)

    class SceneViewer(ViserPlayViewer):
        def setup(self) -> None:
            super().setup()
            _add_action_sliders(self, policy, env)

    viewer = SceneViewer(
        env=env,
        play_scene=play_scene,
        policy=policy,
        port=args.port,
        share=not args.no_share,
    )
    print(f"[view_scene] preset={args.preset} sim={args.sim} num_envs={args.num_envs}")
    print(f"[view_scene] entities={list(env.scene_manager.entities)}")
    print(f"[view_scene] rigid objects={list(env.scene_manager.rigid_objects)}")
    print(f"[view_scene] action dim={env.act_manager.num_actions}")
    _report_geometry(play_scene)
    viewer.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
