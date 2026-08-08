"""Interactive COM-probe viewer — push the K1 trunk COM and watch it tip / drift.

Loads a trained policy into the viser play viewer and adds two sliders
(COM forward x / COM lateral y). Every step the Trunk COM is pinned to the
slider offset (added onto the MJCF default via randomize_body_com_offset), so
you can push the COM forward and SEE whether the robot leans / near-falls, and
whether lateral commands then drift forward — the hardware symptom, reproduced
on demand.

Numeric counterpart (same forcing, buckets by command direction, prints tilt /
drift / falls):
    jaxpy -m rlworld.scripts.diag.k1_gait_direction_diag \\
        --wandb-run-path <run> --sim mujoco --com-x 0.03

viser works on all three backends (mujoco / newton / genesis). Drive the robot
with the command panel while sweeping the COM sliders.

Run (JAX -> jaxpy):
    jaxpy -m rlworld.scripts.diag.k1_com_probe_viser_diag \\
        --wandb-run-path jsw7460/K1_Joystick/<run> --sim mujoco
"""

from __future__ import annotations

import argparse


def main() -> int:
    import torch

    from rlworld.rl.configs.scene import SceneEntitySelector
    from rlworld.rl.envs.mdp.events.dr import unified as unified_dr
    from rlworld.rl.evals import PolicyEvaluator
    from rlworld.rl.vis.viser.play_viewer import ViserPlayViewer

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wandb-run-path", required=True)
    ap.add_argument("--sim", default="mujoco", choices=("mujoco", "newton", "genesis"))
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    ev = PolicyEvaluator(
        wandb_run_path=args.wandb_run_path,
        eval_target=args.sim,
        num_evals=1,
        seed=0,
        record_video=False,
        save_data=False,
        use_rich_display=False,
        extra_overrides={"env": {"num_envs": 1}},
    )

    # DR backends expect a RESOLVED entity (body_ids); resolve once like the event manager does.
    trunk_resolved = ev.env.resolve_selector(SceneEntitySelector(name="robot", body_names=("Trunk",)))

    class ComProbeViewer(ViserPlayViewer):
        def setup(self) -> None:
            super().setup()
            with self._server.gui.add_folder("COM probe"):
                self._com_x = self._server.gui.add_slider(
                    "COM forward x (m)", min=-0.05, max=0.10, step=0.005, initial_value=0.0
                )
                self._com_y = self._server.gui.add_slider(
                    "COM lateral y (m)", min=-0.05, max=0.05, step=0.005, initial_value=0.0
                )

        def _force_com(self) -> None:
            cx = float(self._com_x.value)
            cy = float(self._com_y.value)
            ranges = {}
            if cx != 0.0:
                ranges[0] = (cx, cx)
            if cy != 0.0:
                ranges[1] = (cy, cy)
            if not ranges:
                return  # slider at 0 -> leave COM at the MJCF default
            unified_dr.randomize_body_com_offset(
                self.env,
                env_ids=torch.arange(self.env.num_envs, device=self.env.device),
                asset_cfg=trunk_resolved,
                ranges=ranges,
                operation="add",
            )

        def _execute_step(self) -> bool:
            self._force_com()  # add=baseline+offset, so re-applying every step does not accumulate
            return super()._execute_step()

    play_scene = ev._create_play_scene()
    viewer = ComProbeViewer(
        env=ev.env,
        play_scene=play_scene,
        policy=ev.policy,
        port=args.port,
    )
    viewer.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
