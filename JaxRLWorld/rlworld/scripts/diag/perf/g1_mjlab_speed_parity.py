"""Speed parity rig: our G1 flat mujoco preset with Mjlab's exact physics.

``uv run train Mjlab-Velocity-Flat-Unitree-G1`` and our g1_29dof mujoco
preset differ in four physics-cost knobs, all of which make OUR runs
slower per step:

  ==================  =============  ====================
  knob                ours           Mjlab flat G1
  ==================  =============  ====================
  solver iterations   50             10
  solver ls_iters     50             20
  njmax               1500           300
  nconmax             100            None (auto)
  actuator            DelayedPD      builtin position
                      (torch, per    (PD inside mjwarp,
                      substep,       no delay)
                      delay 0-2)
  ==================  =============  ====================

  Identical already: asset (mjlab zoo G1 spec), timestep 0.005,
  decimation 4, newton solver, implicitfast, pyramidal cone,
  impratio 1, ccd_iterations 50, contact_sensor_maxmatch 64,
  num_steps_per_env 24, plane terrain, feet+self-collision sensors.

This rig rebuilds our preset with Mjlab's values (and the actuator as
``ImplicitActuatorCfg`` — the same builtin position path, our gains) and
runs the normal training loop so the printed Throughput is directly
comparable to rsl_rl's ``fps`` (both are num_steps * num_envs /
(collect + learn)).

Usage:
    jaxpy -m rlworld.scripts.diag.perf.g1_mjlab_speed_parity --num-envs 16384 --iterations 60
"""

from __future__ import annotations

import argparse

from rlworld.rl.actuators.actuator_cfg import ImplicitActuatorCfg
from rlworld.rl.configs.presets.g1_29dof.base import G1FlatConfig
from rlworld.rl.configs.scene.unified_entity_config import ArticulationCfg
from rlworld.rl.runners import BaseRunner


class G1FlatMjlabSpeedParityConfig(G1FlatConfig):
    """G1 flat mujoco with Mjlab-Velocity-Flat-Unitree-G1's physics knobs."""

    def build(self):
        from mjlab.sim.sim import MujocoCfg, SimulationCfg

        cfgs = super().build()
        sc = cfgs.scene
        # Verbatim Mjlab-Velocity-Flat-Unitree-G1 SimulationCfg. Passing it
        # whole (scene manager uses it as-is) also removes the
        # disableflags=("nativeccd",) our scene manager otherwise sets —
        # mjlab's train runs with native CCD enabled.
        sc.mjlab_sim_cfg = SimulationCfg(
            nconmax=None,
            njmax=300,
            mujoco=MujocoCfg(
                timestep=0.005,
                iterations=10,
                ls_iterations=20,
            ),
            contact_sensor_maxmatch=64,
        )

        # Match mjlab's self_collision sensor shape: one subtree-vs-subtree
        # pair (2 mj sensors). Our preset's per-link body match expands to
        # 31 links x 2 fields = 62 mj sensors, all evaluated inside the
        # mjwarp step graph every substep.
        from rlworld.rl.configs.sensors import ContactMatch, ContactSensorCfg

        sensors = []
        for s in sc.sensors:
            if s.name == "self_collision":
                s = ContactSensorCfg(
                    name="self_collision",
                    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
                    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
                    fields=("found", "force"),
                    reduce="none",
                    num_slots=1,
                    history_length=s.history_length,
                )
            sensors.append(s)
        sc.sensors = tuple(sensors)

        robot = sc.entities["robot"]
        old = robot.articulation.actuators[0]
        robot.articulation = ArticulationCfg(
            actuators=(
                ImplicitActuatorCfg(
                    target_names_expr=old.target_names_expr,
                    stiffness=old.stiffness,
                    damping=old.damping,
                    effort_limit=old.effort_limit,
                    armature=old.armature,
                    frictionloss=old.frictionloss,
                ),
            ),
        )
        return cfgs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", type=int, default=16384)
    ap.add_argument("--iterations", type=int, default=60)
    args = ap.parse_args()

    cfg = G1FlatMjlabSpeedParityConfig(sim_type="mujoco", num_envs=args.num_envs)
    cfgs = cfg.build().with_cli_overrides()
    runner = BaseRunner.create_with_env(cfgs, use_wandb=False)
    runner.learn(
        num_learning_iterations=args.iterations,
        init_at_random_ep_len=cfgs.runner.init_at_random_ep_len,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
