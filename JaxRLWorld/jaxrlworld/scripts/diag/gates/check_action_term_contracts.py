"""Do action terms keep their contracts across entities and simulators?

CPU only, on the real :class:`ActionManagerBase` and joint action terms with
the engine writes stubbed:

1. cross-sim eval: :func:`_check_term_action_parity` accepts the same terms
   (also with the eval simulator's joint order permuted) and rejects a
   changed scale, offset, clip, residual alpha, term type, entity, term
   order, joint set, or a checkpoint without terms when the env has them;
   the manager-level check alone passes a changed term scale (why the
   per-term check exists);
2. dispatch: with explicit actuators on the driven robot, a second
   entity's joint target is written as a position target (its own
   simulator PD), never through the robot's explicit torque path;
3. PD-gain DR refuses an entity other than the driven robot instead of
   randomizing the robot's gains.

    python -m jaxrlworld.scripts.diag.gates.check_action_term_contracts
"""

from __future__ import annotations

import copy
import sys
from types import SimpleNamespace

import torch

from jaxrlworld.rl.configs.scene import SceneEntitySelector
from jaxrlworld.rl.envs.managers.common.action import ActionManagerBase, ActionManagerBaseConfig
from jaxrlworld.rl.envs.mdp.actions.joint_actions import (
    JointPositionAction,
    JointPositionActionCfg,
    MotionResidualJointPositionAction,
    MotionResidualJointPositionActionCfg,
)
from jaxrlworld.rl.envs.mdp.events.dr.unified import randomize_pd_gains
from jaxrlworld.rl.envs.mdp.events.keyed_sampler import STAGE_RESET_DR, TermSampler
from jaxrlworld.rl.evals.evaluator import _check_resolved_action_parity, _check_term_action_parity

JOINTS = ["j0", "j1", "j2"]


class _Manager(ActionManagerBase):
    def _build_indexing(self):
        return self.env.entity_indexing(self.env.robot_entity_name)

    def _build_actuators_from_entity(self) -> None:
        pass


def _env(joint_names: list[str]):
    indexing = SimpleNamespace(joint_names=list(joint_names), sim_indices=torch.arange(len(joint_names)))
    indexing.num_joints = len(joint_names)
    return SimpleNamespace(
        device="cpu",
        num_envs=1,
        robot_entity_name="robot",
        entity_indexing=lambda name: indexing,
        scene_manager=SimpleNamespace(config=SimpleNamespace(entities={})),
        _resolve_default_joint_pos=lambda name: torch.zeros(len(joint_names)),
    )


def _manager(joint_names=JOINTS, **overrides) -> _Manager:
    arm = dict(joint_names=["j0", "j1"], scale={"j0": 0.25, "j1": 0.5}, offset={"j0": 0.1, "j1": -0.2}, clip=None)
    arm.update(overrides.get("arm", {}))
    residual = dict(joint_names=["j2"], alpha={"j2": 0.3})
    residual.update(overrides.get("residual", {}))
    terms = {
        "arm": JointPositionActionCfg(class_type=JointPositionAction, **arm),
        "residual": MotionResidualJointPositionActionCfg(class_type=MotionResidualJointPositionAction, **residual),
    }
    if overrides.get("arm_type") is not None:
        terms["arm"] = overrides["arm_type"](**arm)
    if overrides.get("reverse"):
        terms = dict(reversed(list(terms.items())))
    return _Manager(_env(joint_names), ActionManagerBaseConfig(action_terms=terms))


def _metadata(m: _Manager) -> dict:
    return {
        "canonical_joint_names": list(m.actuated_joint_names),
        "action_scale": m.scale.tolist(),
        "action_offset": m.offset[0].tolist(),
        "action_clip_low": m.clip_low.tolist(),
        "action_clip_high": m.clip_high.tolist(),
        "action_terms": copy.deepcopy(m.resolved_term_mappings()),
    }


def _rejects(fn) -> tuple[bool, str]:
    try:
        fn()
    except (ValueError, NotImplementedError) as e:  # except-ok: the refusal under test
        return True, str(e).splitlines()[-1].strip()[:110]
    return False, "accepted"


def main() -> int:
    failures: list[str] = []

    def chk(name: str, ok: bool, detail: str = "") -> None:
        print(f"  [{'OK' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
        if not ok:
            failures.append(name)

    print("-- 1. cross-sim term parity")
    saved = _metadata(_manager())

    def same(m):
        _check_term_action_parity(saved, m, "train", "eval")
        return True

    chk("identical terms accepted", same(_manager()))
    chk("eval joint order permuted, same values per joint: accepted", same(_manager(joint_names=["j2", "j0", "j1"])))
    for label, kwargs in (
        ("changed arm scale", {"arm": {"scale": {"j0": 0.25, "j1": 1.0}}}),
        ("changed arm offset", {"arm": {"offset": {"j0": 0.1, "j1": 0.0}}}),
        ("changed arm clip", {"arm": {"clip": (-1.0, 1.0)}}),
        ("changed residual alpha", {"residual": {"alpha": {"j2": 0.5}}}),
        ("changed term order", {"reverse": True}),
        ("changed joint set", {"arm": {"joint_names": ["j0"], "scale": {"j0": 0.25}, "offset": {"j0": 0.1}}}),
        (
            "changed term type",
            {
                "arm_type": lambda **kw: MotionResidualJointPositionActionCfg(
                    class_type=MotionResidualJointPositionAction, joint_names=kw["joint_names"]
                )
            },
        ),
    ):
        try:
            m = _manager(**kwargs)
        except ValueError as e:  # except-ok: a construction the manager refuses, reported as a failed check
            chk(f"{label}: rejected", False, f"could not build: {e}")
            continue
        ok, detail = _rejects(lambda m=m: _check_term_action_parity(saved, m, "train", "eval"))
        chk(f"{label}: rejected", ok, detail)
    other_entity = copy.deepcopy(saved)
    other_entity["action_terms"]["arm"]["entity"] = "robot2"
    ok, detail = _rejects(lambda: _check_term_action_parity(other_entity, _manager(), "train", "eval"))
    chk("changed entity: rejected", ok, detail)
    no_terms = {k: v for k, v in saved.items() if k != "action_terms"}
    print("  (a checkpoint without action_terms warns:)")
    _check_term_action_parity(no_terms, _manager(), "train", "eval")
    changed = _manager(arm={"scale": {"j0": 0.25, "j1": 1.0}})
    manager_blind = not _rejects(lambda: _check_resolved_action_parity(saved, changed, "train", "eval"))[0]
    chk("the manager-level check alone does not see a term scale (why the term check exists)", manager_blind)

    print("-- 2. dispatch to a second, simulator-PD entity")
    m = _manager()
    writes: list[tuple[str, str]] = []
    m._has_explicit_actuators = True
    m._compute_actuator_torques = lambda target, name: torch.full_like(target, 123.0)
    m._apply_force = lambda value, name: writes.append(("force", name))
    m._apply_position = lambda value, name: writes.append(("position", name))
    m._entity_joint_target = {"robot": torch.ones(1, 3), "second_robot": torch.ones(1, 2)}
    m._pending_target_entities = ["robot", "second_robot"]
    m._flush_entity_commands()
    chk("driven robot: explicit torque path", ("force", "robot") in writes, str(writes))
    chk("second entity: position target", ("position", "second_robot") in writes, str(writes))
    chk("second entity: no torque write", ("force", "second_robot") not in writes, str(writes))

    print("-- 3. PD-gain DR entity scope")
    actuator = SimpleNamespace(stiffness=torch.ones(1, 2), damping=torch.zeros(1, 2))
    env = SimpleNamespace(
        device="cpu",
        num_envs=1,
        robot_entity_name="robot",
        act_manager=SimpleNamespace(has_explicit_actuators=True, actuators=[(actuator, torch.arange(2))]),
    )
    sampler = TermSampler.bind(
        base_seed=1, term_name="pd", stage=STAGE_RESET_DR, counter=torch.zeros(1, dtype=torch.long)
    )
    ok, detail = _rejects(
        lambda: randomize_pd_gains(
            env, torch.tensor([0]), sampler, asset_cfg=SceneEntitySelector(name="second_robot"), kp_range=(2.0, 2.0)
        )
    )
    chk("another entity's gains: refused", ok, detail)
    chk("... and the robot's gains untouched", torch.equal(actuator.stiffness, torch.ones(1, 2)))

    print("=" * 78)
    print("=== RESULT: ALL OK ===" if not failures else f"=== RESULT: {len(failures)} FAILED ===")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
