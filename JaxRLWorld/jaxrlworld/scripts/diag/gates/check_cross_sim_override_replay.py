"""Gate: cross-sim evaluation keeps the training run's overrides.

Same-sim evaluation rebuilds the preset and restores every saved
difference (``load_config_from_checkpoint``). Cross-sim evaluation built
the target simulator's preset from class and constructor kwargs alone and
copied only ``algorithm`` and ``nn`` from the checkpoint, so a reward
weight, command range, event or episode length changed on the training
command line was silently replaced by the target preset's default. The
evaluator now replays the saved run's simulator-independent overrides
onto the target (``checkpoint.replay_training_overrides``).

No simulator is needed: a unified preset defined here builds a Newton or
a MuJoCo ``ConfigsForRun`` with the same task terms and per-simulator
observation functions and scene.

1. a run trained on one simulator with overrides on reward, command,
   event, env and runner, saved through the checkpoint's YAML path and
   resolved for the other simulator, carries every one of them after
   the replay, keeps the target's scene and observation functions, and
   reports the scene override it could not carry;
2. ``PolicyEvaluator._prepare_cross_sim_configs`` applies the replay
   before the caller's own overrides, which still win;
3. a run without overrides replays nothing;
4. a saved term the target has no name for raises, and a term the
   preset gained since training raises.

Run::

    python -m jaxrlworld.scripts.diag.gates.check_cross_sim_override_replay
"""

from __future__ import annotations

import copy
import os
import tempfile
from dataclasses import dataclass
from types import SimpleNamespace

from jaxrlworld.rl.configs.algorithms import get_algorithm_config_class
from jaxrlworld.rl.configs.base_config import iter_terms
from jaxrlworld.rl.configs.common_config_classes import EventConfig, ObservationGroupConfig, RewardConfig
from jaxrlworld.rl.configs.events.event_term_config import EventTermConfig
from jaxrlworld.rl.configs.mujoco_config_classes import MujocoConfigsForRun
from jaxrlworld.rl.configs.newton_config_classes import NewtonConfigsForRun
from jaxrlworld.rl.configs.observations.observation_term_config import ObservationTermConfig
from jaxrlworld.rl.configs.rewards.reward_term_config import RewardTermConfig
from jaxrlworld.rl.evals.evaluator import PolicyEvaluator
from jaxrlworld.rl.evals.sim_initializers import resolve_cross_sim_config
from jaxrlworld.rl.utils.checkpoint import EVALUATOR_OWNED_SECTIONS, SIM_BOUND_SECTIONS, replay_training_overrides
from jaxrlworld.rl.utils.resolve import resolve_callable
from jaxrlworld.rl.utils.yaml_io import dump_yaml, load_yaml

_CONFIGS_FOR_RUN = {"newton": NewtonConfigsForRun, "mujoco": MujocoConfigsForRun}


@dataclass
class _VelocityCommand:
    lin_vel_x: tuple[float, float] = (-1.0, 1.0)
    resampling_time_s: float = 10.0


@dataclass
class GatePreset:
    """A unified preset: one class, ``sim_type`` selects the backend."""

    sim_type: str = "newton"
    num_envs: int = 64

    def build(self):
        # Term classes are defined per build, as presets do, so every build
        # holds its own term objects (a class-level term is shared by all
        # instances and an override on one would show on the "fresh" one).
        @dataclass
        class _Rewards(RewardConfig):
            track = RewardTermConfig(func=resolve_callable, weight=2.0, params={"std": 0.5})
            upright = RewardTermConfig(func=resolve_callable, weight=-1.0)
            smooth = RewardTermConfig(func=resolve_callable, weight=-0.1)

        @dataclass
        class _Events(EventConfig):
            reset_root = EventTermConfig(func=resolve_callable, mode="reset")
            mass_dr = EventTermConfig(func=resolve_callable, mode="reset_dr")
            push_robot = EventTermConfig(func=resolve_callable, mode="interval", interval_range_s=(2.0, 20.0))

        cfgs = _CONFIGS_FOR_RUN[self.sim_type](algorithm=get_algorithm_config_class("PPO")())
        cfgs.env.num_envs = self.num_envs
        cfgs.env.task_name = "gate"
        cfgs.reward = _Rewards()
        cfgs.event = _Events()
        cfgs.command.terms = {"velocity": _VelocityCommand()}
        actor = ObservationGroupConfig()
        actor.ang_vel = ObservationTermConfig(func=f"jaxrlworld.rl.envs.mdp.observations.{self.sim_type}:base_ang_vel")
        actor.dof_pos = ObservationTermConfig(func="jaxrlworld.rl.envs.mdp.observations.common:dof_pos", scale=1.0)
        cfgs.observation.actor = actor
        cfgs.preset_module = type(self).__module__
        cfgs.preset_class_name = type(self).__name__
        cfgs.preset_kwargs = {"sim_type": self.sim_type, "num_envs": self.num_envs}
        return cfgs


def yaml_round_trip(cfgs) -> dict:
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "config.yaml")
        dump_yaml(path, cfgs.recursive_to_dict())
        return load_yaml(path)


def trained_config():
    """A Newton run with command-line style and programmatic overrides."""
    train = GatePreset(sim_type="newton", num_envs=64).build()
    train.apply_overrides(
        reward={"track": {"weight": 3.5, "params": {"std": 0.9}}, "upright": None},
        command={"terms": {"velocity": {"lin_vel_x": [-2.0, 2.0]}}},
        event={"push_robot": None},
        env={"episode_length_s": 7.0},
        algorithm={"gamma": 0.93},
    )
    train.runner.save_interval = 123
    train.scene.rigid_gap = 0.0  # Newton-only plant setting
    return train


def main() -> int:
    failures: list[str] = []

    def chk(name: str, ok: bool, detail: str = "") -> None:
        print(f"  [{'OK' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
        if not ok:
            failures.append(name)

    print("=" * 78)
    print("CROSS-SIM OVERRIDE REPLAY")
    print("=" * 78)
    chk(
        "simulator-bound sections (kept from the target, reported)",
        SIM_BOUND_SECTIONS == {"scene", "observation", "action"},
        str(sorted(SIM_BOUND_SECTIONS)),
    )
    chk(
        "evaluator-owned sections (copied or set by the evaluator, not reported)",
        EVALUATOR_OWNED_SECTIONS == {"algorithm", "nn", "visualization"},
        str(sorted(EVALUATOR_OWNED_SECTIONS)),
    )

    print("\n-- 1. replay onto the other simulator's preset --")
    saved = yaml_round_trip(trained_config())
    metadata = {"config": saved}
    target = resolve_cross_sim_config(metadata, "mujoco")
    chk("target resolves to the MuJoCo config class", isinstance(target, MujocoConfigsForRun), type(target).__name__)
    chk("constructor kwargs carried (num_envs)", target.env.num_envs == 64, str(target.env.num_envs))
    chk("before the replay the target holds preset defaults", target.reward.track.weight == 2.0)
    fresh_scene = target.scene.recursive_to_dict()

    cfgs = copy.deepcopy(target)
    applied, skipped = replay_training_overrides(cfgs, saved)
    print(f"  applied: {applied}")
    print(f"  skipped: {skipped}")
    rewards = iter_terms(cfgs.reward, RewardTermConfig)
    chk(
        "reward weight and params replayed",
        rewards["track"].weight == 3.5 and rewards["track"].params == {"std": 0.9},
        str(rewards["track"]),
    )
    chk("reward term disabled in training stays disabled", "upright" not in rewards, str(sorted(rewards)))
    chk("replayed term is still a RewardTermConfig", isinstance(cfgs.reward.track, RewardTermConfig))
    velocity = cfgs.command.terms["velocity"]
    chk(
        "command range replayed onto the term object",
        isinstance(velocity, _VelocityCommand) and tuple(velocity.lin_vel_x) == (-2.0, 2.0),
        str(velocity),
    )
    events = iter_terms(cfgs.event, EventTermConfig)
    chk("event term disabled in training stays disabled", "push_robot" not in events, str(sorted(events)))
    chk("env.episode_length_s replayed", cfgs.env.episode_length_s == 7.0, str(cfgs.env.episode_length_s))
    chk("runner.save_interval replayed", cfgs.runner.save_interval == 123, str(cfgs.runner.save_interval))
    chk(
        "algorithm is neither replayed nor reported (copied whole by the evaluator)",
        "algorithm.gamma" not in applied and "algorithm.gamma" not in skipped and cfgs.algorithm.gamma != 0.93,
    )
    chk("target scene untouched", cfgs.scene.recursive_to_dict() == fresh_scene)
    chk("the Newton-only scene override is reported as skipped", skipped == ["scene.rigid_gap"], str(skipped))
    chk(
        "observation functions stay the target simulator's",
        "observations.mujoco:" in cfgs.observation.actor.ang_vel.func,
        cfgs.observation.actor.ang_vel.func,
    )
    expected = {
        "command.terms.velocity.lin_vel_x",
        "env.episode_length_s",
        "event.push_robot",
        "reward.track.params.std",
        "reward.track.weight",
        "reward.upright",
        "runner.save_interval",
    }
    chk("applied paths are exactly the training overrides", set(applied) == expected, str(sorted(applied)))

    print("\n-- 2. the evaluator's cross-sim path --")
    evaluator = SimpleNamespace(_train_sim_type="Newton", sim_type="MujocoEnv")
    # A fresh target: the evaluator works on a deep copy, but term objects
    # are class attributes of the build's term classes and an override of
    # one shows on every config of that build, so only instance fields
    # (env.*) can show whether the passed-in object was left alone.
    target = resolve_cross_sim_config(metadata, "mujoco")
    prepared = PolicyEvaluator._prepare_cross_sim_configs(
        evaluator,
        eval_cfgs=target,
        metadata=metadata,
        extra_overrides={"env": {"num_envs": 1}, "reward": {"smooth": {"weight": -0.5}}},
        record_video=False,
        video_dir=None,
    )
    chk("algorithm copied from the checkpoint", prepared.algorithm.gamma == 0.93, str(prepared.algorithm.gamma))
    chk("training overrides replayed", prepared.reward.track.weight == 3.5 and prepared.env.episode_length_s == 7.0)
    chk(
        "the caller's overrides are applied after the replay and win",
        prepared.env.num_envs == 1 and prepared.reward.smooth.weight == -0.5,
        f"num_envs={prepared.env.num_envs} smooth={prepared.reward.smooth.weight}",
    )
    chk("target scene kept", prepared.scene.recursive_to_dict() == fresh_scene)
    chk(
        "the passed-in target's instance fields are not mutated",
        target.env.num_envs == 64 and target.env.episode_length_s == 20.0,
        f"num_envs={target.env.num_envs} episode_length_s={target.env.episode_length_s}",
    )

    print("\n-- 3. a run without overrides replays nothing --")
    plain_saved = yaml_round_trip(GatePreset(sim_type="newton", num_envs=64).build())
    plain_target = resolve_cross_sim_config({"config": plain_saved}, "mujoco")
    before = plain_target.recursive_to_dict()
    applied, skipped = replay_training_overrides(plain_target, plain_saved)
    chk("nothing applied, nothing skipped", applied == [] and skipped == [], f"{applied} {skipped}")
    chk("config unchanged", plain_target.recursive_to_dict() == before)

    print("\n-- 4. refusals --")
    ghost = copy.deepcopy(saved)
    ghost["reward"]["ghost"] = {"func": "x:y", "weight": 1.0, "params": {}}
    try:
        replay_training_overrides(copy.deepcopy(target), ghost)
        chk("a saved term the target lacks raises", False, "no exception")
    except ValueError as e:
        chk("a saved term the target lacks raises", "reward.ghost" in str(e), str(e)[:90])
    grown = copy.deepcopy(saved)
    del grown["reward"]["smooth"]
    try:
        replay_training_overrides(copy.deepcopy(target), grown)
        chk("a term the preset gained since training raises", False, "no exception")
    except ValueError as e:
        chk("a term the preset gained since training raises", "reward.smooth" in str(e), str(e)[:90])

    print("=" * 78)
    print("=== RESULT: ALL OK ===" if not failures else f"=== RESULT: {len(failures)} FAILED ===")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
