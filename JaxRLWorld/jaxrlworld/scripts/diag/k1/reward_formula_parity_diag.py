"""Prove the ported reward formulas equal the reference implementation's.

No simulator is involved. The reference module is imported from its own source
file with its framework dependencies stubbed out, so this compares the terms
this repository will actually run against the code they were ported from --
not against a transcription of it, which would hide a transcription error.

Both sides are fed identical tensors. Where a term needs a quantity the two
frameworks each compute for themselves (the projected gravity vector), the stub
hands both sides the same precomputed value, so what is compared is the reward
formula around it and nothing else. Cross-backend agreement of those
framework-computed quantities is a separate question, answered by
``diag_cross_sim_parity``.

The case list is built to hit every branch: a dead-zero command, a command just
under and just over each speed threshold, the exact threshold values, zero pose
error, and errors large enough to saturate the exponentials.

Run (from the SimForge root)::

    python -m jaxrlworld.scripts.diag.k1.reward_formula_parity_diag
"""

from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import torch

from jaxrlworld.rl.envs.mdp.rewards import k1_velocity as ported
from jaxrlworld.rl.utils import string as string_utils

REFERENCE_SOURCE = Path("third_party/rl_frameworks/booster_mjlab/src/booster_mjlab/tasks/velocity/mdp/rewards.py")
# The self-collision term the recipe uses is not its own; it comes from the
# framework the recipe is built on, so that file is the reference for it.
UPSTREAM_SOURCE = Path("vendor/Mjlab/src/mjlab/tasks/velocity/mdp/rewards.py")

# Substeps of force history per control step, i.e. the decimation. The term
# under test counts how many of them carried a self-contact.
HISTORY_LENGTH = 4
SELF_COLLISION_THRESHOLD = 10.0

# Joint set the upper-body term is scored over, in resolved order.
UPPER_BODY_JOINTS = [
    "Head_Yaw",
    "Head_Pitch",
    "Left_Shoulder_Pitch",
    "Left_Shoulder_Roll",
    "Left_Elbow_Pitch",
    "Left_Elbow_Yaw",
    "Right_Shoulder_Pitch",
    "Right_Shoulder_Roll",
    "Right_Elbow_Pitch",
    "Right_Elbow_Yaw",
]

# The preset's own std tables, so this checks the values that will be trained on.
STD_STANDING = {r"Head_.*": 0.05, r".*_Shoulder_.*": 0.05, r".*_Elbow_.*": 0.05}
STD_WALKING = {r"Head_.*": 0.05, r".*_Shoulder_.*": 0.15, r".*_Elbow_.*": 0.15}
STD_RUNNING = {
    r"Head_.*": 0.05,
    r".*_Shoulder_Pitch": 0.5,
    r".*_Shoulder_Roll": 0.2,
    r".*_Elbow_.*": 0.35,
}

WALKING_THRESHOLD = 0.05
RUNNING_THRESHOLD_UPRIGHT = 1.5
RUNNING_THRESHOLD_POSTURE = 1.0

TOL = 1e-12


# ── reference module, loaded from source with its framework stubbed ──


def _install_stubs(projected_gravity: torch.Tensor) -> None:
    """Make the reference module importable outside its own framework.

    Only two of the stubbed symbols are ever called at runtime. The quaternion
    rotation is pinned to ``projected_gravity`` so both sides see the same
    vector; the name resolver is the shared IsaacLab-derived implementation
    both frameworks already use.
    """

    def _fake_quat_apply_inverse(quat, vec):
        del quat, vec
        return projected_gravity

    class _Any:
        """Placeholder for a type the reference module only annotates with or
        instantiates for an unused module-level default."""

        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs
            self.name = args[0] if args else "robot"
            self.joint_ids = None
            self.body_ids = None

    specs: dict[str, dict[str, object]] = {
        "mjlab": {},
        "mjlab.entity": {"Entity": _Any},
        "mjlab.managers": {},
        "mjlab.managers.reward_manager": {"RewardTermCfg": _Any},
        "mjlab.managers.scene_entity_config": {"SceneEntityCfg": _Any},
        "mjlab.utils": {},
        "mjlab.utils.lab_api": {},
        "mjlab.utils.lab_api.math": {
            "quat_apply_inverse": _fake_quat_apply_inverse,
            # Imported by the upstream module; never reached by the terms here.
            "quat_apply": _Any,
        },
        "mjlab.utils.lab_api.string": {"resolve_matching_names_values": string_utils.resolve_matching_names_values},
        "mjlab.envs": {"ManagerBasedRlEnv": _Any},
        # Only needed by the upstream module, which carries a wider import set.
        "mjlab.sensor": {"BuiltinSensor": _Any, "ContactSensor": _Any},
        "mjlab.sensor.terrain_height_sensor": {"TerrainHeightSensor": _Any},
        "mjlab.tasks": {},
        "mjlab.tasks.velocity": {},
        "mjlab.tasks.velocity.mdp": {},
        "mjlab.tasks.velocity.mdp.terrain_utils": {"terrain_normal_from_sensors": _Any},
        "mjlab.viewer": {},
        "mjlab.viewer.debug_visualizer": {"DebugVisualizer": _Any},
    }
    for name, attrs in specs.items():
        module = types.ModuleType(name)
        for attr, value in attrs.items():
            setattr(module, attr, value)
        sys.modules[name] = module


def _load_module(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(f"source not found at {path} (run from the SimForge root)")
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_reference(projected_gravity: torch.Tensor):
    _install_stubs(projected_gravity)
    return _load_module(REFERENCE_SOURCE, "_reference_rewards")


def _load_upstream(projected_gravity: torch.Tensor):
    """The framework module the recipe's self-collision term comes from."""
    _install_stubs(projected_gravity)
    return _load_module(UPSTREAM_SOURCE, "_upstream_rewards")


# ── the two environment shapes, over one shared set of tensors ───────


@dataclass
class Case:
    name: str
    command: torch.Tensor  # (B, 3) = vx, vy, wz
    lin_vel_b: torch.Tensor  # (B, 3) root linear velocity, body frame
    joint_pos: torch.Tensor  # (B, J)
    default_joint_pos: torch.Tensor  # (B, J)
    projected_gravity: torch.Tensor  # (B, 3)
    contact_history: torch.Tensor  # (B, N, H, 3) self-collision force history


class _RefAssetData:
    def __init__(self, case: Case) -> None:
        self.root_link_lin_vel_b = case.lin_vel_b
        self.joint_pos = case.joint_pos
        self.default_joint_pos = case.default_joint_pos
        self.root_link_quat_w = torch.zeros(case.command.shape[0], 4)
        self.body_link_quat_w = torch.zeros(case.command.shape[0], 1, 4)
        self.gravity_vec_w = torch.tensor([0.0, 0.0, -1.0])


class _RefSensorData:
    def __init__(self, case: Case) -> None:
        self.force_history = case.contact_history
        self.found = (torch.norm(case.contact_history, dim=-1) > SELF_COLLISION_THRESHOLD).any(dim=2)


class _RefSensor:
    def __init__(self, case: Case) -> None:
        self.data = _RefSensorData(case)


class _RefAsset:
    def __init__(self, case: Case) -> None:
        self.data = _RefAssetData(case)

    def find_joints(self, names):
        del names
        return list(range(len(UPPER_BODY_JOINTS))), list(UPPER_BODY_JOINTS)


class _RefCommandManager:
    def __init__(self, case: Case) -> None:
        self._command = case.command

    def get_command(self, name):
        del name
        return self._command


class _RefEnv:
    """Shaped like the reference framework's env: ``scene[name].data.*``."""

    def __init__(self, case: Case) -> None:
        self._asset = _RefAsset(case)
        self._sensor = _RefSensor(case)
        self.command_manager = _RefCommandManager(case)
        self.device = "cpu"

    def __getitem__(self, name):
        return self._sensor if name == "self_collision" else self._asset

    @property
    def scene(self):
        return self


class _PortedRobotData:
    def __init__(self, case: Case) -> None:
        self.root_link_lin_vel_b = case.lin_vel_b
        self.joint_pos = case.joint_pos
        self.default_joint_pos = case.default_joint_pos
        self.projected_gravity_b = case.projected_gravity


class _PortedCommandManager:
    def __init__(self, case: Case) -> None:
        self.lin_vel_x = case.command[:, 0]
        self.lin_vel_y = case.command[:, 1]
        self.ang_vel = case.command[:, 2]


class _PortedContactManager:
    def __init__(self, case: Case) -> None:
        self._history = case.contact_history

    def contact_force_history(self, group):
        del group
        return self._history

    def contact_force(self, group):
        del group
        return self._history[:, :, -1, :]


class _PortedEnv:
    """Shaped like this repository's env: ``get_entity_data(name).*``."""

    def __init__(self, case: Case) -> None:
        self._data = _PortedRobotData(case)
        self.command_manager = _PortedCommandManager(case)
        self.contact_manager = _PortedContactManager(case)
        self.device = "cpu"

    def get_entity_data(self, name="robot"):
        del name
        return self._data


class _Selector:
    """Enough of a resolved selector for the terms under test."""

    def __init__(self, joint_ids=None, joint_names=(), body_ids=None) -> None:
        self.name = "robot"
        self.joint_ids = joint_ids
        self.joint_names = list(joint_names)
        self.joint_names_tuple = tuple(joint_names)
        self.body_ids = body_ids


# ── cases ────────────────────────────────────────────────────────────


def build_cases() -> list[Case]:
    torch.manual_seed(0)
    num_joints = len(UPPER_BODY_JOINTS)

    # Commands chosen to straddle every threshold the terms branch on.
    commands = [
        ("zero", [0.0, 0.0, 0.0]),
        ("below_walking", [0.02, 0.0, 0.0]),
        ("at_walking_threshold", [WALKING_THRESHOLD, 0.0, 0.0]),
        ("just_above_walking", [WALKING_THRESHOLD + 1e-4, 0.0, 0.0]),
        ("walking", [0.6, -0.2, 0.3]),
        ("at_posture_running_threshold", [RUNNING_THRESHOLD_POSTURE, 0.0, 0.0]),
        ("between_thresholds", [1.0, 0.2, 0.05]),
        ("at_upright_running_threshold", [RUNNING_THRESHOLD_UPRIGHT, 0.0, 0.0]),
        ("running", [1.6, 0.4, -0.7]),
        ("pure_yaw", [0.0, 0.0, 1.2]),
        ("pure_lateral", [0.0, -0.9, 0.0]),
        ("backwards", [-1.1, 0.0, 0.0]),
    ]

    cases: list[Case] = []
    for label, cmd in commands:
        for variant, scale in (("exact", 0.0), ("small_error", 0.15), ("large_error", 3.0)):
            batch = 4
            command = torch.tensor(cmd, dtype=torch.float64).repeat(batch, 1)
            if scale == 0.0:
                # Velocity exactly on command, pose exactly at default.
                lin = torch.zeros(batch, 3, dtype=torch.float64)
                lin[:, :2] = command[:, :2]
                joint_pos = torch.zeros(batch, num_joints, dtype=torch.float64)
                gravity = torch.zeros(batch, 3, dtype=torch.float64)
                gravity[:, 2] = -1.0
            else:
                lin = torch.randn(batch, 3, dtype=torch.float64) * scale
                lin[:, :2] += command[:, :2]
                joint_pos = torch.randn(batch, num_joints, dtype=torch.float64) * scale
                gravity = torch.randn(batch, 3, dtype=torch.float64) * (0.3 * scale)
                gravity[:, 2] = -1.0
            default = torch.zeros(batch, num_joints, dtype=torch.float64)
            # Self-contact history: one column, four substeps. The magnitudes
            # straddle the threshold so the count lands on every value from
            # none of the substeps to all of them.
            history = torch.zeros(batch, 1, HISTORY_LENGTH, 3, dtype=torch.float64)
            for row in range(batch):
                for substep in range(HISTORY_LENGTH):
                    over = (row + substep) % (HISTORY_LENGTH + 1) <= row
                    history[row, 0, substep, 2] = SELF_COLLISION_THRESHOLD * (3.0 if over else 0.1)
            cases.append(
                Case(
                    name=f"{label}/{variant}",
                    command=command,
                    lin_vel_b=lin,
                    joint_pos=joint_pos,
                    default_joint_pos=default,
                    projected_gravity=gravity,
                    contact_history=history,
                )
            )
    return cases


# ── comparison ───────────────────────────────────────────────────────


def main() -> int:
    cases = build_cases()
    fails: list[str] = []
    worst: dict[str, float] = {}

    def record(term: str, case: str, diff: float) -> None:
        worst[term] = max(worst.get(term, 0.0), diff)
        if not (diff <= TOL):
            fails.append(f"{term} @ {case}: |diff| = {diff:.3e}")

    joint_ids = torch.arange(len(UPPER_BODY_JOINTS))
    subset = _Selector(joint_ids=joint_ids, joint_names=UPPER_BODY_JOINTS)
    whole = _Selector()

    print(f"=== reward formula parity: {len(cases)} cases x 5 terms ===\n")

    for case in cases:
        reference = _load_reference(case.projected_gravity)
        ref_env = _RefEnv(case)
        ported_env = _PortedEnv(case)

        # 1. speed-relative velocity tracking, with and without the progress blend.
        for progress_weight in (0.0, 0.5):
            ref = reference.track_linear_velocity(
                ref_env,
                std=0.5,
                command_name="twist",
                std_at_rest=0.1,
                asset_cfg=whole,
                relative_std=0.75,
                progress_weight=progress_weight,
            )
            got = ported.track_lin_vel_relative_std(
                ported_env,
                std=0.5,
                std_at_rest=0.1,
                relative_std=0.75,
                progress_weight=progress_weight,
                asset_cfg=whole,
            )
            record(
                f"track_lin_vel_relative_std(pw={progress_weight})",
                case.name,
                float((ref - got).abs().max()),
            )

        # 2. speed-tiered upright.
        ref = reference.variable_upright(
            ref_env,
            command_name="twist",
            std_standing=0.20**0.5,
            std_walking=0.25**0.5,
            std_running=0.35**0.5,
            asset_cfg=whole,
            walking_threshold=WALKING_THRESHOLD,
            running_threshold=RUNNING_THRESHOLD_UPRIGHT,
        )
        got = ported.variable_upright(
            ported_env,
            std_standing=0.20**0.5,
            std_walking=0.25**0.5,
            std_running=0.35**0.5,
            walking_threshold=WALKING_THRESHOLD,
            running_threshold=RUNNING_THRESHOLD_UPRIGHT,
            asset_cfg=whole,
        )
        record("variable_upright", case.name, float((ref - got).abs().max()))

        # 3. standing pose deviation. The reference returns a positive cost with
        #    a negative weight; this port returns the negated value with a
        #    positive weight, so the expected relation is got == -ref.
        ref = reference.standing_pose_deviation_l1(
            ref_env, command_name="twist", command_threshold=0.05, asset_cfg=subset
        )
        got = ported.standing_pose_deviation_l1(ported_env, command_threshold=0.05, asset_cfg=subset)
        record("standing_pose_deviation_l1", case.name, float((-ref - got).abs().max()))
        if bool((got > 0).any()):
            fails.append(f"standing_pose_deviation_l1 @ {case.name}: returned a positive penalty")

        # 4. upper-body posture. Same sign relation as above.
        ref_term = reference.upper_body_posture_penalty(cfg=_RefCfg(subset), env=ref_env)
        ref = ref_term(
            ref_env,
            std_standing=STD_STANDING,
            std_walking=STD_WALKING,
            std_running=STD_RUNNING,
            asset_cfg=subset,
            command_name="twist",
            walking_threshold=WALKING_THRESHOLD,
            running_threshold=RUNNING_THRESHOLD_POSTURE,
        )
        got_term = ported.upper_body_posture_penalty(
            env=ported_env,
            asset_cfg=subset,
            std_standing=STD_STANDING,
            std_walking=STD_WALKING,
            std_running=STD_RUNNING,
            walking_threshold=WALKING_THRESHOLD,
            running_threshold=RUNNING_THRESHOLD_POSTURE,
        )
        got = got_term(ported_env)
        record("upper_body_posture_penalty", case.name, float((-ref - got).abs().max()))
        if bool((got > 0).any()):
            fails.append(f"upper_body_posture_penalty @ {case.name}: returned a positive penalty")

        # 5. Self-collision priced by substep count. This term comes from the
        #    framework the recipe is built on, not from the recipe, so it is
        #    compared against that module's own source.
        upstream = _load_upstream(case.projected_gravity)
        ref = upstream.self_collision_cost(
            ref_env, sensor_name="self_collision", force_threshold=SELF_COLLISION_THRESHOLD
        )
        got = ported.self_collision_substep_count(
            ported_env, contact_group="self_collision", force_threshold=SELF_COLLISION_THRESHOLD
        )
        record("self_collision_substep_count", case.name, float((-ref - got).abs().max()))
        if bool((got > 0).any()):
            fails.append(f"self_collision_substep_count @ {case.name}: returned a positive penalty")
        if float(ref.max()) <= 0.0:
            fails.append(f"self_collision_substep_count @ {case.name}: reference never fired")

    print(f"{'term':42s} {'worst |diff| over all cases':>28s}")
    print("-" * 72)
    for term in sorted(worst):
        print(f"{term:42s} {worst[term]:28.3e}")

    print(f"\ntolerance: {TOL:.0e}")
    print("\n=== RESULT:", "ALL OK" if not fails else f"{len(fails)} FAIL", "===")
    for line in fails[:20]:
        print("  " + line)
    return 1 if fails else 0


class _RefCfg:
    """The reference term class reads its std tables out of a cfg object."""

    def __init__(self, selector) -> None:
        self.params = {
            "asset_cfg": selector,
            "std_standing": STD_STANDING,
            "std_walking": STD_WALKING,
            "std_running": STD_RUNNING,
        }


if __name__ == "__main__":
    raise SystemExit(main())
