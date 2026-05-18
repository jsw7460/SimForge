"""NN config YAML save/load roundtrip diag.

Exercises every typed NN-config shape we ship (PPO/SAC/TD3/FastTD3 ×
MLP/SpaceTimeTransformer actor + critic × OrthoInit/DefaultInit ×
representative Activation/StdType/DistributionType values) and verifies
that:

  1. ``NNConfig.recursive_to_dict()`` produces only YAML-safe values
     (no Enum / dataclass instances leaking through). A leak would show
     up downstream as ``!!python/object/apply:...`` tags that
     ``yaml.safe_load`` refuses to construct — the bug pattern that
     broke ``load_checkpoint_metadata`` after the strict-typed-NN-config
     migration.
  2. ``yaml.safe_dump → yaml.safe_load → NNConfig.from_dict`` reproduces
     the original policy / actor / critic *types* and field values.
  3. The framework's own ``dump_yaml`` / ``load_yaml`` (used by the
     runner to persist checkpoint configs) survives the same roundtrip.
  4. Each emitted YAML string contains no ``!!python/object`` tag.

Output: a plain-text report to
``JaxRLWorld/rlworld/scripts/diag/output/nn_config_yaml_roundtrip_<ts>.txt``,
with one line per case and a final PASS/FAIL count. Non-zero exit code
on any failure so CI can hook this in later.

Run:
    jaxpy JaxRLWorld/rlworld/scripts/diag/check_nn_config_yaml_roundtrip.py
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys
import traceback
from dataclasses import fields as dc_fields
from enum import Enum
from typing import Any

import yaml

from rlworld.rl.configs.common_config_classes import (
    Activation,
    DefaultInit,
    DistributionType,
    FastTD3PolicyConfig,
    MLPActorCfg,
    MLPCriticCfg,
    NNConfig,
    OrthoInit,
    PPOPolicyConfig,
    SACPolicyConfig,
    SpaceTimeTransformerActorCfg,
    SpaceTimeTransformerCriticCfg,
    StdType,
    TD3PolicyConfig,
)
from rlworld.rl.utils.yaml_io import dump_yaml, load_yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_OUTPUT_DIR = os.path.join(_HERE, "output")


# ── Case enumeration ───────────────────────────────────────────────


def _build_cases() -> list[tuple[str, NNConfig]]:
    """Mirror every preset / benchmark NN config we ship today plus a
    few coverage-driven variations (DefaultInit, transformer critic,
    each std_type / distribution_type value).
    """
    cases: list[tuple[str, NNConfig]] = []

    # 7 preset baselines.
    cases.append(
        (
            "g1_29dof (PPO MLP scalar)",
            NNConfig(
                policy=PPOPolicyConfig(
                    actor=MLPActorCfg(
                        hidden_dims=[512, 256, 128],
                        activation=Activation.ELU,
                        init=OrthoInit(output_gain=0.01),
                    ),
                    critic=MLPCriticCfg(
                        hidden_dims=[1024, 512, 256],
                        activation=Activation.ELU,
                        init=OrthoInit(output_gain=0.01),
                    ),
                    init_noise_std=1.0,
                    distribution_type=DistributionType.GAUSSIAN,
                    std_type=StdType.SCALAR,
                )
            ),
        )
    )
    cases.append(
        (
            "go2_flat (PPO MLP state_independent)",
            NNConfig(
                policy=PPOPolicyConfig(
                    actor=MLPActorCfg(hidden_dims=[256, 128, 64], activation=Activation.ELU),
                    critic=MLPCriticCfg(hidden_dims=[256, 128, 64], activation=Activation.ELU),
                    init_noise_std=1.0,
                    distribution_type=DistributionType.GAUSSIAN,
                    std_type=StdType.STATE_INDEPENDENT,
                )
            ),
        )
    )
    cases.append(
        (
            "t1_tracking (PPO MLP)",
            NNConfig(
                policy=PPOPolicyConfig(
                    actor=MLPActorCfg(
                        hidden_dims=[512, 256, 128],
                        activation=Activation.ELU,
                        init=OrthoInit(output_gain=1.0),
                    ),
                    critic=MLPCriticCfg(
                        hidden_dims=[512, 256, 128],
                        activation=Activation.ELU,
                        init=OrthoInit(output_gain=1.0),
                    ),
                    init_noise_std=1.0,
                    distribution_type=DistributionType.GAUSSIAN,
                    std_type=StdType.STATE_INDEPENDENT,
                )
            ),
        )
    )
    cases.append(
        (
            "t1_tracking_transformer (PPO Transformer)",
            NNConfig(
                policy=PPOPolicyConfig(
                    actor=SpaceTimeTransformerActorCfg(
                        tracked_body_names=["pelvis", "left_ankle", "right_ankle"],
                        future_offsets=[1, 2, 4, 8, 16],
                        embed_dim=128,
                        num_heads=4,
                        num_layers=3,
                        dim_feedforward=256,
                        bottleneck_dim=32,
                        decoder_hidden_dim=128,
                        decoder_activation=Activation.ELU,
                        pe_type="learned",
                        attention_mode="factorized",
                    ),
                    critic=SpaceTimeTransformerCriticCfg(
                        tracked_body_names=["pelvis", "left_ankle", "right_ankle"],
                        future_offsets=[1, 2, 4, 8, 16],
                        embed_dim=128,
                        num_heads=4,
                        num_layers=3,
                        dim_feedforward=256,
                        pe_type="learned",
                        attention_mode="factorized",
                    ),
                    init_noise_std=1.0,
                    distribution_type=DistributionType.GAUSSIAN,
                    std_type=StdType.STATE_INDEPENDENT,
                )
            ),
        )
    )
    cases.append(
        (
            "cf2x_hover (PPO MLP tanh scalar)",
            NNConfig(
                policy=PPOPolicyConfig(
                    actor=MLPActorCfg(
                        hidden_dims=[128, 128],
                        activation=Activation.TANH,
                        init=OrthoInit(output_gain=1.0),
                    ),
                    critic=MLPCriticCfg(
                        hidden_dims=[128, 128],
                        activation=Activation.TANH,
                        init=OrthoInit(output_gain=1.0),
                    ),
                    init_noise_std=0.5,
                    distribution_type=DistributionType.GAUSSIAN,
                    std_type=StdType.SCALAR,
                )
            ),
        )
    )

    # Algorithm variants.
    cases.append(
        (
            "SAC (squashed gaussian + ReLU)",
            NNConfig(
                policy=SACPolicyConfig(
                    actor=MLPActorCfg(hidden_dims=[256, 256], activation=Activation.RELU),
                    critic=MLPCriticCfg(hidden_dims=[64, 64, 64], activation=Activation.RELU),
                    init_noise_std=0.05,
                )
            ),
        )
    )
    cases.append(
        (
            "TD3",
            NNConfig(
                policy=TD3PolicyConfig(
                    actor=MLPActorCfg(hidden_dims=[256, 256], activation=Activation.RELU),
                    critic=MLPCriticCfg(hidden_dims=[256, 256], activation=Activation.RELU),
                )
            ),
        )
    )
    cases.append(
        (
            "FastTD3 (DefaultInit)",
            NNConfig(
                policy=FastTD3PolicyConfig(
                    actor=MLPActorCfg(
                        hidden_dims=[256, 128, 128],
                        activation=Activation.RELU,
                        init=DefaultInit(),
                    ),
                    critic=MLPCriticCfg(
                        hidden_dims=[256, 128, 128],
                        activation=Activation.RELU,
                        init=DefaultInit(),
                    ),
                )
            ),
        )
    )

    # Coverage for every std_type + distribution_type combination.
    for std in StdType:
        cases.append(
            (
                f"PPO std_type={std.value}",
                NNConfig(
                    policy=PPOPolicyConfig(
                        actor=MLPActorCfg(activation=Activation.GELU),
                        critic=MLPCriticCfg(activation=Activation.GELU),
                        std_type=std,
                    )
                ),
            )
        )
    for dist in DistributionType:
        cases.append(
            (
                f"PPO distribution_type={dist.value}",
                NNConfig(
                    policy=PPOPolicyConfig(
                        actor=MLPActorCfg(activation=Activation.SELU),
                        critic=MLPCriticCfg(activation=Activation.SELU),
                        distribution_type=dist,
                    )
                ),
            )
        )

    return cases


# ── Roundtrip checks ───────────────────────────────────────────────


def _no_python_object_tag(yaml_text: str) -> bool:
    return "!!python/object" not in yaml_text and "!!python/name" not in yaml_text


def _values_match(a: NNConfig, b: NNConfig) -> tuple[bool, str]:
    """Deep equality of policy / actor / critic fields after a roundtrip."""
    if type(a.policy) is not type(b.policy):
        return False, f"policy type {type(a.policy).__name__} != {type(b.policy).__name__}"
    if type(a.policy.actor) is not type(b.policy.actor):
        return False, f"actor type {type(a.policy.actor).__name__} != {type(b.policy.actor).__name__}"
    if type(a.policy.critic) is not type(b.policy.critic):
        return False, f"critic type {type(a.policy.critic).__name__} != {type(b.policy.critic).__name__}"

    def _field_values(obj):
        return {f.name: _coerce(getattr(obj, f.name)) for f in dc_fields(obj)}

    def _coerce(v):
        if isinstance(v, Enum):
            return v.value
        if isinstance(v, list | tuple):
            return list(v)
        return v

    for label, x, y in (
        ("policy", a.policy, b.policy),
        ("actor", a.policy.actor, b.policy.actor),
        ("critic", a.policy.critic, b.policy.critic),
    ):
        # Compare scalar-ish fields (skip nested Cfg sub-objects, handled separately).
        xv = _field_values(x)
        yv = _field_values(y)
        for k in xv:
            xk = xv[k]
            yk = yv.get(k, "<missing>")
            # Skip nested Cfg fields (actor/critic of the policy and the
            # init scheme dataclass) — those are compared by type above
            # and via their own field walk in the helpers below.
            if k in ("actor", "critic"):
                continue
            if k == "init":
                if type(xk) is not type(yk):
                    return False, f"{label}.init type {type(xk).__name__} != {type(yk).__name__}"
                # OrthoInit has output_gain to compare.
                if isinstance(xk, OrthoInit):
                    if xk.output_gain != yk.output_gain:
                        return False, f"{label}.init.output_gain {xk.output_gain} != {yk.output_gain}"
                continue
            if xk != yk:
                return False, f"{label}.{k}: {xk!r} != {yk!r}"
    return True, "OK"


def _run_one_case(name: str, nn: NNConfig, tmp_dir: str) -> dict:
    """Apply all 3 roundtrip variants + the tag-leak check. Returns
    a dict with per-step results.
    """
    result: dict[str, Any] = {"case": name, "checks": {}}

    try:
        d = nn.recursive_to_dict()
    except Exception as e:
        result["checks"]["recursive_to_dict"] = f"FAIL: {type(e).__name__}: {e}"
        result["passed"] = False
        return result
    result["checks"]["recursive_to_dict"] = "OK"

    # Variant 1: yaml.safe_dump + yaml.safe_load
    try:
        text = yaml.safe_dump(d, sort_keys=False)
        if not _no_python_object_tag(text):
            result["checks"]["safe_dump_no_tag"] = "FAIL: !!python/object tag present"
            result["passed"] = False
            return result
        result["checks"]["safe_dump_no_tag"] = "OK"
        loaded = yaml.safe_load(text)
        nn2 = NNConfig.from_dict(loaded)
        ok, msg = _values_match(nn, nn2)
        result["checks"]["safe_dump_roundtrip"] = "OK" if ok else f"FAIL: {msg}"
        if not ok:
            result["passed"] = False
            return result
    except Exception as e:
        result["checks"]["safe_dump_roundtrip"] = f"FAIL: {type(e).__name__}: {e}\n{traceback.format_exc()}"
        result["passed"] = False
        return result

    # Variant 2: yaml.dump (unsafe) + yaml.safe_load
    # If dump-stage leaks an Enum instance, safe_load will refuse.
    try:
        text2 = yaml.dump(d, sort_keys=False)
        if not _no_python_object_tag(text2):
            result["checks"]["unsafe_dump_no_tag"] = "FAIL: !!python/object tag present"
            result["passed"] = False
            return result
        result["checks"]["unsafe_dump_no_tag"] = "OK"
        loaded2 = yaml.safe_load(text2)
        nn3 = NNConfig.from_dict(loaded2)
        ok, msg = _values_match(nn, nn3)
        result["checks"]["unsafe_dump_roundtrip"] = "OK" if ok else f"FAIL: {msg}"
        if not ok:
            result["passed"] = False
            return result
    except Exception as e:
        result["checks"]["unsafe_dump_roundtrip"] = f"FAIL: {type(e).__name__}: {e}\n{traceback.format_exc()}"
        result["passed"] = False
        return result

    # Variant 3: framework's own dump_yaml / load_yaml.
    try:
        # Filename based on slug; spaces / parens replaced.
        slug = "".join(c if c.isalnum() else "_" for c in name)
        path = os.path.join(tmp_dir, f"{slug}.yaml")
        dump_yaml(path, d)
        loaded3 = load_yaml(path)
        nn4 = NNConfig.from_dict(loaded3)
        ok, msg = _values_match(nn, nn4)
        result["checks"]["framework_dump_load"] = "OK" if ok else f"FAIL: {msg}"
        if not ok:
            result["passed"] = False
            return result
    except Exception as e:
        result["checks"]["framework_dump_load"] = f"FAIL: {type(e).__name__}: {e}\n{traceback.format_exc()}"
        result["passed"] = False
        return result

    result["passed"] = True
    return result


# ── Reporter ───────────────────────────────────────────────────────


def _format_report(results: list[dict]) -> str:
    L: list[str] = []
    L.append("=" * 80)
    L.append("NN config YAML save/load roundtrip diag")
    L.append("=" * 80)
    L.append("")
    n_pass = sum(1 for r in results if r.get("passed"))
    n_fail = len(results) - n_pass
    L.append(f"summary: {n_pass} PASS, {n_fail} FAIL  (out of {len(results)} cases)")
    L.append("")

    for r in results:
        verdict = "PASS" if r.get("passed") else "FAIL"
        L.append(f"[{verdict}] {r['case']}")
        for check_name, status in r["checks"].items():
            short_status = status.split("\n")[0]
            L.append(f"    - {check_name:25s} {short_status}")
        L.append("")

    if n_fail > 0:
        L.append("=" * 80)
        L.append("FAILURE DETAILS")
        L.append("=" * 80)
        for r in results:
            if r.get("passed"):
                continue
            L.append(f"\n[FAIL] {r['case']}")
            for check_name, status in r["checks"].items():
                if not status.startswith("FAIL"):
                    continue
                L.append(f"  {check_name}:")
                for line in status.split("\n"):
                    L.append(f"    {line}")

    return "\n".join(L)


# ── Main ───────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=str, default=_DEFAULT_OUTPUT_DIR)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.output_dir, f"nn_config_yaml_roundtrip_{ts}.txt")

    # Per-case tmp dir for variant 3 (framework dump_yaml).
    import tempfile

    with tempfile.TemporaryDirectory(prefix="nn_cfg_diag_") as tmp_dir:
        results: list[dict] = []
        for name, nn in _build_cases():
            results.append(_run_one_case(name, nn, tmp_dir))

    text = _format_report(results)
    with open(out_path, "w") as f:
        f.write(text)
    print(text)
    print()
    print(f"[diag] wrote {out_path}")

    n_fail = sum(1 for r in results if not r.get("passed"))
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
