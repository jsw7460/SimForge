"""Verify the declarative model-field expansion for MuJoCo domain randomization.

MuJoCo Warp keeps model fields as one shared ``(1, ...)`` row until
``expand_model_fields`` allocates a per-world copy. Two failures follow from
getting the expansion list wrong, and neither of them raises:

  * a DR term writing an unexpanded field makes every environment take the
    same value -- the logs still show resampling every reset, but the policy
    trains against a single draw;
  * ``recompute_constants`` writes derived constants per world, so a derived
    field left at ``(1, ...)`` is an out-of-bounds store: absorbed by memory
    pool slack at small ``num_envs``, a hard CUDA fault at large ones.

The list used to be a lookup table hand-maintained in ``mjlab_env.py``,
duplicating knowledge that lives in the DR backends. It now comes from a
``@requires_model_fields`` declaration on each DR function, collected by
``rlworld.rl.envs.mdp.events.dr._model_fields.collect_expand_fields``.

Run (GPU box)::

    jaxpy -m rlworld.scripts.diag.dr_model_fields_diag
    jaxpy -m rlworld.scripts.diag.dr_model_fields_diag --static-only  # no env build

Checks
------
Static (no environment, no GPU):
  C1  the collected field set is byte-identical to what the retired lookup
      table produced, for a config registering every DR term. This is the
      no-behaviour-change gate for the migration.
  C2  every ``randomize_*`` entry point in ``unified.py`` carries a
      declaration -- a new DR function cannot silently skip the expansion.
  C3  CONTROL: an undecorated DR-package function makes ``collect_expand_fields``
      raise. If this passes, C1/C2 prove nothing.
  C4  the derived-field table matches what the installed mujoco-warp actually
      writes, recovered from ``_src/io.py`` by walking the ``set_const*`` call
      graph. Re-run after every mujoco-warp bump.
  C5  ``RecomputeLevel`` member names still resolve on mjlab's enum and on
      mujoco-warp, which is what ``recompute_constants`` dispatches through.

Live (builds one Go2 MuJoCo environment):
  C6  every collected field is really ``(num_envs, ...)`` in the warp model,
      and a field nobody asked for is still ``(1, ...)``.
  C7  after a reset the randomized fields actually differ across environments.
  C8  the deferred ``set_const`` recompute leaves the physics state finite.
"""

from __future__ import annotations

import argparse
import pathlib
import re

# ``stat`` is written by set_const_0 but never needs expansion: its kernel is
# launched with ``dim=m.stat.meaninertia.shape[0]``, so it sizes itself to
# whatever the array already is and can never store out of bounds.
_WRITE_EXEMPT: frozenset[str] = frozenset({"stat"})

# Fields the Go2 MuJoCo preset randomizes directly (as opposed to derived
# constants recomputed from them). C7 requires cross-env variance for these.
_GO2_DIRECTLY_RANDOMIZED: tuple[str, ...] = ("geom_friction", "body_mass", "dof_frictionloss")


def _hdr(title: str) -> None:
    print("-" * 92)
    print(title)
    print("-" * 92)


# ══════════════════════════════════════════════════════════════════════════
# Frozen copy of the retired mjlab_env.py lookup table (C1 reference)
# ══════════════════════════════════════════════════════════════════════════


def _retired_collect(event_cfg) -> tuple[str, ...]:
    """Reproduce the hand-maintained table that ``collect_expand_fields`` replaced.

    Copied verbatim from ``MujocoEnv._pre_manager_setup`` as it stood before the
    migration, so C1 compares against the real previous behaviour rather than a
    paraphrase of it. Do not "fix" this table -- it is a frozen reference.
    """
    from rlworld.rl.configs.base_config import iter_terms
    from rlworld.rl.configs.events.event_term_config import EventTermConfig
    from rlworld.rl.envs.mdp.events.dr import unified as _unified
    from rlworld.rl.envs.mdp.events.dr._model_fields import RecomputeLevel

    set_const_0_fields = (
        "dof_invweight0",
        "body_invweight0",
        "tendon_length0",
        "tendon_invweight0",
        "actuator_acc0",
        "actuator_biasprm",
        "cam_pos0",
        "cam_poscom0",
        "cam_mat0",
        "light_pos0",
        "light_poscom0",
        "light_dir0",
        "eq_data",
    )
    derived_fields = {
        RecomputeLevel.none: (),
        RecomputeLevel.set_const_fixed: ("body_subtreemass",),
        RecomputeLevel.set_const_0: set_const_0_fields,
        RecomputeLevel.set_const: ("body_subtreemass",) + set_const_0_fields + ("tendon_lengthspring",),
    }
    unified_to_fields = {
        _unified.randomize_friction: (("geom_friction",), RecomputeLevel.none),
        _unified.randomize_body_mass: (("body_mass",), RecomputeLevel.set_const),
        _unified.randomize_body_com_offset: (("body_ipos",), RecomputeLevel.set_const),
        _unified.randomize_pd_gains: (("actuator_gainprm", "actuator_biasprm"), RecomputeLevel.none),
        _unified.randomize_joint_armature: (("dof_armature",), RecomputeLevel.set_const_0),
        _unified.randomize_joint_friction: (("dof_frictionloss",), RecomputeLevel.none),
        _unified.randomize_joint_damping: (("dof_damping",), RecomputeLevel.none),
    }

    out: list[str] = []
    for _name, term in iter_terms(event_cfg, EventTermConfig).items():
        if term.mode == "startup" and "field" in term.params:
            out.append(term.params["field"])
        entry = unified_to_fields.get(term.func)
        if entry is not None:
            fields, level = entry
            out.extend(fields)
            out.extend(derived_fields[level])
    return tuple(dict.fromkeys(out))


def _all_dr_config():
    """EventConfig registering every unified DR term plus an mjlab-native one."""
    from rlworld.rl.configs.common_config_classes import EventConfig
    from rlworld.rl.configs.events import EventTermConfig
    from rlworld.rl.envs.mdp.events.dr import unified as u

    terms = (
        u.randomize_friction,
        u.randomize_body_mass,
        u.randomize_body_com_offset,
        u.randomize_pd_gains,
        u.randomize_joint_armature,
        u.randomize_joint_friction,
        u.randomize_joint_damping,
    )
    cfg = EventConfig()
    for i, func in enumerate(terms):
        setattr(cfg, f"dr_{i}", EventTermConfig(func=func, mode="reset_dr"))
    # An mjlab-native startup DR term names its field in the config itself;
    # that second nomination source must survive the migration too.
    cfg.native_startup = EventTermConfig(func=_native_startup_term, mode="startup", params={"field": "geom_solref"})
    return cfg


def _native_startup_term(env, env_ids, field):
    """Stand-in for an mjlab-native startup DR term (never executed here)."""
    del env, env_ids, field


# ══════════════════════════════════════════════════════════════════════════
# Static checks
# ══════════════════════════════════════════════════════════════════════════


def check_c1() -> bool:
    from rlworld.rl.envs.mdp.events.dr._model_fields import collect_expand_fields

    _hdr("C1  collected fields identical to the retired lookup table")
    cfg = _all_dr_config()
    new, old = collect_expand_fields(cfg), _retired_collect(cfg)
    print(f"  new: {len(new)} fields")
    print(f"  old: {len(old)} fields")
    only_new = [f for f in new if f not in old]
    only_old = [f for f in old if f not in new]
    print(f"  only in new: {only_new or '-'}")
    print(f"  only in old: {only_old or '-'}")
    print(f"  order preserved: {list(new) == list(old)}")
    print(f"  fields: {list(new)}")
    ok = tuple(new) == tuple(old)
    print(f"\n  C1: {'PASS' if ok else 'FAIL'}\n")
    return ok


def check_c2() -> bool:
    from rlworld.rl.envs.mdp.events.dr import unified as u

    _hdr("C2  every randomize_* entry point declares its model fields")
    entries = sorted(n for n in dir(u) if n.startswith("randomize_") and callable(getattr(u, n)))
    ok = True
    for name in entries:
        func = getattr(u, name)
        declared = getattr(func, "model_fields", None)
        level = getattr(func, "recompute", None)
        status = "OK" if declared is not None else "MISSING @requires_model_fields"
        ok &= declared is not None
        level_name = level.name if level is not None else "-"
        print(f"  {name:<32} {status:<30} recompute={level_name:<16} fields={list(declared or [])}")
    print(f"\n  C2: {'PASS' if ok else 'FAIL'}  ({len(entries)} entry points)\n")
    return ok


def check_c3() -> bool:
    from rlworld.rl.configs.common_config_classes import EventConfig
    from rlworld.rl.configs.events import EventTermConfig
    from rlworld.rl.envs.mdp.events.dr import unified as u
    from rlworld.rl.envs.mdp.events.dr._model_fields import collect_expand_fields

    _hdr("C3  CONTROL: an undecorated DR-package term must raise")
    # A private backend helper: lives in the DR package, carries no declaration.
    cfg = EventConfig()
    cfg.undeclared = EventTermConfig(func=u._genesis_friction_backend, mode="reset_dr")
    try:
        fields = collect_expand_fields(cfg)
    except RuntimeError as exc:
        print(f"  raised RuntimeError as expected:\n    {str(exc)[:200]}...")
        print("\n  C3: PASS\n")
        return True
    print(f"  no exception; collected {fields}")
    print("\n  C3: FAIL -- the guard does not discriminate, so C1/C2 prove nothing.\n")
    return False


def _mujoco_warp_write_sets() -> dict[str, set[str]] | None:
    """Fields each ``set_const*`` writes, from the installed mujoco-warp source.

    Walks the call graph so ``set_const`` picks up what its callees write.
    Writes are detected as top-level ``m.<field> =`` assignments plus anything
    named inside a kernel launch's ``outputs=[...]``; reads are ignored because
    only writes can store out of bounds. Returns ``None`` when the source is
    unavailable or unparseable, so the caller reports INCONCLUSIVE rather than
    a spurious failure.
    """
    try:
        import mujoco_warp
    except ImportError:
        return None
    io_path = pathlib.Path(mujoco_warp.__file__).parent / "_src" / "io.py"
    if not io_path.is_file():
        return None
    lines = io_path.read_text().splitlines()
    defs = [(i, m.group(1)) for i, line in enumerate(lines) if (m := re.match(r"^def (\w+)", line))]
    if not defs:
        return None
    bodies = {
        name: "\n".join(lines[i : (defs[k + 1][0] if k + 1 < len(defs) else len(lines))])
        for k, (i, name) in enumerate(defs)
    }

    def direct_writes(body: str) -> set[str]:
        found = set(re.findall(r"^\s*m\.([a-z_0-9]+)\s*=(?!=)", body, re.M))
        for span in re.findall(r"outputs\s*=\s*\[(.*?)\]", body, re.S):
            found |= set(re.findall(r"\bm\.([a-z_0-9]+)", span))
        return found

    def callees(body: str, own: str) -> set[str]:
        return (set(re.findall(r"^\s*(?:return\s+)?([a-z_][a-z_0-9]*)\(", body, re.M)) & set(bodies)) - {own}

    out: dict[str, set[str]] = {}
    for entry in ("set_const_fixed", "set_const_0", "set_const"):
        if entry not in bodies:
            return None
        seen: set[str] = set()
        stack = [entry]
        acc: set[str] = set()
        while stack:
            fn = stack.pop()
            if fn in seen:
                continue
            seen.add(fn)
            acc |= direct_writes(bodies[fn])
            stack.extend(callees(bodies[fn], fn))
        out[entry] = acc - _WRITE_EXEMPT
    return out


def check_c4() -> bool | None:
    from rlworld.rl.envs.mdp.events.dr._model_fields import DERIVED_FIELDS, RecomputeLevel

    _hdr("C4  derived-field table vs the installed mujoco-warp")
    detected = _mujoco_warp_write_sets()
    if detected is None:
        print("  mujoco_warp source unavailable or unparseable -- cannot verify.")
        print("\n  C4: INCONCLUSIVE\n")
        return None

    levels = {
        "set_const_fixed": RecomputeLevel.set_const_fixed,
        "set_const_0": RecomputeLevel.set_const_0,
        "set_const": RecomputeLevel.set_const,
    }
    ok = True
    print(f"  write-detection exemptions (self-sized kernels): {sorted(_WRITE_EXEMPT)}")
    for entry, level in levels.items():
        ours = set(DERIVED_FIELDS[level])
        theirs = detected[entry]
        missing = sorted(theirs - ours)  # dangerous: written per world, never expanded
        extra = sorted(ours - theirs)  # harmless: expanded but no longer written
        ok &= not missing
        print(f"\n  {entry}: ours={len(ours)}  mujoco-warp writes={len(theirs)}")
        print(f"    MISSING from our table (per-world write with no expansion): {missing or '-'}")
        print(f"    extra in our table (stale, harmless):                       {extra or '-'}")
    print(f"\n  C4: {'PASS' if ok else 'FAIL'}\n")
    return ok


def check_c5() -> bool:
    from rlworld.rl.envs.mdp.events.dr._model_fields import RecomputeLevel

    _hdr("C5  RecomputeLevel name contract")
    ok = True
    try:
        from mjlab.managers.event_manager import RecomputeLevel as MjlabRecomputeLevel

        theirs = {lvl.name for lvl in MjlabRecomputeLevel}
        unknown = {lvl.name for lvl in RecomputeLevel} - theirs
        print(f"  ours   : {sorted(lvl.name for lvl in RecomputeLevel)}")
        print(f"  mjlab  : {sorted(theirs)}")
        print(f"  not on mjlab's enum: {sorted(unknown) or '-'}")
        ok &= not unknown
    except ImportError as exc:
        # Printed rather than swallowed: on a training box mjlab is always
        # importable, so a failure here is itself the finding.
        print(f"  mjlab half SKIPPED -- import failed: {exc}")

    try:
        import mujoco_warp

        for level in RecomputeLevel:
            if level is RecomputeLevel.none:
                continue  # never dispatched; recompute is skipped at this level
            resolves = hasattr(mujoco_warp, level.name)
            ok &= resolves
            print(f"  getattr(mujoco_warp, {level.name!r}) resolves: {resolves}")
    except ImportError as exc:
        print(f"  dispatch half SKIPPED -- mujoco_warp import failed: {exc}")
    print(f"\n  C5: {'PASS' if ok else 'FAIL'}\n")
    return ok


# ══════════════════════════════════════════════════════════════════════════
# Live checks
# ══════════════════════════════════════════════════════════════════════════


def _field_view(wp_model, field: str):
    """Torch view of ``wp_model.<field>``, or ``None`` when the field is absent.

    Reads the RAW mujoco-warp Model, not ``sim.model`` -- the latter is a
    ``WarpBridge`` that hands back ``TorchArray`` wrappers, so the underlying
    stored shape (the thing that tells us whether the field was expanded) is
    not what a caller would observe there. Conversion errors are deliberately
    left to propagate: a field that exists but cannot be read is a finding,
    not something to paper over.
    """
    import warp as wp

    arr = getattr(wp_model, field, None)
    if arr is None:
        return None
    return wp.to_torch(arr)


def check_live(num_envs: int, steps: int) -> bool:
    import torch

    from rlworld.rl.configs.presets.go2.base import Go2FlatConfig
    from rlworld.rl.envs.mdp.events.dr._model_fields import collect_expand_fields
    from rlworld.rl.runners import BaseRunner

    cfgs = Go2FlatConfig(sim_type="mujoco", num_envs=num_envs).build()
    collected = collect_expand_fields(cfgs.event)
    env = BaseRunner.create_with_env(cfgs).env
    env.reset()

    sim = env.scene_manager.sim
    wp_model = sim.wp_model
    print()
    _hdr("C6  collected fields are expanded per world; others are not")
    print(f"  num_envs = {env.num_envs}")
    print(f"  collected ({len(collected)}): {list(collected)}")
    print(f"  sim.expanded_fields ({len(sim.expanded_fields)}): {sorted(sim.expanded_fields)}")

    # Independent of any array read: mjlab records what it expanded, so every
    # field we asked for has to appear there.
    not_expanded = [f for f in collected if f not in sim.expanded_fields]
    print(f"  collected but NOT in sim.expanded_fields: {not_expanded or '-'}")
    c6_ok = not not_expanded

    print()
    print(f"  {'field':<24}{'shape':<22}{'per-world':>12}{'numel':>10}")
    checked = 0
    present: list[str] = []
    for field in collected:
        view = _field_view(wp_model, field)
        if view is None:
            # Every collected field names a real mujoco-warp model field; an
            # absent one means the declaration is wrong.
            print(f"  {field:<24}{'<absent from model>':<22}{'-':>12}{'-':>10}")
            c6_ok = False
            continue
        present.append(field)
        if view.numel() == 0:
            # No cameras / lights / tendons on this robot: the field cannot be
            # expanded and cannot be written either.
            print(f"  {field:<24}{str(tuple(view.shape)):<22}{'n/a (empty)':>12}{0:>10}")
            continue
        per_world = view.shape[0] == env.num_envs
        checked += 1
        c6_ok &= per_world
        print(f"  {field:<24}{str(tuple(view.shape)):<22}{str(per_world):>12}{view.numel():>10}")

    if checked == 0:
        print("\n  no non-empty collected field was inspected -- the check proved nothing.")
        c6_ok = False

    control = next(
        (
            f
            for f in ("body_pos", "body_quat", "dof_armature", "dof_damping", "body_ipos", "jnt_range")
            if f not in collected and (v := _field_view(wp_model, f)) is not None and v.numel() > 0
        ),
        None,
    )
    if control is None:
        print("\n  control: no unrequested non-empty field available -- selectivity unverified.")
        c6_ok = False
    else:
        view = _field_view(wp_model, control)
        shared = view.shape[0] == 1
        c6_ok &= shared
        print(f"\n  control (never requested): {control} shape={tuple(view.shape)} still shared: {shared}")

    print(f"\n  C6: {'PASS' if c6_ok else 'FAIL'}  ({checked} fields inspected)\n")

    _hdr("C7  randomized fields actually differ across environments")
    print(f"  {'field':<24}{'across-env spread':>20}{'varies':>9}   note")
    c7_ok = True
    seen_required: set[str] = set()
    for field in present:
        view = _field_view(wp_model, field)
        if view.numel() == 0 or view.shape[0] != env.num_envs:
            print(f"  {field:<24}{'-':>20}{'-':>9}   skipped (empty or shared)")
            continue
        flat = view.reshape(view.shape[0], -1).float()
        spread = float((flat.max(dim=0).values - flat.min(dim=0).values).max())
        varies = spread > 0.0
        required = field in _GO2_DIRECTLY_RANDOMIZED
        if required:
            seen_required.add(field)
            c7_ok &= varies
        note = "REQUIRED" if required else "derived / not randomized by this preset"
        print(f"  {field:<24}{spread:>20.6e}{str(varies):>9}   {note}")

    unchecked = [f for f in _GO2_DIRECTLY_RANDOMIZED if f not in seen_required]
    if unchecked:
        print(f"\n  required fields never evaluated: {unchecked}")
        c7_ok = False
    print(f"\n  C7: {'PASS' if c7_ok else 'FAIL'}  (required: {list(_GO2_DIRECTLY_RANDOMIZED)})\n")

    _hdr("C8  deferred set_const recompute leaves the state finite")
    zero_act = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    for _ in range(steps):
        env.step(zero_act)
    rd = env.get_robot_data("robot")
    checks = {
        "joint_pos": rd.joint_pos,
        "joint_vel": rd.joint_vel,
        "root_pos": rd.root_link_pos_w,
        "root_lin_vel": rd.root_link_lin_vel_w,
    }
    c8_ok = True
    for name, tensor in checks.items():
        finite = bool(torch.isfinite(tensor).all())
        c8_ok &= finite
        peak = float(torch.nan_to_num(tensor.abs(), nan=float("inf")).max())
        print(f"  {name:<16} finite={finite}   max|.|={peak:.6e}")
    print(f"\n  C8: {'PASS' if c8_ok else 'FAIL'}  (after {steps} steps)\n")

    return c6_ok and c7_ok and c8_ok


# ══════════════════════════════════════════════════════════════════════════


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--static-only", action="store_true", help="Skip the checks that build an environment.")
    ap.add_argument("--num-envs", type=int, default=8)
    ap.add_argument("--steps", type=int, default=5)
    args = ap.parse_args()

    print("=" * 92)
    print("DR MODEL-FIELD EXPANSION DIAG")
    print("=" * 92)
    print()

    results: dict[str, bool | None] = {
        "C1 parity with retired table": check_c1(),
        "C2 decorator coverage": check_c2(),
        "C3 guard control": check_c3(),
        "C4 derived table vs mujoco-warp": check_c4(),
        "C5 RecomputeLevel names": check_c5(),
    }
    if not args.static_only:
        results["C6-C8 live expansion"] = check_live(args.num_envs, args.steps)

    print("=" * 92)
    print("SUMMARY")
    print("=" * 92)
    failed = False
    for name, ok in results.items():
        verdict = "INCONCLUSIVE" if ok is None else ("PASS" if ok else "FAIL")
        failed |= ok is False
        print(f"  {name:<36}{verdict}")
    print("-" * 92)
    print(f"OVERALL: {'FAIL' if failed else 'PASS'}")
    print("=" * 92)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
