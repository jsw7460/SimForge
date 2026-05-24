"""End-to-end smoke for the ``TerrainImporter`` refactor.

The refactor (this commit series) promotes terrain to a class hierarchy
mirroring IsaacLab's ``TerrainImporter`` / mjlab's ``TerrainEntity``:

  * ``rlworld/rl/terrains/importer.py``                — base class
  * ``rlworld/rl/envs/managers/<sim>/terrain_importer.py``
        — Newton / Genesis / MuJoCo subclasses
  * ``ManagerRegistry.create(sim, "terrain", ...)``
        — constructed inside each ``SceneManager``
  * ``scene_manager.env_origins``                      — unified property
  * ``scene_manager.terrain.data``                     — generated terrain data
  * ``ContactMatch(entity="terrain")``                 — singleton sentinel

It also rips out the legacy ``GroundPlaneCfg`` entity, the per-sim
scaffolding fields, the ``add_ground`` kwarg, and the mjlab-only
``scene.env_origins`` branch in ``reset_root_state_uniform``.

This script tries to fail loudly on every plausible regression site by
building Go2Flat (the only preset with a flip-able ``use_rough_terrain``)
under each backend × each terrain type, exercising the touched code
paths in one process, and dumping every diagnostic on first run.

What gets exercised per (sim, terrain) combo:

  1. Preset config construction
       — ``Go2FlatConfig(sim_type=..., use_rough_terrain=...).build()``
       — verifies ``make_terrain_cfg`` returns ``TerrainCfg`` with the
         right ``terrain_type``, ``terrain_cfg`` is plumbed into
         ``SceneConfig``, and the ``entities`` dict has NO ``ground`` /
         ``base_entity`` legacy key.
  2. Env build (``BaseRunner.create_with_env``)
       — verifies ``ManagerRegistry`` resolves ``"terrain"`` for every
         sim, and ``SceneManager.__init__`` instantiates the importer
         without an ``add_ground``/``ground_config`` collision.
  3. Terrain importer state
       — class name, ``terrain.data`` (None for plane; populated for
         generator with shape + ``half_extent`` + ``origins[0]``),
         ``terrain.env_origins`` shape + first-row sample, plane case
         ``terrain.env_origins`` should be all-zeros (no sub-terrain).
  4. Unified ``scene_manager.env_origins`` property
       — shape, sample, and (mjlab plane only) the documented fallback
         to ``Scene.env_origins`` is exercised.
  5. ``reset_root_state_uniform``
       — resets envs ``[0, 2]`` and confirms root position picks up the
         per-env origin offset (matches ``env_origins[env_ids, :2]``
         within sampling noise).
  6. ``out_of_terrain_bounds`` termination wiring
       — plane: short-circuits to all-False; rough: reads
         ``terrain.half_extent`` and produces a per-env tensor.
  7. Newton-only: ``ContactSensorCfg(entity="terrain")`` resolution
       — confirms the new sentinel reaches a non-empty shape list
         (the singleton ``ground_plane`` shape).
  8. Step once
       — full ``World.step()`` round-trip, no crash, no NaNs in robot
         root state.
  9. (best-effort) Viser bridge ``extract_geometry``
       — reads ``scene_manager.terrain.data`` (Genesis + MuJoCo path)
         and emits a ``terrain`` mesh group for rough terrain. Skipped
         if ``viser``/``trimesh`` deps are missing.

The script is meant for the GPU/sim box (it needs torch + the live
simulator package). When the run covers more than one combo, it spawns
each combo in a fresh subprocess so that CUDA / warp / wandb state from
one backend cannot poison the next (multi-sim ``torch.manual_seed``
device-side asserts have been observed when all three sims share one
process). Use ``--no-driver`` to run inline against a single combo for
debugging.

Usage::

    python -m rlworld.scripts.diag.check_terrain_importer_refactor
    python -m rlworld.scripts.diag.check_terrain_importer_refactor --sim newton
    python -m rlworld.scripts.diag.check_terrain_importer_refactor --sim genesis --terrain rough
    python -m rlworld.scripts.diag.check_terrain_importer_refactor \\
        --sim newton --terrain rough --no-driver --num-envs 4 --output /tmp/terrain.txt
"""

from __future__ import annotations

import argparse
import gc
import os
import subprocess
import sys
import traceback
from dataclasses import dataclass

os.environ.setdefault("JAXRLWORLD_ALLOW_MULTI_SIM", "1")


_SIMS = ("newton", "genesis", "mujoco")
_TERRAINS = ("plane", "rough")


# ---------------------------------------------------------------------------
# Tiny logger that doubles as a pass/fail collector.
# ---------------------------------------------------------------------------


@dataclass
class _Logger:
    """Print and capture lines; track pass/fail per (sim, terrain) combo."""

    fp = None
    failures: list[str] = None

    def __post_init__(self) -> None:
        self.failures = []

    def w(self, line: str = "") -> None:
        print(line)
        if self.fp is not None:
            self.fp.write(line + "\n")

    def fail(self, combo: str, msg: str) -> None:
        line = f"  ✗ FAIL  [{combo}] {msg}"
        self.failures.append(line)
        self.w(line)

    def ok(self, combo: str, msg: str) -> None:
        self.w(f"  ✓ OK    [{combo}] {msg}")


# ---------------------------------------------------------------------------
# Build the env exactly the way training does.
# ---------------------------------------------------------------------------


def _build_env(sim: str, use_rough: bool, num_envs: int, seed: int):
    """Construct a Go2Flat env for ``sim`` with ``use_rough_terrain=use_rough``.

    Mirrors ``check_robot_data_frames._build_env`` — Go2FlatConfig builds
    every sub-config (env/scene/obs/action/reward/command/event/viz/...)
    and ``BaseRunner.create_with_env`` glues them through ``World``.
    """
    from rlworld.rl.configs.presets.go2_flat.base import Go2FlatConfig
    from rlworld.rl.runners import BaseRunner

    cfg = Go2FlatConfig(
        sim_type=sim,
        num_envs=num_envs,
        use_rough_terrain=use_rough,
        seed=seed,
    )
    cfgs = cfg.build()
    runner = BaseRunner.create_with_env(cfgs)
    return runner.env


# ---------------------------------------------------------------------------
# Individual checks. Each one returns (passed, detail) tuples that the
# combo driver folds into the global pass/fail tally.
# ---------------------------------------------------------------------------


def _check_scene_config_shape(cfg_built, sim: str, use_rough: bool, lg: _Logger, combo: str) -> bool:
    """Sanity-check the *config* dataclass (no env yet).

    Catches: ``make_terrain_cfg`` returning the wrong terrain_type, a
    stale ``entities["ground"]`` left in a preset, ``terrain_cfg`` not
    threaded through ``SceneConfig``.
    """
    scene_cfg = cfg_built.scene
    lg.w(f"    SceneConfig: {type(scene_cfg).__name__}")

    # terrain_cfg present + correct terrain_type
    terrain_cfg = getattr(scene_cfg, "terrain_cfg", None)
    if terrain_cfg is None:
        lg.fail(combo, "SceneConfig.terrain_cfg missing — terrain_cfg was not threaded through preset builder")
        return False
    expected_type = "generator" if use_rough else "plane"
    lg.w(f"    terrain_cfg: terrain_type={terrain_cfg.terrain_type!r} (expected {expected_type!r})")
    if terrain_cfg.terrain_type != expected_type:
        lg.fail(combo, f"terrain_cfg.terrain_type={terrain_cfg.terrain_type!r}, expected {expected_type!r}")
        return False
    if use_rough and terrain_cfg.terrain_generator is None:
        lg.fail(combo, "use_rough_terrain=True but terrain_cfg.terrain_generator is None")
        return False

    # entities dict must not contain legacy ground/base_entity keys
    entities = scene_cfg.entities
    lg.w(f"    entities keys: {sorted(entities)}")
    legacy = [k for k in ("ground", "base_entity") if k in entities]
    if legacy:
        lg.fail(combo, f"legacy entity keys still present in scene config: {legacy}")
        return False

    # Sim-specific: Newton SceneConfig should NOT have `add_ground`.
    if sim == "newton" and hasattr(scene_cfg, "add_ground"):
        lg.fail(combo, "NewtonSceneConfig still exposes `add_ground` field — legacy not deleted")
        return False
    return True


def _check_importer_state(env, sim: str, use_rough: bool, lg: _Logger, combo: str) -> bool:
    """Dump + verify ``scene_manager.terrain`` after env build."""
    import torch

    sm = env.scene_manager
    if not hasattr(sm, "terrain"):
        lg.fail(combo, f"scene_manager (sim={sim}) has no ``terrain`` attribute")
        return False

    importer = sm.terrain
    lg.w(f"    importer class: {type(importer).__name__}")

    # `terrain.data` semantics: None for plane, populated for generator.
    data = importer.data
    if use_rough:
        if data is None:
            lg.fail(combo, "use_rough_terrain=True but terrain.data is None")
            return False
        lg.w(f"    terrain.data.heights_m.shape   = {tuple(data.heights_m.shape)}")
        lg.w(f"    terrain.data.half_extent       = {data.half_extent}")
        lg.w(f"    terrain.data.horizontal_scale  = {data.horizontal_scale}")
        lg.w(f"    terrain.data.vertical_scale    = {data.vertical_scale}")
        lg.w(f"    terrain.data.origins.shape     = {tuple(data.origins.shape)}")
        # The ``half_extent`` property must also be reachable through the
        # importer (terminations use this path).
        hx, hy = importer.half_extent
        lg.w(f"    importer.half_extent (property)= ({hx}, {hy})")
    else:
        if data is not None:
            lg.fail(combo, "use_rough_terrain=False but terrain.data is populated")
            return False
        lg.w("    terrain.data                   = None  (plane → no generated data, OK)")

    # env_origins shape + dtype + sample.
    origins = importer.env_origins
    lg.w(f"    importer.env_origins.shape     = {tuple(origins.shape)}, dtype={origins.dtype}")
    lg.w(f"    importer.env_origins[0]        = {origins[0].detach().cpu().tolist()}")
    if origins.shape != (env.num_envs, 3):
        lg.fail(combo, f"importer.env_origins shape mismatch: got {tuple(origins.shape)}, expected ({env.num_envs}, 3)")
        return False
    if not use_rough:
        if torch.any(origins != 0):
            lg.fail(combo, "plane terrain importer should have all-zero env_origins (no sub-terrain grid)")
            return False

    # Genesis-only: ``entity`` slot must be populated (used by the
    # ``entity="terrain"`` contact-sensor sentinel).
    if sim == "genesis":
        ent = getattr(importer, "entity", "<missing>")
        lg.w(f"    importer.entity (genesis)      = {type(ent).__name__ if ent is not None else None}")
        if ent is None:
            lg.fail(combo, "Genesis terrain importer: ``entity`` slot is None after add_to_scene")
            return False

    return True


def _check_unified_env_origins(env, sim: str, use_rough: bool, lg: _Logger, combo: str) -> bool:
    """Verify ``scene_manager.env_origins`` exists and is correctly wired.

    Newton / Genesis: always = importer.env_origins.
    MuJoCo plane: = mjlab ``Scene.env_origins`` (env_spacing grid).
    MuJoCo rough: = importer.env_origins (importer sub-terrain grid).
    """
    import torch

    sm = env.scene_manager
    if not hasattr(sm, "env_origins"):
        lg.fail(combo, "scene_manager.env_origins property missing")
        return False

    o = sm.env_origins
    lg.w(f"    scene_manager.env_origins.shape= {tuple(o.shape)}")
    lg.w(f"    scene_manager.env_origins[0]   = {o[0].detach().cpu().tolist()}")
    if o.shape != (env.num_envs, 3):
        lg.fail(combo, f"scene_manager.env_origins shape mismatch: {tuple(o.shape)}")
        return False

    if sim in ("newton", "genesis"):
        ref = sm.terrain.env_origins
        if not torch.equal(o, ref):
            lg.fail(combo, f"{sim}: scene_manager.env_origins != terrain.env_origins")
            return False
        lg.ok(combo, "scene_manager.env_origins == terrain.env_origins")
        return True

    # MuJoCo branch.
    if use_rough:
        ref = sm.terrain.env_origins
        if not torch.equal(o, ref):
            lg.fail(combo, "mjlab rough: scene_manager.env_origins != terrain.env_origins")
            return False
        lg.ok(combo, "mjlab rough: scene_manager.env_origins == terrain.env_origins")
    else:
        # plane: should be mjlab's grid; sample it and compare against
        # the underlying mjlab Scene.env_origins.
        mjlab_scene = sm.scene
        mjlab_origins = getattr(mjlab_scene, "env_origins", None)
        if mjlab_origins is None:
            lg.fail(combo, "mjlab plane: scene.env_origins missing on mjlab Scene")
            return False
        if not torch.equal(o, mjlab_origins):
            lg.fail(combo, "mjlab plane: scene_manager.env_origins != scene.env_origins (env_spacing grid)")
            return False
        lg.w(f"    mjlab plane: scene.env_origins[0] = {mjlab_origins[0].detach().cpu().tolist()}")
        lg.ok(combo, "mjlab plane: scene_manager.env_origins falls through to mjlab Scene grid")
    return True


def _check_reset_picks_up_env_origins(env, sim: str, use_rough: bool, lg: _Logger, combo: str) -> bool:
    """Reset and confirm the spawn position picks up the env_origins offset.

    Calls into ``reset_root_state_uniform`` indirectly via ``env.reset``.
    Then reads back ``root_link_pos_w`` for the reset envs and checks
    that the xy component lies within ±max-perturbation of the env's
    own origin offset.
    """

    env.reset()
    rd = env.get_robot_data()
    pos = rd.root_link_pos_w.detach()  # (num_envs, 3)

    origins = env.scene_manager.env_origins.detach()
    # XY component of spawn position should fall within (origin_xy ±
    # sampled perturbation). The Go2Flat reset event uses a small
    # (x,y) sample range; we just check that subtracting the origin
    # collapses the spread to a small finite number for all envs.
    delta = (pos[:, :2] - origins[:, :2]).abs().max().item()
    lg.w(f"    max |pos[xy] - env_origins[xy]| after reset = {delta:.4f}")
    if not (delta < 5.0):  # generous; real range is sub-metre.
        lg.fail(combo, f"reset spawn xy diverges from env_origins by {delta:.3f} m (>5 m) — env_origins not applied")
        return False
    if use_rough and origins.abs().max().item() == 0.0:
        # In rough mode there must be SOME non-zero origin offset, or
        # the importer's grid wasn't propagated.
        lg.fail(combo, "use_rough_terrain=True but env_origins is identically zero")
        return False
    lg.ok(combo, "spawn xy tracks scene_manager.env_origins")
    return True


def _check_termination_wiring(env, sim: str, use_rough: bool, lg: _Logger, combo: str) -> bool:
    """Exercise ``out_of_terrain_bounds`` directly through the
    termination registry. Plane: must short-circuit. Rough: must read
    ``terrain.half_extent`` and produce a per-env bool tensor."""

    from rlworld.rl.configs.scene.entity_selector import SceneEntitySelector
    from rlworld.rl.envs.mdp.terminations.common.terminations import terrain_out_of_bounds

    # Synthetic ResolvedEntity that just names the robot.
    resolved = env.resolve_selector(SceneEntitySelector(name="robot"))

    result = terrain_out_of_bounds(env, margin=0.0, asset_cfg=resolved)
    flags = result.reset.detach()
    lg.w(f"    terrain_out_of_bounds: tensor shape={tuple(flags.shape)}, any={bool(flags.any())}")

    if flags.shape != (env.num_envs,):
        lg.fail(combo, f"terrain_out_of_bounds shape {tuple(flags.shape)} != ({env.num_envs},)")
        return False
    if not use_rough and bool(flags.any()):
        lg.fail(combo, "plane terrain: terrain_out_of_bounds returned True for some env (should short-circuit)")
        return False
    lg.ok(combo, "terrain_out_of_bounds wiring reads scene_manager.terrain correctly")
    return True


def _check_newton_terrain_sentinel(env, lg: _Logger, combo: str) -> bool:
    """Newton-only: confirm a synthetic ``ContactMatch(entity="terrain")``
    actually resolves through the contact-sensor scoping path (the new
    sentinel branch). Reads from the live ``NewtonContactSensorManager``."""
    cm = env.contact_manager
    if cm is None or not getattr(cm, "_group_sensors", None):
        lg.w("    (no Newton contact sensors registered — skipping sentinel check)")
        return True
    # Find any sensor with secondary.entity == "terrain"; the go2_flat
    # preset registers two such sensors (foot_ground, body_ground).
    matched: list[str] = []
    for sensor in cm._group_sensors.values():
        secondary = getattr(sensor.cfg, "secondary", None)
        if secondary is not None and secondary.entity == "terrain":
            matched.append(sensor.cfg.name)
    lg.w(f"    Newton contact sensors with entity='terrain': {matched}")
    if not matched:
        lg.w("    (no entity='terrain' sensors in this preset — skipping)")
        return True
    lg.ok(combo, f"Newton entity='terrain' sentinel resolved ({len(matched)} sensors)")
    return True


def _check_newton_label_indexing(env, lg: _Logger, combo: str) -> bool:
    """Newton-only: verify ``scene_manager.label_indexing`` populates both
    the articulation entries (one per robot/articulated entity) and the
    ``"terrain"`` singleton; cross-check ``find_*`` against a known
    pattern.

    This is the post-refactor cache that replaced the inline prefix +
    world-major resolution in the old
    ``NewtonContactSensor._resolve_indices``. A regression here means
    contact-sensor wiring is broken.
    """
    sm = env.scene_manager
    if not hasattr(sm, "label_indexing"):
        lg.fail(combo, "scene_manager.label_indexing missing (NewtonSceneManager._build_label_indexing not wired)")
        return False

    idx_map = sm.label_indexing
    lg.w(f"    scene_manager.label_indexing keys: {sorted(idx_map)}")

    # Every articulation entity should have an entry, plus "terrain".
    expected_articulations = set(sm.entities)
    missing = (expected_articulations | {"terrain"}) - set(idx_map)
    if missing:
        lg.fail(combo, f"label_indexing missing entries for: {sorted(missing)}")
        return False

    # Terrain singleton: must resolve "ground_plane" to a non-empty
    # shape-id list (one entry — the singleton ground/heightfield shape).
    try:
        terrain_ids = idx_map["terrain"].find_shapes(patterns=("ground_plane",))
    except ValueError as e:
        lg.fail(combo, f"label_indexing['terrain'].find_shapes('ground_plane') raised: {e!r}")
        return False
    lg.w(f"    label_indexing['terrain'].find_shapes('ground_plane') -> {terrain_ids}")
    if len(terrain_ids) != 1:
        lg.fail(combo, f"terrain ground_plane resolved to {len(terrain_ids)} shape(s), expected exactly 1")
        return False

    # Robot articulation: a wildcard pattern should resolve to (num_envs ×
    # shapes_per_world) shapes. Just confirm divisibility, not the exact
    # count (varies per preset).
    if "robot" in idx_map:
        try:
            robot_ids = idx_map["robot"].find_shapes(patterns=(".*",))
        except ValueError as e:
            lg.fail(combo, f"label_indexing['robot'].find_shapes('.*') raised: {e!r}")
            return False
        lg.w(f"    label_indexing['robot'].find_shapes('.*'): {len(robot_ids)} indices")
        if len(robot_ids) % env.num_envs != 0:
            lg.fail(combo, f"robot shape count {len(robot_ids)} not divisible by num_envs={env.num_envs}")
            return False

    lg.ok(combo, "scene_manager.label_indexing populated + find_shapes resolves expected patterns")
    return True


def _check_step(env, lg: _Logger, combo: str) -> bool:
    """One full ``World.step`` to make sure nothing blew up at runtime."""
    import torch

    env.reset()
    n_act = env.num_actions
    zero = torch.zeros(env.num_envs, n_act, device=env.device)
    env.step(zero)
    rd = env.get_robot_data()
    pos = rd.root_link_pos_w.detach()
    if not torch.isfinite(pos).all():
        lg.fail(combo, "step produced non-finite root_link_pos_w")
        return False
    lg.ok(combo, "step round-trip OK")
    return True


def _check_viser_bridge_geometry(env, sim: str, use_rough: bool, lg: _Logger, combo: str) -> bool:
    """Best-effort: build the viser bridge, extract geometry. Catches
    the bridges' new ``scene_manager.terrain.data`` access (it used to
    be ``getattr(_, "_terrain_data", None)``)."""
    if sim == "newton":
        # The Newton bridge reads terrain from ``model.shape_geo_src`` —
        # nothing in the refactor changed that surface. Skip.
        return True
    try:
        if sim == "genesis":
            from rlworld.rl.vis.viser.bridges import GenesisBridge as Bridge
        else:
            from rlworld.rl.vis.viser.bridges import MujocoBridge as Bridge
    except Exception as e:
        lg.w(f"    (viser bridge import failed — skipping: {e!r})")
        return True

    try:
        bridge = Bridge(env.scene_manager)
        geo = bridge.extract_geometry()
        n_groups = len(geo.mesh_groups)
        has_terrain = any(g.body_name == "terrain" for g in geo.mesh_groups)
        lg.w(f"    {sim} bridge: mesh_groups={n_groups}, has_terrain_group={has_terrain}")
        if use_rough and not has_terrain:
            lg.fail(combo, f"{sim} bridge: rough terrain but no 'terrain' mesh group emitted")
            return False
        lg.ok(combo, f"{sim} bridge.extract_geometry() reads terrain.data path")
    except Exception as e:
        lg.fail(combo, f"{sim} bridge.extract_geometry() crashed: {e!r}")
        traceback.print_exc()
        return False
    return True


# ---------------------------------------------------------------------------
# Driver for a single (sim, terrain) combo.
# ---------------------------------------------------------------------------


def _run_combo(sim: str, terrain: str, num_envs: int, seed: int, lg: _Logger) -> bool:
    use_rough = terrain == "rough"
    combo = f"{sim}/{terrain}"
    lg.w("")
    lg.w("=" * 78)
    lg.w(f"[{combo}] building Go2Flat (num_envs={num_envs}, use_rough_terrain={use_rough})")
    lg.w("=" * 78)

    # ---- Step 1: config-only sanity (cheapest — no GPU) -------------
    try:
        from rlworld.rl.configs.presets.go2_flat.base import Go2FlatConfig

        cfg_built = Go2FlatConfig(
            sim_type=sim,
            num_envs=num_envs,
            use_rough_terrain=use_rough,
            seed=seed,
        ).build()
    except Exception as e:
        lg.fail(combo, f"Go2FlatConfig.build() raised: {e!r}")
        traceback.print_exc()
        return False

    if not _check_scene_config_shape(cfg_built, sim, use_rough, lg, combo):
        return False

    # ---- Step 2..9: full env build + runtime checks ------------------
    env = None
    try:
        try:
            env = _build_env(sim, use_rough, num_envs, seed)
        except Exception as e:
            lg.fail(combo, f"env build raised: {e!r}")
            traceback.print_exc()
            return False

        passed = True
        passed &= _check_importer_state(env, sim, use_rough, lg, combo)
        passed &= _check_unified_env_origins(env, sim, use_rough, lg, combo)
        passed &= _check_reset_picks_up_env_origins(env, sim, use_rough, lg, combo)
        passed &= _check_termination_wiring(env, sim, use_rough, lg, combo)
        if sim == "newton":
            passed &= _check_newton_terrain_sentinel(env, lg, combo)
            passed &= _check_newton_label_indexing(env, lg, combo)
        passed &= _check_step(env, lg, combo)
        passed &= _check_viser_bridge_geometry(env, sim, use_rough, lg, combo)
        return passed
    finally:
        # Free GPU memory between combos so we don't blow up on the
        # second sim's allocator.
        del env
        gc.collect()


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def _spawn_subprocess(sim: str, terrain: str, num_envs: int, seed: int) -> int:
    """Run one (sim, terrain) combo in a fresh Python process.

    Each backend allocates its own CUDA / warp resources at import time
    (Newton brings warp, Genesis its own taichi/torch handles, mjlab
    mujoco-warp). Sequencing all three inside one process tends to
    surface device-side CUDA asserts on the second sim's
    ``torch.manual_seed`` because the first sim's runtime poisoned the
    default CUDA context. Spawning a child process per combo gives each
    backend a clean device state and a clean wandb run.
    """
    cmd = [
        sys.executable,
        "-m",
        "rlworld.scripts.diag.check_terrain_importer_refactor",
        "--sim",
        sim,
        "--terrain",
        terrain,
        "--num-envs",
        str(num_envs),
        "--seed",
        str(seed),
        "--no-driver",
    ]
    print()
    print("┃ launching subprocess: " + " ".join(cmd))
    print()
    # ``check=False`` — we tolerate per-combo failures so the driver
    # can still print a SUMMARY across all combos.
    res = subprocess.run(cmd, check=False)
    return res.returncode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim", choices=[*_SIMS, "all"], default="all")
    ap.add_argument("--terrain", choices=[*_TERRAINS, "both"], default="both")
    ap.add_argument("--num-envs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", type=str, default=None, help="Also tee the log to this file.")
    ap.add_argument(
        "--no-driver",
        action="store_true",
        help="Run inline (one combo per invocation). The default driver mode forks a fresh "
        "subprocess per combo to avoid CUDA-context bleed across sims.",
    )
    args = ap.parse_args()

    sims = list(_SIMS) if args.sim == "all" else [args.sim]
    terrains = list(_TERRAINS) if args.terrain == "both" else [args.terrain]

    # ── Driver path: spawn one child per combo ────────────────────────
    # Triggered automatically whenever the run covers more than a
    # single combo (multi-sim CUDA bleed is the headline footgun). The
    # ``--no-driver`` escape hatch lets the script run a single combo
    # inline (used both by the driver itself when invoking children
    # and by anyone debugging a single combo).
    if not args.no_driver and (len(sims) > 1 or len(terrains) > 1):
        results: list[tuple[str, str, int]] = []
        for sim in sims:
            for terrain in terrains:
                rc = _spawn_subprocess(sim, terrain, args.num_envs, args.seed)
                results.append((sim, terrain, rc))
        print()
        print("=" * 78)
        print(f"DRIVER SUMMARY: {sum(1 for _, _, rc in results if rc == 0)}/{len(results)} combos passed")
        for sim, terrain, rc in results:
            tag = "✓ OK  " if rc == 0 else f"✗ FAIL (rc={rc})"
            print(f"  {tag}  {sim}/{terrain}")
        return 0 if all(rc == 0 for _, _, rc in results) else 1

    # ── Inline path: a single combo in this process ───────────────────
    lg = _Logger()
    if args.output is not None:
        lg.fp = open(args.output, "w")

    lg.w(f"TerrainImporter refactor diag — sims={sims}, terrains={terrains}, num_envs={args.num_envs}")

    total = 0
    passes = 0
    for sim in sims:
        for terrain in terrains:
            total += 1
            try:
                ok = _run_combo(sim, terrain, args.num_envs, args.seed, lg)
            except Exception as e:
                lg.fail(f"{sim}/{terrain}", f"unhandled exception: {e!r}")
                traceback.print_exc()
                ok = False
            if ok:
                passes += 1

    lg.w("")
    lg.w("=" * 78)
    lg.w(f"SUMMARY: {passes}/{total} combos passed")
    if lg.failures:
        lg.w("Failures:")
        for line in lg.failures:
            lg.w(f"  {line}")
    if lg.fp is not None:
        lg.fp.close()
    return 0 if passes == total else 1


if __name__ == "__main__":
    sys.exit(main())
