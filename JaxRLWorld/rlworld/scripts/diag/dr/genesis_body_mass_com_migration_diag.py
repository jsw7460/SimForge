"""Prove the Genesis body-mass / body-COM domain-randomization migration is correct.

Genesis removed the ``dyn_state`` shift API (``set_mass_shift`` / ``set_COM_shift``
and their getters) in favour of the absolute ``dyn_info`` setters
``set_links_mass`` / ``set_links_COM``. The DR backends in
``rlworld/rl/envs/mdp/events/dr/unified.py`` were migrated to the new API, which
(a) needs ``RigidOptions(batch_links_info=True)`` for per-environment writes and
(b) must scale/offset a snapshotted URDF baseline rather than the current value,
so ratios do not compound across resets.

This diag builds real Genesis presets and verifies, with exact numeric checks,
that the migrated path reproduces the former semantics:

  A. API contract (deterministic, self-set values):
     A1 build has ``batch_links_info=True``
     A2 ``set_links_mass(base*r)`` -> ``get_links_mass`` == ``base*r`` exactly
     A3 mass write leaves per-link inertia unchanged (``scale_inertia=False``)
     A4 ``set_links_COM(base+off)`` -> ``get_links_COM`` == ``base+off`` exactly
     A5 per-environment independence (distinct values survive per env)
  B. EventTerm end-to-end (the actual migrated code path):
     B1 ``randomize_body_mass`` -> effective mass / baseline in ``mass_range``
     B2 firing twice does NOT compound (baseline stays the build value)
     B3 ``randomize_body_com_offset`` -> per-axis offset within ``ranges``
     B4 COM read-modify-write preserves a prior axis (no clobber)
  C. Guard: ``batch_links_info=False`` makes the terms raise, not silently
     clobber every environment.

Run (JAX-free build, but the runner imports JAX -> use jaxpy):
    jaxpy -m rlworld.scripts.diag.dr.genesis_body_mass_com_migration_diag \
        --num-envs 16 --out genesis_body_mass_com_migration_diag.txt
"""

import argparse
import os
from contextlib import contextmanager
from pathlib import Path

import torch

from rlworld.rl.configs.presets.g1_29dof.base import G1FlatConfig
from rlworld.rl.configs.presets.go2.base import Go2FlatConfig
from rlworld.rl.configs.scene import SceneEntitySelector
from rlworld.rl.envs.mdp.events.dr.unified import (
    _genesis_require_batched_links_info,
    randomize_body_com_offset,
    randomize_body_mass,
)
from rlworld.rl.runners import BaseRunner

_PRESETS = {"go2": Go2FlatConfig, "g1": G1FlatConfig}


@contextmanager
def _silence_fd():
    """Redirect fd 1/2 to /dev/null so warp/genesis C-level build noise stays
    out of the report; Python-level prints go through a saved handle."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = (os.dup(1), os.dup(2))
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        os.close(saved[0])
        os.close(saved[1])
        os.close(devnull)


def _build(preset: str, num_envs: int, seed: int):
    cfg = _PRESETS[preset](sim_type="genesis", num_envs=num_envs, seed=seed)
    cfgs = cfg.build()
    with _silence_fd():
        env = BaseRunner.create_with_env(cfgs, use_wandb=False).env
        env.reset()
    return env


class _Report:
    def __init__(self):
        self.lines: list[str] = []
        self.failures = 0

    def log(self, msg: str = "") -> None:
        print(msg)
        self.lines.append(msg)

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        tag = "PASS" if ok else "FAIL"
        if not ok:
            self.failures += 1
        self.log(f"  [{tag}] {name}" + (f"  ({detail})" if detail else ""))

    def section(self, title: str) -> None:
        self.log("")
        self.log("=" * 78)
        self.log(title)
        self.log("=" * 78)


def _resolve(env, body_names):
    sel = SceneEntitySelector(name="robot", body_names=body_names)
    resolved = env.resolve_selector(sel)
    links_idx = resolved.body_ids.tolist()
    return resolved, links_idx


def _stats(t: torch.Tensor) -> str:
    return f"shape={tuple(t.shape)} min={t.min().item():.6g} max={t.max().item():.6g} mean={t.mean().item():.6g}"


# ─────────────────────────────────────────────────────────────────────────────
# SECTION A — absolute-setter API contract (deterministic, exact)
# ─────────────────────────────────────────────────────────────────────────────
def _section_a(rep: _Report, env) -> None:
    rep.section("SECTION A — absolute set/get API contract (go2, all links)")
    entity = env.scene_manager["robot"]
    _, links_idx = _resolve(env, (".*",))
    W = env.num_envs
    env_ids = torch.arange(W, device=env.device)

    batched = entity._solver.is_links_info_batched
    rep.check("A1 build uses batch_links_info=True", batched, f"is_links_info_batched={batched}")
    if not batched:
        rep.log("  (A2-A5 skipped: per-env writes impossible without the batched store)")
        return

    base_mass = entity.get_links_mass(links_idx_local=links_idx).clone()
    base_inertia = entity.get_links_inertia(links_idx_local=links_idx).clone()
    base_com = entity.get_links_COM(links_idx_local=links_idx).clone()
    rep.log(f"  baseline mass    {_stats(base_mass)}")
    rep.log(f"  baseline inertia {_stats(base_inertia)}")
    rep.log(f"  baseline COM     {_stats(base_com)}")

    # A2: exact absolute mass round-trip with a per-env, per-link factor.
    torch.manual_seed(101)
    r = torch.empty_like(base_mass).uniform_(0.5, 1.5)
    target = base_mass * r
    entity.set_links_mass(mass=target, links_idx_local=links_idx, envs_idx=env_ids)
    back = entity.get_links_mass(links_idx_local=links_idx)
    err = (back - target).abs().max().item()
    rep.check("A2 get_links_mass == base*r exactly", err < 1e-6, f"max_abs_err={err:.3e}")

    # A3: mass write must leave inertia untouched (matches former set_mass_shift).
    inertia_after = entity.get_links_inertia(links_idx_local=links_idx)
    ierr = (inertia_after - base_inertia).abs().max().item()
    rep.check("A3 inertia unchanged by set_links_mass", ierr < 1e-9, f"max_abs_err={ierr:.3e}")

    # A4: exact absolute COM round-trip on one axis.
    torch.manual_seed(102)
    off = torch.empty(base_com.shape[:-1], device=env.device).uniform_(-0.05, 0.05)
    com_target = base_com.clone()
    com_target[..., 0] = base_com[..., 0] + off
    entity.set_links_COM(com=com_target, links_idx_local=links_idx, envs_idx=env_ids)
    com_back = entity.get_links_COM(links_idx_local=links_idx)
    cerr = (com_back - com_target).abs().max().item()
    rep.check("A4 get_links_COM == base+off exactly", cerr < 1e-6, f"max_abs_err={cerr:.3e}")

    # A5: per-env independence — variance across envs proves the batched store.
    per_env_spread = back.std(dim=0).max().item()
    rep.check("A5 per-env mass independent", per_env_spread > 0.0, f"max_std_over_envs={per_env_spread:.3e}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION B — EventTerm end-to-end (the migrated backend path)
# ─────────────────────────────────────────────────────────────────────────────
def _section_b_mass(rep: _Report, env) -> None:
    rep.section("SECTION B (mass) — randomize_body_mass end-to-end (go2, all links)")
    entity = env.scene_manager["robot"]
    resolved, links_idx = _resolve(env, (".*",))
    W = env.num_envs
    env_ids = torch.arange(W, device=env.device)

    base = entity.get_links_mass(links_idx_local=links_idx).clone()
    rep.log(f"  build mass {_stats(base)}  n_links={len(links_idx)}")

    torch.manual_seed(201)
    randomize_body_mass(env, env_ids, asset_cfg=resolved, mass_range=(0.8, 1.2), operation="scale")
    back1 = entity.get_links_mass(links_idx_local=links_idx)
    ratio1 = back1 / base
    in_range1 = bool((ratio1 >= 0.8 - 1e-4).all() and (ratio1 <= 1.2 + 1e-4).all())
    rep.check("B1 effective mass / baseline within mass_range", in_range1, f"ratio {_stats(ratio1)}")

    cached = env._genesis_dr_baselines[(resolved.name, "links_mass")]
    cache_ok = torch.equal(cached, base)
    rep.check("B1b cached baseline == build mass", cache_ok, f"max_abs_err={(cached - base).abs().max().item():.3e}")

    # Fire again with a different seed; a current-value (compounding) implementation
    # would push the ratio outside mass_range, a baseline implementation stays inside.
    torch.manual_seed(202)
    randomize_body_mass(env, env_ids, asset_cfg=resolved, mass_range=(0.8, 1.2), operation="scale")
    back2 = entity.get_links_mass(links_idx_local=links_idx)
    ratio2 = back2 / base
    in_range2 = bool((ratio2 >= 0.8 - 1e-4).all() and (ratio2 <= 1.2 + 1e-4).all())
    rep.check("B2 second firing does NOT compound (still in mass_range)", in_range2, f"ratio {_stats(ratio2)}")


def _section_b_com(rep: _Report, env) -> None:
    rep.section("SECTION B (COM) — randomize_body_com_offset end-to-end (g1 torso)")
    entity = env.scene_manager["robot"]
    resolved, links_idx = _resolve(env, ("torso_link",))
    W = env.num_envs
    env_ids = torch.arange(W, device=env.device)
    ranges = {0: (-0.025, 0.025), 1: (-0.025, 0.025), 2: (-0.03, 0.03)}

    base = entity.get_links_COM(links_idx_local=links_idx).clone()
    rep.log(f"  build COM {_stats(base)}  links={links_idx}")

    torch.manual_seed(203)
    randomize_body_com_offset(env, env_ids, ranges=ranges, asset_cfg=resolved, operation="add")
    back = entity.get_links_COM(links_idx_local=links_idx)
    delta = back - base
    ok = True
    detail = []
    for axis, (lo, hi) in ranges.items():
        d = delta[..., axis]
        axis_ok = bool((d >= lo - 1e-4).all() and (d <= hi + 1e-4).all())
        ok = ok and axis_ok
        detail.append(f"ax{axis}[{lo},{hi}] min={d.min().item():.4f} max={d.max().item():.4f}")
    rep.check("B3 per-axis COM offset within ranges", ok, "; ".join(detail))

    # B4: read-modify-write must not clobber an axis a prior term already set.
    torch.manual_seed(204)
    randomize_body_com_offset(env, env_ids, ranges={0: (-0.025, 0.025)}, asset_cfg=resolved, operation="add")
    com_a = entity.get_links_COM(links_idx_local=links_idx).clone()
    axis0_first = com_a[..., 0].clone()
    torch.manual_seed(205)
    randomize_body_com_offset(env, env_ids, ranges={2: (-0.03, 0.03)}, asset_cfg=resolved, operation="add")
    com_b = entity.get_links_COM(links_idx_local=links_idx)
    clobber_err = (com_b[..., 0] - axis0_first).abs().max().item()
    axis2_moved = (com_b[..., 2] - com_a[..., 2]).abs().max().item()
    rep.check("B4 untouched axis preserved (no clobber)", clobber_err < 1e-6, f"axis0_drift={clobber_err:.3e}")
    rep.check("B4b targeted axis actually moved", axis2_moved > 0.0, f"axis2_delta_max={axis2_moved:.3e}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION C — guard against silent cross-env clobber
# ─────────────────────────────────────────────────────────────────────────────
def _section_c(rep: _Report) -> None:
    rep.section("SECTION C — batch_links_info=False guard raises (no silent clobber)")

    class _FakeSolver:
        is_links_info_batched = False

    class _FakeEntity:
        _solver = _FakeSolver()

    raised = False
    try:
        _genesis_require_batched_links_info(_FakeEntity(), "randomize_body_mass")
    except RuntimeError as exc:
        raised = True
        rep.log(f"  raised: {exc}")
    rep.check("C1 guard raises RuntimeError when unbatched", raised)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--num-envs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("genesis_body_mass_com_migration_diag.txt"))
    args = ap.parse_args()

    rep = _Report()
    rep.log("Genesis body-mass / body-COM DR migration diagnostic")
    rep.log(f"num_envs={args.num_envs} seed={args.seed} torch={torch.__version__}")

    # go2 exercises randomize_body_mass. Run the EventTerm section FIRST so its
    # snapshotted baseline is the untouched build mass (B1b checks exactly that);
    # the absolute-API section afterwards overwrites mass/COM for its own checks.
    env_go2 = _build("go2", args.num_envs, args.seed)
    rep.log(f"go2 built: {env_go2.num_envs} envs, device={env_go2.device}")
    _section_b_mass(rep, env_go2)
    _section_a(rep, env_go2)

    # g1 exercises randomize_body_com_offset (torso COM).
    env_g1 = _build("g1", args.num_envs, args.seed)
    rep.log("")
    rep.log(f"g1 built: {env_g1.num_envs} envs, device={env_g1.device}")
    _section_b_com(rep, env_g1)

    _section_c(rep)

    rep.section("SUMMARY")
    rep.check("ALL CHECKS PASSED", rep.failures == 0, f"{rep.failures} failure(s)")

    report = "\n".join(rep.lines) + "\n"
    args.out.write_text(report)
    print(f"\nwrote {args.out}")
    raise SystemExit(1 if rep.failures else 0)


if __name__ == "__main__":
    main()
