"""Do the three backends test, and resolve, the same self-collision pairs of every entity?

For one preset on one backend this builds the real env and reads the engine's
own compiled collision tables and contact lists:

* candidate pairs: the geom pairs the broadphase will test, as the engine
  stores them (mujoco-warp ``nxn_pairid`` on the mjlab and Newton backends,
  the collider's valid pair list on Genesis), reduced to intra-entity link
  pairs by bare link name;
* self contacts: ``--samples`` random joint configurations inside the soft
  joint limits, the same ones on every backend (seeded, written in actuated
  order), the floating entities held in the air, then one contact pass of the
  engine (mjlab ``forward``, Newton ``mujoco_warp.forward`` on the synced
  state, Genesis ``collider.detection``) and the intra-entity link pairs that
  actually touch, per sample.

An entity with ``enable_self_collisions=False`` (Genesis: the scene-wide
``RigidOptions.enable_self_collision``) must show an empty candidate set and no
self contact in any sample. With the flag on, the candidate sets must agree
across backends and the touching pairs should mostly agree (narrowphase
differences allowed). ``--out`` writes the result as json; ``--compare`` diffs
two or three such files.

    jaxpy -m jaxrlworld.scripts.diag.parity.check_self_collision_parity --preset yam_lift --sim mujoco --out runs/sc/yam_lift_mujoco.json
    jaxpy -m jaxrlworld.scripts.diag.parity.check_self_collision_parity --preset yam_lift --sim newton --out runs/sc/yam_lift_newton.json
    jaxpy -m jaxrlworld.scripts.diag.parity.check_self_collision_parity --preset yam_lift --sim genesis --out runs/sc/yam_lift_genesis.json
    jaxpy -m jaxrlworld.scripts.diag.parity.check_self_collision_parity --compare runs/sc/yam_lift_*.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from jaxrlworld.rl.runners import BaseRunner
from jaxrlworld.scripts.diag.gates.check_all_presets import _PUBLIC, _extra_specs, _load

_LIFT_Z = 1.5  # m, floating entities are held this high so only self contacts can occur


def _bare(name: str) -> str:
    return name.rsplit("/", 1)[-1]


def _pair(a: str, b: str) -> list[str]:
    return [a, b] if a <= b else [b, a]


# ---------------------------------------------------------------------
# per-backend engine access
# ---------------------------------------------------------------------


def _active_rows(probe, geom, world, dist, includemargin):
    """``[(world, geom_a, geom_b)]`` of the rows MuJoCo hands to the solver (``dist < includemargin``).

    mujoco-warp also stores rows for pairs inside the geom margin but outside
    ``includemargin`` (Newton's 0.1 m default margin fills ``d.contact`` with
    them); those never exert force, so they are counted on the probe and
    left out of the touching pairs.
    """
    active = dist < includemargin
    probe.margin_only_rows += int((~active).sum())
    return [(int(w), int(g[0]), int(g[1])) for w, g in zip(world[active], geom[active])]


def _mjlab_np(field):
    """mjlab serves model/data fields as ``TorchArray`` views on the GPU; read the warp array behind them."""
    return field.wp_array.numpy()


class _MujocoProbe:
    def __init__(self, env, entity_names):
        import mujoco

        self.env = env
        self.margin_only_rows = 0
        self.mjm = env.scene_manager.mj_model
        self.body_name = [mujoco.mj_id2name(self.mjm, mujoco.mjtObj.mjOBJ_BODY, b) or "" for b in range(self.mjm.nbody)]
        self.body_entity = [
            next((n for n in entity_names if name.startswith(f"{n}/")), None) for name in self.body_name
        ]

    def geom_owner(self, geom: int) -> tuple[str | None, str]:
        body = int(self.mjm.geom_bodyid[geom])
        return self.body_entity[body], _bare(self.body_name[body])

    def candidate_geom_pairs(self):
        model = self.env.scene_manager.model
        pairs = _mjlab_np(model.nxn_geom_pair)
        # (npairs, 2): column 0 is the contact pair id, -2 where the filter dropped the pair.
        pairid = _mjlab_np(model.nxn_pairid)
        return pairs[pairid[:, 0] != -2]

    def contacts(self):
        """``[(world, geom_a, geom_b)]`` after one forward of the current state."""
        self.env.scene_manager.forward()
        d = self.env.scene_manager.data
        n = int(_mjlab_np(d.nacon)[0])
        return _active_rows(
            self,
            _mjlab_np(d.contact.geom)[:n],
            _mjlab_np(d.contact.worldid)[:n],
            _mjlab_np(d.contact.dist)[:n],
            _mjlab_np(d.contact.includemargin)[:n],
        )

    def flags(self, cfgs) -> dict[str, bool]:
        return {n: c.enable_self_collisions for n, c in cfgs.scene.entities.items()}


class _NewtonProbe:
    def __init__(self, env, entity_names):
        import mujoco_warp

        self.env = env
        self.margin_only_rows = 0
        self.mujoco_warp = mujoco_warp
        sm = env.scene_manager
        if not sm._use_mujoco_contacts:
            raise NotImplementedError("the Newton probe reads the mjwarp contact tables; use_mujoco_contacts=True only")
        self.sm = sm
        self.solver = sm.solver
        labels = sm.model.body_label
        prefixes = {n: sm.entities[n]["config"].body_label_prefix for n in entity_names}
        self.body_entity = [next((n for n, p in prefixes.items() if lab.startswith(f"{p}/")), None) for lab in labels]
        self.body_bare = [_bare(lab) for lab in labels]
        self.shape_body = sm.model.shape_body.numpy()
        self.geom_to_shape = self.solver.mjc_geom_to_newton_shape.numpy()  # (nworld, ngeom)

    def geom_owner(self, geom: int, world: int = 0) -> tuple[str | None, str]:
        shape = int(self.geom_to_shape[world, geom])
        if shape < 0:
            return None, "<static>"
        body = int(self.shape_body[shape])
        if body < 0:
            return None, "<world>"
        return self.body_entity[body], self.body_bare[body]

    def candidate_geom_pairs(self):
        model = self.solver.mjw_model
        pairs = model.nxn_geom_pair.numpy()
        # (npairs, 2): column 0 is the contact pair id, -2 where the filter dropped the pair.
        pairid = model.nxn_pairid.numpy()
        return pairs[pairid[:, 0] != -2]

    def contacts(self):
        self.solver._update_mjc_data(self.solver.mjw_data, self.sm.model, self.sm.state_0)
        self.mujoco_warp.forward(self.solver.mjw_model, self.solver.mjw_data)
        d = self.solver.mjw_data
        n = int(d.nacon.numpy()[0])
        return _active_rows(
            self,
            d.contact.geom.numpy()[:n],
            d.contact.worldid.numpy()[:n],
            d.contact.dist.numpy()[:n],
            d.contact.includemargin.numpy()[:n],
        )

    def flags(self, cfgs) -> dict[str, bool]:
        return {n: c.enable_self_collisions for n, c in cfgs.scene.entities.items()}


class _GenesisProbe:
    def __init__(self, env, entity_names):
        from genesis.utils.misc import qd_to_torch

        self.env = env
        self.margin_only_rows = 0
        self.qd_to_torch = qd_to_torch
        self.solver = env.scene_manager.scene.sim.rigid_solver
        self.link_owner: dict[int, tuple[str, str]] = {}
        for n in entity_names:
            for link in env.scene_manager[n].links:
                self.link_owner[link.idx] = (n, link.name)

    def geom_owner(self, geom: int, world: int = 0) -> tuple[str | None, str]:
        link = self.solver.geoms[geom].link
        return self.link_owner.get(link.idx, (None, link.name))

    def link_owner_of(self, link_idx: int) -> tuple[str | None, str]:
        return self.link_owner.get(link_idx, (None, f"link{link_idx}"))

    def candidate_geom_pairs(self):
        return self.solver.collider._valid_collision_pairs

    def contacts(self):
        """``[(world, link_a, link_b)]`` after one detection pass of the current state."""
        self.solver.collider.clear()
        self.solver.collider.detection()
        cd = self.solver.collider.get_contacts(as_tensor=True, to_torch=True)
        n_live = self.qd_to_torch(self.solver.collider.collider_state.n_contacts).cpu()
        link_a, link_b = cd["link_a"].cpu(), cd["link_b"].cpu()
        out = []
        for w in range(link_a.shape[0]):
            for row in range(int(n_live[w])):
                out.append((w, int(link_a[w, row]), int(link_b[w, row])))
        return out

    def flags(self, cfgs) -> dict[str, bool]:
        on = bool(self.solver._enable_self_collision)
        return {n: on for n in cfgs.scene.entities}


# ---------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------


def _owner_pairs(probe, pairs, by_link: bool):
    """Group engine pairs into ``{entity: set(link pair)}``; cross-entity pairs under ``"<cross>"``."""
    out: dict[str, set[tuple[str, str]]] = {}
    for item in pairs:
        if by_link:
            w, a, b = item
            (ea, na), (eb, nb) = probe.link_owner_of(a), probe.link_owner_of(b)
        else:
            # Candidates arrive as ``(geom_a, geom_b)``, contacts as ``(world, geom_a, geom_b)``.
            a, b = int(item[-2]), int(item[-1])
            if a < 0 or b < 0:
                continue
            (ea, na), (eb, nb) = probe.geom_owner(a), probe.geom_owner(b)
        if ea is None or eb is None:
            continue
        key = ea if ea == eb else "<cross>"
        out.setdefault(key, set()).add(tuple(_pair(na, nb)))
    return out


def run(args) -> dict:
    loader = next(
        loader
        for label, loader, sims in list(_PUBLIC) + (list(_extra_specs(args.specs).PRESETS) if args.specs else [])
        if label == args.preset and args.sim in sims
    )
    cfgs = _load(loader, args.sim, args.num_envs)
    env = BaseRunner.create_with_env(cfgs, use_wandb=False).env
    env.reset()
    entity_names = list(cfgs.scene.entities)
    probe = {"mujoco": _MujocoProbe, "newton": _NewtonProbe, "genesis": _GenesisProbe}[args.sim](env, entity_names)
    flags = probe.flags(cfgs)
    by_link = args.sim == "genesis"

    cand = _owner_pairs(probe, probe.candidate_geom_pairs(), by_link=False)
    result = {"preset": args.preset, "sim": args.sim, "entities": {}}
    env_ids = torch.arange(env.num_envs, device=env.device)
    gen = torch.Generator().manual_seed(args.seed)
    writers = {n: env.get_root_state_writer(n) for n in entity_names}
    limits = {n: env.get_entity_data(n).soft_joint_pos_limits for n in entity_names}
    touching: dict[str, list[set[tuple[str, str]]]] = {n: [] for n in entity_names}
    cross_touch = 0
    for _ in range(args.samples):
        for n in entity_names:
            lo, hi = limits[n]
            u = torch.rand((env.num_envs, lo.shape[0]), generator=gen).to(env.device)
            writers[n].set_dof_positions(lo + (hi - lo) * u, env_ids)
            if cfgs.scene.entities[n].floating:
                pos = torch.tensor([0.0, 0.0, _LIFT_Z], device=env.device).expand(env.num_envs, 3).clone()
                quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device).expand(env.num_envs, 4).clone()
                writers[n].set_root_pose(pos, quat, env_ids)
            writers[n].eval_fk(env_ids)
        env._invalidate_cache()
        per_world: dict[int, list] = {}
        for item in probe.contacts():
            per_world.setdefault(item[0], []).append(item)
        for w in range(env.num_envs):
            groups = _owner_pairs(probe, per_world.get(w, []), by_link=by_link)
            for n in entity_names:
                touching[n].append(groups.get(n, set()))
            cross_touch += len(groups.get("<cross>", set())) > 0

    for n in entity_names:
        hits = [s for s in touching[n] if s]
        result["entities"][n] = {
            "flag": flags[n],
            "candidate_pairs": sorted(list(p) for p in cand.get(n, set())),
            "samples": len(touching[n]),
            "samples_with_self_contact": len(hits),
            "touching_pairs": sorted(list(p) for p in set().union(*hits)) if hits else [],
        }
    result["cross_entity_candidate_pairs"] = len(cand.get("<cross>", set()))
    result["margin_only_rows"] = probe.margin_only_rows
    result["samples_with_cross_entity_contact"] = cross_touch
    return result


def report(result: dict) -> bool:
    ok = True
    print(
        f"{result['preset']}:{result['sim']}  cross-entity candidate pairs={result['cross_entity_candidate_pairs']}  "
        f"margin-only contact rows dropped={result['margin_only_rows']}"
    )
    for n, r in result["entities"].items():
        print(
            f"  [{n}] flag={r['flag']} candidate intra pairs={len(r['candidate_pairs'])} "
            f"samples with self contact={r['samples_with_self_contact']}/{r['samples']} "
            f"distinct touching pairs={len(r['touching_pairs'])}"
        )
        if not r["flag"] and (r["candidate_pairs"] or r["samples_with_self_contact"]):
            print("    FAIL flag off but the engine tests or resolves intra-entity pairs")
            ok = False
    return ok


def compare(paths: list[str]) -> bool:
    results = [json.loads(Path(p).read_text()) for p in paths]
    ok = True
    names = list(results[0]["entities"])
    for n in names:
        cands = {r["sim"]: {tuple(p) for p in r["entities"][n]["candidate_pairs"]} for r in results}
        touch = {r["sim"]: {tuple(p) for p in r["entities"][n]["touching_pairs"]} for r in results}
        hits = {r["sim"]: r["entities"][n]["samples_with_self_contact"] for r in results}
        sims = list(cands)
        print(f"[{n}]")
        for s in sims:
            print(
                f"  {s:8s} candidates={len(cands[s]):4d} touching={len(touch[s]):4d} samples_with_self_contact={hits[s]}"
            )
        base = cands[sims[0]]
        for s in sims[1:]:
            if cands[s] != base:
                ok = False
                print(
                    f"  FAIL candidate sets differ {sims[0]} vs {s}: only-{sims[0]}={sorted(base - cands[s])[:8]} "
                    f"only-{s}={sorted(cands[s] - base)[:8]}"
                )
        if any(touch.values()):
            union = set().union(*touch.values())
            inter = set.intersection(*touch.values())
            print(f"  touching-pair agreement across sims: {len(inter)}/{len(union)} (intersection/union)")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--preset", help="label from check_all_presets' table")
    ap.add_argument("--sim", choices=("mujoco", "newton", "genesis"))
    ap.add_argument("--num-envs", type=int, default=16)
    ap.add_argument("--samples", type=int, default=20, help="random joint configurations (x num_envs worlds)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--specs", default=None, help="module exposing PRESETS for presets outside this package")
    ap.add_argument("--out", default=None, help="write the result json here")
    ap.add_argument("--compare", nargs="+", default=None, help="result json files to diff instead of running")
    args = ap.parse_args()

    if args.compare:
        return 0 if compare(args.compare) else 1
    if not args.preset or not args.sim:
        ap.error("--preset and --sim are required unless --compare is given")
    result = run(args)
    ok = report(result)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=1))
        print("wrote", args.out)
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
