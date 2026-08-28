"""Genesis native contact sensors vs contact-list reconstruction — value parity.

Gate for the contact-list migration (candidate 4): before replacing the
per-link ``gs.sensors.Contact``/``ContactForce`` pairs with one batched
``collider.get_contacts()`` read + torch filtering, prove the list path
reproduces the native sensors' VALUES, per substep, on a contact-rich
rollout:

    found  : native Contact flag  ==  "an unfiltered pair exists in the list"
             (bit-exact required)
    force  : native ContactForce  ==  sum over unfiltered pairs of the
             signed contact force (a-side: -f, b-side: +f), rotated into
             the LINK LOCAL frame (native kernel applies
             inv_transform_by_quat with the link quaternion —
             Genesis/genesis/engine/sensors/contact_force.py)

List-side footgun handled the way production code must: rows past each
env's live ``n_contacts`` counter are stale on the zero-copy path, so
validity comes from the counter, never from field sentinels.

Scene: pure Genesis (no JaxRLWorld env) — g1 preset physics options, the
same two sensor groups the g1 preset uses (feet-vs-ground with all robot
links blacklisted, self-collision with ground blacklisted), and a flailing
PD drive (large random joint targets + periodic resets) so ground
contacts, lift-offs, falls AND self-collisions all actually occur.

Usage (GPU box):
    python -m rlworld.scripts.diag.engine.genesis_contact_list_parity_diag
    python -m rlworld.scripts.diag.engine.genesis_contact_list_parity_diag --num-envs 4096
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rlworld.scripts.diag.perf.g1_step_benchmark import _DECIMATION, _DT, _G1_XML, _SPAWN_Z, _setup_active_pd


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num-envs", type=int, default=1024)
    ap.add_argument("--num-steps", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--flail", type=float, default=0.8, help="random joint-target amplitude [rad]")
    ap.add_argument("--out", default="genesis_contact_list_parity_diag.txt")
    args = ap.parse_args()

    import genesis as gs
    import torch
    from genesis.utils.geom import inv_transform_by_quat
    from genesis.utils.misc import qd_to_torch

    torch.manual_seed(args.seed)
    gs.init(backend=gs.gpu, logging_level="warning", seed=args.seed)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=_DT, substeps=1),
        rigid_options=gs.options.RigidOptions(
            dt=_DT,
            constraint_solver=gs.constraint_solver.Newton,
            iterations=10,
            ls_iterations=20,
            tolerance=1e-5,
            constraint_timeconst=0.02,
            enable_collision=True,
            enable_self_collision=True,
            enable_joint_limit=True,
            max_collision_pairs=100,
            batch_dofs_info=True,
        ),
        show_viewer=False,
    )
    ground = scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(gs.morphs.MJCF(file=str(_G1_XML), pos=(0.0, 0.0, _SPAWN_Z)))

    # filter_link_idx is a GLOBAL-link-index blacklist (the native kernel
    # compares it against the collider's global link_a/link_b — see
    # GenesisContactSensor, which builds it from link_start/link_end).
    ground_links_global = sorted(link.idx for link in ground.links)
    robot_links_global = sorted(link.idx for link in robot.links)
    robot_links_local = sorted(link.idx_local for link in robot.links)
    feet_local = sorted(link.idx_local for link in robot.links if "ankle_roll" in link.name)

    # Native sensors, exactly as GenesisContactSensor builds them
    # (history_length=0 -> read_ground_truth returns the CURRENT substep frame).
    def add_group(primary_local: list[int], blacklist_global: list[int]):
        contact, force = [], []
        for link in primary_local:
            contact.append(
                scene.add_sensor(
                    gs.sensors.Contact(
                        entity_idx=robot.idx, link_idx_local=link, filter_link_idx=tuple(blacklist_global)
                    )
                )
            )
            force.append(
                scene.add_sensor(
                    gs.sensors.ContactForce(
                        entity_idx=robot.idx, link_idx_local=link, filter_link_idx=tuple(blacklist_global)
                    )
                )
            )
        return contact, force

    feet_contact, feet_force = add_group(feet_local, robot_links_global)  # feet vs ground: blacklist all-but-ground
    self_contact, self_force = add_group(robot_links_local, ground_links_global)  # self: blacklist all-but-robot

    scene.build(n_envs=args.num_envs)
    drive = _setup_active_pd(robot)
    solver = scene.rigid_solver
    dev = torch.device("cuda:0")

    dof_ids = [j.dofs_idx_local[0] for j in robot.joints if j.n_dofs == 1]
    dofs0 = robot.get_dofs_position(dof_ids).clone()

    # Global link ids + group definitions for the list reader.
    link_start = robot.links[0].idx  # entity's links are contiguous in the solver
    robot_links_g = torch.tensor([link.idx for link in robot.links], device=dev)
    ground_links_g = torch.tensor([link.idx for link in ground.links], device=dev)
    feet_links_g = torch.tensor([link.idx for link in robot.links if "ankle_roll" in link.name], device=dev)

    def list_group(link_a, link_b, force, row_valid, links_quat, primary, counterpart):
        """found (B, P) + LOCAL-frame force (B, P, 3) from the contact list."""
        a_is_p = (link_a.unsqueeze(-1) == primary).any(-1)
        b_is_p = (link_b.unsqueeze(-1) == primary).any(-1)
        a_is_c = (link_a.unsqueeze(-1) == counterpart).any(-1)
        b_is_c = (link_b.unsqueeze(-1) == counterpart).any(-1)
        pair = row_valid & ((a_is_p & b_is_c) | (b_is_p & a_is_c))
        pmask_a = (link_a.unsqueeze(-1) == primary) & (pair & b_is_c).unsqueeze(-1)
        pmask_b = (link_b.unsqueeze(-1) == primary) & (pair & a_is_c).unsqueeze(-1)
        found = (pmask_a | pmask_b).any(1)
        # world-frame signed sum: -f on the a side, +f on the b side
        f_world = torch.einsum("ncp,nci->npi", pmask_b.float() - pmask_a.float(), force)
        q = links_quat[:, (primary - link_start)]  # (B, P, 4) — primaries are robot links
        return found, inv_transform_by_quat(f_world, q)

    def read_list():
        cd = solver.collider.get_contacts(as_tensor=True, to_torch=True)
        link_a, link_b, force = cd["link_a"], cd["link_b"], cd["force"]
        n_live = qd_to_torch(solver.collider._collider_state.n_contacts, copy=False)
        row_valid = torch.arange(link_a.shape[1], device=link_a.device)[None, :] < n_live[:, None]
        links_quat = robot.get_links_quat()
        feet = list_group(link_a, link_b, force, row_valid, links_quat, feet_links_g, ground_links_g)
        selfc = list_group(link_a, link_b, force, row_valid, links_quat, robot_links_g, robot_links_g)
        return feet, selfc

    def read_native(contact_sensors, force_sensors):
        found = torch.stack([cs.read_ground_truth()[..., 0] != 0 for cs in contact_sensors], dim=1)
        force = torch.stack([fs.read_ground_truth() for fs in force_sensors], dim=1)
        return found, force

    groups = {
        "feet_ground": (feet_contact, feet_force),
        "self_collision": (self_contact, self_force),
    }
    stats = {
        name: {"frames": 0, "found_mismatch": 0, "found_events": 0, "force_max_diff": 0.0, "force_bad": 0}
        for name in groups
    }
    atol, rtol = 1e-2, 1e-3

    total_ctrl = args.warmup + args.num_steps
    for k in range(total_ctrl):
        drive(k)  # reset anchor every 50 steps + default-pose target
        target = dofs0 + (torch.rand_like(dofs0) * 2.0 - 1.0) * args.flail
        robot.control_dofs_position(target, dof_ids)  # flail overrides the default target
        for _ in range(_DECIMATION):
            scene.step()
            if k < args.warmup:
                continue
            (lf_found, lf_force), (ls_found, ls_force) = read_list()
            for name, (nat_found, nat_force) in (
                ("feet_ground", read_native(*groups["feet_ground"])),
                ("self_collision", read_native(*groups["self_collision"])),
            ):
                li_found, li_force = (lf_found, lf_force) if name == "feet_ground" else (ls_found, ls_force)
                s = stats[name]
                s["frames"] += 1
                s["found_events"] += int(nat_found.sum())
                if not torch.equal(nat_found, li_found):
                    s["found_mismatch"] += int((nat_found != li_found).sum())
                diff = (nat_force - li_force).abs().max()
                s["force_max_diff"] = max(s["force_max_diff"], float(diff))
                if not torch.allclose(nat_force, li_force, atol=atol, rtol=rtol):
                    s["force_bad"] += 1

    lines: list[str] = []
    lines.append("=" * 100)
    lines.append(f"Genesis native-sensor vs contact-list parity — num_envs={args.num_envs} flail={args.flail}")
    lines.append("=" * 100)
    lines.append(f"substep frames compared: {args.num_steps * _DECIMATION} (x {args.num_envs} envs)")
    lines.append(f"force tolerance: atol={atol} rtol={rtol}")
    lines.append("")
    ok = True
    for name, s in stats.items():
        coverage = s["found_events"] / max(s["frames"], 1)
        group_ok = s["found_mismatch"] == 0 and s["force_bad"] == 0
        ok &= group_ok
        lines.append(
            f"[{name}] {'PASS' if group_ok else 'FAIL'}: "
            f"found mismatches {s['found_mismatch']} (bit-exact required), "
            f"force frames out of tol {s['force_bad']}/{s['frames']}, "
            f"force max|diff| {s['force_max_diff']:.4g}, "
            f"contact events/frame {coverage:.1f} (coverage check — must be >0)"
        )
        if coverage == 0:
            ok = False
            lines.append(f"    !! no contact events in group {name} — rollout did not exercise this group")
    lines.append("")
    lines.append(f"OVERALL: {'PASS' if ok else 'FAIL'}")

    report = "\n".join(lines)
    Path(args.out).write_text(report + "\n")
    print()
    print(report)
    print(f"\nReport written to: {Path(args.out).resolve()}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
