"""Check Genesis plane geom world position.

Print plane's geoms_state.pos and quat after env reset. We suspect Genesis
places the plane at z != 0 (e.g. z ≈ +0.00131), which would mean foot
capsules need to penetrate FURTHER than the analytical lowest_z=0
calculation suggests.
"""

from __future__ import annotations


def main() -> None:
    import genesis as gs

    gs.init(backend=gs.gpu, logging_level="warning")
    from rlworld.rl.configs.presets.g1_29dof.base import G1FlatConfig
    from rlworld.rl.runners import BaseRunner

    cfg = G1FlatConfig(sim_type="genesis", num_envs=16, seed=42)
    cfgs = cfg.build()
    runner = BaseRunner.create_with_env(cfgs)
    env = runner.env
    env.reset()

    rs = env.scene_manager.scene.sim.rigid_solver
    ground_ent = env.scene_manager.terrain.entity

    plane_geom_idx = None
    for link in ground_ent.links:
        for g in link.geoms:
            plane_geom_idx = int(getattr(g, "_idx", getattr(g, "idx", None)))
            break
        if plane_geom_idx is not None:
            break
    print(f"plane_geom_idx: {plane_geom_idx}")

    # Convert qd arrays via the qd_to_torch FUNCTION (not method)
    try:
        from genesis.utils.misc import qd_to_torch  # may or may not exist
    except Exception:
        qd_to_torch = None

    def _as_list(arr, idx):
        try:
            if qd_to_torch is not None:
                t = qd_to_torch(arr)
                return t[idx[0], idx[1]].detach().cpu().tolist()
        except Exception:
            pass
        try:
            import numpy as _np

            t = _np.asarray(arr)
            return t[idx[0], idx[1]].tolist()
        except Exception as e:
            return f"<err: {e!r}>"

    print("plane geoms_state.pos[plane,env=0]:", _as_list(rs.geoms_state.pos, (plane_geom_idx, 0)))
    print("plane geoms_state.quat[plane,env=0]:", _as_list(rs.geoms_state.quat, (plane_geom_idx, 0)))
    print("plane geoms_state.aabb_min[plane,env=0]:", _as_list(rs.geoms_state.aabb_min, (plane_geom_idx, 0)))
    print("plane geoms_state.aabb_max[plane,env=0]:", _as_list(rs.geoms_state.aabb_max, (plane_geom_idx, 0)))

    # Plane entity direct access
    try:
        ent_pos = ground_ent.get_pos()
        print("ground_ent.get_pos():", ent_pos)
    except Exception as e:
        print("ground_ent.get_pos err:", repr(e))

    # Foot capsule positions for reference
    print()
    print("Foot capsule world positions (env 0):")
    robot = env.scene_manager["robot"]
    for link in robot.links:
        if "ankle_roll" not in link.name:
            continue
        for g in link.geoms:
            gi = int(getattr(g, "_idx", getattr(g, "idx", None)))
            try:
                pos = g.get_pos(envs_idx=None)
                p0 = pos[0].detach().cpu().tolist() if hasattr(pos, "__getitem__") else None
                print(f"  geom #{gi} ({link.name}): pos={p0}")
            except Exception as e:
                print(f"  err on geom #{gi}:", repr(e))


if __name__ == "__main__":
    main()
