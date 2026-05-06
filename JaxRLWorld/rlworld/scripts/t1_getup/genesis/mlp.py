"""Train T1 fall-recovery (getup) policy in Genesis."""

from rlworld.rl.configs.presets.t1_getup.base import T1GetupConfig
from rlworld.rl.runners import BaseRunner


def _dump_genesis_friction(env) -> None:
    """Print per-geom base friction, DR ratio, and effective friction.

    ``env.reset()`` must have run at least once — DR is a reset event,
    so ``geoms_state.friction_ratio`` is all-1.0 before the first reset.
    """
    robot_entity = env.scene_manager["robot"]
    solver = robot_entity._solver

    base_friction = solver.geoms_info.friction.to_numpy()  # (n_geoms,)
    friction_ratio = solver.geoms_state.friction_ratio.to_numpy()  # (n_geoms, n_envs)
    link_idx = solver.geoms_info.link_idx.to_numpy()  # (n_geoms,)

    n_geoms, n_envs = friction_ratio.shape
    ratio_env0 = friction_ratio[:, 0]
    effective_env0 = base_friction * ratio_env0

    robot_geom_start = robot_entity.geom_start
    robot_geom_end = robot_entity.geom_end
    robot_links = {lk.idx: lk.name for lk in robot_entity.links}

    print()
    print("=" * 96)
    print(f"Genesis per-geom friction  (n_geoms={n_geoms}, n_envs={n_envs})")
    print("=" * 96)
    print(f"{'idx':>4} | {'base_mu':>8} | {'ratio[0]':>8} | {'eff[0]':>8} | {'ratio range (env-wide)':>28} | link")
    print("-" * 96)
    for gi in range(n_geoms):
        li = int(link_idx[gi])
        if robot_geom_start <= gi < robot_geom_end:
            link_label = f"robot/{robot_links.get(li, f'link{li}')}"
        else:
            link_label = "ground_or_other"
        lo = float(friction_ratio[gi].min())
        hi = float(friction_ratio[gi].max())
        tag = "(randomized)" if (hi - lo) > 1e-6 else "(const)"
        print(
            f"{gi:>4} | {base_friction[gi]:>8.4f} | {ratio_env0[gi]:>8.4f} | "
            f"{effective_env0[gi]:>8.4f} | [{lo:>.4f}, {hi:>.4f}] {tag:<14} | "
            f"{link_label}"
        )

    print("-" * 96)
    if robot_geom_start > 0:
        ground_base = base_friction[:robot_geom_start].max()
        ground_ratio_env0 = ratio_env0[:robot_geom_start].max()
        ground_eff = ground_base * ground_ratio_env0
        print(f"Ground effective (env 0): {ground_eff:.4f}")
        print("Robot foot geoms vs ground (MAX rule per Genesis contact.py:211):")
        for gi in range(robot_geom_start, robot_geom_end):
            li = int(link_idx[gi])
            name = robot_links.get(li, f"link{li}")
            if "foot" in name.lower():
                eff = float(effective_env0[gi])
                mx = max(eff, ground_eff)
                print(f"  geom {gi:3} ({name}): robot_eff={eff:.4f} ground_eff={ground_eff:.4f} → MAX={mx:.4f}")
    print("=" * 96)
    print()


def main():
    cfgs_for_run = T1GetupConfig(sim_type="genesis").build().with_cli_overrides()

    runner = BaseRunner.create_with_env(cfgs_for_run)

    # # DR이 적용된 상태의 per-geom friction 확인용 진단 블록.
    # # ``runner.env.reset()`` 이 reset 이벤트(= DR)을 한 번 돌려
    # # ``friction_ratio`` 를 env별 랜덤 값으로 채움.
    # runner.env.reset()
    # _dump_genesis_friction(runner.env)
    # ipdb.set_trace()

    runner.learn(
        num_learning_iterations=cfgs_for_run.runner.max_iterations,
        init_at_random_ep_len=cfgs_for_run.runner.init_at_random_ep_len,
    )


if __name__ == "__main__":
    main()
