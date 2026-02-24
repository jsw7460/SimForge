"""
Migration test script for go2_flat/newton preset.

Run this to verify that the new component-based config produces
identical output to the old go2_newton approach.

Usage:
    python -m rlworld.rl.configs.presets.go2_flat.newton.test_migration
"""

from deepdiff import DeepDiff


def get_old_config():
    """Get config using old go2_newton approach."""
    from rlworld.rl.configs.presets.go2_newton.mlp import get_config
    return get_config()


def get_new_config():
    """Get config using new component-based approach."""
    from .base import Go2FlatNewtonConfig
    return Go2FlatNewtonConfig().to_dict(
        actor_class_name="MLPActor",
        run_name="Go2_Newton_MLP"
    )


def compare_configs():
    """Compare old and new config outputs."""
    old = get_old_config()
    new = get_new_config()

    diff = DeepDiff(old, new, ignore_order=True)

    if not diff:
        print("SUCCESS: Configs are identical!")
        return True
    else:
        print("DIFFERENCE FOUND:")
        print(diff)
        return False


def print_summary():
    """Print config summary for manual inspection."""
    new = get_new_config()

    print("=== New Newton Config Summary ===")
    print(f"env.num_envs: {new['env'].num_envs}")
    print(f"scene.dt: {new['scene'].dt}")
    print(f"observation.obs_group.actor count: {len(new['observation'].obs_group['actor'])}")
    print(f"observation.obs_group.critic count: {len(new['observation'].obs_group['critic'])}")
    print(f"reward.reward_terms count: {len(new['reward'].reward_terms)}")
    print(f"nn.policy.actor_class_name: {new['nn']['policy']['actor_class_name']}")
    print(f"runner.run_name: {new['runner']['run_name']}")


if __name__ == "__main__":
    print("Testing go2_flat/newton migration...\n")
    print_summary()
    print()
    compare_configs()
