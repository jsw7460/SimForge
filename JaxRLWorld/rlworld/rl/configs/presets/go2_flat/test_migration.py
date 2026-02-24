"""
Migration test script for go2_flat preset.

Run this to verify that the new component-based config produces
identical output to the old default_dict approach.

Usage:
    python -m rlworld.rl.configs.presets.go2_flat.test_migration
"""

from deepdiff import DeepDiff


def get_old_config():
    """Get config using old default_dict approach."""
    from .default_dict import config as old_config
    import copy

    # Simulate what mlp.py used to do
    config = copy.deepcopy(old_config)
    config["runner"].update({
        "policy_class_name": "PPOActorCritic",
        "run_name": "Go2_MLP",
    })
    return config


def get_new_config():
    """Get config using new component-based approach."""
    from .base import Go2FlatConfig
    return Go2FlatConfig().to_dict(
        actor_class_name="MLPActor",
        run_name="Go2_MLP"
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

    print("=== New Config Summary ===")
    print(f"env.num_envs: {new['env']['num_envs']}")
    print(f"env.episode_length_s: {new['env']['episode_length_s']}")
    print(f"action.num_active_joint_actions: {new['action']['num_active_joint_actions']}")
    print(f"observation.obs_group.actor count: {len(new['observation']['obs_group']['actor'])}")
    print(f"observation.obs_group.critic count: {len(new['observation']['obs_group']['critic'])}")
    print(f"reward.reward_terms count: {len(new['reward']['reward_terms'])}")
    print(f"nn.policy.actor_class_name: {new['nn']['policy']['actor_class_name']}")
    print(f"runner.run_name: {new['runner']['run_name']}")


if __name__ == "__main__":
    print("Testing go2_flat migration...\n")
    print_summary()
    print()
    compare_configs()
