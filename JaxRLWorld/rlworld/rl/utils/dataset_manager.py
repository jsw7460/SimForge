import os
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import torch

from rlworld.rl.utils.dynamics_dataset import DynamicsDataset

if TYPE_CHECKING:
    from rlworld.rl.runners import BaseRunner

# Color codes
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"


class DatasetCheckpointHandler:
    """Handles dataset saving and loading with runner state."""

    @staticmethod
    def save_dataset_checkpoint(
        runner: "BaseRunner",
        dataset: DynamicsDataset,
        path: str,
        include_policy: bool = True
    ) -> None:
        """
        Save dataset along with runner configuration and optionally policy state.

        Args:
            runner: The runner instance
            dataset: Dataset to save
            path: Path to save checkpoint
            include_policy: Whether to include policy weights
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        # Build checkpoint dictionary
        checkpoint = {
            # Dataset
            'dataset': {
                'observations': dataset.observations,
                'actions': dataset.actions,
                'next_observations': dataset.next_observations,
                'auxiliary_obs': dataset.auxiliary_obs,
                'next_auxiliary_obs': dataset.next_auxiliary_obs,
                'dones': dataset.dones,
                'metadata': dataset.metadata,
                'size': len(dataset),
                'obs_dim': dataset.get_obs_dim(),
                'action_dim': dataset.get_action_dim()
            },

            # Runner information
            'runner_class': runner.__class__.__name__,
            'configs': runner.cfgs.recursive_to_dict(),

            # Training state at collection time
            'collection_info': {
                'iteration': runner.current_learning_iteration,
                'total_timesteps': runner.total_timesteps,
                'total_time': runner.total_time,
            },

            # Environment info
            'env_info': {
                'num_envs': runner.env.num_envs,
                'max_episode_length': runner.env.max_episode_length,
                'obs_dim': runner.env.obs_manager.calculate_obs_dim(),
                'action_dim': runner.env.num_actions,
            }
        }

        # Optionally include policy state
        if include_policy:
            checkpoint['policy_state'] = {
                name: module.state_dict()
                for name, module in runner.training_modules.items()
            }
            print(f"{GREEN}Including policy state in checkpoint{RESET}")

        # Save
        try:
            torch.save(checkpoint, path)
            print(f"{GREEN}✓ Successfully saved dataset checkpoint to {path}{RESET}")
            print(f"  - Dataset size: {len(dataset)} transitions")
            print(f"  - Obs dim: {dataset.get_obs_dim()}, Action dim: {dataset.get_action_dim()}")

            if dataset.auxiliary_obs:
                print(f"  - Auxiliary terms: {list(dataset.auxiliary_obs.keys())}")

            print(f"  - Policy included: {include_policy}")
        except Exception as e:
            print(f"{RED}✗ Failed to save dataset checkpoint: {e}{RESET}")
            raise

    @staticmethod
    def load_dataset_checkpoint(
        path: str,
        load_policy: bool = False,
        runner: Optional["BaseRunner"] = None
    ) -> tuple[DynamicsDataset, dict]:
        """
        Load dataset checkpoint.

        Args:
            path: Path to checkpoint file
            load_policy: Whether to load policy weights (requires runner)
            runner: Runner instance (required if load_policy=True)

        Returns:
            Tuple of (dataset, info_dict)
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

        print(f"\n{GREEN}Loading dataset checkpoint from {path}...{RESET}")

        # Load checkpoint
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)

        # Reconstruct dataset
        dataset_data = checkpoint['dataset']
        dataset = DynamicsDataset(
            observations=dataset_data['observations'],
            actions=dataset_data['actions'],
            next_observations=dataset_data['next_observations'],
            auxiliary_obs=dataset_data.get('auxiliary_obs', {}),  # ⭐ 추가
            next_auxiliary_obs=dataset_data.get('next_auxiliary_obs', {}),  # ⭐ 추가
            metadata=dataset_data.get('metadata', {})
        )

        print(f"{GREEN}✓ Loaded dataset:{RESET}")
        print(f"  - Size: {len(dataset)} transitions")
        print(f"  - Obs dim: {dataset.get_obs_dim()}, Action dim: {dataset.get_action_dim()}")

        # ⭐ Auxiliary info 추가
        if dataset.auxiliary_obs:
            print(f"  - Auxiliary terms: {list(dataset.auxiliary_obs.keys())}")
            for term_name, tensor in dataset.auxiliary_obs.items():
                print(f"    * {term_name}: {tensor.shape}")

        # Load policy if requested
        if load_policy:
            if runner is None:
                raise ValueError("Runner must be provided to load policy state")

            if 'policy_state' not in checkpoint:
                print(f"{YELLOW}⚠ No policy state found in checkpoint{RESET}")
            else:
                print(f"\n{GREEN}Loading policy state...{RESET}")
                policy_state = checkpoint['policy_state']

                for name, state_dict in policy_state.items():
                    if name in runner.training_modules:
                        try:
                            runner.training_modules[name].load_state_dict(state_dict)
                            print(f"{GREEN}✓ Loaded policy module: {name}{RESET}")
                        except Exception as e:
                            print(f"{RED}✗ Failed to load module {name}: {e}{RESET}")
                    else:
                        print(f"{YELLOW}⚠ Module {name} not found in runner{RESET}")

        # Prepare info dictionary
        info = {
            'runner_class': checkpoint.get('runner_class'),
            'configs': checkpoint.get('configs'),
            'collection_info': checkpoint.get('collection_info'),
            'env_info': checkpoint.get('env_info')
        }

        return dataset, info
