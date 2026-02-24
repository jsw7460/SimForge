from typing import Optional, Dict, Any, List

import torch

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"


class DynamicsDataset:
    """
    Dataset for dynamics learning with auxiliary observations.

    Attributes:
        observations: (N, obs_dim) - Policy observations
        actions: (N, action_dim)
        next_observations: (N, obs_dim) - Policy next observations
        dones: (N,) - Episode termination flags
        auxiliary_obs: Dict[str, Tensor] - Auxiliary observations (NOT in policy obs)
        next_auxiliary_obs: Dict[str, Tensor] - Next auxiliary observations
        metadata: Additional information about dataset
        episode_starts: List of episode start indices
        episode_ends: List of episode end indices
    """

    def __init__(
        self,
        observations: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        next_observations: Optional[torch.Tensor] = None,
        dones: Optional[torch.Tensor] = None,
        auxiliary_obs: Optional[Dict[str, torch.Tensor]] = None,
        next_auxiliary_obs: Optional[Dict[str, torch.Tensor]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize dataset.

        Args:
            observations: Policy observation tensor
            actions: Action tensor
            next_observations: Next policy observation tensor
            dones: Episode termination flags
            auxiliary_obs: Dict mapping term names to auxiliary observation tensors
            next_auxiliary_obs: Dict mapping term names to next auxiliary observation tensors
            metadata: Additional metadata
        """
        if observations is not None:
            assert observations.shape[0] == actions.shape[0] == next_observations.shape[0], \
                "All tensors must have the same number of samples"

            self.observations = observations.cpu()
            self.actions = actions.cpu()
            self.next_observations = next_observations.cpu()

            # Handle dones
            if dones is not None:
                assert dones.shape[0] == observations.shape[0], \
                    f"Dones has wrong size: {dones.shape[0]} vs {observations.shape[0]}"
                self.dones = dones.cpu()
            else:
                # If no dones provided, assume no episode terminations
                self.dones = torch.zeros(observations.shape[0], dtype=torch.bool)
        else:
            # Empty dataset
            self.observations = torch.tensor([])
            self.actions = torch.tensor([])
            self.next_observations = torch.tensor([])
            self.dones = torch.tensor([], dtype=torch.bool)

        # Store auxiliary observations
        self.auxiliary_obs = auxiliary_obs or {}
        self.next_auxiliary_obs = next_auxiliary_obs or {}

        # Validate auxiliary observations have correct size
        if auxiliary_obs:
            for key, tensor in auxiliary_obs.items():
                assert tensor.shape[0] == self.observations.shape[0], \
                    f"Auxiliary obs '{key}' has wrong size: {tensor.shape[0]} vs {self.observations.shape[0]}"

        if next_auxiliary_obs:
            for key, tensor in next_auxiliary_obs.items():
                assert tensor.shape[0] == self.observations.shape[0], \
                    f"Next auxiliary obs '{key}' has wrong size: {tensor.shape[0]} vs {self.observations.shape[0]}"

        self.metadata = metadata or {}

        # Extract episode boundaries from dones
        self.episode_starts, self.episode_ends = self._extract_episode_boundaries()

    def _extract_episode_boundaries(self) -> tuple[List[int], List[int]]:
        """Extract episode start and end indices from done flags"""
        if len(self.dones) == 0:
            return [], []

        episode_starts = [0]
        episode_ends = []

        done_indices = torch.where(self.dones)[0].tolist()

        for done_idx in done_indices:
            episode_ends.append(done_idx + 1)  # End is exclusive
            if done_idx + 1 < len(self.dones):
                episode_starts.append(done_idx + 1)

        # Handle last episode if not terminated
        if len(episode_ends) < len(episode_starts):
            episode_ends.append(len(self.dones))

        return episode_starts, episode_ends

    def has_episode_info(self) -> bool:
        """Check if dataset has episode boundary information"""
        return len(self.episode_starts) > 0 and len(self.episode_ends) > 0

    def get_num_episodes(self) -> int:
        """Get number of episodes in dataset"""
        return len(self.episode_starts)

    def __len__(self) -> int:
        """Return number of transitions in dataset"""
        return self.observations.shape[0] if len(self.observations.shape) > 0 else 0

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single transition with auxiliary data.

        Returns:
            Dictionary with keys:
            - 'obs': Policy observation
            - 'action': Action
            - 'next_obs': Next policy observation
            - 'done': Episode termination flag
            - 'aux_{name}': Auxiliary observation (for each auxiliary term)
            - 'next_aux_{name}': Next auxiliary observation
        """
        item = {
            'obs': self.observations[idx],
            'action': self.actions[idx],
            'next_obs': self.next_observations[idx],
            'done': self.dones[idx]
        }

        # Add auxiliary observations with 'aux_' prefix
        for key, tensor in self.auxiliary_obs.items():
            item[f'aux_{key}'] = tensor[idx]

        for key, tensor in self.next_auxiliary_obs.items():
            item[f'next_aux_{key}'] = tensor[idx]

        return item

    def get_obs_dim(self) -> int:
        """Get policy observation dimension"""
        return self.observations.shape[1] if len(self) > 0 else 0

    def get_action_dim(self) -> int:
        """Get action dimension"""
        return self.actions.shape[1] if len(self) > 0 else 0

    def get_auxiliary_term_dim(self, term_name: str) -> int:
        """Get dimension of a specific auxiliary term"""
        if term_name in self.auxiliary_obs:
            return self.auxiliary_obs[term_name].shape[1]
        return 0

    def has_auxiliary_term(self, term_name: str) -> bool:
        """Check if dataset has a specific auxiliary term"""
        return term_name in self.auxiliary_obs

    def list_auxiliary_terms(self) -> List[str]:
        """List all available auxiliary terms"""
        return list(self.auxiliary_obs.keys())

    def filter_outliers(
        self,
        percentile: float = 99.0,
        method: str = 'delta_norm'
    ) -> 'DynamicsDataset':
        """
        Filter outliers and return new DynamicsDataset.

        Args:
            percentile: Percentile threshold (keep bottom percentile%)
            method: 'delta_norm', 'any_field', or 'obs_norm'

        Returns:
            New filtered DynamicsDataset
        """
        if method == 'delta_norm':
            deltas = self.next_observations - self.observations
            delta_norm = deltas.norm(dim=-1)
            threshold = delta_norm.quantile(percentile / 100)
            mask = delta_norm <= threshold

        elif method == 'any_field':
            obs_abs = self.observations.abs()
            actions_abs = self.actions.abs()
            deltas = self.next_observations - self.observations
            deltas_abs = deltas.abs()

            obs_threshold = obs_abs.quantile(percentile / 100)
            action_threshold = actions_abs.quantile(percentile / 100)
            delta_threshold = deltas_abs.quantile(percentile / 100)

            mask = (
                (obs_abs <= obs_threshold).all(dim=-1) &
                (actions_abs <= action_threshold).all(dim=-1) &
                (deltas_abs <= delta_threshold).all(dim=-1)
            )

        elif method == 'obs_norm':
            obs_norm = self.observations.norm(dim=-1)
            threshold = obs_norm.quantile(percentile / 100)
            mask = obs_norm <= threshold

        else:
            raise ValueError(f"Unknown method: {method}")

        # Filter all data
        filtered_obs = self.observations[mask]
        filtered_actions = self.actions[mask]
        filtered_next_obs = self.next_observations[mask]
        filtered_dones = self.dones[mask]  # Filter dones too

        # Filter auxiliary observations
        filtered_auxiliary_obs = {}
        for key, tensor in self.auxiliary_obs.items():
            filtered_auxiliary_obs[key] = tensor[mask]

        filtered_next_auxiliary_obs = {}
        for key, tensor in self.next_auxiliary_obs.items():
            filtered_next_auxiliary_obs[key] = tensor[mask]

        # Statistics
        n_removed = (~mask).sum().item()
        n_total = len(self)
        pct_removed = n_removed / n_total * 100

        print(f"{YELLOW}Filtered {n_removed:,} / {n_total:,} samples ({pct_removed:.2f}%){RESET}")
        print(f"{GREEN}Remaining: {mask.sum().item():,} samples{RESET}")

        # Create new dataset
        return DynamicsDataset(
            observations=filtered_obs,
            actions=filtered_actions,
            next_observations=filtered_next_obs,
            dones=filtered_dones,
            auxiliary_obs=filtered_auxiliary_obs,
            next_auxiliary_obs=filtered_next_auxiliary_obs,
            metadata=self.metadata.copy()
        )

    def analyze_outliers(
        self,
        percentiles: List[float] = [90, 95, 99, 99.5, 99.9],
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze outlier distribution in dataset.

        Args:
            percentiles: Percentiles to compute
            verbose: Print detailed statistics

        Returns:
            Dictionary containing outlier statistics
        """
        results = {
            'observations': {},
            'actions': {},
            'next_observations': {},
            'deltas': {}
        }

        if verbose:
            print("\n" + "=" * 70)
            print("DATASET OUTLIER ANALYSIS")
            print("=" * 70)
            print(f"Total samples: {len(self):,}")
            print(f"Episodes: {self.get_num_episodes()}")
            print(f"Episode ends: {self.dones.sum().item()}\n")

        # Analyze observations
        obs = self.observations
        obs_abs = obs.abs()
        obs_norm = obs.norm(dim=-1)

        if verbose:
            print("--- Observations ---")
            print(f"Shape: {obs.shape}")
            print(f"Range: [{obs.min():.6f}, {obs.max():.6f}]")
            print(f"Mean: {obs.mean():.6f}, Std: {obs.std():.6f}")

        results['observations']['shape'] = obs.shape
        results['observations']['min'] = obs.min().item()
        results['observations']['max'] = obs.max().item()
        results['observations']['mean'] = obs.mean().item()
        results['observations']['std'] = obs.std().item()
        results['observations']['percentiles'] = {}

        if verbose:
            print("\nPer-element absolute value percentiles:")
        for p in percentiles:
            val = obs_abs.quantile(p / 100).item()
            results['observations']['percentiles'][p] = val
            if verbose:
                print(f"  P{p:>5.1f}: {val:.6f}")

        if verbose:
            print("\nPer-sample L2 norm percentiles:")
        for p in percentiles:
            val = obs_norm.quantile(p / 100).item()
            if verbose:
                print(f"  P{p:>5.1f}: {val:.6f}")

        # Count extreme outliers
        for threshold_p in [99, 99.5, 99.9]:
            threshold = obs_abs.quantile(threshold_p / 100)
            outlier_mask = (obs_abs > threshold).any(dim=-1)
            n_outliers = outlier_mask.sum().item()
            pct = n_outliers / len(self) * 100

            if verbose and threshold_p == 99:
                print(f"\nSamples with any element > P{threshold_p} ({threshold:.6f}):")
                print(f"  {n_outliers:,} / {len(self):,} ({pct:.2f}%)")

        # Analyze actions
        actions = self.actions
        actions_abs = actions.abs()

        if verbose:
            print("\n--- Actions ---")
            print(f"Shape: {actions.shape}")
            print(f"Range: [{actions.min():.6f}, {actions.max():.6f}]")
            print(f"Mean: {actions.mean():.6f}, Std: {actions.std():.6f}")

        results['actions']['shape'] = actions.shape
        results['actions']['min'] = actions.min().item()
        results['actions']['max'] = actions.max().item()
        results['actions']['mean'] = actions.mean().item()
        results['actions']['std'] = actions.std().item()
        results['actions']['percentiles'] = {}

        if verbose:
            print("\nPer-element absolute value percentiles:")
        for p in percentiles:
            val = actions_abs.quantile(p / 100).item()
            results['actions']['percentiles'][p] = val
            if verbose:
                print(f"  P{p:>5.1f}: {val:.6f}")

        # Analyze deltas (next_obs - obs)
        deltas = self.next_observations - self.observations
        deltas_abs = deltas.abs()
        deltas_norm = deltas.norm(dim=-1)

        if verbose:
            print("\n--- State Deltas (next_obs - obs) ---")
            print(f"Range: [{deltas.min():.6f}, {deltas.max():.6f}]")
            print(f"Mean: {deltas.mean():.6f}, Std: {deltas.std():.6f}")

        results['deltas']['min'] = deltas.min().item()
        results['deltas']['max'] = deltas.max().item()
        results['deltas']['mean'] = deltas.mean().item()
        results['deltas']['std'] = deltas.std().item()
        results['deltas']['percentiles'] = {}

        if verbose:
            print("\nPer-element absolute value percentiles:")
        for p in percentiles:
            val = deltas_abs.quantile(p / 100).item()
            results['deltas']['percentiles'][p] = val
            if verbose:
                print(f"  P{p:>5.1f}: {val:.6f}")

        if verbose:
            print("\nPer-sample L2 norm percentiles:")
        for p in percentiles:
            val = deltas_norm.quantile(p / 100).item()
            if verbose:
                print(f"  P{p:>5.1f}: {val:.6f}")

        # Joint analysis: samples with extreme values in any field
        if verbose:
            print("\n--- Joint Outlier Analysis ---")

        for threshold_p in [99, 99.5, 99.9]:
            obs_threshold = obs_abs.quantile(threshold_p / 100)
            action_threshold = actions_abs.quantile(threshold_p / 100)
            delta_threshold = deltas_abs.quantile(threshold_p / 100)

            outlier_mask = (
                (obs_abs > obs_threshold).any(dim=-1) |
                (actions_abs > action_threshold).any(dim=-1) |
                (deltas_abs > delta_threshold).any(dim=-1)
            )

            n_outliers = outlier_mask.sum().item()
            pct = n_outliers / len(self) * 100

            results[f'joint_outliers_p{threshold_p}'] = {
                'count': n_outliers,
                'percentage': pct
            }

            if verbose:
                print(f"\nSamples with ANY field > P{threshold_p}:")
                print(f"  {n_outliers:,} / {len(self):,} ({pct:.2f}%)")

        # Per-dimension analysis (find which dimensions have outliers)
        if verbose:
            print("\n--- Per-Dimension Outlier Count (P99) ---")
            obs_p99 = obs_abs.quantile(0.99, dim=0)
            outlier_counts = (obs_abs > obs_p99).sum(dim=0)

            print("Observation dimensions with most outliers:")
            top_dims = torch.argsort(outlier_counts, descending=True)[:5]
            for rank, dim_idx in enumerate(top_dims):
                count = outlier_counts[dim_idx].item()
                pct = count / len(self) * 100
                print(f"  Dim {dim_idx:2d}: {count:,} samples ({pct:.2f}%)")

        if verbose:
            print("\n" + "=" * 70)

        return results

    def save(self, path: str) -> None:
        """Save dataset to file"""
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            'observations': self.observations,
            'actions': self.actions,
            'next_observations': self.next_observations,
            'dones': self.dones,
            'auxiliary_obs': self.auxiliary_obs,
            'next_auxiliary_obs': self.next_auxiliary_obs,
            'metadata': self.metadata,
            'dataset_size': len(self),
            'obs_dim': self.get_obs_dim(),
            'action_dim': self.get_action_dim()
        }

        torch.save(save_dict, path)
        print(f"{GREEN}Saved dataset ({len(self)} transitions) to {path}{RESET}")
        print(f"  - Episodes: {self.get_num_episodes()}")
        print(f"  - Episode ends: {self.dones.sum().item()}")
        if self.auxiliary_obs:
            print(f"  - Auxiliary terms: {list(self.auxiliary_obs.keys())}")

    @classmethod
    def load(cls, path: str) -> 'DynamicsDataset':
        """Load dataset from file (supports both old and new formats)"""
        import os
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset file not found: {path}")

        data = torch.load(path, map_location='cpu', weights_only=False)

        # Handle backward compatibility (old format with 'dataset' wrapper)
        if 'dataset' in data:
            observations = data['dataset']['observations']
            actions = data['dataset']['actions']
            next_observations = data['dataset']['next_observations']
            dones = data['dataset']['dones']
            metadata = data['dataset'].get('metadata', {})
            auxiliary_obs = data['dataset'].get('auxiliary_obs', {})
            next_auxiliary_obs = data['dataset'].get('next_auxiliary_obs', {})
        else:
            # New format (flat structure)
            observations = data['observations']
            actions = data['actions']
            next_observations = data['next_observations']
            dones = data['dones']
            metadata = data.get('metadata', {})
            auxiliary_obs = data.get('auxiliary_obs', {})
            next_auxiliary_obs = data.get('next_auxiliary_obs', {})

        dataset = cls(
            observations=observations.cpu(),
            actions=actions.cpu(),
            next_observations=next_observations.cpu(),
            dones=dones.cpu() if dones is not None else None,
            auxiliary_obs=auxiliary_obs,
            next_auxiliary_obs=next_auxiliary_obs,
            metadata=metadata
        )

        print(f"{GREEN}Loaded dataset from {path}:{RESET}")
        print(f"  - Size: {len(dataset)} transitions")
        print(f"  - Episodes: {dataset.get_num_episodes()}")
        print(f"  - Episode ends: {dataset.dones.sum().item()}")
        print(f"  - Obs dim: {dataset.get_obs_dim()}")
        print(f"  - Action dim: {dataset.get_action_dim()}")
        print(f"  - Device: {dataset.observations.device}")
        if dataset.auxiliary_obs:
            print(f"  - Auxiliary terms: {list(dataset.auxiliary_obs.keys())}")
            for term_name, tensor in dataset.auxiliary_obs.items():
                print(f"    * {term_name}: {tensor.shape}")
        if dataset.metadata:
            print(f"  - Metadata: {list(dataset.metadata.keys())}")

        return dataset


def compute_auxiliary_dimensions(dataset, aux_terms: List[str]) -> int:
    """
    Compute total dimension for multiple auxiliary terms.

    Args:
        dataset: DynamicsDataset
        aux_terms: List of term names (e.g., ['dof_pos', 'dof_vel', 'dof_acc'])

    Returns:
        Total dimension (sum of all terms)
    """
    total_dim = 0
    for term in aux_terms:
        term_dim = dataset.get_auxiliary_term_dim(term)
        total_dim += term_dim
        print(f"  {term}: {term_dim}")

    print(f"  Total: {total_dim}")
    return total_dim


def compute_normalization_stats(dataset, aux_terms: List[str]) -> dict:
    """
    Compute normalization statistics for auxiliary terms.

    Args:
        dataset: DynamicsDataset
        aux_terms: List of term names

    Returns:
        Dict mapping keys to (mean, std) tuples
    """
    all_data = []
    all_next_data = []

    print("Per-term statistics:")
    for term in aux_terms:
        data = dataset.auxiliary_obs[term]
        next_data = dataset.next_auxiliary_obs[term]

        # Clip outliers at 1% and 99% percentiles
        q01 = data.quantile(0.01)
        q99 = data.quantile(0.99)
        data_clipped = data.clamp(q01, q99)
        next_data_clipped = next_data.clamp(q01, q99)

        all_data.append(data_clipped)
        all_next_data.append(next_data_clipped)

        print(f"  {term} raw range: [{data.min():.2f}, {data.max():.2f}]")
        print(f"  {term} clipped: [{q01:.2f}, {q99:.2f}]")

    # Concatenate along feature dimension
    combined_data = torch.cat(all_data, dim=-1)
    combined_next_data = torch.cat(all_next_data, dim=-1)

    # Compute statistics
    mean = combined_data.mean(dim=0)
    std = combined_data.std(dim=0)

    print(f"\nCombined statistics:")
    print(f"  Mean: {mean.mean():.2f}, Std: {std.mean():.2f}")

    # Create normalization dict
    input_key = f'aux_{"_".join(aux_terms)}'
    output_key = f'next_aux_{"_".join(aux_terms)}'

    return {
        input_key: (mean, std),
        output_key: (mean, std)
    }


class AuxiliaryBatchLoader:
    """Batch loader that handles multiple auxiliary terms or observations"""

    def __init__(
        self,
        dataset,
        batch_size: int,
        aux_terms: List[str] = None,
        shuffle: bool = True,
        normalize_keys: dict = None
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.aux_terms = aux_terms or []
        self.shuffle = shuffle
        self.normalize_keys = normalize_keys or {}

        # Handle Subset objects
        if hasattr(dataset, 'dataset'):
            self.base_dataset = dataset.dataset
            self.indices = list(dataset.indices)
        else:
            self.base_dataset = dataset
            self.indices = list(range(len(dataset)))

    def __iter__(self):
        if self.shuffle:
            import random
            random.shuffle(self.indices)

        for i in range(0, len(self.indices), self.batch_size):
            batch_indices = self.indices[i:i + self.batch_size]

            # Build batch
            batch = {}

            # Actions (always needed)
            batch['action'] = torch.stack([
                self.base_dataset.actions[idx] for idx in batch_indices
            ])

            # Input/Output data
            if not self.aux_terms:
                # Use observations
                batch['obs'] = torch.stack([
                    self.base_dataset.observations[idx] for idx in batch_indices
                ])
                batch['next_obs'] = torch.stack([
                    self.base_dataset.next_observations[idx] for idx in batch_indices
                ])
            else:
                # Use auxiliary terms
                aux_data = []
                next_aux_data = []

                for term in self.aux_terms:
                    term_data = torch.stack([
                        self.base_dataset.auxiliary_obs[term][idx] for idx in batch_indices
                    ])
                    next_term_data = torch.stack([
                        self.base_dataset.next_auxiliary_obs[term][idx] for idx in batch_indices
                    ])

                    aux_data.append(term_data)
                    next_aux_data.append(next_term_data)

                # Concatenate along feature dimension
                input_key = f'aux_{"_".join(self.aux_terms)}'
                output_key = f'next_aux_{"_".join(self.aux_terms)}'

                batch[input_key] = torch.cat(aux_data, dim=-1)
                batch[output_key] = torch.cat(next_aux_data, dim=-1)

            # Apply normalization
            for key in batch.keys():
                if key in self.normalize_keys:
                    mean, std = self.normalize_keys[key]
                    batch[key] = (batch[key] - mean) / (std + 1e-8)

            yield batch

    def __len__(self):
        return (len(self.indices) + self.batch_size - 1) // self.batch_size