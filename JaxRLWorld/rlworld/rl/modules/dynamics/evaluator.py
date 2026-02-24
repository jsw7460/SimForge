from typing import Dict, List, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb


class DynamicsEvaluator:
    """
    Evaluator for dynamics models with multi-step rollout capabilities.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        model_name: str = 'Model',
        use_residual: bool = False,
        max_delta: torch.Tensor | float = 1.0
    ):
        """
        Initialize evaluator.

        Args:
            model: Dynamics model to evaluate
            device: Device to run on
            model_name: Name for logging
        """
        self.model = model
        self.device = device
        self.model_name = model_name
        self.use_residual = use_residual
        self.max_delta = max_delta

    def _predict(self, x: torch.Tensor, action: torch.Tensor,) -> torch.Tensor:
        """Get prediction with residual handling."""
        output = self.model(x, action)
        if self.use_residual:
            max_delta = self.max_delta
            output = torch.clamp(output, -max_delta, max_delta)
            return x + output
        else:
            return output

    # def evaluate_multistep_rollout(
    #     self,
    #     dataset,
    #     input_key: str,
    #     output_key: str,
    #     aux_terms: List[str],
    #     normalize_keys: dict,
    #     horizons: List[int] = [1, 5, 10, 20, 50, 100],
    #     num_episodes: int = 1000,
    #     use_random_starts: bool = True,
    #     return_trajectories: bool = False,
    #     verbose: bool = True
    # ) -> Dict[str, Any]:
    #     """
    #     Evaluate multi-step prediction error with comprehensive statistics.
    #
    #     Args:
    #         dataset: Dataset or Subset with episode_starts and episode_ends attributes
    #         input_key: Key for input data (must match output_key)
    #         output_key: Key for output data (must match input_key)
    #         aux_terms: List of auxiliary term names
    #         normalize_keys: Normalization parameters
    #         horizons: List of prediction horizons to evaluate
    #         num_episodes: Number of episodes to evaluate
    #         return_trajectories: Whether to return full trajectory errors
    #         verbose: Print progress
    #
    #     Returns:
    #         Dictionary containing mean, std, median, percentiles per horizon
    #     """
    #     self.model.eval()
    #
    #     # Handle Subset objects
    #     if hasattr(dataset, 'dataset'):
    #         base_dataset = dataset.dataset
    #         subset_indices = set(dataset.indices)
    #     else:
    #         base_dataset = dataset
    #         subset_indices = None
    #
    #     # Check episode information exists
    #     if not hasattr(base_dataset, 'episode_starts') or not hasattr(base_dataset, 'episode_ends'):
    #         raise ValueError("Dataset must have 'episode_starts' and 'episode_ends' attributes")
    #
    #     max_horizon = max(horizons)
    #     total_samples = len(base_dataset)
    #
    #
    #     # Get valid episode start indices
    #     valid_episode_starts = []
    #     for start, end in zip(base_dataset.episode_starts, base_dataset.episode_ends):
    #         # Check if episode is long enough for max_horizon
    #         if end - start >= max_horizon:
    #             # If using Subset, check if start index is in subset
    #             if subset_indices is None or start in subset_indices:
    #                 valid_episode_starts.append(start)
    #
    #     if len(valid_episode_starts) == 0:
    #         print(f"Warning: No valid episodes for horizon {max_horizon}")
    #         return {
    #             'mean': {h: float('nan') for h in horizons},
    #             'std': {h: float('nan') for h in horizons},
    #             'median': {h: float('nan') for h in horizons},
    #         }
    #
    #     if len(valid_episode_starts) < num_episodes:
    #         if verbose:
    #             print(f"Warning: Only {len(valid_episode_starts)} valid episodes, using all")
    #         num_episodes = len(valid_episode_starts)
    #
    #     # Store errors per horizon
    #     rollout_errors = {h: [] for h in horizons}
    #
    #     # Store full trajectories if requested
    #     if return_trajectories:
    #         trajectory_errors = []
    #
    #     # Get normalization parameters
    #     if input_key in normalize_keys:
    #         mean, std = normalize_keys[input_key]
    #         mean = mean.to(self.device)
    #         std = std.to(self.device)
    #         normalize = True
    #     else:
    #         normalize = False
    #
    #     horizons_set = set(horizons)
    #
    #     with torch.no_grad():
    #         sampled_starts = np.random.choice(
    #             valid_episode_starts,
    #             size=num_episodes,
    #             replace=False
    #         )
    #
    #         for episode_idx, start_idx in enumerate(sampled_starts):
    #             # Get initial state
    #             if not aux_terms:
    #                 x = base_dataset.observations[start_idx:start_idx + 1].to(self.device)
    #             else:
    #                 aux_data = []
    #                 for term in aux_terms:
    #                     aux_data.append(
    #                         base_dataset.auxiliary_obs[term][start_idx:start_idx + 1]
    #                     )
    #                 x = torch.cat(aux_data, dim=-1).to(self.device)
    #
    #             # Apply normalization
    #             if normalize:
    #                 x = (x - mean) / (std + 1e-8)
    #
    #             # Track errors for this trajectory
    #             traj_errors = []
    #
    #             for t in range(max_horizon):
    #                 action = base_dataset.actions[start_idx + t:start_idx + t + 1].to(self.device)
    #
    #                 # Get true next state
    #                 if not aux_terms:
    #                     next_x_true = base_dataset.next_observations[start_idx + t:start_idx + t + 1].to(self.device)
    #                 else:
    #                     next_aux_data = []
    #                     for term in aux_terms:
    #                         next_aux_data.append(
    #                             base_dataset.next_auxiliary_obs[term][start_idx + t:start_idx + t + 1]
    #                         )
    #                     next_x_true = torch.cat(next_aux_data, dim=-1).to(self.device)
    #
    #                 # Apply normalization
    #                 if normalize:
    #                     next_x_true = (next_x_true - mean) / (std + 1e-8)
    #
    #                 # Model prediction
    #                 next_x_pred = self._predict(x, action)
    #
    #                 # Compute error
    #                 error = F.mse_loss(next_x_pred, next_x_true).item()
    #                 traj_errors.append(error)
    #
    #                 # Store error at relevant horizons
    #                 if (t + 1) in horizons_set:
    #                     rollout_errors[t + 1].append(error)
    #
    #                 # Use prediction as next input
    #                 x = next_x_pred
    #
    #             if return_trajectories:
    #                 trajectory_errors.append(traj_errors)
    #
    #             # Progress indicator
    #             if verbose and (episode_idx + 1) % 10 == 0:
    #                 print(f"  Evaluated {episode_idx + 1}/{num_episodes} episodes")
    #
    #     # Compute statistics
    #     results = {
    #         'mean': {},
    #         'std': {},
    #         'median': {},
    #         'p25': {},
    #         'p75': {},
    #         'p90': {},
    #         'p95': {},
    #         'min': {},
    #         'max': {},
    #     }
    #
    #     for h in horizons:
    #         errors = rollout_errors[h]
    #         if len(errors) > 0:
    #             errors_arr = np.array(errors)
    #             results['mean'][h] = np.mean(errors_arr)
    #             results['std'][h] = np.std(errors_arr)
    #             results['median'][h] = np.median(errors_arr)
    #             results['p25'][h] = np.percentile(errors_arr, 25)
    #             results['p75'][h] = np.percentile(errors_arr, 75)
    #             results['p90'][h] = np.percentile(errors_arr, 90)
    #             results['p95'][h] = np.percentile(errors_arr, 95)
    #             results['min'][h] = np.min(errors_arr)
    #             results['max'][h] = np.max(errors_arr)
    #         else:
    #             for key in results:
    #                 results[key][h] = float('nan')
    #
    #     if return_trajectories:
    #         results['trajectories'] = np.array(trajectory_errors)
    #
    #     return results

    def evaluate_multistep_rollout(
        self,
        dataset,
        input_key: str,
        output_key: str,
        aux_terms: List[str],
        normalize_keys: dict,
        horizons: List[int] = [1, 5, 10, 20, 50, 100],
        num_episodes: int = 1000,
        use_random_starts: bool = True,
        return_trajectories: bool = False,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate multi-step prediction error with comprehensive statistics.
        """
        self.model.eval()

        # Handle Subset objects
        if hasattr(dataset, 'dataset'):
            base_dataset = dataset.dataset
        else:
            base_dataset = dataset

        max_horizon = max(horizons)
        total_samples = len(base_dataset)

        # Get valid start indices
        if use_random_starts:
            # 랜덤 시작점: episode 경계를 넘지 않는 인덱스
            valid_starts = []
            for idx in range(total_samples - max_horizon):
                # start부터 max_horizon 내에 done이 없는지 확인
                if not base_dataset.dones[idx:idx + max_horizon].any():
                    valid_starts.append(idx)
        else:
            # 기존: Episode 시작점에서만
            valid_starts = []
            for start, end in zip(base_dataset.episode_starts, base_dataset.episode_ends):
                if end - start >= max_horizon:
                    valid_starts.append(start)

        if len(valid_starts) == 0:
            print(f"Warning: No valid starts for horizon {max_horizon}")
            return {
                'mean': {h: float('nan') for h in horizons},
                'std': {h: float('nan') for h in horizons},
                'median': {h: float('nan') for h in horizons},
            }

        if len(valid_starts) < num_episodes:
            if verbose:
                print(f"Warning: Only {len(valid_starts)} valid starts, using all")
            num_episodes = len(valid_starts)

        # Store errors per horizon
        rollout_errors = {h: [] for h in horizons}

        # Store full trajectories if requested
        if return_trajectories:
            trajectory_errors = []

        # Get normalization parameters
        if input_key in normalize_keys:
            mean, std = normalize_keys[input_key]
            mean = mean.to(self.device)
            std = std.to(self.device)
            normalize = True
        else:
            normalize = False

        horizons_set = set(horizons)

        with torch.no_grad():
            sampled_starts = np.random.choice(
                valid_starts,
                size=num_episodes,
                replace=False
            )

            for episode_idx, start_idx in enumerate(sampled_starts):
                # Get initial state
                if not aux_terms:
                    x = base_dataset.observations[start_idx:start_idx + 1].to(self.device)
                else:
                    aux_data = []
                    for term in aux_terms:
                        aux_data.append(
                            base_dataset.auxiliary_obs[term][start_idx:start_idx + 1]
                        )
                    x = torch.cat(aux_data, dim=-1).to(self.device)

                # Apply normalization
                if normalize:
                    x = (x - mean) / (std + 1e-8)

                # Track errors for this trajectory
                traj_errors = []

                for t in range(max_horizon):
                    action = base_dataset.actions[start_idx + t:start_idx + t + 1].to(self.device)

                    # Get true next state
                    if not aux_terms:
                        next_x_true = base_dataset.next_observations[start_idx + t:start_idx + t + 1].to(self.device)
                    else:
                        next_aux_data = []
                        for term in aux_terms:
                            next_aux_data.append(
                                base_dataset.next_auxiliary_obs[term][start_idx + t:start_idx + t + 1]
                            )
                        next_x_true = torch.cat(next_aux_data, dim=-1).to(self.device)

                    # Apply normalization
                    if normalize:
                        next_x_true = (next_x_true - mean) / (std + 1e-8)

                    # Model prediction
                    next_x_pred = self._predict(x, action)

                    # Compute error
                    error = F.mse_loss(next_x_pred, next_x_true).item()
                    traj_errors.append(error)

                    # Store error at relevant horizons
                    if (t + 1) in horizons_set:
                        rollout_errors[t + 1].append(error)

                    # Use prediction as next input
                    x = next_x_pred

                if return_trajectories:
                    trajectory_errors.append(traj_errors)

                # Progress indicator
                if verbose and (episode_idx + 1) % 10 == 0:
                    print(f"  Evaluated {episode_idx + 1}/{num_episodes} episodes")

        # Compute statistics
        results = {
            'mean': {},
            'std': {},
            'median': {},
            'p25': {},
            'p75': {},
            'p90': {},
            'p95': {},
            'min': {},
            'max': {},
        }

        for h in horizons:
            errors = rollout_errors[h]
            if len(errors) > 0:
                errors_arr = np.array(errors)
                results['mean'][h] = np.mean(errors_arr)
                results['std'][h] = np.std(errors_arr)
                results['median'][h] = np.median(errors_arr)
                results['p25'][h] = np.percentile(errors_arr, 25)
                results['p75'][h] = np.percentile(errors_arr, 75)
                results['p90'][h] = np.percentile(errors_arr, 90)
                results['p95'][h] = np.percentile(errors_arr, 95)
                results['min'][h] = np.min(errors_arr)
                results['max'][h] = np.max(errors_arr)
            else:
                for key in results:
                    results[key][h] = float('nan')

        if return_trajectories:
            results['trajectories'] = np.array(trajectory_errors)

        return results

def print_comparison_results(
    physics_results: Dict[str, float],
    mlp_results: Dict[str, float],
    physics_rollout: Dict[str, Any],
    mlp_rollout: Dict[str, Any]
):
    """
    Print comprehensive comparison results.

    Args:
        physics_results: Physics model single-step results
        mlp_results: MLP model single-step results
        physics_rollout: Physics model rollout results
        mlp_rollout: MLP model rollout results
    """
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    # Single-step performance
    print("\n--- Single-step Prediction (Final Model on Full Val Set) ---")
    print(f"{'Metric':<20} {'Physics':<15} {'MLP':<15} {'Improvement':<12}")
    print("-" * 65)

    p_mean = physics_results['final_mean_loss']
    m_mean = mlp_results['final_mean_loss']
    imp_mean = (m_mean - p_mean) / m_mean * 100
    print(f"{'Mean Loss':<20} {p_mean:<15.6f} {m_mean:<15.6f} {imp_mean:>+10.2f}%")

    p_median = physics_results['final_median_loss']
    m_median = mlp_results['final_median_loss']
    imp_median = (m_median - p_median) / m_median * 100
    print(f"{'Median Loss':<20} {p_median:<15.6f} {m_median:<15.6f} {imp_median:>+10.2f}%")

    p_p90 = physics_results['final_p90']
    m_p90 = mlp_results['final_p90']
    imp_p90 = (m_p90 - p_p90) / m_p90 * 100
    print(f"{'P90':<20} {p_p90:<15.6f} {m_p90:<15.6f} {imp_p90:>+10.2f}%")

    # Multi-step performance - Mean
    print("\n--- Multi-step Rollout Errors (Mean ± Std) ---")
    print(f"{'Horizon':<10} {'Physics':<25} {'MLP':<25} {'Improvement':<12}")
    print("-" * 75)

    for h in sorted(physics_rollout['mean'].keys()):
        p_mean = physics_rollout['mean'][h]
        p_std = physics_rollout['std'][h]
        m_mean = mlp_rollout['mean'][h]
        m_std = mlp_rollout['std'][h]
        imp = (m_mean - p_mean) / m_mean * 100

        print(f"{h:<10} {p_mean:>8.6f} ± {p_std:<8.6f}   "
              f"{m_mean:>8.6f} ± {m_std:<8.6f}   {imp:>+10.2f}%")

    # Multi-step performance - Median & P90
    print("\n--- Multi-step Rollout Errors (Median / P90) ---")
    print(f"{'Horizon':<10} {'Physics':<25} {'MLP':<25} {'Improvement':<12}")
    print("-" * 75)

    for h in sorted(physics_rollout['median'].keys()):
        p_med = physics_rollout['median'][h]
        p_p90 = physics_rollout['p90'][h]
        m_med = mlp_rollout['median'][h]
        m_p90 = mlp_rollout['p90'][h]
        imp_med = (m_med - p_med) / m_med * 100

        print(f"{h:<10} {p_med:>8.6f} / {p_p90:<8.6f}   "
              f"{m_med:>8.6f} / {m_p90:<8.6f}   {imp_med:>+10.2f}%")

    # Worst-case analysis
    print("\n--- Worst-case Performance (P95) ---")
    print(f"{'Horizon':<10} {'Physics':<15} {'MLP':<15} {'Improvement':<12}")
    print("-" * 55)

    for h in sorted(physics_rollout['p95'].keys()):
        p_p95 = physics_rollout['p95'][h]
        m_p95 = mlp_rollout['p95'][h]
        imp_p95 = (m_p95 - p_p95) / m_p95 * 100
        print(f"{h:<10} {p_p95:<15.6f} {m_p95:<15.6f} {imp_p95:>+10.2f}%")


def log_results_to_wandb(
    physics_results: Dict[str, float],
    mlp_results: Dict[str, float],
    physics_rollout: Dict[str, Any],
    mlp_rollout: Dict[str, Any]
):
    """
    Log all results to Weights & Biases.

    Args:
        physics_results: Physics model single-step results
        mlp_results: MLP model single-step results
        physics_rollout: Physics model rollout results
        mlp_rollout: MLP model rollout results
    """

    # Single-step metrics
    physics_final_loss = physics_results['final_mean_loss']
    mlp_final_loss = mlp_results['final_mean_loss']

    wandb.log({
        # Single-step metrics
        "final/physics_final_mean_loss": physics_final_loss,
        "final/mlp_final_mean_loss": mlp_final_loss,
        "final/improvement_pct": (mlp_final_loss - physics_final_loss) / mlp_final_loss * 100,

        "final/physics_final_median_loss": physics_results['final_median_loss'],
        "final/mlp_final_median_loss": mlp_results['final_median_loss'],

        "final/physics_best_val_loss": physics_results['best_val_loss'],
        "final/mlp_best_val_loss": mlp_results['best_val_loss'],
        "final/physics_best_epoch": physics_results['best_epoch'],
        "final/mlp_best_epoch": mlp_results['best_epoch'],
    })

    # Multi-step rollout metrics
    for h in sorted(physics_rollout['mean'].keys()):
        wandb.log({
            f"rollout/horizon_{h}/physics_mean": physics_rollout['mean'][h],
            f"rollout/horizon_{h}/mlp_mean": mlp_rollout['mean'][h],
            f"rollout/horizon_{h}/physics_median": physics_rollout['median'][h],
            f"rollout/horizon_{h}/mlp_median": mlp_rollout['median'][h],
            f"rollout/horizon_{h}/physics_p90": physics_rollout['p90'][h],
            f"rollout/horizon_{h}/mlp_p90": mlp_rollout['p90'][h],
            f"rollout/horizon_{h}/improvement_pct": (
                (mlp_rollout['mean'][h] - physics_rollout['mean'][h]) / mlp_rollout['mean'][h] * 100
            )
        })
