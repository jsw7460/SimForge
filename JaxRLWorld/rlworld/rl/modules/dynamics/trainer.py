"""
Dynamics model trainer with comprehensive metrics and logging.
"""

from typing import Dict, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class DynamicsTrainer:
    """
    Trainer for dynamics models with single-step prediction.

    Handles training loop, validation, and comprehensive metrics collection.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = 'cuda',
        model_name: str = 'Model',
        use_wandb: bool = True,
        clip_grad_norm: Optional[float] = 1.0,
        max_loss_threshold: Optional[float] = 10.0,
        use_residual: bool = False,
        max_delta: torch.Tensor | float = 1.0,
    ):
        """
        Initialize trainer.

        Args:
            model: Dynamics model to train
            optimizer: PyTorch optimizer
            scheduler: Learning rate scheduler (optional)
            device: Device to train on
            model_name: Name for logging
            use_wandb: Whether to log to Weights & Biases
            clip_grad_norm: Gradient clipping threshold (None to disable)
            max_loss_threshold: Skip batches with loss above this (None to disable)
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.model_name = model_name
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.clip_grad_norm = clip_grad_norm
        self.max_loss_threshold = max_loss_threshold
        self.use_residual = use_residual
        self.max_delta = max_delta

        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_epoch = 0

        # History
        self.train_loss_history = []
        self.val_loss_history = []

        self.best_model_state = None

        # ⭐ Define independent metrics for this model
        if self.use_wandb:
            wandb.define_metric(f"{self.model_name}/*")

    def _predict(self, input_data: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Get prediction, handling residual mode.

        Args:
            input_data: Current state
            action: Action

        Returns:
            Predicted next state
        """
        output = self.model(input_data, action)
        max_delta = self.max_delta

        if self.use_residual:
            output = torch.clamp(output, -max_delta, max_delta)
            return input_data + output
        else:
            return output

    def train_epoch(
        self,
        train_loader,
        input_key: str,
        output_key: str
    ) -> float:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = []

        for batch in train_loader:
            input_data = batch[input_key].to(self.device)
            action = batch['action'].to(self.device)
            target = batch[output_key].to(self.device)

            # Skip extreme outliers
            if input_data.abs().max() > 1000 or target.abs().max() > 1000:
                print(f"[{self.model_name}] Skipping batch with extreme values")
                continue

            # Forward pass
            pred = self._predict(input_data, action)
            loss = F.mse_loss(pred, target)

            # Add model-specific losses
            if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'compute_auxiliary_loss'):
                orth_loss = self.model.encoder.compute_auxiliary_loss(input_data)
                loss = loss + orth_loss

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_grad_norm)

            self.optimizer.step()
            epoch_losses.append(loss.item())

            # Log batch loss
            if self.use_wandb:
                wandb.log({
                    f'{self.model_name}/train_loss_batch': loss.item(),
                    f'{self.model_name}/global_step': self.global_step
                })

            self.global_step += 1

        avg_loss = np.mean(epoch_losses) if epoch_losses else float('inf')
        return avg_loss

    def validate(
        self,
        val_loader,
        input_key: str,
        output_key: str
    ) -> float:
        """Validate the model."""
        self.model.eval()
        val_losses = []

        with torch.no_grad():
            for batch in val_loader:
                input_data = batch[input_key].to(self.device)
                action = batch['action'].to(self.device)
                target = batch[output_key].to(self.device)

                pred = self._predict(input_data, action)
                loss = F.mse_loss(pred, target)
                val_losses.append(loss.item())

        avg_loss = np.mean(val_losses) if val_losses else float('inf')
        return avg_loss

    def compute_final_metrics(
        self,
        val_loader,
        input_key: str,
        output_key: str
    ) -> Dict[str, float]:
        """Compute comprehensive validation metrics."""
        print(f"\n[{self.model_name}] Computing final validation metrics...")
        self.model.eval()

        batch_losses = []
        sample_errors = []

        with torch.no_grad():
            for batch in val_loader:
                input_data = batch[input_key].to(self.device)
                action = batch['action'].to(self.device)
                target = batch[output_key].to(self.device)

                pred = self._predict(input_data, action)

                # Batch-level loss
                loss = F.mse_loss(pred, target)
                batch_losses.append(loss.item())

                # Sample-level errors
                per_sample_error = F.mse_loss(pred, target, reduction='none').mean(dim=-1)
                sample_errors.extend(per_sample_error.cpu().numpy())

        # Compute statistics
        mean_loss = np.mean(batch_losses)
        std_loss = np.std(batch_losses)
        median_loss = np.median(batch_losses)

        error_percentiles = np.percentile(sample_errors, [25, 50, 75, 90, 95, 99])

        metrics = {
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'final_mean_loss': mean_loss,
            'final_std_loss': std_loss,
            'final_median_loss': median_loss,
            'final_p25': error_percentiles[0],
            'final_p50': error_percentiles[1],
            'final_p75': error_percentiles[2],
            'final_p90': error_percentiles[3],
            'final_p95': error_percentiles[4],
            'final_p99': error_percentiles[5],
        }

        print(f"[{self.model_name}] Final validation metrics:")
        print(f"  Mean loss: {mean_loss:.6f} ± {std_loss:.6f}")
        print(f"  Median loss: {median_loss:.6f}")
        print(f"  P50/P75/P90/P95: {error_percentiles[1]:.6f} / {error_percentiles[2]:.6f} / "
              f"{error_percentiles[3]:.6f} / {error_percentiles[4]:.6f}")

        if self.use_wandb:
            wandb.log({
                f'{self.model_name}/final_mean_loss': mean_loss,
                f'{self.model_name}/final_std_loss': std_loss,
                f'{self.model_name}/final_median_loss': median_loss,
                f'{self.model_name}/final_p90': error_percentiles[3],
                f'{self.model_name}/final_p95': error_percentiles[4],
            })

        return metrics

    def train(
        self,
        train_loader,
        val_loader,
        input_key: str,
        output_key: str,
        num_epochs: int,
        verbose: bool = True
    ) -> Dict[str, float]:
        """Full training loop."""
        if self.use_residual:
            print(f"[{self.model_name}] Using RESIDUAL prediction mode")

        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader, input_key, output_key)
            self.train_loss_history.append(train_loss)

            val_loss = self.validate(val_loader, input_key, output_key)
            self.val_loss_history.append(val_loss)

            if self.scheduler is not None:
                self.scheduler.step()

            if verbose:
                print(f"[{self.model_name}] Epoch {epoch + 1:3d}/{num_epochs} | "
                      f"Train: {train_loss:.6f} | Val: {val_loss:.6f}")

            if self.use_wandb:
                log_dict = {
                    f'{self.model_name}/train_loss_epoch': train_loss,
                    f'{self.model_name}/val_loss_epoch': val_loss,
                    f'{self.model_name}/epoch': epoch + 1
                }

                if self.scheduler is not None:
                    log_dict[f'{self.model_name}/learning_rate'] = self.optimizer.param_groups[0]['lr']

                wandb.log(log_dict)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch + 1

                self.best_model_state = {
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'use_residual': self.use_residual,  # Save this for inference
                }

                if self.use_wandb:
                    wandb.log({
                        f'{self.model_name}/best_val_loss': self.best_val_loss,
                        f'{self.model_name}/best_epoch': self.best_epoch
                    })

        final_metrics = self.compute_final_metrics(val_loader, input_key, output_key)
        return final_metrics

    def save_checkpoint(self, path: str):
        """Save best model checkpoint"""
        if self.best_model_state is not None:
            torch.save(self.best_model_state, path)
            print(f"[{self.model_name}] Saved checkpoint to {path}")


def create_trainer(
    model: nn.Module,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    num_epochs: int = 100,
    device: str = 'cuda',
    model_name: str = 'Model',
    use_wandb: bool = True,
    use_residual: bool = False,
    max_delta: torch.Tensor | float = 1.0,
) -> DynamicsTrainer:
    """
    Factory function to create a trainer with optimizer and scheduler.

    Args:
        model: Model to train
        learning_rate: Learning rate
        weight_decay: Weight decay
        num_epochs: Number of epochs (for scheduler)
        device: Device
        model_name: Name for logging
        use_wandb: Whether to use WandB
        use_residual: Whether to use residual
        max_delta: Max delta

    Returns:
        Configured DynamicsTrainer
    """
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs
    )

    trainer = DynamicsTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        model_name=model_name,
        use_wandb=use_wandb,
        use_residual=use_residual,
        max_delta=max_delta
    )

    return trainer