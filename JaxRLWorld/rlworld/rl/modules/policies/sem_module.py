import torch
import torch.nn as nn


class SEM(nn.Module):
    """
    Simplicial Embeddings Module that applies group-wise softmax normalization.

    This module transforms features into a product of simplices by:
    1. Projecting hidden_dim to L*V dimensions
    2. Reshaping into L groups of size V
    3. Applying temperature-scaled softmax within each group
    4. Flattening and projecting back to hidden_dim

    Args:
        hidden_dim (int): Hidden dimension size
        L (int): Number of simplices (groups)
        V (int): Vocabulary size per simplex
        tau (float): Temperature parameter controlling sparsity
    """

    def __init__(self, hidden_dim: int, L: int, V: int, tau: float = 0.5):
        super().__init__()
        self.L = L
        self.V = V
        self.tau = tau
        self.hidden_dim = hidden_dim

        # Projection from hidden_dim to L*V
        self.projection = nn.Linear(hidden_dim, L * V)

        # Projection from L*V back to hidden_dim
        self.output = nn.Linear(L * V, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SEM transformation.

        Args:
            x: Input tensor of shape (batch_size, hidden_dim)

        Returns:
            Transformed tensor of shape (batch_size, hidden_dim)
        """
        # Project hidden_dim -> L*V
        z = self.projection(x)

        # Reshape to L groups of V dimensions
        z = z.view(-1, self.L, self.V)

        # Apply temperature-scaled softmax
        z = z / self.tau
        z_tilde = torch.softmax(z, dim=-1)

        # Flatten back
        z_tilde = z_tilde.view(-1, self.L * self.V)

        # Project back to hidden_dim
        output = self.output(z_tilde)

        return output
