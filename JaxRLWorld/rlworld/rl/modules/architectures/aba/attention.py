import math
import torch
from torch import nn


class MaskedTransformerLayer(nn.Module):
    """
    Masked Transformer Layer for body-structured attention.

    Similar to Body Transformer, applies masked self-attention where each body
    can only attend to itself and its direct neighbors in the kinematic tree.

    Args:
        embed_dim: Embedding dimension
        nhead: Number of attention heads
        adjacency_matrix: (num_bodies, num_bodies) binary adjacency matrix
        dim_feedforward: Hidden dimension in feedforward network
    """

    def __init__(
        self,
        embed_dim: int,
        nhead: int,
        adjacency_matrix: torch.Tensor,
        dim_feedforward: int = 512,
        use_mask: bool = True
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=nhead,
            batch_first=True
        )

        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, embed_dim)
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self._init_weights()

        if use_mask:
            # Register attention mask
            # mask[i,j] = True means i can attend to j
            # We allow self + direct neighbors
            mask = torch.eye(adjacency_matrix.shape[0]) + adjacency_matrix
            mask = mask.bool()

            # Convert to attention mask format (0 = can attend, -inf = cannot attend)
            attn_mask = torch.zeros_like(mask, dtype=torch.float)
            attn_mask[~mask] = float('-inf')

            self.register_buffer('attn_mask', attn_mask)
        else:
            self.attn_mask = None

    def _init_weights(self):
        modules = [m for m in self.ffn if isinstance(m, nn.Linear)]
        for module in modules[:-1]:
            nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
            nn.init.zeros_(module.bias)
        nn.init.orthogonal_(modules[-1].weight, gain=1.0)
        nn.init.zeros_(modules[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, num_bodies, embed_dim)

        Returns:
            output: (batch, num_bodies, embed_dim)
        """
        # Masked self-attention with residual connection
        attn_out, _ = self.attention(
            x, x, x,
            attn_mask=self.attn_mask,
            need_weights=False
        )
        x = self.norm1(x + attn_out)

        # Feedforward with residual connection
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x


class PairwiseBilinearBias(nn.Module):
    """
    Compute pairwise attention bias using bilinear form.

    bias_ij = f_i @ W @ f_j.T
    """

    def __init__(self, feature_dim: int, num_heads: int = 1):
        super().__init__()
        self.num_heads = num_heads

        if num_heads == 1:
            # Scalar bias: (B, N, N)
            self.W = nn.Parameter(torch.empty(feature_dim, feature_dim))
        else:
            # Per-head bias: (B, H, N, N)
            self.W = nn.Parameter(torch.empty(num_heads, feature_dim, feature_dim))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W, gain=0.1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, num_bodies, feature_dim)

        Returns:
            bias: (B, num_bodies, num_bodies) if num_heads=1
                  (B, num_heads, num_bodies, num_bodies) if num_heads>1
        """
        if self.num_heads == 1:
            # f_i @ W @ f_j.T
            # (B, N, D) @ (D, D) → (B, N, D)
            # (B, N, D) @ (B, D, N) → (B, N, N)
            fW = features @ self.W  # (B, N, D)
            bias = fW @ features.transpose(-1, -2)  # (B, N, N)
        else:
            # Per-head: (H, D, D)
            # (B, N, D) @ (H, D, D) → (B, H, N, D)
            fW = torch.einsum('bnd,hdk->bhnk', features, self.W)
            # (B, H, N, D) @ (B, D, N) → (B, H, N, N)
            bias = torch.einsum('bhnk,bmk->bhnm', fW, features)

        return bias


class DualBiasedAttentionLayer(nn.Module):
    """
    Transformer layer with precomputed attention bias.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dim_feedforward: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Q, K, V projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Feedforward
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            nn.ELU(),
            nn.Linear(dim_feedforward, embed_dim),
        )

        # Layer norms (post-norm)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

        for module in self.ffn:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        precomputed_bias: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, N, _ = x.shape

        Q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Combine bias and mask
        if attn_mask is not None:
            # attn_mask: (N, N), True = cannot attend
            mask_bias = torch.zeros(N, N, device=x.device, dtype=x.dtype)
            mask_bias.masked_fill_(attn_mask, float('-inf'))
            attn_bias = precomputed_bias + mask_bias.unsqueeze(0).unsqueeze(0)
        else:
            attn_bias = precomputed_bias

        # Flash attention with bias
        attn_out = torch.nn.functional.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=attn_bias,
            dropout_p=self.dropout.p if self.training else 0.0,
        )

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, N, self.embed_dim)
        attn_out = self.out_proj(attn_out)

        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        return x
