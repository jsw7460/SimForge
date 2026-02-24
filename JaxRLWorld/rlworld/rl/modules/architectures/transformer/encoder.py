import torch
from torch import nn


class BiasedAttentionLayer(nn.Module):
    """
    Transformer layer with relational embedding bias.
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
            nn.GELU(),
            nn.Linear(dim_feedforward, embed_dim),
        )

        # Layer norms (pre-norm)
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
        re_bias: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, D)
            re_bias: (H, N, N) or (B, H, N, N) relational embedding bias
            attn_mask: (N, N) True = cannot attend

        Returns:
            output: (B, N, D)
        """
        B, N, _ = x.shape

        # Pre-norm
        x_norm = self.norm1(x)

        Q = self.q_proj(x_norm).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x_norm).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x_norm).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Build attention bias
        attn_bias = None

        if re_bias is not None:
            if re_bias.dim() == 3:
                # (H, N, N) -> (1, H, N, N)
                attn_bias = re_bias.unsqueeze(0)
            else:
                # (B, H, N, N) -> keep as is
                attn_bias = re_bias

        if attn_mask is not None:
            mask_bias = torch.zeros(N, N, device=x.device, dtype=x.dtype)
            mask_bias.masked_fill_(attn_mask, float('-inf'))
            mask_bias = mask_bias.unsqueeze(0).unsqueeze(0)  # (1, 1, N, N)

            if attn_bias is not None:
                attn_bias = attn_bias + mask_bias
            else:
                attn_bias = mask_bias

        # Scaled dot-product attention
        attn_out = torch.nn.functional.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=attn_bias,
            dropout_p=self.dropout.p if self.training else 0.0,
        )

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, N, self.embed_dim)
        attn_out = self.out_proj(attn_out)

        # Residual
        x = x + self.dropout(attn_out)

        # FFN with pre-norm and residual
        x = x + self.dropout(self.ffn(self.norm2(x)))

        return x
