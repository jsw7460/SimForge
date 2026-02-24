from abc import ABC, abstractmethod

import torch
from torch import nn


class Tokenizer(nn.Module, ABC):
    """Base class for observation tokenizers"""

    @property
    @abstractmethod
    def output_dim(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Args:
            observations: (B, obs_dim)
        Returns:
            tokens: (B, N, D)
        """
        raise NotImplementedError


class MLPTokenizer(Tokenizer):
    """Per-body MLP tokenizer"""

    def __init__(
        self,
        num_bodies: int,
        obs_dim: int,
        embed_dim: int,
        hidden_mult: int = 2,
    ):
        super().__init__()
        self.num_bodies = num_bodies
        self.embed_dim = embed_dim

        hidden_dim = embed_dim * hidden_mult
        self.tokenizers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, embed_dim),
            )
            for _ in range(num_bodies)
        ])

        self._init_weights()

    def _init_weights(self):
        for tokenizer in self.tokenizers:
            for module in tokenizer:
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight, gain=1.0)
                    nn.init.zeros_(module.bias)

    @property
    def output_dim(self) -> int:
        return self.embed_dim

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        tokens = torch.stack([
            tokenizer(observations) for tokenizer in self.tokenizers
        ], dim=1)
        return tokens  # (B, N, D)
