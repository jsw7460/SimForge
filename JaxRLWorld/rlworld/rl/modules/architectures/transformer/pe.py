from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from rlworld.rl.envs.managers.scene_manager import KinematicTree


class PositionalEmbedding(nn.Module, ABC):
    """Base class for positional embeddings"""

    @abstractmethod
    def forward(self) -> torch.Tensor:
        """
        Returns:
            pe: (N, D)
        """
        raise NotImplementedError


class LearnedPositionalEmbedding(PositionalEmbedding):
    """Simple learnable positional embedding"""

    def __init__(self, num_bodies: int, embed_dim: int):
        super().__init__()
        self.num_bodies = num_bodies
        self.embedding = nn.Embedding(num_bodies, embed_dim)

        # Pre-compute indices as buffer
        self.register_buffer('indices', torch.arange(num_bodies))

    def forward(self) -> torch.Tensor:
        return self.embedding(self.indices)  # (N, D)


class TraversalPositionalEmbedding(PositionalEmbedding):
    """SWAT-style traversal-based positional embedding"""

    def __init__(
        self,
        kinematic_tree: "KinematicTree",
        embed_dim: int,
    ):
        super().__init__()
        self.num_bodies = kinematic_tree.num_bodies
        self.embed_dim = embed_dim

        # Compute traversal indices
        pre_order, in_order, post_order = self._compute_traversals(kinematic_tree)

        self.register_buffer('pre_order', pre_order)
        self.register_buffer('in_order', in_order)
        self.register_buffer('post_order', post_order)

        # Learnable embeddings for each traversal
        max_idx = self.num_bodies
        self.embed_dim_per_traversal = embed_dim // 3

        self.pre_embedding = nn.Embedding(max_idx, self.embed_dim_per_traversal)
        self.in_embedding = nn.Embedding(max_idx, self.embed_dim_per_traversal)
        self.post_embedding = nn.Embedding(max_idx, self.embed_dim_per_traversal)

        # Handle remaining dimensions
        remaining = embed_dim - 3 * self.embed_dim_per_traversal
        if remaining > 0:
            self.extra_proj = nn.Linear(3 * self.embed_dim_per_traversal, embed_dim)
        else:
            self.extra_proj = None

    def _compute_traversals(
        self,
        tree: "KinematicTree"
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute pre-order, in-order, post-order indices for each node"""
        num_bodies = tree.num_bodies
        root_idx = tree.root_idx

        pre_indices = [0] * num_bodies
        in_indices = [0] * num_bodies
        post_indices = [0] * num_bodies

        pre_counter = [0]
        in_counter = [0]
        post_counter = [0]

        def traverse(node: int):
            children = tree.get_children(node)

            # Pre-order: visit node first
            pre_indices[node] = pre_counter[0]
            pre_counter[0] += 1

            if len(children) == 0:
                in_indices[node] = in_counter[0]
                in_counter[0] += 1
            elif len(children) == 1:
                traverse(children[0])
                in_indices[node] = in_counter[0]
                in_counter[0] += 1
            else:
                mid = len(children) // 2
                for child in children[:mid]:
                    traverse(child)

                in_indices[node] = in_counter[0]
                in_counter[0] += 1

                for child in children[mid:]:
                    traverse(child)

            # Post-order: visit node last
            post_indices[node] = post_counter[0]
            post_counter[0] += 1

        traverse(root_idx)

        return (
            torch.tensor(pre_indices, dtype=torch.long),
            torch.tensor(in_indices, dtype=torch.long),
            torch.tensor(post_indices, dtype=torch.long),
        )

    def forward(self) -> torch.Tensor:
        pre_embed = self.pre_embedding(self.pre_order)  # (N, D//3)
        in_embed = self.in_embedding(self.in_order)  # (N, D//3)
        post_embed = self.post_embedding(self.post_order)  # (N, D//3)

        combined = torch.cat([pre_embed, in_embed, post_embed], dim=-1)

        if self.extra_proj is not None:
            combined = self.extra_proj(combined)

        return combined  # (N, D)