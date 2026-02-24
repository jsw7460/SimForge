from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from rlworld.rl.envs.managers.scene_manager import KinematicTree


class RelationalEmbedding(nn.Module, ABC):
    """Base class for relational embeddings"""

    @abstractmethod
    def forward(self) -> torch.Tensor:
        """
        Returns:
            re: (H, N, N)
        """
        raise NotImplementedError


class GraphRelationalEmbedding(RelationalEmbedding):
    """SWAT-style graph-based relational embedding"""

    def __init__(
        self,
        kinematic_tree: "KinematicTree",
        num_heads: int = 1,
        use_laplacian: bool = True,
        use_spd: bool = True,
        use_ppr: bool = True,
        ppr_alpha: float = 0.15,
    ):
        super().__init__()
        self.num_bodies = kinematic_tree.num_bodies
        self.num_heads = num_heads

        # Compute and register graph features
        features = []

        if use_laplacian:
            lap = self._compute_normalized_laplacian(kinematic_tree)
            self.register_buffer('laplacian', lap)
            features.append('laplacian')

        if use_spd:
            spd = self._compute_shortest_path_distance(kinematic_tree)
            self.register_buffer('spd', spd)
            features.append('spd')

        if use_ppr:
            ppr = self._compute_ppr(kinematic_tree, ppr_alpha)
            self.register_buffer('ppr', ppr)
            features.append('ppr')

        self.features = features
        num_features = len(features)

        # Learnable projection to num_heads
        self.projection = nn.Linear(num_features, num_heads)

    def _compute_normalized_laplacian(self, tree: "KinematicTree") -> torch.Tensor:
        """Compute normalized graph Laplacian: L = I - D^{-1/2} A D^{-1/2}"""
        adj = tree.get_adjacency_matrix()
        degree = adj.sum(dim=1)
        degree_inv_sqrt = torch.where(
            degree > 0,
            degree.pow(-0.5),
            torch.zeros_like(degree)
        )
        D_inv_sqrt = torch.diag(degree_inv_sqrt)
        L = torch.eye(self.num_bodies) - D_inv_sqrt @ adj @ D_inv_sqrt
        return L

    def _compute_shortest_path_distance(self, tree: "KinematicTree") -> torch.Tensor:
        """Compute shortest path distance matrix using BFS"""
        n = self.num_bodies
        adj = tree.get_adjacency_matrix()
        spd = torch.zeros(n, n)

        for i in range(n):
            dist = torch.full((n,), float('inf'))
            dist[i] = 0
            queue = [i]
            head = 0

            while head < len(queue):
                curr = queue[head]
                head += 1

                for j in range(n):
                    if adj[curr, j] > 0 and dist[j] == float('inf'):
                        dist[j] = dist[curr] + 1
                        queue.append(j)

            spd[i] = dist

        # Normalize by number of nodes
        spd = spd / n
        return spd

    def _compute_ppr(self, tree: "KinematicTree", alpha: float) -> torch.Tensor:
        """Compute Personalized PageRank matrix"""
        n = self.num_bodies
        adj = tree.get_adjacency_matrix()

        # Transition matrix
        degree = adj.sum(dim=1, keepdim=True)
        degree = torch.where(degree > 0, degree, torch.ones_like(degree))
        P = adj / degree

        # PPR: alpha * (I - (1-alpha) * P^T)^{-1}
        I = torch.eye(n)
        ppr = alpha * torch.inverse(I - (1 - alpha) * P.T)

        return ppr

    def forward(self) -> torch.Tensor:
        """
        Returns:
            re: (H, N, N) - relational bias for each attention head
        """
        feature_list = []

        if 'laplacian' in self.features:
            feature_list.append(self.laplacian)
        if 'spd' in self.features:
            feature_list.append(self.spd)
        if 'ppr' in self.features:
            feature_list.append(self.ppr)

        stacked = torch.stack(feature_list, dim=-1)  # (N, N, num_features)
        re = self.projection(stacked)  # (N, N, H)
        re = re.permute(2, 0, 1)  # (H, N, N)

        return re