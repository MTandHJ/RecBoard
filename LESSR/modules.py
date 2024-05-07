

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_dense_batch, softmax


class EOPA(MessagePassing):
    r"""Edge-Order Preserving Aggregation."""

    def __init__(
        self, input_dim: int, output_dim: int,
        dropout_rate: float = 0., activation = None, batch_norm: bool = True, 
    ):
        super().__init__()

        self.batch_norm = nn.BatchNorm1d(input_dim) if batch_norm else None
        self.feat_drop = nn.Dropout(dropout_rate)
        self.gru = nn.GRU(input_dim, input_dim, batch_first=True)
        self.fc_self = nn.Linear(input_dim, output_dim, bias=False)
        self.fc_neigh = nn.Linear(input_dim, output_dim, bias=False)
        self.activation = activation

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x  = self.feat_drop(x)
        x = self.fc_self(x) + self.propagate(edge_index, x=x)
        if self.activation is not None:
            x = self.activation(x)
        return x

    def aggregate(
        self, x_j: torch.Tensor, edge_index_i: torch.Tensor, size_i: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Parameters:
        -----------
        x_j: torch.Tensor
            source node features
        edge_index_i: torch.Tensor
            target node index
        size_i: torch.Tensor
            the number of target nodes
        
        Flows:
        ------
        1. `to_dense_batch` collects neighbors for each session
        2. `gru` aggregates neighbors for each session

        Notes:
        ------
        During the aggregation of `gru`, zero-padding is also involved.
        However, the official code seems ignore this issue, and thus I implement this in a similar way.
        """
        x = to_dense_batch(x_j, edge_index_i, batch_size=size_i)[0]
        _, hn = self.gru(x)
        return hn.squeeze(0)

    def update(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_neigh(x)


class SGAT(MessagePassing):
    r"""
    Shortcut Graph Attention.

    SGAT removes repeated edges.
    """

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int,
        dropout_rate: float = 0., activation = None, batch_norm: bool = True, 
    ):
        super().__init__(aggr='add')

        self.batch_norm = nn.BatchNorm1d(input_dim) if batch_norm else None
        self.feat_drop = nn.Dropout(dropout_rate)
        self.fc_q = nn.Linear(input_dim, hidden_dim, bias=True)
        self.fc_k = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc_v = nn.Linear(input_dim, output_dim, bias=False)
        self.fc_e = nn.Linear(hidden_dim, 1, bias=False)
        self.activation = activation

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x  = self.feat_drop(x)
        q = self.fc_q(x)
        k = self.fc_k(x)
        v = self.fc_k(x)

        alpha = self.edge_updater(edge_index, q=q, k=k)

        x = self.propagate(edge_index, x=v, alpha=alpha)
        if self.activation is not None:
            x = self.activation(x)
        return x

    def edge_update(
        self, q_j: torch.Tensor, k_j: torch.Tensor,
        edge_index_i: torch.Tensor, size_i: int
    ) -> torch.Tensor:
        r"""
        Parameters:
        -----------
        edge_index_i: torch.Tensor
            target node index, i.e., edge_index[1]
        size_i: int 
            the number of target nodes
        """
        alpha =  self.fc_e((q_j + k_j).sigmoid())
        alpha = softmax(alpha, index=edge_index_i, num_nodes=size_i)
        return alpha

    def aggregate(
        self, x_j: torch.Tensor, alpha: torch.Tensor,
        edge_index_i: torch.Tensor, size_i: int
    ) -> torch.Tensor:
        return super().aggregate(x_j.mul(alpha), edge_index_i, dim_size=size_i)


class AttnReadout(MessagePassing):

    def __init__(
        self,
        input_dim: int, hidden_dim: int, output_dim: int,
        dropout_rate: float = 0., activation = None, batch_norm: bool =True,
    ):
        super().__init__(aggr='add')
        self.batch_norm = nn.BatchNorm1d(input_dim) if batch_norm else None
        self.feat_drop = nn.Dropout(dropout_rate)
        self.fc_u = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc_v = nn.Linear(input_dim, hidden_dim, bias=True)
        self.fc_e = nn.Linear(hidden_dim, 1, bias=False)
        self.fc_out = (
            nn.Linear(input_dim, output_dim, bias=False)
            if output_dim != input_dim else None
        )
        self.activation = activation

    def forward(
        self, x: torch.Tensor, lasts: torch.Tensor, 
        edge_index: torch.Tensor, groups: torch.Tensor
    ) -> torch.Tensor:
        # edge_index: (BS, D)
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x  = self.feat_drop(x)
        x_u = self.fc_u(x) # (*, D)
        x_v = self.fc_v(lasts) # (*, D)

        alpha = self.edge_updater(edge_index, q=x_u, k=x_v, groups=groups) # (BS, D)

        x = self.propagate(edge_index, x=x, alpha=alpha, groups=groups)
        if self.fc_out is not None:
            x  = self.fc_out(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

    def edge_update(
        self, q: torch.Tensor, k: torch.Tensor, 
        edge_index_i: torch.Tensor, groups: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Parameters:
        -----------
        groups: torch.Tensor
            batch index for grouping
        """
        alpha =  self.fc_e((q + k).sigmoid())
        alpha = softmax(alpha, index=groups)
        return alpha

    def message(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def aggregate(
        self, x: torch.Tensor, edge_index_i: torch.Tensor,
        alpha: torch.Tensor, groups: torch.Tensor
    ) -> torch.Tensor:
        return super().aggregate(x.mul(alpha), index=groups, dim_size=groups.max() + 1)