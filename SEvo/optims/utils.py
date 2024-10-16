

from typing import Literal

import torch
import freerec
from freerec.data.datasets.base import RecDataSet
from freerec.data.tags import USER, ITEM, ID, SEQUENCE
from collections import defaultdict


class Smoother:
    r"""
    Smoother.

    Parameters:
    -----------
    Adj: sparse Tensor
    beta: float
        The hyper-parameter for SEvo.
    L: int
        The number of layers for approximation.
    aggr: str
        - `neumann`: Neuman series approximation.
        - `iterative`: Iterative approximation.
    """

    def __init__(
        self, Adj: torch.Tensor,
        beta: float = 0.99, L: int = 3, 
        aggr: Literal['neumann', 'iterative'] = 'neumann'
    ) -> None:
        self.Adj = Adj
        self.beta = beta
        self.L = L
        self.aggr = aggr
        self.scale_correction = 1 - self.beta ** (self.L + 1)

    @torch.no_grad()
    def __call__(self, features: torch.Tensor):
        smoothed = features
        if self.aggr == 'neumann':
            for _ in range(self.L):
                features = (self.Adj @ features) * self.beta
                smoothed = smoothed + features
            smoothed = smoothed.mul(1 - self.beta).div(self.scale_correction)
        elif self.aggr == 'iterative':
            for _ in range(self.L):
                smoothed = (self.Adj @ features) * self.beta + features * (1 - self.beta)
        else:
            raise ValueError(f"aggr should be average|iterative but {self.aggr} received ...")
        return smoothed


def get_item_graph(
    dataset: RecDataSet, 
    hops: int = 1,
    maxlen: int = None,
    NUM_PADS: int = 0,
):
    r"""
    Get an item-item adjacency matrix.

    Parameters:
    -----------
    dataset: RecDataSet
    hops: int
        A maximum walk length allowing for a pair of neighbors.
    maxlen: int
        Only the last `maxlen` items in a sequence will be used for construciton.
    NUM_PADS: int
        The number paddings.
    """

    Item = dataset.fields[ITEM, ID]
    ISeq = Item.fork(SEQUENCE)

    # seqs:
    # [{USER: 0, ISeq: [...]}, {USER: 1, ISeq: [...]}, ...]
    seqs = dataset.train().to_seqs(maxlen=maxlen)
    edge = defaultdict(int)
    for row in seqs:
        seq = row[ISeq]
        for i in range(len(seq) - 1):
            x = seq[i] + NUM_PADS
            for h, j in enumerate(range(i + 1, min(i + hops + 1, len(seq))), start=1):
                y = seq[j] + NUM_PADS
                edge[(x, y)] += 1. / h
                edge[(y, x)] += 1. / h

    edge_index, edge_weight = zip(*edge.items())
    edge_index = torch.LongTensor(
        edge_index
    ).t() # (N, 2)
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)
    edge_index, edge_weight = freerec.graph.to_normalized(
        edge_index, edge_weight, normalization='sym'
    )
    Adj = freerec.graph.to_adjacency(
        edge_index, edge_weight, num_nodes=Item.count + NUM_PADS
    ) # CSR sparse tensor

    return Adj

def get_user_item_graph(
    dataset: RecDataSet, NUM_PADS: int = 0
):
    r"""
    Get a user-item adjacency matrix.

    Parameters:
    -----------
    dataset: RecDataSet
    NUM_PADS: int
        The number paddings.
    """

    User = dataset.fields[USER, ID]
    Item = dataset.fields[ITEM, ID]
    graph = dataset.train().to_heterograph(((USER, ID), '2', (ITEM, ID)))
    graph[User.name].x = torch.empty((User.count, 0), dtype=torch.long)
    graph[Item.name].x = torch.empty((Item.count + NUM_PADS, 0), dtype=torch.long)
    graph[User.name, '2', Item.name].edge_index[1].add_(NUM_PADS)
    graph = graph.to_homogeneous()

    edge_index = freerec.graph.to_undirected(graph.edge_index)
    edge_index, edge_weight = freerec.graph.to_normalized(
        edge_index, normalization='sym'
    )
    Adj = freerec.graph.to_adjacency(
        edge_index, edge_weight, num_nodes=User.count + Item.count + NUM_PADS
    )
    return Adj

def get_graph(
    cfg,
    dataset: RecDataSet, 
    NUM_PADS: int = 0,
    itemonly: bool = True
):
    if itemonly:
        graph = get_item_graph(dataset, hops=cfg.H, maxlen=cfg.maxlen4graph, NUM_PADS=NUM_PADS)
    else:
        graph = get_user_item_graph(dataset, NUM_PADS=NUM_PADS)
    return graph