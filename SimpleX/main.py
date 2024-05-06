

from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import bipartite_subgraph
import freerec
from freerec.data.tags import NEGATIVE

freerec.declare(version='1.0.1')

cfg = freerec.parser.Parser()
cfg.add_argument("--embedding-dim", type=int, default=64)
cfg.add_argument("--num-negs", type=int, default=1000) # 100,500,1000
cfg.add_argument("--gamma", type=float, default=1.) # 0., 0.5, 0.1
cfg.add_argument("--margin", type=float, default=.9) # 0:1:0.1
cfg.add_argument("--weight-for-negative", type=float, default=150)
cfg.add_argument("--dropout-rate", type=float, default=.1)
cfg.add_argument("--unseen-only", type=eval, choices=(True, False), default=False)

cfg.set_defaults(
    description="SimpleX",
    root="../../data",
    dataset='Yelp2018_10104811_ROU',
    epochs=1000,
    batch_size=2048,
    optimizer='adam',
    lr=1e-3,
    weight_decay=1e-4,
    seed=1
)
cfg.compile()


class AvgConv(MessagePassing):
    
    def __init__(self, embedding_dim: int, size: Tuple[int, int]):
        super().__init__(aggr="mean", flow='target_to_source')
        self.size = size
        self.linear = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(self, items: torch.Tensor, edge_index: torch.Tensor):
        """
        Parameters:
        ---

        x: item embeddings.
        edge_index: (2, N), torch.Tensor
            (users (i), items (j)): target2source
        """
        return self.propagate(edge_index=edge_index, items=items, size=self.size)

    def message(self, items_j: torch.Tensor) -> torch.Tensor:
        return items_j

    def update(self, items: torch.Tensor) -> torch.Tensor:
        return self.linear(items)


class CosineContrastiveLoss(freerec.criterions.BaseCriterion):

    def __init__(
        self, margin: float = cfg.margin, 
        negative_weight: Optional[float] = cfg.weight_for_negative,
        reduction: str = 'mean'
    ):
        super(CosineContrastiveLoss, self).__init__(reduction)
        self.margin = margin
        self.negative_weight = negative_weight

    def forward(self, scores: torch.Tensor):
        logits_pos = scores[:, 0]
        loss_pos = (1 - logits_pos).relu()
        logits_neg = scores[:, 1:]
        loss_neg = (logits_neg - self.margin).relu()
        loss = loss_pos + loss_neg.mean(dim=-1) * self.negative_weight
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class SimpleX(freerec.models.GenRecArch):

    def __init__(
        self, dataset: freerec.data.datasets.RecDataSet,
        gamma: float = 1., embedding_dim: int = 64
    ) -> None:
        super().__init__(dataset)

        self.gamma = gamma

        self.User.add_module(
            "embeddings", nn.Embedding(
                self.User.count, embedding_dim
            )
        )

        self.Item.add_module(
            "embeddings", nn.Embedding(
                self.Item.count, embedding_dim
            )
        )

        self.register_buffer(
            "edge_index",
            self.dataset.train().to_bigraph(edge_type='U2I')['U2I'].edge_index
        )

        self.aggregator = AvgConv(
            embedding_dim=embedding_dim,
            size=(self.User.count, self.Item.count)
        )
        self.dropout = nn.Dropout(p=cfg.dropout_rate)

        self.criterion = CosineContrastiveLoss(reduction='mean')

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=1.e-4)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)

    def sure_trainpipe(self, batch_size: int):
        if cfg.unseen_only:
            return self.dataset.train().shuffled_pairs_source(
            ).gen_train_sampling_neg_(
                num_negatives=cfg.num_negs
            ).batch_(batch_size).tensor_()
        else:
            return self.dataset.train().shuffled_pairs_source(
            ).batch_(batch_size).tensor_()

    def aggregate(self, users: torch.Tensor) -> torch.Tensor:
        users = users
        items = self.Item.embeddings.weight
        edge_index, _ = bipartite_subgraph(
            (users.flatten(), torch.arange(self.Item.count, device=self.device)),
            edge_index=self.edge_index,
            size=(self.User.count, self.Item.count),
            relabel_nodes=False
        )
        userEmbds = self.User.embeddings(users)
        return userEmbds * self.gamma + self.aggregator(items, edge_index)[users] * (1 - self.gamma)

    def encode(self, data: Dict[freerec.data.fields.Field, torch.Tensor]):
        userEmbds = F.normalize(
            self.aggregate(data[self.User]), dim=-1
        ) # (B, 1, D)
        itemEmbds = F.normalize(
            self.Item.embeddings.weight, dim=-1
        ) # (N, D)
        return self.dropout(userEmbds), itemEmbds

    def fit(self, data: Dict[freerec.data.fields.Field, torch.Tensor]):
        userEmbds, itemEmbds = self.encode(data)
        items = torch.cat(
            (data[self.Item], data[self.INeg]),
            dim=1
        )
        itemEmbds = itemEmbds[items] # (B, 1 + K, D)

        scores = torch.einsum("BKD,BKD->BK", userEmbds, itemEmbds)
        rec_loss = self.criterion(scores)

        return rec_loss

    def recommend_from_full(self, data: Dict[freerec.data.fields.Field, torch.Tensor]):
        userEmbds, itemEmbds = self.encode(data)
        return torch.einsum("BKD,ND->BN", userEmbds, itemEmbds)

    def recommend_from_pool(self, data: Dict[freerec.data.fields.Field, torch.Tensor]):
        userEmbds, itemEmbds = self.encode(data)
        itemEmbds = itemEmbds[data[self.IUnseen]] # (B, 101, D)
        return torch.einsum("BKD,BKD->BK", userEmbds, itemEmbds)


class CoachForSimpleX(freerec.launcher.Coach):

    def sample_negs_from_all(self, data: Dict):
        if not self.cfg.unseen_only:
            # Sampling in this way will be much faster.
            bsz = len(data[self.User])
            data[self.Item.fork(NEGATIVE)] = torch.randint(
                0, self.Item.count, 
                size=(bsz, self.cfg.num_negs), 
                device=self.device
            )

    def train_per_epoch(self, epoch: int):
        for data in self.dataloader:
            data = self.dict_to_device(data)
            self.sample_negs_from_all(data)
            loss = self.model(data)

            self.optimizer.zero_grad()
            nn.utils.clip_grad_norm_(self.model.parameters(), 10)
            loss.backward()
            self.optimizer.step()
            
            self.monitor(
                loss.item(), 
                n=len(data[self.User]), reduction="mean", 
                mode='train', pool=['LOSS']
            )


def main():

    try:
        dataset = getattr(freerec.data.datasets, cfg.dataset)(root=cfg.root)
    except AttributeError:
        dataset = freerec.data.datasets.RecDataSet(cfg.root, cfg.dataset, tasktag=cfg.tasktag)
    
    model = SimpleX(
        dataset,
        gamma=cfg.gamma,
        embedding_dim=cfg.embedding_dim
    )
    trainpipe = model.sure_trainpipe(cfg.batch_size)
    validpipe = model.sure_validpipe(cfg.ranking)
    testpipe = model.sure_testpipe(cfg.ranking)

    coach = CoachForSimpleX(
        dataset=dataset,
        trainpipe=trainpipe,
        validpipe=validpipe,
        testpipe=testpipe,
        model=model,
        cfg=cfg
    )
    coach.fit()


if __name__ == "__main__":
    main()