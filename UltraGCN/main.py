

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import freerec
from freerec.data.tags import NEGATIVE

freerec.declare(version='1.0.1')

cfg = freerec.parser.Parser()
cfg.add_argument("--embedding-dim", type=int, default=64)
cfg.add_argument("--num-negs", type=int, default=1500)
cfg.add_argument("--num-neighbors", type=int, default=10)
cfg.add_argument("--neg-weight", type=float, default=300)
cfg.add_argument("--unseen-only", type=eval, choices=(True, False), default=False)
cfg.add_argument("--item-weight", type=float, default=5e-4, help="for item constraint")
cfg.add_argument("--init-weight", type=float, default=1e-4, help="std for init")
cfg.add_argument("--w1", type=float, default=1e-6)
cfg.add_argument("--w2", type=float, default=1.)
cfg.add_argument("--w3", type=float, default=1e-6)
cfg.add_argument("--w4", type=float, default=1.)

cfg.set_defaults(
    description="UltraGCN",
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


class UltraGCN(freerec.models.GenRecArch):

    def __init__(
        self, dataset: freerec.data.datasets.RecDataSet,
        embedding_dim: int = 64
    ) -> None:
        super().__init__(dataset)

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

        self.beta_for_user_item()
        if cfg.item_weight > 0.:
            self.beta_for_item_item()

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=cfg.init_weight)

    def beta_for_user_item(self):
        from torch_geometric.utils import degree
        graph = self.dataset.train().to_bigraph(edge_type='U2I')
        row, col = graph['U2I'].edge_index
        userDeg = degree(row, num_nodes=self.User.count)
        itemDeg = degree(col, num_nodes=self.Item.count)
        userBeta = (userDeg + 1).sqrt() / userDeg
        itemBeta = (itemDeg + 1).pow(-0.5)
        userBeta[torch.isinf(userBeta)].fill_(0.)
        itemBeta[torch.isinf(itemBeta)].fill_(0.)

        self.register_buffer('userBeta', userBeta.flatten())
        self.register_buffer('itemBeta', itemBeta.flatten())

    def beta_for_item_item(self):
        graph = self.dataset.train().to_bigraph(edge_type='U2I')
        edge_index = graph['U2I'].edge_index
        R = torch.sparse_coo_tensor(
            edge_index, torch.ones_like(edge_index[0]).float(),
            size=(self.User.count, self.Item.count)
        )
        G = R.t() @ R
        degs = torch.sparse.sum(G, dim=-1).to_dense().squeeze()
        rowBeta = torch.sqrt((degs + 1)) / degs
        colBeta = 1 / torch.sqrt(degs + 1)
        rowBeta[torch.isinf(rowBeta)] = 0.
        colBeta[torch.isinf(colBeta)] = 0.
        G = rowBeta.reshape(-1, 1) * G * colBeta.reshape(1, -1)
        G = G.coalesce()
        values, indices = torch.topk(G.to_dense(), cfg.num_neighbors, dim=-1)

        self.register_buffer('itemWeights', values.float())
        self.register_buffer('itemIndices', indices.long())

    def sure_trainpipe(self, batch_size: int):
        if cfg.unseen_only:
            return self.dataset.train().shuffled_pairs_source(
            ).gen_train_sampling_neg_(
                num_negatives=cfg.num_negs
            ).batch_(batch_size).tensor_()
        else:
            return self.dataset.train().shuffled_pairs_source(
            ).batch_(batch_size).tensor_()

    def encode(self):
        return self.User.embeddings.weight, self.Item.embeddings.weight

    def fit(self, data: Dict[freerec.data.fields.Field, torch.Tensor]):
        userEmbds, itemEmbds = self.encode()
        users, positives, negatives = data[self.User], data[self.Item], data[self.INeg]

        userEmbds = userEmbds[users] # (B, 1, D)
        iposEmbds = itemEmbds[positives] # (B, 1, D)
        inegEmbds = itemEmbds[negatives] # (B, K, D)

        posLogits = torch.einsum("BKD,BKD->BK", userEmbds, iposEmbds)
        negLogits = torch.einsum("BKD,BKD->BK", userEmbds, inegEmbds)

        # U-I
        rec_pos_loss = F.binary_cross_entropy_with_logits(
            posLogits, torch.ones_like(posLogits, dtype=torch.float32),
            cfg.w1 + cfg.w2 * self.userBeta[users] * self.itemBeta[positives],
            reduction='none'
        ).sum()
        rec_neg_loss = F.binary_cross_entropy_with_logits(
            negLogits, torch.zeros_like(negLogits, dtype=torch.float32),
            cfg.w3 + cfg.w4 * self.userBeta[users] * self.itemBeta[negatives],
            reduction='none'
        ).mean(dim=-1).sum()

        # I-I
        if cfg.item_weight > 0.:
            positives = positives.flatten()
            neighbors = itemEmbds[self.itemIndices[positives]] # (B, K, D)
            weights = self.itemWeights[positives] # (B, K)
            scores = torch.einsum("BKD,BKD->BK", userEmbds, neighbors)
            ii_loss = - weights * scores.sigmoid().log()
            ii_loss = ii_loss.sum()
        else:
            ii_loss = 0.

        return rec_pos_loss, rec_neg_loss, ii_loss

    def reset_ranking_buffers(self):
        userEmbds, itemEmbds = self.encode()
        self.ranking_buffer = dict()
        self.ranking_buffer[self.User] = userEmbds.detach().clone()
        self.ranking_buffer[self.Item] = itemEmbds.detach().clone()

    def recommend_from_full(self, data: Dict[freerec.data.fields.Field, torch.Tensor]):
        userEmbds = self.ranking_buffer[self.User][data[self.User]] # (B, 1, D)
        itemEmbds = self.ranking_buffer[self.Item]
        return torch.einsum("BKD,ND->BN", userEmbds, itemEmbds)

    def recommend_from_pool(self, data: Dict[freerec.data.fields.Field, torch.Tensor]):
        userEmbds = self.ranking_buffer[self.User][data[self.User]] # (B, 1, D)
        itemEmbds = self.ranking_buffer[self.Item][data[self.IUnseen]] # (B, 101, D)
        return torch.einsum("BKD,BKD->BK", userEmbds, itemEmbds)


class CoachForUltraGCN(freerec.launcher.Coach):

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
            rec_pos_loss, rec_neg_loss, ii_loss = self.model(data)

            loss = rec_pos_loss + rec_neg_loss * cfg.neg_weight + ii_loss * cfg.item_weight

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.monitor(
                loss.item(), 
                n=len(data[self.User]), reduction="sum", 
                mode='train', pool=['LOSS']
            )


def main():

    try:
        dataset = getattr(freerec.data.datasets, cfg.dataset)(root=cfg.root)
    except AttributeError:
        dataset = freerec.data.datasets.RecDataSet(cfg.root, cfg.dataset, tasktag=cfg.tasktag)
    
    model = UltraGCN(
        dataset,
        embedding_dim=cfg.embedding_dim
    )
    trainpipe = model.sure_trainpipe(cfg.batch_size)
    validpipe = model.sure_validpipe(cfg.ranking)
    testpipe = model.sure_testpipe(cfg.ranking)

    coach = CoachForUltraGCN(
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