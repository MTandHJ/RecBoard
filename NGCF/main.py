

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import freerec

freerec.declare(version='1.0.1')

cfg = freerec.parser.Parser()
cfg.add_argument("--embedding-dim", type=int, default=64)
cfg.add_argument("--num-layers", type=int, default=3)
cfg.add_argument("--dropout-rate", type=float, default=0.1)

cfg.set_defaults(
    description="NGCF",
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


class NGCFConv(nn.Module):

    def __init__(
        self, 
        in_features: int,
        out_features: int,
        dropout_rate: float = 0.
    ) -> None:
        super().__init__()

        self.linear1 = nn.Linear(in_features, out_features)
        self.linear2 = nn.Linear(in_features, out_features)
        self.act = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x: torch.Tensor, A: torch.Tensor):
        z = A @ x
        return F.normalize(
            self.dropout(
                self.act(self.linear1(z.add(x))) + self.act(self.linear2(z.mul(x)))
            ),
            dim=-1
        )


class NGCF(freerec.models.GenRecArch):

    def __init__(
        self, dataset: freerec.data.datasets.RecDataSet,
        embedding_dim: int = 64, num_layers: int = 3,
        dropout_rate: float = 0.
    ) -> None:
        super().__init__(dataset)

        self.num_layers = num_layers

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

        edge_index = self.dataset.train().to_graph().edge_index
        edge_index, _ = freerec.graph.add_self_loops(edge_index)
        edge_index, edge_weight = freerec.graph.to_normalized(
                edge_index, normalization='left'
        )

        self.register_buffer(
            "Adj", freerec.graph.to_adjacency(
                edge_index, edge_weight,
                num_nodes=self.User.count + self.Item.count
            )
        )
        self.convs = nn.ModuleList([NGCFConv(embedding_dim, embedding_dim, dropout_rate) for _ in range(num_layers)])

        self.criterion = freerec.criterions.BPRLoss(reduction='mean')

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight)

    def sure_trainpipe(self, batch_size: int):
        return self.dataset.train().choiced_user_ids_source(
        ).gen_train_sampling_pos_().gen_train_sampling_neg_(
            num_negatives=1
        ).batch_(batch_size).tensor_()

    def encode(self) -> Tuple[torch.Tensor, torch.Tensor]:
        embds = torch.cat(
            (self.User.embeddings.weight, self.Item.embeddings.weight), dim=0
        ) # (N, D)
        allEmbds = [embds]
        for conv in self.convs:
            embds = conv(embds, self.Adj)
            allEmbds.append(embds)
        allEmbds = torch.cat(allEmbds, dim=-1)
        userEmbds, itemEmbds = torch.split(
            allEmbds, (self.User.count, self.Item.count)
        )
        return userEmbds, itemEmbds

    def fit(self, data: Dict[freerec.data.fields.Field, torch.Tensor]):
        userEmbds, itemEmbds = self.encode()
        users, positives, negatives = data[self.User], data[self.IPos], data[self.INeg]
        userEmbds = userEmbds[users] # (B, 1, D)
        iposEmbds = itemEmbds[positives] # (B, 1, D)
        inegEmbds = itemEmbds[negatives] # (B, K, D)

        rec_loss = self.criterion(
            torch.einsum("BKD,BKD->BK", userEmbds, iposEmbds),
            torch.einsum("BKD,BKD->BK", userEmbds, inegEmbds)
        )
        emb_loss = self.criterion.regularize(
            [
                self.User.embeddings(users),
                self.Item.embeddings(positives),
                self.Item.embeddings(negatives)
            ], rtype='l2'
        ) / len(users)

        return rec_loss, emb_loss

    def reset_ranking_buffers(self):
        """This method will be executed before evaluation."""
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


class CoachForLightGCN(freerec.launcher.Coach):

    def set_optimizer(self):
        if self.cfg.optimizer.lower() == 'sgd':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=self.cfg.lr, 
                momentum=self.cfg.momentum,
                nesterov=self.cfg.nesterov,
                # weight_decay=self.cfg.weight_decay
            )
        elif self.cfg.optimizer.lower() == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.cfg.lr,
                betas=(self.cfg.beta1, self.cfg.beta2),
                # weight_decay=self.cfg.weight_decay
            )
        elif self.cfg.optimizer.lower() == 'adamw':
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=self.cfg.lr,
                betas=(self.cfg.beta1, self.cfg.beta2),
                # weight_decay=self.cfg.weight_decay
            )
        else:
            raise NotImplementedError(
                f"Unexpected optimizer {self.cfg.optimizer} ..."
            )

    def train_per_epoch(self, epoch: int):
        for data in self.dataloader:
            data = self.dict_to_device(data)
            rec_loss, emb_loss = self.model(data)
            loss = rec_loss + self.cfg.weight_decay * emb_loss

            self.optimizer.zero_grad()
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

    model = NGCF(
        dataset,
        embedding_dim=cfg.embedding_dim, 
        num_layers=cfg.num_layers, 
        dropout_rate=cfg.dropout_rate

    )

    trainpipe = model.sure_trainpipe(cfg.batch_size)
    validpipe = model.sure_validpipe(cfg.ranking)
    testpipe = model.sure_testpipe(cfg.ranking)

    coach = CoachForLightGCN(
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