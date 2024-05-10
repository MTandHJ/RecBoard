

from typing import Dict, Tuple

import torch
import torch.nn as nn
import freerec

freerec.declare(version='1.0.1')

cfg = freerec.parser.Parser()
cfg.add_argument("--embedding-dim", type=int, default=64)
cfg.add_argument("--num-layers", type=int, default=3)

cfg.set_defaults(
    description="GTE",
    root="../../data",
    dataset='Yelp2018_10104811_ROU',
    epochs=0,
    batch_size=2048,
    optimizer='adam',
    lr=1e-3,
    weight_decay=1e-4,
    seed=1
)
cfg.compile()

cfg.epochs = 0


class GTE(freerec.models.GenRecArch):

    def __init__(
        self, dataset: freerec.data.datasets.RecDataSet,
        num_layers: int = 3
    ) -> None:
        super().__init__(dataset)

        self.num_layers = num_layers

        self.User.add_module(
            "embeddings", nn.Embedding(
                self.User.count, self.Item.count
            )
        )

        self.Item.add_module(
            "embeddings", nn.Embedding(
                self.Item.count, self.Item.count
            )
        )

        edge_index = self.dataset.train().to_bigraph(edge_type='U2I')['U2I'].edge_index
        self.register_buffer(
            "R",
            torch.sparse_coo_tensor(
                edge_index, 
                torch.ones_like(edge_index[0], dtype=torch.float),
                size=(self.User.count, self.Item.count)
            )
        )

        self.criterion = freerec.criterions.BPRLoss(reduction='mean')

        self.reset_parameters()

    def reset_parameters(self):
        self.User.embeddings.weight.data.fill_(0.)
        self.Item.embeddings.weight.data.copy_(
            torch.eye(self.Item.count)
        )

    def sure_trainpipe(self, batch_size: int):
        return self.dataset.train().choiced_user_ids_source(
        ).gen_train_sampling_pos_().gen_train_sampling_neg_(
            num_negatives=1
        ).batch_(batch_size).tensor_()

    def encode(self) -> Tuple[torch.Tensor, torch.Tensor]:
        userEmbds = self.User.embeddings.weight.data
        itemEmbds = self.Item.embeddings.weight.data
        for _ in range(self.num_layers):
            userEmbds_ = torch.sparse.mm(self.R, itemEmbds) + userEmbds
            itemEmbds_ = torch.sparse.mm(self.R.t(), userEmbds) + itemEmbds
            userEmbds = userEmbds_
            itemEmbds = itemEmbds_
        return userEmbds, itemEmbds

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


class CoachForGTE(freerec.launcher.Coach):

    def train_per_epoch(self, epoch: int):
        ...

def main():

    try:
        dataset = getattr(freerec.data.datasets, cfg.dataset)(root=cfg.root)
    except AttributeError:
        dataset = freerec.data.datasets.RecDataSet(cfg.root, cfg.dataset, tasktag=cfg.tasktag)

    model = GTE(
        dataset,
        num_layers=cfg.num_layers
    )

    trainpipe = model.sure_trainpipe(cfg.batch_size)
    validpipe = model.sure_validpipe(cfg.ranking)
    testpipe = model.sure_testpipe(cfg.ranking)

    coach = CoachForGTE(
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