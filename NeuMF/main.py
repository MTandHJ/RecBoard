

from typing import Dict, Iterable

import torch
import torch.nn as nn
import freerec

freerec.declare(version='1.0.1')

cfg = freerec.parser.Parser()
cfg.add_argument("--embedding-dim", type=int, default=64)
cfg.add_argument("--hidden-sizes", type=str, default="64,32,16,8")
cfg.add_argument("--num-negs", type=int, default=4)

cfg.set_defaults(
    description="NeuMF",
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


class NeuMF(freerec.models.GenRecArch):

    def __init__(
        self, dataset: freerec.data.datasets.RecDataSet,
        embedding_dim: int = 64, hidden_sizes: Iterable[int] = [64,32,16,8]
    ) -> None:
        super().__init__(dataset)

        self.User.add_module(
            "embeddings4mlp", nn.Embedding(
                self.User.count, embedding_dim
            )
        )

        self.User.add_module(
            "embeddings4mf", nn.Embedding(
                self.User.count, embedding_dim
            )
        )

        self.Item.add_module(
            "embeddings4mlp", nn.Embedding(
                self.Item.count, embedding_dim
            )
        )

        self.Item.add_module(
            "embeddings4mf", nn.Embedding(
                self.Item.count, embedding_dim
            )
        )

        hidden_sizes = [embedding_dim * 2] + list(hidden_sizes)
        self.linears = nn.ModuleList(
            [nn.Linear(in_size, out_size) for in_size, out_size in zip(hidden_sizes[:-1], hidden_sizes[1:])]
        )
        self.act = nn.ReLU()

        self.fc = nn.Linear(
            hidden_sizes[-1] + cfg.embedding_dim,
            1
        )

        self.criterion = freerec.criterions.BCELoss4Logits(reduction='mean')

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
        return self.dataset.train().choiced_user_ids_source(
        ).gen_train_sampling_pos_().gen_train_sampling_neg_(
            num_negatives=cfg.num_negs
        ).batch_(batch_size).tensor_()

    def encode(self):
        return self.User.embeddings4mlp.weight, \
            self.User.embeddings4mf.weight, \
            self.Item.embeddings4mlp.weight, \
            self.Item.embeddings4mf.weight

    def fit(self, data: Dict[freerec.data.fields.Field, torch.Tensor]):
        userMLPEmbds, userMFEmbds, itemMLPEmbds, itemMFEmbds = self.encode()
        users, positives, negatives = data[self.User], data[self.IPos], data[self.INeg]
        items = torch.cat((positives, negatives), dim=1) # (B, K + 1)

        userMLPEmbds = userMLPEmbds[users] # (B, 1, D)
        itemMLPEmbds = itemMLPEmbds[items] # (B, K+1, D)
        userMLPEmbds = userMLPEmbds.expand_as(itemMLPEmbds)
        mlpFeats = torch.cat((userMLPEmbds, itemMLPEmbds), dim=-1)
        for linear in self.linears:
            mlpFeats = self.act(linear(mlpFeats)) # (B, K+1, D')

        userMFEmbds = userMFEmbds[users] # (B, 1, D)
        itemMFEmbds = itemMFEmbds[items] # (B, K+1, D)
        mfFeats = userMFEmbds.mul(itemMFEmbds)

        features = torch.cat((mlpFeats, mfFeats), dim=-1) # (B, K+1, D+D')
        logits = self.fc(features).squeeze(-1) # (B, K+1)

        labels = torch.zeros_like(items)
        labels[:, 0].fill_(1)
        rec_loss = self.criterion(logits, labels)

        return rec_loss

    def recommend_from_full(self, data: Dict[freerec.data.fields.Field, torch.Tensor]):
        userMLPEmbds, userMFEmbds, itemMLPEmbds, itemMFEmbds = self.encode()
        users = data[self.User]
        items = torch.tensor(range(self.Item.count), dtype=torch.long, device=self.device).unsqueeze(0)

        userMLPEmbds = userMLPEmbds[users] # (B, 1, D)
        itemMLPEmbds = itemMLPEmbds[items] # (B, N, D)
        userMLPEmbds, itemMLPEmbds = torch.broadcast_tensors(userMLPEmbds, itemMLPEmbds)
        mlpFeats = torch.cat((userMLPEmbds, itemMLPEmbds), dim=-1)
        for linear in self.linears:
            mlpFeats = self.act(linear(mlpFeats)) # (B, N, D')

        userMFEmbds = userMFEmbds[users] # (B, 1, D)
        itemMFEmbds = itemMFEmbds[items] # (B, N, D)
        mfFeats = userMFEmbds.mul(itemMFEmbds)

        features = torch.cat((mlpFeats, mfFeats), dim=-1) # (B, N, D+D')
        logits = self.fc(features).squeeze(-1) # (B, N)

        return logits

    def recommend_from_pool(self, data: Dict[freerec.data.fields.Field, torch.Tensor]):
        userMLPEmbds, userMFEmbds, itemMLPEmbds, itemMFEmbds = self.encode()
        users, items = data[self.User], data[self.IUnseen]

        userMLPEmbds = userMLPEmbds[users] # (B, 1, D)
        itemMLPEmbds = itemMLPEmbds[items] # (B, K+1, D)
        userMLPEmbds, itemMLPEmbds = torch.broadcast_tensors(userMLPEmbds, itemMLPEmbds)
        mlpFeats = torch.cat((userMLPEmbds, itemMLPEmbds), dim=-1)
        for linear in self.linears:
            mlpFeats = self.act(linear(mlpFeats)) # (B, K+1, D')

        userMFEmbds = userMFEmbds[users] # (B, 1, D)
        itemMFEmbds = itemMFEmbds[items] # (B, K+1, D)
        mfFeats = userMFEmbds.mul(itemMFEmbds)

        features = torch.cat((mlpFeats, mfFeats), dim=-1) # (B, K+1, D+D')
        logits = self.fc(features).squeeze(-1) # (B, K+1)

        return logits


class CoachForNeuMF(freerec.launcher.Coach):

    def train_per_epoch(self, epoch: int):
        for data in self.dataloader:
            data = self.dict_to_device(data)
            loss = self.model(data)

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
    
    model = NeuMF(
        dataset,
        embedding_dim=cfg.embedding_dim,
        hidden_sizes=list(map(int, cfg.hidden_sizes.split(',')))
    )
    trainpipe = model.sure_trainpipe(cfg.batch_size)
    validpipe = model.sure_validpipe(cfg.ranking, batch_size=128)
    testpipe = model.sure_testpipe(cfg.ranking, batch_size=128)

    coach = CoachForNeuMF(
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