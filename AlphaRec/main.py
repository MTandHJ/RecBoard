

from typing import Dict, Tuple

import torch, os
import torch.nn as nn
import torch.nn.functional as F

import freerec
from freerec.data.tags import USER, ITEM, TIMESTAMP, ID

freerec.declare(version='1.0.1')

cfg = freerec.parser.Parser()
cfg.add_argument("--embedding-dim", type=int, default=64)
cfg.add_argument("--num_layers", type=int, default=2)

cfg.add_argument("--tfile", type=str, default="textual_modality.pkl", help="the file of textual modality features")

cfg.add_argument("--projector", type=str, choices=('linear', 'mlp'), default='linear')
cfg.add_argument("--num_negs", type=int, default=256)
cfg.add_argument("--tau", type=float, default=0.15)

cfg.set_defaults(
    description="ALPHARec",
    root="../../data",
    dataset='AmazonMovies_Alpha',
    epochs=1000,
    batch_size=4096,
    optimizer='adam',
    lr=5e-4,
    weight_decay=1e-6,
    seed=1
)
cfg.compile()


class AlphaRec(freerec.models.GenRecArch):

    def __init__(
        self, dataset: freerec.data.datasets.RecDataSet,
    ) -> None:
        super().__init__(dataset)

        self.Item.add_module(
            "embeddings",
            nn.Embedding.from_pretrained(
                freerec.utils.import_pickle(
                    os.path.join(dataset.path, cfg.tfile)
                ),
                freeze=True
            )
        )

        # R
        R = dataset.train().to_bigraph(
            (USER, ID), (ITEM, ID),
            edge_type='U2I'
        )
        edge_index, edge_weight = freerec.graph.to_normalized(
            R['U2I'].edge_index, normalization='left'
        )
        R = torch.sparse_coo_tensor(
            edge_index, edge_weight, size=(self.User.count, self.Item.count)
        ).to_sparse_csr()

        # averaged items as user embeddings
        self.User.add_module(
            "embeddings",
            nn.Embedding.from_pretrained(
                R @ self.Item.embeddings.weight,
                freeze=True
            )
        )

        feat_size = self.Item.embeddings.weight.size(1)
        if cfg.projector == 'linear':
            self.projector = nn.Linear(feat_size, cfg.embedding_dim)
        else:
            self.projector = nn.Sequential(
                nn.Linear(feat_size, feat_size // 2),
                nn.LeakyReLU(),
                nn.Linear(feat_size // 2, cfg.embedding_dim)
            )

        self.num_layers = cfg.num_layers

        # U-I Branch
        self.register_buffer(
            "Adj",
            self.dataset.train().to_normalized_adj(
                normalization='sym'
            )
        )

        self.criterion = freerec.criterions.CrossEntropy4Logits(reduction='mean')

        self.reset_parameters()

    def reset_parameters(self):
        ...

    def sure_trainpipe(self, batch_size: int):
        return self.dataset.train().choiced_user_ids_source(
        ).gen_train_sampling_pos_(
        ).gen_train_sampling_neg_(
            num_negatives=cfg.num_negs, unseen_only=True
        ).batch_(batch_size).tensor_()

    def encode(self) -> Tuple[torch.Tensor, torch.Tensor]:
        userEmbs = self.projector(self.User.embeddings.weight)
        itemEmbs = self.projector(self.Item.embeddings.weight)

        features = torch.cat((userEmbs, itemEmbs), dim=0).flatten(1) # N x D
        avgFeats = features / (self.num_layers + 1)
        for _ in range(self.num_layers):
            features = self.Adj @ features
            avgFeats += features / (self.num_layers + 1)
        
        userFeats, itemFeats = torch.split(avgFeats, (self.User.count, self.Item.count))

        return F.normalize(userFeats, dim=-1), F.normalize(itemFeats, dim=-1)

    def fit(self, data: Dict[freerec.data.fields.Field, torch.Tensor]):
        users = data[self.User]
        items = torch.cat((data[self.IPos], data[self.INeg]), dim=-1) # (B, K + 1)
        userFeats, itemFeats = self.encode()

        userFeats = userFeats[users] # B x 1 x D
        itemFeats = itemFeats[items] # B x (K + 1) x D
        labels = torch.zeros((userFeats.size(0),), device=userFeats.device, dtype=torch.long)

        rec_loss = self.criterion(
            torch.einsum("BKD,BKD->BK", userFeats, itemFeats) / cfg.tau,
            labels
        )

        return rec_loss

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


class CoachForAlpha(freerec.launcher.Coach):

    def train_per_epoch(self, epoch: int):
        for data in self.dataloader:
            data = self.dict_to_device(data)
            rec_loss = self.model(data)
            loss = rec_loss

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

    model = AlphaRec(
        dataset
    )

    trainpipe = model.sure_trainpipe(cfg.batch_size)
    validpipe = model.sure_validpipe(cfg.ranking)
    testpipe = model.sure_testpipe(cfg.ranking)

    coach = CoachForAlpha(
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