

from typing import Dict, Tuple

import torch, os
import torch.nn as nn
import torch.nn.functional as F
import freerec

freerec.declare(version='1.0.1')

cfg = freerec.parser.Parser()
cfg.add_argument("--embedding-dim", type=int, default=64)
cfg.add_argument("--num-layers", type=int, default=3)
cfg.add_argument("--fusion-mode", type=str, choices=('cat', 'add'), default="add")

cfg.add_argument("--afile", type=str, default=None, help="the file of acoustic modality features")
cfg.add_argument("--vfile", type=str, default="visual_modality.pkl", help="the file of visual modality features")
cfg.add_argument("--tfile", type=str, default="textual_modality.pkl", help="the file of textual modality features")

cfg.set_defaults(
    description="MMGCN",
    root="../../data",
    dataset='Amazon2014Baby_550_MMRec',
    epochs=100,
    batch_size=1024,
    optimizer='adam',
    lr=1e-3,
    weight_decay=1e-4,
    seed=1
)
cfg.compile()


class GraphConvNet(nn.Module):

    def __init__(
        self, 
        num_users: int,
        feature_dim: int,
        embedding_dim: int,
        num_layers: int = 3,
        fusion_mode: str = "cat",
    ):
        super().__init__()

        self.register_parameter(
            'mUser',
            nn.parameter.Parameter(
                torch.empty((num_users, feature_dim)),
                requires_grad=True
            )
        )
        nn.init.xavier_normal_(self.mUser)

        self.L = num_layers
        self.act = nn.LeakyReLU()

        self.fusion_mode = fusion_mode

        self.aggr_layers = nn.ModuleList([
            nn.Linear(feature_dim, feature_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
        ])
        self.m2id_layers = nn.ModuleList([
            nn.Linear(feature_dim, embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
        ])

        if self.fusion_mode == "cat":
            self.fusion_layers = nn.ModuleList([
                nn.Linear(feature_dim + embedding_dim, embedding_dim),
                nn.Linear(embedding_dim + embedding_dim, embedding_dim),
                nn.Linear(embedding_dim + embedding_dim, embedding_dim),
            ])
        else:
            self.fusion_layers = nn.ModuleList([
                nn.Linear(feature_dim, embedding_dim),
                nn.Linear(embedding_dim, embedding_dim),
                nn.Linear(embedding_dim, embedding_dim),
            ])

    def forward(self, mItem, idEmbds, A: torch.Tensor):
        x = torch.cat((self.mUser, mItem), dim=0) # (N, F_dim)
        x = F.normalize(x, dim=-1)

        for l in range(self.L):
            linear1 = self.aggr_layers[l]
            linear2 = self.m2id_layers[l]
            linear3 = self.fusion_layers[l]

            h = self.act(A @ linear1(x)) # F/E_dim -> F/E_dim
            x_hat = self.act(linear2(x)) + idEmbds # F/E_dim -> E_dim
            if self.fusion_mode == "cat":
                x_hat = torch.cat((h, x_hat), dim=-1) # (F/E_dim + E_dim)
                x = self.act(linear3(x_hat)) # -> E_dim
            else:
                x = self.act(linear3(h) + x_hat)
        
        return x


class MMGCN(freerec.models.GenRecArch):

    def __init__(
        self, dataset: freerec.data.datasets.RecDataSet,
        embedding_dim: int = 64, num_layers: int = 3
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

        self.register_buffer(
            "Adj",
            self.dataset.train().to_normalized_adj(
                normalization='left'
            )
        )

        self.load_feats()

        if cfg.vfile:
            self.vGCN = GraphConvNet(
                self.User.count,
                feature_dim=256, # 256 indicates the hidden size of visual features
                embedding_dim=cfg.embedding_dim,
                fusion_mode=cfg.fusion_mode,
                num_layers=cfg.num_layers,
            )
            self.vProjector = nn.Linear(self.vFeats.size(1), 256)

        if cfg.tfile:
            self.tGCN = GraphConvNet(
                self.User.count,
                feature_dim=self.tFeats.size(1),
                embedding_dim=cfg.embedding_dim,
                fusion_mode=cfg.fusion_mode,
                num_layers=cfg.num_layers,
            )

        if cfg.afile:
            self.aGCN = GraphConvNet(
                self.User.count,
                feature_dim=self.aFeats.size(1),
                embedding_dim=cfg.embedding_dim,
                fusion_mode=cfg.fusion_mode,
                num_layers=cfg.num_layers,
            )

        self.num_modality = len([file_ for file_ in (cfg.afile, cfg.vfile, cfg.tfile) if file_])
        assert self.num_modality > 0

        self.criterion = freerec.criterions.BPRLoss(reduction='mean')

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
        return self.dataset.train().shuffled_pairs_source(
        ).gen_train_sampling_neg_(
            num_negatives=1
        ).batch_(batch_size).tensor_()

    def load_feats(self):
        from freerec.utils import import_pickle
        path = self.dataset.path
        if cfg.vfile:
            self.register_buffer(
                "vFeats", import_pickle(
                    os.path.join(path, cfg.vfile)
                )
            )
        if cfg.tfile:
            self.register_buffer(
                "tFeats", import_pickle(
                    os.path.join(path, cfg.tfile)
                )
            )
        if cfg.afile:
            self.register_buffer(
                "aFeats", import_pickle(
                    os.path.join(path, cfg.afile)
                )
            )

    def encode(self) -> Tuple[torch.Tensor, torch.Tensor]:
        userEmbds = self.User.embeddings.weight
        itemEmbds = self.Item.embeddings.weight
        idEmbds = torch.cat((userEmbds, itemEmbds), dim=0).flatten(1) # N x D

        if cfg.vfile:
            vEmbds = self.vGCN(
                self.vProjector(self.vFeats), idEmbds, self.Adj
            )
        else:
            vEmbds = 0
        if cfg.tfile:
            tEmbds = self.tGCN(
                self.tFeats, idEmbds, self.Adj
            )
        else:
            tEmbds = 0
        if cfg.afile:
            aEmbds = self.aGCN(
                self.aFeats, idEmbds, self.Adj
            )
        else:
            aEmbds = 0
        avgEmbds = (vEmbds + tEmbds + aEmbds) / self.num_modality

        userEmbds, itemEmbds = torch.split(avgEmbds, (self.User.count, self.Item.count))
        return userEmbds, itemEmbds

    def fit(self, data: Dict[freerec.data.fields.Field, torch.Tensor]):
        userEmbds, itemEmbds = self.encode()
        users, positives, negatives = data[self.User], data[self.Item], data[self.INeg]
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

        return rec_loss, emb_loss + self.vGCN.mUser.square().mean()

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


class CoachForMMGCN(freerec.launcher.Coach):

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

    model = MMGCN(
        dataset,
        embedding_dim=cfg.embedding_dim, num_layers=cfg.num_layers
    )

    trainpipe = model.sure_trainpipe(cfg.batch_size)
    validpipe = model.sure_validpipe(cfg.ranking)
    testpipe = model.sure_testpipe(cfg.ranking)

    coach = CoachForMMGCN(
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