

from typing import Dict, Optional, Union

import torch, os
import torch.nn as nn
import torch.nn.functional as F

import freerec
from freerec.data.tags import USER, ITEM, TIMESTAMP, ID

freerec.declare(version='1.0.1')

cfg = freerec.parser.Parser()
cfg.add_argument("-eb", "--embedding-dim", type=int, default=64)
cfg.add_argument("--num_layers", type=int, default=2, help="the number of layers for U-I graph")

cfg.add_argument("--knn-k", type=int, default=10, help="top-k knn graph")
cfg.add_argument("--weight4cl", type=float, default=0.01, help="weight for contrastive loss")
cfg.add_argument("--temperature", type=float, default=0.2, help="temperature for contrastive loss")

cfg.add_argument("--afile", type=str, default=None, help="the file of acoustic modality features")
cfg.add_argument("--vfile", type=str, default="visual_modality.pkl", help="the file of visual modality features")
cfg.add_argument("--tfile", type=str, default="textual_modality.pkl", help="the file of textual modality features")

cfg.set_defaults(
    description="MGCN",
    root="../../data",
    dataset='Amazon2014Baby_550_MMRec',
    epochs=500,
    batch_size=2048,
    optimizer='adam',
    lr=1e-3,
    weight_decay=1e-4,
    seed=1
)
cfg.compile()


class Purifier(nn.Module):

    def __init__(
        self,
        feat_dim: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()

        self.projector = nn.Linear(
            feat_dim, embedding_dim
        )
        self.gate = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.Sigmoid()
        )

    def forward(self, mEmbds: torch.Tensor, iEmbds: torch.Tensor):
        mEmbds = self.projector(mEmbds)
        return iEmbds.mul(
            self.gate(mEmbds)
        )


class MGCN(freerec.models.GenRecArch):

    def __init__(
        self,
        dataset: freerec.data.datasets.RecDataSet,
    ) -> None:
        super().__init__(dataset)

        self.num_layers = cfg.num_layers

        self.User.add_module(
            "embeddings", nn.Embedding(
                self.User.count, cfg.embedding_dim
            )
        )
        self.Item.add_module(
            "embeddings", nn.Embedding(
                self.Item.count, cfg.embedding_dim
            )
        )

        # UI adjacency matrix for U-I View
        self.register_buffer(
            'Adj',
            dataset.train().to_normalized_adj()
        )

        # U-I Interaction matrix for I-I View
        g = dataset.train().to_bigraph(
            (USER, ID), (ITEM, ID),
            edge_type='U2I'
        )
        edge_index, edge_weight = freerec.graph.to_normalized(
            g['U2I'].edge_index, normalization='sym'
        )
        self.register_buffer(
            'R',
            torch.sparse_coo_tensor(
                edge_index, edge_weight, 
                size=(self.User.count, self.Item.count)
            ).to_sparse_csr()
        )

        # Load modalify features
        if cfg.vfile:
            vAdj, vFeats = self.load_feats(
                dataset.path, cfg.vfile
            )
            self.register_buffer(
                'vAdj', vAdj
            )
            self.vFeats = vFeats
            self.vPurifier = Purifier(
                self.vFeats.weight.size(1),
                cfg.embedding_dim
            )

        if cfg.tfile:
            tAdj, tFeats = self.load_feats(
                dataset.path, cfg.tfile
            )
            self.register_buffer(
                'tAdj', tAdj
            )
            self.tFeats = tFeats
            self.tPurifier = Purifier(
                self.tFeats.weight.size(1),
                cfg.embedding_dim
            )

        if cfg.afile:
            aAdj, aFeats = self.load_feats(
                dataset.path, cfg.afile
            )
            self.register_buffer(
                'aAdj', aAdj
            )
            self.aFeats = aFeats
            self.aPurifier = Purifier(
                self.aFeats.weight.size(1),
                cfg.embedding_dim
            )

        self.num_modality = len([file_ for file_ in (cfg.afile, cfg.vfile, cfg.tfile) if file_])

        # Fusing stage
        self.query_common = nn.Sequential(
            nn.Linear(cfg.embedding_dim, cfg.embedding_dim),
            nn.Tanh(),
            nn.Linear(cfg.embedding_dim, 1, bias=False)
        )
        self.preference_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cfg.embedding_dim, cfg.embedding_dim),
                nn.Sigmoid()
            )
            for _ in range(self.num_modality)
        ])

        self.criterion = freerec.criterions.BPRLoss(reduction='mean')

        self.reset_parameters()


    def reset_parameters(self):
        nn.init.xavier_normal_(self.User.embeddings.weight)
        nn.init.xavier_normal_(self.Item.embeddings.weight)

    def sure_trainpipe(self, batch_size: int):
        return self.dataset.train().shuffled_pairs_source(
        ).gen_train_sampling_neg_(
            num_negatives=1
        ).batch_(batch_size).tensor_()

    def get_knn_graph(self, mFeats: torch.Tensor):
        mFeats = F.normalize(mFeats, dim=-1)
        sim = mFeats @ mFeats.t()
        edge_index, edge_weight = freerec.graph.get_knn_graph(
            sim, cfg.knn_k, symmetric=False
        )
        row, col = edge_index[0], edge_index[1]
        deg = freerec.graph.scatter(
            edge_weight, row, 
            dim=0, dim_size=self.Item.count
        )
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt.masked_fill_(torch.isinf(deg_inv_sqrt), 0.)
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        return freerec.graph.to_adjacency(
            edge_index, edge_weight,
            num_nodes=self.Item.count
        )

    def load_feats(self, path: str, filename: str):
        from freeplot.utils import import_pickle
        mFeats = import_pickle(
            os.path.join(path, filename)
        )
        mAdj = self.get_knn_graph(mFeats)
        mFeats = nn.Embedding.from_pretrained(mFeats, freeze=False)
        return mAdj, mFeats

    def user_item_encode(
        self, 
        iEmbds: torch.Tensor,
        Adj: torch.Tensor
    ):
        r"""
        Encode ID embeddings using LightGCN.
        """
        features = iEmbds
        avgFeats = features / (self.num_layers + 1)
        for _ in range(self.num_layers):
            features = Adj @ features
            avgFeats += features / (self.num_layers + 1)
        return avgFeats

    def item_item_encode(
        self, 
        mEmbds: torch.Tensor,
        mAdj: torch.Tensor,
        R: torch.Tensor
    ):
        r"""
        Encode item-item view modality features.

        Parameters:
        -----------
        mEmbds: torch.Tensor,
            Modality embeddings after Purifier
        mAdj: torch.Tensor,
            Modality adjacency matrix
        R: torch.Tensor
            Sym normalized interaction matrix
        """
        mEmbds_i = mAdj @ mEmbds
        mEmbds_u = R @  mEmbds_i
        return torch.cat(
            (mEmbds_u, mEmbds_i),
            dim=0
        )

    def fuse(self, mEmbds: torch.Tensor, preferences: torch.Tensor):
        att_common = self.query_common(mEmbds) # (N, X, 1)
        weight_common = F.softmax(att_common, dim=1) # (N, X, 1)
        sharedEmbds = (mEmbds * weight_common).sum(1, keepdim=True) # (N, 1, D)
        selfEmbds = mEmbds - sharedEmbds # (N, X, D)
        return torch.cat((sharedEmbds, selfEmbds.mul(preferences)), dim=1).mean(1)

    def encode(self):
        userEmbds = self.User.embeddings.weight
        itemEmbds = self.Item.embeddings.weight
        iEmbds = torch.cat((userEmbds, itemEmbds), dim=0)
        mEmbds = []
        if cfg.vfile:
            vEmbds = self.vPurifier(self.vFeats.weight, itemEmbds)
            vEmbds = self.item_item_encode(
                vEmbds, self.vAdj, self.R
            )
            mEmbds.append(vEmbds)
        if cfg.tfile:
            tEmbds = self.tPurifier(self.tFeats.weight, itemEmbds)
            tEmbds = self.item_item_encode(
                tEmbds, self.tAdj, self.R
            )
            mEmbds.append(tEmbds)
        if cfg.afile:
            aEmbds = self.aPurifier(self.aFeats.weight, itemEmbds)
            aEmbds = self.item_item_encode(
                aEmbds, self.aAdj, self.R
            )
            mEmbds.append(aEmbds)
        mEmbds = torch.stack(mEmbds, dim=1) # (N, X, D)
        iEmbds = self.user_item_encode(iEmbds, self.Adj) # (N, D)
        preferences = torch.stack([gate(iEmbds) for gate in self.preference_gates], dim=1) # (N, X, D)

        mEmbds = self.fuse(mEmbds, preferences)
        finalEmbds = iEmbds + mEmbds
        userEmbds, itemEmbds = torch.split(finalEmbds, (self.User.count, self.Item.count), dim=0)
        return userEmbds, itemEmbds, mEmbds, iEmbds

    def InfoNCE(self, view1: torch.Tensor, view2: torch.Tensor, temperature: float):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)
        return torch.mean(cl_loss)

    def fit(self, data: Dict[freerec.data.fields.Field, torch.Tensor]):
        userEmbds, itemEmbds, mEmbds, iEmbds = self.encode()

        users, positives, negatives = data[self.User], data[self.Item], data[self.INeg]
        items = torch.cat([positives, negatives], dim=1)
        userEmbds = userEmbds[users] # (B, 1, D)
        itemEmbds = itemEmbds[items] # (B, 2, D)

        scores = userEmbds.mul(itemEmbds).sum(-1) # (B, 2)
        rec_loss = self.criterion(scores[:, 0], scores[:, 1])

        mEmbds_u, mEmbds_i = torch.split(mEmbds, (self.User.count, self.Item.count), dim=0)
        iEmbds_u, iEmbds_i = torch.split(iEmbds, (self.User.count, self.Item.count), dim=0)
        users, items = users.flatten(), items[:, 0]
        cl_loss = self.InfoNCE(
            mEmbds_u[users], iEmbds_u[users], cfg.temperature
        ) + self.InfoNCE(
            mEmbds_i[items], iEmbds_i[items], cfg.temperature
        )

        emb_loss = self.criterion.regularize(
            [
                self.User.embeddings(users),
                self.Item.embeddings(positives),
                self.Item.embeddings(negatives)
            ], rtype='l2'
        ) / len(users)

        return rec_loss, cl_loss, emb_loss

    def reset_ranking_buffers(self):
        userEmbds, itemEmbds, _, _ = self.encode()
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


class CoachForMGCN(freerec.launcher.Coach):

    def train_per_epoch(self, epoch: int):
        for data in self.dataloader:
            data = self.dict_to_device(data)
            rec_loss, cl_loss, emb_loss = self.model.fit(data)
            loss = rec_loss + emb_loss * self.cfg.weight_decay + cl_loss * self.cfg.weight4cl

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
    
    model = MGCN(dataset=dataset)
    trainpipe = model.sure_trainpipe(cfg.batch_size)
    validpipe = model.sure_validpipe(cfg.ranking)
    testpipe = model.sure_testpipe(cfg.ranking)

    coach = CoachForMGCN(
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