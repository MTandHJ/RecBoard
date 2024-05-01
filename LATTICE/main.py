

from typing import Dict, Tuple

import torch, os
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter

import freerec
from freerec.data.tags import USER, ITEM, TIMESTAMP, ID

freerec.declare(version='1.0.1')

cfg = freerec.parser.Parser()
cfg.add_argument("--embedding-dim", type=int, default=64)
cfg.add_argument("--num_ui_layers", type=int, default=2, help="the number of layers for U-I graph")
cfg.add_argument("--num_ii_layers", type=int, default=1, help="the number of layers for I-I graph")

cfg.add_argument("--knn-k", type=int, default=10, help="top-k knn graph")
cfg.add_argument("--origin-ratio", type=float, default=0.5, help="ratio of fixed graph to learnable graph")

cfg.add_argument("--vfile", type=str, default="visual_modality.pkl", help="the file of visual modality features")
cfg.add_argument("--tfile", type=str, default="textual_modality.pkl", help="the file of textual modality features")

cfg.set_defaults(
    description="LATTICE",
    root="../../data",
    dataset='Amazon2014Baby_550_MMRec',
    epochs=1000,
    batch_size=2048,
    optimizer='adam',
    lr=1e-3,
    weight_decay=1e-4,
    seed=1
)
cfg.compile()


class IISide(nn.Module):

    def __init__(
        self,
        num_items: int,
        num_layers: int,
        embedding_dim: int,
        dataset: freerec.data.datasets.base.RecDataSet
    ) -> None:
        super().__init__()

        self.num_items = num_items
        self.num_layers = num_layers
        self.load_feats(dataset.path)

        if cfg.vfile:
            self.vProjector = nn.Linear(self.vFeats.weight.size(1), embedding_dim)

        if cfg.tfile:
            self.tProjector = nn.Linear(self.tFeats.weight.size(1), embedding_dim)

    def load_feats(self, path: str):
        r"""
        Load v/t Features.

        Note: Following the offical implementation,
        they are stored as nn.Embedding and are trainable in default.
        I tried a frozen variant on Baby and found this operation makes no difference.
        """
        from freeplot.utils import import_pickle
        if cfg.vfile:
            vFeats = import_pickle(
                os.path.join(path, cfg.vfile)
            )
            self.vFeats = nn.Embedding.from_pretrained(vFeats, freeze=False)
            vAdj = self.get_knn_graph(vFeats).detach()
            self.register_buffer(
                'vAdj',
                vAdj
            )

        if cfg.tfile:
            tFeats = import_pickle(
                os.path.join(path, cfg.tfile)
            )
            self.tFeats = nn.Embedding.from_pretrained(tFeats, freeze=False)
            tAdj = self.get_knn_graph(tFeats).detach()
            self.register_buffer(
                'tAdj',
                tAdj
            )

        if not cfg.vfile and cfg.tfile:
            raise NotImplementedError("At least visual or texual modality should be given ...")

        self.alpha = nn.Parameter(torch.Tensor([0.5, 0.5]))
        self.softmax = nn.Softmax(dim=0)
        return

    def get_knn_graph(self, features: torch.Tensor):
        r"""
        Compute the kNN graph.

        Note: Following the offical implementation,
        this graph is not symmetric.
        """
        features = F.normalize(features, dim=-1) # (N, D)
        sim = features @ features.t() # (N, N)
        edge_index, w_ = freerec.graph.get_knn_graph(
            sim, cfg.knn_k, symmetric=False
        )

        rows, cols = edge_index[0], edge_index[1]
        deg = 1.e-7 + scatter(torch.ones_like(rows), rows, dim=0, dim_size=self.num_items) # degree of item (int)
        deg_inv_sqrt = deg.pow(-0.5)
        edge_weight = w_ * deg_inv_sqrt[rows] * deg_inv_sqrt[cols]
        return torch.sparse_coo_tensor(
            edge_index, edge_weight,
            size=(self.num_items, self.num_items)
        )

    def forward(self, itemEmbds: torch.Tensor):
        weight = self.softmax(self.alpha)

        vFeats = self.vProjector(self.vFeats.weight) if cfg.vfile else None
        learned_vAdj = self.get_knn_graph(vFeats)
        final_vAdj = cfg.origin_ratio * self.vAdj + (1 - cfg.origin_ratio) * learned_vAdj

        tFeats = self.tProjector(self.tFeats.weight) if cfg.tfile else None
        learned_tAdj = self.get_knn_graph(tFeats)
        final_tAdj = cfg.origin_ratio * self.tAdj + (1 - cfg.origin_ratio) * learned_tAdj

        adj = weight[0] * final_vAdj + weight[1] * final_tAdj

        for _ in range(self.num_layers):
            itemEmbds = adj @ itemEmbds
        return itemEmbds


class LATTICE(freerec.models.GenRecArch):

    def __init__(
        self, dataset: freerec.data.datasets.RecDataSet,
    ) -> None:
        super().__init__(dataset)

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

        self.num_layers = cfg.num_ui_layers

        # I-I Branch
        self.iiSide = IISide( 
            self.Item.count,
            num_layers=cfg.num_ii_layers,
            embedding_dim=cfg.embedding_dim,
            dataset=dataset
        )

        # U-I Branch
        self.register_buffer(
            "Adj",
            self.dataset.train().to_normalized_adj(
                normalization='sym'
            )
        )

        self.criterion = freerec.criterions.BPRLoss(reduction='mean')

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.User.embeddings.weight)
        nn.init.xavier_normal_(self.Item.embeddings.weight)

    def sure_trainpipe(self, batch_size: int):
        return self.dataset.train().choiced_user_ids_source(
        ).gen_train_sampling_pos_().gen_train_sampling_neg_(
            num_negatives=1
        ).batch_(batch_size).tensor_()

    def encode(self) -> Tuple[torch.Tensor, torch.Tensor]:
        userEmbs = self.User.embeddings.weight
        itemEmbs = self.Item.embeddings.weight

        features = torch.cat((userEmbs, itemEmbs), dim=0).flatten(1) # N x D
        avgFeats = features / (self.num_layers + 1)
        for _ in range(self.num_layers):
            features = self.Adj @ features
            avgFeats += features / (self.num_layers + 1)
        
        iiEmbs = self.iiSide(itemEmbs)
        userFeats, itemFeats = torch.split(avgFeats, (self.User.count, self.Item.count))

        return userFeats, itemFeats + F.normalize(iiEmbs, dim=-1)

    def fit(self, data: Dict[freerec.data.fields.Field, torch.Tensor]):
        users, positives, negatives = data[self.User], data[self.IPos], data[self.INeg]
        userFeats, itemFeats = self.encode()
        userFeats = userFeats[users] # B x 1 x D
        iposFeats = itemFeats[positives]
        inegFeats = itemFeats[negatives]

        rec_loss = self.criterion(
            torch.einsum("BKD,BKD->BK", userFeats, iposFeats),
            torch.einsum("BKD,BKD->BK", userFeats, inegFeats)
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


class CoachForLATTICE(freerec.launcher.Coach):

    def set_optimizer(self):
        if self.cfg.optimizer.lower() == 'sgd':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=self.cfg.lr, 
                momentum=self.cfg.momentum,
                nesterov=self.cfg.nesterov,
                weight_decay=self.cfg.weight_decay
            )
        elif self.cfg.optimizer.lower() == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.cfg.lr,
                betas=(self.cfg.beta1, self.cfg.beta2),
                weight_decay=self.cfg.weight_decay
            )
        elif self.cfg.optimizer.lower() == 'adamw':
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=self.cfg.lr,
                betas=(self.cfg.beta1, self.cfg.beta2),
                weight_decay=self.cfg.weight_decay
            )
        else:
            raise NotImplementedError(
                f"Unexpected optimizer {self.cfg.optimizer} ..."
            )

    def train_per_epoch(self, epoch: int):
        for data in self.dataloader:
            data = self.dict_to_device(data)
            rec_loss, emb_loss = self.model(data)
            loss = rec_loss #+ self.cfg.weight_decay * emb_loss

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

    model = LATTICE(
        dataset
    )

    trainpipe = model.sure_trainpipe(cfg.batch_size)
    validpipe = model.sure_validpipe(cfg.ranking)
    testpipe = model.sure_testpipe(cfg.ranking)

    coach = CoachForLATTICE(
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