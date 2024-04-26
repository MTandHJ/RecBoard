

from typing import Dict, Tuple

import torch, os
import torch.nn as nn
import freerec
from freerec.utils import infoLogger, mkdirs
from utils import calc_node_wise_norm, normalize_edge, \
                    jaccard_similarity, \
                    salton_cosine_similarity, \
                    leicht_holme_nerman_similarity, \
                    common_neighbors_similarity

freerec.declare(version='1.0.1')

cfg = freerec.parser.Parser()
cfg.add_argument("--embedding-dim", type=int, default=64)
cfg.add_argument("--num-layers", type=int, default=3)
cfg.add_argument("--trend-type", type=str, choices=('jc', 'sc', 'lhn', 'cn'), default='jc')
cfg.add_argument("--trend-coeff", type=float, default=2)
cfg.add_argument("--fusion", type=eval, choices=("True", "False"), default='True')

cfg.set_defaults(
    description="CAGCN",
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

assert cfg.fusion in (True, False), "cfg.fusion should be `True' or `False' ..."

class CAGCN(freerec.models.GenRecArch):

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

        self.loadAdj(
            dataset.train().to_bigraph(edge_type='U2I')['U2I'].edge_index
        )

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

    def loadAdj(self, edge_index: torch.Tensor):
        from freerec.utils import import_pickle, export_pickle
        R = torch.sparse_coo_tensor(
            edge_index, torch.ones(edge_index.size(1)),
            size=(self.User.count, self.Item.count)
        )
        path = os.path.join("trends", cfg.dataset, cfg.trend_type)
        mkdirs(path)
        file_ = os.path.join(path, "data.pickle")
        try:
            data = import_pickle(file_)
            trend = data['trend']
            edge_index = data['edge_index']
            edge_weight = data['edge_weight']
            edge_norm = data['edge_norm']
            trend_norm = data['trend_norm']
        except ImportError:
            if cfg.trend_type == 'jc':
                edge_index, trend = jaccard_similarity(R)
            elif cfg.trend_type == 'sc':
                edge_index, trend = salton_cosine_similarity(R)
            elif cfg.trend_type == 'lhn':
                edge_index, trend = leicht_holme_nerman_similarity(R)
            elif cfg.trend_type == 'cn':
                edge_index, trend = common_neighbors_similarity(R)
            edge_weight, _ = normalize_edge(edge_index, self.User.count, self.Item.count)
            edge_norm = calc_node_wise_norm(edge_weight, edge_index[0], self.User.count, self.Item.count)
            trend_norm = calc_node_wise_norm(trend, edge_index[0], self.User.count, self.Item.count)

            data = {
                'trend': trend,
                'edge_index': edge_index,
                'edge_weight': edge_weight,
                'edge_norm': edge_norm,
                'trend_norm': trend_norm
            }
            export_pickle(data, file_)

        if cfg.fusion:
            infoLogger("[CAGCN] >>> Use Trend and Edge Weight together ...")
            trend = cfg.trend_coeff * trend / trend_norm + edge_weight
        else:
            infoLogger("[CAGCN] >>> Use Trend only ...")
            trend = cfg.trend_coeff * trend * edge_norm / trend_norm 

        self.register_buffer(
            'Adj',
            freerec.graph.to_adjacency(
                edge_index, trend
            )
        )

    def sure_trainpipe(self, batch_size: int):
        return self.dataset.train().choiced_user_ids_source(
        ).gen_train_sampling_pos_().gen_train_sampling_neg_(
            num_negatives=1
        ).batch_(batch_size).tensor_()

    def encode(self) -> Tuple[torch.Tensor, torch.Tensor]:
        allEmbds = torch.cat(
            (self.User.embeddings.weight, self.Item.embeddings.weight), dim=0
        ) # (N, D)
        avgEmbds = allEmbds / (self.num_layers + 1)
        for _ in range(self.num_layers):
            allEmbds = self.Adj @ allEmbds
            avgEmbds += allEmbds / (self.num_layers + 1)
        userEmbds, itemEmbds = torch.split(
            avgEmbds, (self.User.count, self.Item.count)
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


class CoachForCAGCN(freerec.launcher.Coach):

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

    model = CAGCN(
        dataset,
        embedding_dim=cfg.embedding_dim, num_layers=cfg.num_layers
    )

    trainpipe = model.sure_trainpipe(cfg.batch_size)
    validpipe = model.sure_validpipe(cfg.ranking)
    testpipe = model.sure_testpipe(cfg.ranking)

    coach = CoachForCAGCN(
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