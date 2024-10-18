

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import freerec

freerec.declare(version='1.0.1')

cfg = freerec.parser.Parser()
cfg.add_argument("--embedding-dim", type=int, default=64)
cfg.add_argument("--num-layers", type=int, default=3)

cfg.add_argument("--aug-type", type=str, choices=('nd', 'ed', 'rw'),  default='ed')
cfg.add_argument("--ssl-drop-ratio", type=float, default=0.1)
cfg.add_argument("--ssl-weight", type=float, default=0.1)
cfg.add_argument("--temperature", type=float, default=0.2)

cfg.set_defaults(
    description="SGL",
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


class SGL(freerec.models.GenRecArch):

    def __init__(
        self, dataset: freerec.data.datasets.RecDataSet
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

        self.register_buffer(
            'edge_index',
            self.dataset.train().to_bigraph(edge_type='u2i')['u2i'].edge_index
        )

        self.register_buffer(
            'Adj',
            self.dataset.train().to_normalized_adj()
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
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)

    def sure_trainpipe(self, batch_size: int):
        return self.dataset.train().choiced_user_ids_source(
        ).gen_train_sampling_pos_().gen_train_sampling_neg_(
            num_negatives=1
        ).batch_(batch_size).tensor_()

    def sample_subgraph(self):
        row, col = self.edge_index[0], self.edge_index[1]
        if cfg.aug_type == 'nd':
            user_weight = (torch.rand(self.User.count, device=self.device) > cfg.ssl_drop_ratio).float()
            item_weight = (torch.rand(self.Item.count, device=self.device) > cfg.ssl_drop_ratio).float()
            edge_weight = user_weight[row] * item_weight[col]
        elif cfg.aug_type in ('ed', 'rw'):
            edge_weight = (torch.rand(len(row), device=self.device) > cfg.ssl_drop_ratio).float()
        else:
            raise NotImplemented
        edge_index = torch.stack((row, col + self.User.count), dim=0) # (2, N)
        edge_index, edge_weight = freerec.graph.to_undirected(
            edge_index, edge_weight,
            num_nodes=self.User.count + self.Item.count
        )
        edge_index, edge_weight = freerec.graph.to_normalized(edge_index, edge_weight)
        Adj = freerec.graph.to_adjacency(
            edge_index, edge_weight,
            num_nodes=self.User.count + self.Item.count
        )
        return Adj

    def resample(self):
        if cfg.aug_type in ('nd', 'ed'):
            Adj = self.sample_subgraph()
            self.sAdjs = [Adj] * cfg.num_layers
            Adj_ = self.sample_subgraph()
            self.sAdjs_ = [Adj_] * cfg.num_layers
        else:
            self.sAdjs = [self.sample_subgraph() for _ in range(cfg.num_layers)]
            self.sAdjs_ = [self.sample_subgraph() for _ in range(cfg.num_layers)]

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
    
    def encode_subgraphs(self):
        allEmbds = torch.cat(
            (self.User.embeddings.weight, self.Item.embeddings.weight), dim=0
        ) # (N, D)
        avgEmbds = allEmbds / (self.num_layers + 1)
        for l in range(self.num_layers):
            allEmbds = self.sAdjs[l] @ allEmbds
            avgEmbds += allEmbds / (self.num_layers + 1)
        userEmbds, itemEmbds = torch.split(
            avgEmbds, (self.User.count, self.Item.count)
        )
        return F.normalize(userEmbds, dim=-1), F.normalize(itemEmbds, dim=-1)

    def encode_subgraphs_(self):
        allEmbds = torch.cat(
            (self.User.embeddings.weight, self.Item.embeddings.weight), dim=0
        ) # (N, D)
        avgEmbds = allEmbds / (self.num_layers + 1)
        for l in range(self.num_layers):
            allEmbds = self.sAdjs_[l] @ allEmbds
            avgEmbds += allEmbds / (self.num_layers + 1)
        userEmbds, itemEmbds = torch.split(
            avgEmbds, (self.User.count, self.Item.count)
        )
        return F.normalize(userEmbds, dim=-1), F.normalize(itemEmbds, dim=-1)

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

        userEmbds, itemEmbds = self.encode_subgraphs()
        userEmbds_, itemEmbds_ = self.encode_subgraphs_()
        userEmbds = userEmbds[users] # (B, 1, D)
        userEmbds_ = userEmbds_[users] # (B, 1, D)
        itemEmbds = itemEmbds[positives] # (B, 1, D)
        itemEmbds_ = itemEmbds_[positives] # (B, 1, D)

        ssl_user_logits = torch.einsum("MKD,NKD->MN", userEmbds, userEmbds_).div(cfg.temperature)
        ssl_item_logits = torch.einsum("MKD,NKD->MN", itemEmbds, itemEmbds_).div(cfg.temperature)
        targets = torch.arange(ssl_user_logits.size(0), device=self.device, dtype=torch.long)
        ssl_loss = freerec.criterions.cross_entropy_with_logits(
            ssl_user_logits, targets
        ) + freerec.criterions.cross_entropy_with_logits(
            ssl_item_logits, targets
        )

        return rec_loss, emb_loss, ssl_loss

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


class CoachForSGL(freerec.launcher.Coach):

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
        self.get_res_sys_arch().resample()
        for data in self.dataloader:
            data = self.dict_to_device(data)
            rec_loss, emb_loss, ssl_loss = self.model(data)
            loss = rec_loss + self.cfg.weight_decay * emb_loss + self.cfg.ssl_weight * ssl_loss

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

    model = SGL(dataset)

    trainpipe = model.sure_trainpipe(cfg.batch_size)
    validpipe = model.sure_validpipe(cfg.ranking)
    testpipe = model.sure_testpipe(cfg.ranking)

    coach = CoachForSGL(
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