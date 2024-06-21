

from typing import Dict, Tuple

import torch
import torch.nn as nn
import freerec

freerec.declare(version='1.0.1')

cfg = freerec.parser.Parser()
cfg.add_argument("--embedding-dim", type=int, default=64)
cfg.add_argument("--num-layers", type=int, default=3)
cfg.add_argument("--dropout-rate", type=float, default=0.1)

cfg.add_argument("--norm-mode", type=str, choices=('None', 'PN', 'PN-SI', 'PN-SCS'), default='PN')
cfg.add_argument("--norm-scale", type=float, default=10.)

cfg.set_defaults(
    description="PairNorm",
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


class PairNorm(nn.Module):
    r"""
    PairNorm borrowed from: https://github.com/LingxiaoShawn/PairNorm/blob/master/layers.py#L85.
    mode:
        'None' : No normalization 
        'PN'   : Original version
        'PN-SI'  : Scale-Individually version
        'PN-SCS' : Scale-and-Center-Simultaneously version
    
    ('SCS'-mode is not in the paper but we found it works well in practice, 
        especially for GCN and GAT.)

    PairNorm is typically used after each graph convolution operation. 
    """
    def __init__(self, mode: str = 'PN', scale: float = 1.):
        assert mode in ['None', 'PN',  'PN-SI', 'PN-SCS']
        super(PairNorm, self).__init__()
        self.mode = mode
        self.scale = scale

        # Scale can be set based on origina data, and also the current feature lengths.
        # We leave the experiments to future. A good pool we used for choosing scale:
        # [0.1, 1, 10, 50, 100]
                
    def forward(self, x):
        if self.mode == 'None':
            return x
        
        col_mean = x.mean(dim=0)      
        if self.mode == 'PN':
            x = x - col_mean
            rownorm_mean = (1e-6 + x.pow(2).sum(dim=1).mean()).sqrt() 
            x = self.scale * x / rownorm_mean

        if self.mode == 'PN-SI':
            x = x - col_mean
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual

        if self.mode == 'PN-SCS':
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual - col_mean

        return x


class SGC(freerec.models.GenRecArch):

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
                normalization='sym'
            )
        )

        self.normLayer = PairNorm(mode=cfg.norm_mode, scale=cfg.norm_scale)
        self.dropout = nn.Dropout(p=cfg.dropout_rate)
        self.linear = nn.Linear(embedding_dim, embedding_dim)

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
        return self.dataset.train().choiced_user_ids_source(
        ).gen_train_sampling_pos_().gen_train_sampling_neg_(
            num_negatives=1
        ).batch_(batch_size).tensor_()

    def encode(self) -> Tuple[torch.Tensor, torch.Tensor]:
        allEmbds = torch.cat(
            (self.User.embeddings.weight, self.Item.embeddings.weight), dim=0
        ) # (N, D)
        allEmbds = self.normLayer(allEmbds)
        for _ in range(self.num_layers):
            allEmbds = self.Adj @ allEmbds
            allEmbds = self.normLayer(allEmbds)
        # XXX: very poor
        allEmbds = self.dropout(allEmbds)
        allEmbds = self.linear(allEmbds)

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


class CoachForPairNorm(freerec.launcher.Coach):

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

    model = SGC(
        dataset,
        embedding_dim=cfg.embedding_dim, num_layers=cfg.num_layers
    )

    trainpipe = model.sure_trainpipe(cfg.batch_size)
    validpipe = model.sure_validpipe(cfg.ranking)
    testpipe = model.sure_testpipe(cfg.ranking)

    coach = CoachForPairNorm(
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