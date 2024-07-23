

from typing import Dict, Tuple

import torch, os
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity
import freerec

freerec.declare(version='1.0.1')

cfg = freerec.parser.Parser()
cfg.add_argument("--embedding-dim", type=int, default=64)
cfg.add_argument("--num-layers", type=int, default=3)


cfg.add_argument("--dropout-rate", type=float, default=0.5)
cfg.add_argument("--second-l", type=float, default=2.)
cfg.add_argument("--reg-weight", type=float, default=1.e-1)

cfg.add_argument("--afile", type=str, default=None, help="the file of acoustic modality features")
cfg.add_argument("--vfile", type=str, default="visual_modality.pkl", help="the file of visual modality features")
cfg.add_argument("--tfile", type=str, default="textual_modality.pkl", help="the file of textual modality features")

cfg.set_defaults(
    description="BM3",
    root="../../data",
    dataset='Amazon2024Baby_550_MMRec',
    epochs=1000,
    batch_size=2048,
    optimizer='adam',
    lr=1e-3,
    weight_decay=1e-4,
    seed=1
)
cfg.compile()


class BM3(freerec.models.GenRecArch):

    def __init__(
        self, dataset: freerec.data.datasets.RecDataSet,
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
            "Adj",
            self.dataset.train().to_normalized_adj(
                normalization='sym'
            )
        )

        self.load_feats(dataset.path)
        self.predictor = nn.Linear(cfg.embedding_dim, cfg.embedding_dim)
        nn.init.xavier_normal_(self.predictor.weight)

        self.reset_parameters()

    def load_feats(self, path: str):
        r"""
        Load v/t Features.

        Note: Following the offical implementation,
        they are stored as nn.Embedding and are trainable in default.
        I tried a frozen variant on Baby and found this operation makes no difference.
        """
        from freerec.utils import import_pickle
        if cfg.vfile:
            vFeats = import_pickle(
                os.path.join(path, cfg.vfile)
            )
            self.vFeats = nn.Embedding.from_pretrained(vFeats, freeze=False)
            self.image_trs = nn.Linear(self.vFeats.weight.size(1), cfg.embedding_dim)
            nn.init.xavier_normal_(self.image_trs.weight)

        if cfg.tfile:
            tFeats = import_pickle(
                os.path.join(path, cfg.tfile)
            )
            self.tFeats = nn.Embedding.from_pretrained(tFeats, freeze=False)
            self.text_trs = nn.Linear(self.tFeats.weight.size(1), cfg.embedding_dim)
            nn.init.xavier_normal_(self.text_trs.weight)

        return

    def reset_parameters(self):
        nn.init.xavier_normal_(self.User.embeddings.weight)
        nn.init.xavier_normal_(self.Item.embeddings.weight)

    def sure_trainpipe(self, batch_size: int):
        return self.dataset.train().shuffled_pairs_source(
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
        return userEmbds, itemEmbds + self.Item.embeddings.weight
    
    def reg_loss(self, *embeddings, norm=2):
        emb_loss = torch.zeros(1).to(embeddings[-1].device)
        for embedding in embeddings:
            emb_loss += torch.norm(embedding, p=norm)
        emb_loss /= embeddings[-1].shape[0]
        return emb_loss

    def fit(self, data: Dict[freerec.data.fields.Field, torch.Tensor]):
        users, items = data[self.User], data[self.Item]
        u_online_ori, i_online_ori = self.encode()
        t_feat_online = self.text_trs(self.tFeats.weight) if cfg.tfile else None
        v_feat_online = self.image_trs(self.vFeats.weight) if cfg.vfile else None

        with torch.no_grad():
            u_target, i_target = u_online_ori.clone(), i_online_ori.clone()
            u_target.detach()
            i_target.detach()
            u_target = F.dropout(u_target, cfg.dropout_rate)
            i_target = F.dropout(i_target, cfg.dropout_rate)

            if cfg.tfile:
                t_feat_target = t_feat_online.clone()
                t_feat_target = F.dropout(t_feat_target, cfg.dropout_rate)

            if cfg.vfile:
                v_feat_target = v_feat_online.clone()
                v_feat_target = F.dropout(v_feat_target, cfg.dropout_rate)

        u_online, i_online = self.predictor(u_online_ori), self.predictor(i_online_ori)

        # users, items = interactions[0], interactions[1]
        u_online = u_online[users, :]
        i_online = i_online[items, :]
        u_target = u_target[users, :]
        i_target = i_target[items, :]

        loss_t, loss_v, loss_tv, loss_vt = 0.0, 0.0, 0.0, 0.0
        if cfg.tfile:
            t_feat_online = self.predictor(t_feat_online)
            t_feat_online = t_feat_online[items, :]
            t_feat_target = t_feat_target[items, :]
            loss_t = 1 - cosine_similarity(t_feat_online, i_target.detach(), dim=-1).mean()
            loss_tv = 1 - cosine_similarity(t_feat_online, t_feat_target.detach(), dim=-1).mean()
        if cfg.vfile:
            v_feat_online = self.predictor(v_feat_online)
            v_feat_online = v_feat_online[items, :]
            v_feat_target = v_feat_target[items, :]
            loss_v = 1 - cosine_similarity(v_feat_online, i_target.detach(), dim=-1).mean()
            loss_vt = 1 - cosine_similarity(v_feat_online, v_feat_target.detach(), dim=-1).mean()

        loss_ui = 1 - cosine_similarity(u_online, i_target.detach(), dim=-1).mean()
        loss_iu = 1 - cosine_similarity(i_online, u_target.detach(), dim=-1).mean()

        return (loss_ui + loss_iu).mean() + cfg.reg_weight * self.reg_loss(u_online_ori, i_online_ori, norm=2) + \
               cfg.second_l * (loss_t + loss_v + loss_tv + loss_vt).mean()

    def reset_ranking_buffers(self):
        """This method will be executed before evaluation."""
        u_online, i_online = self.encode()
        userEmbds, itemEmbds = self.predictor(u_online), self.predictor(i_online)
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


class CoachForBM3(freerec.launcher.Coach):

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

    model = BM3(
        dataset
    )

    trainpipe = model.sure_trainpipe(cfg.batch_size)
    validpipe = model.sure_validpipe(cfg.ranking)
    testpipe = model.sure_testpipe(cfg.ranking)

    coach = CoachForBM3(
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