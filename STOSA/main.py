
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

import freerec

from modules import LayerNorm, DistSAEncoder, wasserstein_distance, kl_distance

freerec.declare(version='1.0.1')

cfg = freerec.parser.Parser()
cfg.add_argument("--maxlen", type=int, default=50)
cfg.add_argument("--embedding-dim", type=int, default=64)
cfg.add_argument("--num-blocks", type=int, default=1)
cfg.add_argument("--num-heads", type=int, default=4)

cfg.add_argument("--hidden-dropout-rate", type=float, default=0.3)
cfg.add_argument("--attn-dropout-rate", type=float, default=0.)
cfg.add_argument("--distance-metric", type=str, choices=("wasserstein", "kl"), default="wasserstein")
cfg.add_argument("--pvn-weight", type=float, default=0.005, help="the weight for postives versus negatives")

cfg.set_defaults(
    description="STOSA",
    root="../../data",
    dataset='Amazon2014Beauty_550_LOU',
    epochs=500,
    batch_size=256,
    optimizer='adam',
    lr=1e-3,
    weight_decay=0.,
    seed=1,
)
cfg.compile()


class STOSA(freerec.models.SeqRecArch):

    def __init__(
        self, dataset: freerec.data.datasets.RecDataSet,
        maxlen: int = 50, embedding_dim: int = 64,
        num_heads: int = 4, num_blocks: int = 1,
        hidden_dropout_rate: float = 0.3,
        attn_dropout_rate: float = 0.,
        distance_metric: str = "wasserstein"
    ) -> None:
        super().__init__(dataset)

        self.embedding_dim = embedding_dim
        self.num_blocks = num_blocks
        self.distance_metric = distance_metric

        self.Item.add_module(
            "mean_embds",
            nn.Embedding(
                self.Item.count + self.NUM_PADS, 
                embedding_dim, padding_idx=self.PADDING_VALUE
            )
        )

        self.Item.add_module(
            "cov_embds",
            nn.Embedding(
                self.Item.count + self.NUM_PADS, 
                embedding_dim, padding_idx=self.PADDING_VALUE
            )
        )

        self.pos_mean_embds = nn.Embedding(cfg.maxlen, embedding_dim)
        self.pos_cov_embds = nn.Embedding(cfg.maxlen, embedding_dim)

        self.embdLN = LayerNorm(embedding_dim, eps=1e-12)
        self.embdDropout = nn.Dropout(p=hidden_dropout_rate)
        self.register_buffer(
            "positions",
            torch.tensor(range(0, cfg.maxlen), dtype=torch.long).unsqueeze(0)
        )

        self.encoder = DistSAEncoder(
            hidden_size=embedding_dim,
            num_heads=num_heads,
            num_layers=num_blocks,
            hidden_dropout_rate=hidden_dropout_rate,
            attn_dropout_rate=attn_dropout_rate,
            distance_metric=distance_metric
        )

        self.register_buffer(
            'attnMask',
            torch.ones((1, 1, maxlen, maxlen), dtype=torch.bool).tril() # (1, 1, maxlen, maxlen)
        )

        self.criterion = freerec.criterions.BPRLoss(reduction='mean')

        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.01, std=0.02)
            elif isinstance(module, LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

    def sure_trainpipe(self, maxlen: int, batch_size: int):
        return self.dataset.train().shuffled_seqs_source(
           maxlen=maxlen
        ).seq_train_yielding_pos_(
            start_idx_for_target=1, end_idx_for_input=-1
        ).seq_train_sampling_neg_(
            num_negatives=1
        ).add_(
            offset=self.NUM_PADS, modified_fields=(self.ISeq,)
        ).lpad_(
            maxlen, modified_fields=(self.ISeq, self.IPos, self.INeg),
            padding_value=self.PADDING_VALUE
        ).batch_(batch_size).tensor_()

    def mark_mean_pos(self, seqs: torch.Tensor):
        seqs = self.Item.mean_embds(seqs)
        positions = self.pos_mean_embds(self.positions) # (1, maxlen, D)
        seqs = seqs + positions
        seqs = self.embdLN(seqs)
        seqs = self.embdDropout(seqs)
        return F.elu(seqs)

    def mark_cov_pos(self, seqs: torch.Tensor):
        seqs = self.Item.cov_embds(seqs)
        positions = self.pos_cov_embds(self.positions) # (1, maxlen, D)
        seqs = seqs + positions
        seqs = self.embdLN(seqs)
        seqs = self.embdDropout(seqs)
        return F.elu(seqs) + 1 # positive semidefinite

    def encode(self, data: Dict[freerec.data.fields.Field, torch.Tensor]):
        seqs = data[self.ISeq]
        attn_mask = seqs.ne(self.PADDING_VALUE).unsqueeze(1).unsqueeze(2) # (B, 1, 1, S)
        attn_mask = attn_mask.logical_and(self.attnMask)
        attn_mask = (1. -  attn_mask.float()) * (- 2 ** 32 + 1)

        mean_seqs = self.mark_mean_pos(seqs)
        cov_seqs = self.mark_cov_pos(seqs)

        all_layer_items = self.encoder(
            mean_seqs, cov_seqs, attn_mask,
            output_all_encoded_layers=False
        )

        mean_users, cov_users, _ = all_layer_items[-1]
        return mean_users, cov_users, \
            self.Item.mean_embds.weight[self.NUM_PADS:], \
            self.Item.cov_embds.weight[self.NUM_PADS:]

    def pvn_loss(self, posLogits: torch.Tensor, pvnLogits: torch.Tensor):
        return  (pvnLogits - posLogits).clamp(0.).mean()

    def fit(self, data: Dict[freerec.data.fields.Field, torch.Tensor]):
        mean_users, cov_users, mean_items, cov_items = self.encode(data)
        indices = data[self.ISeq] != self.PADDING_VALUE
        mean_users = mean_users[indices] # (M, D)
        cov_users = cov_users[indices] # (M, D)

        mean_positives = mean_items[data[self.IPos][indices]] # (M, D)
        cov_positives = F.elu(cov_items[data[self.IPos][indices]]) + 1 # (M, D)
        mean_negatives = mean_items[data[self.INeg][indices]] # (M, D)
        cov_negatives = F.elu(cov_items[data[self.INeg][indices]]) + 1 # (M, D)

        if self.distance_metric == "wasserstein": 
            dist_func = wasserstein_distance
        elif self.distance_metric == "kl":
            dist_func = kl_distance

        posLogits = dist_func(
            mean_users, cov_users,
            mean_positives, cov_positives
        ).neg()

        negLogits = dist_func(
            mean_users, cov_users,
            mean_negatives, cov_negatives
        ).neg()

        pvnLogits = dist_func(
            mean_positives, cov_positives,
            mean_negatives, cov_negatives
        ).neg()

        rec_loss = self.criterion(posLogits, negLogits)
        pvn_loss = self.pvn_loss(posLogits, pvnLogits)

        return rec_loss, pvn_loss

    def recommend_from_full(self, data: Dict[freerec.data.fields.Field, torch.Tensor]):
        mean_users, cov_users, mean_items, cov_items = self.encode(data)

        mean_users = mean_users[:, [-1], :]
        cov_users = cov_users[:, [-1], :]

        mean_items = mean_items
        cov_items = F.elu(cov_items) + 1

        if self.distance_metric == "wasserstein": 
            dist_func = wasserstein_distance
        elif self.distance_metric == "kl":
            dist_func = kl_distance

        return dist_func(
            mean_users, cov_users,
            mean_items, cov_items
        ).neg()

    def recommend_from_pool(self, data: Dict[freerec.data.fields.Field, torch.Tensor]):
        mean_users, cov_users, mean_items, cov_items = self.encode(data)

        mean_users = mean_users[:, [-1], :]
        cov_users = cov_users[:, [-1], :]

        mean_items = mean_items[data[self.IUnseen]]
        cov_items = F.elu(cov_items[data[self.IUnseen]]) + 1

        if self.distance_metric == "wasserstein": 
            dist_func = wasserstein_distance
        elif self.distance_metric == "kl":
            dist_func = kl_distance

        return dist_func(
            mean_users, cov_users,
            mean_items, cov_items
        ).neg()


class CoachForSTOSA(freerec.launcher.Coach):

    def train_per_epoch(self, epoch: int):
        for data in self.dataloader:
            data = self.dict_to_device(data)
            rec_loss, pvn_loss = self.model(data)
            loss = rec_loss + pvn_loss * self.cfg.pvn_weight

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.monitor(
                loss.item(), 
                n=len(data[self.User]), reduction="mean", 
                mode='train', pool=['LOSS']
            )


def main():

    dataset: freerec.data.datasets.RecDataSet
    try:
        dataset = getattr(freerec.data.datasets, cfg.dataset)(root=cfg.root)
    except AttributeError:
        dataset = freerec.data.datasets.RecDataSet(cfg.root, cfg.dataset, tasktag=cfg.tasktag)

    model = STOSA(
        dataset,
        maxlen=cfg.maxlen, embedding_dim=cfg.embedding_dim,
        num_heads=cfg.num_heads, num_blocks=cfg.num_blocks,
        hidden_dropout_rate=cfg.hidden_dropout_rate,
        attn_dropout_rate=cfg.attn_dropout_rate,
        distance_metric=cfg.distance_metric
    )

    # datapipe
    trainpipe = model.sure_trainpipe(cfg.maxlen, cfg.batch_size)
    validpipe = model.sure_validpipe(cfg.maxlen, ranking=cfg.ranking)
    testpipe = model.sure_testpipe(cfg.maxlen, ranking=cfg.ranking)

    coach = CoachForSTOSA(
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