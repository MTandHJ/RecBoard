

from typing import Dict

import torch
import torch.nn as nn
import freerec

freerec.declare(version='1.0.1')

cfg = freerec.parser.Parser()
cfg.add_argument("--maxlen", type=int, default=50)
cfg.add_argument("--embedding-dim", type=int, default=64)
cfg.add_argument("--dropout-rate", type=float, default=0.2)
cfg.add_argument("--alpha", type=float, default=0.2)

cfg.set_defaults(
    description="RUM",
    root="../../data",
    dataset='Amazon2014Beauty_550_LOU',
    epochs=1000,
    batch_size=2048,
    optimizer='adam',
    lr=1e-3,
    weight_decay=1e-4,
    seed=1
)
cfg.compile()


class RUM(freerec.models.SeqRecArch):

    def __init__(
        self, dataset: freerec.data.datasets.RecDataSet,
        embedding_dim: int = 64, dropout_rate: float = 0.,
        alpha: float = 0.2
    ) -> None:
        super().__init__(dataset)

        self.alpha = alpha

        self.User.add_module(
            "embeddings", nn.Embedding(
                self.User.count, embedding_dim
            )
        )

        self.Item.add_module(
            "embeddings", nn.Embedding(
                self.Item.count + self.NUM_PADS, embedding_dim
            )
        )

        self.dropout = nn.Dropout(p=dropout_rate)

        self.criterion = freerec.criterions.BPRLoss(reduction='mean')

        self.reset_parameters()

    def sure_trainpipe(self, maxlen: int, batch_size: int):
        return self.dataset.train().shuffled_seqs_source(
            maxlen
        ).gen_train_sampling_pos_().gen_train_sampling_neg_(
            num_negatives=1
        ).add_(
            self.NUM_PADS, modified_fields=(self.ISeq,)
        ).lpad_(
            maxlen, modified_fields=(self.ISeq,),
            padding_value=self.PADDING_VALUE
        ).batch_(batch_size).tensor_()

    def encode(self, data: Dict[freerec.data.fields.Field, torch.Tensor]):
        seqs = data[self.ISeq]
        seqEmbds = self.Item.embeddings(seqs) # (B, S, D)
        return self.User.embeddings.weight, seqEmbds, self.Item.embeddings.weight[self.NUM_PADS:]

    def read_memory(
        self, 
        seqEmbds: torch.Tensor, # (B, S, D)
        itemEmbds: torch.Tensor, # (B, K + 1, D)
        padding_mask: torch.Tensor # (B, S)
    ):
        sim_matrix = torch.einsum("BSD,BKD->BSK", seqEmbds, itemEmbds) # (B, S, K + 1)
        sim_matrix.masked_fill_(
            padding_mask.unsqueeze(-1).expand_as(sim_matrix),
            -1e23
        )
        sim_matrix = sim_matrix.softmax(dim=1)
        return torch.einsum("BSK,BSD->BKD", sim_matrix, seqEmbds) # (B, K+1, D)

    def fit(self, data: Dict[freerec.data.fields.Field, torch.Tensor]):
        userEmbds, seqEmbds, itemEmbds = self.encode(data)
        users, positives, negatives = data[self.User], data[self.IPos], data[self.INeg]
        userEmbds = userEmbds[users] # (B, 1, D)
        iposEmbds = itemEmbds[positives] # (B, 1, D)
        inegEmbds = itemEmbds[negatives] # (B, K, D)
        itemEmbds = torch.cat((iposEmbds, inegEmbds), dim=1) # (B, K+1, D)

        userEmbds = userEmbds + self.alpha * self.read_memory(
            seqEmbds, itemEmbds, data[self.ISeq].eq(self.PADDING_VALUE)
        )
        scores = self.dropout(userEmbds.mul(itemEmbds)).sum(-1) # ???

        rec_loss = self.criterion(
            scores[:, :1], scores[:, 1:]
        )

        return rec_loss

    def recommend_from_full(self, data: Dict[freerec.data.fields.Field, torch.Tensor]):
        raise NotImplementedError("RUM does not support for full ranking ...")

    def recommend_from_pool(self, data: Dict[freerec.data.fields.Field, torch.Tensor]):
        userEmbds, seqEmbds, itemEmbds = self.encode(data)
        users, items = data[self.User], data[self.IUnseen]
        userEmbds = userEmbds[users] # (B, 1, D)
        itemEmbds = itemEmbds[items] # (B, 101, D)

        userEmbds = userEmbds + self.alpha * self.read_memory(
            seqEmbds, itemEmbds, data[self.ISeq].eq(self.PADDING_VALUE)
        )
        return torch.einsum("BKD,BKD->BK", userEmbds, itemEmbds)


class CoachForRUM(freerec.launcher.Coach):

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
    
    model = RUM(
        dataset,
        embedding_dim=cfg.embedding_dim,
        dropout_rate=cfg.dropout_rate,
        alpha=cfg.alpha
    )
    trainpipe = model.sure_trainpipe(cfg.maxlen, cfg.batch_size)
    validpipe = model.sure_validpipe(cfg.maxlen, cfg.ranking)
    testpipe = model.sure_testpipe(cfg.maxlen, cfg.ranking)

    coach = CoachForRUM(
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