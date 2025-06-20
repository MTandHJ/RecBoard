

from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import freerec
from freerec.data.tags import TIMESTAMP, SEQUENCE

from modules import RelativeBucketedTimeAndPositionBasedBias, \
    LearnablePositionalEmbeddingInputFeaturesPreprocessor, \
    HSTUBlock, truncated_normal
from sampler import shuffled_time_seqs_source

freerec.declare(version='1.0.1')

cfg = freerec.parser.Parser()
cfg.add_argument("--maxlen", type=int, default=50)
cfg.add_argument("--num-heads", type=int, default=8)
cfg.add_argument("--num-blocks", type=int, default=16)
cfg.add_argument("--embedding-dim", type=int, default=64)
cfg.add_argument("--linear-hidden-dim", type=int, default=8)
cfg.add_argument("--attention-dim", type=int, default=8)
cfg.add_argument("--emb-dropout-rate", type=float, default=0.)
cfg.add_argument("--hidden-dropout-rate", type=float, default=0.)
cfg.add_argument("--num-negs", type=int, default=512)
cfg.add_argument("--num-buckets", type=int, default=100)
cfg.add_argument("--temperature", type=float, default=0.05)

cfg.set_defaults(
    description="HSTU",
    root="../../data",
    dataset='Amazon2014Beauty_550_LOU',
    epochs=200,
    batch_size=256,
    optimizer='AdamW',
    lr=1e-3,
    weight_decay=0.,
    seed=1,
)
cfg.compile()


class HSTU(freerec.models.SeqRecArch):

    def __init__(
        self, dataset: freerec.data.datasets.RecDataSet
    ) -> None:
        super().__init__(dataset)

        self.num_blocks = cfg.num_blocks

        self.Item.add_module(
            'embeddings', nn.Embedding(
                num_embeddings=self.Item.count + self.NUM_PADS,
                embedding_dim=cfg.embedding_dim,
                padding_idx=self.PADDING_VALUE
            )
        )
        self.Time = self.fields[TIMESTAMP].fork(SEQUENCE)

        # 0 1 1 ...
        # 0 0 1 ...
        # 0 0 0 ...
        # ....
        # `1` indices that the corresponding position is not allowed to attend!
        self.register_buffer(
            'attnMask',
            torch.ones((cfg.maxlen, cfg.maxlen), dtype=torch.float).triu(diagonal=1)
        )

        self.input_processor = LearnablePositionalEmbeddingInputFeaturesPreprocessor(
            maxlen=cfg.maxlen, embedding_dim=cfg.embedding_dim,
            dropout_rate=cfg.emb_dropout_rate
        )

        self.hstus = nn.ModuleList([
            HSTUBlock(
                embedding_dim=cfg.embedding_dim,
                linear_hidden_dim=cfg.linear_hidden_dim,
                linear_activation='silu',
                attention_dim=cfg.attention_dim,
                num_heads=cfg.num_heads,
                dropout_rate=cfg.hidden_dropout_rate,
                rel_attn_bias_encoder=RelativeBucketedTimeAndPositionBasedBias(
                    maxlen=cfg.maxlen,
                    num_buckets=cfg.num_buckets
                )
            )
            for _ in range(self.num_blocks)
        ])

        self.criterion = freerec.criterions.CrossEntropy4Logits(reduction='mean')

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                truncated_normal(m.weight, mean=0., std=0.02)

    def sure_trainpipe(self, maxlen: int, batch_size: int):
        return shuffled_time_seqs_source(
            dataset=self.dataset.train(), maxlen=maxlen
        ).time_seq_train_yielding_pos_(
            start_idx_for_target=1, end_idx_for_input=-1
        ).add_(
            offset=self.NUM_PADS, modified_fields=(self.ISeq,)
        ).lpad_(
            maxlen, modified_fields=(self.ISeq, self.Time, self.IPos),
            padding_value=self.PADDING_VALUE
        ).batch_(batch_size).tensor_()

    def sure_validpipe(self, maxlen, ranking = 'full', batch_size = 512):
        return self.dataset.valid().ordered_user_ids_source(
        ).time_valid_sampling_(
            ranking
        ).lprune_(
            maxlen, modified_fields=(self.ISeq, self.Time)
        ).add_(
            offset=self.NUM_PADS, modified_fields=(self.ISeq,)
        ).lpad_(
            maxlen, modified_fields=(self.ISeq, self.Time), 
            padding_value=self.PADDING_VALUE
        ).batch_(batch_size).tensor_()

    def sure_testpipe(self, maxlen, ranking = 'full', batch_size = 512):
        return self.dataset.test().ordered_user_ids_source(
        ).time_test_sampling_(ranking).lprune_(
            maxlen, modified_fields=(self.ISeq, self.Time)
        ).add_(
            offset=self.NUM_PADS, modified_fields=(self.ISeq,)
        ).lpad_(
            maxlen, modified_fields=(self.ISeq, self.Time), 
            padding_value=self.PADDING_VALUE
        ).batch_(batch_size).tensor_()

    def encode(
        self, data: Dict[freerec.data.fields.Field, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        seqs = data[self.ISeq]
        timestamps = data[self.Time]

        padding_mask = (seqs == self.PADDING_VALUE).unsqueeze(-1) # (B, L, 1)
        seqs = self.Item.embeddings(seqs) # (B, S) -> (B, S, D)
        seqs = self.input_processor(seqs)
        seqs = seqs.masked_fill(padding_mask, 0.)

        for l in range(self.num_blocks):
            seqs = self.hstus[l](
                seqs, timestamps, self.attnMask
            )
            # !!!
            seqs.masked_fill(padding_mask, 0.)

        userEmbds = F.normalize(seqs, dim=-1)

        return userEmbds, F.normalize(
            self.Item.embeddings.weight[self.NUM_PADS:],
            dim=-1
        ) 

    def _sample_negatives(self, userEmbds: torch.Tensor):
        num_users = userEmbds.size(0)
        negatives = torch.randint(
            0, self.Item.count, size=(num_users, cfg.num_negs),
            device=self.device
        )
        return negatives

    def fit(
        self, data: Dict[freerec.data.fields.Field, torch.Tensor]
    ) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        userEmbds, itemEmbds = self.encode(data)
        indices = data[self.ISeq] != self.PADDING_VALUE
        userEmbds = userEmbds[indices] # (M, D)
        positives = data[self.IPos][indices].unsqueeze(dim=-1) # (M, 1)
        negatives = self._sample_negatives(userEmbds) # (M, K)
        itemEmbds = itemEmbds[
            torch.cat((positives, negatives), dim=1)
        ] # (M, K + 1, D)

        logits = torch.einsum("MD,MKD->MK", userEmbds, itemEmbds) / cfg.temperature # (M, K)
        labels = torch.zeros_like(positives, dtype=torch.long).flatten() # (M,)
        rec_loss = self.criterion(logits, labels)

        return rec_loss

    def recommend_from_full(
        self, data: Dict[freerec.data.fields.Field, torch.Tensor]
    ) -> torch.Tensor:
        userEmbds, itemEmbds = self.encode(data)
        userEmbds = userEmbds[:, -1, :] # (B, D)
        return torch.einsum("BD,ND->BN", userEmbds, itemEmbds)

    def recommend_from_pool(
        self, data: Dict[freerec.data.fields.Field, torch.Tensor]
    ) -> torch.Tensor:
        userEmbds, itemEmbds = self.encode(data)
        userEmbds = userEmbds[:, -1, :] # (B, D)
        itemEmbds = itemEmbds[data[self.IUnseen]] # (B, K, D)
        return torch.einsum("BD,BKD->BK", userEmbds, itemEmbds)


class CoachForHSTU(freerec.launcher.Coach):

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

    dataset: freerec.data.datasets.RecDataSet
    try:
        dataset = getattr(freerec.data.datasets, cfg.dataset)(root=cfg.root)
    except AttributeError:
        dataset = freerec.data.datasets.RecDataSet(cfg.root, cfg.dataset, tasktag=cfg.tasktag)

    model = HSTU(dataset)

    # datapipe
    trainpipe = model.sure_trainpipe(cfg.maxlen, cfg.batch_size)
    validpipe = model.sure_validpipe(cfg.maxlen, ranking=cfg.ranking)
    testpipe = model.sure_testpipe(cfg.maxlen, ranking=cfg.ranking)

    coach = CoachForHSTU(
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