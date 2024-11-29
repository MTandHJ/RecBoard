

from typing import Dict

import torch
import torch.nn as nn
import freerec

freerec.declare(version='1.0.1')

cfg = freerec.parser.Parser()
cfg.add_argument("--maxlen", type=int, default=50)
cfg.add_argument("--num-heads", type=int, default=4)
cfg.add_argument("--num-blocks", type=int, default=2)
cfg.add_argument("--embedding-dim", type=int, default=64)
cfg.add_argument("--mask-ratio", type=float, default=0.3)
cfg.add_argument("--dropout-rate", type=float, default=0.2)

cfg.set_defaults(
    description="BERT4Rec",
    root="../../data",
    dataset='Amazon2014Beauty_550_LOU',
    epochs=200,
    batch_size=256,
    optimizer='adam',
    lr=1e-3,
    weight_decay=0.,
    seed=1,
)
cfg.compile()


class BERT4Rec(freerec.models.SeqRecArch):

    NUM_PADS = 2
    PADDING_VALUE = 0
    MASKING_VALUE = 1

    def __init__(
        self, dataset: freerec.data.datasets.RecDataSet,
        maxlen: int = 50, embedding_dim: int = 64,
        mask_ratio: float = 0.3, dropout_rate: float = 0.2, 
        num_blocks: int = 1, num_heads: int = 2,
    ) -> None:
        super().__init__(dataset)

        self.embedding_dim = embedding_dim
        self.mask_ratio = mask_ratio
        self.num_blocks = num_blocks

        self.Item.add_module(
            "embeddings", nn.Embedding(
                self.Item.count + self.NUM_PADS, embedding_dim,
                padding_idx=self.PADDING_VALUE
            )
        )

        self.Position = nn.Embedding(maxlen, embedding_dim)
        self.embdDropout = nn.Dropout(p=dropout_rate)
        self.register_buffer(
            "positions",
            torch.tensor(range(0, maxlen), dtype=torch.long).unsqueeze(0)
        )

        self.layernorm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                dim_feedforward=embedding_dim * 4,
                dropout=dropout_rate,
                activation='gelu',
                batch_first=True
            ),
            num_layers=num_blocks
        )

        self.fc = nn.Linear(embedding_dim, self.Item.count + self.NUM_PADS)

        self.criterion = freerec.criterions.CrossEntropy4Logits(reduction='mean')

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                m.weight.data.clamp_(-0.02, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight)
                m.weight.data.clamp_(-0.02, 0.02)

    def sure_trainpipe(self, maxlen: int, batch_size: int):
        return self.dataset.train().shuffled_seqs_source(
            maxlen
        ).add_(
            self.NUM_PADS, modified_fields=(self.ISeq,)
        ).lpad_(
            maxlen, modified_fields=(self.ISeq,), padding_value=self.PADDING_VALUE
        ).batch_(batch_size).tensor_()

    def sure_validpipe(self, maxlen: int, ranking: str = 'full', batch_size: int = 256):
        return self.dataset.valid().ordered_user_ids_source(
        ).valid_sampling_(
            ranking=ranking
        ).lprune_(
            maxlen - 1, modified_fields=(self.ISeq,)
        ).add_(
            self.NUM_PADS, modified_fields=(self.ISeq,)
        ).lpad_(
            maxlen - 1, modified_fields=(self.ISeq,), padding_value=self.PADDING_VALUE
        ).rpad_(
            maxlen, modified_fields=(self.ISeq,), padding_value=self.MASKING_VALUE
        ).batch_(batch_size).tensor_()

    def sure_testpipe(self, maxlen: int, ranking: str = 'full', batch_size: int = 256):
        return self.dataset.test().ordered_user_ids_source(
        ).test_sampling_(
            ranking=ranking
        ).lprune_(
            maxlen - 1, modified_fields=(self.ISeq,)
        ).add_(
            self.NUM_PADS, modified_fields=(self.ISeq,)
        ).lpad_(
            maxlen - 1, modified_fields=(self.ISeq,), padding_value=self.PADDING_VALUE
        ).rpad_(
            maxlen, modified_fields=(self.ISeq,), padding_value=self.MASKING_VALUE
        ).batch_(batch_size).tensor_()

    def mark_position(self, seqs: torch.Tensor):
        positions = self.Position(self.positions) # (1, maxlen, D)
        return seqs + positions
    
    def encode(self, data: Dict[freerec.data.fields.Field, torch.Tensor]):
        seqs = data[self.ISeq]
        padding_mask = seqs == self.PADDING_VALUE
        seqs = self.mark_position(self.Item.embeddings(seqs)) # (B, S) -> (B, S, D)
        seqs = self.dropout(self.layernorm(seqs))
        seqs = self.encoder(seqs, src_key_padding_mask=padding_mask) # (B, S, D)

        return seqs

    def random_mask(self, seqs: torch.Tensor, p: float):
        padding_mask = seqs == self.PADDING_VALUE
        rnds = torch.rand(seqs.size(), device=seqs.device)
        masked_seqs = torch.where(
            rnds < p, 
            torch.ones_like(seqs).fill_(self.MASKING_VALUE),
            seqs
        )
        masked_seqs.masked_fill_(padding_mask, self.PADDING_VALUE)
        masks = (masked_seqs == self.MASKING_VALUE) # (B, S)
        labels = seqs[masks]
        return masked_seqs, labels, masks

    def fit(self, data: Dict[freerec.data.fields.Field, torch.Tensor]):
        masked_seqs, labels, masks = self.random_mask(
            seqs=data[self.ISeq], p=self.mask_ratio
        )
        data[self.ISeq] = masked_seqs

        userEmbds = self.encode(data) # (B, S, D)
        logits = self.fc(userEmbds)[masks] # (B, S, N)
        rec_loss = self.criterion(logits, labels)

        return rec_loss

    def recommend_from_full(self, data: Dict[freerec.data.fields.Field, torch.Tensor]):
        userEmbds = self.encode(data)
        userEmbds = userEmbds[:, -1, :] # (B, D)
        return self.fc(userEmbds)[:, self.NUM_PADS:]

    def recommend_from_pool(self, data: Dict[freerec.data.fields.Field, torch.Tensor]):
        userEmbds = self.encode(data)
        userEmbds = userEmbds[:, -1, :] # (B, D)
        scores = self.fc(userEmbds)[:, self.NUM_PADS:] # (B, N)
        return scores.gather(1, data[self.IUnseen]) # (B, 101)

class CoachForBERT4Rec(freerec.launcher.Coach):

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

    model = BERT4Rec(
        dataset, 
        maxlen=cfg.maxlen, embedding_dim=cfg.embedding_dim, 
        mask_ratio=cfg.mask_ratio, dropout_rate=cfg.dropout_rate,
        num_blocks=cfg.num_blocks, num_heads=cfg.num_heads
    )

    # datapipe
    trainpipe = model.sure_trainpipe(cfg.maxlen, cfg.batch_size)
    validpipe = model.sure_validpipe(cfg.maxlen, ranking=cfg.ranking)
    testpipe = model.sure_testpipe(cfg.maxlen, ranking=cfg.ranking)

    coach = CoachForBERT4Rec(
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