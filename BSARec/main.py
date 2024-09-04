

from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
import freerec

from modules import BSARecBlock

freerec.declare(version='1.0.1')

cfg = freerec.parser.Parser()
cfg.add_argument("--maxlen", type=int, default=50)
cfg.add_argument("--num-heads", type=int, default=1)
cfg.add_argument("--num-blocks", type=int, default=2)
cfg.add_argument("--hidden-act", type=str, default='gelu')
cfg.add_argument("--embedding-dim", type=int, default=64)
cfg.add_argument("--hidden-dropout-rate", type=float, default=0.5)
cfg.add_argument("--atten-dropout-rate", type=float, default=0.5)
cfg.add_argument("--loss", type=str, choices=('BPR', 'BCE', 'CE'), default='CE')

cfg.add_argument("--c", type=int, default=5, help="the number of low-pass filters")
cfg.add_argument("--alpha", type=float, default=0.7, help="the ratio for frequency domain")

cfg.set_defaults(
    description="BSARec",
    root="../../data",
    dataset='Amazon2014Beauty_550_LOU',
    epochs=200,
    batch_size=256,
    optimizer='adam',
    lr=1.e-3,
    weight_decay=0.,
    seed=1,
)
cfg.compile()


cfg.hidden_size = cfg.embedding_dim


class BSARec(freerec.models.SeqRecArch):

    def __init__(
        self, dataset: freerec.data.datasets.RecDataSet,
    ) -> None:
        super().__init__(dataset)

        self.embedding_dim = cfg.embedding_dim
        self.num_blocks = cfg.num_blocks

        self.Item.add_module(
            'embeddings', nn.Embedding(
                num_embeddings=self.Item.count + self.NUM_PADS,
                embedding_dim=cfg.embedding_dim,
                padding_idx=self.PADDING_VALUE
            )
        )

        self.Position = nn.Embedding(cfg.maxlen, cfg.embedding_dim)
        self.embdDropout = nn.Dropout(p=cfg.hidden_dropout_rate)
        self.register_buffer(
            "positions",
            torch.tensor(range(0, cfg.maxlen), dtype=torch.long).unsqueeze(0)
        )

        self.layerNorm = nn.LayerNorm(cfg.embedding_dim, eps=1e-12)

        self.blocks = nn.ModuleList([
            BSARecBlock(cfg)
            for _ in range(cfg.num_blocks)
        ])

        if cfg.loss == 'BCE':
            self.criterion = freerec.criterions.BCELoss4Logits(reduction='mean')
        elif cfg.loss == 'BPR':
            self.criterion = freerec.criterions.BPRLoss(reduction='mean')
        elif cfg.loss == 'CE':
            self.criterion = freerec.criterions.CrossEntropy4Logits(reduction='mean')

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Embedding):
                m.weight.data.normal_(mean=0., std=0.02)
            elif isinstance(m, nn.LayerNorm):
                m.bias.data.zero_()
                m.weight.data.fill_(1.)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)

    def sure_trainpipe(self, maxlen: int, batch_size: int):
        return self.dataset.train().shuffled_roll_seqs_source(
            minlen=2, maxlen=maxlen, keep_at_least_itself=True
        ).seq_train_yielding_pos_(
            # [1, 2, 3, 4]
            # Input: [1, 2, 3] 
            # Target: [4]
            start_idx_for_target=-1, end_idx_for_input=-1
        ).seq_train_sampling_neg_(
            num_negatives=1
        ).add_(
            offset=self.NUM_PADS, modified_fields=(self.ISeq,)
        ).lpad_(
            maxlen, modified_fields=(self.ISeq,),
            padding_value=self.PADDING_VALUE
        ).batch_(batch_size).tensor_()

    def mark_position(self, seqs: torch.Tensor):
        positions = self.Position(self.positions) # (1, maxlen, D)
        return seqs + positions

    def after_one_block(self, seqs: torch.Tensor, attention_mask: torch.Tensor, l: int):
        # inputs: (B, S, D)
        return self.blocks[l](seqs, attention_mask)

    def extend_attention_mask(self, seqs: torch.Tensor):
        B, L = seqs.shape
        attention_mask = (seqs != self.PADDING_VALUE).unsqueeze(1).unsqueeze(2) # (B, 1, 1, L)
        attention_mask = attention_mask.expand(-1, -1, L, -1)
        attention_mask = torch.tril(attention_mask)
        attention_mask = torch.where(attention_mask, 0., -1.e4)
        return attention_mask
    
    def encode(
        self, data: Dict[freerec.data.fields.Field, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        seqs = data[self.ISeq]
        attention_mask = self.extend_attention_mask(seqs)

        seqs = self.Item.embeddings(seqs) # (B, S) -> (B, S, D)
        seqs = self.mark_position(seqs)
        seqs = self.layerNorm(seqs)
        seqs = self.embdDropout(seqs)

        for l in range(self.num_blocks):
            seqs = self.after_one_block(seqs, attention_mask, l)

        return seqs, self.Item.embeddings.weight[self.NUM_PADS:]

    def fit(self, data: Dict[freerec.data.fields.Field, torch.Tensor]):
        userEmbds, itemEmbds = self.encode(data)
        posEmbds = itemEmbds[data[self.IPos]] # (B, 1, D)
        negEmbds = itemEmbds[data[self.INeg]] # (B, 1, D)

        if cfg.loss in ('BCE', 'BPR'):
            posEmbds = itemEmbds[data[self.IPos]] # (B, 1, D)
            negEmbds = itemEmbds[data[self.INeg]] # (B, 1, D)
            posLogits = torch.einsum("BD,BSD->BS", userEmbds, posEmbds) # (B, 1)
            negLogits = torch.einsum("BD,BSD->BS", userEmbds, negEmbds) # (B, 1)

            if cfg.loss == 'BCE':
                posLabels = torch.ones_like(posLogits)
                negLabels = torch.zeros_like(negLogits)

                rec_loss = self.criterion(posLogits, posLabels) + \
                    self.criterion(negLogits, negLabels)
            elif cfg.loss == 'BPR':
                rec_loss = self.criterion(posLogits, negLogits)
        elif cfg.loss == 'CE':
            logits = torch.einsum("BD,ND->BN", userEmbds, itemEmbds) # (B, N)
            labels = data[self.IPos].flatten() # (B, S)

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


class CoachForBSARec(freerec.launcher.Coach):

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

    model = BSARec(dataset)

    # datapipe
    trainpipe = model.sure_trainpipe(cfg.maxlen, cfg.batch_size)
    validpipe = model.sure_validpipe(cfg.maxlen, ranking=cfg.ranking)
    testpipe = model.sure_testpipe(cfg.maxlen, ranking=cfg.ranking)

    coach = CoachForBSARec(
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