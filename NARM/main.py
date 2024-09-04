

from typing import Dict

import torch
import torch.nn as nn
import freerec

freerec.declare(version='1.0.1')

cfg = freerec.parser.Parser()
cfg.add_argument("--maxlen", type=int, default=50)
cfg.add_argument("--embedding-dim", type=int, default=64)
cfg.add_argument("--hidden-size", type=int, default=128)
cfg.add_argument("--emb-dropout-rate", type=float, default=0.2)
cfg.add_argument("--hidden-dropout-rate", type=float, default=0.)
cfg.add_argument("--ct-dropout-rate", type=float, default=0.5)
cfg.add_argument("--num-blocks", type=int, default=1, help="the number of GRU layers")

cfg.set_defaults(
    description="NARM",
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


class NARM(freerec.models.SeqRecArch):

    def __init__(
        self, dataset: freerec.data.datasets.RecDataSet,
        embedding_dim: int = 64, hidden_size: int = 128,
        emb_dropout_rate: float = 0.2, 
        hidden_dropout_rate: float = 0., 
        ct_dropout_rate: float = 0.5,
        num_blocks: int = 1
    ) -> None:
        super().__init__(dataset)

        self.embedding_dim = embedding_dim
        self.num_blocks = num_blocks

        self.Item.add_module(
            'embeddings', nn.Embedding(
                num_embeddings=self.Item.count + self.NUM_PADS,
                embedding_dim=embedding_dim,
                padding_idx=self.PADDING_VALUE
            )
        )
        self.emb_dropout = nn.Dropout(emb_dropout_rate)

        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_blocks,
            bias=False,
            batch_first=True,
            dropout=hidden_dropout_rate
        )
        self.a_1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.a_2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_t = nn.Linear(hidden_size, 1, bias=False)
        self.ct_dropout = nn.Dropout(ct_dropout_rate)
        self.b = nn.Linear(2 * hidden_size, embedding_dim, bias=False)

        self.criterion = freerec.criterions.BCELoss4Logits(reduction='mean')

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight)

    def sure_trainpipe(self, maxlen: int, batch_size: int):
        return self.dataset.train().shuffled_roll_seqs_source(
            minlen=2, maxlen=None
        ).seq_train_yielding_pos_(
            start_idx_for_target=-1 # only last item as the target
        ).seq_train_sampling_neg_(
            num_negatives=1
        ).lprune_(
            maxlen, modified_fields=(self.ISeq,)
        ).add_(
            self.NUM_PADS, modified_fields=(self.ISeq,)
        ).rpad_( # [i, j, k, ..., 0, ..., 0]
            maxlen, modified_fields=(self.ISeq,), padding_value=self.PADDING_VALUE
        ).batch_(batch_size).tensor_()

    def sure_validpipe(self, maxlen: int, ranking: str = 'full', batch_size: int = 256):
        return self.dataset.valid().ordered_user_ids_source(
        ).valid_sampling_(
            ranking
        ).lprune_(
            maxlen, modified_fields=(self.ISeq,)
        ).add_(
            self.NUM_PADS, modified_fields=(self.ISeq,)
        ).rpad_(
            maxlen, modified_fields=(self.ISeq,), padding_value=self.PADDING_VALUE
        ).batch_(batch_size).tensor_()

    def sure_testpipe(self, maxlen: int, ranking: str = 'full', batch_size: int = 256):
        return self.dataset.test().ordered_user_ids_source(
        ).test_sampling_(
            ranking
        ).lprune_(
            maxlen, modified_fields=(self.ISeq,)
        ).add_(
            self.NUM_PADS, modified_fields=(self.ISeq,)
        ).rpad_(
            maxlen, modified_fields=(self.ISeq,), padding_value=self.PADDING_VALUE
        ).batch_(batch_size).tensor_()

    def shrink_pads(self, seqs: torch.Tensor):
        mask = seqs.ne(self.PADDING_VALUE)
        keeped = mask.any(dim=0, keepdim=False)
        return seqs[:, keeped], mask[:, keeped].unsqueeze(-1) # (B, S), (B, S, 1)
    
    def encode(self, data: Dict[freerec.data.fields.Field, torch.Tensor]):
        seqs, mask = self.shrink_pads(data[self.ISeq])

        seqs = self.Item.embeddings(seqs)
        seqs = self.emb_dropout(seqs)
        gru_out, _ = self.gru(seqs) # (B, S, H)

        c_global = ht = gru_out.gather(
            dim=1,
            index=mask.sum(1, keepdim=True).add(-1).clamp_min(0).expand((-1, 1, gru_out.size(-1)))
        ) # (B, 1, H)

        q1 = self.a_1(gru_out) # (B, S, H)
        q2 = self.a_2(ht) # (B, 1, H)

        alpha = self.v_t(mask * torch.sigmoid(q1 + q2)) # (B, S, 1)
        c_local = torch.sum(alpha * gru_out, 1) # (B, H)
        c_t = torch.cat([c_local, c_global.squeeze(1)], 1) # (B, 2H)
        c_t = self.ct_dropout(c_t)
        userEmbds = self.b(c_t) # (B, D)
        return userEmbds, self.Item.embeddings.weight[self.NUM_PADS:]

    def fit(self, data: Dict[freerec.data.fields.Field, torch.Tensor]):
        userEmbds, itemEmbds = self.encode(data)
        posEmbds = itemEmbds[data[self.IPos]] # (B, 1, D)
        negEmbds = itemEmbds[data[self.INeg]] # (B, 1, D)

        posLogits = torch.einsum("BD,BSD->BS", userEmbds, posEmbds)
        negLogits = torch.einsum("BD,BSD->BS", userEmbds, negEmbds)
        posLabels = torch.ones_like(posLogits)
        negLabels = torch.zeros_like(negLogits)

        rec_loss = self.criterion(posLogits, posLabels) + \
            self.criterion(negLogits, negLabels)

        return rec_loss

    def recommend_from_full(self, data: Dict[freerec.data.fields.Field, torch.Tensor]):
        userEmbds, itemEmbds = self.encode(data)
        return torch.einsum("BD,ND->BN", userEmbds, itemEmbds)

    def recommend_from_pool(self, data: Dict[freerec.data.fields.Field, torch.Tensor]):
        userEmbds, itemEmbds = self.encode(data)
        itemEmbds = itemEmbds[data[self.IUnseen]] # (B, K, D)
        return torch.einsum("BD,BKD->BK", userEmbds, itemEmbds)


class CoachForNARM(freerec.launcher.Coach):

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

    model = NARM(
        dataset, 
        embedding_dim=cfg.embedding_dim, hidden_size=cfg.hidden_size,
        emb_dropout_rate=cfg.emb_dropout_rate, 
        hidden_dropout_rate=cfg.hidden_dropout_rate, 
        ct_dropout_rate=cfg.ct_dropout_rate,
        num_blocks=cfg.num_blocks
    )

    # datapipe
    trainpipe = model.sure_trainpipe(cfg.maxlen, cfg.batch_size)
    validpipe = model.sure_validpipe(cfg.maxlen, ranking=cfg.ranking)
    testpipe = model.sure_testpipe(cfg.maxlen, ranking=cfg.ranking)

    coach = CoachForNARM(
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