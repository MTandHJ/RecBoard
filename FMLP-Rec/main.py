

from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
import freerec

from modules import Encoder, LayerNorm

freerec.declare(version='1.0.1')

cfg = freerec.parser.Parser()
cfg.add_argument("--maxlen", type=int, default=50)
cfg.add_argument("--embedding-dim", default=64, type=int, help="hidden size of model")
cfg.add_argument("--num-blocks", default=2, type=int, help="number of filter-enhanced blocks")
cfg.add_argument("--num-heads", default=2, type=int)
cfg.add_argument("--hidden-act", default="gelu", type=str)
cfg.add_argument("--hidden-dropout-rate", default=0.5, type=float)
cfg.add_argument("--loss", type=str, choices=('BPR', 'BCE', 'CE'), default='BPR')

cfg.set_defaults(
    description="FMLP-Rec",
    root="../../data",
    dataset='Amazon2014Beauty_550_LOU',
    epochs=200,
    batch_size=512,
    optimizer='adam',
    lr=1e-3,
    weight_decay=0.,
    seed=1,
)
cfg.compile()


cfg.hidden_size = cfg.embedding_dim


class FMLPRec(freerec.models.SeqRecArch):

    def __init__(
        self, dataset: freerec.data.datasets.RecDataSet,
        maxlen: int = 50, embedding_dim: int = 64, 
        hidden_dropout_rate: float = 0.5
    ) -> None:
        super().__init__(dataset)

        self.embedding_dim = embedding_dim

        self.Item.add_module(
            'embeddings', nn.Embedding(
                num_embeddings=self.Item.count + self.NUM_PADS,
                embedding_dim=embedding_dim,
                padding_idx=self.PADDING_VALUE
            )
        )

        self.Position = nn.Embedding(maxlen, embedding_dim)
        self.embdDropout = nn.Dropout(p=hidden_dropout_rate)
        self.register_buffer(
            "positions",
            torch.tensor(range(0, maxlen), dtype=torch.long).unsqueeze(0)
        )

        self.layerNorm = LayerNorm(embedding_dim, eps=1.e-12)
        self.itemEncoder = Encoder(cfg)

        if cfg.loss == 'BCE':
            self.criterion = freerec.criterions.BCELoss4Logits(reduction='mean')
        elif cfg.loss == 'BPR':
            self.criterion = freerec.criterions.BPRLoss(reduction='mean')
        elif cfg.loss == 'CE':
            self.criterion = freerec.criterions.CrossEntropy4Logits(reduction='mean')

        self.reset_parameters()

    def reset_parameters(self):
        """Initializes the module parameters."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0., std=0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0., std=0.02)
            elif isinstance(m, LayerNorm):
                m.bias.data.zero_()
                m.weight.data.fill_(1.)

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
        S = seqs.size(1)
        positions = torch.arange(0, S, dtype=torch.long, device=self.device).unsqueeze(0)
        positions = self.Position(positions) # (1, maxlen, D)
        return self.embdDropout(self.layerNorm(seqs + positions))

    def create_mask(self, seqs: torch.Tensor):
        # seqs: (B, S)
        padding_mask = seqs.ne(self.PADDING_VALUE).long() # (B, S)
        attnMask = padding_mask.unsqueeze(1).unsqueeze(2) # (B, 1, 1, S)
        max_len = padding_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        causalMask = torch.triu(torch.ones(attn_shape), diagonal=1).to(self.device)
        causalMask = (causalMask == 0).unsqueeze(1).long() # (1, S, S)

        attnMask = attnMask * causalMask
        attnMask = (1.0 - attnMask) * -10000.0

        return attnMask # (B, 1, S, S)

    def encode(
        self, data: Dict[freerec.data.fields.Field, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        seqs = data[self.ISeq]
        attnMask = self.create_mask(seqs)
        seqs = self.Item.embeddings(seqs) # (B, S) -> (B, S, D)
        seqs = self.mark_position(seqs)

        userEmbds = self.itemEncoder(
            seqs, attnMask,
            output_all_encoded_layers=True
        )[-1] # (B, S, D)
        userEmbds = userEmbds[:, -1, :] # (B, D)

        return userEmbds, self.Item.embeddings.weight[self.NUM_PADS:]

    def fit(self, data: Dict[freerec.data.fields.Field, torch.Tensor]):
        userEmbds, itemEmbds = self.encode(data)
        posEmbds = itemEmbds[data[self.IPos]] # (B, 1, D)
        negEmbds = itemEmbds[data[self.INeg]] # (B, 1, K, D)

        posLogits = torch.einsum("BD,BSD->BS", userEmbds, posEmbds)
        negLogits = torch.einsum("BD,BSKD->BK", userEmbds, negEmbds)
        if cfg.loss in ('BCE', 'BPR'):
            posEmbds = itemEmbds[data[self.IPos]] # (B, 1, D)
            negEmbds = itemEmbds[data[self.INeg]] # (B, 1, K, D)
            posLogits = torch.einsum("BD,BSD->BS", userEmbds, posEmbds) # (B, 1)
            negLogits = torch.einsum("BD,BSKD->BK", userEmbds, negEmbds) # (B, K)

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

    def recommend_from_full(self, data: Dict[freerec.data.fields.Field, torch.Tensor]):
        userEmbds, itemEmbds = self.encode(data)
        return torch.einsum("BD,ND->BN", userEmbds, itemEmbds)

    def recommend_from_pool(self, data: Dict[freerec.data.fields.Field, torch.Tensor]):
        userEmbds, itemEmbds = self.encode(data)
        itemEmbds = itemEmbds[data[self.IUnseen]] # (B, K, D)
        return torch.einsum("BD,BKD->BK", userEmbds, itemEmbds)


class CoachForFMLPRec(freerec.launcher.Coach):

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

    model = FMLPRec(
        dataset, maxlen=cfg.maxlen, 
        embedding_dim=cfg.embedding_dim, hidden_dropout_rate=cfg.hidden_dropout_rate
    )

    # datapipe
    trainpipe = model.sure_trainpipe(cfg.maxlen, cfg.batch_size)
    validpipe = model.sure_validpipe(cfg.maxlen, ranking=cfg.ranking)
    testpipe = model.sure_testpipe(cfg.maxlen, ranking=cfg.ranking)

    coach = CoachForFMLPRec(
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