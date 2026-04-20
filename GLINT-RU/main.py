from typing import Dict

import freerec
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import MultiHeadAttention

freerec.declare(version="1.0.1")

cfg = freerec.parser.Parser()
cfg.add_argument("--maxlen", type=int, default=50)
cfg.add_argument("--num-heads", type=int, default=8)
cfg.add_argument("--num-layers", type=int, default=1)
cfg.add_argument("--embedding-dim", type=int, default=128)
cfg.add_argument("--hidden-size", type=int, default=128)
cfg.add_argument("--emb-dropout-rate", type=float, default=0.0)
cfg.add_argument("--hidden-dropout-rate", type=float, default=0.2)
cfg.add_argument("--attn-dropout-rate", type=float, default=0.2)
cfg.add_argument("--layer-norm-eps", type=float, default=1.0e-12)
cfg.add_argument("--loss", type=str, choices=("BPR", "BCE", "CE"), default="BCE")

cfg.set_defaults(
    description="GLINT-RU",
    root="../../data",
    dataset="Amazon2014Beauty_550_LOU",
    epochs=200,
    batch_size=2048,
    optimizer="Adam",
    lr=1e-3,
    weight_decay=0.0,
    seed=1,
)
cfg.compile()


class GLINTRU(freerec.models.SeqRecArch):
    """
    item embds -> dropout
    -> dual-path: (1) Conv1D -> GRU -> selective gate -> projection -> Conv1D,
    (2) multi-head linear attention
    -> softmax-weighted mixture-of-experts fusion -> element-wise multiply with GELU branch
    -> linear -> dropout -> LayerNorm residual
    -> SwiGLU-style FFN (dense3 * GELU(dense4)) -> linear -> dropout -> LayerNorm residual
    -> last-position gathering -> dot product with item embds -> BCE/BPR/CE loss.
    """

    def __init__(
        self,
        dataset: freerec.data.datasets.RecDataSet,
    ) -> None:
        super().__init__(dataset)

        self.embedding_dim = cfg.embedding_dim

        self.Item.add_module(
            "embeddings",
            nn.Embedding(
                num_embeddings=self.Item.count + self.NUM_PADS,
                embedding_dim=cfg.embedding_dim,
                padding_idx=self.PADDING_VALUE,
            ),
        )

        self.emb_dropout = nn.Dropout(cfg.emb_dropout_rate)
        self.dense1 = nn.Linear(self.embedding_dim, cfg.hidden_size)
        self.dense2 = nn.Linear(self.embedding_dim, cfg.hidden_size)
        self.conv1d = nn.Conv1d(
            cfg.hidden_size, cfg.hidden_size, kernel_size=3, padding=1
        )
        self.gru_layers = nn.GRU(
            input_size=cfg.hidden_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            bias=False,
            batch_first=True,
        )
        self.conv1dforgru = nn.Conv1d(
            cfg.hidden_size, cfg.hidden_size, kernel_size=3, padding=1
        )
        self.linearattention = MultiHeadAttention(
            cfg.num_heads,
            cfg.hidden_size,
            cfg.hidden_dropout_rate,
            cfg.attn_dropout_rate,
            cfg.layer_norm_eps,
        )
        self.weights = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32))
        self.dense = nn.Linear(cfg.hidden_size, cfg.hidden_size)

        self.dense3 = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.dense4 = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.denseout = nn.Linear(cfg.hidden_size, self.embedding_dim)
        self.dropdense = nn.Dropout(0.3)
        self.dropmix = nn.Dropout(0.3)
        self.LayerNorm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.gelu = nn.GELU()

        self.projection = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.selective_gate = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size // 2),
            nn.SiLU(),
            nn.Linear(cfg.hidden_size // 2, cfg.hidden_size),
            nn.Dropout(0.3),
        )

        if cfg.loss == "BCE":
            self.criterion = freerec.criterions.BCELoss4Logits(reduction="mean")
        elif cfg.loss == "BPR":
            self.criterion = freerec.criterions.BPRLoss(reduction="mean")
        elif cfg.loss == "CE":
            self.criterion = freerec.criterions.CrossEntropy4Logits(reduction="mean")

        self.reset_parameters()

    def reset_parameters(self):
        from torch.nn.init import xavier_normal_, xavier_uniform_

        for m in self.modules():
            if isinstance(m, nn.Embedding):
                xavier_normal_(m.weight)
            elif isinstance(m, nn.GRU):
                xavier_uniform_(m.weight_hh_l0)
                xavier_uniform_(m.weight_ih_l0)
            elif isinstance(m, nn.Linear):
                xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

    def sure_trainpipe(self, maxlen: int, batch_size: int):
        return (
            self.dataset.train()
            .shuffled_roll_seqs_source(minlen=2, maxlen=None)
            .seq_train_yielding_pos_(
                start_idx_for_target=-1  # last item as the target
            )
            .seq_train_sampling_neg_(num_negatives=1)
            .lprune_(maxlen, modified_fields=(self.ISeq,))
            .add_(self.NUM_PADS, modified_fields=(self.ISeq,))
            .rpad_(  # [i, j, k, ..., 0, ..., 0]
                maxlen, modified_fields=(self.ISeq,), padding_value=self.PADDING_VALUE
            )
            .batch_(batch_size)
            .tensor_()
        )

    def sure_validpipe(self, maxlen: int, ranking: str = "full", batch_size: int = 256):
        return (
            self.dataset.valid()
            .ordered_user_ids_source()
            .valid_sampling_(ranking)
            .lprune_(maxlen, modified_fields=(self.ISeq,))
            .add_(self.NUM_PADS, modified_fields=(self.ISeq,))
            .rpad_(
                maxlen, modified_fields=(self.ISeq,), padding_value=self.PADDING_VALUE
            )
            .batch_(batch_size)
            .tensor_()
        )

    def sure_testpipe(self, maxlen: int, ranking: str = "full", batch_size: int = 256):
        return (
            self.dataset.test()
            .ordered_user_ids_source()
            .test_sampling_(ranking)
            .lprune_(maxlen, modified_fields=(self.ISeq,))
            .add_(self.NUM_PADS, modified_fields=(self.ISeq,))
            .rpad_(
                maxlen, modified_fields=(self.ISeq,), padding_value=self.PADDING_VALUE
            )
            .batch_(batch_size)
            .tensor_()
        )

    def shrink_pads(self, seqs: torch.Tensor):
        mask = seqs.ne(self.PADDING_VALUE)
        keeped = mask.any(dim=0, keepdim=False)
        return seqs[:, keeped], mask[:, keeped].unsqueeze(-1)  # (B, S), (B, S, 1)

    def encode(self, data: Dict[freerec.data.fields.Field, torch.Tensor]):
        seqs, mask = self.shrink_pads(data[self.ISeq])

        self.gru_layers.flatten_parameters()
        seqEmbds = self.Item.embeddings(seqs)
        seqEmbds = self.emb_dropout(seqEmbds)
        # -------------------------First Layer-------------------------------
        attention_output = self.linearattention(seqEmbds)
        h1 = self.dense1(seqEmbds)
        h2 = self.dense2(seqEmbds)
        h1 = self.conv1d(h1.transpose(1, 2))
        h1 = h1.transpose(1, 2)
        h2 = self.gelu(h2)

        # -------------------------Mixed Temporal Block----------------------
        gru_output, _ = self.gru_layers(h1)
        selective_gate = self.selective_gate(h1)
        gru_output = self.projection(gru_output)
        gru_output = selective_gate * gru_output
        gru_output = self.conv1dforgru(gru_output.transpose(1, 2))
        gru_output = gru_output.transpose(1, 2)

        weights = F.softmax(self.weights, dim=0)
        expert_output = weights[0] * gru_output + weights[1] * attention_output
        h = expert_output * h2
        h = self.dense(h)
        h = self.dropmix(h)
        h = self.LayerNorm(h + seqEmbds)
        # the embedding of the predicted item, shape of (batch_size, embedding_size)

        x1 = self.dense3(h)
        x2 = self.dense4(h)
        x2 = self.gelu(x2)
        x = x1 * x2
        x = self.denseout(x)
        x = self.dropdense(x)
        x = self.LayerNorm(x + h)

        userEmbds = x.gather(
            dim=1,
            index=mask.sum(1, keepdim=True)
            .add(-1)
            .clamp_min(0)
            .expand((-1, 1, x.size(-1))),
            # clamp_min(0) used for empty sequence
        ).squeeze(1)  # (B, D)

        return userEmbds, self.Item.embeddings.weight[self.NUM_PADS :]

    def fit(self, data: Dict[freerec.data.fields.Field, torch.Tensor]):
        userEmbds, itemEmbds = self.encode(data)
        posEmbds = itemEmbds[data[self.IPos]]  # (B, 1, D)
        negEmbds = itemEmbds[data[self.INeg]]  # (B, 1, D)

        if cfg.loss in ("BCE", "BPR"):
            posEmbds = itemEmbds[data[self.IPos]]  # (B, 1, D)
            negEmbds = itemEmbds[data[self.INeg]]  # (B, 1, D)
            posLogits = torch.einsum("BD,BSD->BS", userEmbds, posEmbds)  # (B, 1)
            negLogits = torch.einsum("BD,BSD->BS", userEmbds, negEmbds)  # (B, 1)

            if cfg.loss == "BCE":
                posLabels = torch.ones_like(posLogits)
                negLabels = torch.zeros_like(negLogits)

                rec_loss = self.criterion(posLogits, posLabels) + self.criterion(
                    negLogits, negLabels
                )
            elif cfg.loss == "BPR":
                rec_loss = self.criterion(posLogits, negLogits)
        elif cfg.loss == "CE":
            logits = torch.einsum("BD,ND->BN", userEmbds, itemEmbds)  # (B, N)
            labels = data[self.IPos].flatten()  # (B, S)

            rec_loss = self.criterion(logits, labels)

        return {"rec_loss": rec_loss}

    def recommend_from_full(self, data: Dict[freerec.data.fields.Field, torch.Tensor]):
        userEmbds, itemEmbds = self.encode(data)
        return torch.einsum("BD,ND->BN", userEmbds, itemEmbds)

    def recommend_from_pool(self, data: Dict[freerec.data.fields.Field, torch.Tensor]):
        userEmbds, itemEmbds = self.encode(data)
        itemEmbds = itemEmbds[data[self.IUnseen]]  # (B, K, D)
        return torch.einsum("BD,BKD->BK", userEmbds, itemEmbds)


class CoachForGLINTRU(freerec.launcher.Coach):
    """Coach for GLINT-RU training."""

    def train_per_epoch(self, epoch: int):
        for data in self.dataloader:
            data = self.dict_to_device(data)
            losses = self.model(data)
            loss = losses["rec_loss"]

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.monitor(
                loss.item(),
                n=len(data[self.User]),
                reduction="mean",
                mode="train",
                pool=["LOSS"],
            )


def main():

    dataset: freerec.data.datasets.RecDataSet
    try:
        dataset = getattr(freerec.data.datasets, cfg.dataset)(root=cfg.root)
    except AttributeError:
        dataset = freerec.data.datasets.RecDataSet(
            cfg.root, cfg.dataset, tasktag=cfg.tasktag
        )

    model = GLINTRU(dataset)

    # datapipe
    trainpipe = model.sure_trainpipe(cfg.maxlen, cfg.batch_size)
    validpipe = model.sure_validpipe(cfg.maxlen, ranking=cfg.ranking)
    testpipe = model.sure_testpipe(cfg.maxlen, ranking=cfg.ranking)

    coach = CoachForGLINTRU(
        dataset=dataset,
        trainpipe=trainpipe,
        validpipe=validpipe,
        testpipe=testpipe,
        model=model,
        cfg=cfg,
    )
    coach.fit()


if __name__ == "__main__":
    main()
