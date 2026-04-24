from typing import Dict, Tuple

import freerec
import torch
import torch.nn as nn

freerec.declare(version="1.0.1")

cfg = freerec.parser.Parser()
cfg.add_argument("--maxlen", type=int, default=50, help="just for trainpipe")
cfg.add_argument("--embedding-dim", default=64, type=int, help="hidden size of model")
cfg.add_argument("--loss", type=str, choices=("BPR", "BCE", "CE"), default="BPR")

cfg.set_defaults(
    description="FPMC",
    root="../../data",
    dataset="Amazon2014Beauty_550_LOU",
    epochs=200,
    batch_size=512,
    optimizer="AdamW",
    lr=1e-3,
    weight_decay=0.0,
    seed=1,
)
cfg.compile()


class FPMC(freerec.models.SeqRecArch):
    """
    (user & last item) -> (i2u item embds * i2l item embds)
    -> scores -> BPR/BCE/CE loss
    """

    NUM_PADS = 0

    def __init__(
        self,
        dataset: freerec.data.datasets.RecDataSet,
    ) -> None:
        super().__init__(dataset)

        self.embedding_dim = cfg.embedding_dim

        self.User.add_module(
            "embeddings",
            nn.Embedding(
                num_embeddings=self.User.count,
                embedding_dim=cfg.embedding_dim,
            ),
        )

        self.Item.add_module(
            "embeddings",
            nn.ModuleDict(
                {
                    "i2u": nn.Embedding(self.Item.count, cfg.embedding_dim),
                    "i2l": nn.Embedding(self.Item.count, cfg.embedding_dim),
                    "l2i": nn.Embedding(self.Item.count, cfg.embedding_dim),
                }
            ),
        )

        if cfg.loss == "BCE":
            self.criterion = freerec.criterions.BCELoss4Logits(reduction="mean")
        elif cfg.loss == "BPR":
            self.criterion = freerec.criterions.BPRLoss(reduction="mean")
        elif cfg.loss == "CE":
            self.criterion = freerec.criterions.CrossEntropy4Logits(reduction="mean")

        self.reset_parameters()

    def reset_parameters(self):
        """Initializes the module parameters."""
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight)

    def sure_trainpipe(self, maxlen: int, batch_size: int):
        return (
            self.dataset.train()
            .shuffled_roll_seqs_source(
                minlen=2, maxlen=maxlen, keep_at_least_itself=True
            )
            .lprune_(
                # [1, 2, 3, 4] -> [3, 4]
                maxlen=2,
                modified_fields=(self.ISeq,),
            )
            .seq_train_yielding_pos_(
                # [3, 4]
                # Input: [3]
                # Target: [4]
                start_idx_for_target=-1,
                end_idx_for_input=-1,
            )
            .seq_train_sampling_neg_(num_negatives=1)
            .batch_(batch_size)
            .tensor_()
        )

    def sure_validpipe(self, maxlen: int, ranking: str = "full", batch_size: int = 512):
        return (
            self.dataset.valid()
            .ordered_user_ids_source()
            .valid_sampling_(ranking)
            .lprune_(maxlen=1, modified_fields=(self.ISeq,))
            .batch_(batch_size)
            .tensor_()
        )

    def sure_testpipe(self, maxlen: int, ranking: str = "full", batch_size: int = 512):
        return (
            self.dataset.test()
            .ordered_user_ids_source()
            .test_sampling_(ranking)
            .lprune_(maxlen=1, modified_fields=(self.ISeq,))
            .batch_(batch_size)
            .tensor_()
        )

    def encode(
        self, data: Dict[freerec.data.fields.Field, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        seqs = data[self.ISeq]

        userEmbds = torch.cat(
            (
                self.User.embeddings(data[self.User]).squeeze(1),
                self.Item.embeddings["l2i"](seqs).squeeze(1),
            ),
            dim=-1,
        )  # (B, 2D)

        itemEmbds = torch.cat(
            (
                self.Item.embeddings["i2u"].weight,
                self.Item.embeddings["i2l"].weight,
            ),
            dim=-1,
        ) # (B, 2D)

        return userEmbds, itemEmbds

    def fit(self, data: Dict[freerec.data.fields.Field, torch.Tensor]):
        userEmbds, itemEmbds = self.encode(data)

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


class CoachForFPMC(freerec.launcher.Coach):
    """Coach for FPMC training."""

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

    model = FPMC(dataset)

    # datapipe
    trainpipe = model.sure_trainpipe(cfg.maxlen, cfg.batch_size)
    validpipe = model.sure_validpipe(cfg.maxlen, ranking=cfg.ranking)
    testpipe = model.sure_testpipe(cfg.maxlen, ranking=cfg.ranking)

    coach = CoachForFPMC(
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
