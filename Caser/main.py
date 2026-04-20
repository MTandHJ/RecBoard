from typing import Dict, Tuple, Union

import freerec
import torch
import torch.nn as nn
import torch.nn.functional as F

freerec.declare(version="1.0.1")

cfg = freerec.parser.Parser()
cfg.add_argument("--maxlen", type=int, default=5)
cfg.add_argument("--embedding-dim", type=int, default=64)
cfg.add_argument("--dropout-rate", type=float, default=0.5)
cfg.add_argument("--num-vert", type=int, default=4, help="number of vertical filters")
cfg.add_argument(
    "--num-horiz", type=int, default=16, help="number of horizontal filters"
)
cfg.add_argument("--num-poss", type=int, default=3, help="number of positive samples")
cfg.add_argument("--num-negs", type=int, default=3, help="number of negative samples")

cfg.set_defaults(
    description="Caser",
    root="../../data",
    dataset="Amazon2014Beauty_550_LOU",
    epochs=200,
    batch_size=256,
    optimizer="adam",
    lr=1e-3,
    weight_decay=0.0,
    seed=1,
)
cfg.compile()


class Caser(freerec.models.SeqRecArch):
    """
    item seq embds -> vertical Conv2d + horizontal Conv2d (multi-scale) -> ReLU -> max pool
    -> concat vert & horiz features -> dropout -> FC -> ReLU
    -> concat with user embd -> dot product with item W2 + bias
    -> BCE loss.
    """

    def __init__(
        self,
        dataset: freerec.data.datasets.RecDataSet,
    ) -> None:
        super().__init__(dataset)

        self.embedding_dim = cfg.embedding_dim

        self.User.add_module(
            "embeddings",
            nn.Embedding(
                num_embeddings=self.User.count, embedding_dim=cfg.embedding_dim
            ),
        )

        self.Item.add_module(
            "embeddings",
            nn.Embedding(
                num_embeddings=self.Item.count + self.NUM_PADS,
                embedding_dim=cfg.embedding_dim,
                padding_idx=self.PADDING_VALUE,
            ),
        )

        self.vert = nn.Conv2d(
            in_channels=1,
            out_channels=cfg.num_vert,
            kernel_size=(cfg.maxlen, 1),
            stride=1,
        )
        self.horizs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=1,
                    out_channels=cfg.num_horiz,
                    kernel_size=(k, cfg.embedding_dim),
                )
                for k in range(1, cfg.maxlen + 1)
            ]
        )
        self.pooling = nn.AdaptiveMaxPool1d((1,))

        self.fc_in_dims = cfg.num_vert * cfg.embedding_dim + cfg.num_horiz * cfg.maxlen

        self.fc1 = nn.Linear(self.fc_in_dims, cfg.embedding_dim)

        self.dropout = nn.Dropout(cfg.dropout_rate)

        self.W2 = nn.Embedding(self.Item.count, cfg.embedding_dim * 2)
        self.b2 = nn.Embedding(self.Item.count, 1)

        self.criterion = freerec.criterions.BCELoss4Logits(reduction="mean")

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight)
                nn.init.normal_(m.weight, std=1.0 / self.embedding_dim)
        self.b2.weight.data.zero_()

    def sure_trainpipe(
        self, maxlen: int, num_poss: int, num_negs: int, batch_size: int
    ):
        return (
            self.dataset.train()
            .shuffled_roll_seqs_source(
                minlen=num_poss + 1,
                maxlen=maxlen + num_poss,
                keep_at_least_itself=False,
            )
            .seq_train_yielding_pos_(
                start_idx_for_target=-num_poss, end_idx_for_input=-num_poss
            )
            .gen_train_sampling_neg_(num_negatives=num_negs)
            .add_(offset=self.NUM_PADS, modified_fields=(self.ISeq,))
            .lpad_(
                maxlen, modified_fields=(self.ISeq,), padding_value=self.PADDING_VALUE
            )
            .batch_(batch_size)
            .tensor_()
        )

    def encode(
        self, data: Dict[freerec.data.fields.Field, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        users, seqs = data[self.User], data[self.ISeq]
        seqEmbds = self.Item.embeddings(seqs).unsqueeze(1)  # (B, 1, S, D)
        userEmbds = self.User.embeddings(users).squeeze(1)  # (B, D)

        vert_features = self.vert(seqEmbds).flatten(1)
        horiz_features = [
            self.pooling(F.relu(conv(seqEmbds).squeeze(3))).squeeze(2)
            for conv in self.horizs
        ]
        horiz_features = torch.cat(horiz_features, dim=1)

        features = self.dropout(torch.cat((vert_features, horiz_features), dim=1))
        features = F.relu(self.fc1(features))
        features = torch.cat([features, userEmbds], dim=1)  # (B, 2D)

        return features

    def fit(
        self, data: Dict[freerec.data.fields.Field, torch.Tensor]
    ) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        userEmbds = self.encode(data)
        items = torch.cat((data[self.IPos], data[self.INeg]), dim=1)  # (B, k1 + k2)

        itemEmbds = self.W2(items)  # (B, K, 2D)
        itemBias = self.b2(items)  # (B, K, 1)

        scores = torch.baddbmm(itemBias, itemEmbds, userEmbds.unsqueeze(2)).squeeze(-1)
        posLogits, negLogits = torch.split(scores, [cfg.num_poss, cfg.num_negs], dim=1)

        posLabels = torch.ones_like(posLogits)
        negLabels = torch.zeros_like(negLogits)
        rec_loss = self.criterion(posLogits, posLabels) + self.criterion(
            negLogits, negLabels
        )

        return {"rec_loss": rec_loss}

    def recommend_from_full(
        self, data: Dict[freerec.data.fields.Field, torch.Tensor]
    ) -> torch.Tensor:
        userEmbds = self.encode(data)  # (B, 2D)
        itemEmbds = self.W2.weight  # (N, 2D)
        itemBias = self.b2.weight  # (N, 1)
        return torch.einsum("BD,ND->BN", userEmbds, itemEmbds) + itemBias.squeeze(1)

    def recommend_from_pool(
        self, data: Dict[freerec.data.fields.Field, torch.Tensor]
    ) -> torch.Tensor:
        userEmbds = self.encode(data)  # (B, 2D)
        items = data[self.IUnseen]

        itemEmbds = self.W2(items)  # (B, K, 2D)
        itemBias = self.b2(items)  # (B, K, 1)

        return torch.einsum("BD,BKD->BK", userEmbds, itemEmbds) + itemBias.squeeze(-1)


class CoachForCaser(freerec.launcher.Coach):
    """Coach for Caser training."""

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

    model = Caser(dataset)

    # datapipe
    trainpipe = model.sure_trainpipe(
        cfg.maxlen, cfg.num_poss, cfg.num_negs, cfg.batch_size
    )
    validpipe = model.sure_validpipe(cfg.maxlen, ranking=cfg.ranking)
    testpipe = model.sure_testpipe(cfg.maxlen, ranking=cfg.ranking)

    coach = CoachForCaser(
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
