import json
import os
from typing import Dict

import freerec
import torch
import torch.nn as nn
import torch.nn.functional as F
from converter import SemIDConverter
from quantizer import ResidualQuantizer

freerec.declare(version="1.0.1")

cfg = freerec.parser.Parser()
cfg.add_argument("--num-codewords", type=int, default=256, help="number of codewords per codebook")
cfg.add_argument("--num-codebooks", type=int, default=3, help="number of codebooks")
cfg.add_argument("--codebook-dim", type=int, default=32, help="codebook dimension size")
cfg.add_argument("--hidden-dims", type=str, default="512,256,128")
cfg.add_argument("--apply-kmeans-init", type=eval, default=True)

cfg.add_argument(
    "--sk-epsilons", type=str, default="0.,0.,0.003", help="epsilon for sinkhorn iteration"
)
cfg.add_argument(
    "--sk-iters", type=int, default=50, help="number of iterations for sinkhorn iteration"
)

cfg.add_argument("--cf-weight", type=float, default=0.01, help="cf loss weight")
cfg.add_argument("--commit-weight", type=float, default=0.25, help="commitment loss weight")
cfg.add_argument("--diversity-weight", type=float, default=0.0001, help="diversity loss weight")

cfg.add_argument("--dropout-rate", type=float, default=0.1)
cfg.add_argument("--sem-feat-file", type=str, default=None, help="file of semantic features")
cfg.add_argument("--collab-feat-file", type=str, default=None, help="file of collaborative features")

cfg.set_defaults(
    description="LETTER-RQVAE",
    root="../../data",
    dataset="Amazon2014Beauty_550_LOU",
    epochs=20000,
    batch_size=1024,
    optimizer="AdamW",
    lr=1e-3,
    weight_decay=1.0e-4,
    seed=1,
)
cfg.compile()

cfg.hidden_dims = list(map(int, cfg.hidden_dims.split(",")))
cfg.sk_epsilons = list(map(float, cfg.sk_epsilons.split(",")))


class RQVAE(freerec.models.RecSysArch):
    r"""Train LETTER quantizers from textual and collaborative item features.

    Workflow
    --------
    1. encode semantic item features into a compact latent space.
    2. quantize the latent with residual codebooks.
    3. reconstruct normalized semantic features.
    4. align reconstructed features with collaborative features through batch
       contrastive loss.
    """

    def __init__(self, dataset: freerec.data.datasets.RecDataSet) -> None:
        super().__init__(dataset)

        self.Item.add_module(
            "semantic_embeddings",
            nn.Embedding.from_pretrained(
                F.normalize(
                    freerec.utils.import_pickle(
                        os.path.join(
                            self.dataset.path,
                            cfg.sem_feat_file,
                        )
                    ),
                    dim=-1,
                ),
                freeze=True,
            ),
        )

        self.Item.add_module(
            "collaborative_embeddings",
            nn.Embedding.from_pretrained(
                F.normalize(
                    freerec.utils.import_pickle(
                        cfg.collab_feat_file,
                    ),
                    dim=-1,
                ),
                freeze=True,
            ),
        )

        ACT = nn.ReLU
        dims = [
            self.Item.semantic_embeddings.weight.size(1),
            *cfg.hidden_dims,
            cfg.codebook_dim,
        ]

        self.encoder = nn.Sequential()
        for l, (input_dim, output_dim) in enumerate(zip(dims[:-1], dims[1:]), start=1):
            self.encoder.append(nn.Dropout(cfg.dropout_rate))
            self.encoder.append(nn.Linear(input_dim, output_dim, bias=False))
            if l < len(dims) - 1:
                self.encoder.append(ACT())

        self.quantizer = ResidualQuantizer(
            hidden_size=cfg.codebook_dim,
            num_codebooks=cfg.num_codebooks,
            num_codewords=cfg.num_codewords,
            commit_weight=cfg.commit_weight,
            diversity_weight=cfg.diversity_weight,
            apply_kmeans_init=cfg.apply_kmeans_init,
            sk_iters=cfg.sk_iters,
            sk_epsilons=cfg.sk_epsilons,
        )

        self.decoder, dims = nn.Sequential(), dims[::-1]
        for l, (input_dim, output_dim) in enumerate(zip(dims[:-1], dims[1:]), start=1):
            self.decoder.append(nn.Dropout(cfg.dropout_rate))
            self.decoder.append(nn.Linear(input_dim, output_dim, bias=False))
            if l < len(dims) - 1:
                self.decoder.append(ACT())

        self.criterion = nn.MSELoss(reduction="mean")

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def sure_trainpipe(self, batch_size: int = 512):
        return (
            freerec.data.postprocessing.source.RandomShuffledSource(
                dataset=self.dataset.train(),
                source=self.dataset.to_rows({self.Item: list(range(self.Item.count))}),
            )
            .batch_(batch_size)
            .tensor_()
        )

    def sure_validpipe(self, batch_size: int = 512):
        return (
            freerec.data.postprocessing.source.OrderedSource(
                dataset=self.dataset.valid(),
                source=self.dataset.to_rows({self.Item: list(range(self.Item.count))}),
            )
            .batch_(batch_size)
            .tensor_()
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)  # (B, D)

    def decode(self, q: torch.Tensor) -> torch.Tensor:
        x_hat = self.decoder(q)  # (B, D)
        return F.normalize(x_hat, dim=-1)

    def fit(self, data: Dict[freerec.data.fields.Field, torch.Tensor]) -> Dict[str, torch.Tensor]:
        items = data[self.Item].flatten()
        x = self.Item.semantic_embeddings(items)

        z = self.encode(x)
        q, auxiliary_loss, _ = self.quantizer(z)
        x_hat = self.decode(q)

        logits = torch.einsum(
            "M D, N D -> M N",
            x_hat,
            self.Item.collaborative_embeddings(items),
        )
        labels = torch.arange(0, x_hat.size(0), device=self.device, dtype=torch.long)
        cf_loss = freerec.criterions.cross_entropy_with_logits(
            logits,
            labels,
        )

        return {
            "recon_loss": self.criterion(x_hat, x),
            "auxiliary_loss": auxiliary_loss,
            "cf_loss": cf_loss,
        }

    @torch.no_grad()
    def generate_sem_ids(self) -> torch.Tensor:
        is_training = self.training
        self.eval()
        try:
            sem_ids = []
            items = torch.arange(0, self.Item.count, device=self.device)
            for items in items.split(cfg.batch_size):
                x = self.Item.semantic_embeddings(items)
                z = self.encode(x)
                _, _, ids = self.quantizer(z)
                sem_ids.append(ids.detach().cpu())

            return torch.cat(sem_ids, dim=0)  # (N, #levels)
        finally:
            self.train(is_training)


class CoachForRQVAE(freerec.launcher.Coach):
    @freerec.ddp.main_process_only
    def save_sid_vocab(self) -> None:
        sem_ids = self.get_res_sys_arch().generate_sem_ids()
        sid_vocab = {}
        for item_id, sids in enumerate(sem_ids.tolist()):
            sids = [
                SemIDConverter.SID_FORMAT.format(level=level, id=sid)
                for level, sid in enumerate(sids)
            ]
            sid_vocab[SemIDConverter.format(item_id)] = tuple(sids)

        vocab_file = os.path.join(self.cfg.LOG_PATH, "sid_vocab.json")
        with open(vocab_file, "w", encoding="utf-8") as file:
            json.dump(sid_vocab, file)

    def set_other(self) -> None:
        self.register_metric("RECON_LOSS", lambda x: x, best_caster=min)
        self.register_metric("COMMIT_LOSS", lambda x: x, best_caster=min)
        self.register_metric("CF_LOSS", lambda x: x, best_caster=min)
        self.register_metric("PPL", lambda x: x, best_caster=max)
        self.register_metric("COLLISION_RATE", lambda x: x, best_caster=min)
        for i in range(self.cfg.num_codebooks):
            self.register_metric(f"PPL#{i}", lambda x: x, best_caster=max)

    def train_per_epoch(self, epoch: int) -> None:
        self.get_res_sys_arch().quantizer.reset_diversity_clusters()

        for data in self.dataloader:
            data = self.dict_to_device(data)
            losses = self.model(data)
            loss = (
                losses["recon_loss"] + losses["auxiliary_loss"] + cfg.cf_weight * losses["cf_loss"]
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.monitor(
                loss.item(),
                n=1,
                reduction="mean",
                mode="train",
                pool=["LOSS"],
            )
            self.monitor(
                losses["recon_loss"].item(),
                n=1,
                reduction="mean",
                mode="train",
                pool=["RECON_LOSS"],
            )
            self.monitor(
                losses["auxiliary_loss"].item(),
                n=1,
                reduction="mean",
                mode="train",
                pool=["COMMIT_LOSS"],
            )
            self.monitor(
                losses["cf_loss"].item(),
                n=1,
                reduction="mean",
                mode="train",
                pool=["CF_LOSS"],
            )

        if epoch % self.cfg.eval_freq == 0:
            self.save_sid_vocab()

    def evaluate(self, epoch: int, step: int = -1, mode: str = "valid") -> None:
        sem_ids = self.get_res_sys_arch().generate_sem_ids().cpu()
        counts = torch.zeros((cfg.num_codewords, cfg.num_codebooks))
        counts.scatter_add_(0, sem_ids, torch.ones_like(sem_ids, dtype=torch.float))
        uniques = set([tuple(id_) for id_ in sem_ids.tolist()])

        freqs = counts.div(counts.sum(dim=0, keepdim=True))
        perplexity = ((freqs + 1.0e-8).log() * freqs).sum(dim=0).neg().exp().tolist()

        ppls = []
        for i, ppl in enumerate(perplexity):
            ppls.append(ppl)
            self.monitor(ppl, n=1, mode="valid", pool=[f"PPL#{i}"])

        self.monitor(
            sum(ppls),
            n=len(ppls),
            mode=mode,
            reduction="sum",
            pool=["PPL"],
        )
        self.monitor(
            (self.Item.count - len(uniques)) / self.Item.count,
            n=1,
            mode=mode,
            pool=["COLLISION_RATE"],
        )


def main() -> None:
    dataset: freerec.data.datasets.RecDataSet
    try:
        dataset = getattr(freerec.data.datasets, cfg.dataset)(root=cfg.root)
    except AttributeError:
        dataset = freerec.data.datasets.RecDataSet(
            cfg.root,
            cfg.dataset,
            tasktag=cfg.tasktag,
        )

    model = RQVAE(dataset)

    trainpipe = model.sure_trainpipe(cfg.batch_size)
    validpipe = model.sure_validpipe(cfg.batch_size)

    coach = CoachForRQVAE(
        dataset=dataset,
        trainpipe=trainpipe,
        validpipe=validpipe,
        testpipe=None,
        model=model,
        cfg=cfg,
    )
    coach.fit()


if __name__ == "__main__":
    main()
