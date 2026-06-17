import json
import os
from typing import Dict

import freerec
import torch
import torch.nn as nn
import torch.nn.functional as F
from converter import SemIDConverter
from quantizer import RatingResidualQuantizer

freerec.declare(version="1.0.1")

cfg = freerec.parser.Parser()
cfg.add_argument("--num-codebooks", type=int, default=3, help="number of codebooks")
cfg.add_argument("--num-codewords", type=int, default=256, help="number of codewords per codebook")
cfg.add_argument("--sem-feat-file", type=str, default=None, help="file of semantic features")

cfg.add_argument("--rec-loss-weight", type=float, default=1.0, help="weight for reconstruction loss")
cfg.add_argument(
    "--sc-loss-weight", type=float, default=0.1, help="weight for semantic cohesion loss"
)
cfg.add_argument(
    "--pd-loss-weight", type=float, default=0.05, help="weight for preference discrimination loss"
)
cfg.add_argument(
    "--pd-temperature",
    type=float,
    default=2.0,
    help="temperature for preference discrimination loss",
)

cfg.set_defaults(
    description="R3-VAE",
    root="../../data",
    dataset="Amazon2014Beauty_550_LOU",
    epochs=10000,
    batch_size=1024,
    optimizer="AdamW",
    lr=1e-3,
    weight_decay=0.0,
    seed=1,
)
cfg.compile()


class R3VAE(freerec.models.RecSysArch):
    def __init__(self, dataset: freerec.data.datasets.RecDataSet) -> None:
        super().__init__(dataset)

        self.Item.add_module(
            "embeddings",
            nn.Embedding.from_pretrained(
                F.normalize(
                    freerec.utils.import_pickle(os.path.join(self.dataset.path, cfg.sem_feat_file)),
                    dim=-1,
                ),
                freeze=True,
            ),
        )

        input_dim = self.Item.embeddings.weight.size(1)
        self.codebook_dim = input_dim
        self.encoder = nn.Identity()

        self.quantizer = RatingResidualQuantizer(
            self.dataset,
            self.codebook_dim,
            num_codebooks=cfg.num_codebooks,
            num_codewords=cfg.num_codewords,
            pd_temperature=cfg.pd_temperature,
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.codebook_dim, input_dim * 4),
            nn.ReLU(),
            nn.Linear(input_dim * 4, input_dim),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # codebook initialization
        with torch.no_grad():
            for codebook in self.quantizer.codebooks:
                codebook.requires_kmeans_init_ = True

            self.encoder.to(cfg.device)
            self.quantizer.to(cfg.device)

            x = self.Item.embeddings.weight[: cfg.num_codewords * 5].to(cfg.device)
            z = self.encode(x)
            self.quantizer(z)

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
        return self.encoder(x)

    def decode(self, q: torch.Tensor) -> torch.Tensor:
        return self.decoder(q)

    def fit(self, data: Dict[freerec.data.fields.Field, torch.Tensor]) -> Dict[str, torch.Tensor]:
        items = data[self.Item].flatten()
        x = self.Item.embeddings(items)
        z = self.encode(x)
        q, sc_loss, pd_loss, _ = self.quantizer(z)
        x_hat = self.decode(q)
        rec_loss = 1.0 - F.cosine_similarity(x_hat, x, dim=-1).mean()
        loss = (
            rec_loss * cfg.rec_loss_weight
            + sc_loss * cfg.sc_loss_weight
            + pd_loss * cfg.pd_loss_weight
        )

        return {
            "loss": loss,
            "rec_loss": rec_loss,
            "sc_loss": sc_loss,
            "pd_loss": pd_loss,
        }

    @torch.no_grad()
    def generate_sem_ids(self) -> torch.Tensor:
        is_training = self.training
        self.eval()
        try:
            sem_ids = []
            items = torch.arange(0, self.Item.count, device=self.device)
            for items in items.split(cfg.batch_size):
                x = self.Item.embeddings(items)
                z = self.encode(x)
                _, _, _, ids = self.quantizer(z)
                sem_ids.append(ids.detach().cpu())

            return torch.cat(sem_ids, dim=0)
        finally:
            self.train(is_training)


class CoachForR3VAE(freerec.launcher.Coach):
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
        self.register_metric("LOSS", lambda x: x, best_caster=min)
        self.register_metric("REC_LOSS", lambda x: x, best_caster=min)
        self.register_metric("SC_LOSS", lambda x: x, best_caster=min)
        self.register_metric("PD_LOSS", lambda x: x, best_caster=min)
        self.register_metric("PPL", lambda x: x, best_caster=max)
        self.register_metric("COLLISION_RATE", lambda x: x, best_caster=min)
        for level in range(self.cfg.num_codebooks):
            self.register_metric(f"PPL#{level}", lambda x: x, best_caster=max)

    def train_per_epoch(self, epoch: int) -> None:
        for data in self.dataloader:
            data = self.dict_to_device(data)
            losses = self.model(data)

            self.optimizer.zero_grad()
            losses["loss"].backward()
            self.optimizer.step()

            for key, val in losses.items():
                self.monitor(
                    val.item(),
                    n=1,
                    reduction="mean",
                    mode="train",
                    pool=[key.upper()],
                )

        if epoch % self.cfg.eval_freq == 0:
            self.save_sid_vocab()

    def evaluate(self, epoch: int, step: int = -1, mode: str = "valid") -> None:
        sem_ids = self.get_res_sys_arch().generate_sem_ids().cpu()
        ppls = []
        for level in range(self.cfg.num_codebooks):
            counts = torch.zeros(self.cfg.num_codewords)
            ids = sem_ids[:, level]
            counts.scatter_add_(0, ids, torch.ones_like(ids, dtype=torch.float))
            freqs = counts.div(counts.sum())
            ppl = ((freqs + 1.0e-8).log() * freqs).sum().neg().exp().item()
            ppls.append(ppl)
            self.monitor(ppl, n=1, mode=mode, pool=[f"PPL#{level}"])

        uniques = set(tuple(ids) for ids in sem_ids.tolist())
        self.monitor(sum(ppls), n=len(ppls), mode=mode, reduction="sum", pool=["PPL"])
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

    model = R3VAE(dataset)
    trainpipe = model.sure_trainpipe(cfg.batch_size)
    validpipe = model.sure_validpipe(cfg.batch_size)

    coach = CoachForR3VAE(
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
