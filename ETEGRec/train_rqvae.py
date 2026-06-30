import os
from typing import Dict

import freerec
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import ETEGRecTokenizer
from transformers import get_constant_schedule_with_warmup, get_linear_schedule_with_warmup

freerec.declare(version="1.0.1")

cfg = freerec.parser.Parser()
cfg.add_argument("--sem-feat-file", type=str, default=None)

cfg.add_argument("--num-codebooks", type=int, default=3)
cfg.add_argument("--num-codewords", type=int, default=256)
cfg.add_argument("--codebook-dim", type=int, default=128)
cfg.add_argument("--hidden-dims", type=str, default="512,256")
cfg.add_argument("--quant-loss-weight", type=float, default=1.0)
cfg.add_argument("--commit-weight", type=float, default=0.25)
cfg.add_argument("--kmeans-init", type=eval, default=True)
cfg.add_argument("--kmeans-iters", type=int, default=100)
cfg.add_argument("--tokenizer-dropout-rate", type=float, default=0.0)
cfg.add_argument("--lr-scheduler-type", type=str, default="linear", choices=("linear", "constant"))
cfg.add_argument("--warmup-epochs", type=int, default=50)

cfg.set_defaults(
    description="ETEGRec-RQVAE",
    root="../../data",
    dataset="Amazon2014Beauty_550_LOU",
    epochs=10000,
    batch_size=1024,
    optimizer="AdamW",
    lr=1e-3,
    weight_decay=1.0e-4,
    eval_freq=50,
    CHECKPOINT_FREQ=50,
    which4best="COLLISION_RATE",
    seed=1,
)
cfg.compile()

cfg.hidden_dims = list(map(int, cfg.hidden_dims.split(","))) if cfg.hidden_dims else []


class RQVAE(freerec.models.RecSysArch):
    r"""Pretrain the ETEGRec item tokenizer from semantic item features."""

    def __init__(self, dataset: freerec.data.datasets.RecDataSet) -> None:
        super().__init__(dataset)

        semantic_features = torch.as_tensor(
            freerec.utils.import_pickle(
                os.path.join(dataset.path, cfg.sem_feat_file)
            ),
            dtype=torch.float,
        )
        tokenizer_config = {
            "num_codebooks": cfg.num_codebooks,
            "num_codewords": cfg.num_codewords,
            "hidden_dims": cfg.hidden_dims,
            "codebook_dim": cfg.codebook_dim,
            "commit_weight": cfg.commit_weight,
            "kmeans_init": cfg.kmeans_init,
            "kmeans_iters": cfg.kmeans_iters,
            "tokenizer_dropout_rate": cfg.tokenizer_dropout_rate,
        }
        self.Item.add_module(
            "semantic_embeddings",
            nn.Embedding.from_pretrained(
                semantic_features,
                freeze=True,
            ),
        )
        self.tokenizer = ETEGRecTokenizer(
            config=tokenizer_config,
            in_dim=semantic_features.size(1),
        )

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

    def fit(self, data: Dict[freerec.data.fields.Field, torch.Tensor]) -> Dict[str, torch.Tensor]:
        items = data[self.Item].flatten()
        semantic_embs = self.Item.semantic_embeddings(items)
        tokenizer_output = self.tokenizer(semantic_embs)
        recon_embs = tokenizer_output.reconstructions
        recon_loss = F.mse_loss(recon_embs, semantic_embs, reduction="mean")

        return {
            "recon_loss": recon_loss,
            "vq_loss": tokenizer_output.loss,
        }

    @torch.no_grad()
    def generate_sem_ids(self) -> torch.Tensor:
        is_training = self.training
        self.eval()
        try:
            sem_ids = []
            items = torch.arange(0, self.Item.count, device=self.device)
            for item_ids in items.split(cfg.batch_size):
                semantic_embs = self.Item.semantic_embeddings(item_ids)
                sem_ids.append(self.tokenizer.get_indices(semantic_embs).detach().cpu())
            return torch.cat(sem_ids, dim=0)
        finally:
            self.train(is_training)


class CoachForRQVAE(freerec.launcher.Coach):
    def set_lr_scheduler(self) -> None:
        steps_per_epoch = max(len(self.trainloader), 1)
        warmup_steps = cfg.warmup_epochs * steps_per_epoch
        max_steps = max(cfg.epochs * steps_per_epoch, 1)
        if cfg.lr_scheduler_type == "linear":
            self.lr_scheduler = get_linear_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=max_steps,
            )
        elif cfg.lr_scheduler_type == "constant":
            self.lr_scheduler = get_constant_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=warmup_steps,
            )
        else:
            raise ValueError(f"unsupported lr_scheduler_type: {cfg.lr_scheduler_type}")

    def set_other(self) -> None:
        self.register_metric("RECON_LOSS", lambda x: x, best_caster=min)
        self.register_metric("VQ_LOSS", lambda x: x, best_caster=min)
        self.register_metric("PPL", lambda x: x, best_caster=max)
        self.register_metric("COLLISION_RATE", lambda x: x, best_caster=min)
        for level in range(cfg.num_codebooks):
            self.register_metric(f"PPL#{level}", lambda x: x, best_caster=max)

    def train_per_epoch(self, epoch: int) -> None:
        for data in self.dataloader:
            data = self.dict_to_device(data)
            losses = self.model(data)
            loss = losses["recon_loss"] + cfg.quant_loss_weight * losses["vq_loss"]

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.lr_scheduler.step()

            self.monitor(loss.item(), n=1, reduction="mean", mode="train", pool=["LOSS"])
            self.monitor(
                losses["recon_loss"].item(),
                n=1,
                reduction="mean",
                mode="train",
                pool=["RECON_LOSS"],
            )
            self.monitor(
                losses["vq_loss"].item(),
                n=1,
                reduction="mean",
                mode="train",
                pool=["VQ_LOSS"],
            )

    def evaluate(self, epoch: int, step: int = -1, mode: str = "valid") -> None:
        sem_ids = self.get_res_sys_arch().generate_sem_ids()
        uniques = {tuple(code) for code in sem_ids.tolist()}

        ppls = []
        for level in range(cfg.num_codebooks):
            counts = torch.bincount(sem_ids[:, level], minlength=cfg.num_codewords).float()
            freqs = counts.div(counts.sum())
            ppl = ((freqs + 1.0e-8).log() * freqs).sum().neg().exp().item()
            ppls.append(ppl)
            self.monitor(ppl, n=1, mode=mode, pool=[f"PPL#{level}"])

        self.monitor(sum(ppls), n=len(ppls), mode=mode, reduction="sum", pool=["PPL"])
        self.monitor(
            (self.Item.count - len(uniques)) / self.Item.count,
            n=1,
            mode=mode,
            pool=["COLLISION_RATE"],
        )

    def save_last(self):
        if freerec.ddp.is_main_process():
            self.save(self.get_res_sys_arch().tokenizer.state_dict(), "tokenizer.pt")
        return super().save_last()


def main():
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
        testpipe=validpipe,
        model=model,
        cfg=cfg,
    )
    coach.fit()


if __name__ == "__main__":
    main()
