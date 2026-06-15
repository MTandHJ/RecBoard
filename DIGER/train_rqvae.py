import os
from typing import Dict

import freerec
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import DIGERIDDecoder, DIGERIDEncoder
from quantizer import ResidualQuantizer

freerec.declare(version="1.0.1")

cfg = freerec.parser.Parser()
cfg.add_argument("--item-feat-file", "--sem-feat-file", type=str, default=None)
cfg.add_argument("--num-codebooks", type=int, default=3)
cfg.add_argument("--num-codewords", type=int, default=256)
cfg.add_argument("--codebook-dim", "--e-dim", type=int, default=256)
cfg.add_argument("--hidden-dims", "--layers", type=str, default="2048,1024,512")
cfg.add_argument("--commit-weight", "--beta", type=float, default=0.25)
cfg.add_argument("--dist", type=str, default="l2")
cfg.add_argument("--tokenizer-dropout-rate", "--dropout-prob", type=float, default=0.0)
cfg.add_argument("--bn", type=eval, default=False)
cfg.add_argument("--sk-epsilons", type=str, default="0.003,0.003,0.003")
cfg.add_argument("--sk-iters", type=int, default=50)
cfg.add_argument("--vq-loss-weight", "--quant-loss-weight", type=float, default=1.0)
cfg.add_argument("--kmeans-init", type=eval, default=True)
cfg.add_argument("--kmeans-iters", type=int, default=100)
cfg.add_argument("--checkpoint-file", type=str, default="rqvae.pt")

cfg.add_argument("--gumbel-hard-switch-epoch", type=int, default=50)
cfg.add_argument("--use-adaptive-selection", type=eval, default=False)
cfg.add_argument("--hot-threshold-ratio", type=float, default=1.5)
cfg.add_argument("--usage-momentum", type=float, default=0.99)
cfg.add_argument("--use-learnable-sigma-gumbel", type=eval, default=False)
cfg.add_argument("--use-simple-uncertainty-loss", type=eval, default=False)
cfg.add_argument("--initial-std", type=float, default=None)
cfg.add_argument("--initial-sigma", type=float, default=1.0)
cfg.add_argument("--noise-type", type=str, default="gumbel")

cfg.set_defaults(
    description="DIGER-RQVAE",
    root="../../RecData",
    dataset="Amazon2014Beauty_550_LOU",
    tasktag="NEXTITEM",
    epochs=1000,
    batch_size=1024,
    optimizer="AdamW",
    lr=1.0e-3,
    weight_decay=0.0,
    seed=2020,
    which4best="COLLISION_RATE",
)
cfg.compile()

cfg.hidden_dims = list(map(int, cfg.hidden_dims.split(",")))
cfg.sk_epsilons = list(map(float, cfg.sk_epsilons.split(",")))


class DIGERIDTokenizer(freerec.models.RecSysArch):
    r"""Pretrain DIGER's item ID tokenizer on item semantic features.

    Workflow
    --------
    1. encode item semantic features into latent vectors.
    2. quantize latents with the same residual quantizer used by DIGER.
    3. reconstruct semantic features through a decoder.
    4. save encoder and quantizer state dicts for `main.py`.
    """

    def __init__(self, dataset: freerec.data.datasets.RecDataSet) -> None:
        super().__init__(dataset)

        item_feat_file = cfg.item_feat_file
        if not os.path.isabs(item_feat_file):
            item_feat_file = os.path.join(self.dataset.path, item_feat_file)
        item_features = freerec.utils.import_pickle(item_feat_file)
        if item_features.size(0) != self.Item.count:
            raise ValueError(
                f"item feature rows should equal Item.count={self.Item.count}, "
                f"but got {item_features.size(0)}"
            )

        self.Item.add_module(
            "embeddings",
            nn.Embedding.from_pretrained(item_features, freeze=True),
        )
        self.id_encoder = DIGERIDEncoder(
            in_dim=item_features.size(1),
            hidden_dims=cfg.hidden_dims,
            codebook_dim=cfg.codebook_dim,
            dropout_rate=cfg.tokenizer_dropout_rate,
            bn=bool(cfg.bn),
        )
        self.id_quantizer = ResidualQuantizer(cfg)
        self.id_decoder = DIGERIDDecoder(
            out_dim=item_features.size(1),
            hidden_dims=cfg.hidden_dims,
            codebook_dim=cfg.codebook_dim,
            dropout_rate=cfg.tokenizer_dropout_rate,
            bn=bool(cfg.bn),
        )
        self.initialize_codebooks()

    def sure_trainpipe(self, batch_size: int):
        return (
            freerec.data.postprocessing.source.RandomShuffledSource(
                dataset=self.dataset.train(),
                source=self.dataset.to_rows({self.Item: list(range(self.Item.count))}),
            )
            .batch_(batch_size)
            .tensor_()
        )

    def sure_validpipe(self, batch_size: int):
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
        x = self.Item.embeddings(items)
        z = self.id_encoder(x)
        quantizer_outputs = self.id_quantizer(z, use_gumbel=False)
        x_hat = self.id_decoder(quantizer_outputs.quantized)
        recon_loss = F.mse_loss(x_hat, x)
        return {
            "loss": recon_loss + cfg.vq_loss_weight * quantizer_outputs.loss,
            "recon_loss": recon_loss,
            "vq_loss": quantizer_outputs.loss,
        }

    @torch.no_grad()
    def initialize_codebooks(self) -> None:
        r"""Initialize RQ codebooks with TIGER-style warm-start samples."""
        if not cfg.kmeans_init:
            return

        sample_size = min(self.Item.count, cfg.num_codewords * 5)
        x = self.Item.embeddings.weight[:sample_size]
        z = self.id_encoder(x)
        residual = z.detach()
        for quantizer in self.id_quantizer.quantizers:
            centers = self.kmeans(residual.view(-1, cfg.codebook_dim), quantizer.num_codewords)
            quantizer.codebook.weight.data.copy_(centers)
            level_q, _, _, _, _, _ = quantizer(residual)
            residual = residual - level_q

    @staticmethod
    def kmeans(samples: torch.Tensor, num_clusters: int) -> torch.Tensor:
        from k_means_constrained import KMeansConstrained

        if samples.size(0) < num_clusters:
            raise ValueError(
                f"k-means requires at least {num_clusters} samples, but got {samples.size(0)}"
            )

        samples_np = samples.detach().cpu().numpy()
        size_min = max(1, min(len(samples_np) // (num_clusters * 2), 50))
        cluster = KMeansConstrained(
            n_clusters=num_clusters,
            size_min=size_min,
            max_iter=cfg.kmeans_iters,
            n_init=10,
            n_jobs=10,
            verbose=False,
        ).fit(samples_np)
        return torch.as_tensor(
            cluster.cluster_centers_,
            dtype=samples.dtype,
            device=samples.device,
        )

    @torch.no_grad()
    def generate_sem_ids(self) -> torch.Tensor:
        was_training = self.training
        self.eval()

        sem_ids = []
        items = torch.arange(self.Item.count, device=self.device)
        for chunk in items.split(cfg.batch_size):
            x = self.Item.embeddings(chunk)
            z = self.id_encoder(x)
            sem_ids.append(self.id_quantizer.get_indices(z).detach().cpu())

        self.train(was_training)
        return torch.cat(sem_ids, dim=0)

    def tokenizer_state(self) -> dict:
        return {
            "encoder": self.id_encoder.state_dict(),
            "quantizer": self.id_quantizer.state_dict(),
            "decoder": self.id_decoder.state_dict(),
            "meta": {
                "dataset": cfg.dataset,
                "item_feat_file": cfg.item_feat_file,
                "hidden_dims": cfg.hidden_dims,
                "codebook_dim": cfg.codebook_dim,
                "num_codebooks": cfg.num_codebooks,
                "num_codewords": cfg.num_codewords,
                "dist": cfg.dist,
                "kmeans_init": cfg.kmeans_init,
                "kmeans_iters": cfg.kmeans_iters,
            },
        }


class CoachForDIGERIDTokenizer(freerec.launcher.Coach):
    r"""Coach for DIGER RQ-VAE pretraining."""

    def set_other(self) -> None:
        self.register_metric("RECON_LOSS", lambda x: x, best_caster=min)
        self.register_metric("VQ_LOSS", lambda x: x, best_caster=min)
        self.register_metric("COLLISION_RATE", lambda x: x, best_caster=min)

    def train_per_epoch(self, epoch: int) -> None:
        for data in self.dataloader:
            data = self.dict_to_device(data)
            losses = self.model(data)

            self.optimizer.zero_grad()
            losses["loss"].backward()
            self.optimizer.step()

            self.monitor(
                losses["recon_loss"].detach().item(),
                n=1,
                reduction="mean",
                mode="train",
                pool=["RECON_LOSS"],
            )
            self.monitor(
                losses["vq_loss"].detach().item(),
                n=1,
                reduction="mean",
                mode="train",
                pool=["VQ_LOSS"],
            )

    def evaluate(self, epoch: int, step: int = -1, mode: str = "valid") -> None:
        sem_ids = self.get_res_sys_arch().generate_sem_ids()
        unique_codes = len({tuple(code) for code in sem_ids.tolist()})
        collision_rate = (self.Item.count - unique_codes) / self.Item.count
        self.monitor(
            collision_rate,
            n=1,
            reduction="mean",
            mode=mode,
            pool=["COLLISION_RATE"],
        )

    def save_tokenizer(self) -> None:
        if freerec.ddp.is_main_process():
            model = self.get_res_sys_arch()
            torch.save(
                model.tokenizer_state(),
                os.path.join(self.cfg.LOG_PATH, self.cfg.checkpoint_file),
            )
        freerec.ddp.synchronize()

    def save_best(self) -> None:
        super().save_best()
        self.save_tokenizer()


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

    model = DIGERIDTokenizer(dataset)
    trainpipe = model.sure_trainpipe(cfg.batch_size)
    validpipe = model.sure_validpipe(cfg.batch_size)

    coach = CoachForDIGERIDTokenizer(
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
