

from typing import Dict, Tuple, Union, Literal

import torch, os
import torch.nn as nn
import torch.nn.functional as F
import freerec
from freerec.data.tags import TIMESTAMP, SEQUENCE

from utils import straight_through_estimator, gumbel_softmax_estimator, rotation_trick_estimator

freerec.declare(version='1.0.1')

cfg = freerec.parser.Parser()
cfg.add_argument("--num-codes", type=int, default=32)
cfg.add_argument("--codebook-dim", type=int, default=16)
cfg.add_argument("--hidden-dims", type=str, default="256,64")
cfg.add_argument("--num-buckets", type=int, default=100)
cfg.add_argument("--tau4estimator", type=float, default=0.05)
cfg.add_argument("--apply-kmeans-init", type=eval, default=True)
cfg.add_argument("--apply-sim-vq", type=eval, default=False)
cfg.add_argument("--commit-weight", type=float, default=0.25)
cfg.add_argument("--dropout-rate", type=float, default=0.1)
cfg.add_argument("--ffile", type=str, default="all-minilm-l12-v2_title.pkl")

cfg.set_defaults(
    description="RQVAE",
    root="../../data",
    dataset='Amazon2014Beauty_550_LOU',
    epochs=200,
    batch_size=256,
    optimizer='AdamW',
    lr=1e-3,
    weight_decay=0.,
    seed=1,
)
cfg.compile()

cfg.hidden_dims = list(map(int, cfg.hidden_dims.split(',')))

class Quantizer(nn.Module):

    def __init__(
        self,
        num_codes: int,
        codebook_dim: int,
        apply_sim_vq: bool = False,
        apply_kmeans_init: bool = True,
        gradient_estimator: Literal['ste'] = 'ste'
    ):
        super().__init__()

        self.gradient_estimator = gradient_estimator.lower()

        self.num_codes = num_codes
        self.codebook = nn.Parameter(
            torch.randn(num_codes, codebook_dim) * 0.02, 
            requires_grad=True
        )

        if apply_sim_vq:
            self.projector = nn.Linear(codebook_dim, codebook_dim, bias=False)
        else:
            self.projector = nn.Identity()

        self.apply_kmeans_init = apply_kmeans_init 

    @torch.no_grad()
    def set_kmeans_codebook(self, z: torch.Tensor):
        if self.apply_kmeans_init:
            from scipy.cluster.vq import kmeans
            z = z.detach().cpu().numpy()
            codebook = kmeans(z, k_or_guess=self.num_codes)
            self.codebook.data.copy_(
                torch.from_numpy(codebook).to(device=self.device, dtype=self.codebook.dtype)
            )
            self.apply_kmeans_init = False

    def forward(self, z: torch.Tensor):
        self.set_kmeans_codebook(z)

        codebook = self.projector(self.codebook)
        dist = torch.cdist(z, codebook, p=2) # (B, K)
        ids = torch.argmin(dist, dim=-1) # (B,)
        q = codebook[ids] # (B, D)
        loss = F.mse_loss(z.detach(), q) + cfg.commit_weight * F.mse_loss(q.detach(), z)

        if self.training:
            if self.gradient_estimator == 'ste':
                q = straight_through_estimator(z, q)
            elif self.gradient_estimator == 'gumbel-softmax':
                q = gumbel_softmax_estimator(dist, codebook, temperature=cfg.tau4estimator)
            elif self.gradient_estimator == 'rotation':
                q = rotation_trick_estimator(z, q)
            else:
                raise NotImplementedError

        return q, loss, ids


class RQVAE(freerec.models.RecSysArch):

    def __init__(
        self, dataset: freerec.data.datasets.RecDataSet
    ) -> None:
        super().__init__(dataset)

        self.Item.add_module(
            "embeddings",
            nn.Embedding.from_pretrained(
                freerec.utils.import_pickle(cfg.ffile),
                freeze=True
            )
        )

        dims = [self.Item.embeddings.weight.size(0)] + cfg.hidden_dims + [cfg.codebook_dim]

        self.encoder = nn.Sequential()
        for l, (input_dim, output_dim) in enumerate(zip(dims[:-1], dims[1:]), start=1):
            self.encoder.append(
                nn.Linear(input_dim, output_dim, bias=False)
            )
            if l < len(dims) - 1:
                self.encoder.append(
                    nn.SiLU()
                )
                self.encoder.append(
                    nn.Dropout(cfg.dropout_rate)
                )

        self.quantizers = nn.ModuleList([
            Quantizer(
                num_codes=cfg.num_codes,
                codebook_dim=cfg.codebook_dim, 
                apply_sim_vq=cfg.apply_sim_vq
            )
            for _ in range(cfg.num_buckets)
        ])

        self.decoder, dims = nn.Sequential(), dims[::-1]
        for l, (input_dim, output_dim) in enumerate(zip(dims[:-1], dims[1:]), start=1):
            self.decoder.append(
                nn.Linear(input_dim, output_dim, bias=False)
            )
            if l < len(dims) - 1:
                self.decoder.append(
                    nn.SiLU()
                )
                self.decoder.append(
                    nn.Dropout(cfg.dropout_rate)
                )

        self.criterion = freerec.criterions.CrossEntropy4Logits(reduction='mean')

        self.reset_parameters()

    def sure_trainpipe(self, batch_size: int = 512):
        return freerec.data.postprocessing.source.RandomShuffledSource(
            dataset=self.dataset.train(),
            source=self.dataset.to_rows(
                {self.Item: list(range(self.Item.count))}
            )
        ).batch_(batch_size).tensor_()

    def sure_validpipe(self, batch_size: int = 512):
        return freerec.data.postprocessing.source.OrderedSource(
            dataset=self.dataset.valid(),
            source=self.dataset.to_rows(
                {self.Item: list(range(self.Item.count))}
            )
        ).batch_(batch_size).tensor_()

    def encode(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x) # (B, D)

        loss = 0
        qs, residuals, ids = [], [], []

        q_ = z
        for quantizer in self.quantizers:
            q_, loss_, id_ = quantizer(z)
            z = z - q_

            qs.append(q_)
            ids.append(id_)
            loss += loss_

        q = torch.stack(qs, dim=-1).sum(dim=-1)
        return q, loss, torch.stack(ids, dim=-1)

    def decode(
        self,
        q: torch.Tensor
    ):
        x_hat = self.decoder(q) # (B, D)
        return x_hat

    def fit(
        self, data: Dict[freerec.data.fields.Field, torch.Tensor]
    ) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        items = data[self.Item]
        x = self.Item.embeddings[items]

        q, auxiliary_loss, _ = self.encode(x)
        x_hat = self.decode(q)

        loss = F.mse_loss(x_hat, x) + auxiliary_loss

        return loss

    @torch.no_grad()
    def generate_sem_ids(self):
        sem_ids = []
        items = torch.arange(0, self.Item.embeddings.weight.size(0), device=self.device)
        for items in items.split(cfg.batch_size):
            x = self.Item.embeddings(items)
            _, _, ids = self.encode(x)
            sem_ids.append(ids.detach().cpu())
        
        sem_ids = torch.cat(sem_ids, dim=0) # (N, #Buckets)
        return sem_ids

class CoachForRQVAE(freerec.launcher.Coach):

    def save_checkpoint(self, epoch):
        super().save_checkpoint(epoch)
        if freerec.ddp.is_main_process():
            sem_ids = self.get_res_sys_arch().generate_sem_ids()
            freerec.utils.export_pickle(
                sem_ids,
                os.path.join(
                    self.cfg.LOG_PATH,
                    f"sem_id_{epoch}"
                )
            )

    def set_other(self):
        for i in range(self.num_buckets):
            self.register_metric(
                f"ENT{i}", lambda x: x, best_caster=max
            )
            self.register_metric(
                f"DIST{i}", lambda x: x, best_caster=max
            )

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

    def evaluate(self, epoch, step = -1, mode = 'valid'):

        counts = torch.zeros((cfg.num_codes, cfg.num_buckets))

        for data in self.dataloader:
            data = self.dict_to_device(data)
            items = data[self.Item]
            x = self.model.Item.embeddings[items]
            _, _, ids = self.model.encode(x)

            ids = ids.cpu()
            counts.scatter_add(
                0, ids, torch.ones_like(ids, dtype=torch.float)
            )

        freqs = counts.div(counts.sum(dim=0, keepdim=True))
        entropy = ((freqs + 1.e-8).log() * freqs).sum(dim=0).neg().tolist()

        for i, ent in enumerate(entropy):
            self.monitor(
                ent, n=1, mode='valid',
                pool=[f"ENT{i}"]
            )

        for i, quantizer in enumerate(self.model.quantizers):
            codebook = quantizer.codebook
            dist = torch.cdist(codebook, codebook).mean().item()
            self.monitor(
                dist, n=1, mode='valid',
                pool=[f"DIST{i}"]
            )


def main():

    dataset: freerec.data.datasets.RecDataSet
    try:
        dataset = getattr(freerec.data.datasets, cfg.dataset)(root=cfg.root)
    except AttributeError:
        dataset = freerec.data.datasets.RecDataSet(cfg.root, cfg.dataset, tasktag=cfg.tasktag)

    model = RQVAE(dataset)

    # datapipe
    trainpipe = model.sure_trainpipe(cfg.batch_size)
    validpipe = model.sure_validpipe(cfg.batch_size)

    coach = CoachForRQVAE(
        dataset=dataset,
        trainpipe=trainpipe,
        validpipe=validpipe,
        testpipe=None,
        model=model,
        cfg=cfg
    )
    coach.fit()


if __name__ == "__main__":
    main()