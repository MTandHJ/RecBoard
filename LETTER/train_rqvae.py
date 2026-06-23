

from typing import Dict, Tuple, Union, Literal

import random
import torch, os
import torch.nn as nn
import torch.nn.functional as F
import freerec
from k_means_constrained import KMeansConstrained 

from utils import sinkhorn_algorithm

freerec.declare(version='1.0.1')

cfg = freerec.parser.Parser()
cfg.add_argument("--num-codewords", type=int, default=256, help="number of codewords per codebook")
cfg.add_argument("--num-codebooks", type=int, default=3, help="number of codebooks")
cfg.add_argument("--codebook-dim", type=int, default=32, help="codebook dimension size")
cfg.add_argument("--hidden-dims", type=str, default="512,256,128")
cfg.add_argument("--apply-kmeans-init", type=eval, default=True)

cfg.add_argument("--sk-epsilons", type=str, default='0.,0.,0.,0.003', help="epsilon for sinkhorn iteration")
cfg.add_argument("--sk-iters", type=float, default=50, help="number of iterations for sinkhorn iteration")

cfg.add_argument("--cf-weight", type=float, default=0.01, help="cf loss weight")
cfg.add_argument("--commit-weight", type=float, default=0.25, help="commitment loss weight")
cfg.add_argument("--diversity-weight", type=float, default=0.0001, help="diversity loss weight")

cfg.add_argument("--dropout-rate", type=float, default=0.1)
cfg.add_argument("--sem-feat-file", type=str, default=None, help="file of semantic features")
cfg.add_argument("--collab-feat-file", type=str, default=None, help="file of collaborative features")
 
cfg.set_defaults(
    description="LETTER",
    root="../../data",
    dataset='Amazon2014Beauty_1000_LOU',
    epochs=20000,
    batch_size=1024,
    optimizer='AdamW',
    lr=1e-3,
    weight_decay=1.e-4,
    seed=1,
)
cfg.compile()

cfg.hidden_dims = list(map(int, cfg.hidden_dims.split(',')))
cfg.sk_epsilons = list(map(float, cfg.sk_epsilons.split(',')))


class Quantizer(nn.Module):

    def __init__(
        self,
        num_codewords: int,
        codebook_dim: int,
        apply_kmeans_init: bool = True,
        sk_epsilon: float = 0., sk_iters: int = 50
    ):
        super().__init__()

        self.num_codewords = num_codewords
        self.codebook = nn.Parameter(
            torch.rand(num_codewords, codebook_dim),
            requires_grad=True
        )
        self.apply_kmeans_init = apply_kmeans_init 
        self.sk_epsilon = sk_epsilon
        self.sk_iters = sk_iters

        self.criterion = nn.MSELoss(reduction="mean")

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(
            self.codebook, 
            -1. / self.num_codewords,
            1. / self.num_codewords
        )

    def reset_training_buffers(self, num_clusters: int = 10):
        r"""
        Determining the clusters and corresponding labels for the diversity loss.
        This method is invoked at the beginning of each epoch.
        """
        z = self.codebook.detach().cpu().numpy()
        size_min = min(len(z) // (self.num_codewords * 2), 10)

        clf = KMeansConstrained(
            n_clusters=num_clusters,
            size_min=size_min, size_max=num_clusters * 6,
            max_iter=10, n_init=10, n_jobs=10, 
            verbose=False
        )
        clf.fit(z)
        
        self.register_buffer(
            "_labels", # (#num_codewords,)
            torch.from_numpy(clf.labels_).to(self.codebook.device)
        )
        self._clusters = [
            set(torch.where(self._labels == l)[0].tolist())
            for l in range(num_clusters)
        ]

    @torch.no_grad()
    def set_kmeans_codebook(self, z: torch.Tensor):
        if self.apply_kmeans_init:
            z = z.detach().cpu().numpy()

            size_min = min(len(z) // (self.num_codewords * 2), 50)

            clf = KMeansConstrained(
                n_clusters=self.num_codewords, 
                size_min=size_min, size_max=size_min * 4, 
                max_iter=10, n_init=10, n_jobs=10, 
                verbose=False
            )
            clf.fit(z)
            codebook = torch.from_numpy(clf.cluster_centers_)

            self.codebook.data.copy_(
                codebook.to(device=self.codebook.device, dtype=self.codebook.dtype)
            )

            self.apply_kmeans_init = False

    def _sample_positive(self, id_: int, label_: int):
        candidates = self._clusters[label_] - {id_} # remove 'self' from the candidates
        return random.choice(list(candidates))

    def calc_diversity_loss(
        self, q: torch.Tensor, ids: torch.Tensor
    ):
        labels = torch.gather(
            self._labels, dim=0, index=ids
        ).tolist()
        positives = [ # sample positives within the same cluster except itself
            self._sample_positive(id_, label_)
            for id_, label_ in zip(ids, labels)
        ]
        positives = torch.tensor(positives, dtype=torch.long, device=q.device)

        # no cosine similarity?
        ids = ids.view(-1, 1)
        logits = torch.einsum("B D, N D -> B N", q, self.codebook)
        logits = torch.scatter( # I think this operation is not necessary
            logits, dim=-1, index=ids, 
            src=torch.ones_like(ids).float().fill_(-1.e4)
        )
        return freerec.criterions.cross_entropy_with_logits(
            logits, positives, reduction="mean"
        )
    
    def forward(
        self, z: torch.Tensor,
        apply_sinkhorn_distance: bool = True
    ):
        self.set_kmeans_codebook(z)

        codebook = self.codebook
        dist = torch.cdist(z, codebook, p=2) # (B, K)
        if apply_sinkhorn_distance and self.sk_epsilon > 0:
            dist = -sinkhorn_algorithm(dist, self.sk_epsilon, self.sk_iters)
        ids = torch.argmin(dist, dim=-1) # (B,)

        q = codebook[ids] # (B, D)
        loss = self.criterion(z.detach(), q) \
            + cfg.commit_weight * self.criterion(q.detach(), z) \
            + cfg.diversity_weight * self.calc_diversity_loss(q, ids)

        return z + (q - z).detach(), loss, ids


class LETTER(freerec.models.RecSysArch):

    def __init__(
        self, dataset: freerec.data.datasets.RecDataSet
    ) -> None:
        super().__init__(dataset)

        self.Item.add_module(
            "semEmbds",
            nn.Embedding.from_pretrained(
                F.normalize(
                    freerec.utils.import_pickle(
                        os.path.join(
                            self.dataset.path,
                            cfg.sem_feat_file,
                        )
                    ),
                    dim=-1
                ),
                freeze=True
            )
        )

        self.Item.add_module(
            "collabEmbds",
            nn.Embedding.from_pretrained(
                F.normalize(
                    freerec.utils.import_pickle(
                        cfg.collab_feat_file,
                    ),
                    dim=-1
                ),
                freeze=True
            )
        )

        ACT = nn.ReLU
        dims = [self.Item.semEmbds.weight.size(1)] + cfg.hidden_dims + [cfg.codebook_dim]

        self.encoder = nn.Sequential()
        for l, (input_dim, output_dim) in enumerate(zip(dims[:-1], dims[1:]), start=1):
            self.encoder.append(
                nn.Dropout(cfg.dropout_rate)
            )
            self.encoder.append(
                nn.Linear(input_dim, output_dim, bias=False)
            )
            if l < len(dims) - 1:
                self.encoder.append(
                    ACT()
                )

        self.quantizers = nn.ModuleList([
            Quantizer(
                num_codewords=cfg.num_codewords,
                codebook_dim=cfg.codebook_dim, 
                apply_kmeans_init=cfg.apply_kmeans_init,
                sk_epsilon=cfg.sk_epsilons[l],
                sk_iters=cfg.sk_iters
            )
            for l in range(cfg.num_codebooks)
        ])
        for quantizer in self.quantizers:
            quantizer.reset_training_buffers()

        self.decoder, dims = nn.Sequential(), dims[::-1]
        for l, (input_dim, output_dim) in enumerate(zip(dims[:-1], dims[1:]), start=1):
            self.decoder.append(
                nn.Dropout(cfg.dropout_rate)
            )
            self.decoder.append(
                nn.Linear(input_dim, output_dim, bias=False)
            )
            if l < len(dims) - 1:
                self.decoder.append(
                    ACT()
                )

        self.criterion = nn.MSELoss(reduction="mean")

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)

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
        qs, ids = [], []

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
        return F.normalize(x_hat, dim=-1) # normalization !!!

    def fit(
        self, data: Dict[freerec.data.fields.Field, torch.Tensor]
    ) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        items = data[self.Item].flatten()
        x = self.Item.semEmbds(items)

        q, auxiliary_loss, _ = self.encode(x)
        x_hat = self.decode(q)

        logits = torch.einsum(
            "M D, N D -> M N", 
            x_hat,
            self.Item.collabEmbds(items)
        )
        labels = torch.arange(0, x_hat.size(0), device=self.device, dtype=torch.long)
        cf_loss = freerec.criterions.cross_entropy_with_logits(
            logits, labels
        )

        loss = self.criterion(x_hat, x) + auxiliary_loss + cfg.cf_weight * cf_loss

        return loss

    @torch.no_grad()
    def generate_sem_ids(self):
        sem_ids = []
        items = torch.arange(0, self.Item.count, device=self.device)
        for items in items.split(cfg.batch_size):
            x = self.Item.semEmbds(items)
            _, _, ids = self.encode(x)
            sem_ids.append(ids.detach().cpu())
        
        sem_ids = torch.cat(sem_ids, dim=0) # (N, #levels)
        return sem_ids


class CoachForLETTER(freerec.launcher.Coach):

    @freerec.ddp.main_process_only
    def save_sid_vocab(self) -> None:
        sem_ids = self.get_res_sys_arch().generate_sem_ids()
        freerec.utils.export_pickle(
            sem_ids,
            os.path.join(
                self.cfg.LOG_PATH,
                f"sem_id.pkl"
            )
        )

    def set_other(self):
        self.register_metric("RECON_LOSS", lambda x: x, best_caster=min)
        self.register_metric("COMMIT_LOSS", lambda x: x, best_caster=min)
        self.register_metric("PPL", lambda x: x, best_caster=max)
        self.register_metric("COLLISION_RATE", lambda x: x, best_caster=min)
        for i in range(self.cfg.num_codebooks):
            self.register_metric(f"PPL#{i}", lambda x: x, best_caster=max)

    def train_per_epoch(self, epoch: int):

        quantizer: Quantizer
        for quantizer in self.get_res_sys_arch().quantizers:
            quantizer.reset_training_buffers()

        for data in self.dataloader:
            data = self.dict_to_device(data)
            loss = self.model(data) / data[self.Size]

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
           
            self.monitor(
                loss.item(), 
                n=data[self.Size], reduction="mean", 
                mode='train', pool=['LOSS']
            )

        if epoch % self.cfg.eval_freq == 0:
            self.save_sid_vocab()

    def evaluate(self, epoch, step=-1, mode="valid"):
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


def main():

    dataset: freerec.data.datasets.RecDataSet
    try:
        dataset = getattr(freerec.data.datasets, cfg.dataset)(root=cfg.root)
    except AttributeError:
        dataset = freerec.data.datasets.RecDataSet(cfg.root, cfg.dataset, tasktag=cfg.tasktag)

    model = LETTER(dataset)

    # datapipe
    trainpipe = model.sure_trainpipe(cfg.batch_size)
    validpipe = model.sure_validpipe(cfg.batch_size)

    coach = CoachForLETTER(
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