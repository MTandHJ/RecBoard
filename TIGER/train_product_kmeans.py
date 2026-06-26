import json
import math
import os

import freerec
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from converter import SemIDConverter

freerec.declare(version="1.0.1")

cfg = freerec.parser.Parser()
cfg.add_argument("--num-codebooks", type=int, default=3)
cfg.add_argument("--num-codewords", type=int, default=256)
cfg.add_argument("--num-iters", type=int, default=10)
cfg.add_argument("--sem-feat-file", type=str, default=None)

cfg.set_defaults(
    description="PQ-KMeans",
    root="../../data",
    dataset="Amazon2014Beauty_1000_LOU",
    epochs=1,
    batch_size=256,
    optimizer="AdamW",
    seed=1,
)
cfg.compile()


def num_codewords_to_nbits(num_codewords: int) -> int:
    nbits = int(math.log2(num_codewords))
    if num_codewords <= 1 or 2**nbits != num_codewords:
        raise ValueError("num-codewords must be a power of two greater than 1")
    return nbits


class PQKMeans(freerec.models.RecSysArch):
    def __init__(self, dataset: freerec.data.datasets.RecDataSet) -> None:
        super().__init__(dataset)

        self.Item.add_module(
            "embeddings",
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

        self._sem_ids = None

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

    def fit_product_quantizer(self, z: np.ndarray):
        import faiss

        dim = z.shape[1]
        if dim % cfg.num_codebooks != 0:
            raise ValueError("embedding dimension must be divisible by num-codebooks")

        nbits = num_codewords_to_nbits(cfg.num_codewords)
        pq = faiss.ProductQuantizer(dim, cfg.num_codebooks, nbits)
        pq.cp.niter = cfg.num_iters
        pq.verbose = True
        pq.train(z)

        dsub = dim // cfg.num_codebooks
        return faiss.vector_to_array(pq.centroids).reshape(
            cfg.num_codebooks,
            cfg.num_codewords,
            dsub,
        )

    def assign_product_codes(self, z: np.ndarray, centroids: np.ndarray) -> torch.Tensor:
        sem_ids = []
        for subvector, codebook in zip(np.split(z, cfg.num_codebooks, axis=1), centroids):
            codes = []
            codebook_norm = (codebook**2).sum(axis=1)
            for batch in np.array_split(
                subvector, max(1, math.ceil(len(subvector) / cfg.batch_size))
            ):
                batch_norm = (batch**2).sum(axis=1, keepdims=True)
                dist = batch_norm + codebook_norm[None, :] - 2 * batch @ codebook.T
                codes.append(dist.argmin(axis=-1))
            sem_ids.append(torch.from_numpy(np.concatenate(codes)))
        return torch.stack(sem_ids, dim=-1).long()

    @torch.no_grad()
    def generate_sem_ids(self):
        is_training = self.training
        self.eval()
        try:
            if self._sem_ids is None:
                z = self.Item.embeddings.weight.detach().cpu().numpy()
                z = np.ascontiguousarray(z, dtype=np.float32)

                centroids = self.fit_product_quantizer(z)
                self._sem_ids = self.assign_product_codes(z, centroids)
            return self._sem_ids
        finally:
            self.train(is_training)


class CoachForPQKMeans(freerec.launcher.Coach):
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

    def set_other(self):
        self.register_metric("RECON_LOSS", lambda x: x, best_caster=min)
        self.register_metric("PPL", lambda x: x, best_caster=max)
        self.register_metric("COLLISION_RATE", lambda x: x, best_caster=min)
        for i in range(self.cfg.num_codebooks):
            self.register_metric(f"PPL#{i}", lambda x: x, best_caster=max)

    def train_per_epoch(self, epoch: int):
        self.save_sid_vocab()

    def evaluate(self, epoch, step=-1, mode="valid"):
        sem_ids = self.model.generate_sem_ids().cpu()
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

    model = PQKMeans(dataset)

    # datapipe
    trainpipe = model.sure_trainpipe(cfg.batch_size)
    validpipe = model.sure_validpipe(cfg.batch_size)

    coach = CoachForPQKMeans(
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
