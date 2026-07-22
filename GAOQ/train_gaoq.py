import json
import os
from collections import defaultdict

import freerec
import numpy as np
import torch
import torch.nn as nn
from converter import SemIDConverter
from sklearn.cluster import KMeans
from utils import hungarian_match

freerec.declare(version="1.0.1")

cfg = freerec.parser.Parser()
cfg.add_argument("--sem-feat-file", type=str, default=None)
cfg.add_argument("--num-codewords", type=str, default="96,192")
cfg.add_argument("--l2norm", type=eval, default=False)
cfg.add_argument("--use-balancedkmeans", type=eval, default=True)

cfg.set_defaults(
    description="GAOQ",
    root="../../data",
    dataset="Amazon2014Beauty_550_LOU",
    epochs=1,
    batch_size=256,
    optimizer="AdamW",
    lr=1e-3,
    weight_decay=0.0,
    seed=1,
)
cfg.compile()


NUM_CODEWORDS = tuple(int(size) for size in cfg.num_codewords.split(","))
if not NUM_CODEWORDS or min(NUM_CODEWORDS) <= 0:
    raise ValueError("num-codewords should contain positive comma-separated integers")
NUM_CODEBOOKS = len(NUM_CODEWORDS) + 1


class GAOQ(freerec.models.RecSysArch):
    r"""Generate globally aligned orthogonal semantic IDs."""

    def __init__(self, dataset: freerec.data.datasets.RecDataSet) -> None:
        super().__init__(dataset)

        self.Item.add_module(
            "embeddings",
            nn.Embedding.from_pretrained(
                freerec.utils.import_pickle(
                    os.path.join(
                        self.dataset.path,
                        cfg.sem_feat_file,
                    )
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

    def _make_kmeans(self, n_clusters: int, num_items: int):
        n_clusters = min(n_clusters, max(1, num_items))
        if not cfg.use_balancedkmeans:
            return KMeans(
                n_clusters=n_clusters,
                n_init="auto",
                random_state=0,
            )

        from k_means_constrained import KMeansConstrained

        base = num_items // n_clusters
        rem = num_items % n_clusters
        return KMeansConstrained(
            n_clusters=n_clusters,
            size_min=base,
            size_max=base + int(rem > 0),
            random_state=0,
            n_init=3,
        )

    @staticmethod
    def _make_orthogonal_codebook(dim: int, num_codewords: int) -> np.ndarray:
        assert num_codewords <= dim
        np.random.seed(0)
        A = np.random.normal(size=(dim, num_codewords))
        Q, _ = np.linalg.qr(A)
        return Q.T

    @staticmethod
    def _group_by_prefix(sem_ids):
        groups = defaultdict(list)
        for item_id, prefix in enumerate(zip(*sem_ids)):
            groups[prefix].append(item_id)
        return groups

    def gaoq(self, z: np.ndarray) -> np.ndarray:
        if cfg.l2norm:
            z = z / np.linalg.norm(z, axis=1, keepdims=True)

        N, D = z.shape
        kmeans = self._make_kmeans(NUM_CODEWORDS[0], N)
        kmeans.fit(z)

        codes = kmeans.labels_
        sem_ids = [codes]
        codebook_per_item = kmeans.cluster_centers_[codes]

        for num_codewords in NUM_CODEWORDS[1:]:
            global_codebook = self._make_orthogonal_codebook(D, num_codewords)
            codes = np.zeros(N, dtype=np.int32)
            next_codebook_per_item = np.zeros_like(z)

            for indices in self._group_by_prefix(sem_ids).values():
                if len(indices) == 1:
                    next_codebook_per_item[indices] = codebook_per_item[indices]
                    continue

                sub_z = z[indices]
                kmeans = self._make_kmeans(num_codewords, len(indices))
                kmeans.fit(sub_z)

                local_codebook = kmeans.cluster_centers_
                sub_codes = kmeans.labels_
                next_codebook_per_item[indices] = local_codebook[sub_codes]

                residual_codebook = local_codebook - codebook_per_item[indices[0]]
                global_id_mapping, _ = hungarian_match(residual_codebook, global_codebook)
                codes[indices] = global_id_mapping[sub_codes]

            unique_codes = np.unique(codes)
            code_mapping = {code: idx for idx, code in enumerate(unique_codes)}
            codes = np.array([code_mapping[code] for code in codes], dtype=np.int32)
            sem_ids.append(codes)
            codebook_per_item = next_codebook_per_item

        groups = self._group_by_prefix(sem_ids)
        num_codewords = max(len(indices) for indices in groups.values())
        global_codebook = self._make_orthogonal_codebook(D, num_codewords)
        codes = np.zeros(N, dtype=np.int32)

        for indices in groups.values():
            if len(indices) == 1:
                continue
            residual = z[indices] - codebook_per_item[indices]
            codes[indices], _ = hungarian_match(residual, global_codebook)
        sem_ids.append(codes)

        # TIGER semantic IDs are zero-based; ReSID's final +1 is an export detail.
        return np.stack(sem_ids, axis=-1)

    @torch.no_grad()
    def generate_sem_ids(self):
        is_training = self.training
        self.eval()
        try:
            if self._sem_ids is None:
                z = self.Item.embeddings.weight.cpu().numpy()
                self._sem_ids = torch.from_numpy(self.gaoq(z)).long()
            return self._sem_ids
        finally:
            self.train(is_training)


class CoachForGAOQ(freerec.launcher.Coach):
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
        self.register_metric("PPL", lambda x: x, best_caster=max)
        self.register_metric("COLLISION_RATE", lambda x: x, best_caster=min)
        for i in range(NUM_CODEBOOKS):
            self.register_metric(f"PPL#{i}", lambda x: x, best_caster=max)

    def train_per_epoch(self, epoch: int):
        self.save_sid_vocab()

    def evaluate(self, epoch, step=-1, mode="valid"):
        sem_ids = self.model.generate_sem_ids().cpu()
        max_num_codewords = int(sem_ids.max().item()) + 1
        counts = torch.zeros((max_num_codewords, NUM_CODEBOOKS))
        counts.scatter_add_(0, sem_ids, torch.ones_like(sem_ids, dtype=torch.float))
        uniques = set(map(tuple, sem_ids.tolist()))

        freqs = counts.div(counts.sum(dim=0, keepdim=True))
        perplexity = ((freqs + 1.0e-8).log() * freqs).sum(dim=0).neg().exp().tolist()

        ppls = []
        for i, ppl in enumerate(perplexity):
            ppls.append(ppl)
            self.monitor(ppl, n=1, mode=mode, pool=[f"PPL#{i}"])

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

    model = GAOQ(dataset)

    # datapipe
    trainpipe = model.sure_trainpipe(cfg.batch_size)
    validpipe = model.sure_validpipe(cfg.batch_size)

    coach = CoachForGAOQ(
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
