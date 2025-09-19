

import torch, os
import torch.nn as nn
import torch.nn.functional as F
import freerec

freerec.declare(version='1.0.1')

cfg = freerec.parser.Parser()
cfg.add_argument("--num-codebooks", type=int, default=3)
cfg.add_argument("--num-codewords", type=int, default=256)
cfg.add_argument("--num-iters", type=int, default=10)
cfg.add_argument("--kmeans-init-method", type=str, choices=("random", "points", "++", "matrix"), default="random")
cfg.add_argument("--sem-feat-file", type=str, default=None)

cfg.set_defaults(
    description="R-KMeans",
    root="../../data",
    dataset='Amazon2014Beauty_1000_LOU',
    epochs=1,
    batch_size=256,
    optimizer='AdamW',
    lr=1e-3,
    weight_decay=0.,
    seed=1,
)
cfg.compile()


class RKMeans(freerec.models.RecSysArch):

    def __init__(
        self, dataset: freerec.data.datasets.RecDataSet
    ) -> None:
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
                    dim=-1
                ),
                freeze=True
            )
        )

        self._sem_ids = None

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

    @torch.no_grad()
    def generate_sem_ids(self):
        if self._sem_ids is None:
            from scipy.cluster.vq import kmeans2
            sem_ids = []
            z = self.Item.embeddings.weight.cpu().numpy()

            for l in range(cfg.num_codebooks):
                codebook, codes = kmeans2(
                    z,
                    k=cfg.num_codewords, iter=cfg.num_iters,
                    minit=cfg.kmeans_init_method
                )
                sem_ids.append(codes)
                q = codebook[codes]
                z = z - q

            sem_ids = [torch.from_numpy(ids) for ids in sem_ids]
            sem_ids = torch.stack(sem_ids, dim=-1).long()
            self._sem_ids = sem_ids
        return self._sem_ids


class CoachForRKMeans(freerec.launcher.Coach):

    def save_checkpoint(self, epoch):
        super().save_checkpoint(epoch)
        if freerec.ddp.is_main_process():
            sem_ids = self.get_res_sys_arch().generate_sem_ids()
            freerec.utils.export_pickle(
                sem_ids,
                os.path.join(
                    self.cfg.LOG_PATH,
                    f"sem_id_{epoch}.pkl"
                )
            )

    def set_other(self):
        self.register_metric(
            f"RECON_LOSS", lambda x: x, best_caster=min
        )
        for i in range(self.cfg.num_codebooks):
            self.register_metric(
                f"PPL#{i}", lambda x: x, best_caster=max
            )

    def train_per_epoch(self, epoch: int):
        pass

    def evaluate(self, epoch, step = -1, mode = 'valid'):
        sem_ids = self.get_res_sys_arch().generate_sem_ids()
        counts = torch.zeros((cfg.num_codewords, cfg.num_codebooks))
        counts.scatter_add_(
            0, sem_ids, torch.ones_like(sem_ids).float()
        )
        freqs = counts.div(counts.sum(dim=0, keepdim=True))
        perplexity = ((freqs + 1.e-8).log() * freqs).sum(dim=0).neg().exp().tolist()

        for i, ppl in enumerate(perplexity):
            self.monitor(
                ppl, n=1, mode='valid',
                pool=[f"PPL#{i}"]
            )


def main():

    dataset: freerec.data.datasets.RecDataSet
    try:
        dataset = getattr(freerec.data.datasets, cfg.dataset)(root=cfg.root)
    except AttributeError:
        dataset = freerec.data.datasets.RecDataSet(cfg.root, cfg.dataset, tasktag=cfg.tasktag)

    model = RKMeans(dataset)

    # datapipe
    trainpipe = model.sure_trainpipe(cfg.batch_size)
    validpipe = model.sure_validpipe(cfg.batch_size)

    coach = CoachForRKMeans(
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