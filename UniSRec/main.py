

from typing import Dict, Tuple, Union

import torch, os
import torch.nn as nn
import torch.nn.functional as F
import freerec
from freerec.data.tags import ITEM, ID

from modules import MoEAdaptorLayer, MultiHeadAttention, FeedForward

freerec.declare(version='1.0.1')

cfg = freerec.parser.Parser()
cfg.add_argument("--maxlen", type=int, default=50)
cfg.add_argument("--num-heads", type=int, default=1)
cfg.add_argument("--num-blocks", type=int, default=2)
cfg.add_argument("--embedding-dim", type=int, default=64)
cfg.add_argument("--hidden-dropout-rate", type=float, default=0.2)
cfg.add_argument("--attn-dropout-rate", type=float, default=0.2)
cfg.add_argument("--adaptor-dropout-rate", type=float, default=0.2)
cfg.add_argument("--num-moe-experts", type=int, default=8)
cfg.add_argument("--T", type=float, default=0.02, help="T(emperature)")
cfg.add_argument("--mask-ratio", type=float, default=0.2)
cfg.add_argument("--s2sloss-weight", type=float, default=1.e-3)

cfg.add_argument("--tfile", type=str, default=None)

cfg.set_defaults(
    description="UniSRec",
    root="../../data",
    dataset='Amazon2014Beauty_1000_LOU,Amazon2014Home_1000_LOU,Amazon2014CDs_1000_LOU,Amazon2014Video_1000_LOU,Amazon2014Movies_1000_LOU',
    epochs=200,
    batch_size=256,
    gradient_accumulation_steps=1,
    optimizer='adam',
    lr=1e-3,
    weight_decay=0.,
    seed=1,
)
cfg.compile()

cfg.dataset = cfg.dataset.split(',')


class UniSRec(freerec.models.SeqRecArch):

    def __init__(
        self, datasets: Dict[str, freerec.data.datasets.RecDataSet],
    ) -> None:
        super().__init__(datasets[cfg.dataset[0]])

        self.datasets = datasets
        self.embedding_dim = cfg.embedding_dim
        self.num_blocks = cfg.num_blocks

        # Embedding Blocks
        self.starts_and_ends = dict()
        start = self.NUM_PADS
        for name in cfg.dataset:
            dataset = self.datasets[name]
            Item = dataset.fields[ITEM, ID]
            self.starts_and_ends[name] = (start, start + Item.count)
            start = self.starts_and_ends[name][-1]
        
        embeddings = []
        if cfg.tfile is not None:
            for name in cfg.dataset:
                dataset = self.datasets[name]
                tFeats = freerec.utils.import_pickle(
                    os.path.join(dataset.path, cfg.tfile)
                )
                embeddings.append(tFeats)

            embeddings = torch.cat(embeddings, dim=0)
            embeddings = torch.cat(
                (torch.zeros_like(embeddings)[:self.NUM_PADS], embeddings),
                dim=0
            )

            self.Item.add_module(
                "embeddings", nn.Embedding.from_pretrained(
                    embeddings, freeze=True, padding_idx=self.PADDING_VALUE
                )
            )
        else:
            self.Item.add_module(
                "embeddings", nn.Embedding(
                    start, cfg.embedding_dim, padding_idx=self.PADDING_VALUE
                )
            )

        self.Position = nn.Embedding(cfg.maxlen, cfg.embedding_dim)

        self.inputLN = nn.LayerNorm(cfg.embedding_dim, eps=1.e-12)
        self.inputDrop = nn.Dropout(p=cfg.hidden_dropout_rate)

        self.register_buffer(
            "positions",
            torch.tensor(range(0, cfg.maxlen), dtype=torch.long).unsqueeze(0)
        )

        self.moe_adaptor = MoEAdaptorLayer(
            n_exps=cfg.num_moe_experts, 
            input_size=self.Item.embeddings.weight.size(1),
            output_size=cfg.embedding_dim,
            dropout_rate=cfg.adaptor_dropout_rate
        )
        self.attnLayers = nn.ModuleList()
        self.fwdLayers = nn.ModuleList()

        for _ in range(self.num_blocks):

            self.attnLayers.append(
                MultiHeadAttention(
                    n_heads=cfg.num_heads,
                    hidden_size=cfg.embedding_dim,
                    hidden_dropout_prob=cfg.hidden_dropout_rate,
                    attn_dropout_prob=cfg.attn_dropout_rate,
                    layer_norm_eps=1.e-12
                )
            )

            self.fwdLayers.append(
                FeedForward(
                    hidden_size=cfg.embedding_dim,
                    inner_size=cfg.embedding_dim * 4,
                    hidden_dropout_prob=cfg.hidden_dropout_rate,
                    hidden_act='gelu',
                    layer_norm_eps=1.e-12
                )
            )

        self.reset_parameters()

        self.criterion = freerec.criterions.CrossEntropy4Logits(reduction="mean")

    def reset_parameters(self):
        """Initializes the module parameters."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0., std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)

    def sure_trainpipe(self, maxlen: int, batch_size: int):
        datapipes = []
        for name, dataset in self.datasets.items():
            datapipes.append(
                dataset.train().shuffled_roll_seqs_source(
                    minlen=2, maxlen=maxlen, keep_at_least_itself=True
                ).seq_train_yielding_pos_(
                    # [1, 2, 3, 4] -> Input: [1, 2, 3], Target: [4]
                    start_idx_for_target=-1, end_idx_for_input=-1
                ).add_(
                    offset=self.starts_and_ends[name][0], modified_fields=(self.ISeq, self.IPos)
                ).lpad_(
                    maxlen, modified_fields=(self.ISeq,),
                    padding_value=self.PADDING_VALUE
                )
            )
        pipes_weights = {datapipe: 1. for datapipe in datapipes}
        return freerec.data.postprocessing.SampleMultiplexer(pipes_weights).batch_(batch_size).tensor_()

    def sure_validpipe(
        self, maxlen: int, ranking: str = 'full', batch_size: int = 512,
    ):
        datapipes = []
        for name, dataset in self.datasets.items():
            datapipes.append(
                dataset.valid().ordered_user_ids_source(
                ).valid_sampling_(
                    ranking
                ).lprune_(
                    maxlen, modified_fields=(self.ISeq,)
                ).add_(
                    offset=self.starts_and_ends[name][0], modified_fields=(self.ISeq,)
                ).lpad_(
                    maxlen, modified_fields=(self.ISeq,), 
                    padding_value=self.PADDING_VALUE
                ).batch_(batch_size).tensor_().mark_(dataset=name)
            )
        pipes_weights = {datapipe: 1. for datapipe in datapipes}
        return freerec.data.postprocessing.SampleMultiplexer(pipes_weights)

    def sure_testpipe(
        self, maxlen: int, ranking: str = 'full', batch_size: int = 512,
    ):
        datapipes = []
        for name, dataset in self.datasets.items():
            datapipes.append(
                dataset.test().ordered_user_ids_source(
                ).test_sampling_(
                    ranking
                ).lprune_(
                    maxlen, modified_fields=(self.ISeq,)
                ).add_(
                    offset=self.starts_and_ends[name][0], modified_fields=(self.ISeq,)
                ).lpad_(
                    maxlen, modified_fields=(self.ISeq,), 
                    padding_value=self.PADDING_VALUE
                ).batch_(batch_size).tensor_().mark_(dataset=name)
            )
        pipes_weights = {datapipe: 1. for datapipe in datapipes}
        return freerec.data.postprocessing.SampleMultiplexer(pipes_weights)

    def mark_position(self, seqs: torch.Tensor):
        positions = self.Position(self.positions) # (1, maxlen, D)
        return seqs + positions

    def after_one_block(self, seqs: torch.Tensor, attention_mask: torch.Tensor, l: int):
        # inputs: (B, S, D)

        seqs = self.attnLayers[l](seqs, attention_mask)
        seqs = self.fwdLayers[l](seqs)

        return seqs

    def extend_attention_mask(self, seqs: torch.Tensor):
        B, L = seqs.shape
        attention_mask = (seqs != self.PADDING_VALUE).unsqueeze(1).unsqueeze(2) # (B, 1, 1, L)
        attention_mask = attention_mask.expand(-1, -1, L, -1)
        attention_mask = torch.tril(attention_mask)
        attention_mask = torch.where(attention_mask, 0., -1.e4)
        return attention_mask
    
    def encode(
        self, data: Dict[freerec.data.fields.Field, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        seqs = data[self.ISeq]
        attention_mask = self.extend_attention_mask(seqs)

        seqs = self.Item.embeddings(seqs) # (B, S, D0) 
        seqs = self.moe_adaptor(seqs) # -> (B, S, D)
        seqs = self.mark_position(seqs)
        seqs = self.inputLN(seqs)
        seqs = self.inputDrop(seqs)

        for l in range(self.num_blocks):
            seqs = self.after_one_block(seqs, attention_mask, l)

        return F.normalize(seqs[:, -1, :], dim=-1)

    def random_mask(self, seqs: torch.Tensor, p: float):
        rnds = torch.rand(seqs.size(), device=seqs.device)
        masked_seqs = torch.where(
            rnds < p, 
            torch.zeros_like(seqs).fill_(self.PADDING_VALUE),
            seqs
        )
        return masked_seqs

    def fit(
        self, data: Dict[freerec.data.fields.Field, torch.Tensor]
    ) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        userEmbds = self.encode(data) # userEmbds: (B, D); itemEmbds: (N, D)
        posEmbds = self.moe_adaptor(self.Item.embeddings(data[self.IPos]))
        posEmbds = F.normalize(posEmbds, dim=-1).squeeze(1) # (B, D)

        logits = torch.einsum("BD,KD->BK", userEmbds, posEmbds) / cfg.T # (B, B)
        labels = torch.arange(logits.size(1), dtype=torch.long, device=self.device)
        rec_loss = self.criterion(logits, labels)

        maskedSeqs = self.random_mask(data[self.ISeq], p=cfg.mask_ratio)
        maskedEmbds = self.encode({self.ISeq: maskedSeqs}) # (B, D)
        logits = torch.einsum("BD,KD->BK", userEmbds, maskedEmbds) / cfg.T # (B, B)
        labels = torch.arange(logits.size(1), dtype=torch.long, device=self.device)
        s2s_loss = self.criterion(logits, labels)

        return rec_loss, s2s_loss

    def recommend_from_full(
        self, data: Dict[freerec.data.fields.Field, torch.Tensor]
    ) -> torch.Tensor:
        userEmbds = self.encode(data)
        start, end = self.starts_and_ends[data['dataset']]
        itemEmbds = F.normalize(
            self.moe_adaptor(
                self.Item.embeddings.weight[start:end],
            ),
            dim=-1
        )
        return torch.einsum("BD,ND->BN", userEmbds, itemEmbds)

    def recommend_from_pool(
        self, data: Dict[freerec.data.fields.Field, torch.Tensor]
    ) -> torch.Tensor:
        userEmbds = self.encode(data)
        start, end = self.starts_and_ends[data['dataset']]
        itemEmbds = F.normalize(
            self.moe_adaptor(
                self.Item.embeddings.weight[start:end],
            ),
            dim=-1
        )
        itemEmbds = itemEmbds[data[self.IUnseen]] # (B, K, D)
        return torch.einsum("BD,BKD->BK", userEmbds, itemEmbds)

    
class CoachForUniSRec(freerec.launcher.Coach):

    def __init__(self, *, datasets, trainpipe, validpipe, testpipe, model, cfg):
        self.datasets = datasets
        super().__init__(
            dataset=datasets[cfg.dataset[0]], 
            trainpipe=trainpipe, 
            validpipe=validpipe, 
            testpipe=testpipe, 
            model=model, 
            cfg=cfg
        )

    # def set_model(
    #     self, model: freerec.models.RecSysArch
    # ):
    #     self.model = model.to(self.device)
    #     if freerec.ddp.is_distributed():
    #         self.model = torch.nn.parallel.DistributedDataParallel(
    #             model,
    #             find_unused_parameters=True
    #         )

    def set_dataloader(self):
        return super().set_dataloader()

    def set_other(self):
        from freerec.launcher import DEFAULT_METRICS, DEFAULT_FMTS, DEFAULT_BEST_CASTER
        for monitor in self.cfg.monitors:
            if '@' not in monitor:
                continue
            metric, k = monitor.split('@')

            metric = metric.upper()
            for dataset in cfg.dataset:
                self.register_metric(
                    f"{dataset}${metric}@{k}",
                    func=DEFAULT_METRICS[metric],
                    fmt=DEFAULT_FMTS[metric],
                    best_caster=DEFAULT_BEST_CASTER[metric]
                )

    def train_per_epoch(self, epoch: int):
        step = 0
        for data in self.dataloader:
            step += 1
            data = self.dict_to_device(data)
            rec_loss, s2sloss  = self.model(data)
            loss = rec_loss + cfg.s2sloss_weight * s2sloss
            loss = loss / cfg.gradient_accumulation_steps

            loss.backward()
            if step % cfg.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
           
            self.monitor(
                loss.item(), 
                n=len(data[self.User]), reduction="mean", 
                mode='train', pool=['LOSS']
            )

        if step % cfg.gradient_accumulation_steps != 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

    def evaluate(self, epoch: int, step: int = -1, mode: str = 'valid'):
        self.get_res_sys_arch().reset_ranking_buffers()
        for data in self.dataloader:
            bsz = data[self.Size]

            data = self.dict_to_device(data)
            Item = self.datasets[data['dataset']].fields[ITEM, ID]
            if self.cfg.ranking == 'full':
                scores = self.model(data, ranking='full')
                if self.remove_seen:
                    seen = Item.to_csr(data[self.ISeen]).to(self.device).to_dense().bool()
                    scores[seen] = -1e23
                targets = Item.to_csr(data[self.IUnseen]).to(self.device).to_dense()
            elif self.cfg.ranking == 'pool':
                scores = self.model(data, ranking='pool')
                if self.Label in data:
                    targets = data[self.Label]
                else:
                    targets = torch.zeros_like(scores)
                    targets[:, 0].fill_(1)
            else:
                raise NotImplementedError(
                    f"`ranking` should be 'full' or 'pool' but {self.cfg.ranking} received ..."
                )

            self.monitor(
                scores, targets,
                n=bsz, reduction="mean", mode=mode,
                pool=['HITRATE', 'PRECISION', 'RECALL', 'NDCG', 'MRR']
            )
        
            self.monitor(
                scores, targets,
                n=bsz, reduction="mean", mode=mode,
                pool=[f"{data['dataset']}${metric}" for metric in ['HITRATE', 'PRECISION', 'RECALL', 'NDCG', 'MRR']]
            )


def main():

    datasets: Dict[str, freerec.data.datasets.RecDataSet] = dict()
    for dataset in cfg.dataset:
        try:
            datasets[dataset] = getattr(freerec.data.datasets, dataset)(root=cfg.root)
        except AttributeError:
            datasets[dataset] = freerec.data.datasets.NextItemRecDataSet(cfg.root, dataset, tasktag=cfg.tasktag)

    model = UniSRec(datasets)

    # datapipe
    trainpipe = model.sure_trainpipe(cfg.maxlen, cfg.batch_size)
    validpipe = model.sure_validpipe(cfg.maxlen, ranking=cfg.ranking, batch_size=128)
    testpipe = model.sure_testpipe(cfg.maxlen, ranking=cfg.ranking, batch_size=128)

    coach = CoachForUniSRec(
        datasets=datasets,
        trainpipe=trainpipe,
        validpipe=validpipe,
        testpipe=testpipe,
        model=model,
        cfg=cfg
    )
    coach.fit()


if __name__ == "__main__":
    main()