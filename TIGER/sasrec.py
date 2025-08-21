

from typing import Dict, Tuple, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import freerec

from modules import SemIDEmbedding

freerec.declare(version='1.0.1')

cfg = freerec.parser.Parser()
cfg.add_argument("--maxlen", type=int, default=50, help="the maximum (item id) sequence length")
cfg.add_argument("--num-heads", type=int, default=1)
cfg.add_argument("--num-blocks", type=int, default=2)
cfg.add_argument("--embedding-dim", type=int, default=64)
cfg.add_argument("--dropout-rate", type=float, default=0.2)
cfg.add_argument("--sem-id-ckpt", type=str, default=None, help="checkpoint file of 'sem_ids'")
cfg.add_argument("--num-beams", type=int, default=32)

cfg.set_defaults(
    description="SASRec",
    root="../../data",
    dataset='Amazon2014Beauty_550_LOU',
    epochs=200,
    batch_size=256,
    optimizer='AdamW',
    lr=1e-3,
    weight_decay=0.01,
    seed=1,
)
cfg.compile()


class PointWiseFeedForward(nn.Module):

    def __init__(self, hidden_size: int, dropout_rate: int):
        super(PointWiseFeedForward, self).__init__()

        self.conv1 = nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (B, S, D)
        outputs = self.dropout2(self.conv2(self.relu(
            self.dropout1(self.conv1(inputs.transpose(-1, -2)))
        ))) # -> (B, D, S)
        outputs = outputs.transpose(-1, -2) # -> (B, S, D)
        outputs += inputs
        return outputs


class SASRec(freerec.models.SeqRecArch):

    def __init__(
        self, dataset: freerec.data.datasets.RecDataSet,
    ) -> None:
        super().__init__(dataset)

        self.num_blocks = cfg.num_blocks
        self.embedding_dim = cfg.embedding_dim

        self.Item.add_module(
            'embeddings', SemIDEmbedding(
                sem_ids=freerec.utils.import_pickle(cfg.sem_id_ckpt),
                embedding_dim=cfg.embedding_dim, padding=True
            )
        )
        self.num_codebooks: int = self.Item.embeddings.num_codebooks
        self.num_codewords: List[int] = self.Item.embeddings.num_codewords

        self.maxlen = (cfg.maxlen + 1) * self.num_codebooks # Every Item occupies #levels tokens
        self.posEncoding = nn.Embedding(self.maxlen, cfg.embedding_dim)
        self.semEncoding = nn.Embedding(self.num_codebooks, cfg.embedding_dim)
        self.embdDropout = nn.Dropout(p=cfg.dropout_rate)
        self.register_buffer(
            "positions",
            torch.tensor(
                list(range(0, self.maxlen))[::-1], # Make sure that the last is latest
                dtype=torch.long
            )
        )

        self.attnLNs = nn.ModuleList() # to be Q for self-attention
        self.attnLayers = nn.ModuleList()
        self.fwdLNs = nn.ModuleList()
        self.fwdLayers = nn.ModuleList()

        self.lastLN = nn.LayerNorm(cfg.embedding_dim, eps=1e-8)

        for _ in range(self.num_blocks):
            self.attnLNs.append(nn.LayerNorm(
                cfg.embedding_dim, eps=1e-8
            ))

            self.attnLayers.append(
                nn.MultiheadAttention(
                    embed_dim=cfg.embedding_dim,
                    num_heads=cfg.num_heads,
                    dropout=cfg.dropout_rate,
                    batch_first=True # !!!
                )
            )

            self.fwdLNs.append(nn.LayerNorm(
                cfg.embedding_dim, eps=1e-8
            ))

            self.fwdLayers.append(PointWiseFeedForward(
                cfg.embedding_dim, cfg.dropout_rate
            ))

        # very neccessary !!!
        self.output_projector = nn.Linear(
            cfg.embedding_dim, self.Item.embeddings.weight.size(0), bias=False
        )

        self.criterion = freerec.criterions.CrossEntropy4Logits(reduction='mean')

        self.reset_parameters()

    def reset_parameters(self):
        """Initializes the module parameters."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)

    def sure_trainpipe(self, maxlen: int, batch_size: int):
        return self.dataset.train().shuffled_seqs_source(
           maxlen=maxlen 
        ).add_(
            offset=self.NUM_PADS, modified_fields=(self.ISeq,)
        ).lpad_(
            maxlen, modified_fields=(self.ISeq,),
            padding_value=self.PADDING_VALUE
        ).batch_(batch_size).tensor_()

    def after_one_block(self, seqs: torch.Tensor, padding_mask: torch.Tensor, l: int):
        # seqs: (B, S, D)
        L = seqs.size(1)
        Q = self.attnLNs[l](seqs)
        attnMask = torch.ones((L, L), dtype=torch.bool, device=self.device).triu(diagonal=1)

        seqs = self.attnLayers[l](
            Q, seqs, seqs,
            attn_mask=attnMask,
            need_weights=False
        )[0] + seqs

        seqs = self.fwdLNs[l](seqs)
        seqs = self.fwdLayers[l](seqs)

        return seqs.masked_fill(padding_mask, 0.)

    def shrink_paddings(self, seqs: torch.Tensor):
        mask = seqs.ne(self.PADDING_VALUE)
        keeped = mask.any(dim=0, keepdim=False)
        return seqs[:, keeped] # (B, S)
    
    def mark_position(self, seqs: torch.Tensor):
        S = seqs.size(1)
        positions = torch.arange(S, dtype=torch.long, device=self.device)
        return seqs \
            + self.posEncoding(self.positions)[-S:] \
            + self.semEncoding(positions % self.num_codebooks)

    def encode(
        self, seqs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        padding_mask = (seqs == self.PADDING_VALUE).unsqueeze(-1)
        seqs = self.Item.embeddings(seqs) # (B, S) -> (B, S, D)
        seqs *= self.embedding_dim ** 0.5
        seqs = self.embdDropout(self.mark_position(seqs))
        seqs = seqs.masked_fill(padding_mask, 0.)

        for l in range(self.num_blocks):
            seqs = self.after_one_block(seqs, padding_mask, l)
        
        userEmbds = self.lastLN(seqs) # (B, S, D)

        return userEmbds

    def fit(
        self, data: Dict[freerec.data.fields.Field, torch.Tensor]
    ) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        seqs = data[self.ISeq]
        seqs = self.Item.embeddings.item_ids_to_sem_ids(seqs)
        seqs = self.shrink_paddings(seqs)
        seqs, labels = seqs[:, :-1], seqs[:, 1:]
        userEmbds = self.encode(seqs)

        indices = seqs != self.PADDING_VALUE
        userEmbds = userEmbds[indices] # (M, D)
        logits = self.output_projector(userEmbds) # (M, N)

        labels = labels[indices] # (M,)
        loss = self.criterion(logits, labels)

        return loss

    def beam_search(
        self, 
        data: Dict[freerec.data.fields.Field, torch.Tensor],
        temperature: float = 1.,
        remove_invalid_codes: bool = False
    ) -> torch.Tensor:
        r"""
        Beam search for item recommendation generation.

        Parameters
        ----------
        data : Dict[freerec.data.fields.Field, torch.Tensor]
            Contains the historical sequence.
        temperature : float
            Sampling temperature for controlling exploration. 
            Note: Probabilistic sampling is not supported.
        remove_invalid_codes : bool
            - True: Exclude invalid codes at each decoding step.
            - False: Retain invalid codes during decoding.

        Returns
        -------
        torch.Tensor
            Recommended candidates, returned as original item IDs (not semantic IDs).
        """

        # historical sequence
        seqs: torch.Tensor = self.Item.embeddings.item_ids_to_sem_ids(data[self.ISeq])
        seqs = self.shrink_paddings(seqs)
        itemEmbds = self.output_projector.weight

        (B, S), K = seqs.shape, cfg.num_beams

        seqs = seqs.repeat_interleave(K, dim=0) # (B * K, S)
        scores = torch.zeros((B, K), device=self.device)
        # mask the last K-1 beams to avoid repeated sampling at the first turn
        scores[:, 1:] = -1e9 

        seqs, scores = seqs.view(B * K, -1), scores.view(B * K, 1)

        start = self.NUM_PADS
        for l in range(self.num_codebooks):
            N = self.num_codewords[l]
            end = start + N

            userEmbds = self.encode(
                seqs
            )[:, -1, :] # (B * K, D)

            logits = torch.einsum("MD,ND->MN", userEmbds, itemEmbds[start:end])
            logp = F.log_softmax(logits / temperature, dim=-1).add(scores).view(B, -1) # (B, K * N)
            if remove_invalid_codes:
                is_valid = self.Item.embeddings.check_validity(
                    torch.cat(
                        (
                            seqs[:, S:].repeat_interleave(N, dim=0),
                            torch.arange(start, end, device=self.device).repeat(B * K).unsqueeze(-1)
                        ),
                        dim=1
                    )
                ).view(B, K * N)
                logp[~is_valid] = -1e9
            scores, indices = logp.topk(k=K, dim=-1, largest=True, sorted=True) # (B, K)

            prev_beam_tokens = indices // N # (B, K)
            next_beam_tokens = indices % N # (B, K)
            prev_beam_tokens += (torch.arange(B, dtype=torch.long, device=self.device) * K).unsqueeze(-1)
            next_beam_tokens += start
            seqs = seqs[prev_beam_tokens.flatten()] # (B * K, L)

            seqs = torch.cat(
                (
                    seqs,
                    next_beam_tokens.flatten().unsqueeze(-1) # (B * K, 1)
                ),
                dim=-1
            ) # (B * K, L + 1)
            scores = scores.view(B * K, 1)

            start += N

        seqs = seqs.view(B, K, -1)
        ranked_sem_ids = seqs[..., -self.num_codebooks:] # (B, K, num_codebooks)
        ranked_item_ids = self.Item.embeddings.sem_ids_to_item_ids(ranked_sem_ids) # (B, K)
        num_invalid_item_ids = (ranked_item_ids == self.PADDING_VALUE).sum()
        return ranked_item_ids, num_invalid_item_ids

    def recommend_from_full(
        self, data: Dict[freerec.data.fields.Field, torch.Tensor]
    ) -> torch.Tensor:
        ranked_item_ids, num_invalid_item_ids = self.beam_search(data)
        B, K, N = ranked_item_ids.size(0), ranked_item_ids.size(1), self.Item.count + self.NUM_PADS

        scores = torch.rand((B, N), device=self.device) * 0.001
        scores = scores.scatter(
            dim=1, index=ranked_item_ids,
            src=torch.arange(K + 1, 0, -1, device=self.device, dtype=torch.float).repeat((B, 1))
        )
        return scores[:, self.NUM_PADS:], num_invalid_item_ids

    def recommend_from_pool(
        self, data: Dict[freerec.data.fields.Field, torch.Tensor]
    ) -> torch.Tensor:
        ranked_item_ids, num_invalid_item_ids = self.beam_search(data)
        B, K, N = ranked_item_ids.size(0), ranked_item_ids.size(1), self.Item.count + self.NUM_PADS

        scores = torch.rand((B, N), device=self.device) * 0.001
        scores = scores.scatter(
            dim=1, index=ranked_item_ids,
            src=torch.arange(K + 1, 0, -1, device=self.device, dtype=torch.float).repeat((B, 1))
        )
        scores = scores[:, self.NUM_PADS:]
        candidates = data[self.IUnseen]
        scores = scores.gather(
            dim=1, index=candidates
        )
        return scores, num_invalid_item_ids


class CoachForSASRec(freerec.launcher.Coach):

    def set_other(self):
        self.register_metric(
            "NUM_INVALID", lambda x: x,
            best_caster=min
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

    def evaluate(self, epoch: int, step: int = -1, mode: str = 'valid'):
        self.get_res_sys_arch().reset_ranking_buffers()
        for data in self.dataloader:
            bsz = data[self.Size]

            data = self.dict_to_device(data)
            if self.cfg.ranking == 'full':
                scores, num_invalid_item_ids = self.model(data, ranking='full')
                if self.remove_seen:
                    seen = self.Item.to_csr(data[self.ISeen]).to(self.device).to_dense().bool()
                    scores[seen] = -1e23
                targets = self.Item.to_csr(data[self.IUnseen]).to(self.device).to_dense()
            elif self.cfg.ranking == 'pool':
                scores, num_invalid_item_ids = self.model(data, ranking='pool')
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
                num_invalid_item_ids,
                n=bsz, reduction="sum", mode=mode,
                pool=['NUM_INVALID']
            )


def main():

    dataset: freerec.data.datasets.RecDataSet
    try:
        dataset = getattr(freerec.data.datasets, cfg.dataset)(root=cfg.root)
    except AttributeError:
        dataset = freerec.data.datasets.RecDataSet(cfg.root, cfg.dataset, tasktag=cfg.tasktag)

    model = SASRec(dataset)

    # datapipe
    trainpipe = model.sure_trainpipe(cfg.maxlen, cfg.batch_size)
    validpipe = model.sure_validpipe(cfg.maxlen, ranking=cfg.ranking, batch_size=128)
    testpipe = model.sure_testpipe(cfg.maxlen, ranking=cfg.ranking, batch_size=128)

    coach = CoachForSASRec(
        dataset=dataset,
        trainpipe=trainpipe,
        validpipe=validpipe,
        testpipe=testpipe,
        model=model,
        cfg=cfg
    )
    coach.fit()


if __name__ == "__main__":
    main()