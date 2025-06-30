

from typing import Dict, Tuple, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import freerec

from modules import SemIDEmbedding

freerec.declare(version='1.0.1')

cfg = freerec.parser.Parser()
cfg.add_argument("--maxlen", type=int, default=50)
cfg.add_argument("--num-heads", type=int, default=1)
cfg.add_argument("--num-blocks", type=int, default=2)
cfg.add_argument("--embedding-dim", type=int, default=64)
cfg.add_argument("--dropout-rate", type=float, default=0.2)
cfg.add_argument("--sem-id-ckpt", type=str, default=None, help="checkpoint file of sem_ids")

cfg.set_defaults(
    description="SASRec",
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
        self.num_levels: int = self.Item.embeddings.num_levels
        self.num_codes: List[int] = self.Item.embeddings.num_codes

        self.maxlen = (cfg.maxlen + 2) * self.num_levels # Each Item occupies #levels tokens
        self.Position = nn.Embedding(self.maxlen, cfg.embedding_dim)
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
        ).seq_train_yielding_pos_(
            start_idx_for_target=1, end_idx_for_input=-1
        ).add_(
            offset=self.NUM_PADS, modified_fields=(self.ISeq, self.IPos)
        ).lpad_(
            maxlen, modified_fields=(self.ISeq, self.IPos),
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

    def shrink_pads(self, seqs: torch.Tensor):
        mask = seqs.ne(self.PADDING_VALUE)
        keeped = mask.any(dim=0, keepdim=False)
        return seqs[:, keeped] # (B, S)
    
    def mark_position(self, seqs: torch.Tensor):
        positions = self.Position(self.positions) # (maxlen, D)
        return seqs + positions[-seqs.size(1):]

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
        seqs = self.shrink_pads(seqs)
        userEmbds = self.encode(seqs)
        indices = seqs != self.PADDING_VALUE
        userEmbds = userEmbds[indices] # (M, D)
        itemEmbds = self.Item.embeddings.weight

        logits = torch.einsum("MD,ND->MN", userEmbds, itemEmbds) # (M, N)
        labels = self.Item.embeddings.item_ids_to_sem_ids(data[self.IPos])
        indices = labels != self.PADDING_VALUE
        labels = labels[indices] # (M,)
        rec_loss = self.criterion(logits, labels)

        return rec_loss

    def beam_search(
        self, 
        data: Dict[freerec.data.fields.Field, torch.Tensor],
        temperature: float = 1.
    ) -> torch.Tensor:
        # historical sequence
        seqs: torch.Tensor = self.Item.embeddings.item_ids_to_sem_ids(data[self.ISeq])
        itemEmbds = self.Item.embeddings.weight

        (B, S), K = seqs.shape, cfg.num_beams

        seqs = seqs.repeat_interleave(K, dim=0) # (B * K, S)
        scores = torch.zeros((B, K), device=self.device)
        # mask the last K-1 starts to avoid repeated sampling at first turn
        scores[:, 1:] = -1e9 

        seqs, scores = seqs.view(B * K, -1), scores.view(B * K, 1)

        start = self.NUM_PADS
        for l in range(self.num_levels):
            N = self.num_codes[l]
            end = start + N

            userEmbds = self.encode(
                seqs
            )[:, -1, :] # (B * K, D)

            is_valid = self.Item.embeddings.check_validity(
                torch.cat(
                    (
                        seqs[:, S:].repeat_interleave(N, dim=0),
                        torch.arange(start, end, device=self.device).repeat(B * K).unsqueeze(-1)
                    ),
                    dim=1
                )
            ).view(B, K * N)
            logits = torch.einsum("MD,ND->MN", userEmbds, itemEmbds[start:end])
            logp = F.log_softmax(logits / temperature, dim=-1).add(scores).view(B, -1) # (B, K * N)
            logp[~is_valid] = -1e9
            scores, indices = logp.topk(k=K, dim=-1, largest=True, sorted=True) # (B, K)

            prev_beam_tokens = indices // N # (B, K)
            next_beam_tokens = indices % N # (B, K)
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
        generated = seqs[..., -self.num_levels:] # (B, K, num_levels)
        recommended = self.Item.embeddings.sem_ids_to_item_ids(generated) # (B, K)
        return recommended

    def recommend_from_full(
        self, data: Dict[freerec.data.fields.Field, torch.Tensor]
    ) -> torch.Tensor:
        itemIDs = self.beam_search(data)
        B, K, N = itemIDs.size(0), itemIDs.size(1), self.Item.count + self.NUM_PADS

        scores = torch.zeros((B, N), device=self.device)
        scores = scores.scatter(
            dim=1, index=itemIDs,
            src=torch.arange(K + 1, 0, -1, device=self.device, dtype=torch.float).repeat((B, 1))
        )
        return scores[:, self.NUM_PADS:]

    def recommend_from_pool(
        self, data: Dict[freerec.data.fields.Field, torch.Tensor]
    ) -> torch.Tensor:

        itemIDs = self.beam_search(data)
        itemEmbds = self.Item.embeddings.weight
        B, K, N = itemIDs.size(0), itemIDs.size(1), self.Item.count + self.NUM_PADS

        scores = torch.zeros((B, N), device=self.device)
        scores = scores.scatter(
            dim=1, index=itemIDs,
            src=torch.arange(K + 1, 0, -1, device=self.device, dtype=torch.float).repeat((B, 1))
        )
        scores = scores[:, self.NUM_PADS:]
        candidates = data[self.IUnseen]
        scores = scores.gather(
            dim=1, index=candidates
        )
        return scores

class CoachForSASRec(freerec.launcher.Coach):

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


def main():

    dataset: freerec.data.datasets.RecDataSet
    try:
        dataset = getattr(freerec.data.datasets, cfg.dataset)(root=cfg.root)
    except AttributeError:
        dataset = freerec.data.datasets.RecDataSet(cfg.root, cfg.dataset, tasktag=cfg.tasktag)

    model = SASRec(dataset)

    # datapipe
    trainpipe = model.sure_trainpipe(cfg.maxlen, cfg.batch_size)
    validpipe = model.sure_validpipe(cfg.maxlen, ranking=cfg.ranking, batch_size=64)
    testpipe = model.sure_testpipe(cfg.maxlen, ranking=cfg.ranking, batch_size=64)

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