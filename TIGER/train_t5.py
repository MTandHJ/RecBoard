

from typing import Dict, Tuple, Union, List

import os
import torch, math
import torch.nn as nn
import torch.nn.functional as F
import freerec
from einops import rearrange
from transformers import T5Model, T5Config
from transformers.cache_utils import EncoderDecoderCache, DynamicCache

from tokenizer import SemIDTokenzier

freerec.declare(version='1.0.1')

cfg = freerec.parser.Parser()
cfg.add_argument("--maxlen", type=int, default=50, help="the maximum (item id) sequence length")
cfg.add_argument("--embedding-dim", type=int, default=128, help="d_model")
cfg.add_argument("--attention-size", type=int, default=64, help="d_kv")
cfg.add_argument("--intermediate-size", type=int, default=64 * 4, help="d_ff")
cfg.add_argument("--num-user-buckets", type=int, default=2000, help="the number of buckets for user hashing")
cfg.add_argument("--num-heads", type=int, default=6, help="the number of attention heads")
cfg.add_argument("--num-layers", type=int, default=4, help="the number of layers")
cfg.add_argument("--dropout-rate", type=float, default=0.1, help="the dropout rate")
cfg.add_argument("--input-dropout-rate", type=float, default=0.1, help="the input dropoutrate")

cfg.add_argument("--sem-id-ckpt", type=str, default=None, help="checkpoint file of 'sem_ids'")
cfg.add_argument("--num-beams", type=int, default=32)

cfg.set_defaults(
    description="TIGER-T5",
    root="../../data",
    dataset='Amazon2014Beauty_550_LOU',
    epochs=500,
    batch_size=256,
    optimizer='AdamW',
    lr=1e-3,
    weight_decay=0.1,
    seed=1,
)
cfg.compile()


class TIGERT5(freerec.models.SeqRecArch):

    def __init__(
        self, dataset: freerec.data.datasets.RecDataSet,
    ) -> None:
        super().__init__(dataset)

        self.embedding_dim = cfg.embedding_dim

        self.tokenizer = SemIDTokenzier(
            sem_ids=freerec.utils.import_pickle(cfg.sem_id_ckpt),
        )
        self.num_codebooks: int = self.tokenizer.num_codebooks
        self.num_codewords: List[int] = self.tokenizer.num_codewords
        self.maxlen = (cfg.maxlen + 1) * self.num_codebooks # Every Item occupies #levels tokens

        self.User.add_module(
            "embeddings",
            nn.Embedding(
                cfg.num_user_buckets,
                embedding_dim=cfg.embedding_dim,
            )
        )

        self.Item.add_module(
            "embeddings",
            nn.Embedding(
                self.tokenizer.vocab_size,
                embedding_dim=cfg.embedding_dim,
            )
        )

        self.maxlen = cfg.maxlen * self.num_codebooks # Every Item occupies #levels tokens
        self.posEncoding = nn.Embedding(self.maxlen, cfg.embedding_dim)
        self.codeEncoding = nn.Embedding(self.num_codebooks, cfg.embedding_dim)

        self.model_config = T5Config(
            vocab_size=0,
            d_model=cfg.embedding_dim,
            d_kv=cfg.attention_size,
            d_ff=cfg.intermediate_size,
            num_layers=cfg.num_layers,
            num_decoder_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            dropout_rate=cfg.dropout_rate,
            output_attentions=False,
        )

        self.model = T5Model(self.model_config)

        self.encoder_input_projector = nn.Sequential(
            # nn.RMSNorm(cfg.embedding_dim),
            nn.LayerNorm(cfg.embedding_dim),
            nn.Dropout(p=cfg.input_dropout_rate),
        )
        
        self.decoder_input_projector = nn.Sequential(
            # nn.RMSNorm(cfg.embedding_dim),
            nn.LayerNorm(cfg.embedding_dim),
            nn.Dropout(p=cfg.input_dropout_rate),
        )

        self.output_projector = nn.Linear(
            cfg.embedding_dim, self.tokenizer.vocab_size, bias=False
        )

        self.criterion = freerec.criterions.CrossEntropy4Logits(reduction='mean')

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)

            if isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(
                    m.weight,
                    std=math.sqrt(1. / m.weight.size(1))
                )

    def sure_trainpipe(self, maxlen: int, batch_size: int):
        return self.dataset.train().shuffled_roll_seqs_source(
            minlen=2, maxlen=maxlen, keep_at_least_itself=True
        ).seq_train_yielding_pos_(
            start_idx_for_target=-1,
            end_idx_for_input=-1
        ).add_(
            offset=self.NUM_PADS, modified_fields=(self.ISeq, self.IPos)
        ).rpad_(
            maxlen, modified_fields=(self.ISeq,),
            padding_value=self.PADDING_VALUE
        ).batch_(batch_size).tensor_()

    def sure_validpipe(
        self, maxlen: int, ranking: str = 'full', batch_size: int = 512,
    ):
        return self.dataset.valid().ordered_user_ids_source(
        ).valid_sampling_(
            ranking
        ).lprune_(
            maxlen, modified_fields=(self.ISeq,)
        ).add_(
            offset=self.NUM_PADS, modified_fields=(self.ISeq,)
        ).rpad_(
            maxlen, modified_fields=(self.ISeq,), 
            padding_value=self.PADDING_VALUE
        ).batch_(batch_size).tensor_()

    def sure_testpipe(
        self, maxlen: int, ranking: str = 'full', batch_size: int = 512,
    ):
        return self.dataset.valid().ordered_user_ids_source(
        ).test_sampling_(
            ranking
        ).lprune_(
            maxlen, modified_fields=(self.ISeq,)
        ).add_(
            offset=self.NUM_PADS, modified_fields=(self.ISeq,)
        ).rpad_(
            maxlen, modified_fields=(self.ISeq,), 
            padding_value=self.PADDING_VALUE
        ).batch_(batch_size).tensor_()

    def shrink_paddings(self, seqs: torch.Tensor):
        mask = seqs.ne(self.PADDING_VALUE)
        keeped = mask.any(dim=0, keepdim=False)
        return seqs[:, keeped] # (B, S)

    def mark_position(self, seqs: torch.Tensor):
        S = seqs.size(1)
        positions = torch.arange(S, dtype=torch.long, device=self.device)
        return seqs + self.posEncoding(positions)

    def mark_codebook(self, seqs: torch.Tensor):
        S = seqs.size(1)
        codebooks = torch.arange(S, dtype=torch.long, device=self.device) % self.num_codebooks
        return seqs + self.codeEncoding(codebooks)
    
    def encode(
        self,
        users: torch.Tensor,
        context: torch.Tensor,
        query: torch.Tensor
    ) -> Dict:

        userEmbds = self.User.embeddings(users % cfg.num_user_buckets) # (B, 1, D)
        seqEmbds = self.Item.embeddings(context) # (B, L, D)
        seqEmbds = self.mark_position(seqEmbds) # (B, L, D)

        # attention_mask: '0' indicates the <pad> token
        attention_mask = torch.cat(
            (
                torch.ones_like(users),
                context.not_equal(self.PADDING_VALUE)
            ),
            dim=-1
        )
        context = torch.cat(
            (userEmbds, seqEmbds), dim=1
        ) # (B, L + 1, D)

        out = self.model(
            inputs_embeds=self.encoder_input_projector(context),
            attention_mask=attention_mask,
            decoder_inputs_embeds=self.decoder_input_projector(
                self.mark_codebook(self.Item.embeddings(query)),
            ),
            decoder_attention_mask=None,
            use_cache=not self.training, return_dict=True
        )
        
        return out

    def fit(
        self, data: Dict[freerec.data.fields.Field, torch.Tensor]
    ) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        users = data[self.User]
        context = data[self.ISeq]
        query = data[self.IPos]

        context = self.shrink_paddings(context)
        context = self.tokenizer.encode(context)
        query = self.tokenizer.encode(query)
        query = F.pad(
            query,
            (1, 0, 0, 0),
            value=self.PADDING_VALUE
        ) # <start> token is <pad> token
        query, target = query[:, :-1], query[:, 1:]

        last_hidden_states = self.encode(users, context, query).last_hidden_state
        logits = self.output_projector(last_hidden_states) # (B, L, C)

        loss = self.criterion(rearrange(logits, "B L C -> B C L"), target)

        return loss

    def beam_search(
        self, 
        data: Dict[freerec.data.fields.Field, torch.Tensor],
        temperature: float = 1.,
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

        Returns
        -------
        torch.Tensor
            Recommended candidates, returned as original item IDs (not semantic IDs).
        """

        # historical sequence
        users = data[self.User]
        context: torch.Tensor = self.tokenizer.encode(data[self.ISeq])
        context = self.shrink_paddings(context)
        itemEmbds = self.output_projector.weight
        generated = torch.zeros_like(users, dtype=torch.long).fill_(self.PADDING_VALUE)

        (B, S), K = generated.shape, cfg.num_beams

        users = users.repeat_interleave(K, dim=0)  # (B * K, 1)
        context = context.repeat_interleave(K, dim=0) # (B * K, S)
        generated = generated.repeat_interleave(K, dim=0) # (B * K, 1)
        scores = torch.zeros((B, K), device=self.device)
        # mask the last K-1 beams to avoid repeated sampling at the first turn
        scores[:, 1:] = -1e9 

        generated, scores = generated.view(B * K, -1), scores.view(B * K, 1)

        out = self.encode(users, context, generated)
        # keep encoder outputs as they are unchanged in search
        encoder_last_hidden_state = out.encoder_last_hidden_state

        start = self.NUM_PADS # remove the <pad> token
        for l in range(self.num_codebooks):
            N = self.num_codewords[l]
            end = start + N

            userEmbds = out.last_hidden_state[:, -1, :] # (B * K, D)
            # past_key_values includes:
            # 1. self-attention cache
            # 2. cross-attention cache
            past_key_values: EncoderDecoderCache = out.past_key_values

            logits = torch.einsum("MD,ND->MN", userEmbds, itemEmbds[start:end])
            logp = F.log_softmax(logits / temperature, dim=-1).add(scores).view(B, -1) # (B, K * N)
            scores, indices = logp.topk(k=K, dim=-1, largest=True, sorted=True) # (B, K)

            prev_beam_tokens = indices // N # (B, K)
            next_beam_tokens = indices % N # (B, K)
            prev_beam_tokens += (torch.arange(B, dtype=torch.long, device=self.device) * K).unsqueeze(-1)
            next_beam_tokens += start

            next_beam_tokens = next_beam_tokens.flatten().unsqueeze(-1) # (B * K, 1)
            prev_beam_tokens = prev_beam_tokens.flatten()

            generated = generated[prev_beam_tokens] # (B * K, L)

            self_attention_cache = []
            cross_attention_cache = []
            for layer_cache in past_key_values:
                if layer_cache is not None:
                    self_key_cache, self_value_cache, cross_key_cache, cross_value_cache = layer_cache
                    new_self_key_cache = self_key_cache[prev_beam_tokens]
                    new_self_value_cache = self_value_cache[prev_beam_tokens]
                    new_cross_key_cache = cross_key_cache[prev_beam_tokens]
                    new_cross_value_cache = cross_value_cache[prev_beam_tokens]
                    self_attention_cache.append(
                        (new_self_key_cache, new_self_value_cache)
                    )
                    cross_attention_cache.append(
                        (new_cross_key_cache, new_cross_value_cache)
                    )
                else:
                    self_attention_cache.append(None)
                    cross_attention_cache.append(None)
            past_key_values = EncoderDecoderCache(
                DynamicCache(self_attention_cache, self.model_config),
                DynamicCache(cross_attention_cache, self.model_config)
            )

            generated = torch.cat(
                (
                    generated,
                    next_beam_tokens
                ),
                dim=-1
            ) # (B * K, L + 1)

            scores = scores.view(B * K, 1)

            if l < self.num_codebooks - 1:
                out = self.model(
                    decoder_inputs_embeds=self.decoder_input_projector(
                        self.mark_codebook(self.Item.embeddings(generated)),
                    ),
                    use_cache=True, return_dict=True,
                    past_key_values=past_key_values,
                    encoder_outputs=[encoder_last_hidden_state],
                    output_attentions=False,
                )

            start += N

        generated = generated.view(B, K, -1)
        ranked_sem_ids = generated[..., -self.num_codebooks:] # (B, K, num_codebooks)
        ranked_item_ids = self.tokenizer.decode(ranked_sem_ids) # (B, K)
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


class CoachForTIGER(freerec.launcher.Coach):

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

    model = TIGERT5(dataset)

    # datapipe
    trainpipe = model.sure_trainpipe(cfg.maxlen, cfg.batch_size)
    validpipe = model.sure_validpipe(cfg.maxlen, ranking=cfg.ranking, batch_size=16)
    testpipe = model.sure_testpipe(cfg.maxlen, ranking=cfg.ranking, batch_size=16)

    coach = CoachForTIGER(
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