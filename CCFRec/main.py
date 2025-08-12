

from typing import Dict, Tuple, Union, List, Optional

import torch, os
import torch.nn as nn
import torch.nn.functional as F
import freerec
from einops import rearrange, repeat
from sklearn.decomposition import PCA

from modules import TransformerLayer, CrossAttTransformerLayer

freerec.declare(version='1.0.1')

cfg = freerec.parser.Parser()
cfg.add_argument("--maxlen", type=int, default=50)
cfg.add_argument("--num-heads", type=int, default=1)
cfg.add_argument("--embedding-dim", type=int, default=64)
cfg.add_argument("--hidden-size", type=int, default=64 * 4)
cfg.add_argument("--hidden-act", type=str, choices=("gelu", "relu", "swish", "tanh", "sigmoid"), default="gelu")

# loss
cfg.add_argument("--tau", type=float, default=0.07, help="temperature")
cfg.add_argument("--num-negs", type=int, default=49, help="for rec loss")
cfg.add_argument("--weight4mlm", type=float, default=0.1)
cfg.add_argument("--weight4cl", type=float, default=0.1)
cfg.add_argument("--mask-ratio", type=float, default=0.5)

# QFormer
cfg.add_argument("--num-qformer-blocks", type=int, default=2)
cfg.add_argument("--qformer-dropout-rate", type=float, default=0.3)

# Encoder
cfg.add_argument("--num-encoder-blocks", type=int, default=2)
cfg.add_argument("--encoder-dropout-rate", type=float, default=0.5)

cfg.add_argument("--sem-id-ckpt", type=str, default=None, help="checkpoint file of 'sem_ids'")
cfg.add_argument("--tfiles", type=str, default=None, help="checkpoint file of textual features")

cfg.set_defaults(
    description="CCFRec",
    root="../../data",
    dataset='Amazon2014Beauty_550_LOU',
    epochs=500,
    batch_size=2048,
    optimizer='AdamW',
    lr=1e-3,
    weight_decay=0.,
    seed=1,
)
cfg.compile()


class SemIDEmbedding(nn.Embedding):

    def __init__(
        self,
        sem_ids: torch.Tensor,
        embedding_dim: int,
        padding_token: int = 0,
        masking_token: int = 1
    ):
        assert padding_token == 0
        assert masking_token == 1
        N, self.num_codebooks = sem_ids.shape
        num_codewords = sem_ids.max(dim=0)[0] + 1
        num_embeddings = num_codewords.sum().item()
        sem_ids[:, 1:] += num_codewords.cumsum(dim=0)[:-1]
        self.num_codewords: List[int] = num_codewords.tolist()

        num_embeddings += 2
        sem_ids = sem_ids + 2
        sem_ids = F.pad(
            sem_ids,
            (0, 0, 1, 0),
            value=masking_token
        )
        sem_ids = F.pad(
            sem_ids,
            (0, 0, 1, 0),
            value=padding_token
        )

        super().__init__(num_embeddings, embedding_dim, padding_token)
        
        self.register_buffer(
            "sem_ids", sem_ids
        )


class QFormer(nn.Module):

    def __init__(
        self,
        num_blocks: int = 2,
        num_heads: int = 2,
        hidden_size: int = 64,
        inner_size: int = 256,
        hidden_dropout_prob: int = 0.1,
        attn_dropout_prob: int = 0.1,
        hidden_act: str = "gelu",
        layer_norm_eps: float = 1e-8,
    ):
        super().__init__()

        self.num_blocks = num_blocks
        self.blocks = nn.ModuleList([
            CrossAttTransformerLayer(
                n_heads=num_heads,
                hidden_size=hidden_size,
                intermediate_size=inner_size,
                hidden_dropout_prob=hidden_dropout_prob,
                attn_dropout_prob=attn_dropout_prob,
                hidden_act=hidden_act,
                layer_norm_eps=layer_norm_eps
            )
            for _ in range(num_blocks)
        ])

    def forward(
        self, 
        query: torch.Tensor, 
        context: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None, 
        cross_attention_mask: Optional[torch.Tensor] = None
    ):
        for l in range(self.num_blocks):
            query = self.blocks[l](query, context, attention_mask, cross_attention_mask)
        return query


class Transformer(nn.Module):

    def __init__(
        self,
        num_blocks: int = 2,
        num_heads: int = 2,
        hidden_size: int = 64,
        inner_size: int = 256,
        hidden_dropout_prob: int = 0.1,
        attn_dropout_prob: int = 0.1,
        hidden_act: str = "gelu",
        layer_norm_eps: float = 1e-8,
    ):
        super().__init__()

        self.num_blocks = num_blocks
        self.blocks = nn.ModuleList([
            TransformerLayer(
                n_heads=num_heads,
                hidden_size=hidden_size,
                intermediate_size=inner_size,
                hidden_dropout_prob=hidden_dropout_prob,
                attn_dropout_prob=attn_dropout_prob,
                hidden_act=hidden_act,
                layer_norm_eps=layer_norm_eps
            )
            for _ in range(num_blocks)
        ])

    def forward(
        self, 
        query: torch.Tensor, 
        attention_mask: torch.Tensor, 
    ):
        for l in range(self.num_blocks):
            query = self.blocks[l](query, query, attention_mask)
        return query


class CCFRec(freerec.models.SeqRecArch):

    NUM_PADS = 2
    PADDING_VALUE = 0
    MASKING_VALUE = 1

    def __init__(
        self, dataset: freerec.data.datasets.RecDataSet,
    ) -> None:
        super().__init__(dataset)

        self.embedding_dim = cfg.embedding_dim

        self.Item.add_module(
            'embeddings', SemIDEmbedding(
                sem_ids=freerec.utils.import_pickle(
                    os.path.join(self.dataset.path, cfg.sem_id_ckpt)
                ),
                embedding_dim=self.embedding_dim, 
                padding_token=self.PADDING_VALUE,
                masking_token=self.MASKING_VALUE
            )
        )

        features = []
        num_fields = 0
        for tfile in cfg.tfiles.split(','):
            num_fields += 1
            features.append(
                PCA(
                    n_components=self.embedding_dim, whiten=True
                ).fit_transform(
                    freerec.utils.import_pickle(os.path.join(self.dataset.path, tfile)) # (N, D)
                )
            )
        features = [torch.from_numpy(feat).to(torch.float32) for feat in features]
        features = torch.cat(features, dim=1) # (N, K * D)
        features = F.pad(
            features,
            (0, 0, 1, 0), # PADDING_TOKEN == 0
            value=0.
        )
        features = F.pad(
            features,
            (0, 0, 1, 0), # MASKING_TOKEN == 0
            value=0.
        )

        self.register_buffer(
            "features",
            rearrange(features, "N (K D) -> N K D", K=num_fields)
        )

        self.qformer = QFormer(
            num_blocks=cfg.num_qformer_blocks,
            num_heads=cfg.num_heads,
            hidden_size=cfg.embedding_dim,
            inner_size=cfg.hidden_size,
            hidden_dropout_prob=cfg.qformer_dropout_rate,
            attn_dropout_prob=cfg.qformer_dropout_rate,
            hidden_act=cfg.hidden_act,
            layer_norm_eps=1.e-12
        )

        self.encoder = Transformer(
            num_blocks=cfg.num_encoder_blocks,
            num_heads=cfg.num_heads,
            hidden_size=cfg.embedding_dim,
            inner_size=cfg.hidden_size,
            hidden_dropout_prob=cfg.encoder_dropout_rate,
            attn_dropout_prob=cfg.encoder_dropout_rate,
            hidden_act=cfg.hidden_act,
            layer_norm_eps=1.e-12
        )

        self.Position = nn.Embedding(cfg.maxlen, cfg.embedding_dim)
        self.register_buffer(
            "positions",
            torch.tensor(range(0, cfg.maxlen), dtype=torch.long).unsqueeze(0)
        )

        self.inputLN = nn.LayerNorm(cfg.embedding_dim, eps=1.e-12)
        self.inputDrop = nn.Dropout(p=cfg.encoder_dropout_rate)

        self.criterion = freerec.criterions.CrossEntropy4Logits(reduction='mean')

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.LayerNorm):
                m.bias.data.zero_()
                m.weight.data.fill_(1.0)
        nn.init.normal_(self.Item.embeddings.weight, std=0.02)

    def sure_trainpipe(self, maxlen: int, batch_size: int):
        return self.dataset.train().shuffled_roll_seqs_source(
           maxlen=maxlen, keep_at_least_itself=True
        ).seq_train_yielding_pos_(
            start_idx_for_target=-1, end_idx_for_input=-1
        ).seq_train_sampling_neg_(
            num_negatives=cfg.num_negs
        ).add_(
            offset=self.NUM_PADS, modified_fields=(self.ISeq, self.IPos, self.INeg)
        ).lpad_(
            maxlen, modified_fields=(self.ISeq,),
            padding_value=self.PADDING_VALUE
        ).batch_(batch_size).tensor_()

    def mark_position(self, seqs: torch.Tensor):
        positions = self.Position(self.positions) # (1, maxlen, D)
        return seqs + positions

    def extend_attention_mask(self, seqs: torch.Tensor):
        attention_mask = (seqs != self.PADDING_VALUE).unsqueeze(1).unsqueeze(2) # (B, 1, 1, L)
        attention_mask = rearrange(
            seqs != self.PADDING_VALUE,
            "B S -> B 1 1 S",
        )
        attention_mask = repeat(
            attention_mask,
            "B 1 1 S -> B 1 L S",
            L=seqs.size(1)
        )
        attention_mask = torch.tril(attention_mask)
        attention_mask = torch.where(attention_mask, 0., -1.e4)
        return attention_mask

    def encode_item(
        self, items: torch.Tensor, sem_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        if sem_ids is None:
            sem_ids = self.Item.embeddings.sem_ids[items].flatten(end_dim=-2) # (N, K)
        else:
            sem_ids = sem_ids.flatten(end_dim=-2)
        query = self.Item.embeddings(sem_ids) # (N, K, D)
        context = self.features[items].flatten(end_dim=-3) # (N, C, D)

        itemEmbds = self.qformer(query, context) # (N, K, D)

        return {
            'unpooled': itemEmbds.view(*items.shape, itemEmbds.size(-2), itemEmbds.size(-1)), # (*, K, D)
            'pooled': (itemEmbds + query).mean(dim=1).view(*items.shape, itemEmbds.size(-1)) # (*, D)
        }

    def encode_user(
        self, seqs: torch.Tensor, seqEmbds: torch.Tensor
    ) -> torch.Tensor:
        attention_mask = self.extend_attention_mask(seqs)

        seqEmbds = self.mark_position(seqEmbds)
        seqEmbds = self.inputLN(seqEmbds)
        seqEmbds = self.inputDrop(seqEmbds)

        seqEmbds = self.encoder(seqEmbds, attention_mask)

        return F.normalize(seqEmbds[:, -1, :], dim=-1)

    def mask_and_replace(self, sem_ids: torch.Tensor, p: float):
        MASK_THRESHOLD = 0.8
        REPLACE_THRESHOLD = 0.9
        padding_mask = sem_ids == self.PADDING_VALUE
        rnds = torch.rand(sem_ids.size(), device=sem_ids.device)

        # replace
        masked_seqs = torch.where(
            rnds < p * REPLACE_THRESHOLD,
            torch.randint(0, self.Item.embeddings.weight.size(0), size=sem_ids.size()).to(self.device),
            sem_ids
        )

        # mask
        masked_seqs = torch.where(
            rnds < p * MASK_THRESHOLD, 
            torch.ones_like(sem_ids).fill_(self.MASKING_VALUE),
            sem_ids
        )

        masked_seqs.masked_fill_(padding_mask, self.PADDING_VALUE)
        masks = (masked_seqs != sem_ids) # (B, S, K)
        labels = sem_ids[masks]
        return masked_seqs, labels, masks

    def fit(
        self, data: Dict[freerec.data.fields.Field, torch.Tensor]
    ) -> Union[torch.Tensor, Tuple[torch.Tensor]]:

        # (B, S)
        seqs = data[self.ISeq]

        itemDict = self.encode_item(seqs)
        userEmbds = self.encode_user(seqs, itemDict['pooled']) # (B, D)

        positives = data[self.IPos] # (B, 1)
        negatives = data[self.INeg].squeeze(1) # (B, K)
        itemEmbds = self.encode_item(
            torch.cat((positives, negatives), dim=-1)
        )['pooled'] # (B, K + 1, D)
        itemEmbds = F.normalize(itemEmbds, dim=-1)

        logits = torch.einsum("B D, B K D -> B K", userEmbds, itemEmbds) / cfg.tau
        labels = torch.zeros_like(logits[:, 0], dtype=torch.long)
        rec_loss = self.criterion(logits, labels)

        masked_sem_ids, labels, masks = self.mask_and_replace(
            self.Item.embeddings.sem_ids[seqs],
            p=cfg.mask_ratio
        )
        masked_itemDict = self.encode_item(seqs, masked_sem_ids)
        masked_userEmbds = self.encode_user(seqs, masked_itemDict['pooled'])

        unpooled = F.normalize(masked_itemDict['unpooled'], dim=-1)[masks] # (*, D)
        semEmbds = F.normalize(self.Item.embeddings.weight, dim=-1) # (N, D)
        logits = torch.einsum("M D, N D -> M N", unpooled, semEmbds) / cfg.tau
        mlm_loss = self.criterion(logits, labels)

        logits = torch.einsum("M D, N D -> M N", userEmbds, masked_userEmbds) / cfg.tau
        labels = torch.arange(logits.size(0), device=self.device, dtype=torch.long)

        cl_loss = (self.criterion(logits, labels) + self.criterion(logits.T, labels)) / 2

        return rec_loss, mlm_loss, cl_loss

    def reset_ranking_buffers(self):
        itemEmbds = []

        LOCAL_BATCH_SIZE = 256
        items = torch.arange(self.Item.count + self.NUM_PADS, device=self.device)
        itemEmbds = []
        for items_ in torch.split(items, LOCAL_BATCH_SIZE, dim=0):
            itemEmbds.append(
                self.encode_item(items_)['pooled']
            )
        self._itemEmbds = torch.cat(itemEmbds, dim=0)

    def recommend_from_full(
        self, data: Dict[freerec.data.fields.Field, torch.Tensor]
    ) -> torch.Tensor:
        itemEmbds = self._itemEmbds[self.NUM_PADS:]
        userEmbds = self.encode_user(data[self.ISeq], self._itemEmbds[data[self.ISeq]])
        return torch.einsum("BD,ND->BN", userEmbds, itemEmbds)

    def recommend_from_pool(
        self, data: Dict[freerec.data.fields.Field, torch.Tensor]
    ) -> torch.Tensor:
        itemEmbds = self._itemEmbds[self.NUM_PADS:]
        userEmbds = self.encode_user(data[self.ISeq], self._itemEmbds[data[self.ISeq]])
        itemEmbds = itemEmbds[data[self.IUnseen]] # (B, K, D)
        return torch.einsum("BD,BKD->BK", userEmbds, itemEmbds)


class CoachForCCFRec(freerec.launcher.Coach):

    def train_per_epoch(self, epoch: int):
        for data in self.dataloader:
            data = self.dict_to_device(data)
            rec_loss, mlm_loss, cl_loss = self.model(data)
            loss = rec_loss + cfg.weight4mlm * mlm_loss + cfg.weight4cl * cl_loss

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

    model = CCFRec(dataset)

    # datapipe
    trainpipe = model.sure_trainpipe(cfg.maxlen, cfg.batch_size)
    validpipe = model.sure_validpipe(cfg.maxlen, ranking=cfg.ranking)
    testpipe = model.sure_testpipe(cfg.maxlen, ranking=cfg.ranking)

    coach = CoachForCCFRec(
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