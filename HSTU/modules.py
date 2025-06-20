

from typing import Optional, Tuple

import torch, math
import torch.nn as nn
import torch.nn.functional as F


def truncated_normal(x: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    with torch.no_grad():
        size = x.shape
        tmp = x.new_empty(size + (4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        x.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        x.data.mul_(std).add_(mean)
        return x


class LearnablePositionalEmbeddingInputFeaturesPreprocessor(nn.Module):

    def __init__(
        self,
        maxlen: int,
        embedding_dim: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()

        self.embedding_dim: int = embedding_dim
        self.posEmb: torch.nn.Embedding = torch.nn.Embedding(
            maxlen,
            self.embedding_dim,
        )
        self.register_buffer(
            "positions",
            torch.tensor(range(0, maxlen), dtype=torch.long).unsqueeze(0)
        )
        self.emb_dropout = torch.nn.Dropout(p=dropout_rate)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        truncated_normal(
            self.posEmb.weight.data,
            mean=0.0,
            std=math.sqrt(1.0 / self.embedding_dim),
        )

    def forward(
        self,
        seqs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        seqs = seqs * (self.embedding_dim ** 0.5) + self.posEmb(self.positions)
        seqs = self.emb_dropout(seqs)

        return seqs


class RelativeBucketedTimeAndPositionBasedBias(nn.Module):
    """
    Bucketizes timespans based on ts(next-item) - ts(current-item).
    """

    def __init__(
        self,
        maxlen: int,
        num_buckets: int,
    ) -> None:
        super().__init__()

        self.maxlen: int = maxlen
        self.timestamp_weights = torch.nn.Parameter(
            torch.empty(num_buckets + 1).normal_(mean=0, std=0.02),
        )
        self.position_weights = torch.nn.Parameter(
            torch.empty(2 * maxlen - 1).normal_(mean=0, std=0.02),
        )
        self.num_buckets: int = num_buckets

    def _bucketization_fn(self, x: torch.Tensor):
        # 0.301 is close to log10(e)
        return (x.abs().clamp_min(1.).log() / 0.301).long()

    def forward(
        self,
        timestamps: torch.Tensor,
    ) -> torch.Tensor:
        """
        timestamps: (t_1, t_2, \ldots, t_l)
            -> (t_2 - t_1, t_3 - t_2, \ldots, t_l - t_{l - 1}, 0)
            -> buckets -> weights
        
        Parameters:
        -----------
        timestamps: torch.Tensor, (B, L)

        Returns:
        --------
        rel_pos_encoding: torch.Tensor, (B, L, L)
        """
        B = timestamps.size(0)
        L = self.maxlen
        t = F.pad(self.position_weights[:2 * L - 1], [0, L]).repeat(L) # ((3L - 1) * L)
        t = t[..., :-L].reshape(1, L, 3 * L - 2)
        # Example (L=3):
        # tensor([[[-0.0642,  1.7951,  0.0779,  0.5414, -0.0855,  0.0000,  0.0000],
        #         [ 0.0000, -0.0642,  1.7951,  0.0779,  0.5414, -0.0855,  0.0000],
        #         [ 0.0000,  0.0000, -0.0642,  1.7951,  0.0779,  0.5414, -0.0855]]])
        r = (2 * L - 1) // 2

        # timestamps -> rel_timestamps -> buckted
        ext_timestamps = torch.cat(
            [timestamps, timestamps[:, L - 1:L]], dim=1
        )
        bucketed_timestamps = torch.clamp(
            self._bucketization_fn(
                ext_timestamps[:, 1:].unsqueeze(2) - ext_timestamps[:, :-1].unsqueeze(1)
            ),
            min=0,
            max=self.num_buckets,
        ).detach()

        rel_pos_bias = t[:, :, r:-r]
        rel_ts_bias = torch.index_select(
            self.timestamp_weights, dim=0, index=bucketed_timestamps.view(-1)
        ).view(B, L, L)
        return rel_pos_bias + rel_ts_bias


class HSTUBlock(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        linear_hidden_dim: int,
        linear_activation: str,
        attention_dim: int, num_heads: int,
        dropout_rate: float = 0.,
        rel_attn_bias_encoder: Optional[RelativeBucketedTimeAndPositionBasedBias] = None,
        eps: float = 1.e-6
    ):
        super().__init__()

        self.linear_dim = linear_hidden_dim
        self.attention_dim = attention_dim
        self.num_heads = num_heads

        self.uvqk_linear = nn.Linear(
            embedding_dim,
            linear_hidden_dim * 2 * num_heads + attention_dim * 2 * num_heads,
            bias=False
        )
        self.output_linear = nn.Linear(
            linear_hidden_dim * num_heads,
            embedding_dim
        )
        self.rel_attn_bias_encoder = rel_attn_bias_encoder

        self.linear_act = nn.SiLU() if linear_activation.lower() == 'silu' else nn.Identity()
        self.dropout = nn.Dropout(p=dropout_rate)

        self.hidden_layer_norm = nn.LayerNorm(embedding_dim, eps=eps)
        self.attn_layer_norm = nn.LayerNorm(linear_hidden_dim * num_heads, eps)

    def forward(
        self,
        x: torch.Tensor, # (B, L, D)
        timestamps: torch.Tensor, # (B, L)
        attn_mask: torch.Tensor # (L, L)
    ):
        B, L, _ = x.shape

        z = self.hidden_layer_norm(x) # (B, L, D)
        z = self.linear_act(self.uvqk_linear(z)) # (B, L, 2D1 + 2D2)
        u, v, q, k = torch.split(
            z,
            [
                self.linear_dim * self.num_heads, # (B, L, D1)
                self.linear_dim * self.num_heads, # (B, L, D1)
                self.attention_dim * self.num_heads, # (B, L, D2)
                self.attention_dim * self.num_heads # (B, L, D2)
            ],
            dim=-1
        )
        q, k = q.view(B, L, self.num_heads, self.attention_dim), k.view(B, L, self.num_heads, self.attention_dim)

        qk_attn = torch.einsum("BMHD,BNHD->BHMN", q, k) # (B, H, L, L)
        rel_attn_bias = 0. if self.rel_attn_bias_encoder is None \
            else self.rel_attn_bias_encoder(timestamps).unsqueeze(1) # (B, 1, L, L)
        qk_attn = qk_attn + rel_attn_bias
        qk_attn = F.silu(qk_attn) / L
        qk_attn = qk_attn * (1 - attn_mask)

        z = torch.einsum( # (B, L, D1)
            "BHMN,BNHD->BMHD",
            qk_attn,
            v.reshape(B, L, self.num_heads, self.linear_dim)
        ).reshape(B, L, -1)
        z = self.attn_layer_norm(z) * u
        z = self.output_linear(
            self.dropout(z)
        ) + x

        return z