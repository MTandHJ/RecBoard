from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

__all__ = [
    "CodeBook",
    "QuantizerOutput",
    "ResidualQuantizer",
    "VectorQuantizer",
]


@dataclass
class QuantizerOutput:
    quantized: torch.Tensor
    loss: torch.Tensor
    indices: torch.Tensor
    one_hot: torch.Tensor
    logits: torch.Tensor


class CodeBook(nn.Embedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        _weight: Optional[torch.Tensor] = None,
        _freeze: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            num_embeddings,
            embedding_dim,
            padding_idx,
            max_norm,
            norm_type,
            scale_grad_by_freq,
            sparse,
            _weight,
            _freeze,
            device,
            dtype,
        )
        self.requires_kmeans_init_ = False

    @torch.no_grad()
    def reinit_kmeans_codebook(self, z: torch.Tensor, max_iters: int) -> torch.Tensor:
        if self.requires_kmeans_init_:
            centers = kmeans(z, self.num_embeddings, max_iters)
            self.weight.data.copy_(centers.to(device=self.weight.device, dtype=self.weight.dtype))
            self.requires_kmeans_init_ = False
        return self.weight


def kmeans(samples: torch.Tensor, num_clusters: int, num_iters: int = 10) -> torch.Tensor:
    x = samples.detach().cpu().numpy()
    cluster = KMeans(n_clusters=num_clusters, max_iter=num_iters).fit(x)
    return torch.from_numpy(cluster.cluster_centers_).to(samples.device)


class VectorQuantizer(nn.Module):
    r"""One ETEGRec codebook with source-compatible logits.

    ETEGRec needs the selected code ids for item tokens and the per-codeword
    distance logits for KL alignment between sequence latents and target item
    latents. The module therefore returns both discrete ids and raw logits.
    """

    def __init__(self, config: dict, num_codewords: int) -> None:
        super().__init__()
        self.num_codewords = num_codewords
        self.codebook_dim = config["codebook_dim"]
        self.commit_weight = config["commit_weight"]
        self.kmeans_init = config["kmeans_init"]
        self.kmeans_iters = config["kmeans_iters"]

        self.embedding = CodeBook(num_codewords, self.codebook_dim)
        self.initted = not self.kmeans_init
        self.embedding.weight.data.uniform_(-1.0 / num_codewords, 1.0 / num_codewords)
        self.embedding.requires_kmeans_init_ = self.kmeans_init

    def get_codebook(self) -> torch.Tensor:
        return self.embedding.weight

    def get_codebook_entry(
        self,
        indices: torch.Tensor,
        shape: Optional[Tuple] = None,
    ) -> torch.Tensor:
        quantized = self.embedding(indices)
        if shape is not None:
            quantized = quantized.view(shape)
        return quantized

    def pairwise_distance(self, z: torch.Tensor) -> torch.Tensor:
        return (
            torch.sum(z**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1, keepdim=True).t()
            - 2 * torch.matmul(z, self.embedding.weight.t())
        )

    @torch.no_grad()
    def get_maxk_indices(self, z: torch.Tensor, maxk: int = 1) -> torch.Tensor:
        latents = z.view(-1, self.codebook_dim)
        distances = (
            torch.sum(latents**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1, keepdim=True).t()
            - 2 * torch.matmul(latents, self.embedding.weight.t())
        )
        return torch.topk(distances, k=maxk, largest=False, dim=-1).indices.view(
            *z.shape[:-1],
            maxk,
        )

    def quantization_loss(
        self,
        quantized: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        codebook_loss = F.mse_loss(quantized, z.detach())
        commitment_loss = F.mse_loss(quantized.detach(), z)
        return codebook_loss + self.commit_weight * commitment_loss

    def forward(self, z: torch.Tensor) -> QuantizerOutput:
        latents = z.view(-1, self.codebook_dim)
        if not self.initted and self.training:
            self.embedding.reinit_kmeans_codebook(latents, self.kmeans_iters)
            self.initted = True

        logits = self.pairwise_distance(latents)
        indices = torch.argmin(logits, dim=-1)
        one_hot = F.one_hot(indices, self.num_codewords).float()
        quantized = self.embedding(indices).view(z.shape)
        loss = self.quantization_loss(quantized, z)
        quantized = z + (quantized - z).detach()

        return QuantizerOutput(
            quantized=quantized,
            loss=loss,
            indices=indices.view(z.shape[:-1]),
            one_hot=one_hot,
            logits=logits,
        )


class ResidualQuantizer(nn.Module):
    r"""ETEGRec residual quantizer with source-compatible auxiliary outputs."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.num_codebooks = config["num_codebooks"]
        self.num_codewords = config["num_codewords"]
        self.vq_layers = nn.ModuleList(
            [
                VectorQuantizer(config=config, num_codewords=self.num_codewords)
                for _ in range(self.num_codebooks)
            ]
        )

    def get_codebook(self) -> torch.Tensor:
        codebooks = [quantizer.get_codebook().detach().cpu() for quantizer in self.vq_layers]
        return torch.stack(codebooks)

    @torch.no_grad()
    def get_indices(self, z: torch.Tensor) -> torch.Tensor:
        all_indices = []
        residual = z
        for quantizer in self.vq_layers:
            output = quantizer(residual)
            residual = residual - output.quantized
            all_indices.append(output.indices)
        return torch.stack(all_indices, dim=-1)

    @torch.no_grad()
    def get_maxk_indices(self, z: torch.Tensor, maxk: int = 1, used: bool = False) -> torch.Tensor:
        all_indices = []
        residual = z
        for quantizer in self.vq_layers:
            topk_indices = quantizer.get_maxk_indices(residual, maxk=maxk)
            quantized = quantizer.get_codebook_entry(topk_indices[..., 0], shape=residual.shape)
            residual = residual - quantized
            all_indices.append(topk_indices)
        return torch.stack(all_indices, dim=1)

    def forward(self, z: torch.Tensor) -> QuantizerOutput:
        all_losses = []
        all_indices = []
        all_one_hots = []
        all_logits = []

        quantized = torch.zeros_like(z)
        residual = z
        for quantizer in self.vq_layers:
            output = quantizer(residual)
            residual = residual - output.quantized
            quantized = quantized + output.quantized
            all_losses.append(output.loss)
            all_indices.append(output.indices)
            all_one_hots.append(output.one_hot)
            all_logits.append(output.logits)

        return QuantizerOutput(
            quantized=quantized,
            loss=torch.stack(all_losses).mean(),
            indices=torch.stack(all_indices, dim=-1),
            one_hot=torch.stack(all_one_hots, dim=1),
            logits=torch.stack(all_logits, dim=1),
        )
