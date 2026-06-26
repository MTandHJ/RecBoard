from typing import Optional, Tuple

import freerec
import torch
import torch.nn as nn
import torch.nn.functional as F
from freerec.data.tags import ID, ITEM
from utils import sinkhorn_algorithm


class CodeBook(nn.Embedding):
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        padding_idx=None,
        max_norm=None,
        norm_type=2,
        scale_grad_by_freq=False,
        sparse=False,
        _weight=None,
        _freeze=False,
        device=None,
        dtype=None,
    ):
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
    def reinit_kmeans_codebook(self, z: torch.Tensor) -> torch.Tensor:
        if self.requires_kmeans_init_:
            from k_means_constrained import KMeansConstrained

            z = z.detach().cpu().numpy()

            size_min = max(1, min(len(z) // (self.num_embeddings * 2), 50))

            clf = KMeansConstrained(
                n_clusters=self.num_embeddings,
                size_min=size_min,
                max_iter=10,
                n_init=10,
                n_jobs=10,
                verbose=False,
            )
            clf.fit(z)
            codebook = torch.from_numpy(clf.cluster_centers_)

            codebook = codebook.to(device=self.weight.device, dtype=self.weight.dtype)
            self.weight.data.copy_(codebook)

            self.requires_kmeans_init_ = False
        return self.weight


class Quantizer(nn.Module):
    def __init__(
        self,
        dataset: freerec.data.datasets.base.RecDataSet,
        codebook_dim: int,
        num_codebooks: int = 3,
        num_codewords: int = 256,
        apply_shared_codebook: bool = False,
        commit_weight: float = 0.25,
        sk_iters: int = 50,
        sk_epsilons: Optional[Tuple[float]] = None,
    ):
        super().__init__()

        self.dataset = dataset
        self.Item = self.dataset.fields[ITEM, ID]
        self.codebook_dim = codebook_dim
        self.num_codebooks = num_codebooks
        self.num_codewords = num_codewords

        if apply_shared_codebook:
            self.codebooks = nn.ModuleList(
                [
                    CodeBook(
                        num_codewords,
                        codebook_dim,
                    )
                ]
                * self.num_codebooks
            )
        else:
            self.codebooks = nn.ModuleList(
                [
                    CodeBook(
                        num_codewords,
                        codebook_dim,
                    )
                    for _ in range(self.num_codebooks)
                ]
            )

        self.sk_iters = sk_iters
        self.sk_epsilons = sk_epsilons
        self.commit_weight = commit_weight

    def commit(self, x: torch.Tensor, y: torch.Tensor):
        return F.mse_loss(x, y.detach(), reduction="sum") / x.size(0)

    def match(self, x: torch.Tensor, l: int):
        codebook = self.codebooks[l].reinit_kmeans_codebook(x)
        dist = torch.cdist(x, codebook, p=2)  # (B, K)
        if self.sk_epsilons[l] > 0.0:
            dist = -sinkhorn_algorithm(dist, self.sk_epsilons[l], self.sk_iters)
        ids = torch.argmin(dist, dim=-1)  # (B,)
        c = codebook[ids]
        return ids, c


class ResidualQuantizer(Quantizer):
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor]:
        loss = 0

        ids = []
        z_res = z  # residual
        z_hat = 0.0  # estimation
        for l in range(self.num_codebooks):
            ids_, c = self.match(z_res, l)
            z_hat = z_hat + c
            loss += self.commit(c, z_res) + self.commit(z_res, c) * self.commit_weight
            z_res = z_res - (z_res + (c - z_res).detach())

            ids.append(ids_)

        return (
            z + (z_hat - z).detach(),
            loss / self.num_codebooks,
            torch.stack(ids, dim=-1),
        )


class ProductQuantizer(Quantizer):
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor]:
        z = z.view(z.size(0), self.num_codebooks, self.codebook_dim)

        loss = 0
        ids, cs = [], []
        for l in range(self.num_codebooks):
            ids_, c = self.match(z[:, l, :], l)
            z_l = z[:, l, :]
            loss += self.commit(c, z_l) + self.commit(z_l, c) * self.commit_weight
            ids.append(ids_)
            cs.append(c)

        z_hat = torch.stack(cs, dim=1).view(z.size(0), -1)
        z = z.view(z.size(0), -1)
        return (
            z + (z_hat - z).detach(),
            loss / self.num_codebooks,
            torch.stack(ids, dim=-1),
        )
