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

            self.weight.data.copy_(
                codebook.to(device=self.weight.device, dtype=self.weight.dtype)
            )

            self.requires_kmeans_init_ = False
        return self.weight


class ResidualQuantizer(nn.Module):
    def __init__(
        self,
        dataset: freerec.data.datasets.base.RecDataSet,
        hidden_size: int,
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

        if apply_shared_codebook:
            self.codebooks = nn.ModuleList(
                [
                    CodeBook(
                        num_codewords,
                        hidden_size,
                    )
                ]
                * num_codebooks
            )
        else:
            self.codebooks = nn.ModuleList(
                [
                    CodeBook(
                        num_codewords,
                        hidden_size,
                    )
                    for _ in range(num_codebooks)
                ]
            )

        self.sk_iters = sk_iters
        self.sk_epsilons = sk_epsilons
        self.commit_weight = commit_weight

    def commit(self, x: torch.Tensor, y: torch.Tensor):
        return F.mse_loss(x, y.detach(), reduction="sum") / x.size(0)

    def step(self, r: torch.Tensor, l: int):
        codebook = self.codebooks[l].reinit_kmeans_codebook(r)
        dist = torch.cdist(r, codebook, p=2)  # (B, K)
        if self.sk_epsilons[l] > 0.0:
            dist = -sinkhorn_algorithm(dist, self.sk_epsilons[l], self.sk_iters)
        ids = torch.argmin(dist, dim=-1)  # (B,)
        q = codebook[ids]
        return ids, q

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor]:
        loss = 0

        ids = []
        z_res = z  # residual
        z_hat = 0.0  # estimation
        L = len(self.codebooks)
        for l in range(L):
            ids_, q = self.step(z_res, l)
            z_hat = z_hat + q
            loss += self.commit(q, z_res) + self.commit(z_res, q) * self.commit_weight
            z_res = z_res - q

            ids.append(ids_)

        return z + (z_hat - z).detach(), loss / L, torch.stack(ids, dim=-1)
