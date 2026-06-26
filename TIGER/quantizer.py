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

    @staticmethod
    def straight_through_estimator(z: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        return z + (q - z).detach()

    @staticmethod
    def gumbel_softmax_estimator(
        dist: torch.Tensor, codebook: torch.Tensor, temperature: float
    ) -> torch.Tensor:
        weights = F.gumbel_softmax(-dist, tau=temperature, hard=False)
        return weights @ codebook

    @staticmethod
    def rotation_trick_estimator(
        z: torch.Tensor, q: torch.Tensor, eps: float = 1.0e-8
    ) -> torch.Tensor:
        q = q.detach()
        z_norm = z.detach().norm(dim=-1, keepdim=True).clamp_min(eps)
        q_norm = q.norm(dim=-1, keepdim=True).clamp_min(eps)
        scale = q_norm / z_norm

        u = F.normalize(z, dim=-1).detach()
        v = F.normalize(q, dim=-1).detach()
        r = F.normalize(u + v, dim=-1).detach()

        z_on_r = torch.einsum("BD,BD->B", r, z).unsqueeze(-1).mul(r)
        z_on_u_to_v = torch.einsum("BD,BD->B", u, z).unsqueeze(-1).mul(v)
        return scale * (z - 2 * z_on_r + 2 * z_on_u_to_v)


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
        gumbel_temperature: float = 1.0,
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
        self.gumbel_temperature = gumbel_temperature

    def commit(self, x: torch.Tensor, y: torch.Tensor):
        return F.mse_loss(x, y.detach(), reduction="sum") / x.size(0)

    def match(self, x: torch.Tensor, l: int):
        codebook = self.codebooks[l].reinit_kmeans_codebook(x)
        dist = torch.cdist(x, codebook, p=2)  # (B, K)
        if self.sk_epsilons[l] > 0.0:
            dist = -sinkhorn_algorithm(dist, self.sk_epsilons[l], self.sk_iters)
        ids = torch.argmin(dist, dim=-1)  # (B,)
        c = codebook[ids]
        return ids, c, dist, codebook


class ResidualQuantizer(Quantizer):
    def get_indices(self, z: torch.Tensor) -> torch.Tensor:
        ids = []
        z_res = z
        for l in range(self.num_codebooks):
            ids_, c, _, _ = self.match(z_res, l)
            z_res = z_res - c
            ids.append(ids_)
        return torch.stack(ids, dim=-1)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor]:
        loss = 0

        ids = []
        z_res = z  # residual
        z_hat = 0.0  # estimation
        for l in range(self.num_codebooks):
            ids_, c, _, _ = self.match(z_res, l)
            q = CodeBook.straight_through_estimator(z_res, c)
            z_hat = z_hat + q
            loss += self.commit(c, z_res) + self.commit(z_res, c) * self.commit_weight
            z_res = z_res - q

            ids.append(ids_)

        return z_hat, loss / self.num_codebooks, torch.stack(ids, dim=-1)


class ResidualQuantizerGumbel(ResidualQuantizer):
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor]:
        loss = 0

        ids = []
        z_res = z
        z_hat = 0.0
        for l in range(self.num_codebooks):
            ids_, c, dist, codebook = self.match(z_res, l)
            q = CodeBook.gumbel_softmax_estimator(dist, codebook, self.gumbel_temperature)
            z_hat = z_hat + q
            loss += self.commit(c, z_res) + self.commit(z_res, c) * self.commit_weight
            z_res = z_res - q
            ids.append(ids_)

        return z_hat, loss / self.num_codebooks, torch.stack(ids, dim=-1)


class ResidualQuantizerRotation(ResidualQuantizer):
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor]:
        loss = 0

        ids = []
        z_res = z
        z_hat = 0.0
        for l in range(self.num_codebooks):
            ids_, c, _, _ = self.match(z_res, l)
            q = CodeBook.rotation_trick_estimator(z_res, c)
            z_hat = z_hat + q
            loss += self.commit(c, z_res) + self.commit(z_res, c) * self.commit_weight
            z_res = z_res - q
            ids.append(ids_)

        return z_hat, loss / self.num_codebooks, torch.stack(ids, dim=-1)


class ResidualSimVQQuantizer(Quantizer):
    r"""Residual SimVQ with frozen base codebooks and trainable projections."""

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
        gumbel_temperature: float = 1.0,
    ):
        super().__init__(
            dataset,
            codebook_dim,
            num_codebooks=num_codebooks,
            num_codewords=num_codewords,
            apply_shared_codebook=apply_shared_codebook,
            commit_weight=commit_weight,
            sk_iters=sk_iters,
            sk_epsilons=sk_epsilons,
            gumbel_temperature=gumbel_temperature,
        )

        for codebook in self.codebooks:
            nn.init.normal_(codebook.weight, mean=0.0, std=codebook_dim**-0.5)
            codebook.weight.requires_grad_(False)

        if apply_shared_codebook:
            projection = nn.Linear(codebook_dim, codebook_dim)
            self.projections = nn.ModuleList([projection] * self.num_codebooks)
        else:
            self.projections = nn.ModuleList(
                [nn.Linear(codebook_dim, codebook_dim) for _ in range(self.num_codebooks)]
            )

    def match(self, x: torch.Tensor, l: int):
        codebook = self.projections[l](self.codebooks[l].weight)
        dist = torch.cdist(x, codebook, p=2)  # (B, K)
        if self.sk_epsilons[l] > 0.0:
            dist = -sinkhorn_algorithm(dist, self.sk_epsilons[l], self.sk_iters)
        ids = torch.argmin(dist, dim=-1)
        c = codebook[ids]
        return ids, c, dist, codebook

    def get_indices(self, z: torch.Tensor) -> torch.Tensor:
        ids = []
        z_res = z
        for l in range(self.num_codebooks):
            ids_, c, _, _ = self.match(z_res, l)
            z_res = z_res - c
            ids.append(ids_)
        return torch.stack(ids, dim=-1)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor]:
        loss = 0

        ids = []
        z_res = z
        z_hat = 0.0
        for l in range(self.num_codebooks):
            ids_, c, _, _ = self.match(z_res, l)
            q = CodeBook.straight_through_estimator(z_res, c)
            z_hat = z_hat + q
            loss += self.commit(c, z_res) + self.commit(z_res, c) * self.commit_weight
            z_res = z_res - q
            ids.append(ids_)

        return z_hat, loss / self.num_codebooks, torch.stack(ids, dim=-1)


class ProductQuantizer(Quantizer):
    def get_indices(self, z: torch.Tensor) -> torch.Tensor:
        z = z.view(z.size(0), self.num_codebooks, self.codebook_dim)

        ids = []
        for l in range(self.num_codebooks):
            ids_, _, _, _ = self.match(z[:, l, :], l)
            ids.append(ids_)
        return torch.stack(ids, dim=-1)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor]:
        z = z.view(z.size(0), self.num_codebooks, self.codebook_dim)

        loss = 0
        ids, qs = [], []
        for l in range(self.num_codebooks):
            z_l = z[:, l, :]
            ids_, c, _, _ = self.match(z_l, l)
            loss += self.commit(c, z_l) + self.commit(z_l, c) * self.commit_weight
            ids.append(ids_)
            qs.append(CodeBook.straight_through_estimator(z_l, c))

        z_hat = torch.stack(qs, dim=1).view(z.size(0), -1)
        return z_hat, loss / self.num_codebooks, torch.stack(ids, dim=-1)
