from typing import Optional, Tuple

import freerec
import torch
import torch.nn as nn
import torch.nn.functional as F
from freerec.data.tags import ID, ITEM
from utils import AutoSigmaSimple, sinkhorn_algorithm


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

            self.weight.data.copy_(codebook.to(device=self.weight.device, dtype=self.weight.dtype))

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

    def match(self, r: torch.Tensor, l: int):
        codebook = self.codebooks[l].reinit_kmeans_codebook(r)
        dist = torch.cdist(r, codebook, p=2)  # (B, K)
        if self.sk_epsilons[l] > 0.0:
            assignments, centered_distances = sinkhorn_algorithm(
                dist, self.sk_epsilons[l], self.sk_iters
            )
            ids = torch.argmax(assignments, dim=-1)  # (B,)
        else:
            ids = torch.argmin(dist, dim=-1)  # (B,)
        q = codebook[ids]
        return ids, q

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        loss = 0

        ids = []
        z_res = z  # residual
        z_hat = 0.0  # estimation
        L = len(self.codebooks)
        for l in range(L):
            ids_, c = self.match(z_res, l)
            z_hat = z_hat + c
            loss += self.commit(c, z_res) + self.commit(z_res, c) * self.commit_weight
            z_res = z_res - (z_res + (c - z_res).detach())

            ids.append(ids_)

        return z + (z_hat - z).detach(), loss / L, torch.stack(ids, dim=-1)


class ResidualQuantizerGumbel(ResidualQuantizer):
    def __init__(
        self,
        dataset,
        hidden_size,
        num_codebooks=3,
        num_codewords=256,
        apply_shared_codebook=False,
        commit_weight=0.25,
        sk_iters=50,
        sk_epsilons=None,
        sigma_scale_init_std: float = 1.0,
        hot_threshold_ratio: float = 1.5,
    ):

        super().__init__(
            dataset,
            hidden_size,
            num_codebooks,
            num_codewords,
            apply_shared_codebook,
            commit_weight,
            sk_iters,
            sk_epsilons,
        )

        self.hot_threshold_ratio = hot_threshold_ratio
        self.register_buffer("code_usage_ema", torch.ones(num_codewords) / num_codewords)
        self.autosigmas = nn.ModuleList(
            [AutoSigmaSimple(sigma_scale_init_std) for _ in range(len(self.codebooks))]
        )

    @torch.no_grad()
    def get_hot_code_mask(self, z: torch.Tensor) -> Optional[torch.Tensor]:
        codebook = self.codebooks[0].reinit_kmeans_codebook(z)
        dist = torch.cdist(z, codebook, p=2)  # (B, K)
        deterministic_ids = dist.argmin(dim=-1)
        hot_threshold = self.hot_threshold_ratio / self.num_codewords
        return self.code_usage_ema[deterministic_ids] > hot_threshold

    def sample_indices(
        self,
        level: int,
        logits: torch.Tensor,
        deterministic_ids: torch.Tensor,
        gumbel_temperature: float,
        sample_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        probs = self.auto_sigmas[level](logits, tau=gumbel_temperature, hard=False, dim=-1)
        sampled_ids = probs.argmax(dim=-1)

        if level == 0:
            with torch.no_grad():
                counts = torch.bincount(deterministic_ids, minlength=self.num_codewords).float()
                freqs = counts / counts.sum().clamp_min(1)
                self.code_usage_ema.mul_(self.usage_momentum).add_(
                    freqs,
                    alpha=1.0 - self.usage_momentum,
                )
        use_sampled = sample_mask.view_as(deterministic_ids)
        deterministic_probs = (logits / gumbel_temperature).softmax(dim=-1)
        selected_ids = torch.where(use_sampled, sampled_ids, deterministic_ids)
        probs = torch.where(use_sampled[:, None], probs, deterministic_probs)
        return selected_ids, probs

    @torch.no_grad()
    def get_indices(self, z: torch.Tensor) -> torch.Tensor:
        is_training = self.training
        self.eval()
        _, _, ids = self.forward(z)
        self.train(is_training)
        return ids

    def match(self, r: torch.Tensor, l: int, gumbel_temperature: float, hot_code_mask: torch.Tensor):
        codebook = self.codebooks[l].reinit_kmeans_codebook(r)
        dist = torch.cdist(r, codebook, p=2)  # (B, K)
        if self.sk_epsilons[l] > 0.0:
            assignments, centered_distances = sinkhorn_algorithm(
                dist, self.sk_epsilons[l], self.sk_iters
            )
            dist = centered_distances
            deterministic_ids = torch.argmax(assignments, dim=-1)  # (B,)
        else:
            deterministic_ids = torch.argmin(dist, dim=-1)  # (B,)

        ids, probs = self.sample_indices(
            l,
            -dist,
            deterministic_ids,
            gumbel_temperature=gumbel_temperature,
            sample_mask=hot_code_mask,
        )
        hard_probs = F.one_hot(ids, self.num_codewords).float()
        # indicator STE
        q = (hard_probs - probs.detach() + probs) @ codebook
        return ids, q

    def forward(self, z: torch.Tensor, gumbel_temperature: float = 1.0) -> Tuple[torch.Tensor, ...]:

        if self.training:
            loss = 0
            ids = []
            z_res = z  # residual
            z_hat = 0.0  # estimation
            L = len(self.codebooks)

            hot_code_mask = self.get_hot_code_mask(z)

            for l in range(L):
                ids_, q = self.match(z_res, l, gumbel_temperature, hot_code_mask)
                z_hat = z_hat + q
                loss += self.commit(q, z_res) + self.commit(z_res, q) * self.commit_weight
                z_res = z_res - q

                ids.append(ids_)

            return z_hat, loss / L, torch.stack(ids, dim=-1)
        else:
            loss = 0

            ids = []
            z_res = z  # residual
            z_hat = 0.0  # estimation
            L = len(self.codebooks)
            for l in range(L):
                ids_, c = super().match(z_res, l)
                z_hat = z_hat + c
                loss += self.commit(c, z_res) + self.commit(z_res, c) * self.commit_weight
                z_res = z_res - c

                ids.append(ids_)

            return z + (z_hat - z).detach(), loss / L, torch.stack(ids, dim=-1)
