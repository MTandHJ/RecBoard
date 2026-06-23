import random
from typing import List, Tuple

import freerec
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import sinkhorn_algorithm

__all__ = ["CodeBook", "ResidualQuantizer"]


class CodeBook(nn.Embedding):
    r"""LETTER codebook with k-means initialization and diversity clusters.

    The parameter layout follows TIGER's ``CodeBook``. LETTER additionally
    keeps per-codeword cluster labels for the diversity loss used during RQ-VAE
    training.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings, embedding_dim)

        self.requires_kmeans_init_ = False
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(
            self.weight,
            -1.0 / self.num_embeddings,
            1.0 / self.num_embeddings,
        )

    @torch.no_grad()
    def reinit_kmeans_codebook(self, z: torch.Tensor) -> torch.Tensor:
        if self.requires_kmeans_init_:
            from k_means_constrained import KMeansConstrained

            z = z.detach().cpu().numpy()
            size_min = max(1, min(len(z) // (self.num_embeddings * 2), 50))

            clf = KMeansConstrained(
                n_clusters=self.num_embeddings,
                size_min=size_min,
                size_max=size_min * 4,
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

    def reset_diversity_clusters(self, num_clusters: int = 10) -> None:
        r"""Cluster codewords for LETTER's diversity loss.

        Parameters
        ----------
        num_clusters : int, default=10
            Number of constrained k-means clusters over codewords.
        """
        from k_means_constrained import KMeansConstrained

        z = self.weight.detach().cpu().numpy()
        size_min = max(1, min(len(z) // (num_clusters * 2), 10))

        clf = KMeansConstrained(
            n_clusters=num_clusters,
            size_min=size_min,
            size_max=num_clusters * 6,
            max_iter=10,
            n_init=10,
            n_jobs=10,
            verbose=False,
        )
        clf.fit(z)

        self.register_buffer(
            "_diversity_labels",
            torch.from_numpy(clf.labels_).to(self.weight.device),
        )
        self._diversity_clusters = [
            set(torch.where(self._diversity_labels == label)[0].tolist())
            for label in range(num_clusters)
        ]

    def match(
        self,
        z: torch.Tensor,
        sk_epsilon: float,
        sk_iters: int,
        apply_sinkhorn_distance: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        codebook = self.reinit_kmeans_codebook(z)
        dist = torch.cdist(z, codebook, p=2)
        if apply_sinkhorn_distance and sk_epsilon > 0:
            dist = -sinkhorn_algorithm(dist, sk_epsilon, sk_iters)
        ids = torch.argmin(dist, dim=-1)
        return ids, codebook[ids]

    def _sample_positive(self, id_: int, label: int) -> int:
        candidates = self._diversity_clusters[label] - {id_}
        return random.choice(list(candidates))

    def diversity_loss(self, q: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        labels = torch.gather(
            self._diversity_labels,
            dim=0,
            index=ids,
        ).tolist()
        positives = [
            self._sample_positive(id_, label)
            for id_, label in zip(ids, labels)
        ]
        positives = torch.tensor(positives, dtype=torch.long, device=q.device)

        ids = ids.view(-1, 1)
        logits = torch.einsum("B D, N D -> B N", q, self.weight)
        logits = torch.scatter(
            logits,
            dim=-1,
            index=ids,
            src=torch.ones_like(ids).float().fill_(-1.0e4),
        )
        return freerec.criterions.cross_entropy_with_logits(
            logits,
            positives,
            reduction="mean",
        )


class ResidualQuantizer(nn.Module):
    r"""Residual quantizer used by LETTER RQ-VAE.

    Parameters
    ----------
    hidden_size : int
        Latent vector size.
    num_codebooks : int, default=4
        Number of residual codebooks.
    num_codewords : int, default=256
        Number of codewords in each codebook.
    commit_weight : float, default=0.25
        Weight for the encoder commitment term.
    diversity_weight : float, default=0.0001
        Weight for LETTER's codebook diversity loss.
    apply_kmeans_init : bool, default=True
        Whether to initialize codebooks with constrained k-means on first use.
    sk_iters : int, default=50
        Sinkhorn iterations.
    sk_epsilons : List[float] | None, default=None
        Per-codebook Sinkhorn epsilon values.
    """

    def __init__(
        self,
        hidden_size: int,
        num_codebooks: int = 4,
        num_codewords: int = 256,
        commit_weight: float = 0.25,
        diversity_weight: float = 0.0001,
        apply_kmeans_init: bool = True,
        sk_iters: int = 50,
        sk_epsilons: List[float] | None = None,
    ) -> None:
        super().__init__()

        self.codebooks = nn.ModuleList(
            [CodeBook(num_codewords, hidden_size) for _ in range(num_codebooks)]
        )
        self.commit_weight = commit_weight
        self.diversity_weight = diversity_weight
        self.sk_iters = sk_iters
        self.sk_epsilons = sk_epsilons or [0.0] * num_codebooks

        if apply_kmeans_init:
            for codebook in self.codebooks:
                codebook.requires_kmeans_init_ = True

        self.reset_diversity_clusters()

    def reset_diversity_clusters(self) -> None:
        for codebook in self.codebooks:
            codebook.reset_diversity_clusters()

    def commit(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(x, y.detach(), reduction="mean")

    def match(
        self,
        z: torch.Tensor,
        level: int,
        apply_sinkhorn_distance: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.codebooks[level].match(
            z,
            sk_epsilon=self.sk_epsilons[level],
            sk_iters=self.sk_iters,
            apply_sinkhorn_distance=apply_sinkhorn_distance,
        )

    def forward(
        self,
        z: torch.Tensor,
        apply_sinkhorn_distance: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Quantize latent vectors with residual LETTER codebooks.

        Parameters
        ----------
        z : torch.Tensor
            Latent vectors with shape ``(B, D)``.
        apply_sinkhorn_distance : bool, default=True
            Whether to apply Sinkhorn assignment before nearest-codeword lookup.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Quantized vectors, auxiliary quantization loss, and semantic IDs.
        """
        loss = z.new_zeros(())
        ids = []
        qs = []
        z_res = z

        for level, codebook in enumerate(self.codebooks):
            ids_, c = self.match(
                z_res,
                level,
                apply_sinkhorn_distance=apply_sinkhorn_distance,
            )
            q = z_res + (c - z_res).detach()
            loss = (
                loss
                + self.commit(c, z_res)
                + self.commit(z_res, c) * self.commit_weight
                + codebook.diversity_loss(c, ids_) * self.diversity_weight
            )
            z_res = z_res - q

            qs.append(q)
            ids.append(ids_)

        return torch.stack(qs, dim=-1).sum(dim=-1), loss, torch.stack(ids, dim=-1)
