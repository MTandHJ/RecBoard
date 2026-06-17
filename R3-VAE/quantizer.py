from typing import Tuple

import freerec
import torch
import torch.nn as nn
import torch.nn.functional as F
from freerec.data.tags import ID, ITEM


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


class RatingResidualQuantizer(nn.Module):
    r"""Reference-vector-guided residual rating quantizer from R3-VAE.

    Parameters
    ----------
    dataset : freerec.data.datasets.base.RecDataSet
        RecBoard dataset, used only for local field consistency with TIGER's quantizer.
    hidden_size : int
        Dimension of vectors to quantize.
    num_codebooks : int, default=3
        Number of residual rating codebooks.
    num_codewords : int, default=256
        Number of codewords for each codebook.
    pd_temperature : float, default=2.0
        Temperature for preference discrimination regularization.

    Workflow
    --------
    1. normalize the encoded item vector.
    2. project it onto a learnable reference vector and keep the residual.
    3. optionally initialize each codebook from the current residual with KMeans.
    4. repeatedly select the top-1 cosine codeword and add its rating-weighted vector.
    5. return quantized vectors, semantic IDs, SC loss, and PD loss.
    """

    def __init__(
        self,
        dataset: freerec.data.datasets.base.RecDataSet,
        hidden_size: int,
        num_codebooks: int = 3,
        num_codewords: int = 256,
        pd_temperature: float = 2.0,
    ) -> None:
        super().__init__()

        self.dataset = dataset
        self.Item = self.dataset.fields[ITEM, ID]

        self.hidden_size = hidden_size
        self.num_codebooks = num_codebooks
        self.num_codewords = num_codewords
        self.pd_temperature = pd_temperature

        self.reference_vector = nn.Parameter(torch.empty(1, hidden_size))
        self.codebooks = nn.ModuleList(
            [CodeBook(num_codewords, hidden_size) for _ in range(num_codebooks)]
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.reference_vector, -1.0, 1.0)
        for l, codebook in enumerate(self.codebooks):
            nn.init.uniform_(codebook.weight, -1.0 / (l + 1), 1.0 / (l + 1))

    def reference_step(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = F.normalize(z, p=2, dim=-1)
        reference = F.normalize(self.reference_vector, p=2, dim=-1)
        values = torch.sum(z * reference, dim=-1, keepdim=True)
        reference_out = values * reference
        return z - reference_out, reference_out

    def step(self, r: torch.Tensor, l: int) -> Tuple[torch.Tensor, torch.Tensor]:
        codebook = F.normalize(self.codebooks[l].reinit_kmeans_codebook(r), dim=-1)
        r = F.normalize(r, dim=-1)
        weights, indices = torch.topk(r @ codebook.t(), k=1, dim=-1)
        q = torch.einsum("BKD,BK -> BD", codebook[indices], weights)
        return indices[:, 0], q

    def forward(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z_res, z_hat = self.reference_step(z)
        target = z_res

        ids = []
        sc_loss = 0.0
        pd_loss = 0.0
        L = len(self.codebooks)

        for l in range(L):
            ids_, q = self.step(z_res, l)
            z_hat = z_hat + q
            z_res = z_res - q
            ids.append(ids_)

            sc_loss = sc_loss + (1.0 - F.cosine_similarity(target, z_hat, dim=-1).mean())
            pd_loss = pd_loss + preference_discrimination_loss(
                self.codebooks[l].weight,
                temperature=self.pd_temperature,
            )

        return z_hat, sc_loss / L, pd_loss / L, torch.stack(ids, dim=-1)


def preference_discrimination_loss(x: torch.Tensor, temperature: float = 2.0) -> torch.Tensor:
    r"""Encourage codewords in one codebook to spread over the unit sphere."""
    x = F.normalize(x, dim=-1)
    sim_matrix = x @ x.transpose(-2, -1)
    cos_dist_matrix = 1.0 - sim_matrix
    mask = ~torch.eye(sim_matrix.size(0), dtype=torch.bool, device=x.device)
    return torch.exp(-temperature * cos_dist_matrix[mask]).mean().log()
