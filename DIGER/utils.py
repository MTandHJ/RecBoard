import math
from typing import Tuple

import torch
import torch.nn as nn


def center_distance_for_constraint(distances):
    # distances: B, K
    max_distance = distances.max()
    min_distance = distances.min()

    middle = (max_distance + min_distance) / 2
    amplitude = max_distance - middle + 1e-5
    assert amplitude > 0
    centered_distances = (distances - middle) / amplitude
    return centered_distances


@torch.no_grad()
def sinkhorn_algorithm(
    distances: torch.Tensor, epsilon: float, sinkhorn_iterations: int
) -> Tuple[torch.Tensor, torch.Tensor]:

    _dtype = distances.dtype
    distances = center_distance_for_constraint(distances).double()

    Q = torch.exp(-distances / epsilon)

    B = Q.shape[0]  # number of samples to assign
    K = Q.shape[1]  # how many centroids per block (usually set to 256)

    # make the matrix sums to 1
    sum_Q = Q.sum(-1, keepdim=True).sum(-2, keepdim=True)
    Q /= sum_Q
    for it in range(sinkhorn_iterations):
        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= B

        # normalize each row: total weight per prototype must be 1/K
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= K

    Q *= B  # the colomns must sum to 1 so that Q is an assignment
    return Q.to(_dtype), distances.float()


class AutoSigmaGaussian(nn.Module):
    r"""Gaussian-noise relaxation with a learnable uncertainty scale."""

    def __init__(self, sigma_scale_init_std: float = 1.0) -> None:
        super().__init__()

        self.sigma_scale_init_std = sigma_scale_init_std
        self.reset_sigma()

    def reset_sigma(self):
        self.sigma = nn.Parameter(
            torch.tensor(
                -20.0
                if self.sigma_scale_init_std <= 1.0e-5
                else math.log2(self.sigma_scale_init_std)
            )
        )

    def forward(
        self,
        logits: torch.Tensor,
        tau: float = 1.0,
        hard: bool = False,
        dim: int = -1,
    ) -> torch.Tensor:
        scale = torch.pow(2.0, self.sigma).clamp(min=1.0e-5, max=100.0)
        noisy_logits = logits + torch.randn_like(logits) * scale if self.training else logits
        probs = (noisy_logits / tau).softmax(dim)
        if not hard:
            return probs
        index = probs.max(dim, keepdim=True)[1]
        hard_probs = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        return hard_probs - probs.detach() + probs


class AutoSigmaGumbel(AutoSigmaGaussian):
    r"""Gumbel relaxation with a learnable uncertainty scale."""

    def forward(
        self,
        logits: torch.Tensor,
        tau: float = 1.0,
        hard: bool = False,
        dim: int = -1,
    ) -> torch.Tensor:
        if self.training:
            scale = torch.pow(2.0, self.sigma).clamp(min=1.0e-5, max=100.0)
            noise = -torch.empty_like(logits).exponential_().log() * scale
            probs = ((logits + noise) / tau).softmax(dim)
        else:
            probs = (logits / tau).softmax(dim)
        if not hard:
            return probs, self.sigma

        index = probs.max(dim, keepdim=True)[1]
        hard_probs = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        return hard_probs - probs.detach() + probs


class AutoSigmaSimple(AutoSigmaGaussian):
    r"""Simple uncertainty relaxation used by the SDUD variant."""

    def reset_sigma(self):
        self.sigma = nn.Parameter(torch.tensor(self.sigma_scale_init_std))

    def forward(
        self,
        logits: torch.Tensor,
        tau: float = 1.0,
        hard: bool = False,
        dim: int = -1,
    ) -> torch.Tensor:
        scale = self.sigma.abs().clamp(min=1.0e-5, max=100.0)
        if self.training:
            noise = -torch.empty_like(logits).exponential_().log() * scale
            probs = ((logits + noise) / tau).softmax(dim)
        else:
            probs = (logits / tau).softmax(dim)
        if not hard:
            return probs

        index = probs.max(dim, keepdim=True)[1]
        hard_probs = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        return hard_probs - probs.detach() + probs
