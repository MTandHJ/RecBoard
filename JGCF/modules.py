
from typing import List

import torch
import torch.nn as nn
from functools import partial


def jacobi_conv(
    zs: List[torch.Tensor], A: torch.Tensor, l: int, 
    alpha: float = 1., beta: float = 1.
):
    r"""
    Polynomial convolution with [Jacobi bases](https://en.wikipedia.org/wiki/Jacobi_polynomials#Recurrence_relations).

    Parameters:
    -----------
    zs: List[torch.Tensor]
        .. math:: [z_0, z_1, ... z_{l-1}]
    A: Adj, normalized adjacency matrix 
    (alpha, beta): float, two hyper-parameters for Jacobi Polynomial.
    """
    if l == 0:
        return zs[0]

    assert len(zs) == l, "len(zs) != l for l != 0"

    if l == 1:
        c = (alpha - beta) / 2
        return c * zs[-1] + (alpha + beta + 2) / 2 * (A @ zs[-1])
    else:
        c0 = 2 * l \
                * (l + alpha + beta) \
                * (2 * l + alpha + beta - 2)
        c1 = (2 * l + alpha + beta - 1) \
                * (alpha ** 2 - beta ** 2)
        c2 = (2 * l + alpha + beta - 1) \
                * (2 * l + alpha + beta) \
                * (2 * l + alpha + beta - 2)
        c3 = 2 * (l + alpha - 1) \
                * (l + beta - 1) \
                * (2 * l + alpha + beta)
        
        part1 = c1 * zs[-1]
        part2 = c2 * (A @ zs[-1])
        part3 = c3 * zs[-2]

        return (part1 + part2 - part3) / c0


class JacobiConv(nn.Module):

    def __init__(
        self, 
        scaling_factor: float = 3.,
        L: int = 3,
        alpha: float = 1.,
        beta: float = 1.,
    ):
        super().__init__()

        self.L = L
        self.scaling_factor = scaling_factor

        self.register_parameter(
            'gammas',
            nn.parameter.Parameter(
                torch.empty((L + 1, 1)).fill_(min(1 / scaling_factor, 1.)),
                requires_grad=False
            )
        )

        self.conv_fn = partial(jacobi_conv, alpha=alpha, beta=beta)

    def forward(self, x: torch.Tensor, A: torch.Tensor):
        zs = [self.conv_fn([x], A, 0)]
        for l in range(1, self.L + 1):
            z = self.conv_fn(zs, A, l)
            zs.append(z)
        coefs = (self.gammas.tanh() * self.scaling_factor).cumprod(dim=0) 
        zs = torch.stack(zs, dim=1) # (N, L + 1, D)
        return (zs * coefs).mean(1) # (N, D)