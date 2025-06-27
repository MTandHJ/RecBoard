import torch
import torch.nn.functional as F


def straight_through_estimator(
    z: torch.Tensor,
    q: torch.Tensor,
) -> torch.Tensor:
    return z + (q - z).detach()

def gumbel_softmax_estimator(
    dist: torch.Tensor,
    codebook: torch.Tensor,
    temperature: float = 1.
) -> torch.Tensor:
    r"""
    Gumbel Softmax Sampling.

    Parameters:
    -----------
    dist: torch.Tensor, (B, K)
        distance matrix between inputs and codebook
    codebook: torch.Tensor, (K, D)
    temperature: float
        tau for gumbel softmax sampling
    
    Returns:
    --------
    q: torch.Tensor, (B, D)
    """
    logits = -dist
    weights = F.gumbel_softmax(logits, tau=temperature, hard=False)
    return weights @ codebook

def rotation_trick_estimator(
    z: torch.Tensor,
    q: torch.Tensor,
    eps: float = 1.e-8
) -> torch.Tensor:
    r"""
    Gradient estimator via rotation trick.
    See: https://arxiv.org/pdf/2410.06424

    .. math::

        \text{scale} \cdot [z - 2 rr^T e + 2 vu^T e]

    Parameters:
    -----------
    z: torch.Tensor, (B, D)
        inputs
    q: torch.Tensor, (B, D)
        codes
    eps: float
    """

    scale = q.norm().div(z.norm() + eps).detach()
    u = F.normalize(z, dim=-1).detach()
    v = F.normalize(q, dim=-1).detach()
    r = F.normalize(z + q, dim=-1).detach()

    return scale * (
        z
        - 2 * torch.einsum("BD,BD->B", r, z).unsqueeze(-1).mul(r)
        + 2 * torch.einsum("BD,BD->B", u, z).unsqueeze(-1).mul(v)
    )