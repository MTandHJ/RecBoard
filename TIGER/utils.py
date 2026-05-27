import torch
import torch.nn.functional as F


def straight_through_estimator(
    z: torch.Tensor,
    q: torch.Tensor,
) -> torch.Tensor:
    return z + (q - z).detach()


def gumbel_softmax_estimator(
    dist: torch.Tensor, codebook: torch.Tensor, temperature: float = 1.0
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
    z: torch.Tensor, q: torch.Tensor, eps: float = 1.0e-8
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
) -> torch.Tensor:

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
    return Q.to(_dtype)
