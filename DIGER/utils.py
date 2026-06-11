import torch


def center_distance_for_constraint(distances: torch.Tensor) -> torch.Tensor:
    r"""Center distances before Sinkhorn constrained assignment."""
    max_distance = distances.max()
    min_distance = distances.min()
    middle = (max_distance + min_distance) / 2
    amplitude = max_distance - middle + 1.0e-5
    assert amplitude > 0
    return (distances - middle) / amplitude


@torch.no_grad()
def sinkhorn_algorithm(
    distances: torch.Tensor,
    epsilon: float,
    sinkhorn_iterations: int,
) -> torch.Tensor:
    r"""Compute a balanced assignment matrix from pairwise distances."""
    Q = torch.exp(-distances / epsilon)
    B, K = Q.shape
    Q /= Q.sum()
    for _ in range(sinkhorn_iterations):
        Q /= Q.sum(dim=1, keepdim=True)
        Q /= B
        Q /= Q.sum(dim=0, keepdim=True)
        Q /= K
    return Q * B
