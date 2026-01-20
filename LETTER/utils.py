

import torch


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
    distances: torch.Tensor, 
    epsilon: float, sinkhorn_iterations: int
) -> torch.Tensor:

    _dtype = distances.dtype
    distances = center_distance_for_constraint(distances).double()

    Q = torch.exp(-distances / epsilon)

    B = Q.shape[0] # number of samples to assign
    K = Q.shape[1] # how many centroids per block (usually set to 256)

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


    Q *= B # the colomns must sum to 1 so that Q is an assignment
    return Q.to(_dtype)

