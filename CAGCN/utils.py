

import torch
from torch_geometric.utils import degree, sort_edge_index
from torch_geometric.utils import scatter


def calc_node_wise_norm(
    edge_weight: torch.Tensor, index: torch.Tensor,
    n_users: int, n_items: int
):
    assert edge_weight.ndim == 1
    return scatter(
        edge_weight, index, 
        dim=0, dim_size=n_users + n_items
    )[index]


def normalize_edge(edge_index, n_users, n_items):
    row, col = edge_index
    deg = degree(col, num_nodes=n_users + n_items)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.
    edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    return edge_weight, deg


def jaccard_similarity(R: torch.Tensor):
    M, N = R.shape
    R = R.detach().to_dense()
    R[R > 0.] = 1
    src_nodes = []
    trg_nodes = []
    values = []

    for u in range(M):
        items = torch.where(R[u])[0].flatten()
        U = R[:, items].t() # M x len(items)

        i_cap_i = U.matmul(U.t())
        i_cup_i = (U.sum(-1) + U.sum(-1, keepdim=True)) - i_cap_i

        weights = (i_cap_i / i_cup_i).mean(-1)

        src_nodes.extend([u] * len(items))
        trg_nodes.extend(items + M)
        values.extend(weights)
    
    for i in range(N):
        users = torch.where(R[:, i])[0].flatten()
        I = R[users, :] # len(users) x N

        u_cap_u = I.matmul(I.t())
        u_cup_u = (I.sum(-1) + I.sum(-1, keepdim=True)) - u_cap_u
    
        weights = (u_cap_u / u_cup_u).mean(-1)

        src_nodes.extend([i + M] * len(users))
        trg_nodes.extend(users)
        values.extend(weights)

    edge_index = torch.stack((
        torch.tensor(src_nodes), torch.tensor(trg_nodes)
    ), dim=0)
    edge_index, edge_weight = sort_edge_index(
        edge_index.long(), torch.tensor(values).float(),
        num_nodes= M + N
    )
    return edge_index, edge_weight


def common_neighbors_similarity(R: torch.Tensor):
    M, N = R.shape
    R = R.detach().to_dense()
    R[R > 0.] = 1
    src_nodes = []
    trg_nodes = []
    values = []

    for u in range(M):
        items = torch.where(R[u])[0].flatten()
        U = R[:, items].t() # M x len(items)

        i_cap_i = U.matmul(U.t())

        weights = i_cap_i.mean(-1)

        src_nodes.extend([u] * len(items))
        trg_nodes.extend(items + M)
        values.extend(weights)
    
    for i in range(N):
        users = torch.where(R[:, i])[0].flatten()
        I = R[users, :] # len(users) x N

        u_cap_u = I.matmul(I.t())
    
        weights = u_cap_u.mean(-1)

        src_nodes.extend([i + M] * len(users))
        trg_nodes.extend(users)
        values.extend(weights)

    edge_index = torch.stack((
        torch.tensor(src_nodes), torch.tensor(trg_nodes)
    ), dim=0)
    edge_index, edge_weight = sort_edge_index(
        edge_index.long(), torch.tensor(values).float(),
        num_nodes= M + N
    )
    return edge_index, edge_weight


def leicht_holme_nerman_similarity(R: torch.Tensor):
    M, N = R.shape
    R = R.detach().to_dense()
    R[R > 0.] = 1
    src_nodes = []
    trg_nodes = []
    values = []

    for u in range(M):
        items = torch.where(R[u])[0].flatten()
        U = R[:, items].t() # M x len(items)

        i_cap_i = U.matmul(U.t())
        degs = U.sum(-1, keepdim=True)
        i_mul_i = degs.mul(degs.t())

        weights = (i_cap_i / i_mul_i).mean(-1)

        src_nodes.extend([u] * len(items))
        trg_nodes.extend(items + M)
        values.extend(weights)
    
    for i in range(N):
        users = torch.where(R[:, i])[0].flatten()
        I = R[users, :] # len(users) x N

        u_cap_u = I.matmul(I.t())
        degs = I.sum(-1, keepdim=True)
        u_mul_u = degs.mul(degs.t())
    
        weights = (u_cap_u / u_mul_u).mean(-1)

        src_nodes.extend([i + M] * len(users))
        trg_nodes.extend(users)
        values.extend(weights)

    edge_index = torch.stack((
        torch.tensor(src_nodes), torch.tensor(trg_nodes)
    ), dim=0)
    edge_index, edge_weight = sort_edge_index(
        edge_index.long(), torch.tensor(values).float(),
        num_nodes= M + N
    )
    return edge_index, edge_weight


def salton_cosine_similarity(R: torch.Tensor):
    M, N = R.shape
    R = R.detach().to_dense()
    R[R > 0.] = 1
    src_nodes = []
    trg_nodes = []
    values = []

    for u in range(M):
        items = torch.where(R[u])[0].flatten()
        U = R[:, items].t() # M x len(items)

        i_cap_i = U.matmul(U.t())
        degs = U.sum(-1, keepdim=True)
        i_mul_i = degs.mul(degs.t())

        weights = (i_cap_i / i_mul_i.sqrt()).mean(-1)

        src_nodes.extend([u] * len(items))
        trg_nodes.extend(items + M)
        values.extend(weights)
    
    for i in range(N):
        users = torch.where(R[:, i])[0].flatten()
        I = R[users, :] # len(users) x N

        u_cap_u = I.matmul(I.t())
        degs = I.sum(-1, keepdim=True)
        u_mul_u = degs.mul(degs.t())
    
        weights = (u_cap_u / u_mul_u.sqrt()).mean(-1)

        src_nodes.extend([i + M] * len(users))
        trg_nodes.extend(users)
        values.extend(weights)

    edge_index = torch.stack((
        torch.tensor(src_nodes), torch.tensor(trg_nodes)
    ), dim=0)
    edge_index, edge_weight = sort_edge_index(
        edge_index.long(), torch.tensor(values).float(),
        num_nodes= M + N
    )
    return edge_index, edge_weight