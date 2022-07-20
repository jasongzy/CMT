import torch


@torch.no_grad()
def ipot(C, x_len, y_len, beta, iteration, k):
    """ [B, M, N], [B], [B, M], [B], [B, N], [B, M, N]"""
    b, m, n = C.size()
    sigma = torch.ones(b, m, dtype=C.dtype, device=C.device) / x_len.unsqueeze(1)
    T = torch.ones(b, n, m, dtype=C.dtype, device=C.device)
    A = torch.exp(-C.transpose(1, 2) / beta)

    # broadcastable lengths
    x_len = x_len.unsqueeze(1).unsqueeze(2)
    y_len = y_len.unsqueeze(1).unsqueeze(2)

    for _ in range(iteration):
        Q = A * T  # bs * n * m
        sigma = sigma.view(b, m, 1)
        for _ in range(k):
            delta = 1 / (y_len * Q.matmul(sigma).view(b, 1, n))
            sigma = 1 / (x_len * delta.matmul(Q))
        T = delta.view(b, n, 1) * Q * sigma
    return T


def trace(x):
    """ compute trace of input tensor (batched) """
    b, m, n = x.size()
    assert m == n
    mask = torch.eye(n, dtype=torch.bool, device=x.device).unsqueeze(0).expand_as(x)
    trace = x.masked_select(mask).contiguous().view(b, n).sum(dim=-1, keepdim=False)
    return trace


def ipot_dist(cost, iteration=50):
    """
    :param cost: B,M,N
    :return: B
    """
    B = cost.shape[0]
    T = ipot(cost.detach(), torch.ones([B], device=cost.device), torch.ones([B], device=cost.device), 0.5, iteration, 1)
    # dist = trace(cost.matmul(T.detach()))
    col_ind = T.argmax(1)
    dist = torch.gather(cost, 2, col_ind.unsqueeze(-1)).squeeze().sum(1)
    return dist
