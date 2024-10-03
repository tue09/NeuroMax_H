import torch

def pairwise_euclidean_distance(x, y):
    if x.size(1) != y.size(1):
        cost = torch.sum(x ** 2, axis=1, keepdim=True) + torch.sum(y ** 2, dim=1) - 2 * torch.matmul(x.t(), y.t())
    else:
        cost = torch.sum(x ** 2, axis=1, keepdim=True) + torch.sum(y ** 2, dim=1) - 2 * torch.matmul(x, y.t())
    return cost

