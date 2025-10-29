import einops
import lietorch
import torch


def as_SE3(X):
    if isinstance(X, lietorch.SE3):
        return X
    t, q, s = einops.rearrange(X.data.detach().cpu(), "... c -> (...) c").split(
        [3, 4, 1], -1
    )
    T_WC = lietorch.SE3(torch.cat([t, q], dim=-1))
    return T_WC
