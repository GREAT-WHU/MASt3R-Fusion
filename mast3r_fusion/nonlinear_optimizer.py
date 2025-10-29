import math
import torch


def check_convergence(
    iter,
    rel_error_threshold,
    delta_norm_threshold,
    old_cost,
    new_cost,
    delta,
    verbose=False,
):
    cost_diff = old_cost - new_cost
    rel_dec = math.fabs(cost_diff / old_cost)
    delta_norm = torch.linalg.norm(delta)

    converged = rel_dec < rel_error_threshold or delta_norm < delta_norm_threshold
    if verbose:
        print(
            f"{iter=} | {new_cost=} {cost_diff=} {rel_dec=} {delta_norm=} | {converged=}"
        )

    # print(f"{iter=} | {new_cost=} {cost_diff=} {rel_dec=} {delta_norm=} | {converged=}")
    return converged


def huber(r, k=1.345):
    unit = torch.ones((1), dtype=r.dtype, device=r.device)
    r_abs = torch.abs(r)
    mask = r_abs < k
    w = torch.where(mask, unit, k / r_abs)
    return w


def tukey(r, t=4.6851):
    zero = torch.tensor(0.0, dtype=r.dtype, device=r.device)
    r_abs = torch.abs(r)
    tmp = 1 - torch.square(r_abs / t)
    tmp2 = tmp * tmp
    w = torch.where(r_abs < t, tmp2, zero)
    return w
