# utils/loss.py
from __future__ import annotations
from typing import Dict, Tuple, Union
import torch
import torch.nn.functional as F


def gaussian_kernel(x: torch.Tensor, y: torch.Tensor, sigmas=None) -> torch.Tensor:
    """
    Median heuristic bandwidth selection for stable high-dimensional MMD.
    """
    x = x.unsqueeze(1)
    y = y.unsqueeze(0)
    dist_sq = (x - y).pow(2).sum(-1)

    if sigmas is None:
        with torch.no_grad():
            median_dist2 = torch.median(dist_sq.view(-1))
            median_dist2 = torch.clamp(median_dist2, min=1e-6)
            base_sigma2 = median_dist2 / 2.0
        sigmas_sq = [base_sigma2 * s for s in [0.25, 1.0, 4.0]]
    else:
        sigmas_sq = [s**2 for s in sigmas]

    k = 0.0
    for s2 in sigmas_sq:
        k = k + torch.exp(-dist_sq / (2.0 * s2 + 1e-6))
    return k


def mmd_rbf(x: torch.Tensor, y: torch.Tensor, sigmas=None) -> torch.Tensor:
    K_xx = gaussian_kernel(x, x, sigmas)
    K_yy = gaussian_kernel(y, y, sigmas)
    K_xy = gaussian_kernel(x, y, sigmas)
    return K_xx.mean() + K_yy.mean() - 2.0 * K_xy.mean()


def _as_bt(x: torch.Tensor) -> torch.Tensor:
    """Ensure shape (B, T) for time-series losses."""
    if x.dim() == 3:
        x = x.squeeze(1)
    return x.view(x.size(0), -1)


def diff(x: torch.Tensor, dt: float) -> torch.Tensor:
    """
    Central finite difference using torch.gradient.
    Maintains the same shape through padding handled by PyTorch.
    """
    x_bt = _as_bt(x)
    return torch.gradient(x_bt, spacing=dt, dim=-1)[0]


def loss_rec(real: torch.Tensor, rec: torch.Tensor, sigmas=None) -> torch.Tensor:
    xr = _as_bt(real)
    ap = _as_bt(rec)
    mse = F.mse_loss(ap, xr)
    mmd = mmd_rbf(ap, xr, sigmas=sigmas)
    return mse + mmd


def loss_ddl(
    s_hat: torch.Tensor,
    a_real: torch.Tensor,
    dt: float,
    sigmas=None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Dimensionless differential loss enforcing s ~ a_real.
    Returns (loss, v_derived, a_derived).
    """
    a_target = _as_bt(a_real)
    v_derived = diff(s_hat, dt)
    a_derived = diff(v_derived, dt)
    mse_a = F.mse_loss(a_derived, a_target)
    mmd_a = mmd_rbf(a_derived, a_target, sigmas=sigmas)
    return mse_a + mmd_a, v_derived, a_derived


def loss_dt(
    a_in: torch.Tensor,
    v_derived: torch.Tensor,
    s_hat: torch.Tensor,
    y_onehot: torch.Tensor,
    delta_n_phys: torch.Tensor,
    mass_M: Union[float, torch.Tensor],
    scales: Dict[str, torch.Tensor],
    params_phys: Dict[str, torch.Tensor],
    normal_label: int = 0,
    reg_lambda: float = 0.0,
) -> torch.Tensor:
    """
    Physical residual loss (Ma + Cv + Ks + K*delta^1.5 = W) with optional
    regularization on dimensionless parameters.
    """
    if params_phys is None:
        raise ValueError("params_phys missing")
    if scales is None:
        raise ValueError("scales missing for physical restoration")

    device = s_hat.device
    dtype = s_hat.dtype
    B = s_hat.shape[0]

    scale_a = scales["scale_a"].view(1, 1).to(device=device, dtype=dtype)
    scale_v = scales["scale_v"].view(1, 1).to(device=device, dtype=dtype)
    scale_s = scales["scale_s"].view(1, 1).to(device=device, dtype=dtype)

    a_phys = _as_bt(a_in) * scale_a
    v_phys = _as_bt(v_derived) * scale_v
    s_phys = _as_bt(s_hat) * scale_s

    delta_n_phys = (delta_n_phys.view(-1) * 1e-2).to(device=device, dtype=dtype)
    defect_phys = (y_onehot.float() @ delta_n_phys.float()).view(B, 1)
    fault_mask = 1.0 - y_onehot[:, normal_label].view(B, 1)
    delta_phys = F.relu(s_phys - defect_phys) * fault_mask

    c_phys = params_phys["c_phys"].view(B, 1).to(device=device, dtype=dtype)
    k_phys = params_phys["k_phys"].view(B, 1).to(device=device, dtype=dtype)

    M = mass_M if torch.is_tensor(mass_M) else torch.tensor(mass_M, device=device, dtype=dtype)
    M = M.view(1, 1)

    load_W = torch.tensor(2000.0, device=device, dtype=dtype)

    residual = (
        M * a_phys
        + c_phys * v_phys
        + k_phys * s_phys
        + k_phys * delta_phys.pow(1.5)
        - load_W
    )
    loss_physics = (residual / load_W).pow(2).mean()

    if reg_lambda <= 0.0:
        return loss_physics

    c_tilde = params_phys["c_tilde"]
    k_tilde = params_phys["k_tilde"]
    loss_reg = ((c_tilde - 0.3).pow(2) + (k_tilde - 1.0).pow(2)).mean()
    return loss_physics + reg_lambda * loss_reg


def loss_total(
    Lddl: torch.Tensor,
    Ldt: torch.Tensor,
    Lrec: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    gamma: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    同方差不确定性加权损失
    """
    return alpha * Lddl + beta * Ldt + gamma * Lrec - 0.5 * (
        torch.log(alpha + eps) + torch.log(beta + eps) + torch.log(gamma + eps)
    )

