from __future__ import annotations
from typing import Dict

import torch
import torch.nn as nn


class PIModule(nn.Module):
    """
    Table 3 PI structure:
    1024 -> 512 -> 256 -> 512 -> 1024
    """

    def __init__(self, input_dim: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input is acceleration; output is displacement s.
        return self.net(x)


class Encoder(nn.Module):
    def __init__(self, x_dim=1024, y_dim=10, z_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim + y_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, z_dim)
        )

    def forward(self, x: torch.Tensor, y_onehot: torch.Tensor) -> torch.Tensor:
        h = torch.cat([x, y_onehot], dim=-1)
        return self.net(h)


class Decoder(nn.Module):
    def __init__(self, x_dim=1024, y_dim=10, z_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + y_dim, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, x_dim)
        )

    def forward(self, z: torch.Tensor, y_onehot: torch.Tensor) -> torch.Tensor:
        h = torch.cat([z, y_onehot], dim=-1)
        return self.net(h)


class PIAENet(nn.Module):
    """
    z2 corresponds to [c, k]
    pi2 is ALWAYS used.
    """

    def __init__(
        self,
        x_dim: int = 1024,
        y_dim: int = 10,
        z_dim: int = 2,
        c_scale: float = 1e2,
        k_scale: float = 2e10,
    ):
        super().__init__()
        assert z_dim == 2, "z_dim must be 2 for [c, k]"

        self.c_scale = c_scale
        self.k_scale = k_scale

        self.encoder = Encoder(x_dim=x_dim, y_dim=y_dim, z_dim=z_dim)
        self.decoder = Decoder(x_dim=x_dim, y_dim=y_dim, z_dim=z_dim)
        self.pim = PIModule(input_dim=x_dim)

        self.softplus = nn.Softplus(beta=1.0)

    def _to_phys_params(self, z_raw: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        z_raw: (B,2) -> phys params (positive)
        Order: [c, k]
        """
        eps = 1e-6
        z_pos = self.softplus(z_raw) + eps
        z_pos = torch.clamp(z_pos, min=1e-4, max=50.0)
        c_tilde, k_tilde = z_pos[:, 0], z_pos[:, 1]
        c_phys = c_tilde * self.c_scale
        k_phys = k_tilde * self.k_scale
        return {
            "c_tilde": c_tilde,
            "k_tilde": k_tilde,
            "c_phys": c_phys,
            "k_phys": k_phys,
        }

    def forward(self, acc: torch.Tensor, y_onehot: torch.Tensor) -> Dict[str, torch.Tensor]:
        if acc.dim() == 3:
            acc_in = acc.squeeze(1)
        else:
            acc_in = acc

        z_raw = self.encoder(acc_in, y_onehot)
        params = self._to_phys_params(z_raw)

        z_dimless = torch.stack([params["c_tilde"], params["k_tilde"]], dim=-1)
        acc_hat = self.decoder(z_dimless, y_onehot)
        s_hat = self.pim(acc_in)

        return {
            "z_raw": z_raw,
            "params": params,
            "acc_hat": acc_hat,
            "s_hat": s_hat,
        }
