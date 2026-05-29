from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], latent_dim: int) -> None:
        super().__init__()

        enc_layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            enc_layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        self.encoder_body = nn.Sequential(*enc_layers)
        self.fc_mu = nn.Linear(prev, latent_dim)
        self.fc_logvar = nn.Linear(prev, latent_dim)

        dec_layers: list[nn.Module] = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            dec_layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        dec_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder_body(x)
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar


def vae_loss(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    recon = F.mse_loss(x_hat, x, reduction="mean")
    kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)).mean()
    total = recon + beta * kl
    return total, recon, kl

