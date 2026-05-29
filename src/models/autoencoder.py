from __future__ import annotations

import torch
from torch import nn


class AutoEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], latent_dim: int) -> None:
        super().__init__()

        encoder_layers: list[nn.Module] = []
        dims = [input_dim] + hidden_dims + [latent_dim]
        for i in range(len(dims) - 1):
            encoder_layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                encoder_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: list[nn.Module] = []
        rev_dims = [latent_dim] + list(reversed(hidden_dims)) + [input_dim]
        for i in range(len(rev_dims) - 1):
            decoder_layers.append(nn.Linear(rev_dims[i], rev_dims[i + 1]))
            if i < len(rev_dims) - 2:
                decoder_layers.append(nn.ReLU())
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z
