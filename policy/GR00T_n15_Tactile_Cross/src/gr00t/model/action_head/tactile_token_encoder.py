"""Spatial TacMap token encoder for action-conditioned tactile fusion."""

from __future__ import annotations

import torch
from torch import nn


class TactileTokenEncoder(nn.Module):
    """Encode normalized maps into ``[B,N*grid_size**2,D]`` spatial tokens."""

    def __init__(
        self,
        *,
        num_sites: int,
        output_dim: int,
        encoder_dim: int = 256,
        num_heads: int = 8,
        grid_size: int = 2,
    ) -> None:
        super().__init__()
        if num_sites <= 0 or output_dim <= 0:
            raise ValueError("num_sites and output_dim must be positive")
        if encoder_dim <= 0 or num_heads <= 0:
            raise ValueError("encoder_dim and num_heads must be positive")
        self.num_sites = int(num_sites)
        if not isinstance(grid_size, int) or grid_size <= 0:
            raise ValueError("grid_size must be a positive integer")
        self.num_spatial_tokens = grid_size ** 2
        self.shared_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((grid_size, grid_size)),
        )
        self.output_projection = nn.Linear(128, output_dim)
        self.sensor_embedding = nn.Embedding(self.num_sites, output_dim)
        self.spatial_embedding = nn.Embedding(self.num_spatial_tokens, output_dim)

    def forward(self, tactile: torch.Tensor) -> torch.Tensor:
        if not isinstance(tactile, torch.Tensor):
            raise ValueError("tactile input must be a torch.Tensor")
        if tactile.ndim != 5 or tactile.shape[2] != 1:
            raise ValueError(f"tactile must have shape [B,N,1,H,W], got {tuple(tactile.shape)}")
        batch_size, num_sites, _, height, width = tactile.shape
        if num_sites != self.num_sites:
            raise ValueError(f"tactile has {num_sites} sites, expected {self.num_sites}")
        if not torch.is_floating_point(tactile):
            raise ValueError("tactile input must be floating point raw-depth data normalized to [0,1]")
        if not torch.isfinite(tactile).all() or torch.any(tactile < 0) or torch.any(tactile > 1):
            raise ValueError("tactile values must be finite and normalized to [0,1]")

        maps = tactile.to(dtype=self.output_projection.weight.dtype).reshape(
            batch_size * num_sites, 1, height, width
        )
        tokens = self.shared_cnn(maps).flatten(2).transpose(1, 2)
        tokens = self.output_projection(tokens).reshape(batch_size, num_sites, self.num_spatial_tokens, -1)
        sensor_ids = torch.arange(num_sites, device=tokens.device)
        spatial_ids = torch.arange(self.num_spatial_tokens, device=tokens.device)
        tokens = (
            tokens
            + self.sensor_embedding(sensor_ids)[None, :, None, :]
            + self.spatial_embedding(spatial_ids)[None, None, :, :]
        )
        return tokens.flatten(1, 2)
