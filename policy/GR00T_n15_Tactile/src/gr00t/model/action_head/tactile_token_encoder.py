"""ACT-inspired multi-site TacMap encoder for GR00T N1.5."""

from __future__ import annotations

import torch
from torch import nn


class TactileTokenEncoder(nn.Module):
    """Encode ``[B,N,1,H,W]`` tactile maps into one GR00T context token."""

    def __init__(
        self,
        *,
        num_sites: int,
        output_dim: int,
        encoder_dim: int = 256,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        if num_sites <= 0:
            raise ValueError(f"num_sites must be positive, got {num_sites}")
        if encoder_dim <= 0 or output_dim <= 0:
            raise ValueError("encoder_dim and output_dim must be positive")
        if encoder_dim % num_heads != 0:
            raise ValueError(f"encoder_dim={encoder_dim} must be divisible by num_heads={num_heads}")

        self.num_sites = int(num_sites)
        self.shared_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Conv2d(128, encoder_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(16, encoder_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.site_embedding = nn.Embedding(self.num_sites, encoder_dim)
        self.pool_query = nn.Embedding(1, encoder_dim)
        self.attention_pool = nn.MultiheadAttention(encoder_dim, num_heads, batch_first=True)
        self.pool_norm = nn.LayerNorm(encoder_dim)
        self.output_projection = nn.Linear(encoder_dim, output_dim)
        self.modality_embedding = nn.Embedding(1, output_dim)

    def forward(self, tactile: torch.Tensor) -> torch.Tensor:
        if not isinstance(tactile, torch.Tensor):
            raise ValueError("tactile input must be a torch.Tensor")
        if tactile.ndim != 5 or tactile.shape[2] != 1:
            raise ValueError(f"tactile must have shape [B,N,1,H,W], got {tuple(tactile.shape)}")
        batch_size, num_sites, _, height, width = tactile.shape
        if num_sites != self.num_sites:
            raise ValueError(f"tactile has {num_sites} sites, expected {self.num_sites}")
        if height <= 0 or width <= 0:
            raise ValueError(f"tactile spatial dimensions must be positive, got {(height, width)}")

        if tactile.dtype == torch.uint8:
            tactile = tactile.float().div(255.0)
        elif not torch.is_floating_point(tactile):
            raise ValueError(f"tactile must be uint8 or floating point, got {tactile.dtype}")
        if not torch.isfinite(tactile).all() or torch.any(tactile < 0) or torch.any(tactile > 1):
            raise ValueError("floating tactile values must be finite and in [0,1]")

        dtype = self.output_projection.weight.dtype
        maps = tactile.to(dtype=dtype).reshape(batch_size * num_sites, 1, height, width)
        site_features = self.shared_cnn(maps).flatten(1).reshape(batch_size, num_sites, -1)
        site_ids = torch.arange(num_sites, device=tactile.device)
        site_tokens = site_features + self.site_embedding(site_ids).unsqueeze(0)
        query = self.pool_query.weight.unsqueeze(0).expand(batch_size, -1, -1)
        attended, _ = self.attention_pool(query, site_tokens, site_tokens, need_weights=False)
        pooled = self.pool_norm(query + attended)
        token = self.output_projection(pooled)
        return token + self.modality_embedding.weight.unsqueeze(0)
