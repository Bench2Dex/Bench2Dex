"""Tactile-only policy built from the private copy of ACT.

This module deliberately imports only :mod:`policy.tactile_policy.act`; the
upstream ``policy.ACT`` package remains an untouched reference implementation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from types import SimpleNamespace
from typing import Sequence

import torch
from torch import nn
import torchvision.transforms as transforms

from .act.detr.models import build_ACT_model


@dataclass(frozen=True)
class TactileACTConfig:
    state_dim: int
    camera_names: tuple[str, ...]
    site_names: tuple[str, ...]
    tactile_height: int
    tactile_width: int
    chunk_size: int = 30
    hidden_dim: int = 512
    dim_feedforward: int = 3200
    nheads: int = 8
    enc_layers: int = 4
    dec_layers: int = 7
    dropout: float = 0.1
    backbone: str = "resnet18"
    lr: float = 1e-5
    lr_backbone: float = 1e-5
    weight_decay: float = 1e-4

    def __post_init__(self) -> None:
        if self.state_dim <= 0 or self.chunk_size <= 0:
            raise ValueError("state_dim and chunk_size must be positive")
        if not self.camera_names or not self.site_names:
            raise ValueError("tactile ACT requires at least one camera and tactile site")
        if len(set(self.site_names)) != len(self.site_names):
            raise ValueError("site_names must be unique")
        if self.tactile_height <= 0 or self.tactile_width <= 0:
            raise ValueError("tactile map dimensions must be positive")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "TactileACTConfig":
        values = dict(values)
        values["camera_names"] = tuple(values["camera_names"])
        values["site_names"] = tuple(values["site_names"])
        return cls(**values)


class TactileACTModel(nn.Module):
    """The copied ACT DETRVAE with TacMap tokens enabled unconditionally."""

    def __init__(self, config: TactileACTConfig) -> None:
        super().__init__()
        self.config = config
        args = SimpleNamespace(
            state_dim=config.state_dim,
            chunk_size=config.chunk_size,
            camera_names=list(config.camera_names),
            hidden_dim=config.hidden_dim,
            nheads=config.nheads,
            dim_feedforward=config.dim_feedforward,
            enc_layers=config.enc_layers,
            dec_layers=config.dec_layers,
            dropout=config.dropout,
            pre_norm=False,
            backbone=config.backbone,
            lr_backbone=config.lr_backbone,
            dilation=False,
            position_embedding="sine",
            masks=False,
        )
        self.model = build_ACT_model(
            args,
            tactile_cfg={
                "num_sites": len(config.site_names),
                "height": config.tactile_height,
                "width": config.tactile_width,
                "site_names": config.site_names,
            },
        )
        self._normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    @property
    def num_queries(self) -> int:
        return self.model.num_queries

    def forward(
        self,
        qpos: torch.Tensor,
        images: torch.Tensor,
        tactile: torch.Tensor,
        actions: torch.Tensor | None = None,
        is_pad: torch.Tensor | None = None,
    ):
        images = self._normalize(images.float() / 255.0)
        if actions is not None:
            actions = actions[:, :self.num_queries]
            is_pad = is_pad[:, :self.num_queries]
        return self.model(qpos, images, None, actions, is_pad, tactile=tactile)


def build_optimizer(model: TactileACTModel) -> torch.optim.Optimizer:
    config = model.config
    return torch.optim.AdamW(
        [
            {
                "params": [
                    parameter
                    for name, parameter in model.named_parameters()
                    if parameter.requires_grad and "backbone" not in name
                ]
            },
            {
                "params": [
                    parameter
                    for name, parameter in model.named_parameters()
                    if parameter.requires_grad and "backbones.0" in name
                ],
                "lr": config.lr_backbone,
            },
            {
                "params": [
                    parameter
                    for name, parameter in model.named_parameters()
                    if parameter.requires_grad and "backbones.1" in name
                ],
                "lr": config.lr_backbone,
            },
        ],
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
