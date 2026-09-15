"""Loss and inference wrapper for the tactile-only private ACT copy."""

from __future__ import annotations

import torch
from torch import nn

from .act_tactile_model import TactileACTConfig, TactileACTModel, build_optimizer


class TactileACTPolicy(nn.Module):
    """Compute ACT's masked L1 plus conditional-VAE KL objective."""

    def __init__(self, config: TactileACTConfig, *, kl_weight: float = 10.0) -> None:
        super().__init__()
        self.config = config
        self.kl_weight = float(kl_weight)
        self.model = TactileACTModel(config)

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        actions = batch["actions"][:, :self.model.num_queries]
        is_pad = batch["is_pad"][:, :self.model.num_queries]
        action_pred, pad_logits, (mu, logvar) = self.model(
            batch["qpos"],
            batch["images"],
            batch["tactile"],
            actions,
            is_pad,
        )
        valid = (~is_pad.bool()).unsqueeze(-1)
        absolute_error = (action_pred - actions).abs()
        denominator = valid.sum().clamp_min(1) * action_pred.shape[-1]
        l1 = (absolute_error * valid).sum() / denominator
        if mu is None or logvar is None:
            raise RuntimeError("training loss requires posterior statistics")
        kl = (-0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(dim=-1)).mean()
        total = l1 + self.kl_weight * kl
        return {
            "loss": total,
            "l1": l1,
            "kl": kl,
            "action_pred": action_pred,
            "pad_logits": pad_logits,
        }

    def predict(
        self,
        qpos: torch.Tensor,
        images: torch.Tensor,
        tactile: torch.Tensor,
    ) -> torch.Tensor:
        action_pred, _, _ = self.model(qpos, images, tactile)
        return action_pred

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.compute_loss(batch)

    def configure_optimizer(
        self,
    ) -> torch.optim.Optimizer:
        return build_optimizer(self.model)
