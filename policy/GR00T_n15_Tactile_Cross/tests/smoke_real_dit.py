"""Run separately from unit tests: uses real diffusers DiT on CPU, no checkpoint download."""

import os
import sys
from pathlib import Path

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from transformers.feature_extraction_utils import BatchFeature

from gr00t.model.action_head.flow_matching_action_head import (
    FlowmatchingActionHead,
    FlowmatchingActionHeadConfig,
)


def run():
    torch.set_num_threads(2)
    backbone = BatchFeature({
        "backbone_features": torch.randn(2, 3, 16),
        "backbone_attention_mask": torch.ones(2, 3, dtype=torch.long),
    })
    inputs = BatchFeature({
        "state": torch.randn(2, 1, 4),
        "action": torch.randn(2, 2, 4),
        "action_mask": torch.ones(2, 2, 4),
        "embodiment_id": torch.zeros(2, dtype=torch.long),
        "tactile": torch.rand(2, 2, 1, 32, 32),
    })
    for stage, grid in (("pre_dit", 2), ("post_dit", 2), ("post_dit", 4)):
        config = FlowmatchingActionHeadConfig(
            diffusion_model_cfg=dict(
                num_attention_heads=2, attention_head_dim=8, output_dim=8,
                num_layers=2, cross_attention_dim=16,
                interleave_self_attention=True, dropout=0.0,
                positional_embeddings=None,
            ),
            input_embedding_dim=16, backbone_embedding_dim=16, hidden_size=8,
            max_state_dim=4, action_dim=4, action_horizon=2,
            max_seq_len=16, num_target_vision_tokens=1, max_num_embodiments=1,
            use_vlln=False, use_tactile=True, tactile_num_sites=2,
            tactile_num_heads=2, tactile_dropout_prob=0.0,
            tactile_fusion_stage=stage, tactile_grid_size=grid,
            num_inference_timesteps=2,
        )
        head = FlowmatchingActionHead(config)
        optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3)
        for iteration in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss = head(backbone, inputs).loss
            assert torch.isfinite(loss)
            loss.backward()
            if iteration == 1:
                grad = head.tactile_encoder.shared_cnn[0].weight.grad
                assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
            optimizer.step()
        head.eval()
        restored = FlowmatchingActionHead(FlowmatchingActionHeadConfig(**config.to_dict())).eval()
        restored.load_state_dict(head.state_dict(), strict=True)
        torch.manual_seed(123)
        expected = head.get_action(backbone, inputs).action_pred
        torch.manual_seed(123)
        actual = restored.get_action(backbone, inputs).action_pred
        torch.testing.assert_close(actual, expected)
        assert actual.shape == (2, 2, 4) and torch.isfinite(actual).all()
        print(f"PASS real DiT: {stage}, grid={grid}, backward + inference + checkpoint round-trip")


if __name__ == "__main__":
    run()
