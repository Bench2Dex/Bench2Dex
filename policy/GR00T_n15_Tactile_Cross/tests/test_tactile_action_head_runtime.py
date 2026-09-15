from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path

try:
    import torch
    from torch import nn
except ModuleNotFoundError:
    torch = None
    nn = None


ROOT = Path(__file__).resolve().parents[1]


def _load_action_head_module():
    assert torch is not None and nn is not None

    transformers = types.ModuleType("transformers")

    class PretrainedConfig:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    transformers.PretrainedConfig = PretrainedConfig
    feature_utils = types.ModuleType("transformers.feature_extraction_utils")

    class BatchFeature(dict):
        def __init__(self, data=None, **kwargs):
            super().__init__(data or {}, **kwargs)

        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError as exc:
                raise AttributeError(name) from exc

    feature_utils.BatchFeature = BatchFeature
    sys.modules["transformers"] = transformers
    sys.modules["transformers.feature_extraction_utils"] = feature_utils

    for package_name in ("gr00t", "gr00t.model", "gr00t.model.action_head", "flow_test"):
        package = types.ModuleType(package_name)
        package.__path__ = []
        sys.modules[package_name] = package

    action_encoder = types.ModuleType("gr00t.model.action_head.action_encoder")

    class SinusoidalPositionalEncoding(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.dim = dim

        def forward(self, timesteps):
            return torch.zeros(*timesteps.shape, self.dim, device=timesteps.device)

    action_encoder.SinusoidalPositionalEncoding = SinusoidalPositionalEncoding
    action_encoder.swish = torch.nn.functional.silu
    sys.modules[action_encoder.__name__] = action_encoder

    tactile_path = ROOT / "src/gr00t/model/action_head/tactile_token_encoder.py"
    tactile_spec = importlib.util.spec_from_file_location(
        "gr00t.model.action_head.tactile_token_encoder", tactile_path
    )
    assert tactile_spec is not None and tactile_spec.loader is not None
    tactile_module = importlib.util.module_from_spec(tactile_spec)
    sys.modules[tactile_spec.name] = tactile_module
    tactile_spec.loader.exec_module(tactile_module)

    cross_dit = types.ModuleType("flow_test.cross_attention_dit")

    class DiT(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()

        def forward(self, hidden_states, **kwargs):
            return hidden_states

    class SelfAttentionTransformer(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()

        def forward(self, hidden_states):
            return hidden_states

    cross_dit.DiT = DiT
    cross_dit.SelfAttentionTransformer = SelfAttentionTransformer
    sys.modules[cross_dit.__name__] = cross_dit

    flow_path = ROOT / "src/gr00t/model/action_head/flow_matching_action_head.py"
    flow_spec = importlib.util.spec_from_file_location(
        "flow_test.flow_matching_action_head", flow_path
    )
    assert flow_spec is not None and flow_spec.loader is not None
    flow_module = importlib.util.module_from_spec(flow_spec)
    sys.modules[flow_spec.name] = flow_module
    flow_spec.loader.exec_module(flow_module)
    return flow_module, BatchFeature


@unittest.skipIf(torch is None, "PyTorch is required for action-head runtime tests")
class TactileActionHeadRuntimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module, cls.BatchFeature = _load_action_head_module()

    def _head(self, *, dropout: float = 0.0, stage: str = "pre_dit", hidden_size: int = 8):
        config = self.module.FlowmatchingActionHeadConfig(
            add_pos_embed=False,
            diffusion_model_cfg={},
            input_embedding_dim=8,
            backbone_embedding_dim=8,
            hidden_size=hidden_size,
            max_seq_len=16,
            max_state_dim=4,
            action_dim=4,
            action_horizon=2,
            num_inference_timesteps=2,
            max_num_embodiments=1,
            num_target_vision_tokens=1,
            use_vlln=False,
            use_tactile=True,
            tactile_num_sites=2,
            tactile_num_heads=2,
            tactile_dropout_prob=dropout,
            tactile_fusion_stage=stage,
            tune_projector=True,
            tune_diffusion_model=True,
        )
        return self.module.FlowmatchingActionHead(config)

    def test_zero_initialization_and_residual_gate(self) -> None:
        head = self._head()
        self.assertEqual(torch.count_nonzero(head.tactile_cross_attn.out_proj.weight).item(), 0)

        class OnesAttention(nn.Module):
            def forward(self, query, key, value, need_weights=False):
                return torch.ones_like(query), None

        head.tactile_cross_attn = OnesAttention()
        action = torch.zeros(2, 2, 8)
        tactile = torch.zeros(2, 8, 8)
        keep = torch.tensor([0.0, 1.0]).view(2, 1, 1)
        fused = head._fuse_action_features(action, tactile, keep)
        torch.testing.assert_close(fused[0], torch.zeros_like(fused[0]))
        torch.testing.assert_close(fused[1], torch.ones_like(fused[1]))

    def test_full_tactile_dropout_emits_zero_keep_mask(self) -> None:
        head = self._head(dropout=1.0)
        head.train()
        _, keep_mask = head._encode_tactile_tokens(
            torch.zeros(2, 2, 1, 8, 8), device="cpu", dtype=torch.float32
        )
        torch.testing.assert_close(keep_mask, torch.zeros(2, 1, 1))

    def test_forward_backward_reaches_base_and_tactile_modules(self) -> None:
        head = self._head()
        backbone = self.BatchFeature(
            {
                "backbone_features": torch.zeros(2, 3, 8),
                "backbone_attention_mask": torch.ones(2, 3, dtype=torch.long),
            }
        )
        action_input = self.BatchFeature(
            {
                "state": torch.zeros(2, 1, 4),
                "action": torch.zeros(2, 2, 4),
                "action_mask": torch.ones(2, 2, 4),
                "embodiment_id": torch.zeros(2, dtype=torch.long),
                "tactile": torch.zeros(2, 2, 1, 8, 8),
            }
        )
        output = head(backbone, action_input)
        output.loss.backward()
        self.assertIsNotNone(head.action_encoder.W1.W.grad)
        self.assertIsNotNone(head.tactile_cross_attn.out_proj.weight.grad)
        self.assertIsNotNone(head.tactile_encoder.shared_cnn[0].weight.grad)

    def test_post_dit_zero_init_preserves_predictions_and_dropout(self) -> None:
        legacy = self._head().eval()
        post = self._head(stage="post_dit").eval()
        post.load_state_dict(legacy.state_dict(), strict=False)
        backbone = self.BatchFeature({
            "backbone_features": torch.randn(2, 3, 8),
            "backbone_attention_mask": torch.ones(2, 3, dtype=torch.long),
        })
        inputs = self.BatchFeature({
            "state": torch.randn(2, 1, 4),
            "embodiment_id": torch.zeros(2, dtype=torch.long),
            "tactile": torch.rand(2, 2, 1, 8, 8),
        })
        torch.manual_seed(11)
        expected = legacy.get_action(backbone, inputs).action_pred
        torch.manual_seed(11)
        actual = post.get_action(backbone, inputs).action_pred
        torch.testing.assert_close(expected, actual)
        with torch.no_grad():
            post.tactile_cross_attn.out_proj.weight.normal_(std=0.1)
            post.tactile_cross_attn.out_proj.bias.fill_(1.0)
        features = torch.randn(2, 2, 8)
        tokens = post.tactile_encoder(inputs.tactile)
        dropped = post._fuse_action_features(features, tokens, torch.zeros(2, 1, 1))
        torch.testing.assert_close(dropped, features, rtol=0, atol=0)

    def test_post_dit_uses_contextual_action_tokens_with_different_width(self) -> None:
        head = self._head(stage="post_dit", hidden_size=12)
        seen = []

        class ContextDiT(nn.Module):
            def forward(self, hidden_states, encoder_hidden_states, **kwargs):
                contextual = hidden_states + encoder_hidden_states.mean(dim=1, keepdim=True)
                contextual = torch.nn.functional.pad(contextual, (0, 4))
                seen.append(contextual)
                return contextual

        head.model = ContextDiT()
        queried = []
        hook = head.tactile_query_norm.register_forward_pre_hook(
            lambda module, args: queried.append(args[0])
        )
        backbone = self.BatchFeature({
            "backbone_features": torch.randn(2, 3, 8),
            "backbone_attention_mask": torch.ones(2, 3, dtype=torch.long),
        })
        inputs = self.BatchFeature({
            "state": torch.randn(2, 1, 4),
            "action": torch.randn(2, 2, 4),
            "action_mask": torch.ones(2, 2, 4),
            "embodiment_id": torch.zeros(2, dtype=torch.long),
            "tactile": torch.rand(2, 2, 1, 16, 16),
        })
        head(backbone, inputs).loss.backward()
        torch.testing.assert_close(queried[0], seen[0][:, -2:])
        self.assertGreater(head.tactile_cross_attn.out_proj.weight.grad.abs().sum().item(), 0)
        # The encoder is blocked by zero output weights on the first step only.
        head.zero_grad(set_to_none=True)
        with torch.no_grad():
            head.tactile_cross_attn.out_proj.weight.normal_(std=0.1)
        head(backbone, inputs).loss.backward()
        self.assertGreater(head.tactile_encoder.shared_cnn[0].weight.grad.abs().sum().item(), 0)
        seen.clear()
        queried.clear()
        result = head.eval().get_action(backbone, inputs).action_pred
        self.assertEqual(tuple(result.shape), (2, 2, 4))
        self.assertEqual(len(queried), head.num_inference_timesteps)
        for query, context in zip(queried, seen):
            torch.testing.assert_close(query, context[:, -2:])
        hook.remove()

    def test_legacy_checkpoint_strict_reload_and_invalid_config(self) -> None:
        legacy = self._head()
        config = vars(legacy.config).copy()
        config.pop("tactile_fusion_stage")
        restored = self.module.FlowmatchingActionHead(self.module.FlowmatchingActionHeadConfig(**config))
        restored.load_state_dict(legacy.state_dict(), strict=True)
        self.assertEqual(restored.tactile_fusion_stage, "pre_dit")
        with self.assertRaisesRegex(ValueError, "tactile_fusion_stage"):
            self._head(stage="invalid")
        with self.assertRaisesRegex(ValueError, "tactile_dropout_prob"):
            self._head(dropout=1.1)


if __name__ == "__main__":
    unittest.main()
