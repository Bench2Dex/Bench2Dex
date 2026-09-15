from __future__ import annotations

import unittest
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1] / "src/gr00t/model/action_head/flow_matching_action_head.py"


class CrossAttentionActionHeadSourceTest(unittest.TestCase):
    def test_fuses_tactile_only_after_action_encoder(self) -> None:
        text = SOURCE.read_text(encoding="utf-8")
        self.assertIn("self.tactile_cross_attn = (", text)
        self.assertIn("nn.MultiheadAttention(", text)
        self.assertIn("nn.init.zeros_(self.tactile_cross_attn.out_proj.weight)", text)
        self.assertIn("nn.init.zeros_(self.tactile_cross_attn.out_proj.bias)", text)
        self.assertIn("def _fuse_action_features", text)
        self.assertIn("action_features, tactile_tokens, tactile_keep_mask", text)
        self.assertIn("self._fuse_action_features(action_features, tactile_tokens)", text)
        self.assertNotIn("torch.cat((backbone_features, tactile_token), dim=1)", text)
        self.assertNotIn("(attention_mask, tactile_mask), dim=1", text)

    def test_configures_tactile_modules_and_whole_modality_dropout(self) -> None:
        text = SOURCE.read_text(encoding="utf-8")
        self.assertIn("self.tactile_encoder = (", text)
        self.assertIn("self.tactile_cross_attn = (", text)
        self.assertIn("tactile[drop_mask] = 0", text)
        self.assertIn("tactile_dropout_prob", text)
        self.assertIn("tactile_keep_mask", text)
        self.assertIn("tactile_delta = tactile_delta * tactile_keep_mask", text)

    def test_cross_mode_keeps_original_action_head_and_dit_trainable(self) -> None:
        text = SOURCE.read_text(encoding="utf-8")
        self.assertNotIn(
            "if self.config.use_tactile:\n"
            "            for p in self.parameters():\n"
            "                p.requires_grad = False",
            text,
        )
        self.assertNotIn(
            "if self.config.use_tactile:\n"
            "                self.state_encoder.eval()",
            text,
        )


if __name__ == "__main__":
    unittest.main()
