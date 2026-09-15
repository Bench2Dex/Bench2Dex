from __future__ import annotations

import ast
import unittest
from pathlib import Path


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "src/gr00t/model/action_head/tactile_token_encoder.py"
)


class TactileEncoderSourceTest(unittest.TestCase):
    def test_encoder_declares_required_components(self) -> None:
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
        self.assertIn("TactileTokenEncoder", classes)
        text = SOURCE.read_text(encoding="utf-8")
        for component in (
            "site_embedding",
            "pool_query",
            "attention_pool",
            "output_projection",
            "modality_embedding",
        ):
            self.assertIn(component, text)
        self.assertIn("[B,N,1,H,W]", text)


if __name__ == "__main__":
    unittest.main()
