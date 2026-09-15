from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    SOURCE = Path(__file__).resolve().parents[1] / "src/gr00t/model/action_head/tactile_token_encoder.py"
    SPEC = importlib.util.spec_from_file_location("tactile_token_encoder_under_test", SOURCE)
    assert SPEC is not None and SPEC.loader is not None
    MODULE = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(MODULE)
    TactileTokenEncoder = MODULE.TactileTokenEncoder


@unittest.skipIf(torch is None, "PyTorch is required for tactile encoder runtime tests")
class TactileTokenEncoderTest(unittest.TestCase):
    def test_returns_four_spatial_tokens_per_site(self) -> None:
        encoder = TactileTokenEncoder(num_sites=2, output_dim=32)
        tokens = encoder(torch.zeros(3, 2, 1, 240, 240))
        self.assertEqual(tuple(tokens.shape), (3, 8, 32))

    def test_rejects_wrong_site_count_and_quantized_input(self) -> None:
        encoder = TactileTokenEncoder(num_sites=2, output_dim=32)
        with self.assertRaisesRegex(ValueError, "expected 2"):
            encoder(torch.zeros(1, 1, 1, 240, 240))
        with self.assertRaisesRegex(ValueError, "floating"):
            encoder(torch.zeros(1, 2, 1, 240, 240, dtype=torch.uint8))

    def test_configurable_grid_and_checkpoint_round_trip(self) -> None:
        encoder = TactileTokenEncoder(num_sites=2, output_dim=32, grid_size=4)
        maps = torch.rand(1, 2, 1, 32, 32)
        expected = encoder(maps)
        self.assertEqual(tuple(expected.shape), (1, 32, 32))
        restored = TactileTokenEncoder(num_sites=2, output_dim=32, grid_size=4)
        restored.load_state_dict(encoder.state_dict(), strict=True)
        torch.testing.assert_close(restored(maps), expected)
        with self.assertRaisesRegex(ValueError, "grid_size"):
            TactileTokenEncoder(num_sites=2, output_dim=32, grid_size=0)

    def test_declares_sensor_and_spatial_embeddings_without_pooling(self) -> None:
        names = set(dict(TactileTokenEncoder(num_sites=2, output_dim=32).named_modules()))
        self.assertIn("sensor_embedding", names)
        self.assertIn("spatial_embedding", names)
        self.assertNotIn("attention_pool", names)


if __name__ == "__main__":
    unittest.main()
