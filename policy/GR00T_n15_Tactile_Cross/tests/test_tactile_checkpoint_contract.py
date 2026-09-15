from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from policy.GR00T_n15_Tactile_Cross import tactile_io


class TactileCheckpointContractTest(unittest.TestCase):
    def test_reads_complete_contract_from_config_json(self) -> None:
        self.assertTrue(hasattr(tactile_io, "read_tactile_checkpoint_schema"))
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp)
            (checkpoint / "config.json").write_text(
                json.dumps(
                    {
                        "action_head_cfg": {
                            "use_tactile": True,
                            "tactile_site_names": ["right_thumb", "left_thumb"],
                            "tactile_native_height": 240,
                            "tactile_native_width": 240,
                            "tactile_depth_key": "distance_along_normal_m",
                            "tactile_depth_unit": "m",
                            "tactile_d_max_m": 0.012,
                            "tactile_resolution_step": 2,
                        }
                    }
                ),
                encoding="utf-8",
            )
            schema = tactile_io.read_tactile_checkpoint_schema(checkpoint)
        self.assertEqual(schema.site_names, ("right_thumb", "left_thumb"))
        self.assertEqual(schema.image_shape, (240, 240))
        self.assertEqual(schema.resolution_step, 2)
        self.assertAlmostEqual(schema.d_max_m, 0.012)

    def test_rejects_incomplete_or_non_tactile_checkpoint(self) -> None:
        self.assertTrue(hasattr(tactile_io, "read_tactile_checkpoint_schema"))
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp)
            (checkpoint / "config.json").write_text(
                json.dumps({"action_head_cfg": {"use_tactile": False}}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "tactile"):
                tactile_io.read_tactile_checkpoint_schema(checkpoint)


if __name__ == "__main__":
    unittest.main()
