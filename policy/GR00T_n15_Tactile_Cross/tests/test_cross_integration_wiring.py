from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class CrossIntegrationWiringTest(unittest.TestCase):
    def test_checkpoint_serializes_raw_depth_contract(self) -> None:
        text = (ROOT / "scripts/gr00t_finetune.py").read_text(encoding="utf-8")
        for field in (
            "tactile_depth_key",
            "tactile_depth_unit",
            "tactile_d_max_m",
            "tactile_resolution_step",
            "tactile_dropout_prob",
        ):
            self.assertIn(field, text)

    def test_runtime_uses_raw_depth_not_quantized_tacmap(self) -> None:
        for relative_path in ("deploy_policy.py", "tactile_deployment.py", "run_policy.py"):
            text = (ROOT / relative_path).read_text(encoding="utf-8")
            self.assertIn("distance_along_normal_m", text, relative_path)
        self.assertIn('tactile_capture["distance_along_normal_m"]', (ROOT / "run_policy.py").read_text(encoding="utf-8"))
        self.assertIn('"distance_along_normal_m": tactile', (ROOT / "tactile_deployment.py").read_text(encoding="utf-8"))

    def test_remote_runtime_loads_sensor_contract_from_checkpoint(self) -> None:
        deployment = (ROOT / "tactile_deployment.py").read_text(encoding="utf-8")
        runner = (ROOT / "run_policy.py").read_text(encoding="utf-8")
        self.assertIn("read_tactile_checkpoint_schema", deployment)
        self.assertIn("model_path=args.ckpt_dir", runner)

    def test_runner_has_no_unsupported_attention_dump_option(self) -> None:
        runner = (ROOT / "run_policy.py").read_text(encoding="utf-8")
        self.assertNotIn("--dump-tactile-attention", runner)
        self.assertNotIn("_log_tactile_attention", runner)
        self.assertNotIn("_log_modality_attention", runner)


if __name__ == "__main__":
    unittest.main()
