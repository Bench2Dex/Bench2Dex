from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class CrossCommandCompatibilityTest(unittest.TestCase):
    def test_train_keeps_tag_and_writes_cross_checkpoint(self) -> None:
        text = (ROOT / "train.sh").read_text(encoding="utf-8")
        self.assertIn("--tag", text)
        self.assertIn("gr00t_n15_trunc_tactile_cross", text)
        self.assertIn("GR00T_n15_Tactile_Cross", text)

    def test_tactile_resolution_defaults_to_native_240(self) -> None:
        deploy_config = (ROOT / "deploy_policy.yml").read_text(encoding="utf-8")
        finetune = (ROOT / "scripts/gr00t_finetune.py").read_text(encoding="utf-8")
        dataset = (ROOT / "gr00t_hdf5_dataset.py").read_text(encoding="utf-8")
        self.assertIn("tactile_height: 240", deploy_config)
        self.assertIn("tactile_width: 240", deploy_config)
        self.assertIn("hdf5_tactile_height: int = 240", finetune)
        self.assertIn("hdf5_tactile_width: int = 240", finetune)
        self.assertIn("tactile_height: int = 240", dataset)
        self.assertIn("tactile_width: int = 240", dataset)

    def test_eval_keeps_user_flags_and_only_discovers_cross_checkpoints(self) -> None:
        text = (ROOT / "eval_double_env.sh").read_text(encoding="utf-8")
        for flag in ("--sii", "--model-path", "RUN_POLICY_EXTRA"):
            self.assertIn(flag, text)
        self.assertIn("gr00t_n15_tactile_cross", text)
        self.assertIn("GR00T_n15_Tactile_Cross", text)
        helper = (ROOT / "tactile_checkpoint.sh").read_text(encoding="utf-8")
        self.assertIn("gr00t_n15_tactile_cross", helper)


if __name__ == "__main__":
    unittest.main()
