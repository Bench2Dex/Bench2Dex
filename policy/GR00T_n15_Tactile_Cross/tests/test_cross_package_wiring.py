from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class CrossPackageWiringTest(unittest.TestCase):
    def test_cross_fork_has_isolated_sources_and_entrypoints(self) -> None:
        self.assertTrue((ROOT / "src/gr00t/model/action_head/flow_matching_action_head.py").is_file())
        for script in ("train.sh", "eval_double_env.sh", "eval_tasks_seq.sh"):
            self.assertTrue((ROOT / script).is_file())
        for source in ("train.sh", "eval_double_env.sh", "deploy_policy.py"):
            self.assertIn("GR00T_n15_Tactile_Cross", (ROOT / source).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
