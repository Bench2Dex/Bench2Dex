from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "tactile_checkpoint.sh"


class TactileCheckpointTest(unittest.TestCase):
    def run_helper(self, *args: object) -> str:
        completed = subprocess.run(
            ["bash", str(HELPER), *(str(arg) for arg in args)],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    def make_valid_checkpoint(self, path: Path) -> Path:
        path.mkdir(parents=True)
        (path / "config.json").write_text("{}\n", encoding="utf-8")
        (path / "model-00001.safetensors").touch()
        return path

    def test_discovery_prefers_exact_tactile_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_root = Path(tmp)
            variant = self.make_valid_checkpoint(
                checkpoint_root / "03" / "robot_a" / "gr00t_n15_tactile_past"
            )
            exact = self.make_valid_checkpoint(
                checkpoint_root / "03" / "robot_b" / "gr00t_n15_tactile"
            )

            discovered = self.run_helper("discover", checkpoint_root, "03")

            self.assertNotEqual(str(variant), discovered)
            self.assertEqual(str(exact), discovered)

    def test_discovery_handles_glob_characters_in_root_and_robot_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_root = Path(tmp) / "checkpoints[root]?"
            expected = self.make_valid_checkpoint(
                checkpoint_root / "03" / "robot[arm]?" / "gr00t_n15_tactile"
            )

            discovered = self.run_helper("discover", checkpoint_root, "03")

            self.assertEqual(str(expected), discovered)

    def test_discovery_rejects_model_artifact_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_root = Path(tmp)
            candidate = checkpoint_root / "03" / "robot_a" / "gr00t_n15_tactile"
            candidate.mkdir(parents=True)
            (candidate / "config.json").write_text("{}\n", encoding="utf-8")
            (candidate / "model-fake.safetensors").mkdir()

            discovered = self.run_helper("discover", checkpoint_root, "03")

            self.assertEqual("", discovered)

    def test_discovery_accepts_valid_nested_checkpoint_with_incomplete_parent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_root = Path(tmp)
            nested = self.make_valid_checkpoint(
                checkpoint_root
                / "03"
                / "robot_a"
                / "gr00t_n15_tactile_past"
                / "checkpoint-10000"
            )

            discovered = self.run_helper("discover", checkpoint_root, "03")

            self.assertEqual(str(nested), discovered)

    def test_discovery_prefers_highest_numeric_nested_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_root = Path(tmp)
            parent = (
                checkpoint_root / "03" / "robot_a" / "gr00t_n15_tactile_past"
            )
            self.make_valid_checkpoint(parent / "checkpoint-9000")
            latest = self.make_valid_checkpoint(parent / "checkpoint-10000")

            discovered = self.run_helper("discover", checkpoint_root, "03")

            self.assertEqual(str(latest), discovered)

    def test_discovery_compares_nested_checkpoint_steps_with_leading_zeros(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_root = Path(tmp)
            parent = (
                checkpoint_root / "03" / "robot_a" / "gr00t_n15_tactile_past"
            )
            self.make_valid_checkpoint(parent / "checkpoint-08")
            latest = self.make_valid_checkpoint(parent / "checkpoint-010")

            discovered = self.run_helper("discover", checkpoint_root, "03")

            self.assertEqual(str(latest), discovered)

    def test_discovery_supports_arbitrarily_large_nested_checkpoint_steps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_root = Path(tmp)
            parent = (
                checkpoint_root / "03" / "robot_a" / "gr00t_n15_tactile_past"
            )
            self.make_valid_checkpoint(parent / "checkpoint-9000")
            latest = self.make_valid_checkpoint(
                parent / "checkpoint-18446744073709551616"
            )

            discovered = self.run_helper("discover", checkpoint_root, "03")

            self.assertEqual(str(latest), discovered)

    def test_discovery_ignores_invalid_higher_nested_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_root = Path(tmp)
            parent = (
                checkpoint_root / "03" / "robot_a" / "gr00t_n15_tactile_past"
            )
            latest_valid = self.make_valid_checkpoint(parent / "checkpoint-9000")
            (parent / "checkpoint-10000").mkdir()

            discovered = self.run_helper("discover", checkpoint_root, "03")

            self.assertEqual(str(latest_valid), discovered)

    def test_nested_checkpoint_path_resolves_robot_key(self) -> None:
        model_path = Path(
            "/checkpoints/03/multi_ur5_rh56dfx_with_flange/"
            "gr00t_n15_tactile_past/checkpoint-10000"
        )

        robot_key = self.run_helper("robot-key", model_path)

        self.assertEqual("multi_ur5_rh56dfx_with_flange", robot_key)

    def test_discovery_ignores_ordinary_gr00t_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_root = Path(tmp)
            self.make_valid_checkpoint(
                checkpoint_root / "03" / "robot_a" / "gr00t_n15"
            )

            discovered = self.run_helper("discover", checkpoint_root, "03")

            self.assertEqual("", discovered)


if __name__ == "__main__":
    unittest.main()
