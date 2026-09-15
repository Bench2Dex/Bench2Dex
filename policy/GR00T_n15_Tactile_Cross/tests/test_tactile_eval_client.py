from __future__ import annotations

import sys
import argparse
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from policy.GR00T_n15_Tactile_Cross.tactile_eval_client import (
    REPO_ROOT,
    TactileEvalClient,
    main,
)
from utils.seed_policy import EVAL_BASE_SEED_DEFAULT


class TactileEvalClientTest(unittest.TestCase):
    def test_default_execution_horizon_and_cli_override(self) -> None:
        config_path = REPO_ROOT / "policy/GR00T_n15_Tactile_Cross/deploy_policy.yml"
        config = yaml.safe_load(config_path.read_text())
        config["task_name"] = "03_example"
        self.assertEqual(config["action_horizon"], 16)
        self.assertEqual(config["execution_horizon"], 4)
        command = TactileEvalClient(config).build_run_policy_command()
        self.assert_option(command, "--chunk-size", "4")

        config["run_policy_args"] = ["--chunk-size", "16"]
        command = TactileEvalClient(config).build_run_policy_command()
        parser = argparse.ArgumentParser()
        parser.add_argument("--chunk-size", type=int)
        args, _ = parser.parse_known_args(command[2:])
        self.assertEqual(args.chunk_size, 16)

    def assert_option(self, command: list[str], option: str, value: str) -> None:
        index = command.index(option)
        self.assertEqual(command[index + 1], value)

    def test_builds_tactile_runner_command_with_evaluation_options(self) -> None:
        client = TactileEvalClient(
            {
                "host": "127.0.0.1",
                "port": 9123,
                "task_name": "03_example",
                "robot_key": "multi_ur5_rh56dfx_with_flange",
                "model_path": "/checkpoints/03/robot/gr00t_n15_tactile_cross",
                "ckpt_name": "checkpoint-10000",
                "episode_steps": 800,
                "warmup_steps": 60,
                "num_episodes": 25,
                "start_episode": 26,
                "output_dir": "/results/03/cov_only",
                "append_output": True,
                "record_dir": "/records/03/cov_only",
                "record_all": True,
                "append_record_dir": True,
                "generalization_profile": "cov_only",
                "generalization_split": "unseen",
                "generalization_config": "configs/scene/generalization.yaml",
                "enable_generalization": True,
                "anchor_dir": "/anchors/03",
                "policy_name": "GR00T_n15_Tactile_Cross",
                "policy_display_name": "03/robot/gr00t_n15_tactile_cross",
                "headless": True,
                "seed": 123456,
                "run_policy_args": [
                    "--log-level",
                    "DEBUG",
                ],
            }
        )

        command = client.build_run_policy_command()

        self.assertEqual(command[0], sys.executable)
        self.assertEqual(
            command[1], "policy/GR00T_n15_Tactile_Cross/run_policy.py"
        )
        self.assertNotIn("run_policy.py", command[2:])
        for option, value in (
            ("--host", "127.0.0.1"),
            ("--port", "9123"),
            ("--task", "scenes/03_example.yaml"),
            ("--robot-key", "multi_ur5_rh56dfx_with_flange"),
            ("--ckpt-dir", "/checkpoints/03/robot/gr00t_n15_tactile_cross"),
            ("--ckpt-name", "checkpoint-10000"),
            ("--episode-steps", "800"),
            ("--warmup-steps", "60"),
            ("--num-episodes", "25"),
            ("--start-episode", "26"),
            ("--output-dir", "/results/03/cov_only"),
            ("--record-dir", "/records/03/cov_only"),
            ("--generalization-profile", "cov_only"),
            ("--generalization-split", "unseen"),
            ("--generalization-config", "configs/scene/generalization.yaml"),
            ("--anchor-dir", "/anchors/03"),
            ("--policy-name", "GR00T_n15_Tactile_Cross"),
            ("--policy-display-name", "03/robot/gr00t_n15_tactile_cross"),
            ("--seed", "123456"),
        ):
            self.assert_option(command, option, value)
        for flag in (
            "--enable-rgb",
            "--append-output",
            "--record-all",
            "--append-record-dir",
            "--enable-generalization",
            "--headless",
        ):
            self.assertIn(flag, command)
        self.assertEqual(
            command[-2:], ["--log-level", "DEBUG"]
        )

    def test_maps_false_active_dof_and_omits_optional_values(self) -> None:
        command = TactileEvalClient(
            {"task_name": "03_example", "use_active_dof": "false"}
        ).build_run_policy_command()

        self.assertIn("--no-active-dof", command)
        self.assertNotIn("--active-dof", command)
        self.assert_option(command, "--seed", str(EVAL_BASE_SEED_DEFAULT))
        for option in (
            "--robot-key",
            "--ckpt-dir",
            "--ckpt-name",
            "--episode-steps",
            "--warmup-steps",
            "--num-episodes",
            "--start-episode",
            "--output-dir",
            "--record-dir",
            "--generalization-profile",
            "--generalization-split",
            "--generalization-config",
            "--anchor-dir",
            "--anchor-hdf5",
            "--policy-name",
            "--policy-display-name",
        ):
            self.assertNotIn(option, command)

    def test_rejects_non_sequence_run_policy_args(self) -> None:
        for invalid_value in ("--headless", {"--headless": True}, 7):
            with self.subTest(value=invalid_value):
                client = TactileEvalClient(
                    {
                        "task_name": "03_example",
                        "run_policy_args": invalid_value,
                    }
                )

                with self.assertRaisesRegex(
                    ValueError,
                    "run_policy_args must be a list or tuple",
                ):
                    client.build_run_policy_command()

    @mock.patch("policy.GR00T_n15_Tactile_Cross.tactile_eval_client.subprocess.run")
    def test_run_executes_from_repository_root(self, run: mock.Mock) -> None:
        run.return_value.returncode = 17
        client = TactileEvalClient({"task_name": "03_example"})

        returncode = client.run()

        self.assertEqual(returncode, 17)
        run.assert_called_once_with(
            client.build_run_policy_command(), cwd=REPO_ROOT, check=False
        )

    @mock.patch(
        "policy.GR00T_n15_Tactile_Cross.tactile_eval_client.TactileEvalClient"
    )
    @mock.patch(
        "policy.GR00T_n15_Tactile_Cross.tactile_eval_client.RemotePolicyClient"
    )
    def test_main_preserves_yaml_endpoint_for_connection_and_runner(
        self,
        remote_client: mock.Mock,
        tactile_client: mock.Mock,
    ) -> None:
        tactile_client.return_value.run.return_value = 0
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "eval.yml"
            config_path.write_text(
                "task_name: 03_example\nhost: 10.0.0.8\nport: 9124\n",
                encoding="utf-8",
            )
            argv = [
                "tactile_eval_client.py",
                "--config",
                str(config_path),
                "--check-connection",
            ]

            with mock.patch.object(sys, "argv", argv):
                with self.assertRaises(SystemExit) as exit_context:
                    main()

        self.assertEqual(exit_context.exception.code, 0)
        remote_client.assert_called_once_with("10.0.0.8", 9124)
        remote_client.return_value.close.assert_called_once_with()
        loaded_config = tactile_client.call_args.args[0]
        self.assertEqual(loaded_config["host"], "10.0.0.8")
        self.assertEqual(loaded_config["port"], 9124)


if __name__ == "__main__":
    unittest.main()
