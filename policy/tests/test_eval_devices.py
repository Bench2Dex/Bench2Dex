"""Device routing tests that do not start Isaac Sim or load checkpoints."""

import argparse
import ast
from pathlib import Path
import subprocess
import unittest

from utils.eval_devices import add_eval_device_arguments, normalize_eval_device_overrides
from script.eval_policy_client import Dex2SceneLiveEnv
from policy.GR00T_n15_Tactile.tactile_eval_client import TactileEvalClient
from policy.GR00T_n15_Tactile_Cross.tactile_eval_client import TactileEvalClient as CrossClient


ROOT = Path(__file__).resolve().parents[2]
RUNNERS = (
    "run_policy.py",
    "policy/ACT-Tactile/tactile_run_policy.py",
    "policy/GR00T_n15_Tactile/run_policy.py",
    "policy/GR00T_n15_Tactile_Cross/run_policy.py",
)


def device_parser(tactile=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")  # AppLauncher registration
    add_eval_device_arguments(parser, tactile=tactile)
    return parser


class EvalDeviceTests(unittest.TestCase):
    def test_defaults_and_independent_overrides(self):
        parser = device_parser(tactile=True)
        args = parser.parse_args([])
        self.assertEqual((args.device, args.policy_device, args.tactile_device),
                         ("cpu", "cuda:0", "cuda:0"))
        for flag in ("--device", "--sim-device"):
            args = parser.parse_args([flag, "cuda:1", "--policy-device", "cuda:2"])
            self.assertEqual((args.device, args.policy_device, args.tactile_device),
                             ("cuda:1", "cuda:2", "cuda:0"))

    def test_all_clients_default_to_cpu_without_changing_model_device(self):
        for client_type in (Dex2SceneLiveEnv, TactileEvalClient, CrossClient):
            for policy_name in ("ACT", "DP", "pi05", "GR00T_n15", "GR00T_XE"):
                with self.subTest(client=client_type, policy=policy_name):
                    config = {"task_name": "42_trash_disposal", "device": "cuda:2",
                              "policy_name": policy_name}
                    command = client_type(config).build_run_policy_command()
                    args, _ = device_parser(tactile=True).parse_known_args(command[2:])
                    self.assertEqual(args.device, "cpu")
                    self.assertEqual(config["device"], "cuda:2")

    def test_config_and_last_cli_override_win(self):
        for client_type in (Dex2SceneLiveEnv, TactileEvalClient, CrossClient):
            config = {"task_name": "42_trash_disposal", "sim_device": "cuda:1",
                      "tactile_device": "cuda:2"}
            command = client_type(config).build_run_policy_command()
            args, _ = device_parser(tactile=True).parse_known_args(command[2:])
            self.assertEqual(args.device, "cuda:1")
            if client_type is not Dex2SceneLiveEnv:
                self.assertEqual(args.tactile_device, "cuda:2")
            config["run_policy_args"] = ["--device", "cpu"]
            command = client_type(config).build_run_policy_command()
            args, _ = device_parser(tactile=True).parse_known_args(command[2:])
            self.assertEqual(args.device, "cpu")

    def test_client_device_alias_does_not_rewrite_values(self):
        self.assertEqual(normalize_eval_device_overrides(
            ["--device", "cpu", "--label", "--device"]),
            ["--sim_device", "cpu", "--label", "--device"])
        self.assertEqual(normalize_eval_device_overrides(None), [])

    def test_all_runners_use_shared_defaults_and_separate_model_device(self):
        for runner in RUNNERS:
            with self.subTest(runner=runner):
                source = (ROOT / runner).read_text()
                tree = ast.parse(source)
                self.assertIn("add_eval_device_arguments(parser", source)
                self.assertIn("device=args_cli.device", source)
                for node in tree.body:
                    if isinstance(node, ast.FunctionDef) and node.name in (
                        "_load_act_policy", "_load_dp_policy", "_load_tactile_policy"
                    ):
                        snippet = ast.get_source_segment(source, node)
                        self.assertNotIn("args.device", snippet)
                if runner != "run_policy.py":
                    self.assertIn("_require_tactile_gpu(args_cli.tactile_device)", source)
                    self.assertIn("compute_device=args.tactile_device", source)

    def test_previously_rejecting_wrappers_parse_cpu_and_gpu(self):
        scripts = (
            "ACT/eval_double_env.sh", "ACT-Tactile/act/eval_double_env.sh",
            "GR00T_XE/eval_double_env.sh", "pi05/eval_double_env.sh",
            "pi05/eval_all_channels.sh",
        )
        for script in scripts:
            source = (ROOT / "policy" / script).read_text()
            self.assertIn("SIM_DEVICE=cpu", source)
            start = source.index('while [[ $# -gt 0 ]]; do')
            end = source.index('\ndone', start) + len('\ndone')
            # Exercise only option parsing, without launching servers or creating outputs.
            shell = ('set -eu\nSIM_DEVICE=cpu\n'
                     'require_value() { [[ -n "${2:-}" && "$2" != --* ]]; }\n'
                     + source[start:end] + '\nprintf "%s" "$SIM_DEVICE"\n')
            for flags, expected in (([], "cpu"), (["--device", "cpu"], "cpu"),
                                    (["--device=cuda:0"], "cuda:0"),
                                    (["--sim-device", "cuda:1"], "cuda:1")):
                with self.subTest(script=script, flags=flags):
                    result = subprocess.run(["bash", "-c", shell, "test", *flags],
                                            capture_output=True, text=True, check=True)
                    self.assertEqual(result.stdout, expected)


if __name__ == "__main__":
    unittest.main()
