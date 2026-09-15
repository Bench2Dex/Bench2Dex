from __future__ import annotations

import unittest
import inspect
import ast
from pathlib import Path
from unittest.mock import Mock

import numpy as np

from policy.GR00T_n15_Tactile_Cross.tactile_deployment import RemoteGR00TTactileDeployment


class TactileReplanningTest(unittest.TestCase):
    def test_remote_and_runner_default_to_four_step_execution(self):
        signature = inspect.signature(RemoteGR00TTactileDeployment)
        self.assertEqual(signature.parameters["action_horizon"].default, 4)
        path = Path(__file__).resolve().parents[1] / "run_policy.py"
        tree = ast.parse(path.read_text())
        chunk_option = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument" and node.args
            and isinstance(node.args[0], ast.Constant) and node.args[0].value == "--chunk-size"
        )
        default = next(item.value for item in chunk_option.keywords if item.arg == "default")
        self.assertEqual(ast.literal_eval(default), 4)

    def test_executes_only_requested_prefix_and_queries_latest_observation(self):
        policy = RemoteGR00TTactileDeployment.__new__(RemoteGR00TTactileDeployment)
        policy.chunk_size = 4
        policy.prompt = "task"
        policy._remote_observation = lambda obs, prompt: obs
        full_chunk = np.arange(16 * 3).reshape(16, 3)
        policy.client = Mock()
        policy.client.get_action.return_value = full_chunk
        for observation in ({"frame": 0}, {"frame": 4}):
            prefix = policy.get_action(observation)
            np.testing.assert_array_equal(prefix, full_chunk[:4])
            policy.client.get_action.assert_called_with(observation)
        self.assertEqual(policy.client.get_action.call_count, 2)
        policy.client.get_action.return_value = full_chunk[None]
        np.testing.assert_array_equal(policy.get_action({}), full_chunk[:4])
        policy.client.get_action.return_value = None
        self.assertIsNone(policy.get_action({}))
        policy.client.get_action.return_value = full_chunk[0]
        np.testing.assert_array_equal(policy.get_action({}), full_chunk[0])


if __name__ == "__main__":
    unittest.main()
