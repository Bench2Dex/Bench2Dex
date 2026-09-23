"""Check deployment horizon propagation without loading model weights or JAX."""

import dataclasses
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import ModuleType
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    action_horizon: int = 50
    action_dim: int = 4


@dataclasses.dataclass(frozen=True)
class DataConfig:
    state_dim: int = 4


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig()


def load_module(filename):
    path = Path(__file__).resolve().parents[1] / filename
    spec = importlib.util.spec_from_file_location(f"pi05_test_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ActionHorizonTest(unittest.TestCase):
    def setUp(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.checkpoint = Path(temp_dir.name)
        (self.checkpoint / "assets" / "test_robot").mkdir(parents=True)
        self.base_config = TrainConfig()
        self.create_policy = mock.Mock(side_effect=self.fake_policy)
        policies = ModuleType("openpi.policies")
        policies.policy_config = SimpleNamespace(create_trained_policy=self.create_policy)
        training = ModuleType("openpi.training")
        training.config = SimpleNamespace(get_config=mock.Mock(return_value=self.base_config))
        # Stub only the external model loader; exercise the real deployment wrapper.
        modules = {"openpi": ModuleType("openpi"), "openpi.policies": policies, "openpi.training": training}
        with mock.patch.dict(sys.modules, modules), mock.patch.object(sys, "path", sys.path.copy()):
            model_module = load_module("pi_model.py")
            with mock.patch.dict(sys.modules, {"pi_model": model_module}):
                self.deploy = load_module("deploy_policy.py")

    @staticmethod
    def fake_policy(config, checkpoint_path, **kwargs):
        return SimpleNamespace(
            infer=lambda obs: {
                "actions": np.zeros((config.model.action_horizon, config.model.action_dim), dtype=np.float32)
            }
        )

    def make_model(self, **overrides):
        return self.deploy.get_model({
            "train_config_name": "pi05_base_dex2bench_full",
            "checkpoint_path": str(self.checkpoint),
            "use_active_dof": False,
            **overrides,
        })

    def test_training_override_changes_generation_before_truncation(self):
        model = self.make_model(train_action_horizon="20", eval_action_horizon="20", action_dim=6)
        config = self.create_policy.call_args.args[0]
        self.assertEqual(config.model.action_horizon, 20)
        self.assertEqual(config.model.action_dim, 6)
        self.assertEqual(config.data.state_dim, 6)
        model.observation_window = {}
        self.assertEqual(model.get_action().shape, (20, 6))
        self.assertEqual(model.eval_action_horizon, 20)
        self.assertEqual(self.base_config.model.action_horizon, 50)

    def test_missing_horizon_arguments_default_to_twenty(self):
        model = self.make_model()
        self.assertEqual(self.create_policy.call_args.args[0].model.action_horizon, 20)
        self.assertEqual(model.eval_action_horizon, 20)

    def test_prediction_and_execution_horizons_remain_independent(self):
        model = self.make_model(train_action_horizon=50, eval_action_horizon=20)
        model.observation_window = {}
        self.assertEqual(model.get_action().shape, (50, 4))
        self.assertEqual(model.eval_action_horizon, 20)

    def test_invalid_horizons_rejected_before_loading_weights(self):
        for prediction, execution in ((0, 20), (-1, 20), (20, 0), (20, -1), (20, 21)):
            with self.subTest(prediction=prediction, execution=execution):
                with self.assertRaises(ValueError):
                    self.make_model(train_action_horizon=prediction, eval_action_horizon=execution)
        self.create_policy.assert_not_called()

    def test_deployment_executes_twenty_generated_actions(self):
        model = self.make_model(train_action_horizon=20, eval_action_horizon=20)
        observation = {
            "observation": {
                camera: {"rgb": np.zeros((2, 2, 3), dtype=np.uint8)}
                for camera in ("cam_stereo_left", "cam_stereo_right", "cam_wrist_left", "cam_wrist_right")
            },
            "joint_action": {"vector": np.zeros(4)},
        }
        task = SimpleNamespace(
            get_instruction=lambda: "test task",
            get_obs=lambda: observation,
            take_action=mock.Mock(),
        )
        self.deploy.eval(task, model, observation)
        self.assertEqual(model.get_action().shape, (20, 4))
        self.assertEqual(task.take_action.call_count, 20)


if __name__ == "__main__":
    unittest.main()
