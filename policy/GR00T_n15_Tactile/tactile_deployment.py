"""Local Isaac-Sim deployment adapter for tactile GR00T checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from policy.GR00T_n15_Tactile.tactile_io import stack_tactile_mapping
from collector.tacmap_configs import ROBOT_KEY_TO_TACMAP_CFG
from robots.active_dof_utils import get_active_dof_info
from script.policy_rpc import RemotePolicyClient


CAMERA_NAMES = (
    "cam_stereo_left",
    "cam_stereo_right",
    "cam_right_wrist",
    "cam_left_wrist",
)


class GR00TTactileDeployment:
    """Expose tactile GR00T through the standalone tactile runner interface."""

    returns_action_chunks = True
    temporal_agg = False

    def __init__(
        self,
        model_path: str,
        *,
        robot_key: str,
        prompt: str,
        device: str = "cuda",
        action_horizon: int = 16,
        data_config: str | None = None,
    ) -> None:
        if not robot_key:
            raise ValueError("robot_key is required for tactile GR00T deployment")
        from policy.GR00T_n15_Tactile.deploy_policy import Dex2BenchGR00TPolicy

        args: dict[str, Any] = {
            "model_path": str(Path(model_path).expanduser()),
            "robot_key": robot_key,
            "device": device,
            "action_horizon": int(action_horizon),
            "use_active_dof": True,
        }
        if data_config:
            args["data_config"] = data_config
        self.model = Dex2BenchGR00TPolicy(args)
        self.robot_key = robot_key
        self.prompt = str(prompt or "perform the task")
        self.camera_names = CAMERA_NAMES
        self.site_names = self.model.tactile_schema.site_names
        self.state_dim = int(self.model.state_dim)
        self.chunk_size = int(self.model.action_horizon)
        self.action_horizon = self.chunk_size
        self.checkpoint_path = Path(model_path).expanduser()
        self.tactile_image_size = int(self.model.tactile_schema.image_shape[0])
        if self.model.tactile_schema.image_shape[0] != self.model.tactile_schema.image_shape[1]:
            raise ValueError("TacMapRig deployment requires square checkpoint-native tactile maps")
        self.tactile_resolution_step = 1
        self.tactile_max_distance_m = 0.015

    def resolve_tactile_sensor_settings(
        self,
        resolution_step: int | None,
        max_distance: float | None,
    ) -> tuple[int, float]:
        resolved_step = self.tactile_resolution_step if resolution_step is None else int(resolution_step)
        resolved_distance = self.tactile_max_distance_m if max_distance is None else float(max_distance)
        if resolved_step != self.tactile_resolution_step:
            raise ValueError(
                f"tactile resolution_step={resolved_step} does not match checkpoint "
                f"value {self.tactile_resolution_step}"
            )
        if not np.isclose(resolved_distance, self.tactile_max_distance_m, rtol=0.0, atol=1e-9):
            raise ValueError(
                f"tactile max_distance={resolved_distance} does not match checkpoint "
                f"value {self.tactile_max_distance_m}"
            )
        return resolved_step, resolved_distance

    def validate_runtime(self, robot_key: str, active_indices, active_joint_names) -> None:
        if str(robot_key) != self.robot_key:
            raise ValueError(f"runtime robot {robot_key!r} does not match checkpoint {self.robot_key!r}")
        if len(tuple(active_indices)) != self.state_dim:
            raise ValueError(
                f"runtime active DOF count {len(tuple(active_indices))} does not match checkpoint {self.state_dim}"
            )

    @staticmethod
    def _image(images: dict[str, Any], key: str) -> np.ndarray:
        if key not in images:
            raise KeyError(f"missing RGB image {key!r}")
        image = np.asarray(images[key])
        if image.ndim != 3 or image.shape[-1] < 3:
            raise ValueError(f"RGB image {key!r} must have shape [H,W,3+], got {image.shape}")
        return np.ascontiguousarray(image[..., :3], dtype=np.uint8)

    def get_action(self, observation: dict[str, Any]) -> np.ndarray:
        images = observation.get("images")
        tactile = observation.get("tactile")
        if not isinstance(images, dict) or not isinstance(tactile, dict):
            raise ValueError("observation must contain image and tactile mappings")
        qpos = np.asarray(observation.get("qpos"), dtype=np.float32).reshape(-1)
        if qpos.shape[0] != self.state_dim:
            raise ValueError(f"qpos dimension {qpos.shape[0]} does not match checkpoint {self.state_dim}")
        encoded = {
            "state.qpos": qpos[None, :],
            "annotation.human.action.task_description": [self.prompt],
            "video.stereo_left": self._image(images, "cam_stereo_left")[None, ...],
            "video.stereo_right": self._image(images, "cam_stereo_right")[None, ...],
            "video.right_wrist": self._image(images, "cam_right_wrist")[None, ...],
            "video.left_wrist": self._image(images, "cam_left_wrist")[None, ...],
            "tactile": stack_tactile_mapping(
                tactile,
                self.model.tactile_schema,
                output_shape=self.model.tactile_output_shape,
            ),
        }
        return self.model.get_action(encoded)

    def reset(self) -> None:
        return None


class RemoteGR00TTactileDeployment:
    """Isaac-side wrapper that sends RGB, qpos, language, and TacMap to a GR00T server."""

    returns_action_chunks = True
    uses_raw_observation = True
    temporal_agg = False

    def __init__(
        self,
        host: str,
        port: int,
        *,
        robot_key: str,
        prompt: str,
        action_horizon: int = 16,
        tactile_resolution_step: int = 1,
        tactile_max_distance: float = 0.015,
    ) -> None:
        cfg = ROBOT_KEY_TO_TACMAP_CFG.get(robot_key)
        if cfg is None:
            raise ValueError(f"TacMap is not configured for robot {robot_key!r}")
        dof = get_active_dof_info(robot_key)
        self.client = RemotePolicyClient(str(host), int(port))
        self.robot_key = robot_key
        self.prompt = str(prompt or "perform the task")
        self.camera_names = CAMERA_NAMES
        self.site_names = tuple(cfg.site_names)
        self.state_dim = int(dof.active_dof)
        self.chunk_size = int(action_horizon)
        self.action_horizon = self.chunk_size
        self.checkpoint_path = Path(f"remote://{host}:{port}")
        self.tactile_resolution_step = int(tactile_resolution_step)
        if self.tactile_resolution_step <= 0 or cfg.native_resolution % self.tactile_resolution_step != 0:
            raise ValueError(
                f"tactile_resolution_step={self.tactile_resolution_step} must divide {cfg.native_resolution}"
            )
        self.tactile_image_size = int(cfg.native_resolution // self.tactile_resolution_step)
        self.tactile_max_distance_m = float(tactile_max_distance)

    def resolve_tactile_sensor_settings(
        self, resolution_step: int | None, max_distance: float | None
    ) -> tuple[int, float]:
        step = self.tactile_resolution_step if resolution_step is None else int(resolution_step)
        distance = self.tactile_max_distance_m if max_distance is None else float(max_distance)
        if step != self.tactile_resolution_step:
            raise ValueError(f"runtime tactile resolution_step must be {self.tactile_resolution_step}")
        if not np.isclose(distance, self.tactile_max_distance_m, rtol=0.0, atol=1e-9):
            raise ValueError(f"runtime tactile max_distance must be {self.tactile_max_distance_m}")
        return step, distance

    def validate_runtime(self, robot_key: str, active_indices, active_joint_names) -> None:
        if str(robot_key) != self.robot_key:
            raise ValueError(f"runtime robot {robot_key!r} does not match {self.robot_key!r}")
        if len(tuple(active_indices)) != self.state_dim:
            raise ValueError("runtime active DOF count does not match tactile GR00T")

    @staticmethod
    def _remote_observation(observation: dict[str, Any], prompt: str) -> dict[str, Any]:
        images = observation.get("images")
        tactile = observation.get("tactile")
        if not isinstance(images, dict) or not isinstance(tactile, dict):
            raise ValueError("observation must contain image and tactile mappings")
        qpos = np.asarray(observation.get("qpos"), dtype=np.float32).reshape(-1)
        return {
            "joint_action": {"vector": qpos},
            "observation": {
                "cam_stereo_left": {"rgb": images["cam_stereo_left"]},
                "cam_stereo_right": {"rgb": images["cam_stereo_right"]},
                "cam_wrist_right": {"rgb": images["cam_right_wrist"]},
                "cam_wrist_left": {"rgb": images["cam_left_wrist"]},
            },
            "tactile": {"tacmap": tactile},
            "language": prompt,
        }

    def get_action(self, observation: dict[str, Any]):
        return self.client.get_action(self._remote_observation(observation, self.prompt))

    def update_obs(self, observation: dict[str, Any]):
        return self.client.update_obs(self._remote_observation(observation, self.prompt))

    def reset_model(self, seed: int | None = None) -> None:
        self.client.reset_model(seed=seed)

    def close(self) -> None:
        self.client.close()
