"""Dex2bench deployment adapter for the vendored Isaac-GR00T N1.5 policy."""

from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import Any

import numpy as np

POLICY_DIR = Path(__file__).resolve().parent
SRC_DIR = POLICY_DIR / "src"
REPO_ROOT = POLICY_DIR.parents[1]
for path in (SRC_DIR, REPO_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from gr00t.experiment.data_config import load_data_config  # noqa: E402
from gr00t.model.policy import Gr00tPolicy  # noqa: E402


DEFAULT_DATA_CONFIG = "policy.GR00T_n15.gr00t_dex2bench_config:Dex2BenchGR00TDataConfig"
DEFAULT_ACTION_KEY = "action.qpos"
CAMERA_MODE_VIDEO_KEYS = {
    "3cam": ["video.head", "video.right_wrist", "video.left_wrist"],
    "4cam": ["video.stereo_left", "video.stereo_right", "video.right_wrist", "video.left_wrist"],
}
CAMERA_MODE_ALIASES = {
    "3": "3cam",
    "three": "3cam",
    "three_cam": "3cam",
    "three-camera": "3cam",
    "4": "4cam",
    "four": "4cam",
    "four_cam": "4cam",
    "four-camera": "4cam",
}
_ACTIVE_DOF_INFO = None

CAMERA_OBS_ALIASES = {
    "video.head": ("head_camera", "cam_overhead"),
    "video.stereo_left": ("stereo_left_camera", "left_stereo_camera", "cam_stereo_left"),
    "video.stereo_right": ("stereo_right_camera", "right_stereo_camera", "cam_stereo_right"),
    "video.right_wrist": ("right_camera", "right_wrist_camera", "cam_wrist_right"),
    "video.left_wrist": ("left_camera", "left_wrist_camera", "cam_wrist_left"),
}


def _split_csv_env(value: str) -> list[str]:
    return [item.strip() for item in value.replace(",", " ").split() if item.strip()]


def _normalize_camera_mode(value: Any) -> str:
    mode = str(value or "4cam").strip().lower()
    return CAMERA_MODE_ALIASES.get(mode, mode)


def _video_keys_from_env() -> list[str]:
    explicit = os.environ.get("GR00T_VIDEO_KEYS")
    if explicit:
        return [key if key.startswith("video.") else f"video.{key}" for key in _split_csv_env(explicit)]
    mode = _normalize_camera_mode(os.environ.get("GR00T_CAMERA_MODE", "4cam"))
    if mode not in CAMERA_MODE_VIDEO_KEYS:
        raise ValueError(
            f"Unsupported GR00T_CAMERA_MODE={mode!r}; expected one of {sorted(CAMERA_MODE_VIDEO_KEYS)}"
        )
    return CAMERA_MODE_VIDEO_KEYS[mode]


def _as_uint8_rgb(image: Any) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[-1] < 3:
        raise ValueError(f"Expected HWC RGB image, got shape {arr.shape}")
    arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _read_camera(observation: dict[str, Any], *names: str) -> np.ndarray:
    camera_root = observation.get("observation", {})
    for name in names:
        camera = camera_root.get(name)
        if isinstance(camera, dict) and camera.get("rgb") is not None:
            return _as_uint8_rgb(camera["rgb"])
    raise KeyError(f"Missing RGB camera. Tried: {names}")


def _maybe_select_active(qpos: np.ndarray) -> np.ndarray:
    global _ACTIVE_DOF_INFO
    if _ACTIVE_DOF_INFO is None:
        return qpos
    if qpos.shape[-1] != _ACTIVE_DOF_INFO.full_dof:
        return qpos
    from robots.active_dof_utils import select_active
    return select_active(qpos, _ACTIVE_DOF_INFO).astype(np.float32, copy=False)


def _read_qpos(observation: dict[str, Any]) -> np.ndarray:
    joint_action = observation.get("joint_action", {})
    for key in ("vector", "qpos"):
        value = joint_action.get(key)
        if value is not None:
            return _maybe_select_active(np.asarray(value, dtype=np.float32).reshape(-1))
    for key in ("qpos", "agent_pos"):
        value = observation.get(key)
        if value is not None:
            return _maybe_select_active(np.asarray(value, dtype=np.float32).reshape(-1))
    raise KeyError("Missing joint state: expected joint_action.vector, qpos, or agent_pos")


def _optional_int(value: Any, name: str) -> int | None:
    if value is None or value == "" or value == "null":
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def _validate_local_checkpoint(model_path: str) -> None:
    path = Path(model_path).expanduser()
    if not path.exists():
        return
    metadata_path = path / "experiment_cfg" / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            "GR00T_n15 local model_path must be the finetuned checkpoint root containing "
            f"experiment_cfg/metadata.json, got: {path}. Do not pass the base HF model or "
            "a checkpoint subdirectory unless it includes experiment_cfg."
        )


def encode_obs(observation: dict[str, Any]) -> dict[str, Any]:
    """Map dex2bench raw remote observations to the GR00T modality schema."""

    instruction = (
        observation.get("language")
        or observation.get("instruction")
        or observation.get("raw_lang")
        or "perform the task"
    )
    qpos = _read_qpos(observation)
    encoded = {
        "state.qpos": qpos[None, :],
        "annotation.human.action.task_description": [str(instruction)],
    }
    for key in _video_keys_from_env():
        aliases = CAMERA_OBS_ALIASES.get(key)
        if aliases is None:
            suffix = key.removeprefix("video.")
            aliases = (suffix, f"{suffix}_camera", f"cam_{suffix}")
        encoded[key] = _read_camera(observation, *aliases)[None, ...]
    return encoded


class Dex2BenchGR00TPolicy:
    def __init__(self, usr_args: dict[str, Any]) -> None:
        global _ACTIVE_DOF_INFO
        _ACTIVE_DOF_INFO = None
        if usr_args.get("use_active_dof", True):
            robot_key = usr_args.get("robot_key") or None
            if robot_key:
                from robots.active_dof_utils import get_active_dof_info
                _ACTIVE_DOF_INFO = get_active_dof_info(robot_key)
                usr_args = dict(usr_args)
                usr_args["action_dim"] = _ACTIVE_DOF_INFO.active_dof
                usr_args["state_dim"] = _ACTIVE_DOF_INFO.active_dof
                print(
                    f"[GR00T] Active DOF: robot={robot_key} "
                    f"dims={_ACTIVE_DOF_INFO.active_dof} (full={_ACTIVE_DOF_INFO.full_dof})"
                )

        model_path = usr_args.get("model_path") or usr_args.get("ckpt_dir") or usr_args.get("ckpt_setting")
        if not model_path:
            raise ValueError("GR00T_n15 requires --model_path <checkpoint_or_hf_id>.")
        _validate_local_checkpoint(str(model_path))

        max_state_dim = _optional_int(usr_args.get("max_state_dim"), "max_state_dim")
        max_action_dim = _optional_int(usr_args.get("max_action_dim"), "max_action_dim")
        if max_state_dim is not None:
            os.environ["GR00T_MAX_STATE_DIM"] = str(max_state_dim)
        if max_action_dim is not None:
            os.environ["GR00T_MAX_ACTION_DIM"] = str(max_action_dim)
        camera_mode = usr_args.get("camera_mode")
        if camera_mode:
            os.environ["GR00T_CAMERA_MODE"] = _normalize_camera_mode(camera_mode)
        video_keys = usr_args.get("video_keys")
        if isinstance(video_keys, (list, tuple)):
            os.environ["GR00T_VIDEO_KEYS"] = ",".join(str(key) for key in video_keys)
        elif video_keys:
            os.environ["GR00T_VIDEO_KEYS"] = str(video_keys)

        data_config_name = usr_args.get("data_config") or DEFAULT_DATA_CONFIG
        data_config = load_data_config(str(data_config_name))
        modality_config = data_config.modality_config()
        modality_transform = data_config.transform()

        self.action_keys = list(usr_args.get("action_keys") or [DEFAULT_ACTION_KEY])
        self.action_dim = _optional_int(usr_args.get("action_dim"), "action_dim")
        self.state_dim = _optional_int(usr_args.get("state_dim"), "state_dim")
        if self.action_dim is None:
            self.action_dim = 36
        if self.state_dim is None:
            self.state_dim = self.action_dim
        self.action_horizon = int(usr_args.get("action_horizon") or 16)
        self.policy = Gr00tPolicy(
            model_path=str(model_path),
            modality_config=modality_config,
            modality_transform=modality_transform,
            embodiment_tag=str(usr_args.get("embodiment_tag") or "new_embodiment"),
            denoising_steps=int(usr_args.get("denoising_steps") or 4),
            device=str(usr_args.get("device") or "cuda"),
        )

    @staticmethod
    def _to_numpy_action(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.ndim != 2:
            raise ValueError(f"Expected action chunk shape (H, D), got {arr.shape}")
        return arr

    def get_action(self, obs: dict[str, Any]) -> np.ndarray:
        qpos = np.asarray(obs.get("state.qpos"), dtype=np.float32)
        if qpos.shape[-1] != self.state_dim:
            raise ValueError(f"GR00T state dim mismatch: got {qpos.shape[-1]}, expected {self.state_dim}")
        action_dict = self.policy.get_action(obs)
        chunks = []
        for key in self.action_keys:
            if key not in action_dict:
                raise KeyError(f"GR00T action output missing {key!r}; available: {list(action_dict)}")
            chunks.append(self._to_numpy_action(action_dict[key]))

        actions = np.concatenate(chunks, axis=-1) if len(chunks) > 1 else chunks[0]
        if actions.shape[-1] != self.action_dim:
            raise ValueError(
                f"GR00T action dim mismatch: got {actions.shape[-1]}, expected {self.action_dim}"
            )
        return np.ascontiguousarray(actions[: self.action_horizon], dtype=np.float32)


def get_model(usr_args: dict[str, Any]) -> Dex2BenchGR00TPolicy:
    return Dex2BenchGR00TPolicy(usr_args)


def reset_model(model: Dex2BenchGR00TPolicy) -> None:
    return None
