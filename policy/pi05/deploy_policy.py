"""Pi0.5 deploy entrypoint for the dex2bench server/client eval flow."""

import os
import sys

import numpy as np

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
if parent_directory not in sys.path:
    sys.path.append(parent_directory)

from pi_model import PI0  # noqa: E402


_STEREO_LEFT_CANDIDATES = ("cam_stereo_left",)
_STEREO_RIGHT_CANDIDATES = ("cam_stereo_right",)
_WRIST_LEFT_CANDIDATES = ("cam_wrist_left", "left_camera")
_WRIST_RIGHT_CANDIDATES = ("cam_wrist_right", "right_camera")

_ACTIVE_DOF_INFO = None


def _maybe_select_active(qpos):
    global _ACTIVE_DOF_INFO
    if _ACTIVE_DOF_INFO is None:
        return qpos
    arr = np.asarray(qpos, dtype=np.float32)
    if arr.shape[-1] != _ACTIVE_DOF_INFO.full_dof:
        return qpos
    from robots.active_dof_utils import select_active
    return select_active(arr, _ACTIVE_DOF_INFO)


def _observation_root(observation):
    return observation.get("observation", observation)


def _pick_camera(root, candidates):
    for name in candidates:
        if name in root and "rgb" in root[name]:
            return root[name]["rgb"]
    raise KeyError(f"None of the camera observations are available: {candidates}")


def encode_obs(observation):
    root = _observation_root(observation)
    input_rgb_arr = [
        _pick_camera(root, _STEREO_LEFT_CANDIDATES),
        _pick_camera(root, _STEREO_RIGHT_CANDIDATES),
        _pick_camera(root, _WRIST_LEFT_CANDIDATES),
        _pick_camera(root, _WRIST_RIGHT_CANDIDATES),
    ]

    joint_action = observation.get("joint_action", {})
    if "vector" in joint_action:
        input_state = np.asarray(_maybe_select_active(joint_action["vector"]))
    elif "qpos" in joint_action:
        input_state = np.asarray(_maybe_select_active(joint_action["qpos"]))
    else:
        raise KeyError("joint_action must contain 'vector' or 'qpos'")
    return input_rgb_arr, input_state


def _optional_int(value, name):
    if value is None or value == "" or value == "null":
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def _optional_bool(value, default: bool = True) -> bool:
    if value is None or value == "" or value == "null":
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    raise ValueError(f"use_active_dof must be a boolean, got {value!r}")


def get_model(usr_args):
    global _ACTIVE_DOF_INFO
    _ACTIVE_DOF_INFO = None
    train_config_name = usr_args.get("train_config_name")
    if not train_config_name:
        raise ValueError("Pi05 deploy requires 'train_config_name' (e.g. pi05_base_dex2bench_lora).")

    checkpoint_path = usr_args.get("checkpoint_path")
    if not checkpoint_path:
        raise ValueError(
            "Pi05 deploy requires 'checkpoint_path' (absolute step dir, e.g. "
            "outputs/logs/pi05/pi05_base_dex2bench_lora/<exp>/30000)."
        )

    eval_action_horizon = int(usr_args.get("eval_action_horizon", 50))
    action_dim = _optional_int(usr_args.get("action_dim"), "action_dim")
    state_dim = _optional_int(usr_args.get("state_dim"), "state_dim")

    if _optional_bool(usr_args.get("use_active_dof"), True):
        robot_key = usr_args.get("robot_key") or None
        if robot_key:
            from robots.active_dof_utils import get_active_dof_info
            info = get_active_dof_info(robot_key)
            _ACTIVE_DOF_INFO = info
            action_dim = info.active_dof
            state_dim = info.active_dof
            print(f"[pi05] Active DOF: robot={robot_key} dims={info.active_dof} (full={info.full_dof})")

    if action_dim is not None and state_dim is None:
        state_dim = action_dim
    return PI0(
        train_config_name,
        checkpoint_path,
        eval_action_horizon,
        action_dim=action_dim,
        state_dim=state_dim,
    )


def eval(TASK_ENV, model, observation):
    if model.observation_window is None:
        instruction = TASK_ENV.get_instruction()
        model.set_language(instruction)

    input_rgb_arr, input_state = encode_obs(observation)
    model.update_observation_window(input_rgb_arr, input_state)

    actions = model.get_action()[: model.eval_action_horizon]

    for action in actions:
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
        input_rgb_arr, input_state = encode_obs(observation)
        model.update_observation_window(input_rgb_arr, input_state)


def reset_model(model):
    model.reset_observation_windows()
