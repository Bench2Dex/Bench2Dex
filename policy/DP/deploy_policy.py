import numpy as np
import yaml

from .dp_model import DP


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


def _encode_camera(observation, candidates):
    root = _observation_root(observation)
    for name in candidates:
        if name in root and "rgb" in root[name]:
            return np.moveaxis(root[name]["rgb"], -1, 0) / 255
    raise KeyError(f"None of the camera observations are available: {candidates}")


def encode_obs(observation):
    obs = {}
    camera_candidates = {
        "head_cam":         ("head_camera", "cam_overhead"),
        "right_cam":        ("cam_wrist_right", "right_camera"),
        "left_cam":         ("cam_wrist_left", "left_camera"),
        "stereo_left_cam":  ("cam_stereo_left",),
        "stereo_right_cam": ("cam_stereo_right",),
    }
    for obs_key, candidates in camera_candidates.items():
        try:
            obs[obs_key] = _encode_camera(observation, candidates)
        except KeyError:
            pass

    joint_action = observation.get("joint_action", {})
    if "vector" in joint_action:
        obs["agent_pos"] = _maybe_select_active(joint_action["vector"])
    elif "qpos" in joint_action:
        obs["agent_pos"] = _maybe_select_active(joint_action["qpos"])
    else:
        raise KeyError("joint_action must contain 'vector' or 'qpos'")
    return obs


def get_model(usr_args):
    global _ACTIVE_DOF_INFO
    _ACTIVE_DOF_INFO = None
    if usr_args.get("use_active_dof", True):
        robot_key = usr_args.get("robot_key") or None
        if robot_key:
            from robots.active_dof_utils import get_active_dof_info
            _ACTIVE_DOF_INFO = get_active_dof_info(robot_key)
            print(
                f"[DP] Active DOF: robot={robot_key} "
                f"dims={_ACTIVE_DOF_INFO.active_dof} (full={_ACTIVE_DOF_INFO.full_dof})"
            )

    ckpt_file = usr_args.get("checkpoint_path")
    if not ckpt_file:
        raise ValueError("DP deploy requires 'checkpoint_path' (path to a .ckpt file).")

    training_config_path = usr_args.get("training_config_path")
    if not training_config_path:
        raise ValueError(
            "DP deploy requires 'training_config_path' (e.g. "
            "policy/DP/diffusion_policy/config/robot_dp_36_dex2scene_pretrained.yaml)."
        )

    with open(training_config_path, "r", encoding="utf-8") as f:
        model_training_config = yaml.safe_load(f)

    n_obs_steps = model_training_config["n_obs_steps"]
    n_action_steps = model_training_config["n_action_steps"]
    device = usr_args.get("device", "cuda:0")

    return DP(ckpt_file, n_obs_steps=n_obs_steps, n_action_steps=n_action_steps, device=device)


def _model_call(model, func_name, obs=None):
    if hasattr(model, "call"):
        return model.call(func_name=func_name, obs=obs)
    method = getattr(model, func_name)
    return method(obs) if obs is not None else method()


def eval(TASK_ENV, model, observation):
    """
    TASK_ENV: Task Environment Class, you can use this class to interact with the environment
    model: The model from 'get_model()' function
    observation: The observation about the environment
    """
    obs = encode_obs(observation)
    TASK_ENV.get_instruction()

    actions = _model_call(model, "get_action", obs)

    for action in actions:
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
        obs = encode_obs(observation)
        _model_call(model, "update_obs", obs)


def reset_model(model):
    _model_call(model, "reset_model")
