import sys
import numpy as np
import torch
import os
import pickle
import cv2
import time  # Add import for timestamp
import h5py  # Add import for HDF5
from datetime import datetime  # Add import for datetime formatting
from .act_policy import ACT
import copy
from argparse import Namespace


_ACTIVE_DOF_INFO = None


def _maybe_select_active(qpos):
    global _ACTIVE_DOF_INFO
    if _ACTIVE_DOF_INFO is None:
        return qpos
    arr = np.asarray(qpos, dtype=np.float32)
    if arr.shape[-1] != _ACTIVE_DOF_INFO.full_dof:
        return qpos
    from robots.active_dof_utils import select_active
    return select_active(arr, _ACTIVE_DOF_INFO).tolist()


def _resolve_dims(usr_args: dict) -> dict:
    """Auto-determine action_dim/state_dim from robot_key.

    Resolution order:
      1. Explicit robot_key in usr_args
      2. Auto-detect from ckpt_dir parent/grandparent directory name
      3. First robot subdirectory under ckpt_dir
      4. Fallback: HDF5-based detection (legacy)
    """
    global _ACTIVE_DOF_INFO
    _ACTIVE_DOF_INFO = None
    from robots.active_dof_utils import get_active_dof_info
    from robots import ROBOT_SPAWNERS
    use_active = usr_args.get("use_active_dof", True)
    robot_key = usr_args.get("robot_key") or None
    if robot_key is None:
        ckpt_dir = usr_args.get("ckpt_dir")
        if ckpt_dir:
            ckpt_path = os.path.normpath(str(ckpt_dir)).rstrip("/")
            # Check parent and grandparent directory names
            for _ in range(2):
                candidate = os.path.basename(ckpt_path)
                if candidate in ROBOT_SPAWNERS:
                    robot_key = candidate
                    break
                ckpt_path = os.path.dirname(ckpt_path)
            # If still not found, scan task-level dir for robot subdirectories
            if robot_key is None:
                import glob
                for entry in sorted(glob.glob(os.path.join(str(ckpt_dir), "*"))):
                    name = os.path.basename(entry)
                    if name in ROBOT_SPAWNERS and os.path.isdir(entry):
                        robot_key = name
                        break
            # Legacy fallback: HDF5-based detection
            if robot_key is None:
                from robots.active_dof_utils import robot_key_from_hdf5
                import glob
                hdf5s = sorted(glob.glob(os.path.join(str(ckpt_dir), "*.hdf5")))
                if hdf5s:
                    robot_key = robot_key_from_hdf5(hdf5s[0])
        if robot_key is None:
            return usr_args
    info = get_active_dof_info(robot_key)
    if use_active:
        _ACTIVE_DOF_INFO = info
    usr_args = dict(usr_args)
    usr_args["action_dim"] = info.active_dof if use_active else info.full_dof
    usr_args["state_dim"] = info.active_dof if use_active else info.full_dof
    print(f"[ACT] robot={robot_key} state_dim={usr_args['state_dim']} "
          f"(active={info.active_dof}, full={info.full_dof}, use_active={use_active})")
    return usr_args

def encode_obs(observation):
    head_cam = cv2.resize(observation["observation"]["head_camera"]["rgb"], (640, 480), interpolation=cv2.INTER_LINEAR)
    left_cam = cv2.resize(observation["observation"]["left_camera"]["rgb"], (640, 480), interpolation=cv2.INTER_LINEAR)
    right_cam = cv2.resize(observation["observation"]["right_camera"]["rgb"], (640, 480), interpolation=cv2.INTER_LINEAR)
    head_cam = np.moveaxis(head_cam, -1, 0)  # model does /255 internally
    left_cam = np.moveaxis(left_cam, -1, 0)
    right_cam = np.moveaxis(right_cam, -1, 0)
    stereo_left_cam = cv2.resize(observation["observation"]["cam_stereo_left"]["rgb"], (640, 480), interpolation=cv2.INTER_LINEAR)
    stereo_right_cam = cv2.resize(observation["observation"]["cam_stereo_right"]["rgb"], (640, 480), interpolation=cv2.INTER_LINEAR)
    stereo_left_cam = np.moveaxis(stereo_left_cam, -1, 0)
    stereo_right_cam = np.moveaxis(stereo_right_cam, -1, 0)

    # Support both legacy 14-dim (ALOHA) format and full 36-dim dex2bench qpos.
    jt = observation["joint_action"]
    if "qpos" in jt:
        # dex2bench format: flat qpos array already assembled by caller
        qpos = _maybe_select_active(jt["qpos"])
    else:
        # Legacy ALOHA format: left_arm(6) + left_gripper(1) + right_arm(6) + right_gripper(1)
        qpos = (list(jt.get("left_arm", [])) + [jt.get("left_gripper", 0)] +
                list(jt.get("right_arm", [])) + [jt.get("right_gripper", 0)])

    return {
        "head_cam": head_cam,
        "left_cam": left_cam,
        "right_cam": right_cam,
        "stereo_left_cam": stereo_left_cam,
        "stereo_right_cam": stereo_right_cam,
        "qpos": qpos,
    }

def get_model(usr_args):
    usr_args = _resolve_dims(usr_args)
    return ACT(usr_args, Namespace(**usr_args))


def eval(TASK_ENV, model, observation):
    obs = encode_obs(observation)

    # Get action from model
    actions = model.get_action(obs)
    for action in actions:
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
    return observation


def reset_model(model):
    # Reset temporal aggregation state if enabled
    if model.temporal_agg:
        model.all_time_actions = torch.zeros([
            model.max_timesteps,
            model.max_timesteps + model.num_queries,
            model.state_dim,
        ]).to(model.device)
        model.t = 0
        print("Reset temporal aggregation state")
    else:
        model.t = 0
