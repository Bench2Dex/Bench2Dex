"""GR00T XE deployment: 加载 checkpoint, 输出可直接使用的关节角 action.

用法:
  model = get_model(usr_args)
  encoded = encode_obs(raw_obs)       # qpos → FK → 统一 64D state + 视频
  action = model.get_action(encoded)  # 推理 → IK arm + hand 反向映射 → 关节角

encode_obs 兼容两种观测格式:
  TACTILE:  observation["robot_key"], observation["qpos"], observation["images"][cam_id].rgb
  REMOTE:   observation["joint_action"]["qpos"], observation["observation"][cam_id]["rgb"]

get_action 使用 Gr00tPolicy wrapper, 自动处理以下 transform:
  VideoToTensor → VideoCrop → VideoResize → VideoColorJitter → VideoToNumpy
  → StateActionToTensor → StateActionTransform → ConcatTransform → GR00TTransform
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

_GR00T_SRC = Path(__file__).resolve().parents[1] / "GR00T_n15" / "src"
if str(_GR00T_SRC) not in sys.path:
    sys.path.insert(0, str(_GR00T_SRC))

# Module-level robot_key cache — set by get_model, used by encode_obs for REMOTE
_REMOTE_ROBOT_KEY: str = ""

# How much of the model's [H, D] chunk actually reaches the evaluator.
#
# run_policy.py treats whatever get_action returns as a queue: it extends
# _queued_actions with every row and pops one action per POLICY_STRIDE=3 physics
# steps, re-planning only once the queue drains.  n15 returns its whole
# [16, D] chunk there and is played open loop; this policy used to return a
# single joint vector, so _normalize_chunk wrapped it as a 1-element list and
# the evaluator re-planned through the IK at 20 Hz -- a different controller
# from the one n15 scored 8/50 with, which is why the offline comparison of the
# two never predicted their online behaviour.
#   single : old behaviour (1 joint vector, re-plan every policy tick)
#   chunk  : IK every horizon step, return [H, full_dof] -> same open-loop
#            execution as n15
# Default is "single" so nothing changes for anyone not running this
# experiment; set GR00T_XE_ACTION_MODE=chunk to opt in.
_ACTION_MODES = ("single", "chunk")


def _action_mode() -> str:
    mode = os.environ.get("GR00T_XE_ACTION_MODE", "single").strip().lower()
    if mode not in _ACTION_MODES:
        raise ValueError(
            f"GR00T_XE_ACTION_MODE={mode!r} is not one of {_ACTION_MODES}"
        )
    return mode


def _video_keys() -> list[str]:
    return ["video.stereo_left", "video.stereo_right", "video.right_wrist", "video.left_wrist"]


def _hdf5_camera_map() -> dict[str, str]:
    return {"stereo_left": "cam_stereo_left", "stereo_right": "cam_stereo_right",
            "right_wrist": "cam_wrist_right", "left_wrist": "cam_wrist_left"}


def _read_camera_remote(observation: dict[str, Any], name: str) -> np.ndarray:
    obs = observation.get("observation", {})
    cam = obs.get(name)
    if cam is None:
        raise ValueError(f"Camera not found in REMOTE observation: {name}")
    rgb = cam.get("rgb")
    if rgb is None:
        raise ValueError(f"Camera '{name}' has no RGB data in REMOTE observation")
    return np.asarray(rgb, dtype=np.uint8)


def _read_camera_tactile(observation: dict[str, Any], name: str) -> np.ndarray:
    images = observation.get("images", {})
    frame = images.get(name)
    if frame is None:
        raise ValueError(f"Camera not found: {name}")
    rgb = getattr(frame, "rgb", None) if hasattr(frame, "rgb") else (
        frame.get("rgb") if isinstance(frame, dict) else None
    )
    if rgb is None:
        raise ValueError(f"Camera '{name}' has no RGB data")
    return np.asarray(rgb, dtype=np.uint8)


def _is_remote_format(observation: dict[str, Any]) -> bool:
    return "observation" in observation and "joint_action" in observation


def encode_obs(observation: dict[str, Any]) -> dict[str, Any]:
    """将原始观测转为 GR00T 模型输入格式.

    输出格式兼容 Gr00tPolicy.get_action 的输入要求:
      - video.stereo_left: [T, H, W, C] uint8
      - video.stereo_right: [T, H, W, C] uint8
      - video.right_wrist: [T, H, W, C] uint8
      - video.left_wrist: [T, H, W, C] uint8
      - state.qpos: [T, D] float32

    语言条件不在这里产出: ``instruction`` 由
    ``policy_sessions._prepare_observation`` 挂在 obs 上, 由 ``get_action()``
    映射成 ``annotation.human.action.task_description`` (GR00TTransform 认的 key)。
    """
    from .ik_arm_converter import XEStateActionConverter

    remote = _is_remote_format(observation)

    # 1. robot_key
    if remote:
        robot_key = str(observation.get("robot_key", "") or _REMOTE_ROBOT_KEY)
        if not robot_key:
            raise ValueError(
                "REMOTE observation missing 'robot_key'. "
                "Set it via get_model(usr_args) or add to observation."
            )
    else:
        robot_key = str(observation.get("robot_key", ""))
        if not robot_key:
            raise ValueError("observation missing 'robot_key'")

    # 2. qpos — expand active_dof → full_dof for FK + hand mapping
    if remote:
        qpos_raw = np.asarray(
            observation["joint_action"]["qpos"], dtype=np.float64
        ).reshape(-1)
    else:
        qpos_raw = np.asarray(observation["qpos"], dtype=np.float64).reshape(-1)

    from robots.active_dof_utils import expand_to_full, get_active_dof_info

    adi = get_active_dof_info(robot_key)
    if len(qpos_raw) == adi.active_dof:
        qpos_full = expand_to_full(qpos_raw, adi)
    elif len(qpos_raw) == adi.full_dof:
        qpos_full = qpos_raw
    else:
        raise ValueError(
            f"qpos_raw has {len(qpos_raw)} DOF, expected {adi.active_dof} (active) "
            f"or {adi.full_dof} (full) for robot_key={robot_key}"
        )

    # 3. FK → 统一 64D state (uses full_dof qpos)
    converter = XEStateActionConverter()
    unified_state = converter.qpos_to_unified(qpos_full, robot_key)

    # 4. 视频 — 按 Gr00tPolicy 期望的格式: video.<key> → [T, H, W, C] uint8
    out = {}
    for gr00t_key in _video_keys():
        sub_key = gr00t_key.replace("video.", "")
        hdf5_cam = _hdf5_camera_map().get(sub_key, sub_key)
        if remote:
            img = _read_camera_remote(observation, hdf5_cam)
        else:
            img = _read_camera_tactile(observation, hdf5_cam)
        out[gr00t_key] = np.asarray(img, dtype=np.uint8)[None, ...]  # [T=1, H, W, C]

    # 5. 语言: 不加在这里 —— encode_obs 可能被无 instruction 的场景调用,
    #    真正的映射在 get_action() 里 (那里能拿到 policy_sessions 挂的 instruction)。

    out["state.qpos"] = unified_state.astype(np.float32)[None, :]  # [T=1, D]
    # Forward qpos_raw and robot_key for get_action() IK (popped before transform)
    out["qpos_raw"] = qpos_full.astype(np.float64)
    out["robot_key"] = robot_key
    return out


class GR00TXEPolicy:
    """加载 GR00T XE checkpoint, 使用 Gr00tPolicy wrapper 处理 transform 流水线."""

    def __init__(self, model_path: str, robot_key: str) -> None:
        os.environ.setdefault("GR00T_MAX_STATE_DIM", "64")
        os.environ.setdefault("GR00T_MAX_ACTION_DIM", "64")
        os.environ.setdefault("GR00T_CAMERA_MODE", "4cam")

        self.robot_key = robot_key
        self.state_dim = 64
        self.action_dim = 64
        self.action_horizon = 16

        print(f"[GR00T_XE] Loading model from {model_path}", flush=True)

        # 使用 Gr00tPolicy wrapper, 自动处理 transform 流水线
        from gr00t.model.policy import Gr00tPolicy  # noqa: E402
        from gr00t.data.schema import EmbodimentTag  # noqa: E402
        from gr00t.experiment.data_config import load_data_config  # noqa: E402

        # 加载数据配置 (xe_config.py 定义了 transform 流水线)
        data_config_path = "policy.GR00T_XE.xe_config:Dex2BenchXEDataConfig"
        data_config_cls = load_data_config(data_config_path)
        modality_configs = data_config_cls.modality_config()
        transforms = data_config_cls.transform()
        transforms.eval()

        self.policy = Gr00tPolicy(
            model_path=model_path,
            embodiment_tag=EmbodimentTag("new_embodiment"),
            modality_config=modality_configs,
            modality_transform=transforms,
            denoising_steps=4,
        )
        self.policy.model.eval()

        # Gr00tPolicy.__init__ has just normalized the pipeline with whatever
        # metadata.json held -- i.e. the statistics the mixture merged across all
        # 26 tasks, which is NOT what these weights were trained against (hand
        # slots in particular get dragged toward 0 by the 11 robots that do not
        # author them).  Re-apply the artifact the checkpoint carries.  It must be
        # there: falling back to the merged blob is the bug this removes.
        from xe_norm_stats import reapply_for_deploy

        reapply_for_deploy(
            transforms,
            Path(model_path) / "experiment_cfg",
            "new_embodiment",
            robot_key,
        )
        # reapply_for_deploy rebuilds the normalizers ON THE OBJECT IT IS GIVEN,
        # and Gr00tPolicy keeps the pipeline by reference (policy.py:
        # `self._modality_transform = modality_transform`).  That holds today; if
        # it ever becomes a copy, the fix would silently stop reaching inference
        # -- exactly the 坑1 failure mode -- so assert the identity rather than
        # trust it.
        assert self.policy.modality_transform is transforms, (
            "[XE-NORM] Gr00tPolicy holds a different transform object than the one "
            "reapply_for_deploy corrected; inference would denormalize with the "
            "mixture-merged statistics"
        )

        self.device = self.policy.device
        print(f"[GR00T_XE] Model loaded on {self.device}, robot={robot_key}", flush=True)

    def get_action(self, obs: dict[str, Any]) -> np.ndarray:
        """返回可直接发送到 Isaac Sim 的关节角 action (full DOF).

        obs 必须是 encode_obs() 的输出, 包含:
          - video.<key>, state.qpos
          - qpos_raw, robot_key
        """
        from .ik_arm_converter import XEStateActionConverter

        # 提取 IK 需要的额外信息, 不传给 Gr00tPolicy (否则 transform 会报错)
        if "qpos_raw" not in obs:
            raise ValueError(
                "get_action() requires 'qpos_raw' in obs (full_dof joint positions). "
                "Make sure encode_obs() was called first."
            )
        qpos_raw = np.asarray(obs.pop("qpos_raw"), dtype=np.float64).reshape(-1)
        robot_key = obs.pop("robot_key", self.robot_key)

        # 语言条件: policy_sessions._prepare_observation 把场景描述挂在
        # "instruction"/"language"/"raw_lang" 上, 但 GR00TTransform 只认
        # annotation.* 开头的 key (check_keys_and_batch_size), 认不到就退回
        # default_instruction "Perform the default behavior." —— 和训练时喂的
        # 语言完全对不上 (训练走 gr00t_hdf5_dataset.__getitem__ →
        # get_step_data() 的 ANNOTATION_KEY, 见 gr00t_hdf5_dataset.py:59/614)。
        #
        # 值必须是 **0-d ndarray** (np.array(instruction)) —— 裸 str 和 [str]
        # 都不行, 两者都实测踩过坑:
        #   * 裸 str: get_action 的 unsqueeze_dict_values 只展开 ndarray/list,
        #     str 原样留下, 随后的 np.array(v) 把它变成 **0-d** 数组; 而 eval
        #     路径上 ConcatTransform 把 video 抬到 6 维, GR00TTransform 因此走
        #     apply_batch → tree.map_structure(lambda x: x[i], data) 对 0-d 报
        #     "too many indices for array: array is 0-dimensional, but 1 were indexed"
        #     (2026-09-12 真跑挂掉过, 见 zyd/probe_annotation_batched.py 用例 1)。
        #   * [str] / np.array([str]): 被展开成 (1,1), x[0] 取出来还是数组,
        #     _prepare_language 的 isinstance(list) 分支不再触发,
        #     VLM 看到的是带方括号引号的 "['<描述>']"
        #     (GR00T_n15_Tactile/deploy_policy.py:151 的老写法)。
        # 0-d ndarray → unsqueeze 展开成 (1,) → x[0] 取到 np.str_ (str 子类) →
        # 渲染出的 prompt 与训练时的裸 str 逐字符一致。
        # (2026-09-12 用真实 eagle processor + 真 GR00TTransform 解码 token 验证:
        #  zyd/probe_annotation_batched.py)
        instruction = obs.pop("instruction", None)
        for _k in ("language", "raw_lang"):
            obs.pop(_k, None)
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError(
                "GR00T XE is language-conditioned: get_action() needs a non-empty "
                f"'instruction' on the observation, got {instruction!r}. The client "
                "sends the scene description via policy_sessions._prepare_observation; "
                "without it the policy silently conditions on the default string."
            )
        obs["annotation.human.action.task_description"] = np.array(instruction)
        if not getattr(self, "_logged_language", False):
            self._logged_language = True
            print(f"[GR00T_XE] language conditioning: {instruction!r}", flush=True)

        # 使用 Gr00tPolicy.get_action, 自动处理 transform + 推理 + denormalize
        action_dict = self.policy.get_action(obs)
        unified_action = action_dict["action.qpos"]  # [H, D]

        # unified_action shape 是 [H, D], 下面按 mode 决定给 IK 几个 step.
        unified_action = np.asarray(unified_action, dtype=np.float64)
        if unified_action.ndim == 1:
            raise ValueError(
                f"policy returned a 1-D action of shape {unified_action.shape}; "
                f"expected [H, D] so the horizon is explicit"
            )
        if unified_action.ndim > 2:
            unified_action = unified_action.reshape(-1, unified_action.shape[-1])

        converter = XEStateActionConverter()
        mode = _action_mode()
        if mode == "single":
            # 取第一个 step。必须先取 [0] 再用，否则 reshape(-1) 会把
            # 16×64=1024 维全部混在一起传给 IK。
            return converter.unified_to_joint_action(
                unified_action[0], robot_key, qpos_raw
            ).astype(np.float32)

        # chunk 模式: 对整条 horizon 做 IK, 交给 evaluator 去排队开环执行。
        # 每一步用上一步的解当种子 (step 0 用机器人真实 qpos —— 与 single 分支
        # 完全相同的输入, 所以第一步命令不变, 差别只在后面的 15 步);
        # 若 16 步全用同一个 qpos 当种子, 相邻步可能落到不同的 IK 分支上, 手臂会
        # 在两点之间甩。
        seed = qpos_raw
        steps = []
        for h in range(unified_action.shape[0]):
            seed = converter.unified_to_joint_action(
                unified_action[h], robot_key, seed
            )
            steps.append(np.asarray(seed, dtype=np.float32))
        if not getattr(self, "_logged_mode", False):
            self._logged_mode = True
            print(
                f"[GR00T_XE] action mode={mode}: returning a "
                f"[{unified_action.shape[0]}, {steps[0].shape[0]}] joint chunk for the "
                f"evaluator to play open loop (single mode would return "
                f"[{steps[0].shape[0]}])",
                flush=True,
            )
        return np.stack(steps, axis=0)


def get_model(usr_args: dict[str, Any]) -> GR00TXEPolicy:
    global _REMOTE_ROBOT_KEY
    model_path = str(usr_args.get("model_path") or usr_args.get("ckpt_dir"))
    robot_key = str(usr_args.get("robot_key"))
    _REMOTE_ROBOT_KEY = robot_key
    return GR00TXEPolicy(model_path=model_path, robot_key=robot_key)


def reset_model(model: GR00TXEPolicy) -> None:
    pass