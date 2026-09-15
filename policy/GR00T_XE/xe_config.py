# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cross-Embodiment data config for GR00T XE.

Extends the Dex2Bench config with:
  - max_action_dim=64  (unified cross-embodiment action space)
  - max_state_dim=64   (unified obs space)
  - action_mask support (embodiment-aware loss masking)
"""

from __future__ import annotations

import os

from gr00t.data.dataset import ModalityConfig
from gr00t.data.transform.base import ComposedModalityTransform, ModalityTransform
from gr00t.data.transform.concat import ConcatTransform
from gr00t.data.transform.state_action import StateActionToTensor, StateActionTransform
from gr00t.data.transform.video import VideoColorJitter, VideoCrop, VideoResize, VideoToNumpy, VideoToTensor
from gr00t.experiment.data_config import BaseDataConfig
from gr00t.model.transforms import GR00TTransform


CAMERA_MODE_VIDEO_KEYS = {
    "3cam": ["video.head", "video.right_wrist", "video.left_wrist"],
    "4cam": [
        "video.stereo_left",
        "video.stereo_right",
        "video.right_wrist",
        "video.left_wrist",
    ],
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


def _camera_mode() -> str:
    mode = os.environ.get("GR00T_CAMERA_MODE", "4cam").strip().lower()
    mode = CAMERA_MODE_ALIASES.get(mode, mode)
    if mode not in CAMERA_MODE_VIDEO_KEYS:
        raise ValueError(
            f"Unsupported GR00T_CAMERA_MODE={mode!r}; expected one of "
            f"{sorted(CAMERA_MODE_VIDEO_KEYS)} or set GR00T_VIDEO_KEYS."
        )
    return mode


def _video_keys() -> list[str]:
    explicit = os.environ.get("GR00T_VIDEO_KEYS")
    if explicit:
        return [
            key if key.startswith("video.") else f"video.{key}"
            for key in explicit.replace(",", " ").split()
        ]
    return CAMERA_MODE_VIDEO_KEYS[_camera_mode()]


class Dex2BenchXEDataConfig(BaseDataConfig):
    """Cross-embodiment 4-cam RGB + unified state/action + action_mask."""

    video_keys = _video_keys()
    state_keys = ["state.qpos"]
    action_keys = ["action.qpos"]
    # action_mask is NOT declared here and is NOT auto-generated from the action
    # width.  The dataset builds the per-robot authored-dims mask itself and
    # attaches it to the sample as the bare key "action_mask" (dataset.py
    # get_step_data); GR00TTransform._prepare_action honors it when present and
    # only falls back to "first n_action_dims are real" when it is missing.
    # That fallback is wrong for XE: the action width is always 64, so it would
    # supervise the 8 global pads (56-63) and every hand slot this robot has no
    # joints for -- all constant zero in the data.  Keep the dataset's mask.
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))
    max_state_dim = int(os.environ.get("GR00T_MAX_STATE_DIM", "64"))
    max_action_dim = int(os.environ.get("GR00T_MAX_ACTION_DIM", "64"))

    def modality_config(self) -> dict[str, ModalityConfig]:
        return super().modality_config()

    def transform(self) -> ModalityTransform:
        transforms = [
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(
                apply_to=self.video_keys,
                height=224,
                width=224,
                interpolation="linear",
            ),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
            ),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=self.max_state_dim,
                max_action_dim=self.max_action_dim,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)