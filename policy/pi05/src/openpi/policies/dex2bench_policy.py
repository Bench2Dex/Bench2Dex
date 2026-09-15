"""Pi0 inputs/outputs adapter for dex2bench.

Mirrors ``openpi.policies.aloha_policy`` but for configurable dex2bench
state/action widths with 4 cameras (stereo pair + dual wrist). No special joint-flip /
gripper-angular adaptation is needed because dex2bench data is in raw
joint-position space.
"""

from __future__ import annotations

import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms


# Default layout: ur5_left(6) + rh56dfx_left(12) + ur5_right(6) + rh56dfx_right(12)
DEFAULT_DEX2BENCH_STATE_DIM: int = 36


def make_dex2bench_example(state_dim: int = DEFAULT_DEX2BENCH_STATE_DIM) -> dict:
    """Random example for testing the data transforms."""
    return {
        "state": np.zeros((state_dim,), dtype=np.float32),
        "images": {
            "cam_stereo_left": np.zeros((224, 224, 3), dtype=np.uint8),
            "cam_stereo_right": np.zeros((224, 224, 3), dtype=np.uint8),
            "cam_wrist_left": np.zeros((224, 224, 3), dtype=np.uint8),
            "cam_wrist_right": np.zeros((224, 224, 3), dtype=np.uint8),
        },
        "prompt": "do something",
    }


def _require_last_dim(value: np.ndarray, expected_dim: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape[-1] != expected_dim:
        raise ValueError(
            f"Dex2Bench {name} has dim {arr.shape[-1]}, expected state_dim={expected_dim}. "
            "Set policy/pi05/deploy_policy.yml state_dim to match the data/env."
        )
    return arr


@dataclasses.dataclass(frozen=True)
class Dex2BenchInputs(transforms.DataTransformFn):
    """Inputs for Pi0 trained on dex2bench data.

    Expected input dict (produced by ``Dex2BenchPi0Dataset``):
      - state:   (state_dim,) float32 raw joint positions
      - images:  dict[name, (H, W, 3) uint8 RGB] names subset of EXPECTED_CAMERAS
      - actions: (action_horizon, state_dim) float32 (training only)
      - prompt:  str

    Output dict (consumed by Normalize -> ResizeImages -> TokenizePrompt -> model):
      - image:      {"base_0_rgb", "base_1_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
      - image_mask: same keys, bool
      - state:      (model.action_dim,) float32, zero-padded if needed
      - actions:    (action_horizon, model.action_dim) float32, zero-padded if needed
      - prompt:     str
    """

    # Pi0 model.action_dim used to right-pad state and actions.
    action_dim: int
    # Raw dex2bench state/action width before padding.
    state_dim: int = DEFAULT_DEX2BENCH_STATE_DIM

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = (
        "cam_stereo_left",
        "cam_stereo_right",
        "cam_wrist_left",
        "cam_wrist_right",
    )

    def __post_init__(self) -> None:
        if self.state_dim > self.action_dim:
            raise ValueError(
                f"Dex2Bench state_dim ({self.state_dim}) cannot exceed model action_dim ({self.action_dim})."
            )

    def __call__(self, data: dict) -> dict:
        in_images = data["images"]
        if not isinstance(in_images, dict):
            raise TypeError(f"Expected 'images' to be a dict, got {type(in_images)}")
        unknown = set(in_images) - set(self.EXPECTED_CAMERAS)
        if unknown:
            raise ValueError(
                f"Unexpected cameras {unknown}. Expected subset of {self.EXPECTED_CAMERAS}."
            )

        missing = set(self.EXPECTED_CAMERAS) - set(in_images)
        if missing:
            raise KeyError(
                f"Missing required cameras {sorted(missing)}. "
                f"Expected {self.EXPECTED_CAMERAS}."
            )

        def _to_hwc_uint8(img: np.ndarray) -> np.ndarray:
            img = np.asarray(img)
            # Some upstream pipelines hand back (C, H, W); handle both.
            if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
                img = einops.rearrange(img, "c h w -> h w c")
            if np.issubdtype(img.dtype, np.floating):
                img = (255.0 * img).astype(np.uint8)
            return img

        image_map = {
            "base_0_rgb": "cam_stereo_left",
            "base_1_rgb": "cam_stereo_right",
            "left_wrist_0_rgb": "cam_wrist_left",
            "right_wrist_0_rgb": "cam_wrist_right",
        }
        images: dict[str, np.ndarray] = {
            dest: _to_hwc_uint8(in_images[source])
            for dest, source in image_map.items()
        }
        image_masks: dict[str, np.ndarray] = {dest: np.True_ for dest in images}

        state = _require_last_dim(data["state"], self.state_dim, "state")
        state = transforms.pad_to_dim(state, self.action_dim)

        out = {
            "image": images,
            "image_mask": image_masks,
            "state": state,
        }

        if "actions" in data:
            actions = _require_last_dim(data["actions"], self.state_dim, "actions")
            out["actions"] = transforms.pad_to_dim(actions, self.action_dim)

        if "prompt" in data:
            out["prompt"] = data["prompt"]

        return out


@dataclasses.dataclass(frozen=True)
class Dex2BenchOutputs(transforms.DataTransformFn):
    """Outputs for Pi0 trained on dex2bench data.

    Strips the model's zero-padded action dims back to ``state_dim``.
    """

    state_dim: int = DEFAULT_DEX2BENCH_STATE_DIM

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"][..., : self.state_dim])
        return {"actions": actions}
