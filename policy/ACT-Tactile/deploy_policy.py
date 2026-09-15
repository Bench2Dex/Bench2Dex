"""Checkpoint-backed online inference adapter for the tactile ACT policy."""

from __future__ import annotations

from collections import deque
import math
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np
import torch

from .dataset import NormalizationStats
from .train import load_checkpoint


DEFAULT_CHECKPOINT_NAME = "policy_best.ckpt"


def resolve_tactile_checkpoint(
    checkpoint_or_dir: str | Path,
    checkpoint_name: str | None = None,
) -> Path:
    """Resolve either a checkpoint file or a directory plus checkpoint name."""

    source = Path(checkpoint_or_dir).expanduser()
    if source.is_file():
        return source.resolve()
    name = checkpoint_name or DEFAULT_CHECKPOINT_NAME
    candidate = source / name
    if not candidate.is_file():
        raise FileNotFoundError(candidate.resolve())
    return candidate.resolve()


class TactilePolicyDeployment:
    """Prepare online RGB/TacMap observations and return physical actions."""

    returns_action_chunks = True

    def __init__(
        self,
        checkpoint_or_dir: str | Path,
        *,
        checkpoint_name: str | None = None,
        device: str | torch.device = "cpu",
        temporal_agg: bool = False,
        temporal_agg_k: float = 0.1,
    ) -> None:
        self.device = torch.device(device)
        self.temporal_agg = bool(temporal_agg)
        self.temporal_agg_k = float(temporal_agg_k)
        if not math.isfinite(self.temporal_agg_k) or self.temporal_agg_k < 0:
            raise ValueError("temporal_agg_k must be finite and non-negative")
        self.returns_action_chunks = not self.temporal_agg
        self.checkpoint_path = resolve_tactile_checkpoint(
            checkpoint_or_dir,
            checkpoint_name,
        )
        self.policy, payload = load_checkpoint(self.checkpoint_path, device=self.device)
        self.policy.eval()
        self.payload = payload
        self.stats = NormalizationStats.from_dict(payload["normalization_stats"])
        self.camera_names = tuple(str(name) for name in payload["camera_names"])
        self.site_names = tuple(str(name) for name in payload["site_names"])
        self.robot_key = str(payload["robot_key"])
        self.active_indices = tuple(int(index) for index in payload["active_indices"])
        self.active_joint_names = tuple(
            str(name) for name in payload.get("active_joint_names", ())
        )
        self.image_size = tuple(int(value) for value in payload["image_size"])
        tactile_size = payload.get("tactile_size")
        self.tactile_size = (
            None if tactile_size is None else tuple(int(value) for value in tactile_size)
        )
        sensor_metadata = payload.get("tactile_sensor_metadata")
        if not isinstance(sensor_metadata, Mapping):
            raise ValueError(
                "checkpoint has no tactile_sensor_metadata; retrain or resave it "
                "before online tactile evaluation"
            )
        self.tactile_sensor_metadata = dict(sensor_metadata)
        self._validate_checkpoint_metadata()
        self._current_step = 0
        self._action_history: deque[tuple[int, np.ndarray]] = deque(
            maxlen=self.chunk_size
        )

    @property
    def state_dim(self) -> int:
        return int(self.policy.config.state_dim)

    @property
    def chunk_size(self) -> int:
        return int(self.policy.config.chunk_size)

    def enable_tactile_attention_capture(self) -> None:
        """Turn on attention-weight capture in the tactile pooling module."""
        detr = self.policy.model.model
        detr.capture_tactile_attention = True
        detr.capture_modality_attention = True

    def last_tactile_attention_weights(self) -> np.ndarray | None:
        """Attention weights from the last forward pass, [num_sites] (heads averaged)."""
        detr = self.policy.model.model
        weights = getattr(detr, "last_tactile_attn_weights", None)
        if weights is None:
            return None
        array = weights.detach().cpu().numpy()  # [B, 1, num_sites]
        return array.reshape(-1, array.shape[-1])[0]

    def last_modality_attention(self) -> np.ndarray | None:
        """Cross-attention proportions [latent, proprio, visual, tactile] from last forward."""
        detr = self.policy.model.model
        attn = getattr(detr, "_last_modality_attention", None)
        if attn is None:
            return None
        return attn.detach().cpu().numpy()[0]  # [4]

    def _validate_checkpoint_metadata(self) -> None:
        config = self.policy.config
        if tuple(self.camera_names) != tuple(config.camera_names):
            raise ValueError(
                "checkpoint camera_names do not match the tactile ACT model configuration"
            )
        if len(set(self.camera_names)) != len(self.camera_names):
            raise ValueError("checkpoint camera_names contains duplicates")
        if tuple(self.site_names) != tuple(config.site_names):
            raise ValueError(
                "checkpoint site_names do not match the tactile ACT model configuration"
            )
        if len(set(self.site_names)) != len(self.site_names):
            raise ValueError("checkpoint site_names contains duplicates")
        if len(self.active_indices) != self.state_dim:
            raise ValueError(
                "checkpoint active_indices count does not match state_dim: "
                f"{len(self.active_indices)} != {self.state_dim}"
            )
        if len(self.active_joint_names) != self.state_dim:
            raise ValueError(
                "checkpoint active_joint_names count does not match state_dim; "
                "legacy checkpoints must be resaved before online evaluation"
            )
        if len(set(self.active_joint_names)) != len(self.active_joint_names):
            raise ValueError("checkpoint active_joint_names contains duplicates")
        if self.stats.state_dim != self.state_dim:
            raise ValueError(
                "checkpoint normalization state_dim does not match model config: "
                f"{self.stats.state_dim} != {self.state_dim}"
            )
        if len(self.image_size) != 2 or min(self.image_size) <= 0:
            raise ValueError(f"invalid checkpoint image_size: {self.image_size}")
        if self.tactile_size is not None and (
            len(self.tactile_size) != 2 or min(self.tactile_size) <= 0
        ):
            raise ValueError(f"invalid checkpoint tactile_size: {self.tactile_size}")
        required_sensor_keys = {
            "resolution_step",
            "native_resolution",
            "image_size",
            "max_distance_m",
        }
        missing_sensor_keys = sorted(
            required_sensor_keys - set(self.tactile_sensor_metadata)
        )
        if missing_sensor_keys:
            raise ValueError(
                "checkpoint tactile_sensor_metadata is missing: "
                f"{missing_sensor_keys}"
            )
        self.tactile_resolution_step = int(
            self.tactile_sensor_metadata["resolution_step"]
        )
        self.tactile_native_resolution = int(
            self.tactile_sensor_metadata["native_resolution"]
        )
        self.tactile_image_size = int(self.tactile_sensor_metadata["image_size"])
        self.tactile_max_distance_m = float(
            self.tactile_sensor_metadata["max_distance_m"]
        )
        if self.tactile_resolution_step <= 0:
            raise ValueError("tactile resolution_step must be positive")
        if (
            self.tactile_native_resolution <= 0
            or self.tactile_image_size <= 0
            or self.tactile_native_resolution // self.tactile_resolution_step
            != self.tactile_image_size
        ):
            raise ValueError("inconsistent tactile native_resolution/image_size metadata")
        if self.tactile_max_distance_m <= 0:
            raise ValueError("tactile max_distance_m must be positive")

    def validate_runtime(
        self,
        robot_key: str,
        active_indices: Sequence[int],
        active_joint_names: Sequence[str],
    ) -> None:
        """Reject a runtime whose action/state convention differs from training."""

        if str(robot_key) != self.robot_key:
            raise ValueError(
                f"runtime robot_key '{robot_key}' does not match checkpoint "
                f"robot_key '{self.robot_key}'"
            )
        runtime_indices = tuple(int(index) for index in active_indices)
        if runtime_indices != self.active_indices:
            raise ValueError(
                "runtime active_indices do not match checkpoint: "
                f"{runtime_indices} != {self.active_indices}"
            )
        runtime_names = tuple(str(name) for name in active_joint_names)
        if runtime_names != self.active_joint_names:
            raise ValueError(
                "runtime active_joint_names do not match checkpoint order: "
                f"{runtime_names} != {self.active_joint_names}"
            )

    def resolve_tactile_sensor_settings(
        self,
        resolution_step: int | None,
        max_distance_m: float | None,
    ) -> tuple[int, float]:
        """Resolve CLI values while rejecting TacMap distribution drift."""

        resolved_step = (
            self.tactile_resolution_step
            if resolution_step is None
            else int(resolution_step)
        )
        resolved_distance = (
            self.tactile_max_distance_m
            if max_distance_m is None
            else float(max_distance_m)
        )
        if resolved_step != self.tactile_resolution_step:
            raise ValueError(
                f"--tactile-resolution-step={resolved_step} does not match "
                f"checkpoint resolution_step={self.tactile_resolution_step}"
            )
        if not math.isclose(
            resolved_distance,
            self.tactile_max_distance_m,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(
                f"--tactile-max-distance={resolved_distance} does not match "
                f"checkpoint max_distance_m={self.tactile_max_distance_m}"
            )
        return resolved_step, resolved_distance

    @staticmethod
    def _require_mapping(observation: Mapping, key: str) -> Mapping:
        value = observation.get(key)
        if not isinstance(value, Mapping):
            raise ValueError(f"observation '{key}' must be a mapping")
        return value

    @staticmethod
    def _require_channels(values: Mapping, names: tuple[str, ...], label: str) -> None:
        missing = [name for name in names if name not in values]
        if missing:
            raise ValueError(f"missing {label} channels: {missing}")

    def _encode_images(self, images_by_name: Mapping) -> np.ndarray:
        self._require_channels(images_by_name, self.camera_names, "camera")
        output_h, output_w = self.image_size
        images: list[np.ndarray] = []
        for camera_name in self.camera_names:
            image = np.asarray(images_by_name[camera_name])
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(
                    f"camera '{camera_name}' must have HWC RGB shape, got {image.shape}"
                )
            if image.shape[:2] != self.image_size:
                image = cv2.resize(
                    image,
                    (output_w, output_h),
                    interpolation=cv2.INTER_LINEAR,
                )
            if image.dtype != np.uint8:
                image = np.clip(image, 0, 255).astype(np.uint8)
            images.append(np.moveaxis(image, -1, 0))
        return np.stack(images, axis=0)

    def _encode_tactile(self, tactile_by_name: Mapping) -> np.ndarray:
        self._require_channels(tactile_by_name, self.site_names, "tactile")
        tactile_maps: list[np.ndarray] = []
        for site_name in self.site_names:
            tactile = np.asarray(tactile_by_name[site_name])
            if tactile.ndim != 2:
                raise ValueError(
                    f"tactile site '{site_name}' must have HW shape, got {tactile.shape}"
                )
            if self.tactile_size is not None and tactile.shape != self.tactile_size:
                output_h, output_w = self.tactile_size
                tactile = cv2.resize(
                    tactile,
                    (output_w, output_h),
                    interpolation=cv2.INTER_AREA,
                )
            tactile_maps.append(tactile[None, ...].astype(np.float32) / 255.0)
        return np.stack(tactile_maps, axis=0)

    def encode_observation(
        self,
        observation: Mapping,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert one raw online observation into checkpoint-ordered tensors."""

        qpos = np.asarray(observation.get("qpos"), dtype=np.float32).reshape(-1)
        if qpos.shape != (self.state_dim,):
            raise ValueError(
                f"qpos must contain {self.state_dim} active values, got shape {qpos.shape}"
            )
        if not np.isfinite(qpos).all():
            raise ValueError("qpos contains non-finite values")
        normalized_qpos = (qpos - self.stats.qpos_mean) / self.stats.qpos_std
        images = self._encode_images(self._require_mapping(observation, "images"))
        tactile = self._encode_tactile(self._require_mapping(observation, "tactile"))
        return (
            torch.from_numpy(np.ascontiguousarray(normalized_qpos))[None].to(self.device),
            torch.from_numpy(np.ascontiguousarray(images))[None].to(self.device),
            torch.from_numpy(np.ascontiguousarray(tactile))[None].to(self.device),
        )

    def _predict_action_chunk(self, observation: Mapping) -> np.ndarray:
        """Run prior-only ACT inference and return one physical action chunk."""

        qpos, images, tactile = self.encode_observation(observation)
        with torch.inference_mode():
            normalized = self.policy.predict(qpos, images, tactile)
        normalized_np = normalized.detach().cpu().numpy()
        expected = (1, self.chunk_size, self.state_dim)
        if normalized_np.shape != expected:
            raise ValueError(
                f"policy returned action shape {normalized_np.shape}, expected {expected}"
            )
        actions = (
            normalized_np[0] * self.stats.action_std[None, :]
            + self.stats.action_mean[None, :]
        )
        if not np.isfinite(actions).all():
            raise RuntimeError("policy returned non-finite action values")
        return actions.astype(np.float32, copy=False)

    def _aggregate_current_action(self, action_chunk: np.ndarray) -> np.ndarray:
        self._action_history.append((self._current_step, action_chunk))
        while (
            self._action_history
            and self._action_history[0][0] + self.chunk_size <= self._current_step
        ):
            self._action_history.popleft()

        actions_for_current_step = np.stack(
            [
                chunk[self._current_step - start_step]
                for start_step, chunk in self._action_history
            ],
            axis=0,
        )
        weights = np.exp(
            -self.temporal_agg_k
            * np.arange(len(actions_for_current_step), dtype=np.float64)
        )
        weights /= weights.sum()
        action = (actions_for_current_step * weights[:, None]).sum(axis=0)
        if not np.isfinite(action).all():
            raise RuntimeError("temporal aggregation produced non-finite action values")
        return action.astype(np.float32, copy=False)

    def get_action(self, observation: Mapping) -> np.ndarray:
        """Return a physical chunk, or one temporally aggregated physical action."""

        action_chunk = self._predict_action_chunk(observation)
        if not self.temporal_agg:
            return action_chunk

        action = self._aggregate_current_action(action_chunk)
        self._current_step += 1
        return action

    def reset(self) -> None:
        """Clear temporal aggregation state between episodes."""

        self._current_step = 0
        self._action_history.clear()
