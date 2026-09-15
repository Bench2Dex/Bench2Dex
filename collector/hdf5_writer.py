"""HDF5 episode writer."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Dict

import h5py
import numpy as np

from .episode_buffer import EpisodeBuffer


class HDF5EpisodeWriter:
    """Write one episode to one HDF5 file and keep a manifest index."""

    EPISODE_PATTERN = re.compile(r"^episode_(\d{6})\.hdf5$")

    def __init__(self, dataset_root: str, dataset_name: str, task_name: str, scene_name: str):
        self.output_dir = os.path.abspath(os.path.join(dataset_root, dataset_name, task_name, scene_name))
        os.makedirs(self.output_dir, exist_ok=True)
        self.manifest_path = os.path.join(self.output_dir, "episode_manifest.json")

    def next_episode_id(self) -> int:
        max_id = -1
        for filename in os.listdir(self.output_dir):
            match = self.EPISODE_PATTERN.match(filename)
            if match:
                max_id = max(max_id, int(match.group(1)))
        return max_id + 1

    @staticmethod
    def _write_meta_group(file: h5py.File, meta: Dict) -> str:
        str_dtype = h5py.string_dtype(encoding="utf-8")
        created_at = meta.get("created_at", datetime.now(timezone.utc).isoformat())
        meta_grp = file.create_group("meta")
        meta_payload = dict(meta)
        meta_payload.setdefault("created_at", created_at)
        for key, value in meta_payload.items():
            if isinstance(value, str):
                meta_grp.create_dataset(key, data=np.asarray(value, dtype=str_dtype))
            elif isinstance(value, bool):
                meta_grp.create_dataset(key, data=np.asarray(value, dtype=np.bool_))
            elif isinstance(value, int):
                meta_grp.create_dataset(key, data=np.asarray(value, dtype=np.int64))
            elif isinstance(value, float):
                meta_grp.create_dataset(key, data=np.asarray(value, dtype=np.float64))
            else:
                meta_grp.create_dataset(key, data=np.asarray(json.dumps(value), dtype=str_dtype))
        return created_at

    @classmethod
    def init_labels_artifact(
        cls,
        output_path: str,
        meta: Dict,
        frame_valid: np.ndarray | None = None,
        frame_errors: list[str] | None = None,
        sim_steps: np.ndarray | None = None,
    ) -> str:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with h5py.File(output_path, "w") as f:
            cls._write_meta_group(f, meta)
            if frame_valid is not None:
                f.create_dataset("frame_valid", data=np.asarray(frame_valid, dtype=np.bool_))
            if frame_errors is not None:
                str_dtype = h5py.string_dtype(encoding="utf-8")
                f.create_dataset("frame_errors", data=np.asarray(frame_errors, dtype=str_dtype))
            if sim_steps is not None:
                time_grp = f.create_group("time")
                time_grp.create_dataset("sim_step", data=np.asarray(sim_steps, dtype=np.int64))
            f.require_group("labels")
        return output_path

    def write_episode(self, buffer: EpisodeBuffer, episode_id: int, meta: Dict) -> str:
        if buffer.frame_count <= 0:
            raise ValueError("EpisodeBuffer is empty; no frame to write.")

        episode_path = os.path.join(self.output_dir, f"episode_{episode_id:06d}.hdf5")
        frames = buffer.frames
        frame_count = len(frames)
        str_dtype = h5py.string_dtype(encoding="utf-8")

        meta_payload = dict(meta)
        metrics_episode = meta_payload.pop("metrics_episode", None)
        metrics_timeseries = meta_payload.pop("metrics_timeseries", None)
        meta_payload.setdefault("frame_count", frame_count)

        # Build action metadata from first frame
        action_meta = self._build_action_metadata(buffer, meta_payload)

        with h5py.File(episode_path, "w") as f:
            created_at = self._write_meta_group(f, meta_payload)

            # Write time group
            time_grp = f.create_group("time")
            time_grp.create_dataset(
                "frame_index",
                data=buffer.build_frame_index(),
            )
            time_grp.create_dataset(
                "timestamp_ns",
                data=np.asarray([item.timestamp_ns for item in frames], dtype=np.int64),
            )
            time_grp.create_dataset(
                "sim_step",
                data=np.asarray([item.sim_step for item in frames], dtype=np.int64),
            )

            # Write episode group (is_first, is_last, done, success)
            episode_grp = f.create_group("episode")
            episode_grp.create_dataset("is_first", data=buffer.build_is_first())
            episode_grp.create_dataset("is_last", data=buffer.build_is_last())
            # done is True only for the last frame (episode termination)
            done = np.zeros(frame_count, dtype=np.bool_)
            done[-1] = True
            episode_grp.create_dataset("done", data=done)
            # success - we would need to get this from the buffer/metadata
            success = np.zeros(frame_count, dtype=np.bool_)
            # Mark last frame success based on metadata
            if meta_payload.get("success", False):
                success[-1] = True
            episode_grp.create_dataset("success", data=success)

            # Write action group (v2 schema)
            action_grp = f.create_group("action")
            self._write_action_group(action_grp, buffer, action_meta, meta_payload)

            f.create_dataset(
                "frame_valid",
                data=np.asarray([item.valid for item in frames], dtype=np.bool_),
            )
            f.create_dataset(
                "frame_errors",
                data=np.asarray(["; ".join(item.errors) for item in frames], dtype=str_dtype),
            )

            self._write_cameras(f, buffer)
            self._write_robot(f, buffer)
            self._write_objects(f, buffer)
            self._write_labels(f, buffer)

            # Write /metrics group from meta payload
            self._write_metrics_group(f, metrics_episode, metrics_timeseries, str_dtype)

        self.append_manifest(
            episode_id=episode_id,
            episode_path=episode_path,
            frame_count=frame_count,
            created_at=meta_payload.get("created_at", created_at),
            error_count=buffer.error_count,
        )
        return episode_path

    def _build_action_metadata(self, buffer: EpisodeBuffer, meta: dict) -> dict:
        """Build action-related metadata from episode metadata.

        Uses canonical action schema from metadata (independent of observation modalities).
        """
        # Use action metadata from episode metadata (set by collector)
        action_names = list(meta.get("action_names", []))
        action_dim = int(meta.get("action_dim", len(action_names)))

        return {
            "action_dim": action_dim,
            "action_names": action_names,
            "has_action": meta.get("has_action", True),
        }

    def _write_action_group(self, action_grp: h5py.Group, buffer: EpisodeBuffer, action_meta: dict, meta: dict) -> None:
        """Write action group with commanded actions and metadata."""
        frame_count = buffer.frame_count
        str_dtype = h5py.string_dtype(encoding="utf-8")

        # Get action_dim from metadata
        action_dim = action_meta.get("action_dim", 0)

        # Build action commanded array
        if action_dim > 0:
            commanded = np.full((frame_count, action_dim), np.nan, dtype=np.float32)
            for i, frame in enumerate(buffer.frames):
                if frame.action_commanded is not None:
                    commanded[i] = np.asarray(frame.action_commanded, dtype=np.float32)
            action_grp.create_dataset("commanded", data=commanded)
        else:
            action_grp.create_dataset("commanded", data=np.zeros((frame_count, 0), dtype=np.float32))

        # Build action_valid array
        action_grp.create_dataset("action_valid", data=buffer.build_action_valid())

        # Build action_source array
        sources = [f.action_source if f.action_source else "none" for f in buffer.frames]
        action_grp.create_dataset("source", data=np.asarray(sources, dtype=str_dtype))

        # Write action_names
        if action_meta.get("action_names"):
            action_grp.create_dataset("action_names", data=np.asarray(action_meta["action_names"], dtype=str_dtype))

        # Write control_mode and action_type as scalars (from episode metadata)
        control_mode = meta.get("control_mode", "joint_position")
        action_type = meta.get("action_type", "absolute")
        action_grp.create_dataset("control_mode", data=np.asarray(control_mode, dtype=str_dtype))
        action_grp.create_dataset("action_type", data=np.asarray(action_type, dtype=str_dtype))

    @classmethod
    def write_labels_artifact(
        cls,
        output_path: str,
        buffer: EpisodeBuffer,
        meta: Dict,
        frame_valid: np.ndarray | None = None,
        frame_errors: list[str] | None = None,
        sim_steps: np.ndarray | None = None,
    ) -> str:
        cls.init_labels_artifact(
            output_path,
            meta,
            frame_valid=frame_valid,
            frame_errors=frame_errors,
            sim_steps=sim_steps,
        )
        writer = object.__new__(cls)
        with h5py.File(output_path, "r+") as f:
            writer._write_labels(f, buffer)
        return output_path

    def append_manifest(
        self,
        episode_id: int,
        episode_path: str,
        frame_count: int,
        created_at: str,
        error_count: int,
    ) -> None:
        payload = {"episodes": []}
        if os.path.exists(self.manifest_path):
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
            except json.JSONDecodeError:
                payload = {"episodes": []}
        payload.setdefault("episodes", [])
        payload["episodes"].append(
            {
                "episode_id": episode_id,
                "file": os.path.basename(episode_path),
                "frame_count": frame_count,
                "error_count": error_count,
                "created_at": created_at,
            }
        )
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    @staticmethod
    def _first_camera_frame(buffer: EpisodeBuffer, cam_id: str):
        for frame in buffer.frames:
            cam = frame.camera.get(cam_id)
            if cam is not None:
                return cam
        return None

    @staticmethod
    def _first_camera_modality(buffer: EpisodeBuffer, cam_id: str, modality: str):
        for frame in buffer.frames:
            cam = frame.camera.get(cam_id)
            if cam is None:
                continue
            value = getattr(cam, modality)
            if value is not None:
                return value
        return None

    def _write_cameras(self, file: h5py.File, buffer: EpisodeBuffer) -> None:
        cameras_grp = file.create_group("cameras")
        frame_count = buffer.frame_count
        for cam_id in buffer.camera_ids:
            first_cam = self._first_camera_frame(buffer, cam_id)
            if first_cam is None:
                continue
            cam_grp = cameras_grp.create_group(cam_id)
            cam_grp.create_dataset("intrinsic", data=first_cam.intrinsic.astype(np.float32))

            extr = np.repeat(np.eye(4, dtype=np.float32)[None, ...], frame_count, axis=0)
            for i, frame in enumerate(buffer.frames):
                cam = frame.camera.get(cam_id)
                if cam is not None:
                    extr[i] = cam.extrinsic_world_from_cam.astype(np.float32)
            cam_grp.create_dataset("extrinsic_world_from_cam", data=extr.astype(np.float32))

            first_rgb = self._first_camera_modality(buffer, cam_id, "rgb")
            if first_rgb is not None:
                rgb = np.zeros((frame_count, *first_rgb.shape), dtype=np.uint8)
                for i, frame in enumerate(buffer.frames):
                    cam = frame.camera.get(cam_id)
                    if cam is not None and cam.rgb is not None:
                        rgb[i] = cam.rgb.astype(np.uint8)
                cam_grp.create_dataset("rgb", data=rgb, compression="gzip", compression_opts=4)

            first_depth = self._first_camera_modality(buffer, cam_id, "depth_m")
            if first_depth is not None:
                depth = np.full((frame_count, *first_depth.shape), np.nan, dtype=np.float32)
                depth_semantics = np.asarray(["" for _ in range(frame_count)], dtype=h5py.string_dtype(encoding="utf-8"))
                for i, frame in enumerate(buffer.frames):
                    cam = frame.camera.get(cam_id)
                    if cam is not None and cam.depth_m is not None:
                        depth[i] = cam.depth_m.astype(np.float32)
                        depth_semantics[i] = str(getattr(cam, "depth_semantics", None) or "")
                cam_grp.create_dataset("depth_m", data=depth, compression="gzip", compression_opts=4)
                cam_grp.create_dataset("depth_semantics", data=depth_semantics)

            # ---- camera_model (all cameras) ----
            str_dtype_cam = h5py.string_dtype(encoding="utf-8")
            camera_model = getattr(first_cam, "camera_model", "pinhole")
            cam_grp.create_dataset(
                "camera_model",
                data=np.asarray(camera_model, dtype=str_dtype_cam),
            )

            # ---- clipping_range (all cameras) ----
            clipping_range = getattr(first_cam, "clipping_range", None)
            if clipping_range is not None:
                cam_grp.create_dataset(
                    "clipping_range",
                    data=np.asarray(clipping_range, dtype=np.float32),
                )

            # ---- fisheye-specific fields ----
            if camera_model == "fisheye":
                # Calibration fields (single source of truth for projection)
                if getattr(first_cam, "fisheye_camera_matrix", None) is not None:
                    cam_grp.create_dataset(
                        "fisheye_camera_matrix",
                        data=first_cam.fisheye_camera_matrix.astype(np.float32),
                    )
                if getattr(first_cam, "distortion_coefficients", None) is not None:
                    cam_grp.create_dataset(
                        "distortion_coefficients",
                        data=first_cam.distortion_coefficients.astype(np.float32),
                    )
                if getattr(first_cam, "fisheye_valid_mask", None) is not None:
                    cam_grp.create_dataset(
                        "fisheye_valid_mask",
                        data=first_cam.fisheye_valid_mask.astype(np.bool_),
                        compression="gzip",
                        compression_opts=4,
                    )
                cam_grp.create_dataset(
                    "distortion_model",
                    data=np.asarray("opencv_fisheye_kannala_brandt", dtype=str_dtype_cam),
                )
                cam_grp.create_dataset(
                    "raw_image_semantics",
                    data=np.asarray("raw_fisheye_rectangular", dtype=str_dtype_cam),
                )

    def _write_robot(self, file: h5py.File, buffer: EpisodeBuffer) -> None:
        robot_first = next((frame.robot for frame in buffer.frames if frame.robot is not None), None)
        if robot_first is None:
            return
        str_dtype = h5py.string_dtype(encoding="utf-8")
        robot_grp = file.create_group("robot")
        first_joint_state = next(
            (
                frame.robot
                for frame in buffer.frames
                if frame.robot is not None and frame.robot.get("joint_names") is not None
            ),
            None,
        )
        joint_names = list(first_joint_state.get("joint_names", [])) if first_joint_state is not None else []
        joint_count = len(joint_names)
        robot_grp.create_dataset("joint_names", data=np.asarray(joint_names, dtype=str_dtype))

        qpos = np.full((buffer.frame_count, joint_count), np.nan, dtype=np.float32)
        qvel = np.full((buffer.frame_count, joint_count), np.nan, dtype=np.float32)
        has_effort = any(
            frame.robot is not None and frame.robot.get("qeffort") is not None for frame in buffer.frames
        )
        qeffort = np.full((buffer.frame_count, joint_count), np.nan, dtype=np.float32) if has_effort else None

        for i, frame in enumerate(buffer.frames):
            if frame.robot is None:
                continue
            if joint_count > 0:
                qpos[i] = np.asarray(frame.robot.get("qpos", np.full((joint_count,), np.nan, dtype=np.float32)))
                qvel[i] = np.asarray(frame.robot.get("qvel", np.full((joint_count,), np.nan, dtype=np.float32)))
            if has_effort and frame.robot.get("qeffort") is not None:
                qeffort[i] = frame.robot["qeffort"]

        robot_grp.create_dataset("qpos", data=qpos)
        robot_grp.create_dataset("qvel", data=qvel)
        if has_effort and qeffort is not None:
            robot_grp.create_dataset("qeffort", data=qeffort)
        else:
            robot_grp.create_dataset("qeffort", data=np.zeros((0, 0), dtype=np.float32))

        first_tactile = next(
            (frame.robot.get("tactile") for frame in buffer.frames if frame.robot is not None and frame.robot.get("tactile") is not None),
            None,
        )
        if first_tactile is not None:
            tactile_grp = robot_grp.create_group("tactile")
            self._write_tacmap_tactile(tactile_grp, buffer, str_dtype)

    def _write_tactile_meta_value(self, group: h5py.Group, key: str, value, str_dtype) -> None:
        if isinstance(value, dict):
            child = group.create_group(str(key))
            for child_key, child_value in value.items():
                self._write_tactile_meta_value(child, str(child_key), child_value, str_dtype)
            return
        if isinstance(value, str):
            group.create_dataset(key, data=np.asarray(value, dtype=str_dtype))
            return
        if isinstance(value, bool):
            group.create_dataset(key, data=np.asarray(value, dtype=np.bool_))
            return
        if isinstance(value, int):
            group.create_dataset(key, data=np.asarray(value, dtype=np.int64))
            return
        if isinstance(value, float):
            group.create_dataset(key, data=np.asarray(value, dtype=np.float64))
            return
        if isinstance(value, (list, tuple)) and value and all(isinstance(item, str) for item in value):
            group.create_dataset(key, data=np.asarray(list(value), dtype=str_dtype))
            return
        try:
            group.create_dataset(key, data=np.asarray(value))
        except TypeError:
            group.create_dataset(key, data=np.asarray(json.dumps(value), dtype=str_dtype))

    def _write_tacmap_tactile(self, tactile_grp: h5py.Group, buffer: EpisodeBuffer, str_dtype) -> None:
        first_tactile = next(
            (
                frame.robot.get("tactile")
                for frame in buffer.frames
                if frame.robot is not None and frame.robot.get("tactile") is not None
            ),
            None,
        )
        if first_tactile is None:
            return

        meta = dict(first_tactile.get("meta", {}))
        meta.setdefault("sensor_type", "tacmap")
        first_tacmap = dict(first_tactile.get("tacmap", {}))
        if not first_tacmap:
            raise ValueError("Only TacMap tactile payloads are supported by HDF5EpisodeWriter.")
        site_names = [str(site_name) for site_name in (meta.get("site_names") or first_tacmap.keys())]
        meta["site_names"] = site_names

        meta_grp = tactile_grp.create_group("meta")
        for key, value in meta.items():
            self._write_tactile_meta_value(meta_grp, str(key), value, str_dtype)

        tacmap_grp = tactile_grp.create_group("tacmap")
        for site_name in site_names:
            if site_name not in first_tacmap:
                raise ValueError(f"TacMap first frame is missing site: {site_name}")
            first_arr = np.asarray(first_tacmap[site_name], dtype=np.uint8)
            if first_arr.ndim != 2:
                raise ValueError(f"TacMap site '{site_name}' must be a 2-D uint8 image, got {first_arr.shape}.")

            data = np.zeros((buffer.frame_count, *first_arr.shape), dtype=np.uint8)
            for i, frame in enumerate(buffer.frames):
                tactile = None if frame.robot is None else frame.robot.get("tactile")
                if tactile is None:
                    continue
                payload = tactile.get("tacmap", {}).get(site_name)
                if payload is None:
                    continue
                arr = np.asarray(payload, dtype=np.uint8)
                if arr.shape != first_arr.shape:
                    raise ValueError(
                        f"TacMap site '{site_name}' shape changed from {first_arr.shape} to {arr.shape}."
                    )
                data[i] = arr
            tacmap_grp.create_dataset(site_name, data=data, compression="gzip", compression_opts=4)

        self._write_tacmap_optional_map(
            tactile_grp,
            buffer,
            first_tactile,
            site_names,
            payload_key="distance_along_normal_m",
            dtype=np.float32,
            fillvalue=0.0,
        )
        self._write_tacmap_optional_map(
            tactile_grp,
            buffer,
            first_tactile,
            site_names,
            payload_key="contact_mask",
            dtype=np.bool_,
            fillvalue=False,
        )

    def _write_tacmap_optional_map(
        self,
        tactile_grp: h5py.Group,
        buffer: EpisodeBuffer,
        first_tactile: dict,
        site_names: list[str],
        *,
        payload_key: str,
        dtype,
        fillvalue,
    ) -> None:
        first_payload = dict(first_tactile.get(payload_key, {}))
        if not first_payload:
            return

        payload_grp = tactile_grp.create_group(payload_key)
        for site_name in site_names:
            if site_name not in first_payload:
                raise ValueError(f"TacMap first frame is missing {payload_key} site: {site_name}")
            first_arr = np.asarray(first_payload[site_name], dtype=dtype)
            if first_arr.ndim != 2:
                raise ValueError(
                    f"TacMap {payload_key} site '{site_name}' must be a 2-D image, got {first_arr.shape}."
                )

            data = np.full((buffer.frame_count, *first_arr.shape), fillvalue, dtype=dtype)
            for i, frame in enumerate(buffer.frames):
                tactile = None if frame.robot is None else frame.robot.get("tactile")
                if tactile is None:
                    continue
                payload = tactile.get(payload_key, {}).get(site_name)
                if payload is None:
                    continue
                arr = np.asarray(payload, dtype=dtype)
                if arr.shape != first_arr.shape:
                    raise ValueError(
                        f"TacMap {payload_key} site '{site_name}' shape changed from {first_arr.shape} to {arr.shape}."
                    )
                data[i] = arr
            payload_grp.create_dataset(site_name, data=data, compression="gzip", compression_opts=4)

    def _write_objects(self, file: h5py.File, buffer: EpisodeBuffer) -> None:
        objects_grp = file.create_group("objects")
        str_dtype = h5py.string_dtype(encoding="utf-8")
        for obj_id in buffer.object_ids:
            obj_grp = objects_grp.create_group(obj_id)
            pose = np.full((buffer.frame_count, 7), np.nan, dtype=np.float32)
            lin_vel = np.full((buffer.frame_count, 3), np.nan, dtype=np.float32)
            ang_vel = np.full((buffer.frame_count, 3), np.nan, dtype=np.float32)
            first_joint_state = next(
                (frame.objects.get(obj_id) for frame in buffer.frames if frame.objects.get(obj_id, {}).get("joint_names")),
                None,
            )
            joint_names = list(first_joint_state.get("joint_names", [])) if first_joint_state is not None else []
            qpos = np.full((buffer.frame_count, len(joint_names)), np.nan, dtype=np.float32) if joint_names else None
            qvel = np.full((buffer.frame_count, len(joint_names)), np.nan, dtype=np.float32) if joint_names else None
            has_qeffort = any(
                frame.objects.get(obj_id, {}).get("qeffort") is not None for frame in buffer.frames
            )
            qeffort = (
                np.full((buffer.frame_count, len(joint_names)), np.nan, dtype=np.float32)
                if joint_names and has_qeffort
                else None
            )
            for i, frame in enumerate(buffer.frames):
                state = frame.objects.get(obj_id)
                if state is None:
                    continue
                pose[i] = state["pose_world"]
                lin_vel[i] = state["lin_vel_world"]
                ang_vel[i] = state["ang_vel_world"]
                if joint_names:
                    qpos[i] = np.asarray(state.get("qpos", np.full((len(joint_names),), np.nan, dtype=np.float32)))
                    qvel[i] = np.asarray(state.get("qvel", np.full((len(joint_names),), np.nan, dtype=np.float32)))
                    if qeffort is not None and state.get("qeffort") is not None:
                        qeffort[i] = np.asarray(state["qeffort"], dtype=np.float32)
            obj_grp.create_dataset("pose_world", data=pose)
            obj_grp.create_dataset("lin_vel_world", data=lin_vel)
            obj_grp.create_dataset("ang_vel_world", data=ang_vel)
            if joint_names:
                obj_grp.create_dataset("joint_names", data=np.asarray(joint_names, dtype=str_dtype))
                obj_grp.create_dataset("qpos", data=qpos)
                obj_grp.create_dataset("qvel", data=qvel)
                if qeffort is not None:
                    obj_grp.create_dataset("qeffort", data=qeffort)

    @staticmethod
    def _write_metrics_group(
        file: h5py.File,
        metrics_episode,
        metrics_timeseries,
        str_dtype,
    ) -> None:
        """Write /metrics/episode and /metrics/timeseries from meta payload."""
        if not metrics_episode and not metrics_timeseries:
            return

        metrics_grp = file.create_group("metrics")

        if metrics_episode:
            ep_grp = metrics_grp.create_group("episode")
            for key, value in metrics_episode.items():
                if value is None:
                    ep_grp.create_dataset(key, data=np.asarray("null", dtype=str_dtype))
                elif isinstance(value, bool):
                    ep_grp.create_dataset(key, data=np.asarray(value, dtype=np.bool_))
                elif isinstance(value, int):
                    ep_grp.create_dataset(key, data=np.asarray(value, dtype=np.int64))
                elif isinstance(value, float):
                    ep_grp.create_dataset(key, data=np.asarray(value, dtype=np.float64))
                else:
                    ep_grp.create_dataset(key, data=np.asarray(json.dumps(value), dtype=str_dtype))

        if metrics_timeseries:
            ts_grp = metrics_grp.create_group("timeseries")
            for key, values in metrics_timeseries.items():
                if isinstance(values, list) and values:
                    if any(isinstance(v, str) or v is None for v in values):
                        arr = np.asarray(["" if v is None else str(v) for v in values], dtype=str_dtype)
                    elif any(isinstance(v, bool) or v is None for v in values):
                        arr = np.asarray([False if v is None else bool(v) for v in values], dtype=np.bool_)
                    elif any(isinstance(v, float) or v is None for v in values):
                        arr = np.asarray([np.nan if v is None else float(v) for v in values], dtype=np.float64)
                    elif any(isinstance(v, int) or v is None for v in values):
                        arr = np.asarray([-1 if v is None else int(v) for v in values], dtype=np.int64)
                    else:
                        arr = np.asarray(values)
                    ts_grp.create_dataset(key, data=arr)

    def _write_labels(self, file: h5py.File, buffer: EpisodeBuffer) -> None:
        labels_grp = file.require_group("labels")

        box3d_grp = labels_grp.create_group("box3d")
        for obj_id in buffer.object_ids:
            obj_grp = box3d_grp.create_group(obj_id)
            center = np.full((buffer.frame_count, 3), np.nan, dtype=np.float32)
            size = np.full((buffer.frame_count, 3), np.nan, dtype=np.float32)
            quat = np.full((buffer.frame_count, 4), np.nan, dtype=np.float32)
            for i, frame in enumerate(buffer.frames):
                box3d = frame.labels.get("box3d", {}).get(obj_id)
                if box3d is None:
                    continue
                center[i] = box3d["center_world"]
                size[i] = box3d["size_lwh"]
                quat[i] = box3d["quat_world"]
            obj_grp.create_dataset("center_world", data=center)
            obj_grp.create_dataset("size_lwh", data=size)
            obj_grp.create_dataset("quat_world", data=quat)

        has_box2d = any(frame.labels.get("box2d") for frame in buffer.frames)
        if has_box2d:
            present_camera_ids = sorted(
                {cam_id for frame in buffer.frames for cam_id in frame.labels.get("box2d", {}).keys()}
            )
            if present_camera_ids:
                box2d_grp = labels_grp.create_group("box2d")
                for cam_id in present_camera_ids:
                    cam_grp = box2d_grp.create_group(cam_id)
                    for obj_id in buffer.object_ids:
                        obj_grp = cam_grp.create_group(obj_id)
                        xyxy = np.full((buffer.frame_count, 4), -1, dtype=np.int32)
                        visible = np.zeros((buffer.frame_count,), dtype=np.bool_)
                        for i, frame in enumerate(buffer.frames):
                            box2d = frame.labels.get("box2d", {}).get(cam_id, {}).get(obj_id)
                            if box2d is None:
                                continue
                            xyxy[i] = box2d["xyxy"]
                            visible[i] = bool(box2d["visible"])
                        obj_grp.create_dataset("xyxy", data=xyxy)
                        obj_grp.create_dataset("visible", data=visible)

        self._write_occupancy_labels(labels_grp, buffer)

    @staticmethod
    def _first_occupancy(buffer: EpisodeBuffer):
        for frame in buffer.frames:
            occupancy = frame.labels.get("occupancy")
            if occupancy and "state" in occupancy:
                return occupancy
        return None

    def _write_occupancy_labels(self, labels_grp: h5py.Group, buffer: EpisodeBuffer) -> None:
        first_occupancy = self._first_occupancy(buffer)
        if first_occupancy is None:
            return

        occupancy_grp = labels_grp.create_group("occupancy_tsdf")
        grid_shape = np.asarray(first_occupancy["grid_shape"], dtype=np.int32)
        occupancy_grp.create_dataset("grid_shape", data=grid_shape)
        occupancy_grp.create_dataset("bounds", data=np.asarray(first_occupancy["bounds"], dtype=np.float32))
        occupancy_grp.create_dataset("voxel_size", data=np.asarray(first_occupancy["voxel_size"], dtype=np.float32))

        str_dtype = h5py.string_dtype(encoding="utf-8")
        occupancy_grp.create_dataset("method", data=np.asarray("sensor_depth_tsdf", dtype=str_dtype))
        occupancy_grp.create_dataset("semantics", data=np.asarray("observed_three_state", dtype=str_dtype))

        state = np.zeros((buffer.frame_count, *grid_shape.tolist()), dtype=np.uint8)
        frame_valid = np.zeros((buffer.frame_count,), dtype=np.bool_)
        for i, frame in enumerate(buffer.frames):
            frame_occupancy = frame.labels.get("occupancy")
            if not frame_occupancy or "state" not in frame_occupancy:
                continue
            state[i] = np.asarray(frame_occupancy["state"], dtype=np.uint8)
            frame_valid[i] = True
        occupancy_grp.create_dataset("state", data=state, compression="gzip", compression_opts=4)
        occupancy_grp.create_dataset("frame_valid", data=frame_valid)
