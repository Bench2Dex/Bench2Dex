"""Top-level data collector orchestration."""

from __future__ import annotations

import copy
import os
import time
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict

import isaaclab.sim as sim_utils
import numpy as np
import torch

from build.generalization import relativize_sample_paths as _relativize_gen_sample
from .box_labeler import compute_box2d_labels, compute_box3d_labels
from .cameras import CameraRig
from .config import CollectConfig
from .episode_buffer import EpisodeBuffer, FrameRecord
from .hdf5_writer import HDF5EpisodeWriter
from .metrics_payload import build_metrics_episode_payload
from .state_reader import read_joint_limits, read_joint_state, read_object_states, select_robot_articulation
from .success_tracker import SuccessTracker
from .tacmap_configs import ROBOT_KEY_TO_TACMAP_CFG
from .tacmap_rig import TacMapRig, build_tacmap_raycast_targets
from .task_manifest import get_task_manifest_entry
try:
    from benchmark.metric_tracker import MetricTracker
except ImportError:
    MetricTracker = None  # type: ignore[assignment,misc]
from .tactile import SUPPORTED_TACTILE_ROBOT_KEYS, attach_tactile_to_robot_state


def _is_geometry_capable(cam) -> bool:
    """Return True if *cam* can participate in geometric label generation (box2d, occupancy)."""
    camera_type = str(getattr(cam, "camera_type", "pinhole"))
    if camera_type == "pinhole":
        return True
    if camera_type == "fisheye":
        return (
            getattr(cam, "fisheye_camera_matrix", None) is not None
            and getattr(cam, "fisheye_distortion_coefficients", None) is not None
        )
    return False


class DataCollector:
    """Collect simulation data and persist one episode as one HDF5 file."""

    def __init__(
        self,
        sim: sim_utils.SimulationContext,
        collect_cfg: CollectConfig,
        task_path: str,
        runtime: Dict,
        interactive_objects: Dict[str, object],
        seed: int | None = None,
        base_seed: int | None = None,
        task_seed_id: int | None = None,
        task_base_seed: int | None = None,
        seed_policy: str | None = None,
        seed_namespace: str | None = None,
        episode_step_budget: int | None = None,
        contact_reader: Any | None = None,
    ):
        if collect_cfg.dataset_format != "hdf5_per_episode":
            raise ValueError(f"Unsupported dataset format: {collect_cfg.dataset_format}")

        self._sim = sim
        self._cfg = collect_cfg
        self._task_path = os.path.abspath(task_path)
        self._seed = seed
        self._base_seed = base_seed
        self._task_seed_id = task_seed_id
        self._task_base_seed = task_base_seed
        self._seed_policy = seed_policy
        self._seed_namespace = seed_namespace
        self._episode_index: int | None = None
        self._episode_seed: int | None = seed
        self._interactive_objects = interactive_objects
        self._step_stride = max(1, collect_cfg.step_stride)
        self._flushed = False
        self._closed = False
        self._episode_step_budget = episode_step_budget
        self._skip_before_step_once_at: int | None = None
        self._metric_update_count = 0
        self._last_metric_sim_step: int | None = None

        local_bbox_raw = runtime.get("asset_local_bbox", {})
        self._local_bboxes = {
            obj_id: (tuple(bounds[0]), tuple(bounds[1])) for obj_id, bounds in local_bbox_raw.items()
        }
        self._object_ids = sorted(self._local_bboxes.keys())
        if not self._object_ids:
            self._object_ids = sorted(obj_id for obj_id in interactive_objects.keys() if obj_id != "global_robot")

        self._object_body_types = dict(runtime.get("object_body_types", {}))
        self._object_asset_paths = dict(runtime.get("object_asset_paths", {}))
        self._object_asset_keys = dict(runtime.get("object_asset_keys", {}))
        self._scene_generalization_config = dict(runtime.get("scene_generalization_config", {}))
        self._scene_generalization_sample = dict(runtime.get("scene_generalization_sample", {}))
        self._robot_key = runtime.get("robot_key")
        self._distractor_object_ids = set(runtime.get("distractor_object_ids", []))
        self._contact_reader = contact_reader

        # Load success conditions and instruction from task YAML (passed via runtime)
        self._success_conditions = runtime.get("success_conditions", [])
        self._task_instruction = runtime.get("task_instruction", None)
        self._metrics_spec = runtime.get("metrics", {})
        self._runtime_table_z = runtime.get("table_z")

        self._validate_modality_requirements()
        self._enforce_memory_budget()

        self._robot_articulation = select_robot_articulation(interactive_objects)
        self._geometry_compatible_camera_ids = [
            cam.camera_id for cam in collect_cfg.cameras if _is_geometry_capable(cam)
        ]

        self._camera_rig: CameraRig | None = None
        self._camera_ids: list[str] = []

        self._tactile_rig = self._make_tactile_rig(runtime) if collect_cfg.enabled("tactile") else None

        self._episode_active = False
        self._buffer = self._make_episode_buffer()
        task_name = os.path.basename(os.path.dirname(self._task_path)) or "task"
        scene_name = os.path.splitext(os.path.basename(self._task_path))[0]
        self._scene_name = scene_name
        self._task_manifest_entry = get_task_manifest_entry(scene_name)
        self._writer = HDF5EpisodeWriter(
            dataset_root=collect_cfg.dataset_root,
            dataset_name=collect_cfg.dataset_name,
            task_name=task_name,
            scene_name=scene_name,
        )

    def _sampled_frame_budget(self) -> int | None:
        if self._episode_step_budget is None or self._episode_step_budget <= 0:
            return None
        return max(1, int(np.ceil(float(self._episode_step_budget) / float(self._step_stride))))

    def _make_episode_buffer(self) -> EpisodeBuffer:
        sampled_budget = self._sampled_frame_budget()
        max_frames = sampled_budget if sampled_budget is not None and sampled_budget > 0 else 0
        return EpisodeBuffer(camera_ids=list(self._camera_ids), object_ids=self._object_ids, max_frames=max_frames)

    def _reset_episode_storage(self) -> None:
        self._buffer = self._make_episode_buffer()
        self._flushed = False
        self._skip_before_step_once_at = None
        self._metric_update_count = 0
        self._last_metric_sim_step = None

    @property
    def episode_active(self) -> bool:
        return self._episode_active

    def _validate_modality_requirements(self) -> None:
        if self._cfg.enabled("box2d") and not any(
            _is_geometry_capable(cam) for cam in self._cfg.cameras
        ):
            raise ValueError("box2d requires at least one geometry-compatible camera (pinhole or fisheye with calibration).")

    def _estimate_buffer_bytes(self, frame_budget: int) -> int:
        total_bytes = 0
        camera_frame_bytes = 0
        for cam in self._cfg.cameras:
            pixel_count = int(cam.width) * int(cam.height)
            if self._cfg.enabled("rgb"):
                camera_frame_bytes += pixel_count * 3
            if self._cfg.enabled("depth"):
                camera_frame_bytes += pixel_count * 4
        total_bytes += frame_budget * camera_frame_bytes
        if self._cfg.enabled("tactile"):
            tacmap_cfg = ROBOT_KEY_TO_TACMAP_CFG.get(str(self._robot_key))
            if tacmap_cfg is not None:
                tactile_frame_bytes = len(tacmap_cfg.site_names) * int(tacmap_cfg.native_resolution) ** 2
                total_bytes += frame_budget * tactile_frame_bytes
        return int(total_bytes)

    def _enforce_memory_budget(self) -> None:
        max_buffer_mb = int(self._cfg.capture_max_buffer_mb)
        if max_buffer_mb <= 0:
            return
        frame_budget = self._sampled_frame_budget()
        if frame_budget is None:
            return
        estimated_bytes = self._estimate_buffer_bytes(frame_budget)
        budget_bytes = max_buffer_mb * 1024 * 1024
        if estimated_bytes > budget_bytes:
            estimated_mb = estimated_bytes / (1024.0 * 1024.0)
            raise ValueError(
                f"Estimated collector buffer {estimated_mb:.1f} MB exceeds capture.max_buffer_mb={max_buffer_mb}."
            )

    @property
    def frame_count(self) -> int:
        return self._buffer.frame_count

    def _effective_capture_fps(self) -> float:
        physics_dt = float(self._sim.get_physics_dt())
        if physics_dt <= 0.0 or self._step_stride <= 0:
            return 0.0
        return 1.0 / (physics_dt * float(self._step_stride))

    def _refresh_robot_articulation(self) -> object | None:
        self._robot_articulation = select_robot_articulation(self._interactive_objects)
        if self._camera_rig is not None:
            self._camera_rig.set_robot_articulation(self._robot_articulation)
        return self._robot_articulation

    def _make_camera_rig(self) -> CameraRig | None:
        enable_rgb = self._cfg.enabled("rgb")
        enable_depth = self._cfg.enabled("depth")
        if not (enable_rgb or enable_depth):
            return None
        return CameraRig(
            self._sim,
            self._cfg.cameras,
            enable_rgb,
            enable_depth,
            robot_articulation=self._robot_articulation,
            camera_generalization_sample=self._scene_generalization_sample,
        )

    def prepare_scene_rebuild(self) -> None:
        """Release stage-bound sensors before their USD prims are deleted."""
        if self._episode_active:
            raise RuntimeError("Cannot rebuild the scene while an episode is still recording.")
        if self._camera_rig is not None:
            self._camera_rig.close()
            self._camera_rig = None
        if self._tactile_rig is not None:
            self._tactile_rig.close()
            self._tactile_rig = None
        if self._contact_reader is not None:
            close = getattr(self._contact_reader, "close", None)
            if callable(close):
                close()
            self._contact_reader = None
        self._camera_ids = []
        self._robot_articulation = None

    def refresh_scene_runtime(self, runtime: Dict, interactive_objects: Dict[str, object]) -> None:
        self._interactive_objects = interactive_objects

        local_bbox_raw = runtime.get("asset_local_bbox", {})
        self._local_bboxes = {
            obj_id: (tuple(bounds[0]), tuple(bounds[1])) for obj_id, bounds in local_bbox_raw.items()
        }
        self._object_ids = sorted(self._local_bboxes.keys())
        if not self._object_ids:
            self._object_ids = sorted(obj_id for obj_id in interactive_objects.keys() if obj_id != "global_robot")

        self._object_body_types = dict(runtime.get("object_body_types", {}))
        self._object_asset_paths = dict(runtime.get("object_asset_paths", {}))
        self._object_asset_keys = dict(runtime.get("object_asset_keys", {}))
        self._scene_generalization_config = dict(runtime.get("scene_generalization_config", {}))
        self._scene_generalization_sample = dict(runtime.get("scene_generalization_sample", {}))
        self._robot_key = runtime.get("robot_key")
        self._distractor_object_ids = set(runtime.get("distractor_object_ids", []))
        self._success_conditions = runtime.get("success_conditions", [])
        self._task_instruction = runtime.get("task_instruction", None)
        self._metrics_spec = runtime.get("metrics", {})
        self._runtime_table_z = runtime.get("table_z")

        self._refresh_robot_articulation()
        if self._camera_rig is not None:
            self._camera_rig.close()
            self._camera_rig = None
        self._camera_ids = []

        if self._tactile_rig is not None:
            self._tactile_rig.close()
        if self._cfg.enabled("tactile"):
            self._tactile_rig = self._make_tactile_rig(runtime)
        else:
            self._tactile_rig = None

        self._episode_active = False
        self._reset_episode_storage()

    def create_camera_rig_before_reset(self) -> None:
        """Create CameraRig after robot spawn but before ``sim.reset()``.

        Native-mounted cameras are parented under articulation link prims. They
        must exist before IsaacLab/PhysX initializes the articulation, otherwise
        the child prim can remain stuck at the link's authored USD pose instead
        of following the runtime link transform.
        """
        if self._camera_rig is not None:
            return
        self._refresh_robot_articulation()
        self._camera_rig = self._make_camera_rig()
        self._camera_ids = self._camera_rig.camera_ids if self._camera_rig is not None else []
        self._buffer = self._make_episode_buffer()

    def sync_mounted_camera_poses(self) -> None:
        """Synchronize robot-link cameras before rendering the current state."""
        if self._camera_rig is not None:
            self._camera_rig.sync_mounted_camera_poses()

    def set_episode_seed_context(self, seed_context) -> None:
        self._seed_namespace = getattr(seed_context, "seed_namespace", self._seed_namespace)
        self._base_seed = getattr(seed_context, "base_seed", self._base_seed)
        self._task_seed_id = getattr(seed_context, "task_seed_id", self._task_seed_id)
        self._task_base_seed = getattr(seed_context, "task_base_seed", self._task_base_seed)
        self._episode_index = getattr(seed_context, "episode_index", self._episode_index)
        self._episode_seed = getattr(seed_context, "episode_seed", self._episode_seed)
        self._seed_policy = getattr(seed_context, "seed_policy", self._seed_policy)
        self._seed = self._episode_seed

    def _make_tactile_rig(self, runtime: Dict) -> TacMapRig:
        object_prim_paths, object_body_types = build_tacmap_raycast_targets(
            dict(runtime.get("object_prim_paths", {})),
            dict(runtime.get("object_body_types", {})),
        )
        return TacMapRig(
            robot_key=str(runtime.get("robot_key", SUPPORTED_TACTILE_ROBOT_KEYS[0])),
            object_prim_paths=object_prim_paths,
            object_body_types=object_body_types,
            robot_prim_path=str(runtime.get("robot_pose", {}).get("prim_path", "/World/Objects/GlobalRobot")),
        )

    def initialize_after_reset(self) -> None:
        robot_articulation = self._refresh_robot_articulation()
        if self._cfg.enabled("joint_state") and robot_articulation is None:
            print("[WARN] joint_state is enabled but robot articulation is unavailable after reset.")
        if self._camera_rig is None and (self._cfg.enabled("rgb") or self._cfg.enabled("depth")):
            raise RuntimeError(
                "CameraRig is missing after sim.reset(). Create it before reset with "
                "DataCollector.create_camera_rig_before_reset(); creating robot-link "
                "cameras after articulation initialization causes stale wrist-camera poses."
            )
        if self._camera_rig is not None:
            self._camera_rig.initialize_after_reset()
        tactile_rig = getattr(self, "_tactile_rig", None)
        if tactile_rig is not None:
            tactile_rig.initialize_after_reset(robot_articulation)

        self._success_tracker = None

        # Initialize metric tracker if metrics spec is available
        metrics_spec = getattr(self, '_metrics_spec', None)
        if MetricTracker is not None and metrics_spec:
            self._metric_tracker = MetricTracker(
                self._metrics_spec_for_episode(metrics_spec),
                dt=self._sim.get_physics_dt(),
                robot_key=self._robot_key,
            )
        else:
            self._metric_tracker = None

        print(
            "[INFO] Collector ready: "
            f"objects={len(self._object_ids)}, cameras={len(self._buffer.camera_ids)}, stride={self._step_stride}"
        )

    def start_episode(self, sim_step: int = 0) -> None:
        if self._closed:
            raise RuntimeError("Cannot start a new episode after collector.close().")
        if self._episode_active:
            print("[WARN] Collector episode already active; ignoring start request.")
            return
        self._reset_episode_storage()
        # Reset metric tracker for new episode
        metrics_spec = getattr(self, '_metrics_spec', None)
        if MetricTracker is not None and metrics_spec:
            self._metric_tracker = MetricTracker(
                self._metrics_spec_for_episode(metrics_spec),
                dt=self._sim.get_physics_dt(),
                robot_key=self._robot_key,
            )
        self._homing_start_sim_step: int | None = None  # 本 episode 内首次 homing 的 sim_step
        self._episode_active = True
        self.capture_initial_observation(sim_step=sim_step)
        self._skip_before_step_once_at = sim_step

    def set_homing_start_sim_step(self, sim_step: int) -> None:
        """录制期间触发 homing 时记录当前仿真步，供后续 HDF5 写入用于数据分割。"""
        if not hasattr(self, '_homing_start_sim_step') or self._homing_start_sim_step is None:
            self._homing_start_sim_step = sim_step

    def finish_episode(self) -> str | None:
        if not self._episode_active:
            return None
        episode_path = self.flush_episode()
        self._episode_active = False
        self._reset_episode_storage()
        return episode_path

    def discard_episode(self) -> None:
        """Discard the current episode without saving."""
        if not self._episode_active:
            return
        self._episode_active = False
        self._reset_episode_storage()

    def capture_initial_observation(self, sim_step: int = 0) -> None:
        """Capture the initial observation before any action is taken.

        This creates the first frame with no action (action_commanded=None).
        Used at the start of an episode to capture the initial state.

        Args:
            sim_step: Current simulation step (default 0)
        """
        if not self._episode_active:
            return
        # Capture observation without action - initial frame has no action source
        self._capture_observation(
            sim_step,
            action_commanded=None,
            action_source="none",
            teleop_snapshot=None,
            force_capture=True,
        )

    def before_step(
        self,
        sim_step: int,
        hold_targets: dict,
        controlled_articulations: list,
        action_source: str = "scripted",
    ) -> None:
        """Capture (obs_t, action_t) before physics step — Convention A.

        action_t is the target that **will be executed** in the upcoming
        sim.step().  The caller must ensure teleop targets have already been
        written into *hold_targets* before calling this method.

        Args:
            sim_step: Current simulation step.
            hold_targets: Dictionary mapping articulation id to joint target tensor.
            controlled_articulations: List of articulation objects.
            action_source: Action source label ("teleop", "scripted", …).
        """
        if not self._episode_active:
            return
        if self._skip_before_step_once_at == sim_step:
            self._skip_before_step_once_at = None
            return
        # Extract joint targets from hold_targets
        action_commanded = None
        if controlled_articulations:
            for articulation in controlled_articulations:
                target = hold_targets.get(id(articulation))
                if target is not None:
                    if torch.is_tensor(target):
                        action_commanded = target.detach().cpu().numpy().squeeze(0).copy()
                    else:
                        action_commanded = np.asarray(target, dtype=np.float32).copy()
                    break

        self._capture_observation(
            sim_step,
            action_commanded=action_commanded,
            action_source=action_source,
            teleop_snapshot=None,
        )

    def after_step(self, sim_step: int, dt: float | None = None) -> None:
        """Hook called after physics step. Updates metric tracker if available."""
        if self._metric_tracker is not None and self._metric_tracker.available:
            object_states = read_object_states(self._interactive_objects, self._object_ids)
            self._merge_contact_forces(object_states, dt)
            robot_state = None
            joint_limits = None
            if self._robot_articulation is not None:
                try:
                    robot_state = read_joint_state(self._robot_articulation)
                    joint_limits = read_joint_limits(self._robot_articulation)
                except Exception:
                    pass
            self._metric_tracker.update(
                object_states,
                sim_step=sim_step,
                dt=dt,
                robot_state=robot_state,
                joint_limits=joint_limits,
            )
            self._metric_update_count += 1
            self._last_metric_sim_step = int(sim_step)

    def _merge_contact_forces(self, object_states: Dict[str, Dict], dt: float | None = None) -> None:
        """Merge per-pair contact forces from ContactSensorReader into object_states."""
        if self._contact_reader is None or not self._contact_reader.available:
            return
        self._contact_reader.update(dt or 1.0 / 60.0)
        contact_data = self._contact_reader.read()
        for obj_id, forces in contact_data.items():
            if obj_id in object_states:
                object_states[obj_id]["contact_forces"] = forces

    def _metrics_spec_for_episode(self, metrics_spec: dict) -> dict:
        """Return episode-local metrics with runtime table_z as authoritative."""
        spec = copy.deepcopy(metrics_spec or {})
        if self._runtime_table_z is None:
            return spec

        table_z = float(self._runtime_table_z)
        safety = spec.setdefault("safety", {})
        if isinstance(safety, dict):
            safety["table_z"] = table_z
        grasp = spec.get("grasp")
        if isinstance(grasp, dict):
            grasp["table_z"] = table_z
        return spec

    def set_contact_reader(self, contact_reader: Any | None) -> None:
        """Set or replace the ContactSensorReader (call after sim.reset())."""
        if self._contact_reader is not None and self._contact_reader is not contact_reader:
            close = getattr(self._contact_reader, "close", None)
            if callable(close):
                close()
        self._contact_reader = contact_reader

    def _capture_observation(
        self,
        sim_step: int,
        action_commanded,
        action_source,
        teleop_snapshot,
        force_capture: bool = False,
    ) -> None:
        """Internal method to capture observation and optionally store action."""
        if not force_capture and sim_step % self._step_stride != 0:
            return

        valid = True
        errors: list[str] = []
        camera_frames: Dict = {}
        camera_failed = False
        object_states: Dict = {}
        robot_state: Dict | None = None
        box3d_labels: Dict = {}
        box2d_labels: Dict = {}

        if self._cfg.enabled("object_pose") or self._cfg.enabled("box3d") or self._cfg.enabled("box2d"):
            try:
                object_states = read_object_states(self._interactive_objects, self._object_ids)
            except Exception as exc:
                valid = False
                errors.append(f"object_state: {exc}")

        if self._cfg.enabled("joint_state"):
            if self._robot_articulation is None:
                valid = False
                errors.append("joint_state: robot articulation unavailable")
            else:
                try:
                    robot_state = read_joint_state(self._robot_articulation)
                except Exception as exc:
                    valid = False
                    errors.append(f"joint_state: {exc}")

        if self._camera_rig is not None:
            try:
                camera_frames = self._camera_rig.capture(dt=self._sim.get_physics_dt())
                camera_errors = list(self._camera_rig.last_capture_errors)
                if camera_errors:
                    camera_failed = True
                    valid = False
                    errors.extend(f"camera: {msg}" for msg in camera_errors)
            except Exception as exc:
                camera_failed = True
                valid = False
                errors.append(f"camera: {exc}")

        tactile_rig = getattr(self, "_tactile_rig", None)
        if tactile_rig is not None:
            try:
                tactile_payload = tactile_rig.capture(dt=self._sim.get_physics_dt())
                robot_state = attach_tactile_to_robot_state(robot_state, tactile_payload)
            except Exception as exc:
                valid = False
                errors.append(f"tactile: {exc}")

        if self._cfg.enabled("box3d"):
            try:
                box3d_labels = compute_box3d_labels(object_states, self._local_bboxes)
            except Exception as exc:
                valid = False
                errors.append(f"box3d: {exc}")

        if self._cfg.enabled("box2d") and box3d_labels and camera_frames:
            try:
                box2d_labels = compute_box2d_labels(box3d_labels=box3d_labels, camera_frames=camera_frames)
            except Exception as exc:
                valid = False
                errors.append(f"box2d: {exc}")

        frame_count = self._buffer.frame_count
        self._buffer.add_frame(
            FrameRecord(
                frame_index=frame_count,
                timestamp_ns=time.time_ns(),
                sim_step=sim_step,
                valid=valid,
                errors=errors,
                camera=camera_frames,
                robot=robot_state,
                objects=object_states,
                labels={
                    "box3d": box3d_labels,
                    "box2d": box2d_labels,
                },
                action_commanded=action_commanded,
                action_valid=action_commanded is not None,
                action_source=action_source,
                teleop_snapshot=teleop_snapshot,
            )
        )

    def _get_robot_joint_names(self) -> list[str]:
        """Get robot joint names for action schema."""
        if self._robot_articulation is None:
            return []
        try:
            return list(self._robot_articulation.data.joint_names)
        except Exception:
            return []

    def _object_roles(self) -> Dict[str, str]:
        roles: Dict[str, str] = {}
        distractor_ids = set(getattr(self, "_distractor_object_ids", set()) or set())
        for obj_id in self._object_ids:
            if obj_id in distractor_ids:
                roles[obj_id] = "distractor"
                continue
            body_type = str(self._object_body_types.get(obj_id, "dynamic")).lower()
            if body_type == "articulation":
                roles[obj_id] = "task_articulation"
            elif body_type == "deformable":
                roles[obj_id] = "task_deformable"
            else:
                roles[obj_id] = "task_object"
        return roles

    def close(self) -> str | None:
        if self._closed:
            return None

        episode_path = None
        try:
            if self._episode_active:
                episode_path = self.finish_episode()
            elif self._buffer.frame_count > 0 and not self._flushed:
                episode_path = self.flush_episode()
        finally:
            if self._camera_rig is not None:
                self._camera_rig.close()
                self._camera_rig = None
            if self._tactile_rig is not None:
                self._tactile_rig.close()
                self._tactile_rig = None
            self._closed = True
        return episode_path

    def flush_episode(self) -> str | None:
        if self._flushed:
            return None
        self._flushed = True
        if self._buffer.frame_count <= 0:
            print("[WARN] Collector flush skipped: no frames captured.")
            return None

        episode_id = self._writer.next_episode_id()

        # Build v2 metadata
        meta = self._build_episode_metadata()

        path = self._writer.write_episode(self._buffer, episode_id=episode_id, meta=meta)
        print(f"[INFO] Episode written: {path}")
        return path

    def _build_episode_metadata(self) -> dict:
        """Build episode metadata including v2 schema fields."""
        # Resolve instruction from scene YAML (passed via runtime)
        instruction = getattr(self, '_task_instruction', None)

        # Get success status from metric tracker
        success = False
        metrics_episode = None
        metrics_timeseries = None
        if hasattr(self, '_metric_tracker') and self._metric_tracker is not None and self._metric_tracker.available:
            terminated_reason = self._infer_metric_termination_reason()
            metric_result = self._metric_tracker.finalize(
                steps=self._metric_update_count,
                terminated_reason=terminated_reason,
            )
            success = metric_result.stable_success
            metrics_episode = build_metrics_episode_payload(metric_result)
            metrics_timeseries = metric_result.timeseries
        elif hasattr(self, '_success_tracker') and self._success_tracker is not None:
            success = self._success_tracker.success

        # Determine demo eligibility (scripted = not eligible)
        demo_eligible = False

        # Get action names from robot articulation (canonical source)
        action_names = self._get_robot_joint_names()
        action_dim = len(action_names)

        meta = {
            # v2 schema version
            "schema_version": "raw_hdf5_v2",
            # Collection mode
            "collection_mode": "scripted_hold",
            "source_domain": "sim",
            # Instruction
            "instruction": instruction,
            "has_instruction": instruction is not None,
            # Action metadata - canonical action schema (independent of observation modalities)
            "control_mode": "joint_position",
            "action_type": "absolute",
            "action_dim": action_dim,
            "action_names": action_names,
            "has_action": True,
            # Dataset eligibility
            "demo_eligible": demo_eligible,
            # Modalities
            "has_rgb": self._cfg.enabled("rgb"),
            "has_state": self._cfg.enabled("joint_state") or self._cfg.enabled("object_pose"),
            # Success tracking
            "success_available": bool(
                (hasattr(self, '_metric_tracker') and self._metric_tracker is not None and self._metric_tracker.available)
                or (getattr(self, "_success_tracker", None) is not None and self._success_tracker.available)
            ),
            "success": success,
            "metrics_episode": metrics_episode,
            "metrics_timeseries": metrics_timeseries,
            # Original fields
            "task_name": os.path.basename(os.path.dirname(self._task_path)) or "task",
            "scene_file": self._task_path,
            "scene_name": self._scene_name,
            "seed": int(self._episode_seed if self._episode_seed is not None else (self._seed or 0)),
            "base_seed": self._base_seed,
            "task_seed_id": self._task_seed_id,
            "task_base_seed": self._task_base_seed,
            "episode_index": self._episode_index,
            "episode_seed": self._episode_seed,
            "seed_policy": self._seed_policy,
            "seed_namespace": self._seed_namespace,
            "fps": int(self._cfg.capture_fps),
            "physics_dt": float(self._sim.get_physics_dt()),
            "effective_fps": float(self._effective_capture_fps()),
            "step_stride": int(self._step_stride),
            "dataset_version": self._cfg.dataset_version,
            "sensor_profile_id": self._cfg.sensor_profile_id,
            "sensor_profile_version": self._cfg.sensor_profile_version,
            "modalities": [name for name, enabled in self._cfg.modalities.items() if enabled],
            "camera_definitions": [asdict(cam) for cam in self._cfg.cameras],
            "pinhole_camera_ids": [cam.camera_id for cam in self._cfg.cameras if str(getattr(cam, "camera_type", "pinhole")) == "pinhole"],
            "fisheye_camera_ids": [cam.camera_id for cam in self._cfg.cameras if str(getattr(cam, "camera_type", "pinhole")) == "fisheye"],
            "box2d_camera_ids": list(self._geometry_compatible_camera_ids),
            "collect_config": self._cfg.to_dict(),
            "local_bboxes": {
                obj_id: [list(bounds[0]), list(bounds[1])] for obj_id, bounds in self._local_bboxes.items()
            },
            "object_body_types": self._object_body_types,
            "object_roles": self._object_roles(),
            "object_asset_paths": self._object_asset_paths,
            "object_asset_keys": self._object_asset_keys,
            "scene_generalization_config": dict(getattr(self, "_scene_generalization_config", {})),
            "scene_generalization_sample": _relativize_gen_sample(
                dict(getattr(self, "_scene_generalization_sample", {}))
            ),
            "robot_key": getattr(self, "_robot_key", None),
            "task_manifest": self._task_manifest_entry,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "frame_error_count": int(self._buffer.error_count),
            "homing_start_sim_step": int(getattr(self, '_homing_start_sim_step', None) or -1),
        }
        return meta

    def _infer_metric_termination_reason(self) -> str:
        if self._episode_step_budget is not None and self._episode_step_budget > 0:
            if self._metric_update_count >= int(self._episode_step_budget):
                return "max_steps"
        return "manual_stop"
