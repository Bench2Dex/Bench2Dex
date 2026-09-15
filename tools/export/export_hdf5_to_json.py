import argparse
import json
from pathlib import Path

import h5py
import numpy as np

DATASET_PREFIXES = {
    "time": "/time/",
    "cameras": "/cameras/",
    "robot": "/robot/",
    "tactile": "/robot/tactile/",
    "objects": "/objects/",
    "box3d": "/labels/box3d/",
    "box2d": "/labels/box2d/",
    "occupancy_tsdf": "/labels/occupancy_tsdf/",
}
RAW_MODALITIES = ("rgb", "depth", "joint_state", "object_pose", "box3d", "box2d", "occupancy_tsdf", "tactile")
DERIVED_MODALITIES = ("occupancy_tsdf",)
STATUS_ORDER = {"pass": 0, "warn": 1, "fail": 2}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export HDF5 summaries and a conservative data-quality audit."
    )
    parser.add_argument("raw_path", type=Path, help="Path to the raw HDF5 file.")
    parser.add_argument("derived_path", type=Path, nargs="?", default=None, help="Optional derived HDF5 file.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory for JSON files.")
    parser.add_argument(
        "--require-raw-modalities",
        nargs="*",
        default=None,
        help="Explicit raw modalities to require: rgb depth joint_state object_pose box3d box2d occupancy_tsdf tactile",
    )
    parser.add_argument(
        "--require-derived-modalities",
        nargs="*",
        default=None,
        help="Explicit derived modalities to require: occupancy_tsdf",
    )
    return parser.parse_args()


def decode_scalar(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray) and value.shape == ():
        return decode_scalar(value.item())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _maybe_json(value):
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or text[0] not in "[{\"0123456789tfn-":
        return value
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def _normalize_modality_name(name):
    text = str(name).strip().lower()
    aliases = {
        "depth_m": "depth",
        "state": "occupancy_tsdf",
        "joint": "joint_state",
        "joint_states": "joint_state",
        "object_states": "object_pose",
        "pose": "object_pose",
    }
    return aliases.get(text, text)


def _normalize_modalities(values, allowed):
    seen, out = set(), []
    for value in values or []:
        item = _normalize_modality_name(value)
        if item in allowed and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _safe_group(group, name):
    if group is None:
        return None
    item = group.get(name)
    return item if isinstance(item, h5py.Group) else None


def _safe_dataset(group, name):
    if group is None:
        return None
    item = group.get(name)
    return item if isinstance(item, h5py.Dataset) else None


def _read_meta_group(file):
    meta = {}
    group = file.get("meta")
    if not isinstance(group, h5py.Group):
        return meta
    for key, item in group.items():
        if not isinstance(item, h5py.Dataset):
            continue
        value = decode_scalar(item[()])
        if isinstance(value, np.ndarray):
            value = [decode_scalar(v) for v in value.tolist()]
        meta[key] = _maybe_json(value)
    return meta


def infer_required_raw_modalities(meta):
    required = []
    collect_config = meta.get("collect_config")
    if isinstance(collect_config, dict):
        modalities = collect_config.get("modalities")
        if isinstance(modalities, dict):
            required.extend(name for name, enabled in modalities.items() if bool(enabled))
        elif isinstance(modalities, list):
            required.extend(modalities)
    if isinstance(meta.get("modalities"), list):
        required.extend(meta["modalities"])
    return _normalize_modalities(required, RAW_MODALITIES)


def infer_required_derived_modalities(meta):
    out = []
    if "occupancy_tsdf" in infer_required_raw_modalities(meta):
        out.append("occupancy_tsdf")
    return _normalize_modalities(out, DERIVED_MODALITIES)


def _base_preview(arr, ds):
    info = {"shape": list(ds.shape), "dtype": str(ds.dtype)}
    if arr.shape == ():
        info["value"] = decode_scalar(arr.item())
        return info
    info["size"] = int(arr.size)
    return info


def _flat_sample(arr, limit=16):
    flat = arr.reshape(-1)
    return [decode_scalar(x) for x in flat[: min(limit, flat.size)].tolist()]


def _occupancy_state_summary(arr, ds):
    info = _base_preview(arr, ds)
    if arr.dtype.kind in "biuf":
        finite = np.isfinite(arr)
        info["finite_ratio"] = float(finite.mean()) if arr.size else 1.0
        if finite.any():
            info["min"] = float(np.nanmin(arr))
            info["max"] = float(np.nanmax(arr))
    info["sample_mode"] = "flat_head_16"
    info["sample"] = _flat_sample(arr)
    counts = {"0": 0, "1": 0, "2": 0}
    if arr.size:
        keys, vals = np.unique(arr.astype(np.uint8, copy=False), return_counts=True)
        counts.update({str(int(k)): int(v) for k, v in zip(keys, vals)})
    total = float(arr.size) if arr.size else 1.0
    info["state_counts"] = counts
    info["state_ratios"] = {key: float(value) / total for key, value in counts.items()}
    info["nonzero_ratio"] = float(counts["1"] + counts["2"]) / total
    return info


def preview_dataset(ds, path=""):
    try:
        arr = np.asarray(ds[()])
    except Exception as exc:
        return {"shape": list(ds.shape), "dtype": str(ds.dtype), "read_error": str(exc)}
    if path == "/labels/occupancy_tsdf/state":
        return _occupancy_state_summary(arr, ds)
    info = _base_preview(arr, ds)
    if arr.shape == ():
        return info
    if arr.dtype.kind in "biuf":
        finite = np.isfinite(arr)
        info["finite_ratio"] = float(finite.mean()) if arr.size else 1.0
        if finite.any():
            info["min"] = float(np.nanmin(arr))
            info["max"] = float(np.nanmax(arr))
    info["sample"] = _flat_sample(arr)
    return info


def summarize_group(group, prefix=""):
    out = {"groups": {}, "datasets": {}}
    for key, item in group.items():
        path = f"{prefix}/{key}" if prefix else f"/{key}"
        if isinstance(item, h5py.Group):
            out["groups"][key] = summarize_group(item, path)
        else:
            out["datasets"][key] = preview_dataset(item, path)
    return out


def collect_dataset_map(group, prefix=""):
    out = {}
    for key, item in group.items():
        path = f"{prefix}/{key}" if prefix else f"/{key}"
        if isinstance(item, h5py.Group):
            out.update(collect_dataset_map(item, path))
        else:
            out[path] = {"shape": list(item.shape), "dtype": str(item.dtype)}
    return out


def build_coverage(dataset_map):
    coverage = {"frame_valid": "/frame_valid" in dataset_map, "frame_errors": "/frame_errors" in dataset_map}
    counts = {"frame_valid": int(coverage["frame_valid"]), "frame_errors": int(coverage["frame_errors"])}
    for key, prefix in DATASET_PREFIXES.items():
        count = sum(1 for path in dataset_map if path.startswith(prefix))
        coverage[key] = count > 0
        counts[key] = count
    return coverage, counts


def count_nodes(group):
    groups, datasets = 0, 0

    def visitor(_name, obj):
        nonlocal groups, datasets
        if isinstance(obj, h5py.Group):
            groups += 1
        else:
            datasets += 1

    group.visititems(visitor)
    return groups, datasets


def summarize_file(path):
    with h5py.File(path, "r") as f:
        dataset_map = collect_dataset_map(f)
        coverage, coverage_counts = build_coverage(dataset_map)
        group_count, dataset_count = count_nodes(f)
        return {
            "file": str(path),
            "size_bytes": path.stat().st_size,
            "export": {
                "type": "summary",
                "contains_full_dataset_values": False,
                "dataset_preview_limit": 16,
                "note": "This JSON is a structural summary. Use the audit JSON to judge whether the episode is complete and usable.",
            },
            "top_level_groups": sorted(list(f.keys())),
            "group_count": group_count,
            "dataset_count": dataset_count,
            "coverage": coverage,
            "coverage_counts": coverage_counts,
            "meta": _read_meta_group(f),
            "summary": summarize_group(f),
            "dataset_schema": dataset_map,
        }


def _empty_result(required):
    return {"required": bool(required), "status": "pass", "issues": [], "stats": {}, "checked_paths": []}


def _update_status(result, status):
    if STATUS_ORDER[status] > STATUS_ORDER[result["status"]]:
        result["status"] = status


def _add_issue(result, severity, code, message, path=None, details=None):
    issue = {"severity": severity, "code": code, "message": message}
    if path is not None:
        issue["path"] = path
    if details is not None:
        issue["details"] = details
    result["issues"].append(issue)
    _update_status(result, severity)


def _sample_frame_indices(frame_count):
    if frame_count <= 0:
        return []
    return sorted({0, frame_count // 2, frame_count - 1})


def _bool_ratio(values):
    return float(np.mean(np.asarray(values, dtype=np.float32))) if np.size(values) else 0.0


def _frame_count(file):
    frame_valid = _safe_dataset(file, "frame_valid")
    if frame_valid is not None and frame_valid.ndim == 1:
        return int(frame_valid.shape[0])
    sim_step = _safe_dataset(_safe_group(file, "time"), "sim_step")
    if sim_step is not None and sim_step.ndim == 1:
        return int(sim_step.shape[0])
    return None


def _coarse_frame_sample(frame):
    arr = np.asarray(frame, dtype=np.float32)
    if arr.ndim < 2:
        return np.nan_to_num(arr, nan=-1.0, posinf=1.0e9, neginf=-1.0e9)
    step_h = max(1, arr.shape[0] // 32)
    step_w = max(1, arr.shape[1] // 32)
    sampled = arr[::step_h, ::step_w]
    return np.nan_to_num(sampled, nan=-1.0, posinf=1.0e9, neginf=-1.0e9)


def _pairwise_mean_abs_diffs(frames):
    if len(frames) < 2:
        return []
    diffs = []
    for prev, curr in zip(frames[:-1], frames[1:]):
        diffs.append(float(np.mean(np.abs(curr - prev))))
    return diffs


def _audit_frame_level(file):
    result = _empty_result(True)
    frame_valid_ds = _safe_dataset(file, "frame_valid")
    frame_errors_ds = _safe_dataset(file, "frame_errors")
    if frame_valid_ds is None:
        _add_issue(result, "fail", "missing_frame_valid", "Missing top-level /frame_valid dataset.", path="/frame_valid")
        return result
    frame_valid = np.asarray(frame_valid_ds[()], dtype=np.bool_)
    result["checked_paths"].append("/frame_valid")
    result["stats"]["frame_count"] = int(frame_valid.shape[0])
    result["stats"]["frame_valid_ratio"] = _bool_ratio(frame_valid)
    result["stats"]["invalid_frame_count"] = int(frame_valid.shape[0] - int(np.count_nonzero(frame_valid)))
    if not frame_valid.any():
        _add_issue(result, "fail", "no_valid_frames", "No valid frames are marked in /frame_valid.", path="/frame_valid")
    elif result["stats"]["frame_valid_ratio"] < 0.8:
        _add_issue(result, "warn", "low_valid_frame_ratio", f"Only {result['stats']['frame_valid_ratio']:.3f} of frames are valid.", path="/frame_valid")
    if frame_errors_ds is None:
        _add_issue(result, "warn", "missing_frame_errors", "Missing top-level /frame_errors dataset.", path="/frame_errors")
    else:
        errors = []
        for x in frame_errors_ds[()]:
            text = str(decode_scalar(x))
            if text == "b''":
                text = ""
            errors.append(text)
        non_empty = [text for text in errors if text.strip()]
        result["checked_paths"].append("/frame_errors")
        result["stats"]["frame_error_count"] = len(non_empty)
        if non_empty:
            _add_issue(result, "warn", "frame_errors_present", f"{len(non_empty)} frame error entries are recorded.", path="/frame_errors", details={"examples": non_empty[:5]})
    return result


def _audit_rgb(file, expected_frame_count, required=False):
    result = _empty_result(required)
    cameras = _safe_group(file, "cameras")
    if cameras is None:
        if required:
            _add_issue(result, "fail", "missing_cameras_group", "Missing /cameras group.", path="/cameras")
        return result
    per_camera, count = {}, 0
    for cam_id, cam_grp in cameras.items():
        ds = _safe_dataset(cam_grp, "rgb") if isinstance(cam_grp, h5py.Group) else None
        if ds is None:
            continue
        count += 1
        path = f"/cameras/{cam_id}/rgb"
        result["checked_paths"].append(path)
        stats = {"shape": list(ds.shape), "dtype": str(ds.dtype)}
        per_camera[cam_id] = stats
        if ds.ndim != 4 or ds.shape[-1] != 3:
            _add_issue(result, "fail", "rgb_bad_shape", f"RGB dataset for {cam_id} must have shape [T,H,W,3].", path=path)
            continue
        if expected_frame_count is not None and ds.shape[0] != expected_frame_count:
            _add_issue(result, "fail", "rgb_frame_mismatch", f"RGB frame count for {cam_id} is {ds.shape[0]}, expected {expected_frame_count}.", path=path)
        stds, coarse_samples = [], []
        for idx in _sample_frame_indices(int(ds.shape[0])):
            frame = np.asarray(ds[idx])
            stds.append(float(np.std(frame)))
            coarse_samples.append(_coarse_frame_sample(frame))
        motion_diffs = _pairwise_mean_abs_diffs(coarse_samples)
        stats["sample_std"] = stds
        stats["sample_motion_mean_abs_diff"] = motion_diffs
        if stds and all(value <= 1e-6 for value in stds):
            _add_issue(result, "fail", "rgb_constant", f"Sampled RGB frames for {cam_id} are constant-valued.", path=path)
        elif stds and float(np.mean(stds)) < 5.0:
            _add_issue(result, "warn", "rgb_low_texture", f"Sampled RGB frames for {cam_id} have very low contrast.", path=path)
        if motion_diffs and max(motion_diffs) <= 0.1:
            _add_issue(result, "warn", "rgb_repeated_samples", f"Sampled RGB frames for {cam_id} are nearly identical; verify the camera is updating.", path=path)
    result["stats"]["camera_count_with_rgb"] = count
    result["stats"]["per_camera"] = per_camera
    if count == 0 and required:
        _add_issue(result, "fail", "missing_rgb", "No /cameras/*/rgb dataset was found.", path="/cameras")
    return result


def _audit_depth(file, expected_frame_count, required=False):
    result = _empty_result(required)
    cameras = _safe_group(file, "cameras")
    if cameras is None:
        if required:
            _add_issue(result, "fail", "missing_cameras_group", "Missing /cameras group.", path="/cameras")
        return result
    per_camera, count = {}, 0
    for cam_id, cam_grp in cameras.items():
        ds = _safe_dataset(cam_grp, "depth_m") if isinstance(cam_grp, h5py.Group) else None
        if ds is None:
            continue
        count += 1
        path = f"/cameras/{cam_id}/depth_m"
        result["checked_paths"].append(path)
        stats = {"shape": list(ds.shape), "dtype": str(ds.dtype)}
        per_camera[cam_id] = stats
        if ds.ndim != 3:
            _add_issue(result, "fail", "depth_bad_shape", f"Depth dataset for {cam_id} must have shape [T,H,W].", path=path)
            continue
        if expected_frame_count is not None and ds.shape[0] != expected_frame_count:
            _add_issue(result, "fail", "depth_frame_mismatch", f"Depth frame count for {cam_id} is {ds.shape[0]}, expected {expected_frame_count}.", path=path)
        finite_ratios, positive_values, coarse_samples = [], [], []
        for idx in _sample_frame_indices(int(ds.shape[0])):
            frame = np.asarray(ds[idx], dtype=np.float32)
            finite = np.isfinite(frame)
            finite_ratios.append(float(np.mean(finite)) if finite.size else 0.0)
            positive_values.append(float(np.nanmax(frame[finite])) if finite.any() else 0.0)
            coarse_samples.append(_coarse_frame_sample(frame))
        motion_diffs = _pairwise_mean_abs_diffs(coarse_samples)
        stats["sample_finite_ratio"] = finite_ratios
        stats["sample_motion_mean_abs_diff"] = motion_diffs
        if finite_ratios and max(finite_ratios) <= 0.0:
            _add_issue(result, "fail", "depth_all_nonfinite", f"All sampled depth frames for {cam_id} are non-finite.", path=path)
        elif finite_ratios and float(np.mean(finite_ratios)) < 0.5:
            _add_issue(result, "warn", "depth_low_finite_ratio", f"Sampled depth frames for {cam_id} contain many non-finite pixels.", path=path)
        if positive_values and max(positive_values) <= 0.0:
            _add_issue(result, "fail", "depth_nonpositive", f"Sampled depth frames for {cam_id} contain no positive finite depth values.", path=path)
        if motion_diffs and max(motion_diffs) <= 1.0e-4:
            _add_issue(result, "warn", "depth_repeated_samples", f"Sampled depth frames for {cam_id} are nearly identical; verify the camera is updating.", path=path)
    result["stats"]["camera_count_with_depth"] = count
    result["stats"]["per_camera"] = per_camera
    if count == 0 and required:
        _add_issue(result, "fail", "missing_depth", "No /cameras/*/depth_m dataset was found.", path="/cameras")
    return result


def _audit_joint_state(file, expected_frame_count, required=False):
    result = _empty_result(required)
    robot = _safe_group(file, "robot")
    if robot is None:
        if required:
            _add_issue(result, "fail", "missing_robot_group", "Missing /robot group.", path="/robot")
        return result
    qpos_ds, qvel_ds = _safe_dataset(robot, "qpos"), _safe_dataset(robot, "qvel")
    if qpos_ds is None or qvel_ds is None:
        if required:
            _add_issue(result, "fail", "missing_joint_state", "Missing /robot/qpos or /robot/qvel dataset.", path="/robot")
        return result
    result["checked_paths"].extend(["/robot/qpos", "/robot/qvel"])
    qpos, qvel = np.asarray(qpos_ds[()], dtype=np.float32), np.asarray(qvel_ds[()], dtype=np.float32)
    result["stats"]["qpos_shape"] = list(qpos.shape)
    result["stats"]["qvel_shape"] = list(qvel.shape)
    if qpos.ndim != 2 or qvel.ndim != 2 or qpos.shape != qvel.shape:
        _add_issue(result, "fail", "joint_state_shape_mismatch", "Joint state datasets must both have shape [T,J].", path="/robot")
        return result
    if expected_frame_count is not None and qpos.shape[0] != expected_frame_count:
        _add_issue(result, "fail", "qpos_frame_mismatch", f"/robot/qpos has {qpos.shape[0]} frames, expected {expected_frame_count}.", path="/robot/qpos")
    if qpos.size == 0:
        _add_issue(result, "fail", "empty_joint_state", "Joint state dataset is empty.", path="/robot/qpos")
    elif not (np.isfinite(qpos).all() and np.isfinite(qvel).all()):
        _add_issue(result, "fail", "nonfinite_joint_state", "Joint state contains NaN or Inf values.", path="/robot")
    return result


def _audit_object_pose(file, expected_frame_count, required=False):
    result = _empty_result(required)
    objects = _safe_group(file, "objects")
    if objects is None or not list(objects.keys()):
        if required:
            _add_issue(result, "fail", "missing_objects", "No /objects/* groups were found.", path="/objects")
        return result
    valid_count = 0
    for obj_id, obj_grp in objects.items():
        pose_ds = _safe_dataset(obj_grp, "pose_world") if isinstance(obj_grp, h5py.Group) else None
        path = f"/objects/{obj_id}/pose_world"
        if pose_ds is None:
            _add_issue(result, "fail", "missing_pose_world", f"Object {obj_id} is missing pose_world.", path=path)
            continue
        result["checked_paths"].append(path)
        pose = np.asarray(pose_ds[()], dtype=np.float32)
        if pose.ndim != 2 or pose.shape[1] != 7:
            _add_issue(result, "fail", "pose_world_bad_shape", f"Object {obj_id} pose_world must have shape [T,7].", path=path)
            continue
        if expected_frame_count is not None and pose.shape[0] != expected_frame_count:
            _add_issue(result, "fail", "pose_world_frame_mismatch", f"Object {obj_id} pose_world has {pose.shape[0]} frames, expected {expected_frame_count}.", path=path)
        finite_rows = np.isfinite(pose).all(axis=1)
        if not finite_rows.any():
            _add_issue(result, "fail", "pose_world_all_nonfinite", f"Object {obj_id} pose_world contains no fully finite rows.", path=path)
            continue
        quat_norm = np.linalg.norm(pose[finite_rows, 3:7], axis=1)
        if np.any(np.abs(quat_norm - 1.0) > 0.05):
            _add_issue(result, "warn", "pose_world_quat_not_normalized", f"Object {obj_id} pose_world quaternion norm drifts away from 1.", path=path)
        valid_count += 1
    result["stats"]["valid_object_count"] = valid_count
    if required and valid_count == 0:
        _add_issue(result, "fail", "no_valid_object_pose", "No object pose stream contains usable finite poses.", path="/objects")
    return result


def _audit_box3d(file, expected_frame_count, required=False):
    result = _empty_result(required)
    box3d = _safe_group(_safe_group(file, "labels"), "box3d")
    if box3d is None or not list(box3d.keys()):
        if required:
            _add_issue(result, "fail", "missing_box3d", "No /labels/box3d/* groups were found.", path="/labels/box3d")
        return result
    valid_count = 0
    for obj_id, obj_grp in box3d.items():
        center_ds = _safe_dataset(obj_grp, "center_world") if isinstance(obj_grp, h5py.Group) else None
        size_ds = _safe_dataset(obj_grp, "size_lwh") if isinstance(obj_grp, h5py.Group) else None
        quat_ds = _safe_dataset(obj_grp, "quat_world") if isinstance(obj_grp, h5py.Group) else None
        base = f"/labels/box3d/{obj_id}"
        if center_ds is None or size_ds is None or quat_ds is None:
            _add_issue(result, "fail", "incomplete_box3d", f"3D box for {obj_id} is missing required datasets.", path=base)
            continue
        center = np.asarray(center_ds[()], dtype=np.float32)
        size = np.asarray(size_ds[()], dtype=np.float32)
        quat = np.asarray(quat_ds[()], dtype=np.float32)
        result["checked_paths"].extend([f"{base}/center_world", f"{base}/size_lwh", f"{base}/quat_world"])
        if center.shape != size.shape or center.ndim != 2 or center.shape[1] != 3 or quat.ndim != 2 or quat.shape[1] != 4 or quat.shape[0] != center.shape[0]:
            _add_issue(result, "fail", "box3d_bad_shape", f"3D box tensors for {obj_id} have inconsistent shapes.", path=base)
            continue
        if expected_frame_count is not None and center.shape[0] != expected_frame_count:
            _add_issue(result, "fail", "box3d_frame_mismatch", f"3D box for {obj_id} has {center.shape[0]} frames, expected {expected_frame_count}.", path=base)
        valid_rows = np.isfinite(center).all(axis=1) & np.isfinite(size).all(axis=1) & np.isfinite(quat).all(axis=1) & np.all(size > 0.0, axis=1)
        if valid_rows.any():
            valid_count += 1
        else:
            _add_issue(result, "fail", "box3d_no_valid_rows", f"3D box for {obj_id} has no valid finite positive-size rows.", path=base)
    result["stats"]["valid_object_count"] = valid_count
    if required and valid_count == 0:
        _add_issue(result, "fail", "box3d_empty", "No usable 3D boxes were found.", path="/labels/box3d")
    return result


def _audit_box2d(file, expected_frame_count, required=False):
    result = _empty_result(required)
    box2d = _safe_group(_safe_group(file, "labels"), "box2d")
    if box2d is None or not list(box2d.keys()):
        if required:
            _add_issue(result, "fail", "missing_box2d", "No /labels/box2d/* groups were found.", path="/labels/box2d")
        return result
    visible_count, valid_visible_count = 0, 0
    for cam_id, cam_grp in box2d.items():
        if not isinstance(cam_grp, h5py.Group):
            continue
        for obj_id, obj_grp in cam_grp.items():
            xyxy_ds = _safe_dataset(obj_grp, "xyxy") if isinstance(obj_grp, h5py.Group) else None
            visible_ds = _safe_dataset(obj_grp, "visible") if isinstance(obj_grp, h5py.Group) else None
            base = f"/labels/box2d/{cam_id}/{obj_id}"
            if xyxy_ds is None or visible_ds is None:
                _add_issue(result, "fail", "incomplete_box2d", f"2D box for {cam_id}/{obj_id} is missing datasets.", path=base)
                continue
            xyxy = np.asarray(xyxy_ds[()], dtype=np.int32)
            visible = np.asarray(visible_ds[()], dtype=np.bool_)
            result["checked_paths"].extend([f"{base}/xyxy", f"{base}/visible"])
            if xyxy.ndim != 2 or xyxy.shape[1] != 4 or visible.ndim != 1 or xyxy.shape[0] != visible.shape[0]:
                _add_issue(result, "fail", "box2d_bad_shape", f"2D box for {cam_id}/{obj_id} has invalid shapes.", path=base)
                continue
            if expected_frame_count is not None and xyxy.shape[0] != expected_frame_count:
                _add_issue(result, "fail", "box2d_frame_mismatch", f"2D box for {cam_id}/{obj_id} has {xyxy.shape[0]} frames, expected {expected_frame_count}.", path=base)
            valid_xyxy = (xyxy[:, 2] > xyxy[:, 0]) & (xyxy[:, 3] > xyxy[:, 1]) & np.all(xyxy >= 0, axis=1)
            vis_count = int(np.count_nonzero(visible))
            ok_count = int(np.count_nonzero(visible & valid_xyxy))
            visible_count += vis_count
            valid_visible_count += ok_count
            if vis_count > 0 and ok_count == 0:
                _add_issue(result, "fail", "visible_box2d_invalid", f"2D boxes for {cam_id}/{obj_id} are marked visible but coordinates are invalid.", path=base)
    result["stats"]["visible_count"] = visible_count
    result["stats"]["valid_visible_count"] = valid_visible_count
    if required and visible_count == 0:
        _add_issue(result, "warn", "box2d_all_invisible", "All 2D boxes are invisible; verify camera placement and label projection.", path="/labels/box2d")
    return result


def _audit_occupancy_group(occupancy_grp, expected_frame_count, required=False, base_path="/labels/occupancy_tsdf"):
    result = _empty_result(required)
    if occupancy_grp is None:
        if required:
            _add_issue(result, "fail", "missing_occupancy", f"Missing {base_path} group.", path=base_path)
        return result
    state_ds = _safe_dataset(occupancy_grp, "state")
    frame_valid_ds = _safe_dataset(occupancy_grp, "frame_valid")
    grid_shape_ds = _safe_dataset(occupancy_grp, "grid_shape")
    bounds_ds = _safe_dataset(occupancy_grp, "bounds")
    voxel_size_ds = _safe_dataset(occupancy_grp, "voxel_size")
    result["checked_paths"].extend([f"{base_path}/state", f"{base_path}/frame_valid", f"{base_path}/grid_shape", f"{base_path}/bounds", f"{base_path}/voxel_size"])
    if None in (state_ds, frame_valid_ds, grid_shape_ds, bounds_ds, voxel_size_ds):
        _add_issue(result, "fail", "incomplete_occupancy", f"{base_path} is missing required datasets.", path=base_path)
        return result
    frame_valid = np.asarray(frame_valid_ds[()], dtype=np.bool_)
    grid_shape = np.asarray(grid_shape_ds[()], dtype=np.int32)
    result["stats"]["state_shape"] = list(state_ds.shape)
    result["stats"]["frame_valid_ratio"] = _bool_ratio(frame_valid)
    if state_ds.ndim != 4:
        _add_issue(result, "fail", "occupancy_bad_shape", f"{base_path}/state must have shape [T,X,Y,Z].", path=f"{base_path}/state")
        return result
    if frame_valid.ndim != 1 or state_ds.shape[0] != frame_valid.shape[0]:
        _add_issue(result, "fail", "occupancy_frame_valid_mismatch", f"{base_path}/frame_valid length does not match state frame count.", path=base_path)
    if expected_frame_count is not None and state_ds.shape[0] != expected_frame_count:
        _add_issue(result, "fail", "occupancy_frame_mismatch", f"{base_path}/state has {state_ds.shape[0]} frames, expected {expected_frame_count}.", path=f"{base_path}/state")
    if grid_shape.shape != (3,) or tuple(grid_shape.tolist()) != tuple(int(v) for v in state_ds.shape[1:4]):
        _add_issue(result, "fail", "occupancy_grid_shape_mismatch", f"{base_path}/grid_shape does not match state spatial shape.", path=base_path)
    if not frame_valid.any():
        _add_issue(result, "fail", "occupancy_no_valid_frames", f"{base_path}/frame_valid marks no valid occupancy frame.", path=f"{base_path}/frame_valid")
        return result
    valid_indices = np.flatnonzero(frame_valid)
    nonzero_frames, invalid_values = 0, set()
    for idx in valid_indices.tolist():
        frame = np.asarray(state_ds[int(idx)], dtype=np.uint8)
        if np.count_nonzero(frame) > 0:
            nonzero_frames += 1
        invalid_values.update(value for value in np.unique(frame).tolist() if int(value) not in (0, 1, 2))
    result["stats"]["valid_frame_count"] = int(valid_indices.shape[0])
    result["stats"]["nonzero_frame_count"] = nonzero_frames
    if invalid_values:
        _add_issue(result, "fail", "occupancy_invalid_state_values", f"{base_path}/state contains values outside {{0,1,2}}.", path=f"{base_path}/state", details={"values": sorted(int(v) for v in invalid_values)})
    if nonzero_frames == 0:
        _add_issue(result, "fail", "occupancy_all_empty", f"All valid occupancy frames in {base_path} are entirely empty.", path=f"{base_path}/state")
    elif nonzero_frames < int(valid_indices.shape[0]):
        _add_issue(result, "warn", "occupancy_some_empty_frames", f"Some valid occupancy frames in {base_path} contain no observed voxels.", path=f"{base_path}/state")
    return result


def _audit_tactile(file, expected_frame_count, required=False):
    result = _empty_result(required)
    tactile = _safe_group(_safe_group(file, "robot"), "tactile")
    if tactile is None:
        if required:
            _add_issue(result, "fail", "missing_tactile", "Missing /robot/tactile group.", path="/robot/tactile")
        return result

    tacmap = _safe_group(tactile, "tacmap")
    if tacmap is not None and list(tacmap.keys()):
        result["stats"]["schema"] = "tacmap"
        result["stats"]["site_count"] = len(tacmap.keys())
        usable_sites = 0
        for site_name, ds in tacmap.items():
            if not isinstance(ds, h5py.Dataset):
                continue
            path = f"/robot/tactile/tacmap/{site_name}"
            if ds.ndim != 3:
                _add_issue(result, "fail", "tacmap_bad_shape", f"TacMap site {site_name} must have shape [T,H,W].", path=path)
                continue
            if expected_frame_count is not None and ds.shape[0] != expected_frame_count:
                _add_issue(result, "fail", "tacmap_frame_mismatch", f"TacMap site {site_name} has {ds.shape[0]} frames, expected {expected_frame_count}.", path=path)
            if ds.dtype != np.dtype("uint8"):
                _add_issue(result, "warn", "tacmap_unexpected_dtype", f"TacMap site {site_name} has dtype {ds.dtype}, expected uint8.", path=path)
            usable_sites += 1
        if usable_sites == 0:
            _add_issue(result, "fail", "missing_tacmap_sites", "No usable /robot/tactile/tacmap/* datasets were found.", path="/robot/tactile/tacmap")
        return result

    camera = _safe_group(tactile, "camera")
    if camera is None or not list(camera.keys()):
        _add_issue(result, "fail", "missing_tactile_data", "Missing /robot/tactile/tacmap/* or legacy /robot/tactile/camera/* streams.", path="/robot/tactile")
        return result
    result["stats"]["schema"] = "legacy_camera_force_field"
    return result


def _audit_fisheye_metadata(file):
    result = _empty_result(False)
    cameras = _safe_group(file, "cameras")
    if cameras is None:
        return result
    fisheye_count = 0
    for cam_id, cam_grp in cameras.items():
        if not isinstance(cam_grp, h5py.Group):
            continue
        model_ds = _safe_dataset(cam_grp, "camera_model")
        model = str(decode_scalar(model_ds[()]) if model_ds is not None else "pinhole")
        # Accept current and legacy fisheye model names
        if model not in {"fisheye", "opencv_fisheye", "isaacsim_fisheye"}:
            continue
        fisheye_count += 1
        base = f"/cameras/{cam_id}"
        sem_ds = _safe_dataset(cam_grp, "raw_image_semantics")
        semantics = decode_scalar(sem_ds[()]) if sem_ds is not None else None
        if semantics != "raw_fisheye_rectangular":
            _add_issue(result, "warn", "unexpected_fisheye_semantics", f"Fisheye camera {cam_id} raw_image_semantics is {semantics!r}, expected 'raw_fisheye_rectangular'.", path=f"{base}/raw_image_semantics")

        # Calibration fields (required for geometry-compatible projection)
        matrix_ds = _safe_dataset(cam_grp, "fisheye_camera_matrix")
        dist_ds = _safe_dataset(cam_grp, "distortion_coefficients")
        mask_ds = _safe_dataset(cam_grp, "fisheye_valid_mask")
        rgb_ds = _safe_dataset(cam_grp, "rgb")
        depth_ds = _safe_dataset(cam_grp, "depth_m")
        if matrix_ds is None:
            _add_issue(result, "fail", "missing_fisheye_matrix", f"Fisheye camera {cam_id} is missing fisheye_camera_matrix.", path=f"{base}/fisheye_camera_matrix")
        if dist_ds is None:
            _add_issue(result, "fail", "missing_distortion_coefficients", f"Fisheye camera {cam_id} is missing distortion_coefficients.", path=f"{base}/distortion_coefficients")
        if mask_ds is None:
            _add_issue(result, "fail", "missing_fisheye_valid_mask", f"Fisheye camera {cam_id} is missing fisheye_valid_mask.", path=f"{base}/fisheye_valid_mask")
        else:
            mask = np.asarray(mask_ds[()], dtype=np.bool_)
            if mask.size == 0 or not mask.any():
                _add_issue(result, "fail", "empty_fisheye_valid_mask", f"Fisheye valid mask for {cam_id} has no valid pixels.", path=f"{base}/fisheye_valid_mask")
            elif bool(mask.all()):
                _add_issue(result, "warn", "full_fisheye_valid_mask", f"Fisheye valid mask for {cam_id} is entirely true; verify calibration/mask export.", path=f"{base}/fisheye_valid_mask")
            spatial = tuple(int(v) for v in (rgb_ds.shape[1:3] if rgb_ds is not None else depth_ds.shape[1:3])) if (rgb_ds is not None or depth_ds is not None) else None
            if spatial is not None and tuple(mask.shape) != spatial:
                _add_issue(result, "fail", "fisheye_mask_shape_mismatch", f"Fisheye valid mask for {cam_id} has shape {tuple(mask.shape)}, expected {spatial}.", path=f"{base}/fisheye_valid_mask")

        # Clipping range (accept both new and old field names)
        clipping_ds = _safe_dataset(cam_grp, "clipping_range") or _safe_dataset(cam_grp, "render_clipping_range")
        if clipping_ds is not None and tuple(np.asarray(clipping_ds[()]).shape) != (2,):
            _add_issue(result, "fail", "bad_clipping_range", f"Fisheye camera {cam_id} clipping_range must have shape [2].", path=f"{base}/clipping_range")
    result["stats"]["fisheye_camera_count"] = fisheye_count
    return result


def audit_raw_file(path, required_modalities=None):
    with h5py.File(path, "r") as f:
        meta = _read_meta_group(f)
        expected = _frame_count(f)
        required = _normalize_modalities(required_modalities, RAW_MODALITIES) if required_modalities is not None else infer_required_raw_modalities(meta)
        modalities = {
            "rgb": _audit_rgb(f, expected, required="rgb" in required),
            "depth": _audit_depth(f, expected, required="depth" in required),
            "joint_state": _audit_joint_state(f, expected, required="joint_state" in required),
            "object_pose": _audit_object_pose(f, expected, required="object_pose" in required),
            "box3d": _audit_box3d(f, expected, required="box3d" in required),
            "box2d": _audit_box2d(f, expected, required="box2d" in required),
            "occupancy_tsdf": _audit_occupancy_group(_safe_group(_safe_group(f, "labels"), "occupancy_tsdf"), expected, required="occupancy_tsdf" in required),
            "tactile": _audit_tactile(f, expected, required="tactile" in required),
        }
        frame_level = _audit_frame_level(f)
        fisheye = _audit_fisheye_metadata(f)
    statuses = [frame_level["status"], fisheye["status"]] + [payload["status"] for payload in modalities.values() if payload["required"]]
    return {"file": str(path), "type": "raw", "meta": meta, "expected_frame_count": expected, "required_modalities": required, "frame_level": frame_level, "modalities": modalities, "fisheye": fisheye, "status": max(statuses, key=lambda item: STATUS_ORDER[item]) if statuses else "pass"}


def audit_derived_file(path, required_modalities=None):
    with h5py.File(path, "r") as f:
        meta = _read_meta_group(f)
        expected = _frame_count(f)
        required = _normalize_modalities(required_modalities, DERIVED_MODALITIES) if required_modalities is not None else infer_required_derived_modalities(meta)
        frame_level = _audit_frame_level(f)
        occupancy = _audit_occupancy_group(_safe_group(_safe_group(f, "labels"), "occupancy_tsdf"), expected, required="occupancy_tsdf" in required)
    statuses = [frame_level["status"]] + [occupancy["status"] if occupancy["required"] else "pass"]
    return {"file": str(path), "type": "derived", "meta": meta, "expected_frame_count": expected, "required_modalities": required, "frame_level": frame_level, "modalities": {"occupancy_tsdf": occupancy}, "status": max(statuses, key=lambda item: STATUS_ORDER[item]) if statuses else "pass"}


def build_cross_file_audit(raw_audit, derived_audit):
    result = _empty_result(derived_audit is not None)
    if derived_audit is None:
        result["stats"]["paired"] = False
        return result
    raw_count, derived_count = raw_audit.get("expected_frame_count"), derived_audit.get("expected_frame_count")
    result["stats"]["paired"] = True
    result["stats"]["raw_frame_count"] = raw_count
    result["stats"]["derived_frame_count"] = derived_count
    if raw_count is not None and derived_count is not None and raw_count != derived_count:
        _add_issue(result, "fail", "frame_count_mismatch", f"Raw frame count ({raw_count}) does not match derived frame count ({derived_count}).")
    if "occupancy_tsdf" in derived_audit.get("required_modalities", []) and raw_audit.get("modalities", {}).get("depth", {}).get("status") == "fail":
        _add_issue(result, "fail", "derived_occupancy_without_valid_depth", "Derived occupancy is required, but the paired raw depth stream is not usable.")
    raw_occ = raw_audit.get("modalities", {}).get("occupancy_tsdf", {})
    derived_occ = derived_audit.get("modalities", {}).get("occupancy_tsdf", {})
    if raw_occ.get("status") == "fail" and derived_occ.get("status") == "pass":
        _add_issue(result, "warn", "occupancy_only_offline_valid", "Online occupancy in the raw file is unusable, but offline/derived occupancy passed.")
    return result


def build_pair_report(raw_audit, derived_audit):
    cross = build_cross_file_audit(raw_audit, derived_audit)
    failures, warnings = [], []

    def collect(section, prefix):
        for issue in section.get("issues", []):
            text = f"{prefix}: {issue['message']}"
            if issue["severity"] == "fail":
                failures.append(text)
            elif issue["severity"] == "warn":
                warnings.append(text)

    collect(raw_audit["frame_level"], "raw.frame_level")
    collect(raw_audit["fisheye"], "raw.fisheye")
    for name, section in raw_audit["modalities"].items():
        if section.get("required") or section.get("status") != "pass":
            collect(section, f"raw.{name}")
    if derived_audit is not None:
        collect(derived_audit["frame_level"], "derived.frame_level")
        for name, section in derived_audit["modalities"].items():
            if section.get("required") or section.get("status") != "pass":
                collect(section, f"derived.{name}")
    collect(cross, "cross_file")
    statuses = [raw_audit["status"], cross["status"]] + ([derived_audit["status"]] if derived_audit is not None else [])
    overall = max(statuses, key=lambda item: STATUS_ORDER[item]) if statuses else "pass"
    return {
        "export": {
            "type": "data_quality_audit",
            "contains_full_dataset_values": False,
            "note": "This audit is conservative: pass means the checked schema and sampled statistics look usable; warn means manual review is still needed; fail means the file is unfit for the required modality.",
        },
        "raw": raw_audit,
        "derived": derived_audit,
        "cross_file": cross,
        "overall": {
            "status": overall,
            "usable_for_downstream": overall != "fail",
            "recommended_for_training": overall == "pass",
            "failure_reasons": failures,
            "warning_reasons": warnings,
        },
    }


def build_diff(raw_summary, derived_summary):
    raw_ds, derived_ds = raw_summary["dataset_schema"], derived_summary["dataset_schema"]
    raw_only = {k: raw_ds[k] for k in sorted(set(raw_ds) - set(derived_ds))}
    derived_only = {k: derived_ds[k] for k in sorted(set(derived_ds) - set(raw_ds))}
    shared = {k: {"raw": raw_ds[k], "derived": derived_ds[k]} for k in sorted(set(raw_ds) & set(derived_ds)) if raw_ds[k] != derived_ds[k]}
    return {
        "raw_file": raw_summary["file"],
        "derived_file": derived_summary["file"],
        "export": {"type": "schema_diff", "contains_full_dataset_values": False, "note": "This JSON compares dataset presence and schema between two HDF5 files."},
        "raw_size_bytes": raw_summary["size_bytes"],
        "derived_size_bytes": derived_summary["size_bytes"],
        "raw_coverage": raw_summary["coverage"],
        "derived_coverage": derived_summary["coverage"],
        "top_level_groups": {
            "raw": raw_summary["top_level_groups"],
            "derived": derived_summary["top_level_groups"],
            "raw_only": sorted(set(raw_summary["top_level_groups"]) - set(derived_summary["top_level_groups"])),
            "derived_only": sorted(set(derived_summary["top_level_groups"]) - set(raw_summary["top_level_groups"])),
        },
        "raw_only_datasets": raw_only,
        "derived_only_datasets": derived_only,
        "shared_dataset_schema_differences": shared,
    }


def main():
    args = parse_args()
    raw_path = args.raw_path.resolve()
    derived_path = args.derived_path.resolve() if args.derived_path else None
    out_dir = args.out_dir.resolve() if args.out_dir else raw_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_summary = summarize_file(raw_path)
    raw_summary_path = out_dir / f"{raw_path.stem}.summary.json"
    raw_summary_path.write_text(json.dumps(raw_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    raw_required = _normalize_modalities(args.require_raw_modalities, RAW_MODALITIES) if args.require_raw_modalities is not None else None
    raw_audit = audit_raw_file(raw_path, raw_required)

    derived_summary_path = None
    diff_path = None
    derived_audit = None
    if derived_path is not None:
        derived_summary = summarize_file(derived_path)
        derived_summary_path = out_dir / f"{derived_path.stem}.summary.json"
        derived_summary_path.write_text(json.dumps(derived_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        diff_path = out_dir / f"{raw_path.stem}_vs_{derived_path.stem}.diff.json"
        diff_path.write_text(json.dumps(build_diff(raw_summary, derived_summary), ensure_ascii=False, indent=2), encoding="utf-8")
        derived_required = _normalize_modalities(args.require_derived_modalities, DERIVED_MODALITIES) if args.require_derived_modalities is not None else None
        derived_audit = audit_derived_file(derived_path, derived_required)

    audit_report = build_pair_report(raw_audit, derived_audit)
    audit_path = out_dir / f"{raw_path.stem}.audit.json"
    audit_path.write_text(json.dumps(audit_report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "raw_summary_json": str(raw_summary_path),
        "derived_summary_json": str(derived_summary_path) if derived_summary_path is not None else None,
        "diff_json": str(diff_path) if diff_path is not None else None,
        "audit_json": str(audit_path),
        "raw_status": raw_audit["status"],
        "derived_status": derived_audit["status"] if derived_audit is not None else None,
        "overall_status": audit_report["overall"]["status"],
        "usable_for_downstream": audit_report["overall"]["usable_for_downstream"],
        "recommended_for_training": audit_report["overall"]["recommended_for_training"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
