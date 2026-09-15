"""Generate ground-truth occupancy labels from collision mesh geometry.

Reads object poses and articulation states from HDF5 episode files,
extracts collision/visual meshes from USD assets, and voxelizes the
scene to produce zero-noise occupancy ground truth.

No Isaac Sim dependency — runs purely offline using pxr + trimesh.

Usage:
    python tools/labels/generate_occupancy_gt.py <episode.hdf5>
    python tools/labels/generate_occupancy_gt.py <episode.hdf5> --voxel-size 0.01
    python tools/labels/generate_occupancy_gt.py <directory> --write-mode inplace
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import warnings
from copy import deepcopy
from typing import Any

import h5py
import numpy as np
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from collector.config import DEFAULT_OCCUPANCY_GT_CONFIG, normalize_occupancy_gt_config
from tools.labels._label_common import (
    _read_meta_dict,
    _scene_table_height_offset,
    compute_default_occupancy_bounds,
    decode_hdf5_string,
    default_sidecar_path,
    find_hdf5_files,
    prepare_label_slot,
    read_frame_metadata,
    read_local_bboxes,
    read_object_states,
    resolve_scene_path as _resolve_scene_path,
    sync_sidecar,
)
from tools.labels._accel import read_episode_batch, get_device, parallel_process
from tools.labels._mesh_voxelizer import (
    ArticulationFK,
    LinkMesh,
    _voxel_centers,
    build_semantic_map,
    clear_voxelizer_caches,
    compute_table_aabb,
    extract_meshes_from_usd,
    parse_scene_table_spec,
    voxelize_scene_frame,
)


# ---------------------------------------------------------------------------
# HDF5 metadata reading (helpers shared via tools.labels._label_common)
# ---------------------------------------------------------------------------


def _resolve_asset_path(asset_path: str, episode_path: str) -> str:
    """Resolve an asset path to an absolute file system path.

    Asset paths may be relative to the repo root or absolute.
    For USD files, we look for the .usd file in the asset directory.
    """
    if os.path.isabs(asset_path) and os.path.isfile(asset_path):
        return asset_path

    # Try relative to repo root
    repo_candidate = os.path.join(REPO_ROOT, asset_path)
    if os.path.isfile(repo_candidate):
        return repo_candidate

    # Try relative to episode directory
    ep_dir = os.path.dirname(os.path.abspath(episode_path))
    ep_candidate = os.path.join(ep_dir, asset_path)
    if os.path.isfile(ep_candidate):
        return ep_candidate

    # Cross-platform fallback: normalize Windows backslashes and search
    # for dataset-relative suffix (e.g. "dex2bench_dataset/Objects/...")
    # in sibling directories of the repo root.
    normalized = asset_path.replace("\\", "/")
    for marker in ("dex2bench_dataset/", "Dex2Assets/"):
        idx = normalized.find(marker)
        if idx >= 0:
            rel_tail = normalized[idx:]
            parent_dir = os.path.dirname(REPO_ROOT)
            candidate = os.path.join(parent_dir, rel_tail)
            if os.path.isfile(candidate):
                # Prefer usd_sdf_linux variant if it exists
                sdf_candidate = _try_sdf_linux_variant(candidate)
                return sdf_candidate if sdf_candidate else candidate

    return asset_path  # return as-is; will warn later if not found


def _try_sdf_linux_variant(usd_path: str) -> str | None:
    """If a ``usd_sdf_linux/`` sibling directory contains the same filename,
    return that path instead.  Handles common USD directory layouts:
      Objects/XXX/usd/foo.usd        -> Objects/XXX/usd_sdf_linux/foo.usd
      Objects/XXX/foo.usd            -> Objects/XXX/usd_sdf_linux/foo.usd
      Objects/XXX/.../subdir/foo.usd -> Objects/XXX/.../usd_sdf_linux/foo.usd
    """
    dirname = os.path.dirname(usd_path)
    basename = os.path.basename(usd_path)
    # Try replacing the immediate parent directory with usd_sdf_linux
    parent = os.path.dirname(dirname)
    candidate = os.path.join(parent, "usd_sdf_linux", basename)
    if os.path.isfile(candidate):
        return candidate
    # Try usd_sdf_linux as a sibling of current directory
    candidate = os.path.join(dirname, "usd_sdf_linux", basename)
    if os.path.isfile(candidate):
        return candidate
    return None


def _get_object_scale(file: h5py.File, obj_id: str) -> float:
    """Get the scale for an object from metadata."""
    # Try object_asset_keys which may encode scale info
    # For now, default to 1.0 — scale is baked into USD meshes in most cases
    return 1.0


def _read_asset_scales_from_scene(
    scene_path: str | None,
    object_asset_keys: dict[str, str],
) -> dict[str, float | tuple[float, float, float]]:
    """Read per-object scale from scene YAML assets section.

    Returns {obj_id: scale}; scale may be scalar or xyz tuple.
    """
    import yaml

    scales: dict[str, float | tuple[float, float, float]] = {}
    if scene_path is None or not os.path.isfile(scene_path):
        return scales

    with open(scene_path, "r") as fh:
        scene_cfg = yaml.safe_load(fh) or {}

    assets_cfg = scene_cfg.get("assets", {})
    for obj_id, asset_key in object_asset_keys.items():
        asset_def = assets_cfg.get(asset_key)
        if asset_def is None:
            continue
        raw_scale = asset_def.get("scale")
        if raw_scale is None:
            continue
        if isinstance(raw_scale, (int, float)):
            scales[obj_id] = float(raw_scale)
        elif isinstance(raw_scale, (list, tuple)):
            vals = [float(v) for v in raw_scale]
            if len(vals) == 3:
                scales[obj_id] = vals[0] if abs(vals[0] - vals[1]) < 1e-6 and abs(vals[1] - vals[2]) < 1e-6 else tuple(vals)
            else:
                warnings.warn(f"Invalid scale {raw_scale} for {obj_id} ({asset_key}); using 1.0")
    return scales


# ---------------------------------------------------------------------------
# Frame processing: GPU with CPU fallback
# ---------------------------------------------------------------------------

def _process_frames_gpu_or_cpu(
    hdf5_path: str,
    f: h5py.File,
    batch_data,
    frame_valid: np.ndarray,
    frame_count: int,
    asset_meshes: dict,
    fk_solvers: dict,
    table_aabb,
    bounds: np.ndarray,
    voxel_size: float,
    centers_np: np.ndarray,
    grid_shape_tuple: tuple[int, int, int],
    semantic_map,
    local_bboxes,
    device_pref: str,
) -> tuple[list, list, tuple | None]:
    """Try GPU path, fall back to CPU on failure."""
    # Try GPU
    try:
        dev = get_device(device_pref)
        if dev.type != "cpu":
            return _process_frames_gpu(
                hdf5_path, batch_data, frame_valid, frame_count,
                asset_meshes, fk_solvers, table_aabb, bounds,
                centers_np, grid_shape_tuple, voxel_size, semantic_map, local_bboxes, dev,
            )
    except Exception as exc:
        warnings.warn(f"{hdf5_path}: GPU GT voxelization failed ({exc}), falling back to CPU")

    # CPU fallback
    return _process_frames_cpu(
        hdf5_path, f, frame_valid, frame_count,
        asset_meshes, fk_solvers, table_aabb,
        bounds, voxel_size, semantic_map, local_bboxes,
    )


def _process_frames_gpu(
    hdf5_path, batch_data, frame_valid, frame_count,
    asset_meshes, fk_solvers, table_aabb, bounds,
    centers_np, grid_shape_tuple, voxel_size, semantic_map, local_bboxes, device,
):
    from tools.labels._gpu_voxelizer import (
        BatchFK, MeshContainmentGPU, MeshSurfaceProximityGPU,
        _uses_surface_proximity, voxelize_scene_frame_gpu,
    )
    from tools.labels._accel import to_torch

    # Pre-compute occupancy tests (reused across all frames).
    # Cache per mesh, not per link: one link can own multiple collision meshes.
    containment_cache = {}
    for obj_id, meshes in asset_meshes.items():
        for lm in meshes:
            key = id(lm.mesh)
            try:
                if _uses_surface_proximity(lm.mesh):
                    containment_cache[key] = MeshSurfaceProximityGPU.from_trimesh(
                        lm.mesh, voxel_size, device,
                    )
                else:
                    containment_cache[key] = MeshContainmentGPU.from_trimesh(lm.mesh, device)
            except Exception:
                pass  # will fall back to trimesh in _mark_mesh_gpu

    # Pre-compute BatchFK cache
    batch_fk_cache = {}
    for obj_id, fk in fk_solvers.items():
        try:
            batch_fk_cache[obj_id] = BatchFK(fk)
        except Exception:
            pass

    # Pre-compute per-mesh local AABB (frame-invariant) for cheap 8-corner
    # world AABB derivation inside _mark_mesh_gpu — avoids transforming every
    # vertex every frame.
    aabb_cache: dict[int, tuple[np.ndarray, np.ndarray] | None] = {}
    for obj_id, meshes in asset_meshes.items():
        for lm in meshes:
            key = id(lm.mesh)
            if key in aabb_cache:
                continue
            verts = np.asarray(lm.mesh.vertices, dtype=np.float32)
            if verts.shape[0] == 0:
                aabb_cache[key] = None
            else:
                aabb_cache[key] = (verts.min(axis=0), verts.max(axis=0))

    # Upload voxel centers to device once; keep as a 3D grid so each mesh can
    # slice its candidate sub-box as a view (no copy, no host sync).
    centers_gpu = to_torch(centers_np, device)  # (N, 3) float32
    nx, ny, nz = grid_shape_tuple
    centers_grid = centers_gpu.view(nx, ny, nz, 3)

    # Table occupies a static AABB — compute its voxel mask once per episode
    # instead of re-scanning the full voxel grid every frame.
    table_mask_3d = None
    if table_aabb is not None:
        lo = to_torch(np.asarray(table_aabb[0], dtype=np.float32), device)
        hi = to_torch(np.asarray(table_aabb[1], dtype=np.float32), device)
        table_mask = ((centers_gpu >= lo) & (centers_gpu <= hi)).all(dim=1)
        table_mask_3d = table_mask.view(nx, ny, nz)

    all_states = []
    all_semantics = []
    grid_shape = grid_shape_tuple

    for fi in tqdm(range(frame_count), desc=f"GPU {os.path.basename(hdf5_path)}", unit="frame"):
        if not bool(frame_valid[fi]):
            all_states.append(None)
            all_semantics.append(None)
            continue

        # Build object states from batch data
        obj_states = {}
        for obj_id in batch_data.object_ids:
            pose = batch_data.poses[obj_id][fi]
            if not np.all(np.isfinite(pose)):
                continue
            state = {"pose_world": pose}
            if obj_id in batch_data.qpos:
                state["qpos"] = batch_data.qpos[obj_id][fi]
            if obj_id in batch_data.joint_names:
                state["joint_names"] = batch_data.joint_names[obj_id]
            obj_states[obj_id] = state

        # NaN check
        nan_objects = set(asset_meshes.keys()) - set(obj_states.keys())
        if nan_objects:
            warnings.warn(
                f"frame {fi}: objects with NaN pose: {sorted(nan_objects)}"
            )
            all_states.append(None)
            all_semantics.append(None)
            continue

        try:
            result = voxelize_scene_frame_gpu(
                object_states=obj_states,
                asset_meshes=asset_meshes,
                batch_fk_cache=batch_fk_cache,
                table_mask_3d=table_mask_3d,
                centers_grid=centers_grid,
                grid_shape=grid_shape,
                containment_cache=containment_cache,
                aabb_cache=aabb_cache,
                bounds=bounds,
                device=device,
                semantic_map=semantic_map,
                local_bboxes=local_bboxes,
                voxel_size=voxel_size,
            )
            all_states.append(result["state"])
            all_semantics.append(result.get("semantic_id"))
        except Exception as exc:
            warnings.warn(f"frame {fi} GPU voxelization failed: {exc}")
            all_states.append(None)
            all_semantics.append(None)

    return all_states, all_semantics, grid_shape


def _process_frames_cpu(
    hdf5_path, f, frame_valid, frame_count,
    asset_meshes, fk_solvers, table_aabb,
    bounds, voxel_size, semantic_map, local_bboxes,
):
    """Original per-frame CPU path."""
    all_states = []
    all_semantics = []
    grid_shape = None

    for fi in tqdm(range(frame_count), desc=f"CPU {os.path.basename(hdf5_path)}", unit="frame"):
        if not bool(frame_valid[fi]):
            all_states.append(None)
            all_semantics.append(None)
            continue

        obj_states = read_object_states(f, fi)
        nan_objects = set(asset_meshes.keys()) - set(obj_states.keys())
        if nan_objects:
            warnings.warn(
                f"{hdf5_path}: frame {fi}: objects with NaN pose: {sorted(nan_objects)}; "
                f"frame marked invalid"
            )
            all_states.append(None)
            all_semantics.append(None)
            continue

        try:
            result = voxelize_scene_frame(
                object_states=obj_states,
                asset_meshes=asset_meshes,
                fk_solvers=fk_solvers,
                table_aabb=table_aabb,
                bounds=bounds,
                voxel_size=voxel_size,
                semantic_map=semantic_map,
                local_bboxes=local_bboxes,
            )
            if grid_shape is None:
                grid_shape = tuple(result["grid_shape"].tolist())
            all_states.append(result["state"])
            all_semantics.append(result.get("semantic_id"))
        except Exception as exc:
            warnings.warn(f"{hdf5_path}: frame {fi} voxelization failed: {exc}")
            all_states.append(None)
            all_semantics.append(None)

    return all_states, all_semantics, grid_shape


# ---------------------------------------------------------------------------
# Per-episode GT occupancy generation
# ---------------------------------------------------------------------------

def generate_occupancy_gt_for_file(
    hdf5_path: str,
    *,
    occupancy_config: dict[str, Any],
    write_mode: str = "sidecar",
    overwrite: bool = False,
    output_path: str | None = None,
    scene_override: str | None = None,
    no_table: bool = False,
    no_semantic: bool = False,
    device: str = "auto",
    bounds_override: list[list[float]] | None = None,
) -> bool:
    """Generate GT occupancy labels for a single episode.

    Returns True if labels were written, False if skipped.

    When ``bounds_override`` is None (no explicit ``--bounds``), the occupancy
    bounds are derived per-episode from the actual object-table geometry
    (covering only the object table, floor at the real tabletop height).
    """
    occupancy_config = deepcopy(occupancy_config)

    # Clear module-level voxelizer caches to prevent stale data from a previous
    # episode's meshes leaking into this episode.
    clear_voxelizer_caches()

    with h5py.File(hdf5_path, "r") as f:
        frame_valid, frame_errors, sim_steps, camera_ids, object_ids = read_frame_metadata(f)

        # Read metadata needed for mesh resolution
        object_body_types_raw = _read_meta_dict(f, "object_body_types") or {}
        object_asset_paths_raw = _read_meta_dict(f, "object_asset_paths") or {}

        # Resolve scene path
        scene_path = _resolve_scene_path(hdf5_path, f, scene_override)
        if scene_path is None and not no_table:
            raise ValueError(
                f"{hdf5_path}: scene_file not found in metadata; "
                f"pass --scene <path> or --no-table to proceed without table."
            )

        # Table AABB — only the tabletop slab (per-episode actual height)
        table_aabb = None
        if scene_path and not no_table:
            try:
                table_size, nominal_height = parse_scene_table_spec(scene_path)
                thickness = float(table_size[2])
                actual_z = nominal_height + _scene_table_height_offset(f)
                table_aabb = compute_table_aabb(table_size, actual_z, thickness)
            except Exception as exc:
                warnings.warn(f"{hdf5_path}: failed to parse table spec: {exc}")

        # Build semantic map
        semantic_map = None
        semantic_legend = None
        if not no_semantic:
            semantic_map, id_to_label = build_semantic_map(object_ids, object_body_types_raw)
            semantic_legend = id_to_label

        # Read per-object scales from scene YAML
        object_asset_keys_raw = _read_meta_dict(f, "object_asset_keys") or {}
        scene_scales = _read_asset_scales_from_scene(scene_path, object_asset_keys_raw)

        # Load meshes and FK solvers per object
        asset_meshes: dict[str, list[LinkMesh]] = {}
        fk_solvers: dict[str, ArticulationFK] = {}
        geometry_source = occupancy_config.get("geometry_source", "collision")

        for obj_id in object_ids:
            asset_path = object_asset_paths_raw.get(obj_id, "")
            if not asset_path:
                warnings.warn(f"{hdf5_path}: no asset path for {obj_id}, skipping")
                continue

            resolved = _resolve_asset_path(asset_path, hdf5_path)
            if not os.path.isfile(resolved):
                warnings.warn(f"{hdf5_path}: asset not found for {obj_id}: {resolved}")
                continue

            scale = scene_scales.get(obj_id, _get_object_scale(f, obj_id))

            # Determine body type early — needed for mesh extraction mode
            body_type = object_body_types_raw.get(obj_id, "")
            is_articulated = body_type == "articulation"

            # Extract meshes
            try:
                meshes = extract_meshes_from_usd(
                    resolved, scale=scale, source=geometry_source,
                    articulated=is_articulated,
                )
                if meshes:
                    asset_meshes[obj_id] = meshes
                else:
                    warnings.warn(f"{hdf5_path}: no meshes extracted for {obj_id} from {resolved}")
            except Exception as exc:
                warnings.warn(f"{hdf5_path}: mesh extraction failed for {obj_id}: {exc}")

            # Build FK solver for articulated bodies
            if is_articulated:
                try:
                    fk_solvers[obj_id] = ArticulationFK(resolved, scale=scale)
                except Exception as exc:
                    warnings.warn(f"{hdf5_path}: FK solver failed for {obj_id}: {exc}")

        frame_count = int(frame_valid.shape[0])

        # Derive bounds from the actual object-table geometry unless explicitly
        # overridden via --bounds. Floor pins to the real tabletop height; XY
        # covers only the object table (excludes the rear robot-support table).
        if bounds_override is not None:
            occupancy_config["bounds"] = [list(b) for b in bounds_override]
            print(f"[INFO] occupancy bounds: override {occupancy_config['bounds']}")
        else:
            auto_bounds = compute_default_occupancy_bounds(f, scene_path)
            if auto_bounds is not None:
                occupancy_config["bounds"] = auto_bounds
                print(f"[INFO] occupancy bounds: auto {auto_bounds} (table z={auto_bounds[0][2]:.4f})")
            else:
                print(f"[INFO] occupancy bounds: fallback {occupancy_config.get('bounds')} "
                      f"(scene YAML unavailable for {hdf5_path})")

        bounds = np.asarray(occupancy_config["bounds"], dtype=np.float32)
        voxel_size = float(occupancy_config["voxel_size"])
        local_bboxes = read_local_bboxes(f) or {}

        # Pre-compute voxel centers ONCE (reused across all frames)
        centers_np, grid_shape_tuple = _voxel_centers(bounds, voxel_size)

        # Read all object data in bulk
        batch_data = read_episode_batch(f, need_depth=False)

        # Try GPU path
        all_states, all_semantics, grid_shape = _process_frames_gpu_or_cpu(
            hdf5_path, f, batch_data, frame_valid, frame_count,
            asset_meshes, fk_solvers, table_aabb, bounds, voxel_size,
            centers_np, grid_shape_tuple, semantic_map, local_bboxes, device,
        )

    if grid_shape is None:
        warnings.warn(f"{hdf5_path}: no valid frames produced, skipping")
        return False

    # Write to HDF5
    if write_mode == "inplace":
        with h5py.File(hdf5_path, "r+") as target:
            labels_grp = target.require_group("labels")
            prepare_label_slot(labels_grp, "occupancy_gt", target_path=hdf5_path, overwrite=overwrite)
            _write_gt_labels(labels_grp, grid_shape, bounds, voxel_size,
                             all_states, all_semantics, semantic_legend, frame_valid)
    else:
        sidecar = output_path or default_sidecar_path(hdf5_path)
        sync_sidecar(
            sidecar,
            source_path=hdf5_path,
            meta={
                "artifact_type": "derived_labels",
                "occupancy_gt_config": occupancy_config,
                "occupancy_gt_semantics": "gt_mesh_voxelization",
            },
            frame_valid=frame_valid,
            frame_errors=frame_errors,
            sim_steps=sim_steps,
        )
        with h5py.File(sidecar, "r+") as target:
            labels_grp = target.require_group("labels")
            prepare_label_slot(labels_grp, "occupancy_gt", target_path=sidecar, overwrite=overwrite)
            _write_gt_labels(labels_grp, grid_shape, bounds, voxel_size,
                             all_states, all_semantics, semantic_legend, frame_valid)

    return True


def _write_gt_labels(
    labels_grp: h5py.Group,
    grid_shape: tuple[int, int, int],
    bounds: np.ndarray,
    voxel_size: float,
    all_states: list[np.ndarray | None],
    all_semantics: list[np.ndarray | None],
    semantic_legend: dict[int, str] | None,
    frame_valid: np.ndarray,
) -> None:
    """Write GT occupancy labels into an HDF5 labels group.

    Uses chunked datasets to avoid materializing the full grid in memory.
    """
    gt_grp = labels_grp.create_group("occupancy_gt")
    gt_grp.create_dataset("grid_shape", data=np.asarray(grid_shape, dtype=np.int32))
    gt_grp.create_dataset("bounds", data=np.asarray(bounds, dtype=np.float32))
    gt_grp.create_dataset("voxel_size", data=np.asarray(voxel_size, dtype=np.float32))

    str_dtype = h5py.string_dtype(encoding="utf-8")
    gt_grp.create_dataset("method", data=np.asarray("gt_mesh_voxelization", dtype=str_dtype))
    gt_grp.create_dataset("semantics", data=np.asarray("gt_complete_scene", dtype=str_dtype))

    n_frames = len(all_states)
    chunk_shape = (1, *grid_shape)
    has_semantics = any(s is not None for s in all_semantics)

    # Create chunked, extendable datasets.
    # gzip opts=9: strongest lossless level (restored bytes are identical to
    # opts=4, only the compressed size shrinks ~5-10%).
    state_ds = gt_grp.create_dataset(
        "state",
        shape=(n_frames, *grid_shape),
        dtype=np.uint8,
        chunks=chunk_shape,
        compression="gzip",
        compression_opts=9,
    )
    gt_frame_valid = np.zeros((n_frames,), dtype=np.bool_)

    if has_semantics:
        # semantic_id values are small (0=free, 1=table, 2+=objects); the
        # legend never exceeds a handful of ids, so uint8 is lossless and halves
        # the raw storage vs uint16.
        sem_ds = gt_grp.create_dataset(
            "semantic_id",
            shape=(n_frames, *grid_shape),
            dtype=np.uint8,
            chunks=chunk_shape,
            compression="gzip",
            compression_opts=9,
        )

    # Write frame-by-frame
    for i, st in enumerate(all_states):
        if st is not None:
            state_ds[i] = st
            gt_frame_valid[i] = True
        if has_semantics and all_semantics[i] is not None:
            # Cast to uint8 on write (lossless: ids <= 255). Guards against a
            # pathological episode with >254 objects corrupting the cast.
            sem_arr = np.asarray(all_semantics[i])
            if sem_arr.dtype != np.uint8:
                if sem_arr.max(initial=0) > 255:
                    raise ValueError(
                        f"semantic_id {sem_arr.max()} exceeds uint8 range; "
                        f"revert semantic_id dataset to uint16."
                    )
                sem_arr = sem_arr.astype(np.uint8)
            sem_ds[i] = sem_arr

    gt_grp.create_dataset("frame_valid", data=gt_frame_valid)

    # Semantic legend (JSON string)
    if semantic_legend is not None:
        str_dtype = h5py.string_dtype(encoding="utf-8")
        gt_grp.create_dataset(
            "semantic_legend",
            data=np.asarray(json.dumps(semantic_legend), dtype=str_dtype),
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate ground-truth occupancy labels from collision mesh geometry."
    )
    parser.add_argument("input", help="HDF5 file or directory to scan for *.hdf5")
    parser.add_argument(
        "--write-mode",
        choices=["inplace", "sidecar"],
        default="sidecar",
        help="Write into source episode or derived sidecar (default: sidecar)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing occupancy_gt labels")
    parser.add_argument("--voxel-size", type=float, default=0.01, help="Voxel size in meters (default: 0.01)")
    parser.add_argument(
        "--bounds", type=str, default=None,
        help="Override bounds as 'x0,y0,z0,x1,y1,z1' (e.g. '-0.5,-0.4,0.70,0.5,0.4,1.05')",
    )
    parser.add_argument(
        "--scene", type=str, default=None,
        help="Override scene YAML path (default: read from HDF5 metadata)",
    )
    parser.add_argument(
        "--geometry-source", choices=["collision", "visual"],
        default="collision",
        help="Mesh source for voxelization (default: collision)",
    )
    parser.add_argument(
        "--no-table", action="store_true",
        help="Do not include table in occupancy",
    )
    parser.add_argument(
        "--no-semantic", action="store_true",
        help="Do not output semantic_id grid",
    )
    parser.add_argument("--workers", type=int, default=0,
                        help="Parallel workers (0=auto, 1=sequential; GPU+workers supported: "
                             "each worker is a separate process sharing one GPU)")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "mps", "cpu"],
                        help="Compute device (default: auto)")
    args = parser.parse_args()

    config: dict[str, Any] = {
        "voxel_size": float(args.voxel_size),
        "geometry_source": args.geometry_source,
        "label_version": "occupancy_gt_v1",
    }
    if args.bounds is not None:
        vals = [float(v) for v in args.bounds.split(",")]
        if len(vals) != 6:
            print("[ERROR] --bounds requires exactly 6 comma-separated values: x0,y0,z0,x1,y1,z1")
            return 2
        config["bounds"] = [[vals[0], vals[1], vals[2]], [vals[3], vals[4], vals[5]]]

    occupancy_config = normalize_occupancy_gt_config(config)

    device_pref = args.device

    # Resolve worker count. GPU + multi-worker is now supported: each worker
    # is a separate process with its own CUDA context sharing one GPU. The
    # single-process GPU path only reaches ~53% util / ~1.6 GB VRAM, so a few
    # concurrent episodes fill the idle compute without exhausting 48 GB. We
    # deliberately do NOT probe CUDA here — initializing a driver context in
    # the parent would break the fork()-ed worker processes.
    if args.workers == 0:
        workers = 4 if device_pref != "cpu" else 0   # 0 => parallel_process auto (cpu)
    else:
        workers = args.workers

    print(f"[INFO] geometry_source={occupancy_config['geometry_source']}, "
          f"voxel_size={occupancy_config['voxel_size']}, "
          f"device={device_pref}, workers={workers if workers else 'auto'}, "
          f"bounds={occupancy_config['bounds']}")
    if workers > 1 and device_pref != "cpu":
        print(f"[INFO] GPU multi-worker: {workers} processes share one GPU "
              f"(~2-3 GB VRAM each). Lower --workers if you hit OOM.")

    paths = find_hdf5_files(args.input)
    if not paths:
        print(f"[ERROR] No HDF5 files found at: {args.input}")
        return 2

    # Explicit --bounds is forwarded verbatim; otherwise bounds are derived
    # per-episode from the object-table geometry inside generate_occupancy_gt_for_file.
    bounds_override = config.get("bounds")

    process_one = functools.partial(
        generate_occupancy_gt_for_file,
        occupancy_config=occupancy_config,
        write_mode=args.write_mode,
        overwrite=args.overwrite,
        scene_override=args.scene,
        no_table=args.no_table,
        no_semantic=args.no_semantic,
        device=device_pref,
        bounds_override=bounds_override,
    )

    n_ok, n_err = parallel_process(process_one, paths, workers=workers, label="occupancy-gt")
    return 0 if n_ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
