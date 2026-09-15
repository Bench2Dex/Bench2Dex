#!/usr/bin/env python3
"""Generate TacMap surface point and normal maps from fingertip meshes.

This script has two modes:

1. Single geometry mode:
   Convert one mesh file into TacMap point/normal npy files.

   Example:
     python tools/asset/gen_tacmap_npy.py \\
       --mesh /path/to/fingertip.obj \\
       --output-dir /tmp/tacmap \\
       --output-stem tactileSensor_map_R4F \\
       --native-resolution 240 \\
       --sensor-normal auto \\
       --mesh-units m

   Useful single-geometry options:
     --origin-xyz "x,y,z"      Apply an extra mesh origin translation in meters.
     --origin-rpy "r,p,y"      Apply an extra mesh origin rotation in radians.
     --mesh-scale "sx,sy,sz"   Apply an extra mesh scale before generation.
     --mesh-units m|mm         Interpret mesh coordinates as meters or millimeters.

2. Robot batch mode:
   Generate configured TacMap files for one robot, or for all configured robots.
   Batch mode reads robot URDFs under --dataset-root/Robots_p/<dataset_key>/ and
   writes into each robot's tactile_sensor directory.

   Examples:
     python tools/asset/gen_tacmap_npy.py \\
       --robot-key multi_panda_with_allegro \\
       --dataset-root ../dex2bench_dataset

     python tools/asset/gen_tacmap_npy.py \\
       --robot-key all \\
       --dataset-root ../dex2bench_dataset

     python tools/asset/gen_tacmap_npy.py \\
       --robot-key multi_ur5_wuji_with_flange \\
       --dataset-root ../dex2bench_dataset \\
       --side right

     python tools/asset/gen_tacmap_npy.py \\
       --robot-key all \\
       --dataset-root ../dex2bench_dataset \\
       --include-sharpa

   Useful batch options:
     --robot-key all           Generate every configured non-Sharpa robot.
     --include-sharpa          Also regenerate Sharpa maps; omitted by default
                               to preserve old Sharpa assets unless requested.
     --side both|right|left    Generate both hands, right hand only, or left hand only.
     --geometry-role auto      Use per-robot default geometry role; default.
                               Orca uses collision skin meshes, others use visual.
     --geometry-role visual    Use visual geometry from URDF links.
     --geometry-role collision Use collision geometry instead.
     --dry-run                 Print selected jobs and hit rates without writing files.

Common tuning examples:
  Inspect selected jobs and hit rates without writing:
    python tools/asset/gen_tacmap_npy.py \\
      --robot-key multi_xarm7_with_leap \\
      --dataset-root ../dex2bench_dataset \\
      --dry-run

  Use an explicit sensor normal and relax the hit-rate threshold:
    python tools/asset/gen_tacmap_npy.py \\
      --robot-key multi_xarm7_with_leap \\
      --dataset-root ../dex2bench_dataset \\
      --sensor-normal 0,0,1 \\
      --min-hit-rate 0.45

Known --robot-key values:
  all
  multi_iiwa7_with_sharpa
  multi_ur5_rh56dfx_with_flange
  multi_ur5_rh5dg2_with_flange
  multi_ur5_shadow_hand_with_flange
  multi_ur5_schunk_hand_with_flange
  multi_ur5_wuji_with_flange
  multi_panda_with_allegro
  multi_panda_with_orca
  multi_xarm7_with_ability
  multi_xarm7_with_leap
  multi_jaka_zu7_dexhand021_with_flange

Output files:
  <output_stem>_point.npy   Surface sample points in millimeters.
  <output_stem>_normal.npy  Inward-facing surface normals.

Batch-mode note:
  Batch mode resolves each TacMap group to a representative contact link. If a
  future configuration intentionally separates the TacMap runtime attachment
  link from the geometry source link, the script transforms source geometry back
  into the attach-link local frame before writing npy files.
"""

from __future__ import annotations

import argparse
import math
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from collector.tacmap_configs import ROBOT_KEY_TO_TACMAP_CFG, TacMapNpyGroup, resolve_tacmap_site_body_name


URDF_BY_ROBOT_KEY = {
    "multi_iiwa7_with_sharpa": "multi_iiwa7_with_sharpa.urdf",
    "multi_ur5_rh56dfx_with_flange": "urdf/Multi_UR5_RH56DFX_with_flange.urdf",
    "multi_ur5_rh5dg2_with_flange": "urdf/Multi_UR5_RH5DG2_with_flange.urdf",
    "multi_ur5_shadow_hand_with_flange": "urdf/Multi_UR5_shadow_hand_with_flange.urdf",
    "multi_ur5_schunk_hand_with_flange": "urdf/Multi_UR5_schunk_hand_with_flange.urdf",
    "multi_ur5_wuji_with_flange": "urdf/Multi_UR5_wuji_with_flange.urdf",
    "multi_panda_with_allegro": "multi_panda_with_allegro.urdf",
    "multi_panda_with_orca": "multi_panda_with_orca.urdf",
    "multi_xarm7_with_ability": "multi_xarm7_with_ability.urdf",
    "multi_xarm7_with_leap": "multi_xarm7_with_leap.urdf",
    "multi_jaka_zu7_dexhand021_with_flange": "urdf/Multi_jaka_zu7_dexhand021_with_flange_with_flange.urdf",
    "multi_rm_65_with_revo2": "urdf/muitl_rm_65_with_revo2_.urdf",
    "multi_rm_75_with_rohand": "multi_rm_75_with_rohand.urdf",
}


Matrix4 = tuple[tuple[float, float, float, float], ...]
IDENTITY_MATRIX: Matrix4 = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)


@dataclass(frozen=True)
class MeshSpec:
    mesh_path: Path | None = None
    origin_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    origin_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)
    mesh_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    primitive_kind: str | None = None
    primitive_size: tuple[float, ...] = ()
    source_to_target: Matrix4 = IDENTITY_MATRIX


@dataclass(frozen=True)
class BatchJob:
    robot_key: str
    dataset_key: str
    group: TacMapNpyGroup
    site_name: str
    attach_link: str
    source_link: str
    urdf_path: Path
    geometry_role: str
    output_dir: Path
    output_stem: str


def _parse_vec3(value: str | None, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if not value:
        return default
    parts = value.replace(",", " ").split()
    if len(parts) != 3:
        raise ValueError(f"Expected three numeric values, got {value!r}")
    return float(parts[0]), float(parts[1]), float(parts[2])


def _normalize(value: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(value)
    if norm <= 1e-12:
        raise ValueError("Cannot normalize a zero vector.")
    return value / norm


def _rpy_matrix(rpy: tuple[float, float, float]) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rz @ ry @ rx


def _origin_matrix(xyz: tuple[float, float, float], rpy: tuple[float, float, float]) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = _rpy_matrix(rpy)
    matrix[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return matrix


def _load_trimesh(spec: MeshSpec, mesh_units: str):
    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError("trimesh is required to generate TacMap .npy assets.") from exc

    if spec.mesh_path is not None:
        mesh = trimesh.load(spec.mesh_path.as_posix(), force="mesh")
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
    elif spec.primitive_kind == "sphere":
        if len(spec.primitive_size) != 1:
            raise ValueError(f"Sphere primitive requires radius, got {spec.primitive_size!r}")
        mesh = trimesh.creation.icosphere(subdivisions=4, radius=float(spec.primitive_size[0]))
    elif spec.primitive_kind == "box":
        if len(spec.primitive_size) != 3:
            raise ValueError(f"Box primitive requires size xyz, got {spec.primitive_size!r}")
        mesh = trimesh.creation.box(extents=spec.primitive_size)
    elif spec.primitive_kind == "cylinder":
        if len(spec.primitive_size) != 2:
            raise ValueError(f"Cylinder primitive requires radius and length, got {spec.primitive_size!r}")
        mesh = trimesh.creation.cylinder(radius=float(spec.primitive_size[0]), height=float(spec.primitive_size[1]))
    else:
        raise ValueError("MeshSpec must provide either mesh_path or primitive geometry.")

    if mesh.is_empty:
        raise ValueError(f"Geometry is empty: {_spec_geometry_label(spec)}")

    mesh = mesh.copy()
    unit_scale = 1.0 if spec.mesh_path is None or mesh_units == "m" else 1e-3
    scale_matrix = np.eye(4, dtype=np.float64)
    scale_matrix[:3, :3] = np.diag(np.asarray(spec.mesh_scale, dtype=np.float64) * unit_scale)
    mesh.apply_transform(scale_matrix)
    mesh.apply_transform(_origin_matrix(spec.origin_xyz, spec.origin_rpy))
    mesh.apply_transform(np.asarray(spec.source_to_target, dtype=np.float64))
    mesh.remove_unreferenced_vertices()
    return mesh


def _spec_geometry_label(spec: MeshSpec) -> str:
    if spec.mesh_path is not None:
        return spec.mesh_path.as_posix()
    return f"{spec.primitive_kind}{tuple(spec.primitive_size)}"


def _basis_from_normal(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normal = _normalize(np.asarray(normal, dtype=np.float64))
    tmp = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(tmp, normal))) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    u_axis = _normalize(np.cross(normal, tmp))
    v_axis = _normalize(np.cross(normal, u_axis))
    return u_axis, v_axis


def _area_weighted_sign(mesh, axis: np.ndarray) -> np.ndarray:
    face_normals = np.asarray(mesh.face_normals, dtype=np.float64)
    face_areas = np.asarray(mesh.area_faces, dtype=np.float64)
    pos_score = np.sum(np.maximum(face_normals @ axis, 0.0) * face_areas)
    neg_score = np.sum(np.maximum(face_normals @ -axis, 0.0) * face_areas)
    return axis if pos_score >= neg_score else -axis


_SENSOR_NORMAL_OVERRIDES: dict[tuple[str, str], tuple[float, float, float]] = {
    ("ur5+RH56DFX", "RTH"): (0, 0, 1),
    ("ur5+RH56DFX", "R4F"): (0, 0, 1),
    ("ur5+RH56DFX", "RIDX"): (0, 0, 1),
    ("ur5+RH56DFX", "RMID"): (0, 0, 1),
    ("ur5+RH56DFX", "RRING"): (0, 0, 1),
    ("ur5+RH56DFX", "RLIT"): (0, 0, 1),
    ("ur5+RH56DFX", "LTH"): (0, 0, 1),
    ("ur5+RH56DFX", "L4F"): (0, 0, 1),
    ("ur5+RH56DFX", "LIDX"): (0, 0, 1),
    ("ur5+RH56DFX", "LMID"): (0, 0, 1),
    ("ur5+RH56DFX", "LRING"): (0, 0, 1),
    ("ur5+RH56DFX", "LLIT"): (0, 0, 1),
    ("ur5+schunk_hand", "R4F"): (0, 1, 0),
    ("ur5+schunk_hand", "L4F"): (0, 1, 0),
    ("ur5+wuji", "RTH"): (-1, 0, 0),
    ("ur5+wuji", "LTH"): (1, 0, 0),
    ("panda+allegro", "RTH"): (1, 0, 0),
    ("panda+allegro", "LTH"): (-1, 0, 0),
    ("panda+allegro", "R4F"): (1, 0, 0),
    ("panda+allegro", "L4F"): (-1, 0, 0),
}

_GEOMETRY_ROLE_OVERRIDES = {
    "multi_panda_with_orca": "collision",
}


def _auto_sensor_normal(mesh, *, urdf_hint: np.ndarray | None = None) -> np.ndarray:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    centered = vertices - vertices.mean(axis=0)
    eigvals, eigvecs = np.linalg.eigh(np.cov(centered.T))

    thin_ratio = float(np.sqrt(eigvals[0] / max(eigvals[2], 1e-12)))
    if urdf_hint is not None and thin_ratio > 0.4:
        axis = _normalize(np.asarray(urdf_hint, dtype=np.float64))
    else:
        axis = _normalize(eigvecs[:, int(np.argmin(eigvals))])

    return _area_weighted_sign(mesh, axis)


def _urdf_pad_normal(root: ET.Element, link_name: str) -> np.ndarray | None:
    """Compute pad-facing normal from URDF joint chain: cross(toward_parent, curl_axis)."""

    child_to_parent = _child_to_parent_joints(root)
    joint_types: dict[str, str] = {}
    joint_axes: dict[str, np.ndarray] = {}
    for joint in root.findall("joint"):
        child_elem = joint.find("child")
        if child_elem is None:
            continue
        cname = child_elem.get("link", "")
        joint_types[cname] = joint.get("type", "fixed")
        axis_elem = joint.find("axis")
        if axis_elem is not None:
            joint_axes[cname] = np.asarray(
                [float(x) for x in axis_elem.get("xyz", "0 0 1").split()], dtype=np.float64,
            )
        else:
            joint_axes[cname] = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    current = str(link_name)
    r_accum = np.eye(3, dtype=np.float64)
    toward_parent: np.ndarray | None = None
    curl_axis: np.ndarray | None = None

    for _ in range(15):
        if current not in child_to_parent:
            break
        parent_name, parent_to_child = child_to_parent[current]
        r_joint = parent_to_child[:3, :3]
        xyz = parent_to_child[:3, 3]

        if toward_parent is None and np.linalg.norm(xyz) > 1e-6:
            toward_parent = r_accum.T @ (r_joint.T @ (-xyz / np.linalg.norm(xyz)))

        jtype = joint_types.get(current, "fixed")
        if jtype in ("revolute", "continuous"):
            curl_axis = r_accum.T @ joint_axes.get(current, np.array([0.0, 0.0, 1.0]))

        r_accum = r_joint @ r_accum
        current = parent_name

    if toward_parent is None or curl_axis is None:
        return None
    pad = np.cross(toward_parent, curl_axis)
    if np.linalg.norm(pad) < 1e-6:
        return None
    return _normalize(pad)


def _raycast_with_trimesh(mesh, origins: np.ndarray, directions: np.ndarray):
    try:
        locations, index_ray, index_tri = mesh.ray.intersects_location(
            origins,
            directions,
            multiple_hits=False,
        )
    except Exception:
        return None

    points = np.zeros_like(origins, dtype=np.float64)
    tri_ids = np.full((origins.shape[0],), -1, dtype=np.int64)
    hit_mask = np.zeros((origins.shape[0],), dtype=bool)
    if len(index_ray):
        points[index_ray] = locations
        tri_ids[index_ray] = index_tri
        hit_mask[index_ray] = True
    return points, tri_ids, hit_mask


def _raycast_moller(mesh, origins: np.ndarray, directions: np.ndarray):
    triangles = np.asarray(mesh.triangles, dtype=np.float64)
    v0 = triangles[:, 0]
    edge1 = triangles[:, 1] - v0
    edge2 = triangles[:, 2] - v0
    eps = 1e-10

    points = np.zeros_like(origins, dtype=np.float64)
    tri_ids = np.full((origins.shape[0],), -1, dtype=np.int64)
    hit_mask = np.zeros((origins.shape[0],), dtype=bool)

    for ray_idx, (origin, direction) in enumerate(zip(origins, directions)):
        pvec = np.cross(np.broadcast_to(direction, edge2.shape), edge2)
        det = np.einsum("ij,ij->i", edge1, pvec)
        valid = np.abs(det) > eps
        if not np.any(valid):
            continue

        inv_det = np.zeros_like(det)
        inv_det[valid] = 1.0 / det[valid]
        tvec = origin - v0
        u = np.einsum("ij,ij->i", tvec, pvec) * inv_det
        qvec = np.cross(tvec, edge1)
        v = np.einsum("j,ij->i", direction, qvec) * inv_det
        t = np.einsum("ij,ij->i", edge2, qvec) * inv_det

        candidates = valid & (u >= 0.0) & (v >= 0.0) & ((u + v) <= 1.0) & (t > eps)
        if not np.any(candidates):
            continue
        candidate_ids = np.flatnonzero(candidates)
        best_local = int(np.argmin(t[candidate_ids]))
        tri_id = int(candidate_ids[best_local])
        points[ray_idx] = origin + direction * t[tri_id]
        tri_ids[ray_idx] = tri_id
        hit_mask[ray_idx] = True

    return points, tri_ids, hit_mask


def _raycast(mesh, origins: np.ndarray, directions: np.ndarray):
    result = _raycast_with_trimesh(mesh, origins, directions)
    if result is not None:
        return result
    return _raycast_moller(mesh, origins, directions)


def _fill_missing_nearest(
    points: np.ndarray,
    normals: np.ndarray,
    hit_mask: np.ndarray,
    native_resolution: int,
) -> None:
    if bool(np.all(hit_mask)):
        return
    valid_ids = np.flatnonzero(hit_mask)
    if len(valid_ids) == 0:
        return

    all_ids = np.arange(native_resolution * native_resolution, dtype=np.int64)
    rows = all_ids // native_resolution
    cols = all_ids % native_resolution
    valid_rows = rows[valid_ids]
    valid_cols = cols[valid_ids]

    for missing_id in np.flatnonzero(~hit_mask):
        d_rows = valid_rows - rows[missing_id]
        d_cols = valid_cols - cols[missing_id]
        nearest_id = valid_ids[int(np.argmin(d_rows * d_rows + d_cols * d_cols))]
        points[missing_id] = points[nearest_id]
        normals[missing_id] = normals[nearest_id]


def generate_maps(
    *,
    mesh,
    native_resolution: int,
    sensor_normal: str,
    padding_ratio: float,
    min_hit_rate: float,
    urdf_hint: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    if native_resolution <= 0:
        raise ValueError(f"native_resolution must be positive, got {native_resolution!r}")

    if sensor_normal == "auto":
        normal = _auto_sensor_normal(mesh, urdf_hint=urdf_hint)
    else:
        normal = _normalize(np.asarray(_parse_vec3(sensor_normal, (0.0, 0.0, 1.0)), dtype=np.float64))
    u_axis, v_axis = _basis_from_normal(normal)

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    u_coord = vertices @ u_axis
    v_coord = vertices @ v_axis
    n_coord = vertices @ normal
    u_min, u_max = float(u_coord.min()), float(u_coord.max())
    v_min, v_max = float(v_coord.min()), float(v_coord.max())
    span = max(u_max - u_min, v_max - v_min, 1e-6)
    padding = span * float(padding_ratio)
    u_grid = np.linspace(u_min - padding, u_max + padding, native_resolution, dtype=np.float64)
    v_grid = np.linspace(v_min - padding, v_max + padding, native_resolution, dtype=np.float64)
    uu, vv = np.meshgrid(u_grid, v_grid, indexing="xy")

    margin = span * 0.1 + 1e-3
    start_n = float(n_coord.max()) + margin
    origins = (
        uu.reshape(-1, 1) * u_axis.reshape(1, 3)
        + vv.reshape(-1, 1) * v_axis.reshape(1, 3)
        + start_n * normal.reshape(1, 3)
    )
    ray_direction = -normal
    directions = np.broadcast_to(ray_direction.reshape(1, 3), origins.shape).copy()

    hit_points_m, tri_ids, hit_mask = _raycast(mesh, origins, directions)
    hit_rate = float(np.mean(hit_mask))
    if hit_rate < float(min_hit_rate):
        raise RuntimeError(
            f"Hit rate {hit_rate:.3f} is below --min-hit-rate {min_hit_rate:.3f}. "
            "Pass an explicit --sensor-normal or inspect the mesh/origin."
        )

    normals = np.zeros_like(hit_points_m, dtype=np.float64)
    face_normals = np.asarray(mesh.face_normals, dtype=np.float64)
    for ray_idx in np.flatnonzero(hit_mask):
        inward = -face_normals[int(tri_ids[ray_idx])]
        inward = _normalize(inward)
        if np.dot(inward, ray_direction) < 0.0:
            inward = -inward
        normals[ray_idx] = inward

    _fill_missing_nearest(hit_points_m, normals, hit_mask, native_resolution)

    points_mm = (hit_points_m * 1000.0).reshape(native_resolution, native_resolution, 3)
    normals = normals.reshape(native_resolution, native_resolution, 3)
    return points_mm.astype(np.float32), normals.astype(np.float32), hit_rate


def _find_link(root: ET.Element, link_name: str, urdf_path: Path) -> ET.Element:
    link = next((elem for elem in root.findall("link") if elem.get("name") == link_name), None)
    if link is None:
        raise ValueError(f"Link {link_name!r} not found in {urdf_path}")
    return link


def _geometry_spec_from_element(
    *,
    geom_parent: ET.Element,
    urdf_path: Path,
    source_to_target: Matrix4,
) -> MeshSpec | None:
    geometry = geom_parent.find("geometry")
    if geometry is None:
        return None

    origin = geom_parent.find("origin")
    xyz = _parse_vec3(origin.get("xyz") if origin is not None else None, (0.0, 0.0, 0.0))
    rpy = _parse_vec3(origin.get("rpy") if origin is not None else None, (0.0, 0.0, 0.0))

    mesh = geometry.find("mesh")
    if mesh is not None and mesh.get("filename"):
        scale = _parse_vec3(mesh.get("scale"), (1.0, 1.0, 1.0))
        mesh_path = (urdf_path.parent / mesh.get("filename")).resolve()
        return MeshSpec(
            mesh_path=mesh_path,
            origin_xyz=xyz,
            origin_rpy=rpy,
            mesh_scale=scale,
            source_to_target=source_to_target,
        )

    sphere = geometry.find("sphere")
    if sphere is not None and sphere.get("radius"):
        return MeshSpec(
            origin_xyz=xyz,
            origin_rpy=rpy,
            primitive_kind="sphere",
            primitive_size=(float(sphere.get("radius")),),
            source_to_target=source_to_target,
        )

    box = geometry.find("box")
    if box is not None and box.get("size"):
        return MeshSpec(
            origin_xyz=xyz,
            origin_rpy=rpy,
            primitive_kind="box",
            primitive_size=_parse_vec3(box.get("size"), (0.0, 0.0, 0.0)),
            source_to_target=source_to_target,
        )

    cylinder = geometry.find("cylinder")
    if cylinder is not None and cylinder.get("radius") and cylinder.get("length"):
        return MeshSpec(
            origin_xyz=xyz,
            origin_rpy=rpy,
            primitive_kind="cylinder",
            primitive_size=(float(cylinder.get("radius")), float(cylinder.get("length"))),
            source_to_target=source_to_target,
        )

    return None


def _joint_link_name(joint: ET.Element, tag: str) -> str | None:
    elem = joint.find(tag)
    return elem.get("link") if elem is not None else None


def _child_to_parent_joints(root: ET.Element) -> dict[str, tuple[str, np.ndarray]]:
    joints: dict[str, tuple[str, np.ndarray]] = {}
    for joint in root.findall("joint"):
        parent = _joint_link_name(joint, "parent")
        child = _joint_link_name(joint, "child")
        if not parent or not child:
            continue
        origin = joint.find("origin")
        xyz = _parse_vec3(origin.get("xyz") if origin is not None else None, (0.0, 0.0, 0.0))
        rpy = _parse_vec3(origin.get("rpy") if origin is not None else None, (0.0, 0.0, 0.0))
        joints[child] = (parent, _origin_matrix(xyz, rpy))
    return joints


def _root_to_link_transform(root: ET.Element, link_name: str) -> tuple[str, np.ndarray]:
    child_to_parent = _child_to_parent_joints(root)
    transform = np.eye(4, dtype=np.float64)
    current = str(link_name)
    visited = set()
    while current in child_to_parent:
        if current in visited:
            raise ValueError(f"Cycle detected while resolving URDF transform for {link_name!r}")
        visited.add(current)
        parent, parent_to_child = child_to_parent[current]
        transform = parent_to_child @ transform
        current = parent
    return current, transform


def _source_to_attach_transform(root: ET.Element, attach_link: str, source_link: str) -> Matrix4:
    if attach_link == source_link:
        return IDENTITY_MATRIX

    attach_root, root_to_attach = _root_to_link_transform(root, attach_link)
    source_root, root_to_source = _root_to_link_transform(root, source_link)
    if attach_root != source_root:
        raise ValueError(
            f"Attach link {attach_link!r} and source link {source_link!r} are in different URDF trees: "
            f"{attach_root!r} vs {source_root!r}"
        )
    transform = np.linalg.inv(root_to_attach) @ root_to_source
    return tuple(tuple(float(value) for value in row) for row in transform)


def _has_skin_mesh(geom_parent: ET.Element) -> bool:
    geometry = geom_parent.find("geometry")
    if geometry is None:
        return False
    mesh = geometry.find("mesh")
    if mesh is None:
        return False
    filename = mesh.get("filename", "")
    return "skin" in Path(filename).stem.lower()


def _mesh_spec_from_urdf(urdf_path: Path, attach_link: str, source_link: str, geometry_role: str) -> MeshSpec:
    root = ET.parse(urdf_path).getroot()
    link = _find_link(root, source_link, urdf_path)
    _find_link(root, attach_link, urdf_path)
    source_to_attach = _source_to_attach_transform(root, attach_link, source_link)

    roles = (geometry_role, "collision" if geometry_role == "visual" else "visual")
    for role in roles:
        elements = link.findall(role)
        elements.sort(key=lambda e: 0 if _has_skin_mesh(e) else 1)
        for geom_parent in elements:
            spec = _geometry_spec_from_element(
                geom_parent=geom_parent,
                urdf_path=urdf_path,
                source_to_target=source_to_attach,
            )
            if spec is not None:
                return spec

    raise ValueError(f"No supported geometry found for link {source_link!r} in {urdf_path}")


def _robots_root(dataset_root: Path) -> Path:
    dataset_root = dataset_root.resolve()
    if (dataset_root / "Robots_p").is_dir():
        return dataset_root / "Robots_p"
    return dataset_root


def _output_stem(points_npy: str) -> str:
    suffix = "_point.npy"
    if not points_npy.endswith(suffix):
        raise ValueError(f"Unexpected TacMap points filename: {points_npy}")
    return points_npy[: -len(suffix)]


def _source_link_for_job(robot_key: str, site_name: str, attach_link: str) -> str:
    if robot_key == "multi_panda_with_allegro" and not attach_link.endswith("_tip"):
        return f"{attach_link}_tip"
    if robot_key == "multi_ur5_wuji_with_flange" and attach_link.endswith("_link4"):
        return attach_link[: -len("_link4")] + "_tip_link"
    return attach_link


def _geometry_role_for_job(robot_key: str, requested_role: str) -> str:
    if requested_role != "auto":
        return requested_role
    return _GEOMETRY_ROLE_OVERRIDES.get(str(robot_key), "visual")


def _build_batch_jobs(args: argparse.Namespace) -> list[BatchJob]:
    selected_keys = (
        tuple(ROBOT_KEY_TO_TACMAP_CFG)
        if args.robot_key == "all"
        else (str(args.robot_key),)
    )
    side = str(args.side)
    robots_root = _robots_root(Path(args.dataset_root))
    jobs: list[BatchJob] = []

    for robot_key in selected_keys:
        cfg = ROBOT_KEY_TO_TACMAP_CFG.get(robot_key)
        if cfg is None:
            raise ValueError(f"TacMap not configured for robot key: {robot_key}")
        if robot_key == "multi_iiwa7_with_sharpa" and not args.include_sharpa:
            continue

        robot_dir = robots_root / cfg.dataset_key
        urdf_rel = URDF_BY_ROBOT_KEY[robot_key]
        urdf_path = (robot_dir / urdf_rel).resolve()
        if not urdf_path.is_file():
            raise FileNotFoundError(f"URDF not found for {robot_key}: {urdf_path}")

        for group in cfg.npy_groups:
            if not group.site_names:
                continue
            if side == "right" and not group.name.startswith("R"):
                continue
            if side == "left" and not group.name.startswith("L"):
                continue
            site_name = group.site_names[0]
            attach_link = resolve_tacmap_site_body_name(robot_key, site_name)
            source_link = _source_link_for_job(robot_key, site_name, attach_link)
            geometry_role = _geometry_role_for_job(robot_key, str(args.geometry_role))
            jobs.append(
                BatchJob(
                    robot_key=robot_key,
                    dataset_key=cfg.dataset_key,
                    group=group,
                    site_name=site_name,
                    attach_link=attach_link,
                    source_link=source_link,
                    urdf_path=urdf_path,
                    geometry_role=geometry_role,
                    output_dir=robot_dir / "tactile_sensor",
                    output_stem=_output_stem(group.points_npy),
                )
            )
    return jobs


def _write_maps(output_dir: Path, output_stem: str, points: np.ndarray, normals: np.ndarray) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    points_path = output_dir / f"{output_stem}_point.npy"
    normals_path = output_dir / f"{output_stem}_normal.npy"
    np.save(points_path, points)
    np.save(normals_path, normals)
    return points_path, normals_path


def _run_one(
    *,
    spec: MeshSpec,
    output_dir: Path,
    output_stem: str,
    native_resolution: int,
    sensor_normal: str,
    mesh_units: str,
    geometry_label: str,
    padding_ratio: float,
    min_hit_rate: float,
    dry_run: bool,
    urdf_hint: np.ndarray | None = None,
) -> None:
    mesh = _load_trimesh(spec, mesh_units)
    points, normals, hit_rate = generate_maps(
        mesh=mesh,
        native_resolution=native_resolution,
        sensor_normal=sensor_normal,
        padding_ratio=padding_ratio,
        min_hit_rate=min_hit_rate,
        urdf_hint=urdf_hint,
    )
    print(
        f"[gen_tacmap_npy] {geometry_label}: shape={points.shape} dtype={points.dtype} "
        f"hit_rate={hit_rate:.3f} geometry={_spec_geometry_label(spec)}"
    )
    if dry_run:
        return
    points_path, normals_path = _write_maps(output_dir, output_stem, points, normals)
    print(f"[gen_tacmap_npy] wrote {points_path}")
    print(f"[gen_tacmap_npy] wrote {normals_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--mesh", type=Path, help="Single mesh path to convert.")
    mode.add_argument("--robot-key", help="Configured robot key to generate, or 'all'.")
    parser.add_argument("--dataset-root", type=Path, default=REPO_ROOT / "../dex2bench_dataset")
    parser.add_argument("--output-dir", type=Path, help="Output directory for single-mesh mode.")
    parser.add_argument("--output-stem", default="tactileSensor_map", help="Output basename without _point/_normal.npy.")
    parser.add_argument("--native-resolution", type=int, default=240)
    parser.add_argument("--sensor-normal", default="auto", help="'auto' or comma-separated vector, e.g. 0,0,1.")
    parser.add_argument("--mesh-units", choices=["m", "mm"], default="m")
    parser.add_argument("--origin-xyz", default=None, help="Single-mesh mode mesh origin xyz in meters.")
    parser.add_argument("--origin-rpy", default=None, help="Single-mesh mode mesh origin rpy in radians.")
    parser.add_argument("--mesh-scale", default=None, help="Single-mesh mode mesh scale.")
    parser.add_argument("--geometry-role", choices=["auto", "visual", "collision"], default="auto")
    parser.add_argument("--side", choices=["both", "right", "left"], default="both")
    parser.add_argument("--include-sharpa", action="store_true", help="Regenerate Sharpa maps in batch mode.")
    parser.add_argument("--padding-ratio", type=float, default=0.0)
    parser.add_argument("--min-hit-rate", type=float, default=0.50)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.mesh is not None:
        output_dir = Path(args.output_dir) if args.output_dir is not None else Path.cwd()
        spec = MeshSpec(
            mesh_path=Path(args.mesh).resolve(),
            origin_xyz=_parse_vec3(args.origin_xyz, (0.0, 0.0, 0.0)),
            origin_rpy=_parse_vec3(args.origin_rpy, (0.0, 0.0, 0.0)),
            mesh_scale=_parse_vec3(args.mesh_scale, (1.0, 1.0, 1.0)),
        )
        _run_one(
            spec=spec,
            output_dir=output_dir,
            output_stem=str(args.output_stem),
            native_resolution=int(args.native_resolution),
            sensor_normal=str(args.sensor_normal),
            mesh_units=str(args.mesh_units),
            geometry_label=args.mesh.as_posix(),
            padding_ratio=float(args.padding_ratio),
            min_hit_rate=float(args.min_hit_rate),
            dry_run=bool(args.dry_run),
        )
        return 0

    jobs = _build_batch_jobs(args)
    if not jobs:
        print("[gen_tacmap_npy] no jobs selected")
        return 0

    for job in jobs:
        spec = _mesh_spec_from_urdf(job.urdf_path, job.attach_link, job.source_link, job.geometry_role)
        cfg = ROBOT_KEY_TO_TACMAP_CFG[job.robot_key]
        override_key = (cfg.dataset_key, job.group.name)
        override = _SENSOR_NORMAL_OVERRIDES.get(override_key)
        if override is not None:
            sensor_normal = ",".join(str(v) for v in override)
            hint = None
        else:
            sensor_normal = str(args.sensor_normal)
            if sensor_normal == "auto":
                urdf_root = ET.parse(job.urdf_path).getroot()
                hint = _urdf_pad_normal(urdf_root, job.attach_link)
            else:
                hint = None
        label = (
            f"{job.robot_key}:{job.group.name}:{job.site_name}"
            f" attach={job.attach_link} source={job.source_link} geometry_role={job.geometry_role}"
        )
        _run_one(
            spec=spec,
            output_dir=job.output_dir,
            output_stem=job.output_stem,
            native_resolution=int(cfg.native_resolution),
            sensor_normal=sensor_normal,
            mesh_units=str(args.mesh_units),
            geometry_label=label,
            padding_ratio=float(args.padding_ratio),
            min_hit_rate=float(args.min_hit_rate),
            dry_run=bool(args.dry_run),
            urdf_hint=hint,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
