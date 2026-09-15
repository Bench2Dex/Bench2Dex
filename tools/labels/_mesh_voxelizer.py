"""GT occupancy via collision-mesh voxelization.

Three core components:
  A. extract_meshes_from_usd() — USD collision/visual mesh extraction with caching
  B. ArticulationFK — forward-kinematics solver for articulated bodies
  C. voxelize_scene_frame() — per-frame scene voxelization orchestrator

No Isaac Sim dependency — uses only pxr (usd-core) + trimesh + numpy.
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from collector.camera_geometry import quat_xyzw_to_rot, quat_wxyz_to_rot

try:
    import trimesh
except ImportError:
    trimesh = None  # type: ignore[assignment]

try:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
except ImportError:
    Usd = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Occupancy state constants (matching collector/occupancy.py)
# ---------------------------------------------------------------------------
_UNKNOWN = 0
_FREE = 1
_OCCUPIED = 2


# ---------------------------------------------------------------------------
# A. USD mesh extraction
# ---------------------------------------------------------------------------

@dataclass
class LinkMesh:
    """A single link's mesh in link-local coordinates."""
    mesh: Any  # trimesh.Trimesh
    link_name: str
    is_collision: bool


# Cache: (usd_path, scale_tuple, source, articulated) -> list[LinkMesh]
_mesh_cache: dict[tuple[str, tuple[float, float, float], str, bool], list[LinkMesh]] = {}


def _scale_vector(scale: Any) -> np.ndarray:
    """Normalize scalar or xyz asset scale to a 3-vector."""
    arr = np.asarray(scale, dtype=np.float64)
    if arr.shape == ():
        return np.full(3, float(arr), dtype=np.float64)
    arr = arr.reshape(-1)
    if arr.size != 3:
        raise ValueError(f"scale must be scalar or length-3, got {scale!r}")
    return arr.astype(np.float64)


def _scale_cache_key(scale: Any) -> tuple[float, float, float]:
    return tuple(float(v) for v in _scale_vector(scale))


def _root_transform_from_pose(
    root_pose_xyzw: np.ndarray,
    scale: Any = 1.0,
) -> np.ndarray:
    """Build root-to-world affine transform with asset scale in root axes."""
    root_rot = quat_xyzw_to_rot(root_pose_xyzw[3:7])
    scale_mat = np.diag(_scale_vector(scale)).astype(np.float32)
    T_root = np.eye(4, dtype=np.float32)
    T_root[:3, :3] = root_rot @ scale_mat
    T_root[:3, 3] = root_pose_xyzw[:3]
    return T_root


def _gf_matrix_to_numpy(m: "Gf.Matrix4d") -> np.ndarray:
    """Convert pxr Gf.Matrix4d to numpy 4x4 (row-major)."""
    arr = np.zeros((4, 4), dtype=np.float64)
    for i in range(4):
        for j in range(4):
            arr[i, j] = m[i][j]
    return arr


def _gf_quatf_to_wxyz(q: "Gf.Quatf") -> np.ndarray:
    """Convert pxr Gf.Quatf (real, imaginary) to [w, x, y, z]."""
    return np.array([q.GetReal(), *q.GetImaginary()], dtype=np.float64)


def _read_usd_mesh_prim(mesh_prim: "UsdGeom.Mesh") -> "trimesh.Trimesh | None":
    """Read triangle mesh data from a UsdGeom.Mesh prim."""
    points_attr = mesh_prim.GetPointsAttr()
    indices_attr = mesh_prim.GetFaceVertexIndicesAttr()
    counts_attr = mesh_prim.GetFaceVertexCountsAttr()

    if not points_attr.HasValue() or not indices_attr.HasValue():
        return None

    points = np.asarray(points_attr.Get(), dtype=np.float64)
    if points.shape[0] == 0:
        return None

    indices = np.asarray(indices_attr.Get(), dtype=np.int64)
    counts = np.asarray(counts_attr.Get(), dtype=np.int64)

    # Only handle triangle meshes; triangulate quads/polygons
    faces = []
    offset = 0
    for n in counts:
        if n < 3:
            offset += n
            continue
        if n == 3:
            faces.append(indices[offset:offset + 3])
        else:
            # Fan triangulation for polygons
            for k in range(1, n - 1):
                faces.append([indices[offset], indices[offset + k], indices[offset + k + 1]])
        offset += n

    if not faces:
        return None

    faces_arr = np.array(faces, dtype=np.int64)
    return trimesh.Trimesh(vertices=points, faces=faces_arr, process=False)


def _extract_link_name(prim_path: str) -> str:
    """Derive link name from prim path.

    Examples:
        /colliders/link_0/Mesh -> link_0
        /link_0/collision -> link_0
        /base_link -> base_link
        /partnet_xxx/base/visuals/frame_2/World/mesh -> base
        /partnet_xxx/link_0/collisions/original_3/World/mesh -> link_0
    """
    parts = prim_path.strip("/").split("/")
    # Skip known wrapper / container names
    skip = {"colliders", "collision", "collisions", "visual", "visuals",
            "mesh", "world", "geometry"}
    for p in reversed(parts):
        low = p.lower()
        if low in skip:
            continue
        # Skip auto-generated container names (original_N, frame_N)
        if low.startswith(("original_", "frame_")):
            continue
        return p
    return parts[-1] if parts else "unknown"


def _find_link_ancestor(
    prim: "Usd.Prim",
    root_prim: "Usd.Prim",
) -> "Usd.Prim | None":
    """Walk up from *prim* to the first ancestor whose parent is *root_prim*."""
    current = prim
    while current and current.IsValid():
        parent = current.GetParent()
        if parent is None or not parent.IsValid() or parent.IsPseudoRoot():
            return root_prim
        if parent.GetPath() == root_prim.GetPath():
            return current
        current = parent
    return root_prim


def extract_meshes_from_usd(
    usd_path: str,
    scale: float | tuple[float, float, float] | np.ndarray = 1.0,
    source: str = "collision",
    articulated: bool = False,
) -> list[LinkMesh]:
    """Extract collision or visual meshes from a USD asset file.

    Args:
        usd_path: Path to USD file.
        scale: Scalar or xyz asset scale. For rigid assets, scale is baked into
            root-frame mesh vertices. For articulations, scale is applied in
            the root transform during FK so link-local axes are not distorted.
        source: "collision" (default) or "visual".
        articulated: If True, return meshes in link-local coordinates
            (for FK-driven bodies).  Otherwise bake full local-to-world.

    Returns:
        List of LinkMesh objects in link-local frame.
    """
    if trimesh is None:
        raise ImportError("trimesh is required for mesh voxelization: pip install trimesh>=4.0")
    if Usd is None:
        raise ImportError("pxr (usd-core) is required for USD reading: pip install usd-core>=26.0")

    scale_vec = _scale_vector(scale)
    cache_key = (usd_path, _scale_cache_key(scale_vec), source, articulated)
    if cache_key in _mesh_cache:
        return _mesh_cache[cache_key]

    if not os.path.isfile(usd_path):
        warnings.warn(f"USD file not found: {usd_path}")
        return []

    stage = Usd.Stage.Open(usd_path, Usd.Stage.LoadAll)
    if stage is None:
        warnings.warn(f"Failed to open USD stage: {usd_path}")
        return []

    # Find root prim for link-ancestor lookup (articulated bodies)
    root_prim = None
    if articulated:
        root_prim = stage.GetDefaultPrim()
        if root_prim is None or not root_prim.IsValid():
            for p in stage.GetPseudoRoot().GetChildren():
                if p.IsValid():
                    root_prim = p
                    break

    # Also try physics layer for collision data (URDF-converted assets)
    physics_usd = Path(usd_path).parent / "configuration" / f"{Path(usd_path).stem}_physics.usd"
    physics_stage = None
    if physics_usd.exists():
        physics_stage = Usd.Stage.Open(str(physics_usd), Usd.Stage.LoadAll)

    results: list[LinkMesh] = []

    # Rigid meshes are already in asset/root coordinates, so xyz scale can be
    # baked into vertices. Articulated meshes are stored in link-local frames;
    # root-axis non-uniform scale must be applied later as part of FK.
    mesh_scale = np.ones(3, dtype=np.float64) if articulated else scale_vec

    if source == "collision":
        results = _extract_collision_meshes(stage, mesh_scale, articulated, root_prim)
        # Also check physics layer if main stage yielded nothing
        if not results and physics_stage is not None:
            results = _extract_collision_meshes(physics_stage, mesh_scale, articulated, root_prim)
        # Fallback to visual if no collision meshes found
        if not results:
            warnings.warn(f"No collision meshes in {usd_path}, falling back to visual meshes")
            results = _extract_visual_meshes(stage, mesh_scale, articulated, root_prim)
    else:
        results = _extract_visual_meshes(stage, mesh_scale, articulated, root_prim)

    _mesh_cache[cache_key] = results
    return results


def _extract_collision_meshes(
    stage: "Usd.Stage",
    scale: np.ndarray,
    articulated: bool = False,
    root_prim: "Usd.Prim | None" = None,
) -> list[LinkMesh]:
    """Extract meshes with CollisionAPI or under /colliders/|/collisions/ paths."""
    results: list[LinkMesh] = []
    seen_paths: set[str] = set()

    # Strategy 1: Meshes under /colliders/ with CollisionAPI or MeshCollisionAPI
    # Use TraverseInstanceProxies to also visit meshes inside USD instances.
    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        path_str = prim.GetPath().pathString

        # Only consider Mesh prims
        if not prim.IsA(UsdGeom.Mesh):
            continue

        # Must be under /colliders/ or /collisions/ or have collision API
        in_colliders = "/colliders/" in path_str or "/collisions/" in path_str
        has_collision_api = prim.HasAPI(UsdPhysics.CollisionAPI) or prim.HasAPI(UsdPhysics.MeshCollisionAPI)

        if not (in_colliders or has_collision_api):
            continue

        if path_str in seen_paths:
            continue
        seen_paths.add(path_str)

        link_prim = _find_link_ancestor(prim, root_prim) if articulated and root_prim else None
        mesh_result = _read_mesh_prim_with_transform(prim, scale, link_prim=link_prim)
        if mesh_result is None:
            continue

        mesh_trimesh, link_name = mesh_result

        # Check for convex hull approximation
        mesh_api = UsdPhysics.MeshCollisionAPI(prim)
        if mesh_api:
            approx_attr = mesh_api.GetApproximationAttr()
            if approx_attr.HasValue():
                approx = approx_attr.Get()
                if approx == "convexHull":
                    try:
                        mesh_trimesh = mesh_trimesh.convex_hull
                    except Exception:
                        pass

        results.append(LinkMesh(
            mesh=mesh_trimesh,
            link_name=link_name,
            is_collision=True,
        ))

    return results


def _extract_visual_meshes(
    stage: "Usd.Stage",
    scale: np.ndarray,
    articulated: bool = False,
    root_prim: "Usd.Prim | None" = None,
) -> list[LinkMesh]:
    """Extract visual (renderable) meshes as fallback."""
    results: list[LinkMesh] = []
    seen_paths: set[str] = set()

    # Use TraverseInstanceProxies to also visit meshes inside USD instances.
    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue

        path_str = prim.GetPath().pathString
        # Skip collision meshes
        if "/colliders/" in path_str or "/collisions/" in path_str:
            continue
        if prim.HasAPI(UsdPhysics.CollisionAPI) or prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            continue

        if path_str in seen_paths:
            continue
        seen_paths.add(path_str)

        link_prim = _find_link_ancestor(prim, root_prim) if articulated and root_prim else None
        mesh_result = _read_mesh_prim_with_transform(prim, scale, link_prim=link_prim)
        if mesh_result is None:
            continue

        mesh_trimesh, link_name = mesh_result

        # Visual meshes may not be watertight — repair
        if not mesh_trimesh.is_watertight:
            try:
                mesh_trimesh.fill_holes()
            except Exception:
                pass
            if not mesh_trimesh.is_watertight:
                try:
                    mesh_trimesh = mesh_trimesh.convex_hull
                except Exception:
                    pass

        results.append(LinkMesh(
            mesh=mesh_trimesh,
            link_name=link_name,
            is_collision=False,
        ))

    return results


def _read_mesh_prim_with_transform(
    prim: "Usd.Prim",
    scale: np.ndarray,
    link_prim: "Usd.Prim | None" = None,
) -> tuple["trimesh.Trimesh", str] | None:
    """Read a mesh prim with transform applied to vertices.

    When *link_prim* is given (articulated bodies), vertices are stored in
    link-local coordinates.  Otherwise full local-to-world is baked.

    USD matrices use row-vector convention: ``v' = v @ M``.
    """
    mesh_prim = UsdGeom.Mesh(prim)
    mesh_trimesh = _read_usd_mesh_prim(mesh_prim)
    if mesh_trimesh is None:
        return None

    # Apply transform (bake into vertices)
    xformable = UsdGeom.Xformable(prim)
    if xformable:
        try:
            T_mesh = _gf_matrix_to_numpy(
                xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            )

            if link_prim is not None:
                T_link = _gf_matrix_to_numpy(
                    UsdGeom.Xformable(link_prim).ComputeLocalToWorldTransform(
                        Usd.TimeCode.Default()
                    )
                )
                # Row-vector: v_link = v_raw @ T_mesh @ inv(T_link)
                T_apply = T_mesh @ np.linalg.inv(T_link)
            else:
                T_apply = T_mesh

            # Row-vector convention: v' = [v 1] @ M
            verts = mesh_trimesh.vertices
            ones = np.ones((verts.shape[0], 1), dtype=np.float64)
            verts_h = np.hstack([verts, ones])
            mesh_trimesh.vertices = (verts_h @ T_apply)[:, :3]
        except Exception:
            pass

    # Apply scale
    if not np.allclose(scale, 1.0):
        mesh_trimesh.vertices = mesh_trimesh.vertices * scale

    link_name = _extract_link_name(prim.GetPath().pathString)
    return mesh_trimesh, link_name


# ---------------------------------------------------------------------------
# B. Articulation FK solver
# ---------------------------------------------------------------------------

@dataclass
class JointDef:
    """Definition of a single joint in an articulation."""
    name: str
    joint_type: str  # "revolute" | "prismatic" | "fixed"
    parent_link: str
    child_link: str
    axis: np.ndarray  # [3], in joint frame
    parent_to_joint: np.ndarray  # [4,4]
    child_to_joint: np.ndarray  # [4,4]


class ArticulationFK:
    """Forward-kinematics solver for articulated bodies from USD joint definitions."""

    def __init__(
        self,
        usd_path: str,
        scale: float | tuple[float, float, float] | np.ndarray = 1.0,
    ) -> None:
        if Usd is None:
            raise ImportError("pxr (usd-core) is required for FK: pip install usd-core>=26.0")

        self.usd_path = usd_path
        self.scale = _scale_vector(scale)
        self.joints: list[JointDef] = []
        self.base_link: str = ""
        self._link_children: dict[str, list[str]] = {}
        self._link_to_joint: dict[str, JointDef] = {}  # child_link -> joint

        self._parse_joints_from_usd(usd_path)

    def _parse_joints_from_usd(self, usd_path: str) -> None:
        """Parse joint definitions from a USD articulation file."""
        if not os.path.isfile(usd_path):
            warnings.warn(f"USD file not found for FK: {usd_path}")
            return

        stage = Usd.Stage.Open(usd_path, Usd.Stage.LoadAll)
        if stage is None:
            warnings.warn(f"Failed to open USD for FK: {usd_path}")
            return

        # Also try physics layer
        physics_usd = Path(usd_path).parent / "configuration" / f"{Path(usd_path).stem}_physics.usd"
        if physics_usd.exists():
            phys_stage = Usd.Stage.Open(str(physics_usd), Usd.Stage.LoadAll)
        else:
            phys_stage = None

        all_links: set[str] = set()

        for prim in stage.Traverse():
            if prim.IsInstanceProxy():
                continue

            joint: "UsdPhysics.Joint | None" = None
            joint_type = None

            # RevoluteJoint / PrismaticJoint / FixedJoint are typed schemas,
            # not API schemas — use IsA(), not HasAPI().
            if prim.IsA(UsdPhysics.RevoluteJoint):
                joint_type = "revolute"
            elif prim.IsA(UsdPhysics.PrismaticJoint):
                joint_type = "prismatic"
            elif prim.IsA(UsdPhysics.FixedJoint):
                joint_type = "fixed"
            else:
                continue

            # Get body references
            body0_targets = []
            body1_targets = []
            local_pos0 = Gf.Vec3f(0, 0, 0)
            local_rot0 = Gf.Quatf(1, 0, 0, 0)
            local_pos1 = Gf.Vec3f(0, 0, 0)
            local_rot1 = Gf.Quatf(1, 0, 0, 0)
            axis = np.array([0.0, 0.0, 1.0])  # default Z axis

            for prop in prim.GetProperties():
                name = prop.GetName()
                try:
                    if name == "physics:body0" and hasattr(prop, "GetTargets"):
                        targets = prop.GetTargets()
                        if targets:
                            body0_targets = [str(t) for t in targets]
                    elif name == "physics:body1" and hasattr(prop, "GetTargets"):
                        targets = prop.GetTargets()
                        if targets:
                            body1_targets = [str(t) for t in targets]
                    elif name == "physics:localPos0":
                        val = prop.Get()
                        if val is not None:
                            local_pos0 = val
                    elif name == "physics:localRot0":
                        val = prop.Get()
                        if val is not None:
                            local_rot0 = val
                    elif name == "physics:localPos1":
                        val = prop.Get()
                        if val is not None:
                            local_pos1 = val
                    elif name == "physics:localRot1":
                        val = prop.Get()
                        if val is not None:
                            local_rot1 = val
                    elif name == "physics:axis":
                        val = prop.Get()
                        if val is not None:
                            axis_str = str(val)
                            axis_map = {"X": [1, 0, 0], "Y": [0, 1, 0], "Z": [0, 0, 1]}
                            if axis_str in axis_map:
                                axis = np.array(axis_map[axis_str], dtype=np.float64)
                except Exception:
                    continue

            parent_link = _extract_link_name(body0_targets[0]) if body0_targets else ""
            child_link = _extract_link_name(body1_targets[0]) if body1_targets else ""

            if not parent_link or not child_link:
                continue

            # Build parent_to_joint and child_to_joint 4x4 transforms.
            # Asset scale is applied once at the articulation root transform;
            # scaling joint-local offsets here would apply non-uniform scale in
            # the wrong coordinate frame for rotated links.
            parent_to_joint = _make_transform(local_pos0, local_rot0)
            child_to_joint = _make_transform(local_pos1, local_rot1)

            joint_name = prim.GetName()
            joint_def = JointDef(
                name=joint_name,
                joint_type=joint_type,
                parent_link=parent_link,
                child_link=child_link,
                axis=axis,
                parent_to_joint=parent_to_joint,
                child_to_joint=child_to_joint,
            )
            self.joints.append(joint_def)
            all_links.add(parent_link)
            all_links.add(child_link)

        # Determine base link: the link that is a parent but never a child
        child_links = {j.child_link for j in self.joints}
        parent_links = {j.parent_link for j in self.joints}
        base_candidates = parent_links - child_links
        self.base_link = next(iter(base_candidates)) if base_candidates else (
            next(iter(all_links)) if all_links else ""
        )

        # Build adjacency
        for j in self.joints:
            self._link_children.setdefault(j.parent_link, []).append(j.child_link)
            self._link_to_joint[j.child_link] = j

    def compute_link_transforms(
        self,
        root_pose_xyzw: np.ndarray,
        qpos: np.ndarray,
        hdf5_joint_names: list[str],
    ) -> dict[str, np.ndarray]:
        """Compute world-frame 4x4 transforms for all links via FK.

        Args:
            root_pose_xyzw: [x, y, z, qx, qy, qz, qw] root pose.
            qpos: Joint positions (radians for revolute, meters for prismatic).
            hdf5_joint_names: Joint names from HDF5, used to map qpos indices.

        Returns:
            Dict mapping link_name -> 4x4 world transform.
        """
        # Root transform. The 3x3 block may include non-uniform scale.
        T_root = _root_transform_from_pose(root_pose_xyzw, self.scale)

        link_transforms: dict[str, np.ndarray] = {self.base_link: T_root}

        # Build joint name -> qpos index mapping
        joint_name_to_idx = _build_joint_name_map(
            [j.name for j in self.joints],
            hdf5_joint_names,
        )

        # Warn on unmatched joints (will use q=0)
        unmatched = set(j.name for j in self.joints) - set(joint_name_to_idx.keys())
        if unmatched:
            warnings.warn(
                f"ArticulationFK: unmatched USD joints (will use q=0): {sorted(unmatched)}"
            )

        # Warn on qpos length mismatch
        if joint_name_to_idx:
            max_idx = max(joint_name_to_idx.values())
            if max_idx >= len(qpos):
                warnings.warn(
                    f"ArticulationFK: qpos length ({len(qpos)}) < max joint index "
                    f"({max_idx}); missing joints zero-padded"
                )

        # BFS from base link
        queue: deque[str] = deque([self.base_link])
        while queue:
            parent = queue.popleft()
            if parent not in self._link_children:
                continue
            for child in self._link_children[parent]:
                joint = self._link_to_joint[child]
                T_parent = link_transforms[parent]

                # Get joint value
                q_val = 0.0
                if joint.name in joint_name_to_idx:
                    idx = joint_name_to_idx[joint.name]
                    if idx < len(qpos):
                        q_val = float(qpos[idx])

                # Compute joint transform
                T_joint = _joint_transform(joint, q_val)

                # FK: T_child = T_parent @ parent_to_joint @ T_joint @ inv(child_to_joint)
                T_child = T_parent @ joint.parent_to_joint @ T_joint @ np.linalg.inv(joint.child_to_joint)
                link_transforms[child] = T_child.astype(np.float32)
                queue.append(child)

        return link_transforms


def _make_transform(pos: "Gf.Vec3f", rot: "Gf.Quatf") -> np.ndarray:
    """Build 4x4 transform from position and quaternion."""
    T = np.eye(4, dtype=np.float64)
    q_wxyz = _gf_quatf_to_wxyz(rot)
    R = quat_wxyz_to_rot(q_wxyz.astype(np.float32))
    T[:3, :3] = R
    T[:3, 3] = [pos[0], pos[1], pos[2]]
    return T


def _joint_transform(joint: JointDef, q_val: float) -> np.ndarray:
    """Compute joint-space 4x4 transform for a given joint value."""
    T = np.eye(4, dtype=np.float64)
    if joint.joint_type == "revolute":
        angle = q_val
        c, s = np.cos(angle), np.sin(angle)
        ax = joint.axis / (np.linalg.norm(joint.axis) + 1e-12)
        # Rodrigues' rotation formula
        K = np.array([
            [0, -ax[2], ax[1]],
            [ax[2], 0, -ax[0]],
            [-ax[1], ax[0], 0],
        ])
        R = np.eye(3) + s * K + (1 - c) * (K @ K)
        T[:3, :3] = R
    elif joint.joint_type == "prismatic":
        ax = joint.axis / (np.linalg.norm(joint.axis) + 1e-12)
        T[:3, 3] = ax * q_val
    # fixed joints: identity
    return T


def _build_joint_name_map(
    usd_joint_names: list[str],
    hdf5_joint_names: list[str],
) -> dict[str, int]:
    """Map USD joint names -> HDF5 qpos index.

    Uses suffix matching: strip common prefixes and match by tail.
    """
    result: dict[str, int] = {}

    # Direct match first
    hdf5_lookup = {name: idx for idx, name in enumerate(hdf5_joint_names)}
    for uname in usd_joint_names:
        if uname in hdf5_lookup:
            result[uname] = hdf5_lookup[uname]

    if len(result) == len(usd_joint_names):
        return result

    # Suffix matching: try stripping prefixes like "joint_", "robot_"
    def _strip_prefixes(name: str) -> str:
        for prefix in ("joint_", "robot_", "base_to_", "fixed_"):
            if name.startswith(prefix):
                return name[len(prefix):]
        return name

    for uname in usd_joint_names:
        if uname in result:
            continue
        uname_stripped = _strip_prefixes(uname)
        for hname, hidx in hdf5_lookup.items():
            hname_stripped = _strip_prefixes(hname)
            if uname_stripped == hname_stripped or uname == hname_stripped or uname_stripped == hname:
                result[uname] = hidx
                break

    return result


# ---------------------------------------------------------------------------
# C. Scene voxelization
# ---------------------------------------------------------------------------

def _voxel_centers(bounds: np.ndarray, voxel_size: float) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Compute world-coordinate centers of all voxels in the grid.

    Returns:
        centers: float32 (N, 3) voxel center positions.
        grid_shape: (nx, ny, nz) grid dimensions.
    """
    nx = int(np.round((bounds[1, 0] - bounds[0, 0]) / voxel_size))
    ny = int(np.round((bounds[1, 1] - bounds[0, 1]) / voxel_size))
    nz = int(np.round((bounds[1, 2] - bounds[0, 2]) / voxel_size))

    xs = bounds[0, 0] + (np.arange(nx) + 0.5) * voxel_size
    ys = bounds[0, 1] + (np.arange(ny) + 0.5) * voxel_size
    zs = bounds[0, 2] + (np.arange(nz) + 0.5) * voxel_size

    # Create grid of centers
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    centers = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1).astype(np.float32)

    return centers, (nx, ny, nz)


def voxelize_scene_frame(
    object_states: dict[str, dict],
    asset_meshes: dict[str, list[LinkMesh]],
    fk_solvers: dict[str, ArticulationFK],
    table_aabb: np.ndarray | None,
    bounds: np.ndarray,
    voxel_size: float,
    semantic_map: dict[str, int] | None = None,
    local_bboxes: dict[str, tuple] | None = None,
) -> dict:
    """Voxelize one frame of the scene from GT mesh geometry.

    Args:
        object_states: Per-object state dict from read_object_states().
            Each value has "pose_world" and optionally "qpos"/"joint_names".
        asset_meshes: {obj_id: list[LinkMesh]} — cached meshes per asset.
        fk_solvers: {obj_id: ArticulationFK} — only for articulated bodies.
        table_aabb: [2, 3] AABB of table solid block, or None to skip.
        bounds: [2, 3] voxelization bounds.
        voxel_size: Voxel edge length in meters.
        semantic_map: {obj_id: semantic_id}. If None, no semantic output.
        local_bboxes: Optional episode-local object AABBs used to clip mesh
            occupancy to the object geometry bounds recorded during collection.

    Returns:
        Dict with keys: state, semantic_id (optional), grid_shape, bounds, voxel_size.
    """
    centers, grid_shape = _voxel_centers(bounds, voxel_size)
    nx, ny, nz = grid_shape
    n_voxels = nx * ny * nz

    # Initialize all as FREE (GT knows the full scene)
    state = np.full(n_voxels, _FREE, dtype=np.uint8)
    semantic = np.zeros(n_voxels, dtype=np.uint16) if semantic_map is not None else None

    # 1. Table — only the top surface layer (1 voxel thick).
    #    The full solid block (z=0..table_height) wastes storage and skews
    #    metrics; downstream tasks only care about the surface.
    if table_aabb is not None:
        surface_aabb = table_aabb.copy()
        surface_aabb[0, 2] = surface_aabb[1, 2] - voxel_size
        inside_table = np.all(
            (centers >= surface_aabb[0]) & (centers <= surface_aabb[1]),
            axis=1,
        )
        state[inside_table] = _OCCUPIED
        if semantic is not None:
            semantic[inside_table] = 1  # table always gets semantic_id=1

    # 2. Objects
    for obj_id, obj_state in object_states.items():
        meshes = asset_meshes.get(obj_id)
        if not meshes:
            continue

        pose = obj_state["pose_world"]
        if not np.all(np.isfinite(pose)):
            continue

        # Determine if articulated
        is_articulated = obj_id in fk_solvers

        if is_articulated:
            _voxelize_articulated(
                obj_id, obj_state, meshes, fk_solvers[obj_id],
                centers, state, semantic, semantic_map, voxel_size,
            )
        else:
            clip_aabb = _object_world_aabb_from_local_bbox(
                obj_state,
                local_bboxes.get(obj_id) if local_bboxes else None,
                margin=voxel_size * 0.5,
            )
            _voxelize_rigid(
                obj_id, obj_state, meshes,
                centers, state, semantic, semantic_map, voxel_size, clip_aabb,
            )

    # Reshape to 3D grid
    state_grid = state.reshape(nx, ny, nz)
    result: dict[str, Any] = {
        "state": state_grid,
        "grid_shape": np.array(grid_shape, dtype=np.int32),
        "bounds": bounds,
        "voxel_size": voxel_size,
    }
    if semantic is not None:
        result["semantic_id"] = semantic.reshape(nx, ny, nz)
    return result


def _object_world_aabb_from_local_bbox(
    obj_state: dict,
    local_bbox: tuple | None,
    *,
    margin: float = 0.0,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Transform an episode local bbox to a world-frame AABB."""
    if local_bbox is None:
        return None
    lo, hi = local_bbox
    lo_arr = np.asarray(lo, dtype=np.float32)
    hi_arr = np.asarray(hi, dtype=np.float32)
    if lo_arr.shape != (3,) or hi_arr.shape != (3,):
        return None
    pose = obj_state["pose_world"]
    rot = quat_xyzw_to_rot(pose[3:7])
    center_local = 0.5 * (lo_arr + hi_arr)
    size = hi_arr - lo_arr
    corners_local = np.asarray([
        [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
        [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1],
    ], dtype=np.float32) * (0.5 * size) + center_local
    corners_world = (corners_local @ rot.T) + pose[:3]
    return (
        corners_world.min(axis=0).astype(np.float32) - margin,
        corners_world.max(axis=0).astype(np.float32) + margin,
    )


def _voxelize_rigid(
    obj_id: str,
    obj_state: dict,
    meshes: list[LinkMesh],
    centers: np.ndarray,
    state: np.ndarray,
    semantic: np.ndarray | None,
    semantic_map: dict[str, int] | None,
    voxel_size: float = 0.01,
    clip_aabb: tuple[np.ndarray, np.ndarray] | None = None,
) -> None:
    """Voxelize a rigid body: apply pose_world to all link meshes."""
    pose = obj_state["pose_world"]
    T_world = np.eye(4, dtype=np.float32)
    T_world[:3, :3] = quat_xyzw_to_rot(pose[3:7])
    T_world[:3, 3] = pose[:3]

    sem_id = semantic_map.get(obj_id, 0) if semantic_map else 0

    for lm in meshes:
        _mark_voxels_for_mesh(lm.mesh, T_world, centers, state, semantic, sem_id, voxel_size, clip_aabb)


def _voxelize_articulated(
    obj_id: str,
    obj_state: dict,
    meshes: list[LinkMesh],
    fk: ArticulationFK,
    centers: np.ndarray,
    state: np.ndarray,
    semantic: np.ndarray | None,
    semantic_map: dict[str, int] | None,
    voxel_size: float = 0.01,
) -> None:
    """Voxelize an articulated body: compute FK, then voxelize each link.

    No ``clip_aabb`` is applied here — for articulated bodies the per-link
    world AABB (computed inside ``_mark_voxels_for_mesh`` from the link's
    own transformed vertices) already provides correct, tight pre-filtering.
    A static root-pose or per-link clip_aabb derived from episode-local bboxes
    is both incorrect for links far from root and unnecessary given the
    ``proximity_radius`` of only 0.75×voxel_size.
    """
    pose = obj_state["pose_world"]
    qpos = obj_state.get("qpos", np.array([], dtype=np.float32))
    joint_names = obj_state.get("joint_names", [])

    link_transforms = fk.compute_link_transforms(pose, qpos, joint_names)

    sem_id = semantic_map.get(obj_id, 0) if semantic_map else 0

    # Build link_name -> mesh lookup
    meshes_by_link: dict[str, list[trimesh.Trimesh]] = {}
    for lm in meshes:
        meshes_by_link.setdefault(lm.link_name, []).append(lm.mesh)

    # Per-link voxelization — no clip_aabb needed; the mesh's own world AABB
    # (computed inside _mark_voxels_for_mesh) is the correct tight pre-filter.
    for link_name, T_world in link_transforms.items():
        link_meshes = meshes_by_link.get(link_name, [])
        for mesh in link_meshes:
            _mark_voxels_for_mesh(mesh, T_world, centers, state, semantic, sem_id, voxel_size, clip_aabb=None)

    # Also process meshes whose link_name doesn't match any FK link
    # (e.g. base_link meshes that might use a different naming convention)
    fk_links = set(link_transforms.keys())
    for link_name, link_mesh_list in meshes_by_link.items():
        if link_name in fk_links:
            continue
        # Use scaled root transform as fallback.
        T_root = _root_transform_from_pose(pose, fk.scale)
        for mesh in link_mesh_list:
            _mark_voxels_for_mesh(mesh, T_root, centers, state, semantic, sem_id, voxel_size, clip_aabb=None)


def _build_surface_kdtree(
    mesh: "trimesh.Trimesh",
    voxel_size: float,
) -> "cKDTree":
    """Build a KD-tree from dense surface samples for fast proximity queries.

    Sampling density ensures no surface point is farther than
    ``voxel_size / 4`` from the nearest sample.
    """
    from scipy.spatial import cKDTree

    target_spacing = voxel_size * 0.25
    n_samples = max(int(mesh.area / (target_spacing ** 2)), mesh.vertices.shape[0])
    # Cap at reasonable maximum to avoid OOM on very large meshes
    n_samples = min(n_samples, 500_000)
    try:
        samples, _ = trimesh.sample.sample_surface(mesh, n_samples, seed=42)
    except Exception:
        samples = mesh.vertices
    return cKDTree(samples)


# Cache: mesh id -> (KD-tree built in local frame)
_kdtree_cache: dict[int, "cKDTree"] = {}
_surface_samples_cache: dict[int, np.ndarray] = {}
# Cache: mesh id -> bool (whether mesh is solid/watertight enough for volumetric containment)
_mesh_classification_cache: dict[int, bool] = {}


def _classify_mesh(mesh: "trimesh.Trimesh") -> bool:
    """Classify a mesh as solid (volumetric containment) or surface (KD-tree proximity).

    Classification uses the ORIGINAL local-frame mesh, so it is frame-invariant
    and can be safely cached.  Mirrors ``_uses_surface_proximity`` in the GPU path.

    Returns True for solid meshes, False for surface-proximity meshes.
    """
    mesh_key = id(mesh)
    if mesh_key in _mesh_classification_cache:
        return _mesh_classification_cache[mesh_key]

    is_solid = mesh.is_watertight
    if is_solid:
        lo = mesh.vertices.min(axis=0)
        hi = mesh.vertices.max(axis=0)
        aabb_vol = float(np.prod(hi - lo + 1e-12))
        if aabb_vol > 0 and mesh.volume / aabb_vol < 0.05:
            is_solid = False  # thin shell — treat as surface

    _mesh_classification_cache[mesh_key] = is_solid
    return is_solid


def clear_voxelizer_caches() -> None:
    """Clear all module-level caches for meshes, KD-trees, and surface samples.

    Call between episodes to prevent stale cached data from leaking across
    episodes with different objects / meshes.
    """
    _kdtree_cache.clear()
    _surface_samples_cache.clear()
    _mesh_classification_cache.clear()


def _is_rotation_matrix_like(linear: np.ndarray) -> bool:
    """True when local Euclidean distances are preserved by ``linear``."""
    gram = linear.T @ linear
    return bool(np.allclose(gram, np.eye(3, dtype=gram.dtype), atol=1e-5, rtol=1e-5))


def _surface_samples_for_mesh(mesh: "trimesh.Trimesh", voxel_size: float) -> np.ndarray:
    mesh_key = id(mesh)
    samples = _surface_samples_cache.get(mesh_key)
    if samples is not None:
        return samples
    target_spacing = voxel_size * 0.25
    n_samples = max(int(mesh.area / (target_spacing ** 2)), mesh.vertices.shape[0])
    n_samples = min(n_samples, 500_000)
    try:
        samples, _ = trimesh.sample.sample_surface(mesh, n_samples, seed=42)
    except Exception:
        samples = mesh.vertices
    samples = np.asarray(samples, dtype=np.float32)
    _surface_samples_cache[mesh_key] = samples
    return samples


def _mark_voxels_for_mesh(
    mesh: "trimesh.Trimesh",
    T_world: np.ndarray,
    centers: np.ndarray,
    state: np.ndarray,
    semantic: np.ndarray | None,
    semantic_id: int,
    voxel_size: float = 0.01,
    clip_aabb: tuple[np.ndarray, np.ndarray] | None = None,
) -> None:
    """Transform mesh to world frame and mark occupied voxels.

    Strategy:
      - Watertight solid (volume/AABB ≥ 5%): volumetric ``contains()``
      - Otherwise (non-watertight shells, thin shells): surface-proximity
        via KD-tree on dense surface samples
    """
    from scipy.spatial import cKDTree

    # Skip degenerate meshes (< 4 faces or zero-thickness)
    if len(mesh.faces) < 4:
        return
    aabb_size = mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0)
    if aabb_size.min() < 1e-6:
        return

    # Transform mesh vertices to world frame
    verts = mesh.vertices.astype(np.float32)
    linear = T_world[:3, :3].astype(np.float32)
    t = T_world[:3, 3].astype(np.float32)
    verts_world = (verts @ linear.T) + t

    # Decide strategy: volumetric fill vs surface proximity.
    # Classification is cached per-mesh on the original (local-frame) mesh
    # so it is frame-invariant — same mesh never flips between solid/surface.
    is_solid = _classify_mesh(mesh)

    # Still need a world_mesh for the solid containment path
    world_mesh = None
    if is_solid:
        world_mesh = trimesh.Trimesh(vertices=verts_world, faces=mesh.faces, process=False)

    # Quick AABB pre-filter (expand by proximity radius for surface mode)
    proximity_radius = voxel_size * 0.75
    margin = 0.0 if is_solid else proximity_radius
    mesh_min = verts_world.min(axis=0) - margin
    mesh_max = verts_world.max(axis=0) + margin
    in_aabb = np.all((centers >= mesh_min) & (centers <= mesh_max), axis=1)
    if clip_aabb is not None:
        clip_min, clip_max = clip_aabb
        in_aabb &= np.all((centers >= clip_min) & (centers <= clip_max), axis=1)

    if not np.any(in_aabb):
        return

    candidate_centers = centers[in_aabb]

    if is_solid:
        # Volumetric containment for true solids
        try:
            inside = world_mesh.contains(candidate_centers)
        except Exception:
            inside = _fallback_contains(world_mesh, candidate_centers)
    else:
        # Surface-proximity via KD-tree (fast path)
        # Reuse local-frame KD-tree across frames for the same mesh
        mesh_key = id(mesh)
        if _is_rotation_matrix_like(linear) and mesh_key in _kdtree_cache:
            tree_local = _kdtree_cache[mesh_key]
            # For pure rotation, local and world distances are identical.
            candidates_local = (candidate_centers - t) @ linear
            dists, _ = tree_local.query(candidates_local)
        elif _is_rotation_matrix_like(linear):
            # Build tree in local frame and cache
            tree_local = _build_surface_kdtree(
                trimesh.Trimesh(vertices=verts, faces=mesh.faces, process=False),
                voxel_size,
            )
            _kdtree_cache[mesh_key] = tree_local
            candidates_local = (candidate_centers - t) @ linear
            dists, _ = tree_local.query(candidates_local)
        else:
            # Non-uniform scale changes distances; query in world coordinates.
            samples_local = _surface_samples_for_mesh(
                trimesh.Trimesh(vertices=verts, faces=mesh.faces, process=False),
                voxel_size,
            )
            samples_world = (samples_local @ linear.T) + t
            tree_world = cKDTree(samples_world)
            dists, _ = tree_world.query(candidate_centers)
        inside = dists <= proximity_radius

    # Map back to flat indices
    aabb_indices = np.where(in_aabb)[0]
    occupied_indices = aabb_indices[inside]
    state[occupied_indices] = _OCCUPIED
    if semantic is not None and semantic_id > 0:
        semantic[occupied_indices] = semantic_id


def _fallback_contains(
    mesh: "trimesh.Trimesh",
    points: np.ndarray,
) -> np.ndarray:
    """Fallback containment test using ray casting (odd intersection count)."""
    try:
        ray_origins = points
        ray_directions = np.tile([1.0, 0.0, 0.0], (len(points), 1)).astype(np.float32)
        # intersects_location returns (locations, ray_indices, face_indices)
        locs, ray_idxs, _ = mesh.ray.intersects_location(ray_origins, ray_directions)
        counts = np.zeros(len(points), dtype=np.int32)
        for ri in ray_idxs:
            counts[ri] += 1
        return (counts % 2) == 1
    except Exception:
        return np.zeros(len(points), dtype=bool)


# ---------------------------------------------------------------------------
# Scene YAML parsing helpers
# ---------------------------------------------------------------------------

def parse_scene_table_spec(scene_path: str) -> tuple[tuple[float, float, float], float]:
    """Parse table size and height from a scene YAML file.

    Returns:
        (table_size, table_height) where table_size = (sx, sy, thickness_from_yaml).
    """
    import yaml
    with open(scene_path, "r", encoding="utf-8") as f:
        task = yaml.safe_load(f)

    table = task.get("table", {}) or {}
    size = tuple(float(v) for v in table.get("size", [2.2, 1.1, 0.04]))
    height = float(table.get("height", 0.75))
    return size, height


def compute_table_aabb(
    table_size: tuple[float, float, float],
    actual_z: float,
    thickness: float | None = None,
) -> np.ndarray:
    """Compute the world-frame AABB for the tabletop slab only.

    Only the top slab (the surface the arm can collide with / objects rest on)
    is relevant for occupancy — the legs and bulk below the table are not.
    The slab spans z in [actual_z - thickness, actual_z], where ``actual_z`` is
    the per-episode tabletop height (nominal height + scene-generalization
    offset) and ``thickness`` defaults to ``table_size[2]`` (e.g. 0.04m).
    """
    sx, sy = table_size[0], table_size[1]
    t = float(thickness if thickness is not None else table_size[2])
    return np.array([
        [-sx / 2, -sy / 2, actual_z - t],
        [sx / 2, sy / 2, actual_z],
    ], dtype=np.float32)


def build_semantic_map(
    object_ids: list[str],
    object_body_types: dict[str, str],
) -> tuple[dict[str, int], dict[int, str]]:
    """Build bidirectional semantic ID mapping.

    IDs: 0=free, 1=table, 2+=objects (sorted by obj_id for determinism).

    Returns:
        obj_to_id: {obj_id: semantic_id}
        id_to_label: {semantic_id: label_string}
    """
    obj_to_id: dict[str, int] = {}
    id_to_label: dict[int, str] = {0: "free", 1: "table"}
    next_id = 2
    for obj_id in sorted(object_ids):
        obj_to_id[obj_id] = next_id
        id_to_label[next_id] = obj_id
        next_id += 1
    return obj_to_id, id_to_label
