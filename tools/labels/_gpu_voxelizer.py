"""GPU-accelerated mesh containment (Moller-Trumbore) and cached FK for GT occupancy."""

from __future__ import annotations

import warnings
from collections import deque
from typing import TYPE_CHECKING, Any

import numpy as np

from tools.labels._accel import torch_available, to_torch, to_numpy

if TYPE_CHECKING:
    import torch

if torch_available():
    import torch as _torch
else:
    _torch = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# A. GPU Mesh Containment via Moller-Trumbore ray-casting
# ---------------------------------------------------------------------------

class MeshContainmentGPU:
    """Point-in-mesh test using Moller-Trumbore ray-casting on GPU.

    Casts a fixed ray ``D = (1, 0, 0)`` from each query point, counts
    triangle intersections; odd count ⇒ inside.  Edge vectors and
    determinants are **precomputed once** from link-local triangle vertices,
    then reused across all frames by transforming query points to link-local
    coordinates instead.
    """

    def __init__(self, triangles: "torch.Tensor", device: "torch.device") -> None:
        """
        Args:
            triangles: ``(T, 3, 3)`` triangle vertices in link-local frame.
        """
        if _torch is None:
            raise RuntimeError("torch is not installed")

        tris = triangles.to(device=device, dtype=_torch.float32)
        self._device = device
        self._n_faces = tris.shape[0]

        self._V0 = tris[:, 0, :]  # (T, 3)
        E1 = tris[:, 1, :] - self._V0  # (T, 3)
        E2 = tris[:, 2, :] - self._V0  # (T, 3)
        self._E1 = E1
        self._E2 = E2

        # P = cross(D, E2) where D = (1, 0, 0)
        # cross((1,0,0), (ex,ey,ez)) = (0*ez-0*ey, 0*ex-1*ez, 1*ey-0*ex) = (0, -ez, ey)
        self._P = _torch.stack([
            _torch.zeros(self._n_faces, device=device),
            -E2[:, 2],
            E2[:, 1],
        ], dim=-1)  # (T, 3)

        det = (E1 * self._P).sum(dim=-1)  # (T,)
        self._valid_face = det.abs() > 1.0e-10  # (T,)
        # Preserve sign for back-facing triangles; invalid faces get det=1 to avoid div-by-zero
        self._det = _torch.where(self._valid_face, det, _torch.ones_like(det))

        # Pre-compute AABB of the mesh (link-local) for fast rejection
        all_verts = tris.reshape(-1, 3)
        self._aabb_min = all_verts.min(dim=0).values  # (3,)
        self._aabb_max = all_verts.max(dim=0).values  # (3,)

    @classmethod
    def from_trimesh(cls, mesh: Any, device: "torch.device") -> "MeshContainmentGPU":
        """Build from a ``trimesh.Trimesh`` object."""
        verts = _torch.from_numpy(mesh.vertices.astype(np.float32))
        faces = _torch.from_numpy(mesh.faces.astype(np.int64))
        triangles = verts[faces]  # (T, 3, 3)
        return cls(triangles, device)

    def contains(self, points: "torch.Tensor", chunk_size: int = 2048) -> "torch.Tensor":
        """Test which points are inside the mesh.

        Args:
            points: ``(N, 3)`` query points in the **same coordinate frame**
                as the mesh triangles (link-local).
            chunk_size: Process this many points at a time to limit GPU memory.

        Returns:
            ``(N,)`` bool tensor — True for points inside the mesh.
        """
        N = points.shape[0]
        if N == 0:
            return _torch.zeros(0, dtype=_torch.bool, device=self._device)

        # Adaptive chunk size based on face count (target ~1 GB peak)
        max_elements = int(1e9 / (self._n_faces * 48))  # ~48 bytes per element
        effective_chunk = min(chunk_size, max(256, max_elements))

        result = _torch.zeros(N, dtype=_torch.bool, device=self._device)
        for start in range(0, N, effective_chunk):
            end = min(N, start + effective_chunk)
            result[start:end] = self._contains_chunk(points[start:end])
        return result

    def _contains_chunk(self, pts: "torch.Tensor") -> "torch.Tensor":
        """Moller-Trumbore for a chunk of points. D = (1, 0, 0)."""
        C = pts.shape[0]
        T = self._n_faces

        # T_vec = O - V0: (C, T, 3)
        T_vec = pts[:, None, :] - self._V0[None, :, :]  # (C, T, 3)

        # u = dot(T_vec, P) / det
        # dot(T_vec, P) for each (c, t): sum over j
        u_num = (T_vec * self._P[None, :, :]).sum(dim=-1)  # (C, T)
        inv_det = 1.0 / self._det  # (T,) -- safe: invalid faces have det=1
        u = u_num * inv_det[None, :]  # (C, T)

        # Q = cross(T_vec, E1)
        Q = _torch.cross(T_vec, self._E1[None, :, :].expand(C, -1, -1), dim=-1)  # (C, T, 3)

        # v = dot(D, Q) / det  →  D = (1,0,0)  →  v = Q[:,:,0] / det
        v = Q[:, :, 0] * inv_det[None, :]  # (C, T)

        # t_param = dot(E2, Q) / det
        t_param = (self._E2[None, :, :] * Q).sum(dim=-1) * inv_det[None, :]  # (C, T)

        # Valid hit: u >= 0, v >= 0, u+v <= 1, t > 0, face non-degenerate
        hit = (
            (u >= 0)
            & (v >= 0)
            & ((u + v) <= 1.0)
            & (t_param > 1.0e-8)
            & self._valid_face[None, :]
        )  # (C, T)

        # Odd intersection count → inside
        return (hit.sum(dim=1) % 2) == 1  # (C,)

    @property
    def aabb_min(self) -> "torch.Tensor":
        return self._aabb_min

    @property
    def aabb_max(self) -> "torch.Tensor":
        return self._aabb_max


class MeshSurfaceProximityGPU:
    """Surface-proximity occupancy for non-watertight or thin-shell meshes.

    The CPU path uses a KD-tree over dense surface samples for these meshes.
    This class keeps the same rule but evaluates nearest-surface distance on
    the selected torch device.
    """

    def __init__(
        self,
        samples: "torch.Tensor",
        radius: float,
        device: "torch.device",
    ) -> None:
        if _torch is None:
            raise RuntimeError("torch is not installed")
        self._samples = samples.to(device=device, dtype=_torch.float32)
        self._radius = float(radius)
        self._device = device

    @classmethod
    def from_trimesh(
        cls,
        mesh: Any,
        voxel_size: float,
        device: "torch.device",
    ) -> "MeshSurfaceProximityGPU":
        """Build dense local-frame surface samples for proximity queries."""
        import trimesh

        target_spacing = float(voxel_size) * 0.25
        n_samples = max(int(mesh.area / (target_spacing ** 2)), mesh.vertices.shape[0])
        n_samples = min(n_samples, 250_000)
        try:
            samples, _ = trimesh.sample.sample_surface(mesh, n_samples, seed=42)
        except Exception:
            samples = mesh.vertices
        samples_t = _torch.from_numpy(samples.astype(np.float32, copy=False))
        return cls(samples_t, radius=float(voxel_size) * 0.75, device=device)

    @property
    def radius(self) -> float:
        return self._radius

    def near_surface(
        self,
        points: "torch.Tensor",
        *,
        sample_linear: "torch.Tensor | None" = None,
        sample_translation: "torch.Tensor | None" = None,
        point_chunk_size: int = 1024,
        sample_chunk_size: int = 8192,
    ) -> "torch.Tensor":
        """Return True where points are within ``radius`` of the sampled surface.

        By default ``points`` and samples are compared in mesh-local coordinates.
        When ``sample_linear`` is provided, samples are transformed into world
        coordinates first; this is required for non-uniform scale because local
        Euclidean distances no longer match world distances.
        """
        N = points.shape[0]
        if N == 0:
            return _torch.zeros(0, dtype=_torch.bool, device=self._device)

        result = _torch.zeros(N, dtype=_torch.bool, device=self._device)
        for p0 in range(0, N, point_chunk_size):
            p1 = min(N, p0 + point_chunk_size)
            pts = points[p0:p1]
            near = _torch.zeros(p1 - p0, dtype=_torch.bool, device=self._device)
            for s0 in range(0, self._samples.shape[0], sample_chunk_size):
                if bool(near.all()):
                    break
                s1 = min(self._samples.shape[0], s0 + sample_chunk_size)
                samples = self._samples[s0:s1]
                if sample_linear is not None:
                    samples = samples @ sample_linear.T
                    if sample_translation is not None:
                        samples = samples + sample_translation
                dists = _torch.cdist(pts, samples)
                near |= dists.min(dim=1).values <= self._radius
            result[p0:p1] = near
        return result


def _uses_surface_proximity(mesh: Any) -> bool:
    """Match the CPU shell/solid decision for GPU precomputation."""
    if not mesh.is_watertight:
        return True
    wm_min = mesh.vertices.min(axis=0)
    wm_max = mesh.vertices.max(axis=0)
    aabb_vol = float(np.prod(wm_max - wm_min + 1e-12))
    return bool(aabb_vol > 0 and mesh.volume / aabb_vol < 0.05)


def _is_rotation_matrix_like(linear: np.ndarray) -> bool:
    gram = linear.T @ linear
    return bool(np.allclose(gram, np.eye(3, dtype=gram.dtype), atol=1e-5, rtol=1e-5))


# ---------------------------------------------------------------------------
# B. Cached Forward Kinematics
# ---------------------------------------------------------------------------

class BatchFK:
    """Optimized FK solver that caches constant data across frames.

    Caches ``inv(child_to_joint)`` per joint (constant per asset) and
    the joint name map (constant per episode), avoiding repeated matrix
    inversions and string matching.
    """

    def __init__(self, fk: Any) -> None:
        """
        Args:
            fk: ``ArticulationFK`` instance from ``_mesh_voxelizer.py``.
        """
        from collector.camera_geometry import quat_xyzw_to_rot

        self._fk = fk
        self._base_link = fk.base_link
        self._scale = getattr(fk, "scale", np.ones(3, dtype=np.float32))
        self._quat_xyzw_to_rot = quat_xyzw_to_rot

        # Pre-compute BFS traversal order
        self._bfs_order: list[tuple[str, Any]] = []  # (child_link, JointDef)
        queue: deque[str] = deque([fk.base_link])
        while queue:
            parent = queue.popleft()
            for child in fk._link_children.get(parent, []):
                joint = fk._link_to_joint[child]
                self._bfs_order.append((child, joint))
                queue.append(child)

        # Cache inv(child_to_joint) — constant per asset
        self._inv_child_to_joint: dict[str, np.ndarray] = {}
        for child, joint in self._bfs_order:
            self._inv_child_to_joint[child] = np.linalg.inv(
                joint.child_to_joint
            ).astype(np.float32)

        # Will be cached after first call
        self._joint_name_map: dict[str, int] | None = None
        self._last_joint_names: list[str] | None = None

    def compute_link_transforms(
        self,
        root_pose_xyzw: np.ndarray,
        qpos: np.ndarray,
        hdf5_joint_names: list[str],
    ) -> dict[str, np.ndarray]:
        """Compute world-frame 4x4 transforms for all links via FK."""
        from tools.labels._mesh_voxelizer import _joint_transform, _build_joint_name_map

        # Cache joint name map (constant per episode)
        if self._last_joint_names is not hdf5_joint_names:
            self._joint_name_map = _build_joint_name_map(
                [j.name for _, j in self._bfs_order],
                hdf5_joint_names,
            )
            self._last_joint_names = hdf5_joint_names

        # Root transform. The 3x3 block may include non-uniform asset scale.
        root_rot = self._quat_xyzw_to_rot(root_pose_xyzw[3:7])
        scale_mat = np.diag(np.asarray(self._scale, dtype=np.float32).reshape(3))
        T_root = np.eye(4, dtype=np.float32)
        T_root[:3, :3] = root_rot @ scale_mat
        T_root[:3, 3] = root_pose_xyzw[:3]

        link_transforms: dict[str, np.ndarray] = {self._base_link: T_root}

        for child, joint in self._bfs_order:
            T_parent = link_transforms.get(joint.parent_link)
            if T_parent is None:
                continue

            q_val = 0.0
            if self._joint_name_map and joint.name in self._joint_name_map:
                idx = self._joint_name_map[joint.name]
                if idx < len(qpos):
                    q_val = float(qpos[idx])

            T_joint = _joint_transform(joint, q_val)
            # Use cached inverse
            T_child = T_parent @ joint.parent_to_joint @ T_joint @ self._inv_child_to_joint[child]
            link_transforms[child] = T_child.astype(np.float32)

        return link_transforms


# ---------------------------------------------------------------------------
# C. GPU-accelerated scene frame voxelization
# ---------------------------------------------------------------------------

_UNKNOWN = 0
_FREE = 1
_OCCUPIED = 2


def voxelize_scene_frame_gpu(
    object_states: dict[str, dict],
    asset_meshes: dict[str, list],
    batch_fk_cache: dict[str, BatchFK],
    table_mask_3d: "torch.Tensor | None",
    centers_grid: "torch.Tensor",
    grid_shape: tuple[int, int, int],
    containment_cache: dict[int, MeshContainmentGPU | MeshSurfaceProximityGPU],
    aabb_cache: dict[int, tuple[np.ndarray, np.ndarray] | None],
    bounds: np.ndarray,
    device: "torch.device",
    semantic_map: dict[str, int] | None = None,
    local_bboxes: dict[str, tuple] | None = None,
    voxel_size: float = 0.01,
) -> dict:
    """GPU-resident scene voxelization for one frame.

    The whole frame pipeline stays on device: ``centers_grid`` (a 3D grid of
    world voxel centers) and ``state``/``semantic`` tensors live on the GPU,
    and per-mesh writes use ``masked_fill_`` (element-wise, no data-dependent
    sync).  The only host sync per frame is the final ``.cpu()`` of the state
    grid.  ``table_mask_3d`` is precomputed once per episode because the table
    is static across frames.
    """
    from collector.camera_geometry import quat_xyzw_to_rot

    nx, ny, nz = grid_shape

    state = _torch.full((nx, ny, nz), _FREE, dtype=_torch.uint8, device=device)
    # int32 on device: masked_fill_ / where are not implemented for uint16 on
    # CUDA; cast to uint16 only when materializing the result for storage.
    semantic = (
        _torch.zeros((nx, ny, nz), dtype=_torch.int32, device=device)
        if semantic_map is not None
        else None
    )

    # 1. Table — static mask precomputed once per episode.
    if table_mask_3d is not None:
        state.masked_fill_(table_mask_3d, _OCCUPIED)
        if semantic is not None:
            semantic.masked_fill_(table_mask_3d, 1)

    # 2. Objects
    for obj_id, obj_state in object_states.items():
        meshes = asset_meshes.get(obj_id)
        if not meshes:
            continue

        pose = obj_state["pose_world"]
        if not np.all(np.isfinite(pose)):
            continue

        # Always compute world transform (needed for rigid bodies and FK fallback)
        T_world = np.eye(4, dtype=np.float32)
        T_world[:3, :3] = quat_xyzw_to_rot(pose[3:7])
        T_world[:3, 3] = pose[:3]

        # Determine link transforms
        is_articulated = obj_id in batch_fk_cache
        if is_articulated:
            qpos = obj_state.get("qpos", np.array([], dtype=np.float32))
            joint_names = obj_state.get("joint_names", [])
            link_transforms = batch_fk_cache[obj_id].compute_link_transforms(
                pose, qpos, joint_names,
            )
        else:
            link_transforms = None

        sem_id = semantic_map.get(obj_id, 0) if semantic_map else 0

        # Build link_name -> mesh lookup
        meshes_by_link: dict[str, list] = {}
        for lm in meshes:
            meshes_by_link.setdefault(lm.link_name, []).append(lm)

        if is_articulated and link_transforms is not None:
            for link_name, T_link in link_transforms.items():
                for lm in meshes_by_link.get(link_name, []):
                    _mark_mesh_gpu(
                        lm, T_link, centers_grid, state, semantic, sem_id,
                        containment_cache, aabb_cache, bounds, voxel_size, device,
                    )
            # Fallback for unmatched links
            fk_links = set(link_transforms.keys())
            root_fallback = link_transforms.get(
                getattr(batch_fk_cache[obj_id], "_base_link", ""),
                T_world,
            )
            for link_name, lm_list in meshes_by_link.items():
                if link_name in fk_links:
                    continue
                for lm in lm_list:
                    _mark_mesh_gpu(
                        lm, root_fallback, centers_grid, state, semantic, sem_id,
                        containment_cache, aabb_cache, bounds, voxel_size, device,
                    )
        else:
            clip_aabb = None
            if local_bboxes and obj_id in local_bboxes:
                from tools.labels._mesh_voxelizer import _object_world_aabb_from_local_bbox
                clip_aabb = _object_world_aabb_from_local_bbox(
                    obj_state, local_bboxes[obj_id], margin=float(voxel_size) * 0.5,
                )
            for lm in meshes:
                _mark_mesh_gpu(
                    lm, T_world, centers_grid, state, semantic, sem_id,
                    containment_cache, aabb_cache, bounds, voxel_size, device, clip_aabb,
                )

    state_np = state.detach().cpu().numpy()
    result: dict[str, Any] = {
        "state": state_np,
        "grid_shape": np.array(grid_shape, dtype=np.int32),
    }
    if semantic is not None:
        result["semantic_id"] = semantic.detach().cpu().numpy().astype(np.uint16)
    return result


def _mark_mesh_gpu(
    lm: Any,
    T_world: np.ndarray,
    centers_grid: "torch.Tensor",
    state_grid: "torch.Tensor",
    semantic_grid: "torch.Tensor | None",
    sem_id: int,
    containment_cache: dict[int, MeshContainmentGPU | MeshSurfaceProximityGPU],
    aabb_cache: dict[int, tuple[np.ndarray, np.ndarray] | None],
    bounds: np.ndarray,
    voxel_size: float,
    device: "torch.device",
    clip_aabb: tuple[np.ndarray, np.ndarray] | None = None,
) -> None:
    """Mark voxels occupied by a single link mesh, fully on the GPU.

    World AABB is derived from the **8 corners** of the cached local AABB
    (not from transforming every vertex), then converted to a deterministic
    voxel index box on the CPU.  The matching sub-grid of ``centers_grid`` is
    sliced (a view — no sync), the containment/proximity test runs on the
    device, and results are written back through the slice view with
    ``masked_fill_`` (element-wise — no data-dependent host sync).  Thus the
    only sync in the whole per-frame loop is the final state ``.cpu()``.
    """
    linear = T_world[:3, :3].astype(np.float32)
    t = T_world[:3, 3].astype(np.float32)

    mesh_key = id(lm.mesh)
    occupancy_test = containment_cache.get(mesh_key)
    margin = occupancy_test.radius if isinstance(occupancy_test, MeshSurfaceProximityGPU) else 0.0

    # Local AABB -> world AABB via 8 corners (avoids full-vertex transform).
    cached = aabb_cache.get(mesh_key)
    if cached is None:
        return
    lo, hi = cached
    lo = np.asarray(lo, dtype=np.float32)
    hi = np.asarray(hi, dtype=np.float32)
    center = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    signs = np.array([
        [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
        [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1],
    ], dtype=np.float32)
    corners_world = (signs * half + center) @ linear.T + t  # (8, 3) on CPU
    wmin = corners_world.min(axis=0) - margin
    wmax = corners_world.max(axis=0) + margin

    if clip_aabb is not None:
        clip_min, clip_max = clip_aabb
        wmin = np.maximum(wmin, np.asarray(clip_min, dtype=np.float32))
        wmax = np.minimum(wmax, np.asarray(clip_max, dtype=np.float32))
        if np.any(wmax <= wmin):
            return

    # World AABB -> voxel index box (deterministic, no GPU sync).
    b0 = bounds[0]
    nx_g, ny_g, nz_g = state_grid.shape

    def _index_range(lo_w: float, hi_w: float, origin: float, n: int) -> tuple[int, int]:
        # Voxel i center is at origin + (i + 0.5) * voxel_size.  Return the
        # inclusive-low / exclusive-high index range whose centers fall in
        # [lo_w, hi_w], clamped to [0, n].
        i0 = int(np.floor((lo_w - origin) / voxel_size - 0.5))
        i1 = int(np.floor((hi_w - origin) / voxel_size - 0.5)) + 1
        return max(0, i0), min(n, i1)

    ix0, ix1 = _index_range(wmin[0], wmax[0], b0[0], nx_g)
    iy0, iy1 = _index_range(wmin[1], wmax[1], b0[1], ny_g)
    iz0, iz1 = _index_range(wmin[2], wmax[2], b0[2], nz_g)
    if ix0 >= ix1 or iy0 >= iy1 or iz0 >= iz1:
        return

    bx, by, bz = ix1 - ix0, iy1 - iy0, iz1 - iz0

    # Slice the candidate sub-grid (a view — no copy, no sync) and flatten to
    # (M, 3) world points for the containment/proximity test.
    sub = centers_grid[ix0:ix1, iy0:iy1, iz0:iz1]  # (bx, by, bz, 3)
    cand_world = sub.reshape(-1, 3)

    linear_gpu = _torch.from_numpy(linear).to(device=device)
    t_gpu = _torch.from_numpy(t).to(device=device)

    if occupancy_test is None:
        # CPU trimesh fallback (rare): sync candidates only, write back on GPU.
        import trimesh
        cand_world_cpu = cand_world.detach().cpu().numpy()
        verts_world = (np.asarray(lm.mesh.vertices, dtype=np.float32) @ linear.T) + t
        world_mesh = trimesh.Trimesh(vertices=verts_world, faces=lm.mesh.faces, process=False)
        try:
            inside = world_mesh.contains(cand_world_cpu)
        except Exception:
            return
        mask_3d = _torch.from_numpy(inside).to(device=device).view(bx, by, bz)
    elif isinstance(occupancy_test, MeshSurfaceProximityGPU):
        if _is_rotation_matrix_like(linear):
            # Pure rotation: world<->local distances are identical.
            cand_local = (cand_world - t_gpu) @ linear_gpu
            inside = occupancy_test.near_surface(cand_local)
        else:
            # Non-uniform scale: query in world frame, transform samples.
            inside = occupancy_test.near_surface(
                cand_world, sample_linear=linear_gpu, sample_translation=t_gpu,
            )
        mask_3d = inside.view(bx, by, bz)
    else:
        # Volumetric containment in link-local frame.
        inv_lin = _torch.linalg.inv(linear_gpu)
        cand_local = (cand_world - t_gpu) @ inv_lin.t()
        inside = occupancy_test.contains(cand_local)
        mask_3d = inside.view(bx, by, bz)

    # Element-wise scatter writes through the slice view — no host sync.
    state_grid[ix0:ix1, iy0:iy1, iz0:iz1].masked_fill_(mask_3d, _OCCUPIED)
    if semantic_grid is not None and sem_id > 0:
        semantic_grid[ix0:ix1, iy0:iy1, iz0:iz1].masked_fill_(mask_3d, sem_id)
