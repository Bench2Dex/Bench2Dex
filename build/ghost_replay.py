"""Ghost replay for teleoperation guidance."""

from __future__ import annotations
import os
from typing import Dict, List, Tuple
import numpy as np

from utils.usd_prims import get_current_stage_compat


class GhostReplayData:
    def __init__(self, hdf5_path: str):
        import h5py
        with h5py.File(hdf5_path, "r") as hf:
            self.frame_count = int(hf["meta/frame_count"][()])
            fps = float(hf["meta/fps"][()]) if "meta/fps" in hf else 20.0
            self.frame_dt = 1.0 / max(fps, 1.0)
            self.object_poses: Dict[str, np.ndarray] = {}
            for obj_key in hf["objects"].keys():
                k = f"objects/{obj_key}/pose_world"
                if k in hf:
                    self.object_poses[obj_key] = hf[k][:].astype(np.float64)
            # Table-height offset of the recorded episode (m). Used to realign
            # ghost z to the *current* episode's (possibly different) table.
            self.table_height_offset_m = self._read_table_offset(hf)

    @staticmethod
    def _read_table_offset(hf) -> float:
        import json
        if "meta/scene_generalization_sample" not in hf:
            return 0.0
        raw = hf["meta/scene_generalization_sample"][()]
        try:
            sample = json.loads(raw.decode() if isinstance(raw, bytes) else str(raw))
        except (ValueError, UnicodeDecodeError):
            return 0.0
        th = (sample.get("spatial", {}) or {}).get("table_height", {}) or {}
        return float(th.get("height_offset_m", 0.0) or 0.0)

    def get_frame_poses(self, frame: int) -> Dict[str, np.ndarray]:
        f = min(max(frame, 0), self.frame_count - 1)
        return {oid: poses[f] for oid, poses in self.object_poses.items()}


def _stage():
    return get_current_stage_compat()


def _make_editable(prim_path: str) -> int:
    """Clear `instanceable` flags so the whole subtree becomes editable.

    Some assets (e.g. 223_chefmate_frypan) reference their geometry as USD
    *instances*. Instanced meshes are exposed only as read-only instance
    proxies, which `Usd.PrimRange` does not descend into and which cannot be
    authored on. Flipping `instanceable` off (top-down) turns the instance into
    a normal, editable subtree so later tinting / physics-disable can reach the
    actual Gprims. Returns the number of prims de-instanced.
    """
    from pxr import Usd
    stage = _stage()
    root = stage.GetPrimAtPath(prim_path)
    if not root.IsValid():
        return 0
    cleared = 0
    stack = [root]
    while stack:
        p = stack.pop()
        if p.IsInstanceable():
            p.SetInstanceable(False)
            cleared += 1
        stack.extend(p.GetChildren())
    return cleared


def _disable_physics_recursive(prim_path: str) -> None:
    from pxr import Usd, UsdPhysics
    stage = _stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return
    for p in Usd.PrimRange(prim):
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            api = UsdPhysics.RigidBodyAPI(p)
            attr = api.GetRigidBodyEnabledAttr()
            if not attr or not attr.IsValid():
                attr = api.CreateRigidBodyEnabledAttr()
            attr.Set(False)
        if p.HasAPI(UsdPhysics.CollisionAPI):
            api = UsdPhysics.CollisionAPI(p)
            attr = api.GetCollisionEnabledAttr()
            if not attr or not attr.IsValid():
                attr = api.CreateCollisionEnabledAttr()
            attr.Set(False)


def _apply_tint(ghost_prim_path: str) -> None:
    """Set blue tint via displayColor primvar on all descendant Gprims."""
    from pxr import Usd, UsdShade, UsdGeom, Gf, Vt, Sdf

    stage = _stage()
    prim = stage.GetPrimAtPath(ghost_prim_path)
    if not prim.IsValid():
        return

    # Unbind materials so primvar colour takes effect
    for p in Usd.PrimRange(prim):
        try:
            UsdShade.MaterialBindingAPI(p).UnbindAllBindings()
        except Exception:
            pass

    color = Vt.Vec3fArray([Gf.Vec3f(0.3, 0.6, 1.0)])
    for p in Usd.PrimRange(prim):
        if not p.IsA(UsdGeom.Gprim):
            continue
        pvar = UsdGeom.PrimvarsAPI(p)
        if pvar.HasPrimvar("displayColor"):
            pvar.RemovePrimvar("displayColor")
        pvar.CreatePrimvar("displayColor", Sdf.ValueTypeNames.Float3Array).Set(color)


def spawn_ghost_objects(
    task_objects: List[str],
    object_asset_paths: Dict[str, str],
    object_scales: Dict[str, Tuple[float, float, float]] = None,
) -> Dict[str, str]:
    from pxr import UsdGeom

    stage = _stage()
    root = "/World/Ghosts"
    if not stage.GetPrimAtPath(root).IsValid():
        stage.DefinePrim(root, "Xform")

    ghost_map: Dict[str, str] = {}
    for obj_id in task_objects:
        asset_path = object_asset_paths.get(obj_id)
        if not asset_path or not os.path.exists(asset_path):
            print(f"[Ghost] SKIP {obj_id}: no asset")
            continue

        ghost_path = f"/World/Ghosts/{obj_id.replace(' ', '_')}"
        if stage.GetPrimAtPath(ghost_path).IsValid():
            stage.RemovePrim(ghost_path)

        stage.DefinePrim(ghost_path, "Xform")
        ref_path = f"{ghost_path}/asset"
        stage.DefinePrim(ref_path, "Xform").GetReferences().AddReference(asset_path)
        # De-instance first: instanced geometry (e.g. frypan) is otherwise
        # unreachable by PrimRange, so its material never unbinds and the tint
        # never applies — it would show the original texture, not blue.
        n_deinst = _make_editable(ref_path)
        _disable_physics_recursive(ref_path)
        _apply_tint(ghost_path)
        if n_deinst:
            print(f"[Ghost] {obj_id}: de-instanced {n_deinst} prim(s) for tint")


        # Set initial ops: translate(0,0,0) + orient(1,0,0,0) + scale
        from pxr import Gf
        xform = UsdGeom.Xformable(stage.GetPrimAtPath(ghost_path))
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0))
        xform.AddOrientOp().Set(Gf.Quatf(1.0, Gf.Vec3f(0, 0, 0)))
        if object_scales:
            sc = object_scales.get(obj_id)
            if sc is not None:
                xform.AddScaleOp().Set(Gf.Vec3f(float(sc[0]), float(sc[1]), float(sc[2])))

        ghost_map[obj_id] = ghost_path
        print(f"[Ghost] {obj_id} -> {ghost_path}")

    return ghost_map


def _apply_pose(prim, pose_xyzw) -> None:
    from pxr import UsdGeom, Gf
    xform = UsdGeom.Xformable(prim)
    ops = {op.GetOpType(): op for op in xform.GetOrderedXformOps()}
    t_op = ops.get(UsdGeom.XformOp.TypeTranslate)
    r_op = ops.get(UsdGeom.XformOp.TypeOrient)
    if t_op is None or r_op is None:
        # Should not happen — spawn always creates them
        return
    x, y, z = float(pose_xyzw[0]), float(pose_xyzw[1]), float(pose_xyzw[2])
    qx, qy, qz, qw = (float(pose_xyzw[3]), float(pose_xyzw[4]),
                      float(pose_xyzw[5]), float(pose_xyzw[6]))
    t_op.Set(Gf.Vec3d(x, y, z))
    r_op.Set(Gf.Quatf(qw, Gf.Vec3f(qx, qy, qz)))


def update_ghost_objects(
    ghost_map: Dict[str, str],
    replay_data: GhostReplayData,
    ghost_frame: int,
    sim_step: int = 0,
    z_offset: float = 0.0,
) -> None:
    """Update ghost positions.

    z_offset shifts every ghost in world-z to compensate for a table-height
    difference between the recorded episode and the current one.
    """
    poses = replay_data.get_frame_poses(ghost_frame)
    stage = _stage()
    visible = (sim_step % 10 == 0) or True
    for obj_id, prim_path in ghost_map.items():
        pose = poses.get(obj_id)
        if pose is None:
            continue
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            continue
        if visible:
            shifted = pose.copy()
            shifted[2] += z_offset
            _apply_pose(prim, shifted)
        else:
            # Hide underground
            _apply_pose(prim, np.array([pose[0], pose[1], -10.0, 1.0, 0.0, 0.0, 0.0]))


def current_table_offset_m(scene_generalization_sample) -> float:
    """Extract table-height offset (m) from a live scene generalization sample.

    The sample is a nested namespace (sample.spatial.table_height.height_offset_m).
    Returns 0.0 if any level is missing.
    """
    spatial = getattr(scene_generalization_sample, "spatial", None)
    th = getattr(spatial, "table_height", None) if spatial is not None else None
    return float(getattr(th, "height_offset_m", 0.0) or 0.0) if th is not None else 0.0


def resolve_ghost_object_ids(
    replay_data: GhostReplayData,
    scene_object_ids: List[str],
) -> List[str]:
    hdf5_ids = set(replay_data.object_poses.keys())
    matched = sorted(hdf5_ids & set(scene_object_ids))
    if not matched:
        print(f"[Ghost] WARN: no match — HDF5={sorted(hdf5_ids)} scene={sorted(scene_ids)}")
    return matched
