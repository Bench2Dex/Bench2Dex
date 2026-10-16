# Rebuilt Tactile Mesh -> TacMap NPY Commands

This document lists the commands for converting rebuilt tactile contact-surface meshes to `.npy` files for each hand under the current four-finger-shared TacMap configuration. It focuses on `--origin-xyz` and `--origin-rpy`.

These commands match the current runtime configuration: one group for the thumb and one shared group for the other fingers. Sharpa therefore uses `TH` / `4F`; other bimanual robots use `RTH` / `R4F` / `LTH` / `L4F`. A per-finger five-digit setup requires more than changing output filenames: update the `TacMapNpyGroup` entries in `collector/tacmap_configs.py` and generate separate point/normal NPY files for every finger.

Run these commands from the repository root:

```bash
cd .
```

The commands assume input meshes are stored at:

```text
../dex2bench_dataset/Robots_p/<dataset_key>/tactile_sensor/tactile_surface_<GROUP>.stl
```

If you exported an `.obj` file or used a name such as `sensor_fingertip.obj`, change only the `--mesh` path.

## Coordinate Frame Conventions

These commands apply when you crop or rebuild a mesh in Blender from the original mesh referenced by the robot URDF, and the exported mesh remains in the original mesh file coordinate frame.

The single-mesh mode of `gen_tacmap_npy.py --mesh ...` does not read the URDF. You must therefore provide the following values from the URDF `<visual>` / `<collision>` element:

- `origin xyz` -> `--origin-xyz`
- `origin rpy` -> `--origin-rpy`
- `mesh scale` -> `--mesh-scale`

If you have already aligned the new mesh in Blender and exported it in the TacMap attach link local frame, do not apply the URDF origin, RPY, or scale again; that would transform it twice. Use:

```bash
--origin-xyz 0,0,0 --origin-rpy 0,0,0 --mesh-scale 1,1,1
```

## Output Naming

Legacy Sharpa assets:

```text
tactileSensor_map_TH_point.npy
tactileSensor_map_TH_normal.npy
tactileSensor_map_4F_point.npy
tactileSensor_map_4F_normal.npy
```

Four groups for other robots:

```text
tactileSensor_map_RTH_point.npy
tactileSensor_map_RTH_normal.npy
tactileSensor_map_R4F_point.npy
tactileSensor_map_R4F_normal.npy
tactileSensor_map_LTH_point.npy
tactileSensor_map_LTH_normal.npy
tactileSensor_map_L4F_point.npy
tactileSensor_map_L4F_normal.npy
```

Suggested names for a per-finger five-digit layout:

```text
tactileSensor_map_RTH_point.npy
tactileSensor_map_RTH_normal.npy
tactileSensor_map_RIDX_point.npy
tactileSensor_map_RIDX_normal.npy
tactileSensor_map_RMID_point.npy
tactileSensor_map_RMID_normal.npy
tactileSensor_map_RRING_point.npy
tactileSensor_map_RRING_normal.npy
tactileSensor_map_RLIT_point.npy
tactileSensor_map_RLIT_normal.npy
tactileSensor_map_LTH_point.npy
tactileSensor_map_LTH_normal.npy
tactileSensor_map_LIDX_point.npy
tactileSensor_map_LIDX_normal.npy
tactileSensor_map_LMID_point.npy
tactileSensor_map_LMID_normal.npy
tactileSensor_map_LRING_point.npy
tactileSensor_map_LRING_normal.npy
tactileSensor_map_LLIT_point.npy
tactileSensor_map_LLIT_normal.npy
```

For four-finger hands such as Allegro or Leap that lack a little/pinky finger, keep only `TH` / `IDX` / `MID` / `RING`. If the left and right hands truly share the same mesh and attach-link local frame, side-shared names without `R` / `L` may be retained. If their URDF links, origins, or mesh mirroring differ, use the side-specific names above.

## Information Still Needed for a Per-Finger Five-Digit Layout

The existing four-finger-shared commands record only one representative non-thumb mesh/origin for `R4F` / `L4F`, usually the index finger. A per-finger five-digit layout needs the following information for every finger:

- `TacMapNpyGroup.name`, for example `RIDX`, `RMID`, `RRING`, or `RLIT`.
- `points_npy` / `normals_npy`, for example `tactileSensor_map_RIDX_point.npy` and `tactileSensor_map_RIDX_normal.npy`.
- `site_names`: the tactile site bound to each NPY; a per-finger layout usually assigns one site per group.
- Rebuilt mesh path: use `tactile_surface_<GROUP>.stl`, for example `tactile_surface_RMID.stl`.
- Original source mesh: use the corresponding finger mesh from the URDF `<visual>` or `<collision>`, rather than reusing the index-finger mesh.
- `attach_link` / `source_link`: the link used to attach TacMap at runtime and the link from which geometry is obtained. For Allegro, Wuji, and some other hands, the source link may need to map to a tip link.
- `origin xyz` / `origin rpy` / `mesh scale`: record values from each finger's URDF geometry; do not assume the `R4F` / `L4F` values apply.
- `sensor-normal`: start with `auto`. If the point cloud lands on the back side or the hit rate is abnormal, record an explicit direction for that finger.
- `geometry-role`: Orca currently uses the collision skin mesh; other hands default to visual geometry. Preserve this choice in a per-finger layout.

Known cases where `R4F` / `L4F` should not be reused blindly:

- RH56DFX, RH5DG2, Wuji, Orca, and DexHand021 usually have separate mesh files for each non-thumb finger.
- LEAP uses the same `fingertip.obj` for the index and middle/ring fingertips, but their URDF visual origins differ slightly. Do not copy only the index finger's `origin-xyz`.
- Ability's right pinky uses a source mesh that differs from those of the other non-thumb fingers.
- Even for hands such as Shadow, Schunk, and Allegro that reuse geometry files extensively, bind a separate site/link for each finger; otherwise runtime sensing remains shared across the four fingers.

Update the runtime code as well, or the new files will not be loaded:

1. In `collector/tacmap_configs.py`, split the aggregated `R4F` / `L4F` groups generated by `_split_side_groups(...)` into one `TacMapNpyGroup` per finger.
2. Ensure the site names in `collector/tactile.py` correspond one-to-one with the new groups.
3. Regenerate the corresponding `.npy` files, then inspect each group's point cloud with `tools/vis/vis_tacmap_npy.py`.

Suggested group-to-site mappings for a per-finger five-digit layout:

| Robot | Right groups | Left groups |
| --- | --- | --- |
| RH56DFX | `RTH=right_thumb_pad`, `RIDX=right_index_pad`, `RMID=right_middle_pad`, `RRING=right_ring_pad`, `RLIT=right_little_pad` | `LTH=left_thumb_pad`, `LIDX=left_index_pad`, `LMID=left_middle_pad`, `LRING=left_ring_pad`, `LLIT=left_little_pad` |
| RH5DG2 | `RTH=right_thumb_pad`, `RIDX=right_index_pad`, `RMID=right_middle_pad`, `RRING=right_ring_pad`, `RLIT=right_little_pad` | `LTH=left_thumb_pad`, `LIDX=left_index_pad`, `LMID=left_middle_pad`, `LRING=left_ring_pad`, `LLIT=left_little_pad` |
| Shadow | `RTH=right_THJ1`, `RIDX=right_FFJ1`, `RMID=right_MFJ1`, `RRING=right_RFJ1`, `RLIT=right_LFJ1` | `LTH=left_THJ1`, `LIDX=left_FFJ1`, `LMID=left_MFJ1`, `LRING=left_RFJ1`, `LLIT=left_LFJ1` |
| Schunk | `RTH=right_Thumb_Flexion`, `RIDX=right_Index_Finger_Distal`, `RMID=right_Middle_Finger_Distal`, `RRING=right_Ring_Finger`, `RLIT=right_Pinky` | `LTH=left_Thumb_Flexion`, `LIDX=left_Index_Finger_Distal`, `LMID=left_Middle_Finger_Distal`, `LRING=left_Ring_Finger`, `LLIT=left_Pinky` |
| Wuji | `RTH=right_thumb_J4`, `RIDX=right_index_J4`, `RMID=right_middle_J4`, `RRING=right_ring_J4`, `RLIT=right_little_J4` | `LTH=left_thumb_J4`, `LIDX=left_index_J4`, `LMID=left_middle_J4`, `LRING=left_ring_J4`, `LLIT=left_little_J4` |
| Allegro | `RTH=right_thumb_3`, `RIDX=right_index_3`, `RMID=right_middle_3`, `RRING=right_ring_3` | `LTH=left_thumb_3`, `LIDX=left_index_3`, `LMID=left_middle_3`, `LRING=left_ring_3` |
| Orca | `RTH=right_thumb_dip`, `RIDX=right_index_pip`, `RMID=right_middle_pip`, `RRING=right_ring_pip`, `RLIT=right_pinky_pip` | `LTH=left_thumb_dip`, `LIDX=left_index_pip`, `LMID=left_middle_pip`, `LRING=left_ring_pip`, `LLIT=left_pinky_pip` |
| Ability | `RTH=right_thumb_q2`, `RIDX=right_index_q2`, `RMID=right_middle_q2`, `RRING=right_ring_q2`, `RLIT=right_pinky_q2` | `LTH=left_thumb_q2`, `LIDX=left_index_q2`, `LMID=left_middle_q2`, `LRING=left_ring_q2`, `LLIT=left_pinky_q2` |
| Leap | `RTH=right_thumb_2`, `RIDX=right_index_2`, `RMID=right_middle_2`, `RRING=right_ring_2` | `LTH=left_thumb_2`, `LIDX=left_index_2`, `LMID=left_middle_2`, `LRING=left_ring_2` |
| DexHand021 | `RTH=right_thumb_J4`, `RIDX=right_index_J4`, `RMID=right_middle_J4`, `RRING=right_ring_J4`, `RLIT=right_little_J4` | `LTH=left_thumb_J4`, `LIDX=left_index_J4`, `LMID=left_middle_J4`, `LRING=left_ring_J4`, `LLIT=left_little_J4` |
| Sharpa | The legacy assets use side-shared `TH` / `4F` groups. Before splitting them into five fingers, confirm whether the left and right elastomer meshes are identical, then choose between `TH/IDX/MID/RING/LIT` and side-specific `R*` / `L*` names. |

Single-mesh command template for a per-finger five-digit layout:

```bash
python tools/asset/gen_tacmap_npy.py \
  --mesh ../dex2bench_dataset/Robots_p/<dataset_key>/tactile_sensor/tactile_surface_<GROUP>.stl \
  --output-dir ../dex2bench_dataset/Robots_p/<dataset_key>/tactile_sensor \
  --output-stem tactileSensor_map_<GROUP> \
  --native-resolution 240 \
  --sensor-normal auto \
  --mesh-units m \
  --origin-xyz <URDF_GEOMETRY_ORIGIN_XYZ> \
  --origin-rpy <URDF_GEOMETRY_ORIGIN_RPY> \
  --mesh-scale <URDF_MESH_SCALE>
```

## 1. Sharpa

Dataset: `kuka+sharpa`

Source mesh:

- TH: `meshes/thumb_elastomer_surface.STL`
- 4F: `meshes/elastomer_surface.STL`

```bash
python tools/asset/gen_tacmap_npy.py \
  --mesh ../dex2bench_dataset/Robots_p/kuka+sharpa/tactile_sensor/tactile_surface_TH.stl \
  --output-dir ../dex2bench_dataset/Robots_p/kuka+sharpa/tactile_sensor \
  --output-stem tactileSensor_map_TH \
  --native-resolution 240 \
  --sensor-normal auto \
  --mesh-units m \
  --origin-xyz 0,0,0 \
  --origin-rpy 0,0,0 \
  --mesh-scale 1,1,1

python tools/asset/gen_tacmap_npy.py \
  --mesh ../dex2bench_dataset/Robots_p/kuka+sharpa/tactile_sensor/tactile_surface_4F.stl \
  --output-dir ../dex2bench_dataset/Robots_p/kuka+sharpa/tactile_sensor \
  --output-stem tactileSensor_map_4F \
  --native-resolution 240 \
  --sensor-normal auto \
  --mesh-units m \
  --origin-xyz 0,0,0 \
  --origin-rpy 0,0,0 \
  --mesh-scale 1,1,1
```

## 2. RH56DFX

Dataset: `ur5+RH56DFX`

Source mesh:

- RTH: `meshes/hand_right/right_thumb_rubber_3.STL`
- R4F: `meshes/hand_right/right_index_rubber_2.STL`
- LTH: `meshes/hand_left/left_thumb_rubber_3.STL`
- L4F: `meshes/hand_left/left_index_rubber_2.STL`

```bash
python tools/asset/gen_tacmap_npy.py \
  --mesh ../dex2bench_dataset/Robots_p/ur5+RH56DFX/tactile_sensor/tactile_surface_RTH.stl \
  --output-dir ../dex2bench_dataset/Robots_p/ur5+RH56DFX/tactile_sensor \
  --output-stem tactileSensor_map_RTH \
  --native-resolution 240 \
  --sensor-normal 0,0,1 \
  --mesh-units m \
  --origin-xyz 0,0,0 \
  --origin-rpy 0,0,0 \
  --mesh-scale 1,1,1

python tools/asset/gen_tacmap_npy.py \
  --mesh ../dex2bench_dataset/Robots_p/ur5+RH56DFX/tactile_sensor/tactile_surface_R4F.stl \
  --output-dir ../dex2bench_dataset/Robots_p/ur5+RH56DFX/tactile_sensor \
  --output-stem tactileSensor_map_R4F \
  --native-resolution 240 \
  --sensor-normal 0,0,1 \
  --mesh-units m \
  --origin-xyz 0,0,0 \
  --origin-rpy 0,0,0 \
  --mesh-scale 1,1,1

python tools/asset/gen_tacmap_npy.py \
  --mesh ../dex2bench_dataset/Robots_p/ur5+RH56DFX/tactile_sensor/tactile_surface_LTH.stl \
  --output-dir ../dex2bench_dataset/Robots_p/ur5+RH56DFX/tactile_sensor \
  --output-stem tactileSensor_map_LTH \
  --native-resolution 240 \
  --sensor-normal 0,0,1 \
  --mesh-units m \
  --origin-xyz 0,0,0 \
  --origin-rpy 0,0,0 \
  --mesh-scale 1,1,1

python tools/asset/gen_tacmap_npy.py \
  --mesh ../dex2bench_dataset/Robots_p/ur5+RH56DFX/tactile_sensor/tactile_surface_L4F.stl \
  --output-dir ../dex2bench_dataset/Robots_p/ur5+RH56DFX/tactile_sensor \
  --output-stem tactileSensor_map_L4F \
  --native-resolution 240 \
  --sensor-normal 0,0,1 \
  --mesh-units m \
  --origin-xyz 0,0,0 \
  --origin-rpy 0,0,0 \
  --mesh-scale 1,1,1
```

## 3. RH5DG2

Dataset: `ur5+RH5DG2`

Source mesh:

- RTH: `meshes/hand_right/right_thumb_force_sensor.STL`
- R4F: `meshes/hand_right/right_index_force_sensor.STL`
- LTH: `meshes/hand_left/left_thumb_force_sensor.STL`
- L4F: `meshes/hand_left/left_index_force_sensor.STL`

```bash
python tools/asset/gen_tacmap_npy.py \
  --mesh ../dex2bench_dataset/Robots_p/ur5+RH5DG2/tactile_sensor/tactile_surface_RTH.stl \
  --output-dir ../dex2bench_dataset/Robots_p/ur5+RH5DG2/tactile_sensor \
  --output-stem tactileSensor_map_RTH \
  --native-resolution 240 \
  --sensor-normal auto \
  --mesh-units m \
  --origin-xyz 0,0,0 \
  --origin-rpy 0,0,0 \
  --mesh-scale 1,1,1

python tools/asset/gen_tacmap_npy.py \
  --mesh ../dex2bench_dataset/Robots_p/ur5+RH5DG2/tactile_sensor/tactile_surface_R4F.stl \
  --output-dir ../dex2bench_dataset/Robots_p/ur5+RH5DG2/tactile_sensor \
  --output-stem tactileSensor_map_R4F \
  --native-resolution 240 \
  --sensor-normal auto \
  --mesh-units m \
  --origin-xyz 0,0,0 \
  --origin-rpy 0,0,0 \
  --mesh-scale 1,1,1

python tools/asset/gen_tacmap_npy.py \
  --mesh ../dex2bench_dataset/Robots_p/ur5+RH5DG2/tactile_sensor/tactile_surface_LTH.stl \
  --output-dir ../dex2bench_dataset/Robots_p/ur5+RH5DG2/tactile_sensor \
  --output-stem tactileSensor_map_LTH \
  --native-resolution 240 \
  --sensor-normal auto \
  --mesh-units m \
  --origin-xyz 0,0,0 \
  --origin-rpy 0,0,0 \
  --mesh-scale 1,1,1

python tools/asset/gen_tacmap_npy.py \
  --mesh ../dex2bench_dataset/Robots_p/ur5+RH5DG2/tactile_sensor/tactile_surface_L4F.stl \
  --output-dir ../dex2bench_dataset/Robots_p/ur5+RH5DG2/tactile_sensor \
  --output-stem tactileSensor_map_L4F \
  --native-resolution 240 \
  --sensor-normal auto \
  --mesh-units m \
  --origin-xyz 0,0,0 \
  --origin-rpy 0,0,0 \
  --mesh-scale 1,1,1
```

## 4. Shadow

Dataset: `ur5+shadow_hand`

Source mesh:

- TH: `meshes/hand/visual/th_distal_pst.obj`
- 4F: `meshes/hand/visual/f_distal_pst.obj`

Important: Shadow URDF applies `mesh scale="0.001 0.001 0.001"`. If your rebuilt mesh is still in the original OBJ coordinate scale, keep `--mesh-scale 0.001,0.001,0.001`. If Blender has already applied this scale and exported meters, change it to `1,1,1`.

```bash
python tools/asset/gen_tacmap_npy.py \
  --mesh ../dex2bench_dataset/Robots_p/ur5+shadow_hand/tactile_sensor/tactile_surface_RTH.stl \
  --output-dir ../dex2bench_dataset/Robots_p/ur5+shadow_hand/tactile_sensor \
  --output-stem tactileSensor_map_RTH \
  --native-resolution 240 \
  --sensor-normal auto \
  --mesh-units m \
  --origin-xyz 0,0,0 \
  --origin-rpy 0,0,0 \
  --mesh-scale 0.001,0.001,0.001

python tools/asset/gen_tacmap_npy.py \
  --mesh ../dex2bench_dataset/Robots_p/ur5+shadow_hand/tactile_sensor/tactile_surface_R4F.stl \
  --output-dir ../dex2bench_dataset/Robots_p/ur5+shadow_hand/tactile_sensor \
  --output-stem tactileSensor_map_R4F \
  --native-resolution 240 \
  --sensor-normal auto \
  --mesh-units m \
  --origin-xyz 0,0,0 \
  --origin-rpy 0,0,0 \
  --mesh-scale 0.001,0.001,0.001

python tools/asset/gen_tacmap_npy.py \
  --mesh ../dex2bench_dataset/Robots_p/ur5+shadow_hand/tactile_sensor/tactile_surface_LTH.stl \
  --output-dir ../dex2bench_dataset/Robots_p/ur5+shadow_hand/tactile_sensor \
  --output-stem tactileSensor_map_LTH \
  --native-resolution 240 \
  --sensor-normal auto \
  --mesh-units m \
  --origin-xyz 0,0,0 \
  --origin-rpy 0,0,0 \
  --mesh-scale 0.001,0.001,0.001

python tools/asset/gen_tacmap_npy.py \
  --mesh ../dex2bench_dataset/Robots_p/ur5+shadow_hand/tactile_sensor/tactile_surface_L4F.stl \
  --output-dir ../dex2bench_dataset/Robots_p/ur5+shadow_hand/tactile_sensor \
  --output-stem tactileSensor_map_L4F \
  --native-resolution 240 \
  --sensor-normal auto \
  --mesh-units m \
  --origin-xyz 0,0,0 \
  --origin-rpy 0,0,0 \
  --mesh-scale 0.001,0.001,0.001
```

## 5. Schunk

Dataset: `ur5+schunk_hand`

Source mesh:

- RTH: `meshes/hand/visual/d13.obj`
- R4F: `meshes/hand/visual/finger_tip.obj`
- LTH: `meshes/hand/visual/d13_left.obj`
- L4F: `meshes/hand/visual/finger_tip.obj`

```bash
python tools/asset/gen_tacmap_npy.py \
  --mesh ../dex2bench_dataset/Robots_p/ur5+schunk_hand/tactile_sensor/tactile_surface_RTH.stl \
  --output-dir ../dex2bench_dataset/Robots_p/ur5+schunk_hand/tactile_sensor \
  --output-stem tactileSensor_map_RTH \
  --native-resolution 240 \
  --sensor-normal auto \
  --mesh-units m \
  --origin-xyz 0,0,0 \
  --origin-rpy 0,0,0 \
  --mesh-scale 1,1,1

python tools/asset/gen_tacmap_npy.py \
  --mesh ../dex2bench_dataset/Robots_p/ur5+schunk_hand/tactile_sensor/tactile_surface_R4F.stl \
  --output-dir ../dex2bench_dataset/Robots_p/ur5+schunk_hand/tactile_sensor \
  --output-stem tactileSensor_map_R4F \
  --native-resolution 240 \
  --sensor-normal 0,1,0 \
  --mesh-units m \
  --origin-xyz 0,0,0 \
  --origin-rpy 0,0,0 \
  --mesh-scale 1,1,1

python tools/asset/gen_tacmap_npy.py \
  --mesh ../dex2bench_dataset/Robots_p/ur5+schunk_hand/tactile_sensor/tactile_surface_LTH.stl \
  --output-dir ../dex2bench_dataset/Robots_p/ur5+schunk_hand/tactile_sensor \
  --output-stem tactileSensor_map_LTH \
  --native-resolution 240 \
  --sensor-normal auto \
  --mesh-units m \
  --origin-xyz 0,0,0 \
  --origin-rpy 0,0,0 \
  --mesh-scale 1,1,1

python tools/asset/gen_tacmap_npy.py \
  --mesh ../dex2bench_dataset/Robots_p/ur5+schunk_hand/tactile_sensor/tactile_surface_L4F.stl \
  --output-dir ../dex2bench_dataset/Robots_p/ur5+schunk_hand/tactile_sensor \
  --output-stem tactileSensor_map_L4F \
  --native-resolution 240 \
  --sensor-normal 0,1,0 \
  --mesh-units m \
  --origin-xyz 0,0,0 \
  --origin-rpy 0,0,0 \
  --mesh-scale 1,1,1
```

## 6. Wuji

Dataset: `ur5+wuji`

Source mesh:

- RTH: `meshes/hand_right/right_finger1_tip_link.STL`
- R4F: `meshes/hand_right/right_finger2_tip_link.STL`
- LTH: `meshes/hand_left/left_finger1_tip_link.STL`
- L4F: `meshes/hand_left/left_finger2_tip_link.STL`

```bash
python tools/asset/gen_tacmap_npy.py \
  --mesh ../dex2bench_dataset/Robots_p/ur5+wuji/tactile_sensor/tactile_surface_RTH.stl \
  --output-dir ../dex2bench_dataset/Robots_p/ur5+wuji/tactile_sensor \
  --output-stem tactileSensor_map_RTH \
  --native-resolution 240 \
  --sensor-normal -1,0,0 \
  --mesh-units m \
  --origin-xyz 0,0,0 \
  --origin-rpy 0,0,0 \
  --mesh-scale 1,1,1

python tools/asset/gen_tacmap_npy.py \
  --mesh ../dex2bench_dataset/Robots_p/ur5+wuji/tactile_sensor/tactile_surface_R4F.stl \
  --output-dir ../dex2bench_dataset/Robots_p/ur5+wuji/tactile_sensor \
  --output-stem tactileSensor_map_R4F \
  --native-resolution 240 \
  --sensor-normal auto \
  --mesh-units m \
  --origin-xyz 0,0,0 \
  --origin-rpy 0,0,0 \
  --mesh-scale 1,1,1

python tools/asset/gen_tacmap_npy.py \
  --mesh ../dex2bench_dataset/Robots_p/ur5+wuji/tactile_sensor/tactile_surface_LTH.stl \
  --output-dir ../dex2bench_dataset/Robots_p/ur5+wuji/tactile_sensor \
  --output-stem tactileSensor_map_LTH \
  --native-resolution 240 \
  --sensor-normal 1,0,0 \
  --mesh-units m \
  --origin-xyz 0,0,0 \
  --origin-rpy 0,0,0 \
  --mesh-scale 1,1,1

python tools/asset/gen_tacmap_npy.py \
  --mesh ../dex2bench_dataset/Robots_p/ur5+wuji/tactile_sensor/tactile_surface_L4F.stl \
  --output-dir ../dex2bench_dataset/Robots_p/ur5+wuji/tactile_sensor \
  --output-stem tactileSensor_map_L4F \
  --native-resolution 240 \
  --sensor-normal auto \
  --mesh-units m \
  --origin-xyz 0,0,0 \
  --origin-rpy 0,0,0 \
  --mesh-scale 1,1,1
```

## 7. Allegro

Dataset: `panda+allegro`

Source mesh:

- TH: `meshes/visual/link_tip.obj`
- 4F: `meshes/visual/link_tip.obj`

All groups use the same mesh file, but right and left groups use different explicit sensor normals.

```bash
python tools/asset/gen_tacmap_npy.py \
  --mesh ../dex2bench_dataset/Robots_p/panda+allegro/tactile_sensor/tactile_surface_RTH.stl \
  --output-dir ../dex2bench_dataset/Robots_p/panda+allegro/tactile_sensor \
  --output-stem tactileSensor_map_RTH \
  --native-resolution 240 \
  --sensor-normal 1,0,0 \
  --mesh-units m \
  --origin-xyz 0,0,-0.012 \
  --origin-rpy 0,0,0 \
  --mesh-scale 1,1,1

python tools/asset/gen_tacmap_npy.py \
  --mesh ../dex2bench_dataset/Robots_p/panda+allegro/tactile_sensor/tactile_surface_R4F.stl \
  --output-dir ../dex2bench_dataset/Robots_p/panda+allegro/tactile_sensor \
  --output-stem tactileSensor_map_R4F \
  --native-resolution 240 \
  --sensor-normal 1,0,0 \
  --mesh-units m \
  --origin-xyz 0,0,-0.012 \
  --origin-rpy 0,0,0 \
  --mesh-scale 1,1,1

python tools/asset/gen_tacmap_npy.py \
  --mesh ../dex2bench_dataset/Robots_p/panda+allegro/tactile_sensor/tactile_surface_LTH.stl \
  --output-dir ../dex2bench_dataset/Robots_p/panda+allegro/tactile_sensor \
  --output-stem tactileSensor_map_LTH \
  --native-resolution 240 \
  --sensor-normal -1,0,0 \
  --mesh-units m \
  --origin-xyz 0,0,-0.012 \
  --origin-rpy 0,0,0 \
  --mesh-scale 1,1,1

python tools/asset/gen_tacmap_npy.py \
  --mesh ../dex2bench_dataset/Robots_p/panda+allegro/tactile_sensor/tactile_surface_L4F.stl \
  --output-dir ../dex2bench_dataset/Robots_p/panda+allegro/tactile_sensor \
  --output-stem tactileSensor_map_L4F \
  --native-resolution 240 \
  --sensor-normal -1,0,0 \
  --mesh-units m \
  --origin-xyz 0,0,-0.012 \
  --origin-rpy 0,0,0 \
  --mesh-scale 1,1,1
```

## 8. Orca

Dataset: `panda+orca`

Source mesh uses collision skin geometry:

- RTH: `meshes/collision/right_collision_thumb_dp_skin_mesh.stl`
- R4F: `meshes/collision/right_collision_index_ip_skin_mesh.stl`
- LTH: `meshes/collision/left_collision_thumb_dp_skin_mesh.stl`
- L4F: `meshes/collision/left_collision_index_ip_skin_mesh.stl`

Do not reuse one side's rebuilt mesh for the other side unless you explicitly mirrored it into that side's original mesh coordinate frame. The left/right origins and rotations differ.

```bash
python tools/asset/gen_tacmap_npy.py \
  --mesh ../dex2bench_dataset/Robots_p/panda+orca/tactile_sensor/tactile_surface_RTH.stl \
  --output-dir ../dex2bench_dataset/Robots_p/panda+orca/tactile_sensor \
  --output-stem tactileSensor_map_RTH \
  --native-resolution 240 \
  --sensor-normal auto \
  --mesh-units m \
  --origin-xyz 0.002785448,-0.000006944,0.016557497 \
  --origin-rpy 3.141592654,-0.396395308,0 \
  --mesh-scale 1,1,1

python tools/asset/gen_tacmap_npy.py \
  --mesh ../dex2bench_dataset/Robots_p/panda+orca/tactile_sensor/tactile_surface_R4F.stl \
  --output-dir ../dex2bench_dataset/Robots_p/panda+orca/tactile_sensor \
  --output-stem tactileSensor_map_R4F \
  --native-resolution 240 \
  --sensor-normal auto \
  --mesh-units m \
  --origin-xyz 0.000621674,0.000002433,0.023021050 \
  --origin-rpy 0,-0.445704230,0 \
  --mesh-scale 1,1,1

python tools/asset/gen_tacmap_npy.py \
  --mesh ../dex2bench_dataset/Robots_p/panda+orca/tactile_sensor/tactile_surface_LTH.stl \
  --output-dir ../dex2bench_dataset/Robots_p/panda+orca/tactile_sensor \
  --output-stem tactileSensor_map_LTH \
  --native-resolution 240 \
  --sensor-normal auto \
  --mesh-units m \
  --origin-xyz -0.002785448,-0.000006944,0.016557497 \
  --origin-rpy 0,-0.396395308,3.141592654 \
  --mesh-scale 1,1,1

python tools/asset/gen_tacmap_npy.py \
  --mesh ../dex2bench_dataset/Robots_p/panda+orca/tactile_sensor/tactile_surface_L4F.stl \
  --output-dir ../dex2bench_dataset/Robots_p/panda+orca/tactile_sensor \
  --output-stem tactileSensor_map_L4F \
  --native-resolution 240 \
  --sensor-normal auto \
  --mesh-units m \
  --origin-xyz -0.000621674,0.000002433,0.023021050 \
  --origin-rpy 0,0.445704230,0 \
  --mesh-scale 1,1,1
```

## 9. Ability

Dataset: `xarm+ability`

Source mesh:

- RTH: `models/thumb_F2_right.STL`
- R4F: `models/idx_F2_Lg.STL`
- LTH: `models/thumb_F2_left.STL`
- L4F: `models/idx_F2_Lg.STL`

```bash
python tools/asset/gen_tacmap_npy.py \
  --mesh ../dex2bench_dataset/Robots_p/xarm+ability/tactile_sensor/tactile_surface_RTH.stl \
  --output-dir ../dex2bench_dataset/Robots_p/xarm+ability/tactile_sensor \
  --output-stem tactileSensor_map_RTH \
  --native-resolution 240 \
  --sensor-normal auto \
  --mesh-units m \
  --origin-xyz 0,0,0 \
  --origin-rpy 0,0,0 \
  --mesh-scale 1,1,1

python tools/asset/gen_tacmap_npy.py \
  --mesh ../dex2bench_dataset/Robots_p/xarm+ability/tactile_sensor/tactile_surface_R4F.stl \
  --output-dir ../dex2bench_dataset/Robots_p/xarm+ability/tactile_sensor \
  --output-stem tactileSensor_map_R4F \
  --native-resolution 240 \
  --sensor-normal auto \
  --mesh-units m \
  --origin-xyz 0,0,0 \
  --origin-rpy 0,0,0 \
  --mesh-scale 1,1,1

python tools/asset/gen_tacmap_npy.py \
  --mesh ../dex2bench_dataset/Robots_p/xarm+ability/tactile_sensor/tactile_surface_LTH.stl \
  --output-dir ../dex2bench_dataset/Robots_p/xarm+ability/tactile_sensor \
  --output-stem tactileSensor_map_LTH \
  --native-resolution 240 \
  --sensor-normal auto \
  --mesh-units m \
  --origin-xyz 0,0,0 \
  --origin-rpy 0,0,0 \
  --mesh-scale 1,1,1

python tools/asset/gen_tacmap_npy.py \
  --mesh ../dex2bench_dataset/Robots_p/xarm+ability/tactile_sensor/tactile_surface_L4F.stl \
  --output-dir ../dex2bench_dataset/Robots_p/xarm+ability/tactile_sensor \
  --output-stem tactileSensor_map_L4F \
  --native-resolution 240 \
  --sensor-normal auto \
  --mesh-units m \
  --origin-xyz 0,0,0 \
  --origin-rpy 0,0,0 \
  --mesh-scale 1,1,1
```

## 10. Leap

Dataset: `xarm+leap`

Source mesh:

- RTH/LTH: `meshes/visual/thumb_fingertip.obj`
- R4F/L4F: `meshes/visual/fingertip.obj`

The `fingertip.obj` mesh has URDF visual origin `xyz="0.0132 -0.0061 0.0144"` and `rpy="3.1415926 0 0"`. If this transform is omitted, the generated surface can land on the wrong side of the fingertip.

```bash
python tools/asset/gen_tacmap_npy.py \
  --mesh ../dex2bench_dataset/Robots_p/xarm+leap/tactile_sensor/tactile_surface_RTH.stl \
  --output-dir ../dex2bench_dataset/Robots_p/xarm+leap/tactile_sensor \
  --output-stem tactileSensor_map_RTH \
  --native-resolution 240 \
  --sensor-normal auto \
  --mesh-units m \
  --origin-xyz 0.0625,0.0784,0.0489 \
  --origin-rpy 0,0,0 \
  --mesh-scale 1,1,1

python tools/asset/gen_tacmap_npy.py \
  --mesh ../dex2bench_dataset/Robots_p/xarm+leap/tactile_sensor/tactile_surface_R4F.stl \
  --output-dir ../dex2bench_dataset/Robots_p/xarm+leap/tactile_sensor \
  --output-stem tactileSensor_map_R4F \
  --native-resolution 240 \
  --sensor-normal auto \
  --mesh-units m \
  --origin-xyz 0.0132,-0.0061,0.0144 \
  --origin-rpy 3.1415926,0,0 \
  --mesh-scale 1,1,1

python tools/asset/gen_tacmap_npy.py \
  --mesh ../dex2bench_dataset/Robots_p/xarm+leap/tactile_sensor/tactile_surface_LTH.stl \
  --output-dir ../dex2bench_dataset/Robots_p/xarm+leap/tactile_sensor \
  --output-stem tactileSensor_map_LTH \
  --native-resolution 240 \
  --sensor-normal auto \
  --mesh-units m \
  --origin-xyz 0.0625,0.0784,0.0489 \
  --origin-rpy 0,0,0 \
  --mesh-scale 1,1,1

python tools/asset/gen_tacmap_npy.py \
  --mesh ../dex2bench_dataset/Robots_p/xarm+leap/tactile_sensor/tactile_surface_L4F.stl \
  --output-dir ../dex2bench_dataset/Robots_p/xarm+leap/tactile_sensor \
  --output-stem tactileSensor_map_L4F \
  --native-resolution 240 \
  --sensor-normal auto \
  --mesh-units m \
  --origin-xyz 0.0132,-0.0061,0.0144 \
  --origin-rpy 3.1415926,0,0 \
  --mesh-scale 1,1,1
```

If you keep the current Leap rebuilt mesh file names, use these equivalent input paths:

```text
RTH/LTH: ../dex2bench_dataset/Robots_p/xarm+leap/tactile_sensor/sensor_thumb_fingertip.obj
R4F/L4F: ../dex2bench_dataset/Robots_p/xarm+leap/tactile_sensor/sensor_fingertip.obj
```

Current workspace note:

- `sensor_fingertip.obj` still needs the R4F/L4F transform shown above. Without `--origin-rpy 3.1415926,0,0`, it lands on the wrong `+Y` side.
- `sensor_thumb_fingertip.obj` is already near the thumb attach-link local frame. For the current file, use `--origin-xyz 0,0,0 --origin-rpy 0,0,0`; applying `0.0625,0.0784,0.0489` again moves it to an obviously wrong offset around `x=0.07,y=0.08,z=0.05`.

Current workspace commands for these two file names:

```bash
python tools/asset/gen_tacmap_npy.py \
  --mesh ../dex2bench_dataset/Robots_p/xarm+leap/tactile_sensor/sensor_thumb_fingertip.obj \
  --output-dir ../dex2bench_dataset/Robots_p/xarm+leap/tactile_sensor \
  --output-stem tactileSensor_map_RTH \
  --native-resolution 240 \
  --sensor-normal auto \
  --mesh-units m \
  --origin-xyz 0,0,0 \
  --origin-rpy 0,0,0 \
  --mesh-scale 1,1,1

python tools/asset/gen_tacmap_npy.py \
  --mesh ../dex2bench_dataset/Robots_p/xarm+leap/tactile_sensor/sensor_fingertip.obj \
  --output-dir ../dex2bench_dataset/Robots_p/xarm+leap/tactile_sensor \
  --output-stem tactileSensor_map_R4F \
  --native-resolution 240 \
  --sensor-normal auto \
  --mesh-units m \
  --origin-xyz 0.0132,-0.0061,0.0144 \
  --origin-rpy 3.1415926,0,0 \
  --mesh-scale 1,1,1

python tools/asset/gen_tacmap_npy.py \
  --mesh ../dex2bench_dataset/Robots_p/xarm+leap/tactile_sensor/sensor_thumb_fingertip.obj \
  --output-dir ../dex2bench_dataset/Robots_p/xarm+leap/tactile_sensor \
  --output-stem tactileSensor_map_LTH \
  --native-resolution 240 \
  --sensor-normal auto \
  --mesh-units m \
  --origin-xyz 0,0,0 \
  --origin-rpy 0,0,0 \
  --mesh-scale 1,1,1

python tools/asset/gen_tacmap_npy.py \
  --mesh ../dex2bench_dataset/Robots_p/xarm+leap/tactile_sensor/sensor_fingertip.obj \
  --output-dir ../dex2bench_dataset/Robots_p/xarm+leap/tactile_sensor \
  --output-stem tactileSensor_map_L4F \
  --native-resolution 240 \
  --sensor-normal auto \
  --mesh-units m \
  --origin-xyz 0.0132,-0.0061,0.0144 \
  --origin-rpy 3.1415926,0,0 \
  --mesh-scale 1,1,1
```

## 11. DexHand021

Dataset: `jaka_zu7+dexhand021`

Source mesh:

- RTH: `meshes/hand/r_f_link1_4.STL`
- R4F: `meshes/hand/r_f_link2_4.STL`
- LTH: `meshes/hand/l_f_link1_4.STL`
- L4F: `meshes/hand/l_f_link2_4.STL`

Note: the left thumb link has `rpy="3.14 0 0"` in the URDF. Keep the LTH transform unless your rebuilt left thumb mesh is already exported in `l_f_link1_4` local coordinates.

```bash
python tools/asset/gen_tacmap_npy.py \
  --mesh ../dex2bench_dataset/Robots_p/jaka_zu7+dexhand021/tactile_sensor/tactile_surface_RTH.stl \
  --output-dir ../dex2bench_dataset/Robots_p/jaka_zu7+dexhand021/tactile_sensor \
  --output-stem tactileSensor_map_RTH \
  --native-resolution 240 \
  --sensor-normal auto \
  --mesh-units m \
  --origin-xyz 0,0,0 \
  --origin-rpy 0,0,0 \
  --mesh-scale 1,1,1

python tools/asset/gen_tacmap_npy.py \
  --mesh ../dex2bench_dataset/Robots_p/jaka_zu7+dexhand021/tactile_sensor/tactile_surface_R4F.stl \
  --output-dir ../dex2bench_dataset/Robots_p/jaka_zu7+dexhand021/tactile_sensor \
  --output-stem tactileSensor_map_R4F \
  --native-resolution 240 \
  --sensor-normal auto \
  --mesh-units m \
  --origin-xyz 0,0,0 \
  --origin-rpy 0,0,0 \
  --mesh-scale 1,1,1

python tools/asset/gen_tacmap_npy.py \
  --mesh ../dex2bench_dataset/Robots_p/jaka_zu7+dexhand021/tactile_sensor/tactile_surface_LTH.stl \
  --output-dir ../dex2bench_dataset/Robots_p/jaka_zu7+dexhand021/tactile_sensor \
  --output-stem tactileSensor_map_LTH \
  --native-resolution 240 \
  --sensor-normal auto \
  --mesh-units m \
  --origin-xyz 0,0,0 \
  --origin-rpy 3.14,0,0 \
  --mesh-scale 1,1,1

python tools/asset/gen_tacmap_npy.py \
  --mesh ../dex2bench_dataset/Robots_p/jaka_zu7+dexhand021/tactile_sensor/tactile_surface_L4F.stl \
  --output-dir ../dex2bench_dataset/Robots_p/jaka_zu7+dexhand021/tactile_sensor \
  --output-stem tactileSensor_map_L4F \
  --native-resolution 240 \
  --sensor-normal auto \
  --mesh-units m \
  --origin-xyz 0,0,0 \
  --origin-rpy 0,0,0 \
  --mesh-scale 1,1,1
```

## 12. RM65 + BrainCo Revo2

Dataset: `rm_65+BrainCo`

Source mesh:

- RTH: `meshes/revo2_right_hand/right_thumb_touch_link.STL`
- R4F: `meshes/revo2_right_hand/right_index_touch_link.STL`
- LTH: `meshes/revo2_left_hand/left_thumb_touch_link.STL`
- L4F: `meshes/revo2_left_hand/left_index_touch_link.STL`

Note: these four source links have zero visual/collision origin and no mesh scale in the URDF. Keep the zero transform below when the rebuilt mesh is still in the original source mesh coordinate system.

```bash
python tools/asset/gen_tacmap_npy.py \
  --mesh ../dex2bench_dataset/Robots_p/rm_65+BrainCo/tactile_sensor/tactile_surface_RTH.stl \
  --output-dir ../dex2bench_dataset/Robots_p/rm_65+BrainCo/tactile_sensor \
  --output-stem tactileSensor_map_RTH \
  --native-resolution 240 \
  --sensor-normal auto \
  --mesh-units m \
  --origin-xyz 0,0,0 \
  --origin-rpy 0,0,0 \
  --mesh-scale 1,1,1

python tools/asset/gen_tacmap_npy.py \
  --mesh ../dex2bench_dataset/Robots_p/rm_65+BrainCo/tactile_sensor/tactile_surface_R4F.stl \
  --output-dir ../dex2bench_dataset/Robots_p/rm_65+BrainCo/tactile_sensor \
  --output-stem tactileSensor_map_R4F \
  --native-resolution 240 \
  --sensor-normal auto \
  --mesh-units m \
  --origin-xyz 0,0,0 \
  --origin-rpy 0,0,0 \
  --mesh-scale 1,1,1

python tools/asset/gen_tacmap_npy.py \
  --mesh ../dex2bench_dataset/Robots_p/rm_65+BrainCo/tactile_sensor/tactile_surface_LTH.stl \
  --output-dir ../dex2bench_dataset/Robots_p/rm_65+BrainCo/tactile_sensor \
  --output-stem tactileSensor_map_LTH \
  --native-resolution 240 \
  --sensor-normal auto \
  --mesh-units m \
  --origin-xyz 0,0,0 \
  --origin-rpy 0,0,0 \
  --mesh-scale 1,1,1

python tools/asset/gen_tacmap_npy.py \
  --mesh ../dex2bench_dataset/Robots_p/rm_65+BrainCo/tactile_sensor/tactile_surface_L4F.stl \
  --output-dir ../dex2bench_dataset/Robots_p/rm_65+BrainCo/tactile_sensor \
  --output-stem tactileSensor_map_L4F \
  --native-resolution 240 \
  --sensor-normal auto \
  --mesh-units m \
  --origin-xyz 0,0,0 \
  --origin-rpy 0,0,0 \
  --mesh-scale 1,1,1
```

## 13. RM75 + RoHand

Dataset: `rm_75+rohand`

Source mesh:

- RTH: `meshes_r/th_distal_link.STL`
- R4F: `meshes_r/if_distal_link.STL`
- LTH: `meshes_l/th_distal_link.STL`
- L4F: `meshes_l/if_distal_link.STL`

Note: these four source links have zero visual/collision origin and no mesh scale in the URDF. Keep the zero transform below when the rebuilt mesh is still in the original source mesh coordinate system.

```bash
python tools/asset/gen_tacmap_npy.py \
  --mesh ../dex2bench_dataset/Robots_p/rm_75+rohand/tactile_sensor/tactile_surface_RTH.stl \
  --output-dir ../dex2bench_dataset/Robots_p/rm_75+rohand/tactile_sensor \
  --output-stem tactileSensor_map_RTH \
  --native-resolution 240 \
  --sensor-normal auto \
  --mesh-units m \
  --origin-xyz 0,0,0 \
  --origin-rpy 0,0,0 \
  --mesh-scale 1,1,1

python tools/asset/gen_tacmap_npy.py \
  --mesh ../dex2bench_dataset/Robots_p/rm_75+rohand/tactile_sensor/tactile_surface_R4F.stl \
  --output-dir ../dex2bench_dataset/Robots_p/rm_75+rohand/tactile_sensor \
  --output-stem tactileSensor_map_R4F \
  --native-resolution 240 \
  --sensor-normal auto \
  --mesh-units m \
  --origin-xyz 0,0,0 \
  --origin-rpy 0,0,0 \
  --mesh-scale 1,1,1

python tools/asset/gen_tacmap_npy.py \
  --mesh ../dex2bench_dataset/Robots_p/rm_75+rohand/tactile_sensor/tactile_surface_LTH.stl \
  --output-dir ../dex2bench_dataset/Robots_p/rm_75+rohand/tactile_sensor \
  --output-stem tactileSensor_map_LTH \
  --native-resolution 240 \
  --sensor-normal auto \
  --mesh-units m \
  --origin-xyz 0,0,0 \
  --origin-rpy 0,0,0 \
  --mesh-scale 1,1,1

python tools/asset/gen_tacmap_npy.py \
  --mesh ../dex2bench_dataset/Robots_p/rm_75+rohand/tactile_sensor/tactile_surface_L4F.stl \
  --output-dir ../dex2bench_dataset/Robots_p/rm_75+rohand/tactile_sensor \
  --output-stem tactileSensor_map_L4F \
  --native-resolution 240 \
  --sensor-normal auto \
  --mesh-units m \
  --origin-xyz 0,0,0 \
  --origin-rpy 0,0,0 \
  --mesh-scale 1,1,1
```

## Validation Recommendations

Perform at least these two checks after generating each map:

1. Check `hit_rate` in the output log; the default target is above `0.50`.
2. Use the visualization tool to confirm that the point cloud lies on the finger-pad contact surface, not on the back of the finger:

```bash
python tools/vis/vis_tacmap_npy.py \
  ../dex2bench_dataset/Robots_p/xarm+leap/tactile_sensor \
  --output /tmp/leap_tacmap_check.png
```

If rebuilding changed a mesh's orientation relative to the original, `--sensor-normal auto` may choose the wrong overall direction. In that case, rerun with an explicit direction, for example:

```bash
--sensor-normal 1,0,0
```

Or:

```bash
--sensor-normal -1,0,0
```
