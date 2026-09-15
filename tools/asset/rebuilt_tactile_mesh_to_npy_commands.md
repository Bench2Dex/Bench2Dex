# Rebuilt Tactile Mesh -> TacMap NPY Commands

本文记录当前四指共用版 TacMap 配置下，每种手把重新构建的触觉接触面 mesh 转成 `.npy` 时应该使用的命令，重点记录 `--origin-xyz` / `--origin-rpy`。

当前文档里的命令只能直接对应现有运行时配置：拇指一组，其他非拇指手指共用一组。也就是说，Sharpa 是 `TH` / `4F`，其他双手机器人是 `RTH` / `R4F` / `LTH` / `L4F`。如果后续要做“五指独立版”，不能只把输出文件名改成五根手指；还需要同步补齐 `collector/tacmap_configs.py` 里的 `TacMapNpyGroup` 分组，并为每根手指分别生成 point/normal npy。

命令假设从仓库根目录执行：

```bash
cd .
```

命令里的输入 mesh 统一假设放在：

```text
../dex2bench_dataset/Robots_p/<dataset_key>/tactile_sensor/tactile_surface_<GROUP>.stl
```

如果你实际导出的是 `.obj`，或者沿用类似 `sensor_fingertip.obj` 的文件名，只改 `--mesh` 路径即可。

## 坐标系约定

这些命令适用于：你在 Blender 中从机器人 URDF 引用的原始 mesh 裁剪/重建出了新 mesh，但导出的新 mesh 仍处在“原始 mesh 文件坐标系”下。

`gen_tacmap_npy.py --mesh ...` 单 mesh 模式不会读取 URDF，所以这里必须手动补上 URDF `<visual>` / `<collision>` 上的：

- `origin xyz` -> `--origin-xyz`
- `origin rpy` -> `--origin-rpy`
- `mesh scale` -> `--mesh-scale`

如果你已经在 Blender 里把新 mesh 对齐并导出到了 TacMap attach link 的局部坐标系下，就不要再套这些 origin/rpy/scale，否则会变换两次。此时应使用：

```bash
--origin-xyz 0,0,0 --origin-rpy 0,0,0 --mesh-scale 1,1,1
```

## 输出命名

Sharpa 旧资产：

```text
tactileSensor_map_TH_point.npy
tactileSensor_map_TH_normal.npy
tactileSensor_map_4F_point.npy
tactileSensor_map_4F_normal.npy
```

其他机器人四组：

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

建议的五指独立版命名：

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

如果某个手只有四指，例如 Allegro / Leap，没有 little/pinky，就只保留 `TH` / `IDX` / `MID` / `RING`。如果左右手确实共用同一套 mesh 和 attach-link 局部坐标，也可以继续用不带 `R`/`L` 的 side-shared 命名；但只要左右 URDF link、origin、mesh 镜像关系不完全一致，就应使用上面的左右独立命名。

## 五指独立版还缺的信息

现有四指共用命令里的 `R4F` / `L4F` 只记录了一根代表性非拇指的 mesh/origin，一般是 index。五指独立版需要为每根手指补齐以下信息：

- `TacMapNpyGroup.name`：例如 `RIDX`、`RMID`、`RRING`、`RLIT`。
- `points_npy` / `normals_npy`：例如 `tactileSensor_map_RIDX_point.npy` 和 `tactileSensor_map_RIDX_normal.npy`。
- `site_names`：该 npy 绑定到哪个 tactile site，五指独立版通常每组只放一个 site。
- rebuilt mesh 路径：建议用 `tactile_surface_<GROUP>.stl`，例如 `tactile_surface_RMID.stl`。
- 原始 source mesh：从 URDF `<visual>` 或 `<collision>` 取对应手指，而不是沿用 index 的 mesh。
- `attach_link` / `source_link`：运行时 TacMap attach 的 link，以及实际取几何的 link；Allegro、Wuji 等存在 source link 需要映射到 tip link 的情况。
- `origin xyz` / `origin rpy` / `mesh scale`：每根手指都要按自己的 URDF geometry 记录，不能默认复用 `R4F` / `L4F`。
- `sensor-normal`：先用 `auto`，如果点云落到背面或 hit rate 异常，再记录该手指的显式方向。
- `geometry-role`：Orca 当前使用 collision skin mesh；其他默认 visual。五指独立时也要保持这个选择。

已知不能盲目复用 `R4F` / `L4F` 的情况：

- RH56DFX、RH5DG2、Wuji、Orca、DexHand021 的非拇指通常都有每根手指自己的 mesh 文件。
- LEAP 的 index fingertip 和 middle/ring fingertip 使用同一个 `fingertip.obj`，但 URDF visual origin 存在细微差异，不能只复制 index 的 `origin-xyz`。
- Ability 的 right pinky 使用的 source mesh 与其他几个非拇指不完全一致。
- Shadow、Schunk、Allegro 这类几何文件高度复用的手，也仍然要按每根手指绑定独立 site/link，否则运行时还是四指共用。

运行时代码也要同步改，否则新文件不会被加载：

1. 在 `collector/tacmap_configs.py` 中把 `_split_side_groups(...)` 生成的 `R4F` / `L4F` 聚合组拆成每根手指一个 `TacMapNpyGroup`。
2. 确认 `collector/tactile.py` 里的 site 名称与新 group 一一对应。
3. 重新生成对应 `.npy` 文件后，再用 `tools/vis/vis_tacmap_npy.py` 检查每个 group 的点云。

五指独立版 group 到 site 的建议映射如下：

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
| Sharpa | 当前是 side-shared `TH` / `4F` 旧资产；如果拆五指，建议先确认左右 elastomer mesh 是否完全共用，再决定使用 `TH/IDX/MID/RING/LIT` 还是左右独立的 `R*` / `L*` 命名。 |

五指独立版单 mesh 命令模板：

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

## 验证建议

每次生成后至少做两步检查：

1. 看输出日志里的 `hit_rate`，默认应大于 `0.50`。
2. 用可视化工具检查点云是否落在 finger pad 的接触面，而不是手指背面：

```bash
python tools/vis/vis_tacmap_npy.py \
  ../dex2bench_dataset/Robots_p/xarm+leap/tactile_sensor \
  --output /tmp/leap_tacmap_check.png
```

如果某组 mesh 因为重建后方向和原始 mesh 不一致，`--sensor-normal auto` 可能会选错宏观方向。此时优先用显式方向重跑，例如：

```bash
--sensor-normal 1,0,0
```

或者：

```bash
--sensor-normal -1,0,0
```
