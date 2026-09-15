# 条件评估器 (condition_evaluator.py) 参考文档

本文档详细说明成功条件评估系统中所有原子条件判断函数的含义和用途。

---

## 一、辅助函数

| 函数 | 作用 | 计算方式 |
|------|------|----------|
| `_get_pos(states, obj_id)` | 获取物体的世界坐标 | 返回 `(x, y, z)` |
| `_dist_xy(states, a, b)` | 计算两物体的水平距离 | `√((x₁-x₂)² + (y₁-y₂)²)` |
| `_dist_3d(states, a, b)` | 计算两物体的三维空间距离 | `√((x₁-x₂)² + (y₁-y₂)² + (z₁-z₂)²)` |

---

## 二、原子条件判断函数

### 1. object_inside - 物体放入容器

**用途**：判断物体是否在容器内部

**判断逻辑**：
- xy平面距离 < tolerance
- 物体z高度 > 容器z高度 - 0.05（物体在容器内或上方）

**参数**：
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `object` | string | 必填 | 物体ID |
| `container` | string | 必填 | 容器ID |
| `tolerance` | float | 0.12 | xy距离容差（米） |

**示例**：
```python
{'type': 'object_inside', 'object': 'obj_spoon', 'container': 'obj_bowl', 'tolerance': 0.15}
```
判断勺子是否放入碗里。

---

### 2. object_on - 物体放在目标上方

**用途**：判断物体是否放在目标物体上面

**判断逻辑**：
- xy平面距离 < tolerance_xy
- 0 < z差 < tolerance_z（物体在目标正上方，有一定高度差但不太高）

**参数**：
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `object` | string | 必填 | 物体ID |
| `target` | string | 必填 | 目标物体ID |
| `tolerance_xy` | float | 0.12 | xy距离容差（米） |
| `tolerance_z` | float | 0.10 | z高度差容差（米） |

**示例**：
```python
{'type': 'object_on', 'object': 'obj_book', 'target': 'obj_table', 'tolerance_xy': 0.15, 'tolerance_z': 0.12}
```
判断书是否放在桌子上。

---

### 3. object_near - 物体靠近目标

**用途**：判断物体是否靠近目标物体

**判断逻辑**：
- 三维空间距离 < threshold

**参数**：
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `object` | string | 必填 | 物体ID |
| `target` | string | 必填 | 目标物体ID |
| `threshold` | float | 0.25 | 距离阈值（米） |

**示例**：
```python
{'type': 'object_near', 'object': 'obj_remote', 'target': 'obj_tv', 'threshold': 0.30}
```
判断遥控器是否靠近电视。

---

### 4. object_in_zone - 物体在区域内

**用途**：判断物体是否在指定的AABB（轴对齐包围盒）区域内

**判断逻辑**：
- 物体坐标在 `[lo, hi]` 范围内

**参数**：
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `object` | string | 必填 | 物体ID |
| `aabb` | tuple | 必填 | `([x_min, y_min, z_min], [x_max, y_max, z_max])` |

**示例**：
```python
{'type': 'object_in_zone', 'object': 'obj_block', 'aabb': [[-0.5, -0.3, 0.75], [0.5, 0.3, 1.0]]}
```
判断物体是否在指定区域范围内。

---

### 5. object_upright - 物体竖直放置

**用途**：判断物体是否竖直放置（局部z轴朝上）

**判断逻辑**：
- 物体局部z轴与世界z轴的点积 > cos(容差角度)

**参数**：
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `object` | string | 必填 | 物体ID |
| `tolerance_deg` | float | 25 | 角度容差（度） |

**示例**：
```python
{'type': 'object_upright', 'object': 'obj_cup', 'tolerance_deg': 30}
```
判断杯子是否竖直放置，允许30度倾斜。

---

### 6. object_flipped - 物体翻转/倒置

**用途**：判断物体是否倒置（局部z轴朝下）

**判断逻辑**：
- 物体局部z轴与世界z轴的点积 < -cos(容差角度)

**参数**：
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `object` | string | 必填 | 物体ID |
| `tolerance_deg` | float | 25 | 角度容差（度） |

**示例**：
```python
{'type': 'object_flipped', 'object': 'obj_calculator', 'tolerance_deg': 30}
```
判断计算器是否翻面倒置。

---

### 7. object_orientation - 物体特定朝向

**用途**：判断物体某个轴是否指向特定方向

**判断逻辑**：
- 指定轴与目标方向的点积 > cos(容差角度)

**参数**：
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `object` | string | 必填 | 物体ID |
| `local_axis` | string | "z" | 局部轴（"x"/"y"/"z"） |
| `target_direction` | list | [0, 0, -1] | 目标方向向量 |
| `tolerance_deg` | float | 30 | 角度容差（度） |

**示例**：
```python
{'type': 'object_orientation', 'object': 'obj_knife', 'local_axis': 'z', 'target_direction': [0, 0, -1], 'tolerance_deg': 35}
```
判断刀具的z轴是否朝下（刀尖朝下）。

---

### 8. relative_position - 相对位置关系

**用途**：判断物体相对参考物的空间位置关系

**判断逻辑**：
- 根据relation类型比较坐标

**参数**：
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `object` | string | 必填 | 物体ID |
| `reference` | string | 必填 | 参考物体ID |
| `relation` | string | 必填 | 位置关系 |
| `margin` | float | 0.02 | 边界裕量（米） |

**支持的relation类型**：
| 关系 | 判断条件 |
|------|----------|
| `left` | object.y > reference.y + margin |
| `right` | object.y < reference.y - margin |
| `front` | object.x < reference.x - margin |
| `behind` | object.x > reference.x + margin |
| `above` | object.z > reference.z + margin |
| `below` | object.z < reference.z - margin |

**示例**：
```python
{'type': 'relative_position', 'object': 'obj_bowl', 'reference': 'obj_pot', 'relation': 'left', 'margin': 0.02}
```
判断碗是否在锅的左边。

---

### 9. joint_state - 关节状态

**用途**：判断关节是否处于目标状态

**判断逻辑**：
- `target="open"`：关节位置绝对值 > tolerance
- `target="closed"`：关节位置绝对值 < tolerance
- `target=数值`：关节位置接近该数值

**参数**：
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `object` | string | 必填 | 物体ID |
| `joint` | string | "joint_0" | 关节名称 |
| `target` | string/float | 必填 | 目标状态 |
| `tolerance` | float | 0.15 | 位置容差 |

**示例**：
```python
{'type': 'joint_state', 'object': 'obj_drawer', 'joint': 'joint_0', 'target': 'open', 'tolerance': 0.10}
```
判断抽屉是否打开。

```python
{'type': 'joint_state', 'object': 'obj_microwave', 'joint': 'joint_1', 'target': 'closed', 'tolerance': 0.15}
```
判断微波炉门是否关闭。

---

### 10. object_lifted - 物体被抬起

**用途**：判断物体是否被抬高到指定高度以上

**判断逻辑**：
- 物体z高度 > min_z

**参数**：
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `object` | string | 必填 | 物体ID |
| `min_z` | float | 0.80 | 最小高度阈值（米） |

**示例**：
```python
{'type': 'object_lifted', 'object': 'obj_briefcase', 'min_z': 0.80}
```
判断公文包是否被抬起到0.80米以上。

---

### 11. height_order - 高度顺序

**用途**：判断多个物体是否按高度顺序排列

**判断逻辑**：
- 每个物体的z ≤ 下一个物体的z + 0.005

**参数**：
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `objects` | list | 必填 | 物体ID列表（从低到高） |

**示例**：
```python
{'type': 'height_order', 'objects': ['obj_cube_1', 'obj_cube_2', 'obj_cube_3']}
```
判断三个立方体是否从下到上堆叠。

---

### 12. object_static - 物体静止

**用途**：判断物体是否静止不动

**判断逻辑**：
- 线速度 < threshold
- 角速度 < threshold * 5

**参数**：
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `object` | string | 必填 | 物体ID |
| `threshold` | float | 0.05 | 速度阈值 |

**示例**：
```python
{'type': 'object_static', 'object': 'obj_ball', 'threshold': 0.05}
```
判断球是否静止不动。

---

## 三、逻辑组合类型

### 1. all - 所有条件都满足

```python
{
    'type': 'all',
    'conditions': [
        {'type': 'object_inside', 'object': 'obj_a', 'container': 'obj_b'},
        {'type': 'object_upright', 'object': 'obj_c'},
    ]
}
```

### 2. any - 任一条件满足

```python
{
    'type': 'any',
    'conditions': [
        {'type': 'object_near', 'object': 'obj_a', 'target': 'obj_b'},
        {'type': 'object_near', 'object': 'obj_a', 'target': 'obj_c'},
    ]
}
```

### 3. sequence - 按顺序完成各步骤

```python
{
    'type': 'sequence',
    'steps': [
        {'type': 'joint_state', 'object': 'obj_door', 'target': 'open'},
        {'type': 'object_near', 'object': 'obj_item', 'target': 'obj_door'},
        {'type': 'joint_state', 'object': 'obj_door', 'target': 'closed'},
    ]
}
```

### 4. not - 条件取反

```python
{
    'type': 'not',
    'condition': {'type': 'object_static', 'object': 'obj_ball'}
}
```

### 5. hold_duration - 条件持续满足指定时间

```python
{
    'type': 'hold_duration',
    'condition': {'type': 'object_lifted', 'object': 'obj_plate', 'min_z': 0.82},
    'seconds': 3.0
}
```
判断盘子被抬起并保持3秒。

---

## 四、完整示例

### 示例1：餐具放入碗

```python
CONDITIONS = [{
    'type': 'all',
    'conditions': [
        {'type': 'object_inside', 'object': 'obj_spoon', 'container': 'obj_bowl', 'tolerance': 0.15},
        {'type': 'object_inside', 'object': 'obj_fork', 'container': 'obj_bowl', 'tolerance': 0.15},
        {'type': 'object_inside', 'object': 'obj_knife', 'container': 'obj_bowl', 'tolerance': 0.15},
        {'type': 'object_orientation', 'object': 'obj_knife', 'local_axis': 'z', 'target_direction': [0, 0, -1], 'tolerance_deg': 45},
    ]
}]
```

### 示例2：打开抽屉放入物品

```python
CONDITIONS = [{
    'type': 'sequence',
    'steps': [
        {'type': 'joint_state', 'object': 'obj_drawer', 'joint': 'joint_0', 'target': 'open'},
        {'type': 'object_inside', 'object': 'obj_block', 'container': 'obj_drawer', 'tolerance': 0.20},
    ]
}]
```

### 示例3：抬起并保持平衡

```python
CONDITIONS = [{
    'type': 'all',
    'conditions': [
        {'type': 'object_upright', 'object': 'obj_glass_1', 'tolerance_deg': 30},
        {'type': 'object_upright', 'object': 'obj_glass_2', 'tolerance_deg': 30},
        {'type': 'hold_duration', 'seconds': 2.5, 'condition': {'type': 'object_lifted', 'object': 'obj_tray', 'min_z': 0.80}},
    ]
}]
```

---

## 五、距离类型对比

| 距离类型 | 计算方式 | 适用场景 |
|----------|----------|----------|
| xy距离 | `√((x₁-x₂)² + (y₁-y₂)²)` | 判断水平位置关系，忽略高度 |
| z差 | `z₁ - z₂` | 判断高度关系 |
| 3D距离 | `√((x₁-x₂)² + (y₁-y₂)² + (z₁-z₂)²)` | 判断空间位置关系 |

**选择建议**：
- 放入容器：使用xy距离 + z条件
- 放在物体上：使用xy距离 + z差
- 靠近物体：使用3D距离
- 相对位置：使用坐标比较