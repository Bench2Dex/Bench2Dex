# Condition Evaluator (`condition_evaluator.py`) Reference

This document describes the meaning and use of each atomic predicate in the success-condition evaluation system.

---

## 1. Helper Functions

| Function | Purpose | Computation |
|------|------|----------|
| `_get_pos(states, obj_id)` | Gets an object's world coordinates | Returns `(x, y, z)` |
| `_dist_xy(states, a, b)` | Computes the horizontal distance between two objects | `√((x₁-x₂)² + (y₁-y₂)²)` |
| `_dist_3d(states, a, b)` | Computes the 3D distance between two objects | `√((x₁-x₂)² + (y₁-y₂)² + (z₁-z₂)²)` |

---

## 2. Atomic Predicates

### 1. `object_inside` — Object Inside a Container

**Purpose**: Checks whether an object is inside a container.

**Logic**:
- Horizontal (XY-plane) distance < `tolerance`
- Object Z coordinate > container Z coordinate - 0.05 (the object is inside or above the container).

**Parameters**:
| Parameter | Type | Default | Description |
|------|------|--------|------|
| `object` | string | Required | Object ID |
| `container` | string | Required | Container ID |
| `tolerance` | float | 0.12 | XY-plane distance tolerance (m) |

**Example**:
```python
{'type': 'object_inside', 'object': 'obj_spoon', 'container': 'obj_bowl', 'tolerance': 0.15}
```
Checks whether the spoon is in the bowl.

---

### 2. `object_on` — Object on a Target

**Purpose**: Checks whether an object is positioned on top of a target object.

**Logic**:
- Horizontal (XY-plane) distance < `tolerance_xy`
- `0 < Z difference < tolerance_z` (the object is above the target, with a positive but limited height difference).

**Parameters**:
| Parameter | Type | Default | Description |
|------|------|--------|------|
| `object` | string | Required | Object ID |
| `target` | string | Required | Target object ID |
| `tolerance_xy` | float | 0.12 | XY-plane distance tolerance (m) |
| `tolerance_z` | float | 0.10 | Z-height difference tolerance (m) |

**Example**:
```python
{'type': 'object_on', 'object': 'obj_book', 'target': 'obj_table', 'tolerance_xy': 0.15, 'tolerance_z': 0.12}
```
Checks whether the book is on the table.

---

### 3. `object_near` — Object Near a Target

**Purpose**: Checks whether an object is near a target object.

**Logic**:
- 3D distance < `threshold`

**Parameters**:
| Parameter | Type | Default | Description |
|------|------|--------|------|
| `object` | string | Required | Object ID |
| `target` | string | Required | Target object ID |
| `threshold` | float | 0.25 | Distance threshold (m) |

**Example**:
```python
{'type': 'object_near', 'object': 'obj_remote', 'target': 'obj_tv', 'threshold': 0.30}
```
Checks whether the remote is near the TV.

---

### 4. `object_in_zone` — Object Inside a Zone

**Purpose**: Checks whether an object is inside the specified AABB (axis-aligned bounding box).

**Logic**:
- Object coordinates lie within the `[lo, hi]` bounds.

**Parameters**:
| Parameter | Type | Default | Description |
|------|------|--------|------|
| `object` | string | Required | Object ID |
| `aabb` | tuple | Required | `([x_min, y_min, z_min], [x_max, y_max, z_max])` |

**Example**:
```python
{'type': 'object_in_zone', 'object': 'obj_block', 'aabb': [[-0.5, -0.3, 0.75], [0.5, 0.3, 1.0]]}
```
Checks whether the object lies in the specified zone.

---

### 5. `object_upright` — Object Upright

**Purpose**: Checks whether an object is upright (its local Z axis points upward).

**Logic**:
- Dot product of the object's local Z axis and the world Z axis > cosine of the tolerance angle.

**Parameters**:
| Parameter | Type | Default | Description |
|------|------|--------|------|
| `object` | string | Required | Object ID |
| `tolerance_deg` | float | 25 | Angular tolerance (degrees) |

**Example**:
```python
{'type': 'object_upright', 'object': 'obj_cup', 'tolerance_deg': 30}
```
Checks whether the cup is upright, allowing up to 30 degrees of tilt.

---

### 6. `object_flipped` — Object Flipped or Upside Down

**Purpose**: Checks whether an object is upside down (its local Z axis points downward).

**Logic**:
- Dot product of the object's local Z axis and the world Z axis < negative cosine of the tolerance angle.

**Parameters**:
| Parameter | Type | Default | Description |
|------|------|--------|------|
| `object` | string | Required | Object ID |
| `tolerance_deg` | float | 25 | Angular tolerance (degrees) |

**Example**:
```python
{'type': 'object_flipped', 'object': 'obj_calculator', 'tolerance_deg': 30}
```
Checks whether the calculator is flipped upside down.

---

### 7. `object_orientation` — Object Orientation

**Purpose**: Checks whether a specified object axis points in a target direction.

**Logic**:
- Dot product of the selected axis and target direction > cosine of the tolerance angle.

**Parameters**:
| Parameter | Type | Default | Description |
|------|------|--------|------|
| `object` | string | Required | Object ID |
| `local_axis` | string | "z" | Local axis (`"x"`, `"y"`, or `"z"`) |
| `target_direction` | list | [0, 0, -1] | Target direction vector |
| `tolerance_deg` | float | 30 | Angular tolerance (degrees) |

**Example**:
```python
{'type': 'object_orientation', 'object': 'obj_knife', 'local_axis': 'z', 'target_direction': [0, 0, -1], 'tolerance_deg': 35}
```
Checks whether the knife's Z axis points downward (blade tip down).

---

### 8. `relative_position` — Relative Position

**Purpose**: Checks an object's spatial position relative to a reference object.

**Logic**:
- Compares coordinates according to the specified `relation`.

**Parameters**:
| Parameter | Type | Default | Description |
|------|------|--------|------|
| `object` | string | Required | Object ID |
| `reference` | string | Required | Reference object ID |
| `relation` | string | Required | Spatial relation |
| `margin` | float | 0.02 | Boundary margin (m) |

**Supported relation values**:
| Relation | Condition |
|------|----------|
| `left` | object.y > reference.y + margin |
| `right` | object.y < reference.y - margin |
| `front` | object.x < reference.x - margin |
| `behind` | object.x > reference.x + margin |
| `above` | object.z > reference.z + margin |
| `below` | object.z < reference.z - margin |

**Example**:
```python
{'type': 'relative_position', 'object': 'obj_bowl', 'reference': 'obj_pot', 'relation': 'left', 'margin': 0.02}
```
Checks whether the bowl is to the left of the pot.

---

### 9. `joint_state` — Joint State

**Purpose**: Checks whether a joint is in the target state.

**Logic**:
- `target="open"`: absolute joint position > `tolerance`.
- `target="closed"`: absolute joint position < `tolerance`.
- Numeric `target`: joint position is close to the specified value.

**Parameters**:
| Parameter | Type | Default | Description |
|------|------|--------|------|
| `object` | string | Required | Object ID |
| `joint` | string | "joint_0" | Joint name |
| `target` | string/float | Required | Target state |
| `tolerance` | float | 0.15 | Position tolerance |

**Example**:
```python
{'type': 'joint_state', 'object': 'obj_drawer', 'joint': 'joint_0', 'target': 'open', 'tolerance': 0.10}
```
Checks whether the drawer is open.

```python
{'type': 'joint_state', 'object': 'obj_microwave', 'joint': 'joint_1', 'target': 'closed', 'tolerance': 0.15}
```
Checks whether the microwave door is closed.

---

### 10. `object_lifted` — Object Lifted

**Purpose**: Checks whether an object has been lifted above a specified height.

**Logic**:
- Object Z coordinate > `min_z`.

**Parameters**:
| Parameter | Type | Default | Description |
|------|------|--------|------|
| `object` | string | Required | Object ID |
| `min_z` | float | 0.80 | Minimum height threshold (m) |

**Example**:
```python
{'type': 'object_lifted', 'object': 'obj_briefcase', 'min_z': 0.80}
```
Checks whether the briefcase has been lifted above 0.80 m.

---

### 11. `height_order` — Height Order

**Purpose**: Checks whether multiple objects are ordered by height.

**Logic**:
- Each object's Z coordinate ≤ the next object's Z coordinate + 0.005.

**Parameters**:
| Parameter | Type | Default | Description |
|------|------|--------|------|
| `objects` | list | Required | List of object IDs, ordered from lowest to highest |

**Example**:
```python
{'type': 'height_order', 'objects': ['obj_cube_1', 'obj_cube_2', 'obj_cube_3']}
```
Checks whether the three cubes are stacked from bottom to top.

---

### 12. `object_static` — Object Static

**Purpose**: Checks whether an object is stationary.

**Logic**:
- Linear speed < `threshold`.
- Angular speed < `threshold * 5`.

**Parameters**:
| Parameter | Type | Default | Description |
|------|------|--------|------|
| `object` | string | Required | Object ID |
| `threshold` | float | 0.05 | Speed threshold |

**Example**:
```python
{'type': 'object_static', 'object': 'obj_ball', 'threshold': 0.05}
```
Checks whether the ball is stationary.

---

## 3. Logical Combinators

### 1. `all` — All Conditions Must Be Met

```python
{
    'type': 'all',
    'conditions': [
        {'type': 'object_inside', 'object': 'obj_a', 'container': 'obj_b'},
        {'type': 'object_upright', 'object': 'obj_c'},
    ]
}
```

### 2. `any` — At Least One Condition Must Be Met

```python
{
    'type': 'any',
    'conditions': [
        {'type': 'object_near', 'object': 'obj_a', 'target': 'obj_b'},
        {'type': 'object_near', 'object': 'obj_a', 'target': 'obj_c'},
    ]
}
```

### 3. `sequence` — Complete Steps in Order

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

### 4. `not` — Negate a Condition

```python
{
    'type': 'not',
    'condition': {'type': 'object_static', 'object': 'obj_ball'}
}
```

### 5. `hold_duration` — Condition Must Hold for a Duration

```python
{
    'type': 'hold_duration',
    'condition': {'type': 'object_lifted', 'object': 'obj_plate', 'min_z': 0.82},
    'seconds': 3.0
}
```
Checks whether the plate remains lifted for 3 seconds.

---

## 4. Complete Examples

### Example 1: Place Utensils in a Bowl

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

### Example 2: Open a Drawer and Place an Item Inside

```python
CONDITIONS = [{
    'type': 'sequence',
    'steps': [
        {'type': 'joint_state', 'object': 'obj_drawer', 'joint': 'joint_0', 'target': 'open'},
        {'type': 'object_inside', 'object': 'obj_block', 'container': 'obj_drawer', 'tolerance': 0.20},
    ]
}]
```

### Example 3: Lift and Hold a Tray Level

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

## 5. Distance Types

| Distance type | Formula | Use |
|----------|----------|----------|
| XY distance | `√((x₁-x₂)² + (y₁-y₂)²)` | Horizontal position; ignores height |
| Z difference | `z₁ - z₂` | Height relationship |
| 3D distance | `√((x₁-x₂)² + (y₁-y₂)² + (z₁-z₂)²)` | Spatial relationship |

**Guidance**:
- Inside a container: use XY distance and a Z condition.
- On top of an object: use XY distance and Z difference.
- Near an object: use 3D distance.
- Relative position: compare coordinates.
