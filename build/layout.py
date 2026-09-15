"""Layout sampling and collision-avoidance helpers."""

from utils.seed_policy import get_py_rng as _rng
from typing import Any, List, Sequence, Tuple

from .constants import Aabb, MAX_PLACE_ATTEMPTS, MIN_SEPARATION


def _normalize_aabb(aabb_like) -> Aabb | None:
    if not aabb_like or len(aabb_like) != 2:
        return None
    p0 = aabb_like[0]
    p1 = aabb_like[1]
    if len(p0) != 3 or len(p1) != 3:
        return None

    x0, y0, z0 = (float(v) for v in p0)
    x1, y1, z1 = (float(v) for v in p1)
    return (
        (min(x0, x1), min(y0, y1), min(z0, z1)),
        (max(x0, x1), max(y0, y1), max(z0, z1)),
    )


def _normalize_aabbs(items) -> List[Aabb]:
    out: List[Aabb] = []
    if not items:
        return out
    for item in items:
        norm = _normalize_aabb(item)
        if norm is not None:
            out.append(norm)
    return out


def _inset_aabb_xy(aabb: Aabb, margin: float) -> Aabb | None:
    (x0, y0, z0), (x1, y1, z1) = aabb
    margin = max(0.0, float(margin))
    inset = ((x0 + margin, y0 + margin, z0), (x1 - margin, y1 - margin, z1))
    if inset[0][0] > inset[1][0] or inset[0][1] > inset[1][1]:
        return None
    return inset


def _resolve_explicit_position(
    position_like: Any,
    *,
    obj_id: str,
    table_z: float,
    min_z: float,
) -> Tuple[float, float, float]:
    if isinstance(position_like, (str, bytes)) or not isinstance(position_like, Sequence):
        raise TypeError(f"Object '{obj_id}' position must be a sequence of length 2 or 3.")

    values = list(position_like)
    if len(values) not in (2, 3):
        raise ValueError(f"Object '{obj_id}' position must have length 2 or 3.")

    try:
        x = float(values[0])
        y = float(values[1])
        if len(values) == 3:
            # All 3-element z values are treated as offsets from the table surface.
            z = table_z + float(values[2])
        else:
            # 2-element: only xy given; z defaults to the table surface.
            z = table_z - min_z
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Object '{obj_id}' position values must be numeric.") from exc

    return (x, y, z)


def _explicit_position_candidate_is_valid(
    cx: float,
    cy: float,
    *,
    aabb: Aabb | None,
    footprint: float,
    placed: List[Tuple[str, float, float, float]],
    forbidden_aabbs: Sequence[Aabb],
) -> bool:
    if aabb is not None:
        (x0, y0, _z0), (x1, y1, _z1) = aabb
        if cx < x0 or cx > x1 or cy < y0 or cy > y1:
            return False

    if _hits_forbidden(cx, cy, footprint, forbidden_aabbs):
        return False

    for _, px, py, pr in placed:
        gap = footprint + pr + MIN_SEPARATION
        if (cx - px) ** 2 + (cy - py) ** 2 < gap ** 2:
            return False

    return True


def _sample_explicit_position(
    anchor_pos: Tuple[float, float, float],
    *,
    obj_id: str,
    jitter_xy_m: float,
    aabb: Aabb | None,
    footprint: float,
    placed: List[Tuple[str, float, float, float]],
    forbidden_aabbs: Sequence[Aabb],
    allow_anchor_fallback: bool = False,
) -> Tuple[float, float, float]:
    if jitter_xy_m <= 0.0:
        return anchor_pos

    anchor_x, anchor_y, anchor_z = anchor_pos
    attempted_radii = []
    for scale in (1.0, 0.5, 0.25):
        radius = jitter_xy_m * scale
        if radius <= 0.0 or any(abs(radius - seen) <= 1e-9 for seen in attempted_radii):
            continue
        attempted_radii.append(radius)
        for _ in range(MAX_PLACE_ATTEMPTS):
            cx = anchor_x + _rng().uniform(-radius, radius)
            cy = anchor_y + _rng().uniform(-radius, radius)
            if _explicit_position_candidate_is_valid(
                cx,
                cy,
                aabb=aabb,
                footprint=footprint,
                placed=placed,
                forbidden_aabbs=forbidden_aabbs,
            ):
                return (cx, cy, anchor_z)

    if _explicit_position_candidate_is_valid(
        anchor_x,
        anchor_y,
        aabb=aabb,
        footprint=footprint,
        placed=placed,
        forbidden_aabbs=forbidden_aabbs,
    ):
        return anchor_pos

    jitter_cm = jitter_xy_m * 100.0
    if allow_anchor_fallback:
        print(
            f"[WARN] Explicit-position jitter fallback to anchor for object '{obj_id}' "
            f"after exhausting local samples within x/y +/-{jitter_cm:.2f}cm around "
            f"({anchor_x:.4f}, {anchor_y:.4f})."
        )
        return anchor_pos

    raise ValueError(
        f"Object '{obj_id}' has no valid explicit-position jitter sample within "
        f"x/y +/-{jitter_cm:.2f}cm around ({anchor_x:.4f}, {anchor_y:.4f})."
    )


def _hits_forbidden(
    cx: float,
    cy: float,
    footprint: float,
    forbidden_aabbs: Sequence[Aabb],
) -> bool:
    expand = footprint + MIN_SEPARATION
    for (x0, y0, _z0), (x1, y1, _z1) in forbidden_aabbs:
        if (x0 - expand) <= cx <= (x1 + expand) and (y0 - expand) <= cy <= (y1 + expand):
            return True
    return False


def _sample_position(
    aabb: Aabb,
    table_z: float,
    min_z: float,
    footprint: float,
    placed: List[Tuple[str, float, float, float]],
    yaw_range: Tuple[float, float],
    forbidden_aabbs: Sequence[Aabb],
) -> Tuple[Tuple[float, float, float], float]:
    (x0, y0, _z0), (x1, y1, _z1) = aabb
    valid_xy = None

    for _ in range(MAX_PLACE_ATTEMPTS):
        cx = _rng().uniform(x0, x1)
        cy = _rng().uniform(y0, y1)

        if _hits_forbidden(cx, cy, footprint, forbidden_aabbs):
            continue

        collides = False
        for _, px, py, pr in placed:
            gap = footprint + pr + MIN_SEPARATION
            if (cx - px) ** 2 + (cy - py) ** 2 < gap ** 2:
                collides = True
                break

        if not collides:
            valid_xy = (cx, cy)
            break

    if valid_xy is None:
        raise ValueError(
            f"Failed to find a valid placement after {MAX_PLACE_ATTEMPTS} attempts "
            f"(zone=[({x0:.3f},{y0:.3f})-({x1:.3f},{y1:.3f})], footprint={footprint:.4f}, "
            f"placed={len(placed)} objects, forbidden_aabbs={len(forbidden_aabbs)})."
        )

    z = table_z - min_z
    yaw = _rng().uniform(yaw_range[0], yaw_range[1])
    return (valid_xy[0], valid_xy[1], z), yaw
