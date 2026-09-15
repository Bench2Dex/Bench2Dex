"""Geometry and pose helpers for scene construction.

``rpy_deg_to_quat`` is the single canonical RPY-to-quaternion conversion used
throughout dex2scene (build, robots, collector).
"""

import math
from typing import Tuple

from pxr import Usd, UsdGeom

from .constants import Aabb

# --------------------------------------------------------------------------- #
# USD bounds with per-path cache (#13)
# --------------------------------------------------------------------------- #

_raw_bounds_cache: dict[str, Aabb | None] = {}


def _compute_bounds_uncached(usd_path: str) -> Aabb | None:
    """Open a USD stage and return its axis-aligned bounding box (unscaled)."""
    try:
        stage = Usd.Stage.Open(usd_path)
    except Exception as exc:
        print(f"[WARN] Cannot open '{usd_path}' for bounds: {exc}")
        return None
    if stage is None:
        print(f"[WARN] Cannot open '{usd_path}' for bounds.")
        return None

    up = UsdGeom.GetStageUpAxis(stage)
    if up != UsdGeom.Tokens.z:
        print(
            f"[WARN] '{usd_path}' has upAxis='{up}' (not Z). "
            "You likely need rpy_deg: [90,0,0] to rotate Y-up to Z-up."
        )

    root = stage.GetDefaultPrim()
    if not root or not root.IsValid():
        children = stage.GetPseudoRoot().GetChildren()
        root = children[0] if children else stage.GetPseudoRoot()

    purposes = [UsdGeom.Tokens.default_]
    for tok in ("render", "proxy"):
        if hasattr(UsdGeom.Tokens, tok):
            purposes.append(getattr(UsdGeom.Tokens, tok))

    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), purposes, useExtentsHint=True)
    bound = bbox.ComputeWorldBound(root)
    rng = (
        bound.ComputeAlignedRange()
        if hasattr(bound, "ComputeAlignedRange")
        else bound.GetRange()
    )
    lo = [float(rng.GetMin()[i]) for i in range(3)]
    hi = [float(rng.GetMax()[i]) for i in range(3)]

    if not all(math.isfinite(v) for v in lo + hi):
        print(f"[WARN] Non-finite bounds for '{usd_path}'.")
        return None

    return (lo[0], lo[1], lo[2]), (hi[0], hi[1], hi[2])


def _compute_bounds(
    usd_path: str, scale: Tuple[float, float, float] | None
) -> Aabb | None:
    """Return the AABB of a USD asset, applying *scale* if given.

    The raw (unscaled) result is cached by *usd_path* so that multiple
    objects referencing the same asset only open the stage once.
    """
    if usd_path not in _raw_bounds_cache:
        _raw_bounds_cache[usd_path] = _compute_bounds_uncached(usd_path)

    raw = _raw_bounds_cache[usd_path]
    if raw is None:
        return None

    lo_raw, hi_raw = raw
    lo = list(lo_raw)
    hi = list(hi_raw)

    if scale:
        for i in range(3):
            a, b = lo[i] * scale[i], hi[i] * scale[i]
            lo[i], hi[i] = min(a, b), max(a, b)

    return (lo[0], lo[1], lo[2]), (hi[0], hi[1], hi[2])


# --------------------------------------------------------------------------- #
# Canonical RPY (degrees) -> quaternion (w, x, y, z) conversion (#5)
# --------------------------------------------------------------------------- #

def rpy_deg_to_quat(rpy_deg: Tuple[float, float, float]) -> Tuple[float, float, float, float]:
    """Convert roll/pitch/yaw in **degrees** to a (w, x, y, z) quaternion."""
    r, p, y = (math.radians(v) for v in rpy_deg)
    cr, sr = math.cos(r * 0.5), math.sin(r * 0.5)
    cp, sp = math.cos(p * 0.5), math.sin(p * 0.5)
    cy, sy = math.cos(y * 0.5), math.sin(y * 0.5)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


_rpy_to_quat = rpy_deg_to_quat


def _quat_rotate(q: Tuple[float, float, float, float], p: Tuple[float, float, float]) -> Tuple[float, float, float]:
    w, qx, qy, qz = q
    px, py, pz = p
    t = 2.0 * (qx * px + qy * py + qz * pz)
    s = w * w - (qx * qx + qy * qy + qz * qz)
    c = 2.0 * w
    return (
        s * px + t * qx + c * (qy * pz - qz * py),
        s * py + t * qy + c * (qz * px - qx * pz),
        s * pz + t * qz + c * (qx * py - qy * px),
    )


def _rotate_bounds(
    lo: Tuple[float, float, float],
    hi: Tuple[float, float, float],
    rpy_deg: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    if all(v == 0 for v in rpy_deg):
        dx, dy = hi[0] - lo[0], hi[1] - lo[1]
        return lo[2], hi[2], 0.5 * math.hypot(dx, dy)

    q = _rpy_to_quat(rpy_deg)
    pts = [
        _quat_rotate(q, (x, y, z))
        for x in (lo[0], hi[0])
        for y in (lo[1], hi[1])
        for z in (lo[2], hi[2])
    ]
    xs = [v[0] for v in pts]
    ys = [v[1] for v in pts]
    zs = [v[2] for v in pts]
    return min(zs), max(zs), 0.5 * math.hypot(max(xs) - min(xs), max(ys) - min(ys))
