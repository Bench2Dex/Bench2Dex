"""Active-DOF utilities for robot joint filtering and mimic expansion.

Given a robot_key, provides:
  - active_indices: which full-DOF joints are independently actuated
  - select_active(): slice full-DOF arrays to active-only
  - expand_to_full(): reconstruct full-DOF from active-only via mimic rules

IMPORTANT — joint ordering:
  ``active_dof_maps.yml`` stores indices in **URDF parse order**.  Isaac Sim
  (and therefore HDF5 episode files) may reorder joints differently.  When
  consuming runtime data, call ``reindex_for_runtime(runtime_joint_names)``
  to obtain an ``ActiveDofInfo`` whose indices match the runtime ordering.
"""

from __future__ import annotations

import logging
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_YML_PATH = _THIS_DIR / "active_dof_maps.yml"
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ActiveDofInfo:
    robot_key: str
    full_dof: int
    active_dof: int
    active_indices: np.ndarray
    full_joint_names: List[str]
    active_joint_names: List[str]
    mimic_rules: List[Tuple[int, int, float, float]] = field(default_factory=list)

    def reindex_for_runtime(self, runtime_joint_names: List[str]) -> "ActiveDofInfo":
        """Return a new ``ActiveDofInfo`` with indices remapped to *runtime_joint_names* order.

        ``active_dof_maps.yml`` defines indices in URDF parse order, but
        Isaac Sim / HDF5 may present joints in a different order.  This
        method translates active_indices and mimic_rules so they are valid
        when applied to arrays ordered by *runtime_joint_names*.

        If the runtime order already matches ``full_joint_names``, the
        object is returned unchanged (fast path).

        Raises ``ValueError`` if *runtime_joint_names* contains names not
        present in ``full_joint_names`` or has a different length.
        """
        if list(runtime_joint_names) == self.full_joint_names:
            return self

        if len(runtime_joint_names) != self.full_dof:
            raise ValueError(
                f"runtime_joint_names has {len(runtime_joint_names)} entries, "
                f"expected {self.full_dof} (full_dof of {self.robot_key})"
            )

        # Build old-index → new-index mapping via joint names.
        runtime_name_to_idx = {n: i for i, n in enumerate(runtime_joint_names)}
        missing = [n for n in self.full_joint_names if n not in runtime_name_to_idx]
        if missing:
            raise ValueError(
                f"Joint names from active_dof_maps.yml not found in "
                f"runtime_joint_names: {missing[:5]}"
                f"{'...' if len(missing) > 5 else ''}"
            )

        urdf_to_runtime = np.array(
            [runtime_name_to_idx[n] for n in self.full_joint_names],
            dtype=np.intp,
        )

        new_active_indices = urdf_to_runtime[self.active_indices]
        new_active_indices_sorted = np.sort(new_active_indices)

        new_mimic_rules: List[Tuple[int, int, float, float]] = []
        for mimic_idx, source_idx, mult, offset in self.mimic_rules:
            new_mi = int(urdf_to_runtime[mimic_idx])
            new_si = int(urdf_to_runtime[source_idx])
            new_mimic_rules.append((new_mi, new_si, mult, offset))

        return ActiveDofInfo(
            robot_key=self.robot_key,
            full_dof=self.full_dof,
            active_dof=self.active_dof,
            active_indices=new_active_indices_sorted,
            full_joint_names=list(runtime_joint_names),
            active_joint_names=[runtime_joint_names[i] for i in new_active_indices_sorted],
            mimic_rules=new_mimic_rules,
        )


_cache: dict[str, ActiveDofInfo] = {}
_runtime_cache: dict[tuple, ActiveDofInfo] = {}
_yml_data: Optional[dict] = None


def _load_yml() -> dict:
    global _yml_data
    if _yml_data is not None:
        return _yml_data
    import yaml
    with open(_YML_PATH, "r") as f:
        _yml_data = yaml.safe_load(f)
    return _yml_data


def _resolve_urdf_path(urdf_rel: str) -> Path:
    """Resolve a URDF path from active_dof_maps.yml to an absolute path."""
    env_root = os.environ.get("DEX2BENCH_DATASET_ROOT")
    if env_root:
        base = Path(env_root)
    else:
        base = _THIS_DIR.parent.parent / "dex2bench_dataset"
    resolved = base / urdf_rel.replace("Bench2Dex/", "")
    return resolved


def _parse_urdf_mimic(
    urdf_path: Path,
    full_joint_names: List[str],
) -> List[Tuple[int, int, float, float]]:
    """Parse mimic rules from URDF, topologically sorted for chain dependencies.

    Returns list of (mimic_idx, source_idx, multiplier, offset) in the
    full_joint_names index space.
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    name_to_idx = {n: i for i, n in enumerate(full_joint_names)}

    raw_rules: dict[str, tuple[str, float, float]] = {}
    for joint in root.findall(".//joint"):
        mimic = joint.find("mimic")
        if mimic is not None:
            name = joint.get("name", "")
            src = mimic.get("joint", "")
            mult = float(mimic.get("multiplier", "1.0"))
            off = float(mimic.get("offset", "0.0"))
            if name in name_to_idx and src in name_to_idx:
                raw_rules[name] = (src, mult, off)

    sorted_rules: List[Tuple[int, int, float, float]] = []
    resolved = set(name_to_idx.keys()) - set(raw_rules.keys())
    remaining = dict(raw_rules)
    max_iter = len(remaining) + 1
    while remaining and max_iter > 0:
        max_iter -= 1
        for name, (src, mult, off) in list(remaining.items()):
            if src in resolved:
                sorted_rules.append((name_to_idx[name], name_to_idx[src], mult, off))
                resolved.add(name)
                del remaining[name]

    if remaining:
        logger.warning("Unresolved mimic chains: %s", list(remaining.keys()))

    return sorted_rules


def get_active_dof_info(robot_key: str) -> ActiveDofInfo:
    """Load ActiveDofInfo for a robot. Cached after first call per robot_key."""
    if robot_key in _cache:
        return _cache[robot_key]

    data = _load_yml()
    robots = data.get("robots", {})
    if robot_key not in robots:
        known = ", ".join(sorted(robots.keys()))
        raise ValueError(f"Unknown robot_key '{robot_key}'. Known: {known}")

    entry = robots[robot_key]
    full_dof = int(entry["full_dof"])
    active_dof = int(entry["active_dof"])
    active_indices = np.array(entry["active_indices"], dtype=np.intp)
    full_joint_names = list(entry["full_joint_names"])
    active_joint_names = list(entry["active_joint_names"])

    mimic_rules: List[Tuple[int, int, float, float]] = []
    mimic_dof = int(entry.get("mimic_dof", 0))
    if mimic_dof > 0:
        urdf_rel = entry.get("urdf_path", "")
        urdf_path = _resolve_urdf_path(urdf_rel)
        if urdf_path.exists():
            mimic_rules = _parse_urdf_mimic(urdf_path, full_joint_names)
        else:
            logger.warning(
                "URDF not found for %s: %s. Mimic expansion will zero-fill mimic joints.",
                robot_key,
                urdf_path,
            )

    info = ActiveDofInfo(
        robot_key=robot_key,
        full_dof=full_dof,
        active_dof=active_dof,
        active_indices=active_indices,
        full_joint_names=full_joint_names,
        active_joint_names=active_joint_names,
        mimic_rules=mimic_rules,
    )
    _cache[robot_key] = info
    return info


def select_active(array: np.ndarray, info: ActiveDofInfo) -> np.ndarray:
    """Slice full-DOF array to active-only along the last axis.

    Works for any shape: (full_dof,), (T, full_dof), (B, T, full_dof), etc.
    """
    if info.active_dof == info.full_dof:
        return array
    return array[..., info.active_indices]


def expand_to_full(active_array: np.ndarray, info: ActiveDofInfo) -> np.ndarray:
    """Expand active-DOF array to full-DOF, computing mimic joints.

    Works for any shape: (active_dof,), (T, active_dof), etc.
    """
    if info.active_dof == info.full_dof:
        return active_array

    out_shape = active_array.shape[:-1] + (info.full_dof,)
    full = np.zeros(out_shape, dtype=active_array.dtype)
    full[..., info.active_indices] = active_array
    for mimic_idx, source_idx, mult, offset in info.mimic_rules:
        full[..., mimic_idx] = full[..., source_idx] * mult + offset
    return full


def robot_key_from_hdf5(path: str) -> Optional[str]:
    """Read robot_key from an HDF5 episode's /meta group. Returns None if absent."""
    import h5py
    try:
        with h5py.File(path, "r") as f:
            meta = f.get("meta")
            if meta is None:
                return None
            rk = meta.get("robot_key")
            if rk is None:
                rk_attr = meta.attrs.get("robot_key")
                if rk_attr is not None:
                    return str(rk_attr)
                return None
            val = rk[()]
            if isinstance(val, bytes):
                return val.decode("utf-8")
            return str(val)
    except Exception:
        return None


def get_active_dof_info_for_runtime(
    robot_key: str,
    runtime_joint_names: List[str],
) -> ActiveDofInfo:
    """Load ``ActiveDofInfo`` with indices remapped to match *runtime_joint_names*.

    Use this instead of ``get_active_dof_info`` whenever the consumer
    operates on arrays whose joint order comes from Isaac Sim runtime
    (``robot.data.joint_names``) rather than URDF parse order.

    The reindexed result is cached per ``(robot_key, tuple(names))``.
    """
    cache_key = (robot_key, tuple(runtime_joint_names))
    if cache_key in _runtime_cache:
        return _runtime_cache[cache_key]
    base = get_active_dof_info(robot_key)
    reindexed = base.reindex_for_runtime(runtime_joint_names)
    _runtime_cache[cache_key] = reindexed
    return reindexed


def get_active_dof_info_for_hdf5(hdf5_path: str) -> Optional[ActiveDofInfo]:
    """Load ``ActiveDofInfo`` reindexed to the joint order stored in an HDF5 file.

    Reads ``meta/robot_key`` and ``robot/joint_names`` from the file, then
    returns an ``ActiveDofInfo`` whose ``active_indices`` and ``mimic_rules``
    match the HDF5 array ordering.  Returns ``None`` if robot_key or
    joint_names are absent.
    """
    import h5py
    try:
        with h5py.File(hdf5_path, "r") as f:
            rk_ds = f.get("meta/robot_key")
            if rk_ds is None:
                return None
            rk_val = rk_ds[()]
            robot_key = rk_val.decode("utf-8") if isinstance(rk_val, bytes) else str(rk_val)
            jn_ds = f.get("robot/joint_names")
            if jn_ds is None:
                return None
            runtime_names = [
                n.decode("utf-8") if isinstance(n, bytes) else str(n)
                for n in jn_ds[()]
            ]
        return get_active_dof_info_for_runtime(robot_key, runtime_names)
    except Exception:
        return None


def get_all_robot_keys() -> List[str]:
    """Return all robot keys defined in active_dof_maps.yml."""
    data = _load_yml()
    return sorted(data.get("robots", {}).keys())
