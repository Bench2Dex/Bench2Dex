"""Joint-order permutations between Isaac runtime order and canonical URDF order.

``active_dof_maps.yml`` stores ``full_joint_names`` in URDF parse order, but
Isaac Sim reports joint state in the articulation's own order, which the
recorder writes verbatim into HDF5 (``robot/joint_names``).  For the
jaka/dexhand robot the two are not even block-swapped — they are interleaved::

    runtime:   joint_1, l_joint_1, joint_2, l_joint_2, ..., r_f_joint1_1,
               r_f_joint2_1, l_f_joint1_1, l_f_joint2_1, r_f_joint1_2, ...
    canonical: joint_1..joint_6, r_f_joint1_1..r_f_joint5_4,
               l_joint_1..l_joint_6, l_f_joint1_1..l_f_joint5_4

Anything that hands runtime-ordered values to code indexing by
``get_active_dof_info().full_joint_names`` (or vice versa) must permute first.
This module is the single source of that permutation so the client, the server
and the offline validation scripts cannot drift apart.
"""

from __future__ import annotations

import numpy as np

_perm_cache: dict = {}


def name_perm(canonical_names, runtime_names, *, context: str):
    """Return ``(runtime_to_canonical, canonical_to_runtime)`` index permutations.

    ``arr_runtime[runtime_to_canonical]`` reorders an array from
    *runtime_names* order into *canonical_names* order; the inverse maps back.

    Raises ``ValueError`` if a canonical joint is absent from the runtime
    articulation — silently leaving it out would scramble the remaining
    columns, so a missing name is a hard error (memory: no-silent-error-skipping).
    """
    cache_key = (tuple(canonical_names), tuple(runtime_names))
    cached = _perm_cache.get(cache_key)
    if cached is not None:
        return cached

    runtime_index = {name: idx for idx, name in enumerate(runtime_names)}
    missing = [name for name in canonical_names if name not in runtime_index]
    if missing:
        raise ValueError(
            f"{context}: canonical joints absent from the runtime articulation: "
            f"{missing[:8]}{'...' if len(missing) > 8 else ''}"
        )
    runtime_to_canonical = np.array(
        [runtime_index[name] for name in canonical_names], dtype=np.intp,
    )
    perms = (runtime_to_canonical, np.argsort(runtime_to_canonical))
    _perm_cache[cache_key] = perms
    return perms


def reindex_runtime_to_canonical(
    array: np.ndarray, canonical_names, runtime_names, *, context: str,
) -> np.ndarray:
    """Reorder a runtime-ordered array into canonical (URDF) order."""
    runtime_to_canonical, _ = name_perm(
        canonical_names, runtime_names, context=context,
    )
    array = np.asarray(array)
    if array.shape[-1] != runtime_to_canonical.size:
        raise ValueError(
            f"{context}: expected {runtime_to_canonical.size} values on the last "
            f"axis, got {array.shape[-1]}"
        )
    return array[..., runtime_to_canonical]


def reindex_canonical_to_runtime(
    array: np.ndarray, canonical_names, runtime_names, *, context: str,
) -> np.ndarray:
    """Reorder a canonical (URDF) ordered array into runtime order."""
    _, canonical_to_runtime = name_perm(
        canonical_names, runtime_names, context=context,
    )
    array = np.asarray(array)
    if array.shape[-1] != canonical_to_runtime.size:
        raise ValueError(
            f"{context}: expected {canonical_to_runtime.size} values on the last "
            f"axis, got {array.shape[-1]}"
        )
    return array[..., canonical_to_runtime]
