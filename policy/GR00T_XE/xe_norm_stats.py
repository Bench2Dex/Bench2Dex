# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GR00T XE normalization statistics -- single source of truth.

The unified 64D vector mixes two quantities whose normalization requirements are
opposite:

  dims  0:12   arm EE, 2 x (x, y, z, roll, pitch, yaw), expressed in the SHARED
               world frame (H1).  One physical pose must map to one number no
               matter which robot produced it, so these dims use statistics
               pooled over the whole corpus.
  dims 12:56   hand joint slots (right 12:33, left 34:55).  Joint ranges differ
               per robot and so does the slot layout, so these use the OWNING
               ROBOT's statistics only.  A slot the robot does not author stays
               at its literal 0: min == max == 0 makes the min_max normalizer
               pass it through untouched.
  dims 56:64   pad, always 0, min == max == 0 -> passthrough.

Why the statistics live in an on-disk artifact:

  * The mixture merge (``LeRobotMixtureDataset.compute_overall_statistics``)
    takes ``np.min`` / ``np.max`` across ALL 26 tasks.  Every robot that does
    not author a slot reports 0/0, so a slot whose real range is [0.9, 1.0]
    merges to [0.0, 1.0]: the authoring robot is squeezed into 10% of [-1, 1]
    while the other 11 robots read a constant -1.
  * Weighting that merge by author count is NOT a fix: the author's [0.9, 1.0]
    would map every other robot's literal 0 to 2*(0-0.9)/0.1 - 1 = -19.
  * Finetune builds a single-task dataset.  Left to itself it would normalize
    the arm EE with that one task's statistics, while the weights it starts from
    were trained against the 26-task ones.

Nothing in this module falls back silently.  A missing artifact, an artifact
that was not built from this task, a robot that is not in it, or a normalizer
that does not end up holding the assembled values all raise.
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------
N_DIMS = 64
N_EE = 12
N_HAND = 44
N_PAD = 8
EE_SLICE = slice(0, N_EE)
HAND_SLICE = slice(N_EE, N_EE + N_HAND)
PAD_SLICE = slice(N_EE + N_HAND, N_DIMS)
assert (N_EE, N_HAND, N_PAD) == (12, 44, 8)

STAT_KEYS = ("mean", "std", "min", "max", "q01", "q99")
MODALITIES = ("state", "action")
KEY_TPL = "{modality}.qpos"

ARTIFACT_NAME = "xe_norm_stats.json"
ENV_VAR = "GR00T_XE_NORM_STATS"
ARTIFACT_VERSION = 1

# min_max is exact in float32 once the values agree to ~1e-6; anything past this
# means the normalizer holds somebody else's statistics, not a rounding wobble.
_TOL = 1e-4


def repo_artifact_path() -> Path:
    """Where the canonical artifact lives (env override, else next to this file)."""
    override = os.environ.get(ENV_VAR, "").strip()
    if override:
        return Path(override)
    return Path(__file__).resolve().parent / ARTIFACT_NAME


def task_dir_key(input_dir: str | Path) -> str:
    """Stable name for a dataset dir: the last two path components.

    ``.../26_foo/replay-generalization`` -> ``26_foo/replay-generalization``.
    Absolute prefixes differ between build hosts and training hosts on the
    shared filesystem, so they must not be part of the key.
    """
    parts = Path(input_dir).parts
    if len(parts) < 2:
        raise ValueError(f"[XE-NORM] dataset dir too short to identify: {input_dir}")
    return "/".join(parts[-2:])


# --------------------------------------------------------------------------
# Load / validate
# --------------------------------------------------------------------------
def _bad(path: Any, msg: str) -> "ValueError":
    return ValueError(f"[XE-NORM] {path}: {msg}")


def _check_len(stats: dict, n: int, who: str, path: Any) -> None:
    for k in STAT_KEYS:
        if k not in stats:
            raise _bad(path, f"{who} is missing statistic {k!r}")
        arr = np.asarray(stats[k], dtype=np.float64)
        if arr.shape != (n,):
            raise _bad(path, f"{who}.{k} has shape {arr.shape}, expected ({n},)")
        if not np.all(np.isfinite(arr)):
            raise _bad(path, f"{who}.{k} contains non-finite values")
    lo = np.asarray(stats["min"], dtype=np.float64)
    hi = np.asarray(stats["max"], dtype=np.float64)
    if np.any(hi < lo):
        bad = np.flatnonzero(hi < lo)
        raise _bad(path, f"{who}: max < min on dims {bad.tolist()}")


def validate_artifact(art: dict, path: Any) -> None:
    if art.get("version") != ARTIFACT_VERSION:
        raise _bad(path, f"version {art.get('version')!r} != {ARTIFACT_VERSION}")
    if art.get("n_dims") != N_DIMS:
        raise _bad(path, f"n_dims {art.get('n_dims')!r} != {N_DIMS}")

    robot_keys = art.get("hand_robot_keys")
    if not robot_keys:
        raise _bad(path, "hand_robot_keys is empty -- nothing could be normalized per robot")
    hands = art.get("hands_by_robot")
    if not isinstance(hands, dict) or set(hands) != set(robot_keys):
        raise _bad(path, "hands_by_robot keys do not match hand_robot_keys")

    ee_global = art.get("ee_global")
    if not isinstance(ee_global, dict):
        raise _bad(path, "ee_global missing")
    for mod in MODALITIES:
        if mod not in ee_global:
            raise _bad(path, f"ee_global.{mod} missing")
        _check_len(ee_global[mod], N_EE, f"ee_global.{mod}", path)

        for rk in robot_keys:
            entry = hands[rk]
            if mod not in entry:
                raise _bad(path, f"hands_by_robot[{rk}].{mod} missing")
            _check_len(entry[mod], N_HAND, f"hands_by_robot[{rk}].{mod}", path)

            mask = np.asarray(entry.get("authored_mask", []), dtype=bool)
            if mask.shape != (N_HAND,):
                raise _bad(path, f"hands_by_robot[{rk}].authored_mask shape {mask.shape}, "
                                 f"expected ({N_HAND},)")
            # The whole design rests on this: a slot the robot does not author
            # must be a constant 0 so the min_max normalizer passes it through.
            lo = np.asarray(entry[mod]["min"], dtype=np.float64)[~mask]
            hi = np.asarray(entry[mod]["max"], dtype=np.float64)[~mask]
            if not (np.all(lo == 0.0) and np.all(hi == 0.0)):
                bad = np.flatnonzero((lo != 0.0) | (hi != 0.0))
                raise _bad(path, f"hands_by_robot[{rk}].{mod}: unauthored slots are not a "
                                 f"constant 0 -- they would be normalized instead of "
                                 f"passed through (offsets {bad.tolist()})")

    if not art.get("sources"):
        raise _bad(path, "sources is empty -- the artifact cannot be checked against a "
                         "dataset dir")


def load_artifact(path: str | Path | None = None) -> dict:
    p = Path(path) if path is not None else repo_artifact_path()
    if p.is_dir():
        p = p / ARTIFACT_NAME
    if not p.is_file():
        raise FileNotFoundError(
            f"[XE-NORM] normalization statistics artifact not found: {p}\n"
            f"  GR00T XE deliberately does NOT re-derive statistics from whatever data a\n"
            f"  run happens to see -- that is the bug this artifact exists to prevent\n"
            f"  (pretrain would merge hand slots across all 26 tasks with np.min/np.max\n"
            f"  and squeeze every single-author slot toward 0; finetune would normalize\n"
            f"  the arm EE with its one task).\n"
            f"  Build it with:\n"
            f"      python3 {Path(__file__).resolve().parent / 'build_xe_norm_stats.py'}"
        )
    art = json.loads(p.read_text())
    validate_artifact(art, p)
    return art


def install_artifact(dest_dir: str | Path, artifact: dict | None = None) -> Path:
    """Copy the artifact into a checkpoint's ``experiment_cfg``.

    Deploy re-applies the artifact rather than the merged ``metadata.json``, so
    every checkpoint has to carry the exact statistics it was trained with.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / ARTIFACT_NAME
    if artifact is None:
        src = repo_artifact_path()
        if not src.is_file():
            raise FileNotFoundError(f"[XE-NORM] cannot install artifact: {src} does not exist")
        shutil.copyfile(src, dest)
    else:
        dest.write_text(json.dumps(artifact, indent=2))
    return dest


# --------------------------------------------------------------------------
# Assemble the 64D statistics for one robot
# --------------------------------------------------------------------------
def assemble(robot_key: str, art: dict) -> dict[str, dict[str, list[float]]]:
    """64D state/action statistics for ``robot_key``.

    EE 0:12 from the corpus-wide pool, hand 12:56 from this robot, pad 56:64 a
    constant 0.  Raises if the robot has no hand statistics of its own.
    """
    hands = art["hands_by_robot"]
    if robot_key not in hands:
        raise KeyError(
            f"[XE-NORM] robot_key={robot_key!r} is not in the normalization artifact. "
            f"Known robots: {sorted(hands)}. Hand slots cannot be normalized per robot "
            f"without that robot's own statistics, and borrowing another robot's would "
            f"rescale its joints by a range that means nothing for it."
        )

    out: dict[str, dict[str, list[float]]] = {}
    for mod in MODALITIES:
        ee = art["ee_global"][mod]
        hand = hands[robot_key][mod]
        out[mod] = {
            k: [float(v) for v in ee[k]]
            + [float(v) for v in hand[k]]
            + [0.0] * N_PAD
            for k in STAT_KEYS
        }
        for k in STAT_KEYS:
            if len(out[mod][k]) != N_DIMS:
                raise AssertionError(
                    f"[XE-NORM] assembled {mod}.{k} has {len(out[mod][k])} dims, expected {N_DIMS}"
                )
    return out


def check_task_against_artifact(stats: dict, robot_key: str, input_dir: str | Path,
                                art: dict, *, context: str = "") -> None:
    """Fail if this task's own statistics are not consistent with the artifact.

    Two checks, both cheap and both loud:

    * the task must be one of the recorded sources, and its EE file must not have
      changed size since the artifact was built (a rebuilt ``arm_ee_trajectories``
      makes the frozen EE range stale, which would otherwise go unnoticed because
      it only widens or narrows the range, never invalidates it);
    * every dim this task produces must lie inside the artifact's assembled range
      for this robot.  A range that does not contain the data means the artifact
      was built from a different corpus.
    """
    where = context or str(input_dir)
    key = task_dir_key(input_dir)
    sources = art["sources"]
    if key not in sources:
        raise KeyError(
            f"[XE-NORM] {where}: this dataset is not in the normalization artifact "
            f"({sorted(sources)}). It was added or renamed after the artifact was built; "
            f"rebuild with build_xe_norm_stats.py."
        )
    rec = sources[key]
    if rec["robot_key"] != robot_key:
        raise ValueError(
            f"[XE-NORM] {where}: artifact records robot_key={rec['robot_key']!r} for this "
            f"dataset but the data says {robot_key!r}."
        )

    ee_path = Path(input_dir).parent / art["ee_file"]
    if not ee_path.is_file():
        raise FileNotFoundError(f"[XE-NORM] {where}: EE file missing: {ee_path}")
    size = ee_path.stat().st_size
    if size != rec["ee_size"]:
        raise ValueError(
            f"[XE-NORM] {where}: {ee_path.name} is {size} bytes but the artifact was built "
            f"from {rec['ee_size']} -- the EE trajectories changed (H1 re-export? new "
            f"episodes?) and the frozen EE statistics are stale. Rebuild with "
            f"build_xe_norm_stats.py and retrain."
        )

    built = assemble(robot_key, art)
    for mod in MODALITIES:
        tmin = np.asarray(stats[mod]["qpos"]["min"], dtype=np.float64)
        tmax = np.asarray(stats[mod]["qpos"]["max"], dtype=np.float64)
        amin = np.asarray(built[mod]["min"], dtype=np.float64)
        amax = np.asarray(built[mod]["max"], dtype=np.float64)
        if tmin.shape != (N_DIMS,):
            raise ValueError(f"[XE-NORM] {where}: {mod} statistics have shape "
                             f"{tmin.shape}, expected ({N_DIMS},)")
        # A dim left at 0 by this task must be a passthrough dim in the artifact.
        unauth = (tmin == 0.0) & (tmax == 0.0)
        drift = np.flatnonzero((amin[unauth] != 0.0) | (amax[unauth] != 0.0))
        if drift.size:
            raise ValueError(
                f"[XE-NORM] {where}: {mod} dims {drift.tolist()} are constant 0 in this "
                f"task but the artifact has a real range for them -- the artifact was "
                f"built from a different corpus / robot."
            )
        outside = np.flatnonzero((tmin < amin - _TOL) | (tmax > amax + _TOL))
        if outside.size:
            rows = [(int(d), float(tmin[d]), float(tmax[d]), float(amin[d]), float(amax[d]))
                    for d in outside[:8]]
            raise ValueError(
                f"[XE-NORM] {where}: {mod} dims {outside.tolist()} fall outside the "
                f"artifact range for {robot_key!r} (dim, task_min, task_max, art_min, "
                f"art_max): {rows}. Rebuild with build_xe_norm_stats.py."
            )


# --------------------------------------------------------------------------
# Push the assembled statistics into a live transform pipeline
# --------------------------------------------------------------------------
def _state_action_transforms(transforms) -> list:
    out = []
    for t in getattr(transforms, "transforms", []):
        if hasattr(t, "_normalizers"):
            out.append(t)
    return out


def _verify(transforms, built: dict, robot_key: str, *, context: str) -> None:
    """Read the normalizers back and confirm they hold the assembled values.

    This is the guard against 坑1: ``pretrain.py`` used to build ONE transform and
    hand it to all 26 datasets, so every ``set_metadata`` call overwrote the
    previous one and whichever robot ran last normalized the whole corpus.  The
    only way to notice that is to look at the object the model will actually use.
    """
    want = {KEY_TPL.format(modality=m): built[m] for m in MODALITIES}
    found: dict[str, Any] = {}
    for t in _state_action_transforms(transforms):
        for key, n in getattr(t, "_normalizers", {}).items():
            if key in want:
                if key in found:
                    raise RuntimeError(
                        f"[XE-NORM] {context}: two normalizers claim {key!r}; the transform "
                        f"pipeline is not the one this module expects"
                    )
                found[key] = n
    missing = sorted(set(want) - set(found))
    if missing:
        raise RuntimeError(
            f"[XE-NORM] {context}: the transform pipeline has no normalizer for {missing} "
            f"-- set_metadata did not run on it"
        )

    for key, n in found.items():
        for stat in ("min", "max"):
            got = n.statistics[stat]
            got = got.detach().cpu().numpy() if hasattr(got, "detach") else np.asarray(got)
            exp = np.asarray(want[key][stat], dtype=np.float64)
            if got.shape != exp.shape:
                raise RuntimeError(
                    f"[XE-NORM] {context}: normalizer {key}.{stat} has shape {got.shape}, "
                    f"expected {exp.shape}"
                )
            diff = float(np.abs(got.astype(np.float64) - exp).max())
            if diff > _TOL:
                raise RuntimeError(
                    f"[XE-NORM] {context}: normalizer {key}.{stat} does NOT hold the "
                    f"assembled statistics for robot {robot_key!r} (max abs diff {diff:.6g}). "
                    f"Something called set_metadata after this point -- a shared transform "
                    f"object handed to more than one dataset would do exactly that."
                )


def apply_to_transform(transforms, metadata, robot_key: str, *,
                       artifact: dict | None = None,
                       context: str = "") -> dict:
    """Rewrite ``metadata``'s statistics and rebuild the normalizers from them.

    ``metadata`` may come from anywhere (a dataset's own ``_build_metadata``, the
    mixture's merge, or a checkpoint's ``metadata.json``); only its ``statistics``
    are replaced, so the modality configs it carries stay intact.

    Returns the assembled 64D statistics, keyed by modality.
    """
    from gr00t.data.schema import DatasetStatisticalValues

    art = artifact if artifact is not None else load_artifact()
    built = assemble(robot_key, art)

    for mod in MODALITIES:
        state_key = KEY_TPL.format(modality=mod).split(".")[1]
        # DatasetStatistics declares `state`/`action` as Dict[str, DatasetStatisticalValues]
        # (`gr00t.data.schema`), and StateActionTransform.set_metadata reads them as
        # `getattr(dataset_statistics, modality)[state_key]` before `.model_dump()`.
        # Writing `metadata.statistics.state.qpos = ...` looks just as plausible and
        # never reaches the normalizer -- it raises, which is why this is spelled out.
        modality_stats = getattr(metadata.statistics, mod, None)
        if not isinstance(modality_stats, dict):
            raise TypeError(
                f"[XE-NORM] {context or '<unknown>'}: metadata.statistics.{mod} is "
                f"{type(modality_stats).__name__}, expected a dict keyed by "
                f"{state_key!r} -- refusing to write statistics nothing would read"
            )
        modality_stats[state_key] = DatasetStatisticalValues(**built[mod])

    transforms.set_metadata(metadata)
    _verify(transforms, built, robot_key, context=context or "<unknown>")
    return built


def _read_metadata_json(exp_cfg_dir: str | Path, embodiment_tag: str):
    """Same selection ``Gr00tPolicy._load_metadata`` does, but without a model."""
    from gr00t.data.schema import DatasetMetadata

    path = Path(exp_cfg_dir) / "metadata.json"
    if not path.is_file():
        raise FileNotFoundError(f"[XE-NORM] checkpoint has no metadata.json: {path}")
    metadatas = json.loads(path.read_text())
    raw = metadatas.get(embodiment_tag)
    if raw is None:
        raise ValueError(
            f"[XE-NORM] no metadata for embodiment tag {embodiment_tag!r} in {path}"
        )
    return DatasetMetadata.model_validate(raw)


def reapply_for_deploy(transforms, exp_cfg_dir: str | Path, embodiment_tag: str,
                       robot_key: str, *, artifact: dict | None = None) -> dict:
    """Deploy-side entry point.

    ``Gr00tPolicy._load_metadata`` has just normalized the pipeline with whatever
    ``metadata.json`` held -- i.e. the mixture's merged statistics, which is not
    what the weights were trained with.  Replace them with the artifact that this
    checkpoint was trained against.  The artifact must be there: falling back to
    the merged blob is the bug this whole module exists to remove.
    """
    exp_cfg_dir = Path(exp_cfg_dir)
    art = artifact if artifact is not None else load_artifact(exp_cfg_dir)
    metadata = _read_metadata_json(exp_cfg_dir, embodiment_tag)
    built = apply_to_transform(
        transforms, metadata, robot_key, artifact=art,
        context=f"deploy {exp_cfg_dir} robot={robot_key}",
    )
    ee = np.asarray(built["action"]["max"], dtype=np.float64)[:N_EE]
    print(
        f"[XE-NORM] deploy: denormalizing with artifact {exp_cfg_dir / ARTIFACT_NAME} "
        f"({art['n_tasks']} tasks, built {art.get('generated', '?')}); robot={robot_key}",
        flush=True,
    )
    return built


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
