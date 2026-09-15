#!/usr/bin/env python3
"""Re-evaluate task 44 recorded HDF5 episodes with current success criteria.

Key change since the original Jul-4 evaluation (commit ``6da843b``):
  - ``container_center_offset`` support in ``object_inside`` — the bowl-mouth
    centre is now correctly used instead of falling back to the bowl's
    centre-of-mass.

Limitations of old HDF5 files (no lin_vel / joint data):
  - Velocity approximated from pose delta over the last 10 frames.
  - Microwave door joint state is **not available**; the door-closed check is
    skipped and reported separately.
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List

import h5py
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

from success.condition_evaluator import evaluate_conditions
from success.custom.task_44_microwave_bowl_loading import (
    CONDITIONS,
    BAGUETTE_CENTER_OFFSET,
    BOWL_MOUTH_CENTER_OFFSET,
    BOWL_MOUTH_RADIUS,
    ZONE_HI,
    ZONE_LO,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pose_delta_velocity(poses: np.ndarray, window: int = 10, physics_dt: float = 1.0 / 60.0) -> np.ndarray:
    """Approximate linear velocity from position delta over *window* frames."""
    if len(poses) < 2:
        return np.zeros(3, dtype=np.float64)
    n = min(window, len(poses) - 1)
    dt = n * physics_dt
    vel = (poses[-1, :3] - poses[-(n + 1), :3]) / max(dt, 1e-6)
    return vel.astype(np.float64)


def load_states_from_hdf5(hdf5_path: str) -> Dict[str, Any]:
    """Build a ``states`` dict compatible with the condition evaluator.

    Only the **last frame** is used for position checks; velocity is
    estimated from the preceding *window* frames.
    """
    states: Dict[str, Any] = {}
    with h5py.File(hdf5_path, "r") as f:
        obj_group = f["objects"]
        for obj_id in obj_group:
            ds = obj_group[obj_id]
            poses = ds["pose_world"][:]  # (T, 7)
            if poses.ndim != 2 or poses.shape[1] != 7:
                continue
            lin_vel = _pose_delta_velocity(poses)
            states[obj_id] = {
                "pose_world": poses[-1].astype(np.float64),
                "lin_vel_world": lin_vel,
                "ang_vel_world": np.zeros(3, dtype=np.float64),
            }
    return states


# ---------------------------------------------------------------------------
# Per-condition checks (mirrors evaluate_conditions but reports individually)
# ---------------------------------------------------------------------------

from success.condition_evaluator import (  # noqa: E402
    _check_joint_state,
    _check_object_inside,
    _check_object_in_container_zone,
    _check_object_static,
)


def check_individual(states: Dict[str, Any]) -> Dict[str, bool | None]:
    """Evaluate each terminal sub-condition and return pass/fail/unknown."""
    result: Dict[str, bool | None] = {}

    # 1. Bowl in microwave zone
    result["bowl_in_microwave_zone"] = _check_object_in_container_zone(
        {"object": "obj_024_bowl_3", "container": "obj_104_microwave_2",
         "zone_lo": ZONE_LO, "zone_hi": ZONE_HI},
        states,
    )

    # 2. Baguette in microwave zone
    result["baguette_in_microwave_zone"] = _check_object_in_container_zone(
        {"object": "obj_163_baguette_1", "container": "obj_104_microwave_2",
         "zone_lo": ZONE_LO, "zone_hi": ZONE_HI},
        states,
    )

    # 3. Baguette inside bowl (this is where container_center_offset matters)
    result["baguette_inside_bowl"] = _check_object_inside(
        {"object": "obj_163_baguette_1", "container": "obj_024_bowl_3",
         "object_center_offset": BAGUETTE_CENTER_OFFSET,
         "container_center_offset": BOWL_MOUTH_CENTER_OFFSET,
         "tolerance": BOWL_MOUTH_RADIUS},
        states,
    )

    # 4. Baguette static
    result["baguette_static"] = _check_object_static(
        {"object": "obj_163_baguette_1", "threshold": 0.05, "check_angular": False},
        states,
    )

    # 5. Door closed — joint state NOT available in old HDF5
    if "joint_0" in states.get("obj_104_microwave_2", {}) or (
        states.get("obj_104_microwave_2", {}).get("qpos") is not None
        and len(states.get("obj_104_microwave_2", {}).get("qpos", [])) > 0
    ):
        result["door_closed"] = _check_joint_state(
            {"object": "obj_104_microwave_2", "joint": "joint_0",
             "target": "closed", "tolerance": 0.2},
            states,
        )
    else:
        result["door_closed"] = None  # unknown

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) > 1:
        base_dir = os.path.expanduser(sys.argv[1])
    else:
        base_dir = os.path.expanduser(
            "../output/eval/44/gr00t_n15_all_0704_2356_schunk_hand/none/episodes"
        )
    if not os.path.isdir(base_dir):
        print(f"ERROR: directory not found: {base_dir}")
        sys.exit(1)

    all_results: List[Dict[str, Any]] = []

    for label_dir in ("success", "failure"):
        full = os.path.join(base_dir, label_dir)
        if not os.path.isdir(full):
            continue
        for fname in sorted(os.listdir(full)):
            if not fname.endswith(".hdf5"):
                continue
            path = os.path.join(full, fname)
            ep_idx = int(fname.replace("episode_", "").replace(".hdf5", ""))
            try:
                states = load_states_from_hdf5(path)
                checks = check_individual(states)
                # Overall without door check (spatial-only)
                spatial_ok = all(
                    checks[k] for k in checks
                    if k != "door_closed" and checks[k] is not None
                )
                terminal_ok = evaluate_conditions(CONDITIONS, states, {})
                all_results.append({
                    "episode": ep_idx,
                    "file": f"{label_dir}/{fname}",
                    "orig_label": label_dir,
                    "checks": checks,
                    "spatial_only_ok": spatial_ok,
                    "terminal_ok": terminal_ok,
                })
            except Exception as exc:
                all_results.append({
                    "episode": ep_idx,
                    "file": f"{label_dir}/{fname}",
                    "orig_label": label_dir,
                    "error": str(exc),
                })

    all_results.sort(key=lambda r: r["episode"])

    # ── Summary ────────────────────────────────────────────────────────────
    total = len(all_results)
    errors = [r for r in all_results if "error" in r]
    valid = [r for r in all_results if "error" not in r]

    orig_success = sum(1 for r in valid if r["orig_label"] == "success")
    spatial_success = sum(1 for r in valid if r.get("spatial_only_ok"))
    terminal_success = sum(1 for r in valid if r.get("terminal_ok"))

    print(f"{'='*70}")
    print(f"Task 44 Re-evaluation ({base_dir})")
    print(f"{'='*70}")
    print(f"Total episodes: {total}")
    print(f"Errors: {len(errors)}")
    print(f"")
    print(f"Original success:  {orig_success}/{total} ({orig_success/total*100:.1f}%)")
    print(f"Spatial-only pass: {spatial_success}/{total} ({spatial_success/total*100:.1f}%)")
    print(f"Terminal pass (*):  {terminal_success}/{total} ({terminal_success/total*100:.1f}%)")
    print(f"")
    print(f"(*) Terminal includes door_closed which defaults to True (joint data")
    print(f"    unavailable in old HDF5).  Use spatial-only pass for comparison.")

    # ── Per-condition breakdown ────────────────────────────────────────────
    cond_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"pass": 0, "fail": 0, "unknown": 0})
    for r in valid:
        for cond, val in r["checks"].items():
            if val is True:
                cond_counts[cond]["pass"] += 1
            elif val is False:
                cond_counts[cond]["fail"] += 1
            else:
                cond_counts[cond]["unknown"] += 1

    print(f"\n{'─'*70}")
    print(f"Per-condition breakdown ({len(valid)} episodes):")
    print(f"{'Condition':<35} {'Pass':>6} {'Fail':>6} {'Unknown':>8}")
    print(f"{'─'*57}")
    for cond in ["bowl_in_microwave_zone", "baguette_in_microwave_zone",
                  "baguette_inside_bowl", "baguette_static", "door_closed"]:
        c = cond_counts[cond]
        print(f"{cond:<35} {c['pass']:>6} {c['fail']:>6} {c['unknown']:>8}")

    # ── Changed verdicts ───────────────────────────────────────────────────
    changed = [
        r for r in valid
        if r.get("spatial_only_ok") != (r["orig_label"] == "success")
    ]
    print(f"\n{'─'*70}")
    print(f"Verdict changes (spatial-only vs original): {len(changed)} episodes")
    for r in changed:
        old = "SUCCESS" if r["orig_label"] == "success" else "FAILURE"
        new = "PASS" if r["spatial_only_ok"] else "FAIL"
        arrow = "↑" if r["spatial_only_ok"] and r["orig_label"] == "failure" else "↓"
        print(f"  {arrow} ep {r['episode']:02d}: {old} → {new}  "
              f"bowl_in_zone={r['checks'].get('bowl_in_microwave_zone')} "
              f"bag_in_zone={r['checks'].get('baguette_in_microwave_zone')} "
              f"bag_in_bowl={r['checks'].get('baguette_inside_bowl')} "
              f"bag_static={r['checks'].get('baguette_static')}")

    # ── Write per-episode JSONL ────────────────────────────────────────────
    out_path = os.path.join(os.path.dirname(base_dir), "re_eval_per_episode.jsonl")
    with open(out_path, "w") as f:
        for r in all_results:
            f.write(json.dumps(r, default=str) + "\n")
    print(f"\nPer-episode details: {out_path}")


if __name__ == "__main__":
    main()
