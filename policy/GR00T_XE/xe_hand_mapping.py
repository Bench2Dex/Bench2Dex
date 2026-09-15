"""Single source of truth for the cross-embodiment hand joint → unified-slot mapping.

Both the training dataset (:mod:`dataset`) and the online/offline converter
(:mod:`ik_arm_converter`) resolve a robot's hand joints into the 64-D unified
space through THIS module, so the two paths cannot drift apart (memory rule:
train/inference logic that is duplicated must reference each other).

Three places must stay consistent if the layout ever changes:

* ``embodiment_mapping.yml``  — authored per-robot joint → semantic slot (44)
* ``xe_hand_mapping.py``      — parses the YAML into lookup dicts + validates
* ``cross_embodiment_design.md`` — the 64-D layout spec

Unified 64-D layout (also in the design doc):

    [0:6]   right arm EE pose        (px,py,pz,rx,ry,rz)
    [6:12]  left  arm EE pose
    [12:34] right hand (22 semantic slots)
    [34:56] left  hand (22 semantic slots)
    [56:64] padding (always 0)

The YAML stores right slots 0-21 and left slots 22-43; each slot value is the
global slot index, so BOTH sides need +12 to land in the unified layout
(right 0-21 → 12-33, left 22-43 → 34-55).  Do NOT use different offsets per
side — that silently pushes the left hand into the [56:64] padding zone
(a regression this module once had).

Joint-name convention (must match what Isaac/HDF5 actually records, i.e.
``urdf_non_fixed_with_isaac_name_sanitize``): author the EXACT recorded name
for each physical side.  Which side carries a ``multi_`` prefix is robot
specific and is baked into the recorded names, so the YAML already reflects it:

* iiwa7/sharpa, xarm7+leap, panda+allegro, xarm7+ability: the RIGHT hand is
  recorded with a ``multi_`` prefix (``multi_right_*`` / ``multi_leap_r_*`` /
  ``multi_joint_*`` / ``multi_*``), the LEFT hand unprefixed
* ur5-family (shadow/rh5dg2/rh56dfx/schunk/wuji) & orca/jaka/revo2: RIGHT hand
  unprefixed (``THJ*`` / ``right_*`` / ``r_*``), LEFT with ``l_*`` / ``left_*``

Never add an automatic ``multi_`` alias for the right hand: for robots whose
LEFT hand shares the plain base name (allegro ``joint_13_0`` vs right
``multi_joint_13_0``) or whose LEFT hand is the ``multi_`` one (ability), the
alias makes one real joint resolve to BOTH sides and the left hand silently
stays 0 (a bug this module once had).

Failure rule (memory: no-silent-error-skipping): if a robot's hand cannot be
fully resolved against the real joint names we RAISE here — never run with
hand columns silently stuck at 0.
"""

from __future__ import annotations

from pathlib import Path

import yaml

_MAPPING_PATH = Path(__file__).resolve().parent / "embodiment_mapping.yml"

# robot_key → hand name used as the key inside embodiment_mapping.yml.
# This MUST be an explicit table: string-slicing robot_key used to silently
# extract the wrong name for 10/12 robots (see memory no-silent-error-skipping).
ROBOT_KEY_TO_HAND = {
    "multi_iiwa7_with_sharpa": "sharpa",
    "multi_jaka_zu7_dexhand021_with_flange": "jaka",
    "multi_panda_with_allegro": "allegro",
    "multi_panda_with_orca": "orca",
    "multi_rm_65_with_revo2": "revo2",
    "multi_ur5_rh56dfx_with_flange": "rh56dfx",
    "multi_ur5_rh5dg2_with_flange": "rh5dg2",
    "multi_ur5_schunk_hand_with_flange": "schunk",
    "multi_ur5_shadow_hand_with_flange": "shadow",
    "multi_ur5_wuji_with_flange": "wuji",
    "multi_xarm7_with_ability": "ability",
    "multi_xarm7_with_leap": "leap",
}

# YAML slot indices are GLOBAL (right 0-21, left 22-43).  Both hands shift by
# +12 into the unified layout (right → 12-33, left → 34-55).  Same offset for
# both sides: 22+12 = 34.  (A per-side offset of 34 here previously pushed the
# left hand into the 56-77 padding zone.)
_SIDE_TO_UNIFIED_OFFSET = {"right": 12, "left": 12}


def robot_key_to_hand_name(robot_key: str) -> str:
    """Map a robot_key to its hand name; raises instead of guessing."""
    hand_name = ROBOT_KEY_TO_HAND.get(robot_key)
    if hand_name is None:
        raise ValueError(
            f"Unknown robot_key '{robot_key}'. "
            f"Supported: {sorted(ROBOT_KEY_TO_HAND.keys())}"
        )
    return hand_name


def _load_raw_mapping() -> dict:
    with open(_MAPPING_PATH) as f:
        return yaml.safe_load(f)


def authored_slots(hand_name: str) -> dict:
    """Return ``{side: {slot: joint_name}}`` for one hand.

    slot is the YAML slot (right 0-21 / left 22-43).  Joints not authored for
    this hand are simply absent from the dict (they stay 0 in the unified
    space and are handled by the action/state mask).
    """
    raw = _load_raw_mapping()["hand_slots"]
    out: dict[str, dict[int, str]] = {"right": {}, "left": {}}
    for side in ("right", "left"):
        for slot_def in raw[side]:
            jn = slot_def.get(hand_name)
            if jn is not None:
                out[side][int(slot_def["slot"])] = str(jn)
    return out


def build_name_to_slot_maps(hand_name: str) -> tuple[dict[str, int], dict[str, int]]:
    """Build ``{joint_name: unified_slot}`` lookups for both hands.

    Returns ``(right_name_to_slot, left_name_to_slot)`` where each key is the
    EXACT recorded joint name authored in the YAML (no automatic prefix alias —
    an alias made the left hand of allegro/ability silently map to the right
    slots) and each value is a unified-slot index (right 12-33, left 34-55).
    """
    authored = authored_slots(hand_name)
    right: dict[str, int] = {}
    left: dict[str, int] = {}
    for side, mapping, slot_map in (
        ("right", authored["right"], right),
        ("left", authored["left"], left),
    ):
        offset = _SIDE_TO_UNIFIED_OFFSET[side]
        for yaml_slot, jn in mapping.items():
            slot_map[str(jn)] = yaml_slot + offset
    return right, left


def expected_joint_names(hand_name: str) -> tuple[list[str], list[str]]:
    """Authored (plain) joint names per side, in YAML slot order."""
    authored = authored_slots(hand_name)
    right = [jn for _, jn in sorted(authored["right"].items())]
    left = [jn for _, jn in sorted(authored["left"].items())]
    return right, left


def validate_hand_coverage(
    robot_key: str,
    actual_joint_names: list[str],
    *,
    context: str = "dataset",
) -> dict:
    """Check every authored hand joint resolves against ``actual_joint_names``.

    Raises ``ValueError`` if:
    * the robot_key is unknown, or
    * the hand has no authored rows at all (e.g. an un-authored hand — every
      hand column would be silently 0), or
    * a joint name is authored on BOTH the right and left rows (would make one
      real joint map to two unified slots and leave the other hand at 0), or
    * any authored right/left joint name is missing from the actual names
      (catches naming drift like the allegro ``joint_13.0`` vs ``joint_13_0``
      separator bug, and a right hand that forgot its ``multi_`` prefix).

    Matching is EXACT on both sides: the YAML must hold the verbatim recorded
    name (no automatic ``multi_`` alias).

    Returns per-side matched/total stats (for a startup print).
    """
    hand_name = robot_key_to_hand_name(robot_key)
    right_expected, left_expected = expected_joint_names(hand_name)
    name_set = set(actual_joint_names)

    if not right_expected and not left_expected:
        sample = ", ".join(sorted(name_set)[:8])
        raise ValueError(
            f"[GR00T_XE:{context}] robot_key '{robot_key}' hand '{hand_name}' has NO "
            f"authored joint→slot rows in {_MAPPING_PATH.name}. "
            f"All {len(name_set)} hand columns would be silently 0. "
            f"Recorded joints look like: [{sample}{', ...' if len(name_set) > 8 else ''}]. "
            f"Add '{hand_name}' rows to the YAML (see comment at its bottom for finger order)."
        )

    overlap = sorted(set(right_expected) & set(left_expected))
    if overlap:
        raise ValueError(
            f"[GR00T_XE:{context}] robot_key '{robot_key}' hand '{hand_name}' authors the "
            f"SAME joint name on BOTH the right and left rows: {overlap[:12]}. "
            f"A joint can only live on one hand — one side would silently stay 0. "
            f"Usually the right hand needs its recorded 'multi_' prefix (e.g. "
            f"'joint_13_0' -> 'multi_joint_13_0')."
        )

    def _check(expected: list[str]) -> tuple[int, list[str]]:
        matched = 0
        missing = []
        for jn in expected:
            if jn in name_set:
                matched += 1
            else:
                missing.append(jn)
        return matched, missing

    right_matched, right_missing = _check(right_expected)
    left_matched, left_missing = _check(left_expected)

    problems = []
    if right_missing:
        problems.append(
            f"right hand ({right_matched}/{len(right_expected)} matched) — "
            f"unresolved joint names: {right_missing[:12]}{' ...' if len(right_missing) > 12 else ''}"
        )
    if left_missing:
        problems.append(
            f"left hand ({left_matched}/{len(left_expected)} matched) — "
            f"unresolved joint names: {left_missing[:12]}{' ...' if len(left_missing) > 12 else ''}"
        )
    if problems:
        raise ValueError(
            f"[GR00T_XE:{context}] hand mapping for robot_key '{robot_key}' "
            f"(hand '{hand_name}') does NOT resolve against actual joint names "
            f"(n={len(name_set)}). Refusing to train/eval with silent zero hand "
            f"columns.\n  " + "\n  ".join(problems)
        )

    return {
        "robot_key": robot_key,
        "hand": hand_name,
        "right": right_matched,
        "right_total": len(right_expected),
        "left": left_matched,
        "left_total": len(left_expected),
        "hand_slots_filled": right_matched + left_matched,
    }
