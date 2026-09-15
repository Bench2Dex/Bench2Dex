"""Success evaluator for 79_bimanual_piano_melody.

Both hands play their melody sequence on the piano.
"""

from __future__ import annotations

from typing import Any, Dict

from success.state_utils import get_joint_position


StateMap = Dict[str, Any]


_LEFT_MELODY = (18, 18, 23, 23, 24, 24, 23)


_RIGHT_MELODY = (33, 33, 38, 38, 39, 39, 38)


_PIANO_PRESS_THRESHOLD = 0.004


_PIANO_RELEASE_THRESHOLD = 0.002


def _has(states: StateMap, *object_ids: str) -> bool:
    return all(object_id in states and states[object_id].get("pose_world") is not None for object_id in object_ids)


def _joint(states: StateMap, object_id: str, joint_name: str) -> float:
    state = states.get(object_id)
    if state is None:
        return 0.0
    return get_joint_position(state, joint_name, 0.0)


def _bimanual_melody_side(
    states: StateMap,
    ctx: Dict[str, Any],
    *,
    key: str,
    melody: tuple[int, ...],
) -> bool:
    """Track one hand's melody sequence.

    Only the *expected* note advances progress; pressing other keys is ignored
    and does NOT reset the sequence.  A note is "pressed" when its joint qpos
    crosses above ``_PIANO_PRESS_THRESHOLD`` from below ``_PIANO_RELEASE_THRESHOLD``.
    """
    piano = "obj_325_piano_1"
    index_key = f"{key}_index"
    armed_key = f"{key}_armed"
    index = int(ctx.get(index_key, 0))
    if index >= len(melody):
        return True

    expected_num = melody[index]
    joint_name = f"joint_{expected_num}_to_1"
    value = _joint(states, piano, joint_name)
    armed = bool(ctx.get(armed_key, True))

    if not armed and value < _PIANO_RELEASE_THRESHOLD:
        armed = True
    if armed and value > _PIANO_PRESS_THRESHOLD:
        index += 1
        armed = False

    ctx[armed_key] = armed
    ctx[index_key] = index
    return index >= len(melody)


def check_success(states: StateMap, ctx: Dict[str, Any], params: Dict[str, Any]) -> bool:
    phase = str((params or {}).get("phase", "terminal"))
    if not _has(states, "obj_325_piano_1"):
        return False
    left_ok = _bimanual_melody_side(states, ctx, key="left", melody=_LEFT_MELODY)
    right_ok = _bimanual_melody_side(states, ctx, key="right", melody=_RIGHT_MELODY)
    if phase == "melody_left":
        return left_ok
    if phase == "melody_right":
        return right_ok
    # Terminal: both hands complete the melody
    return left_ok and right_ok
