"""Task-specific success evaluator for 21_condiment_box_loading.

Place MSG (chicken bouillon), soy sauce, and vinegar into the wooden box.
All three objects must be inside the box; the box must be upright.
Soy sauce and vinegar must additionally be placed upright (bottles not toppled);
MSG does not need to be upright (it is a small packet, may lie down).

Modes (via params.mode):
  - ``soy``    — soy sauce inside + upright, box upright
  - ``vinegar``— vinegar inside + upright, box upright
  - ``msg``    — MSG inside, box upright (upright not required)
  - ``final`` (default) — all objects inside + static, soy/vinegar upright, box upright
"""

from __future__ import annotations

from typing import Any, Dict

from success.condition_evaluator import evaluate_conditions


BOX_ID = "obj_154_wooden_box_1"
SOY_ID = "obj_172_soy_sauce_2"
VINEGAR_ID = "obj_173_vinegar_3"
MSG_ID = "obj_171_msg_4"

_BOX_UPRIGHT = {
    "type": "object_upright",
    "object": BOX_ID,
    "tolerance_deg": 30,
    "local_axis": "y",
}


def _inside(obj_id: str) -> dict:
    return {
        "type": "object_inside",
        "object": obj_id,
        "container": BOX_ID,
        "tolerance": 0.18,
        "container_center_offset": [0.0, 0.075, 0.0],
        "object_center_offset": {
            SOY_ID: [0.0, 0.075, 0.0],
            VINEGAR_ID: [-8.6e-06, 0.1125, -2.6e-06],
            MSG_ID: [-2.0e-06, 0.0475284, 2.2e-06],
        }[obj_id],
    }


def _upright(obj_id: str) -> dict:
    """Bottles spawn upright with local y aligned to world +z."""
    return {
        "type": "object_upright",
        "object": obj_id,
        "tolerance_deg": 30,
        "local_axis": "y",
    }


def _static(obj_id: str) -> dict:
    return {
        "type": "object_static",
        "object": obj_id,
        "threshold": 0.05,
        "check_angular": False,
    }


def check_success(states: Dict[str, Any], ctx: Dict[str, Any], params: Dict[str, Any]) -> bool:
    mode = str(params.get("mode", "final"))

    if mode == "soy":
        return evaluate_conditions(
            [_BOX_UPRIGHT, _inside(SOY_ID), _upright(SOY_ID)], states, ctx
        )
    if mode == "vinegar":
        return evaluate_conditions(
            [_BOX_UPRIGHT, _inside(VINEGAR_ID), _upright(VINEGAR_ID)], states, ctx
        )
    if mode == "msg":
        return evaluate_conditions(
            [_BOX_UPRIGHT, _inside(MSG_ID)], states, ctx
        )

    # final
    return evaluate_conditions(
        [
            _BOX_UPRIGHT,
            _inside(SOY_ID), _upright(SOY_ID),
            _inside(VINEGAR_ID), _upright(VINEGAR_ID),
            _inside(MSG_ID),
            _static(SOY_ID), _static(VINEGAR_ID), _static(MSG_ID),
        ],
        states,
        ctx,
    )
