"""Task-specific success evaluator for 61_medicine_shoebox_pack.

Success: all three objects inside the shoe box (rectangular container,
object_in_container_zone with container_frame:true).

Container params derived from USD geometry:
  origin at bottom, rpy_deg:[90,0,0], scale:[0.45,0.45,0.45]
  interior: 0.161(w)×0.103(h)×0.282(d), 3mm walls
  container_center_offset: [0, 0.055, 0] (bottom→interior center)
  zone: ±0.11(x) × [-0.06, 0.12](y) × ±0.15(z), y widened for open top
"""

from __future__ import annotations
from typing import Any, Dict
from success.condition_evaluator import evaluate_conditions

BOX = "obj_129_shoe_box_1"
_Z = {"container_center_offset": [0.0, 0.055, 0.0], "container_frame": True,
      "zone_lo": [-0.11, -0.06, -0.15], "zone_hi": [0.11, 0.12, 0.15]}
_UP = {"type": "object_upright", "object": BOX, "tolerance_deg": 30, "local_axis": "y"}

CONDITIONS = [{"type": "all", "conditions": [
    {"type": "all", "conditions": [_UP, {"type": "object_in_container_zone", "object": "obj_186_pillbottle_2", "container": BOX, **_Z}]},
    {"type": "all", "conditions": [_UP, {"type": "object_in_container_zone", "object": "obj_218_tooth_paste_3", "container": BOX, **_Z}]},
    {"type": "all", "conditions": [_UP, {"type": "object_in_container_zone", "object": "obj_211_hydrating_oil_4", "container": BOX, **_Z}]},
]}]

def check_success(states: Dict[str, Any], ctx: Dict[str, Any], params: Dict[str, Any]) -> bool:
    return evaluate_conditions(CONDITIONS, states, ctx)
