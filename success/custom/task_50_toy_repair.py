"""Task 50 - Toy Repair and Drawer Storage.

Success: the repair gesture has been observed, then drawer is opened,
and finally the toy is stored while the bottom drawer is open.
"""

from __future__ import annotations

import math
from typing import Any, Dict

from success.state_utils import get_joint_position


# The first drawer group in the asset is treated as the bottom drawer.
BOTTOM_DRAWER_JOINT = "joint_0"


def check_success(
    states: Dict[str, Any],
    ctx: Dict[str, Any],
    params: Dict[str, Any],
) -> bool:
    wrench = states.get("obj_042_adjustable_wrench_1")
    toy = states.get("obj_072e_toy_airplane_2")
    storage = states.get("obj_116_storagefurniture_3")
    if not all([wrench, toy, storage]):
        return False

    wp = wrench["pose_world"]
    tp = toy["pose_world"]
    sto_p = storage["pose_world"]

    # Step tracking: 0 = repair, 1 = open drawer, 2 = store toy
    step = int(ctx.get("_t50_step", 0))

    # Step 0: Repair - wrench near toy for 20 frames
    if step == 0:
        d_wt = math.hypot(float(wp[0] - tp[0]), float(wp[1] - tp[1]))
        if d_wt < 0.10:
            ctx["_t50_repair_near"] = int(ctx.get("_t50_repair_near", 0)) + 1
        if int(ctx.get("_t50_repair_near", 0)) >= 20:
            ctx["_t50_step"] = 1
            step = 1

    # Step 1: Open drawer - at least 15cm (0.15)
    if step == 1:
        bottom_drawer_open = abs(get_joint_position(storage, BOTTOM_DRAWER_JOINT, 0.0)) > 0.15
        if bottom_drawer_open:
            ctx["_t50_step"] = 2
            step = 2

    # Step 2: Store toy - toy near storage while drawer is open
    if step == 2:
        bottom_drawer_open = abs(get_joint_position(storage, BOTTOM_DRAWER_JOINT, 0.0)) > 0.15
        d_ts = math.hypot(float(tp[0] - sto_p[0]), float(tp[1] - sto_p[1]))
        toy_stored = bottom_drawer_open and d_ts < 0.22
        if toy_stored:
            return True

    return False
