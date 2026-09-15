import math

from success.custom import task_12_screwdriver_box_and_hammer as task12


def _state(x, y, z, lin=(0.0, 0.0, 0.0), ang=(0.0, 0.0, 0.0), quat=(0.0, 0.0, 0.0, 1.0)):
    return {
        "pose_world": [x, y, z, *quat],
        "lin_vel_world": list(lin),
        "ang_vel_world": list(ang),
    }


def _task_states(
    *,
    tools_placed=True,
    hammer_lin=(0.0, 0.0, 0.0),
    block_lin=(0.0, 0.0, 0.0),
    block_ang=(0.0, 0.0, 0.0),
):
    half_sqrt = math.sqrt(0.5)
    box = _state(0.50, 0.27, 0.75, quat=(half_sqrt, 0.0, 0.0, half_sqrt))
    if tools_placed:
        phillips = _state(0.48, 0.26, 0.79)
        flat = _state(0.52, 0.28, 0.79)
    else:
        phillips = _state(-0.45, 0.12, 0.79)
        flat = _state(0.25, 0.07, 0.79)

    return {
        task12.PHILLIPS: phillips,
        task12.FLAT: flat,
        task12.BOX: box,
        task12.HAMMER: _state(-0.14, 0.10, 0.80, lin=hammer_lin),
        task12.WOOD_BLOCK: _state(-0.10, 0.10, 0.76, lin=block_lin, ang=block_ang),
    }


def test_hammer_near_and_fast_is_not_a_strike_without_block_response():
    ctx = {"dt": 0.05}

    near_fast = _task_states(hammer_lin=(0.35, 0.0, 0.0))

    assert task12.check_success(near_fast, ctx, {}) is False
    assert task12.check_success(near_fast, ctx, {"mode": "hammer_strike"}) is False


def test_strike_requires_nearby_swing_and_block_response():
    ctx = {"dt": 0.05}

    assert task12.check_success(
        _task_states(hammer_lin=(0.35, 0.0, 0.0)),
        ctx,
        {},
    ) is False

    assert task12.check_success(
        _task_states(block_ang=(0.0, 0.0, 0.5)),
        ctx,
        {"mode": "hammer_strike"},
    ) is True


def test_preplacement_hammer_motion_is_not_counted_after_tools_are_placed():
    ctx = {"dt": 0.05}

    assert task12.check_success(
        _task_states(
            tools_placed=False,
            hammer_lin=(0.35, 0.0, 0.0),
            block_ang=(0.0, 0.0, 0.5),
        ),
        ctx,
        {},
    ) is False

    assert task12.check_success(_task_states(), ctx, {}) is False
