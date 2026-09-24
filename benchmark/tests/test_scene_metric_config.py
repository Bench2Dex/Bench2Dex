from pathlib import Path

import pytest
import yaml

from benchmark.stage_tracker import StageTracker
from success.engine import evaluate_condition_node


SCENE_DIR = Path(__file__).resolve().parents[2] / "scenes"


def _official_scene_paths():
    return sorted(
        path
        for path in SCENE_DIR.glob("[0-9][0-9]_*.yaml")
        if not path.name.startswith("00_")
    )


def test_official_scenes_declare_grasp_and_drop_tracked_objects():
    scene_paths = _official_scene_paths()
    assert scene_paths

    for path in scene_paths:
        scene = yaml.safe_load(path.read_text())
        object_ids = {str(obj["id"]) for obj in scene.get("objects", [])}
        metrics = scene.get("metrics") or {}
        grasp = metrics.get("grasp") or {}
        safety = metrics.get("safety") or {}
        drop = safety.get("drop") or {}

        grasp_tracked = [str(obj_id) for obj_id in grasp.get("tracked_objects") or []]
        drop_tracked = [str(obj_id) for obj_id in drop.get("tracked_objects") or []]

        assert safety.get("table_z") is not None, path
        if not grasp_tracked:
            assert grasp.get("enabled") is False, path
            assert drop.get("enabled") is False, path
            assert not drop_tracked, path
            continue

        assert grasp.get("enabled") is True, path
        assert drop.get("enabled") is True, path
        assert drop_tracked == grasp_tracked, path
        assert set(grasp_tracked) <= object_ids, path

        allowed = drop.get("allowed_placed_conditions") or {}
        assert set(str(obj_id) for obj_id in allowed) <= set(drop_tracked), path


def test_task44_metrics_follow_the_demonstrated_task_order_and_cover_both_payloads():
    path = SCENE_DIR / "44_microwave_bowl_loading.yaml"
    scene = yaml.safe_load(path.read_text())
    metrics = scene["metrics"]

    assert [stage["id"] for stage in metrics["stages"]] == [
        "open_door",
        "baguette_in_bowl",
        "loaded_bowl_in_mw",
        "close_door",
    ]
    tracked = {"obj_024_bowl_3", "obj_163_baguette_1"}
    assert set(metrics["grasp"]["tracked_objects"]) == tracked
    assert set(metrics["safety"]["drop"]["tracked_objects"]) == tracked
    assert set(metrics["safety"]["high_speed"]["tracked_objects"]) == tracked
    assert set(metrics["safety"]["drop"]["allowed_placed_conditions"]) == tracked


@pytest.mark.parametrize(
    ("cleaner_quat", "soap_x", "expected_rate"),
    [
        pytest.param((2**-0.5, 0.0, 0.0, 2**-0.5), 0.5, 0.5, id="upright-cleaner-only"),
        pytest.param((2**-0.5, 0.0, 0.0, 2**-0.5), 0.02, 1.0, id="both-objects-placed"),
        pytest.param((0.0, 0.0, 0.0, 1.0), 0.02, 0.0, id="cleaner-on-its-side"),
    ],
)
def test_task09_stage_progress_uses_the_cleaners_local_y_upright_axis(
    cleaner_quat, soap_x, expected_rate
):
    scene = yaml.safe_load((SCENE_DIR / "09_cleaner_moisturizer_box_loading.yaml").read_text())
    metrics = scene["metrics"]
    # The box and cleaner assets stand upright after a 90-degree X rotation:
    # local Y points up, while local Z is horizontal.
    upright_quat = (2**-0.5, 0.0, 0.0, 2**-0.5)
    states = {
        object_id: {
            "pose_world": [x, 0.0, z, *quat],
            "lin_vel_world": [0.0, 0.0, 0.0],
            "ang_vel_world": [0.0, 0.0, 0.0],
        }
        for object_id, x, z, quat in [
            ("obj_154_wooden_box_1", 0.0, 0.75, upright_quat),
            ("obj_200_cleaner_2", 0.01, 0.80, cleaner_quat),
            ("obj_209_soap_3", soap_x, 0.80, upright_quat),
        ]
    }

    result = StageTracker(metrics["stages"]).update(states, sim_step=0)

    assert result.latched_stage_completion_rate == expected_rate
    assert result.completed == {
        "place_cleaner": expected_rate >= 0.5,
        "place_soap": expected_rate == 1.0,
    }
    assert evaluate_condition_node(metrics["terminal"]["raw_condition"], states, {}) == (
        expected_rate == 1.0
    )
