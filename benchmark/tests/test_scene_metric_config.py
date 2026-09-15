from pathlib import Path

import yaml


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
