"""Task manifest helpers for benchmark metadata."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_README_ROW_RE = re.compile(r"^\|\s*(\d+)\s*\|")
_SCENE_FILE_RE = re.compile(r"^(\d{2})_(.+)\.ya?ml$")
_SCENE_DIRS = (("scenes_final", 0), ("scenes", 1))


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _parse_readme_table(readme_path: Path) -> dict[int, dict[str, Any]]:
    if not readme_path.exists():
        return {}
    lines = readme_path.read_text(encoding="utf-8").splitlines()
    entries: dict[int, dict[str, Any]] = {}
    for line in lines:
        if not _README_ROW_RE.match(line):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 6:
            continue
        task_id = int(cells[0])
        entries[task_id] = {
            "task_id": task_id,
            "domain": cells[1],
            "task_name": cells[2],
            "difficulty": cells[3],
            "task_type_tags": [tag.strip() for tag in cells[4].split("+") if tag.strip()],
            "manipulation_tags": [tag.strip() for tag in cells[5].split(",") if tag.strip()],
        }
    return entries


def _scene_flags(scene_path: Path, task_type_tags: list[str]) -> dict[str, Any]:
    scene = yaml.safe_load(scene_path.read_text(encoding="utf-8")) or {}
    assets = scene.get("assets", {}) or {}
    contains_articulation = any(
        str(spec.get("body_type", "dynamic")).lower() == "articulation"
        for spec in assets.values()
        if isinstance(spec, dict)
    )
    return {
        "contains_articulation": contains_articulation,
        "contains_tool_use": "TU" in task_type_tags,
        "contains_sequential": "SP" in task_type_tags,
        "contains_long_range": "LR" in task_type_tags,
    }


def _iter_scene_files(root: Path):
    for dir_name, priority in _SCENE_DIRS:
        scenes_dir = root / dir_name
        if not scenes_dir.exists():
            continue
        for path in sorted(scenes_dir.glob("*.yaml")):
            match = _SCENE_FILE_RE.match(path.name)
            if not match:
                continue
            task_id = int(match.group(1))
            if task_id <= 0:
                continue
            yield priority, dir_name, path, task_id


@lru_cache(maxsize=1)
def load_task_manifest() -> dict[str, dict[str, Any]]:
    root = _repo_root()
    readme_entries = _parse_readme_table(root / "README.md")
    readme_entries.update(_parse_readme_table(root / "scenes" / "task.md"))

    manifest: dict[str, dict[str, Any]] = {}
    priorities: dict[str, int] = {}
    for priority, dir_name, scene_path, task_id in _iter_scene_files(root):
        scene_name = scene_path.stem
        existing_priority = priorities.get(scene_name)
        if existing_priority is not None and existing_priority <= priority:
            continue

        entry = dict(readme_entries.get(task_id, {"task_id": task_id}))
        entry.setdefault("domain", "Unknown")
        entry.setdefault("task_name", scene_name)
        entry.setdefault("difficulty", "Unknown")
        entry.setdefault("task_type_tags", [])
        entry.setdefault("manipulation_tags", [])
        entry["scene_name"] = scene_name
        entry["scene_file"] = f"{dir_name}/{scene_path.name}"
        entry.update(_scene_flags(scene_path, entry["task_type_tags"]))
        manifest[scene_name] = entry
        priorities[scene_name] = priority
    return manifest


def get_task_manifest_entry(scene_name: str) -> dict[str, Any] | None:
    return load_task_manifest().get(scene_name)

