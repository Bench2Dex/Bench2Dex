from __future__ import annotations

from pathlib import Path


def _normalize_scene_name(scene_name: str | None) -> str | None:
    if not scene_name:
        return None
    return scene_name if scene_name.endswith('.yaml') else f'{scene_name}.yaml'



def resolve_scene_yaml_path(
    *,
    cli_scene_path: str | None,
    recorded_scene_path: str | None,
    recorded_scene_name: str | None,
    script_dir: str,
) -> str:
    """Resolve the scene yaml path for replay.

    Resolution order:
    1. Explicit ``--scene`` CLI override.
    2. Recorded absolute/relative path from HDF5 if it still exists.
    3. Local ``<script_dir>/scenes/<scene_name>.yaml`` fallback.
    4. Local ``<script_dir>/scenes/<basename(recorded_scene_path)>`` fallback.
    """

    if cli_scene_path:
        cli_path = Path(cli_scene_path).expanduser()
        if not cli_path.is_file():
            raise FileNotFoundError(f'Explicit scene path not found: {cli_scene_path}')
        return str(cli_path.resolve())

    candidates: list[Path] = []
    if recorded_scene_path:
        candidates.append(Path(recorded_scene_path).expanduser())

    scenes_dir = Path(script_dir) / 'scenes'
    normalized_scene_name = _normalize_scene_name(recorded_scene_name)
    if normalized_scene_name:
        candidates.append(scenes_dir / normalized_scene_name)

    if recorded_scene_path:
        basename = Path(recorded_scene_path).name
        if basename:
            candidates.append(scenes_dir / basename)

    seen: set[str] = set()
    unique_candidates: list[Path] = []
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique_candidates.append(candidate)

    for candidate in unique_candidates:
        if candidate.is_file():
            return str(candidate.resolve())

    candidate_text = '\n'.join(f'  - {candidate}' for candidate in unique_candidates) or '  <none>'
    raise FileNotFoundError(
        'Unable to resolve scene yaml path for replay. Checked:\n'
        f'{candidate_text}'
    )
