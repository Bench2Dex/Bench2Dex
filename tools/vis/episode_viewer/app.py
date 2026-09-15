#!/usr/bin/env python3
"""Interactive Flask viewer for one HDF5 episode or a directory of episodes.

Run one file:
    python tools/vis/episode_viewer/app.py \
      --hdf5 /path/to/episode_000000_replay.hdf5 \
      --host 0.0.0.0 \
      --port 8080

Run a directory:
    python tools/vis/episode_viewer/app.py \
      --hdf5  ../output/eval\
      --host 0.0.0.0 \
      --port 8088


When --hdf5 points to a directory, the viewer shows a one-level-at-a-time
browser in the UI toolbar. Open subdirectories to inspect nested .hdf5 files.
Deleting a file or folder from the UI moves it into this tool's deleted_items
directory instead of permanently removing it.

Open after launch:
    http://127.0.0.1:8080
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime
import io
import shutil
import threading
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request, send_file

try:
    from .hdf5_episode import DEFAULT_HDF5, EpisodeHDF5, placeholder_png
except ImportError:
    from hdf5_episode import DEFAULT_HDF5, EpisodeHDF5, placeholder_png


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve an interactive HDF5 episode viewer.")
    parser.add_argument("--hdf5", type=Path, default=DEFAULT_HDF5, help="Input HDF5 episode file or directory.")
    parser.add_argument("--host", default="0.0.0.0", help="Flask host.")
    parser.add_argument("--port", type=int, default=8080, help="Flask port.")
    parser.add_argument("--debug", action="store_true", help="Run Flask in debug mode.")
    return parser.parse_args()


class EpisodeStore:
    """Owns the selectable episode list and the currently active reader."""

    def __init__(self, source: str | Path, deleted_dir: str | Path | None = None) -> None:
        self.source = Path(source).expanduser().resolve()
        self._lock = threading.RLock()
        self.deleted_dir = (
            Path(deleted_dir).expanduser().resolve()
            if deleted_dir is not None
            else (Path(__file__).resolve().parent / "deleted_items").resolve()
        )
        self.root, initial_file = self._discover_root_and_initial_file(self.source)
        self._current_directory = initial_file.parent if initial_file else self.root
        self._current_path: Path | None = None
        self._current_episode: EpisodeHDF5 | None = None
        self._select_first_episode(initial_file)

    @staticmethod
    def _discover_root_and_initial_file(source: Path) -> tuple[Path, Path | None]:
        if source.is_file():
            if source.suffix.lower() != ".hdf5":
                raise FileNotFoundError(f"Input file is not a .hdf5 episode: {source}")
            return source.parent, source
        if source.is_dir():
            return source, None
        raise FileNotFoundError(f"HDF5 file or directory not found: {source}")

    def episode_list(self) -> dict:
        with self._lock:
            self._ensure_current_episode()
            return self._episode_list_unlocked()

    def _episode_list_unlocked(self) -> dict:
        current_dir = self._current_directory
        entries = self._entries_for_directory(current_dir)
        episodes = [entry for entry in entries if entry["type"] == "file"]
        current_rel = self._relative_text(self._current_path) if self._current_path else None
        return {
            "current_index": self._current_index_for_episodes(episodes, current_rel),
            "current_file": current_rel,
            "current_directory": self._relative_text(current_dir),
            "root": str(self.root),
            "source": str(self.source),
            "source_is_directory": self.source.is_dir(),
            "deleted_dir": str(self.deleted_dir),
            "can_go_up": current_dir != self.root,
            "parent_directory": self._relative_text(current_dir.parent) if current_dir != self.root else None,
            "entries": entries,
            "episodes": episodes,
        }

    def current(self) -> EpisodeHDF5:
        with self._lock:
            self._ensure_current_episode()
            if self._current_episode is None:
                raise FileNotFoundError("No .hdf5 episode selected in the current directory.")
            return self._current_episode

    def select(self, index: int) -> dict:
        with self._lock:
            episodes = [entry for entry in self._entries_for_directory(self._current_directory) if entry["type"] == "file"]
            if index < 0 or index >= len(episodes):
                raise IndexError(f"Episode index out of range: {index}")
            self._set_current_file(self._resolve_relative(episodes[index]["path"]))
            return self._episode_list_unlocked()

    def select_path(self, relative_path: str) -> dict:
        with self._lock:
            path = self._resolve_relative(relative_path)
            if not path.is_file() or path.suffix.lower() != ".hdf5":
                raise FileNotFoundError(f"HDF5 episode file not found: {relative_path}")
            self._current_directory = path.parent
            self._set_current_file(path)
            return self._episode_list_unlocked()

    def open_directory(self, relative_path: str | None) -> dict:
        with self._lock:
            path = self.root if relative_path in (None, "", ".") else self._resolve_relative(str(relative_path))
            if not path.is_dir():
                # Directory was deleted (e.g. inference tmp dir removed) —
                # silently fall back to the initial --hdf5 root directory.
                self._current_directory = self.root
                self._select_first_episode()
                return self._episode_list_unlocked()
            if self._is_deleted_path(path):
                raise PermissionError("The deleted_items directory cannot be opened from the episode browser.")
            self._current_directory = path
            self._select_first_episode()
            return self._episode_list_unlocked()

    def move_to_deleted(self, relative_path: str) -> dict:
        with self._lock:
            target = self._resolve_relative(relative_path)
            if target == self.root:
                raise PermissionError("Cannot delete the browser root directory.")
            if self._is_deleted_path(target) or self._path_contains(target, self.deleted_dir):
                raise PermissionError("Cannot delete or move the deleted_items directory.")
            if not target.exists():
                raise FileNotFoundError(f"Path not found: {relative_path}")

            current_was_inside_target = self._current_path is not None and (
                self._current_path == target or self._path_contains(target, self._current_path)
            )
            current_dir_was_inside_target = (
                self._current_directory == target or self._path_contains(target, self._current_directory)
            )
            destination = self._unique_deleted_path(target)
            self.deleted_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(target), str(destination))

            if current_dir_was_inside_target:
                self._current_directory = target.parent if self._is_under_root(target.parent) else self.root
            if current_was_inside_target:
                self._select_first_episode()
            else:
                self._ensure_current_episode()
            return {
                "deleted_path": str(destination),
                "deleted_name": destination.name,
                "browser": self._episode_list_unlocked(),
            }

    def _entries_for_directory(self, directory: Path) -> list[dict]:
        if not directory.exists() or not directory.is_dir():
            return []
        child_dirs = []
        child_files = []
        for path in directory.iterdir():
            resolved = path.resolve()
            if self._is_deleted_path(resolved):
                continue
            if resolved.is_dir():
                child_dirs.append(resolved)
            elif resolved.is_file() and resolved.suffix.lower() == ".hdf5":
                child_files.append(resolved)
        entries: list[dict] = []
        for path in sorted(child_dirs, key=lambda item: item.name.lower()):
            entries.append(
                {
                    "type": "directory",
                    "name": path.name,
                    "path": self._relative_text(path),
                    "deletable": path != self.root,
                }
            )
        for idx, path in enumerate(sorted(child_files, key=lambda item: item.name.lower())):
            entries.append(
                {
                    "type": "file",
                    "index": idx,
                    "file_name": path.name,
                    "name": path.name,
                    "path": self._relative_text(path),
                    "absolute_path": str(path),
                    "deletable": True,
                }
            )
        return entries

    def _ensure_current_episode(self) -> None:
        if self._current_path is not None and self._current_path.exists():
            return
        self._select_first_episode()

    def _select_first_episode(self, preferred: Path | None = None) -> None:
        if preferred is not None and preferred.exists():
            self._set_current_file(preferred)
            return
        episodes = [entry for entry in self._entries_for_directory(self._current_directory) if entry["type"] == "file"]
        if episodes:
            self._set_current_file(self._resolve_relative(episodes[0]["path"]))
            return
        self._current_path = None
        self._current_episode = None

    def _set_current_file(self, path: Path) -> None:
        resolved = path.resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Episode file not found: {path}")
        self._current_path = resolved
        self._current_episode = EpisodeHDF5(resolved)

    def _current_index_for_episodes(self, episodes: list[dict], current_rel: str | None) -> int:
        if current_rel is None:
            return -1
        for idx, episode in enumerate(episodes):
            if episode["path"] == current_rel:
                return idx
        return -1

    def _resolve_relative(self, relative_path: str) -> Path:
        text = str(relative_path or "").replace("\\", "/")
        if text in ("", "."):
            return self.root
        # Absolute path (e.g. from a symlink pointing outside the root).
        if text.startswith("/"):
            path = Path(text).resolve()
            if not path.exists():
                raise FileNotFoundError(f"Absolute path not found: {relative_path}")
            return path
        text = text.lstrip("/")
        if text in ("", "."):
            return self.root
        path = (self.root / text).resolve()
        if not self._is_under_root(path):
            raise PermissionError(f"Path escapes browser root: {relative_path}")
        return path

    def _relative_text(self, path: Path | None) -> str:
        if path is None:
            return ""
        resolved = path.resolve()
        if resolved == self.root:
            return ""
        try:
            return resolved.relative_to(self.root).as_posix()
        except ValueError:
            # Path is outside root (e.g. symlink target); return absolute path.
            return resolved.as_posix()

    def _unique_deleted_path(self, source: Path) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        if source.is_file():
            base_name = f"{source.stem}_{timestamp}{source.suffix}"
        else:
            base_name = f"{source.name}_{timestamp}"
        candidate = self.deleted_dir / base_name
        counter = 1
        while candidate.exists():
            if source.is_file():
                candidate = self.deleted_dir / f"{source.stem}_{timestamp}_{counter}{source.suffix}"
            else:
                candidate = self.deleted_dir / f"{source.name}_{timestamp}_{counter}"
            counter += 1
        return candidate

    def _is_under_root(self, path: Path) -> bool:
        return self._path_contains(self.root, path.resolve())

    def _is_deleted_path(self, path: Path) -> bool:
        resolved = path.resolve()
        return resolved == self.deleted_dir or self._path_contains(self.deleted_dir, resolved)

    @staticmethod
    def _path_contains(parent: Path, child: Path) -> bool:
        try:
            child.resolve().relative_to(parent.resolve())
            return True
        except ValueError:
            return False


def create_app(hdf5_path: str | Path) -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    episode_store = EpisodeStore(hdf5_path)

    @app.after_request
    def add_no_cache_headers(response: Response) -> Response:
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/api/summary")
    def api_summary():
        try:
            return jsonify(episode_store.current().summary())
        except FileNotFoundError as exc:
            return jsonify({"error": str(exc)}), 404

    @app.route("/api/episodes")
    def api_episodes():
        return jsonify(episode_store.episode_list())

    @app.route("/api/episode/select", methods=["POST"])
    def api_select_episode():
        payload = request.get_json(silent=True) or {}
        try:
            if "path" in payload:
                selected = episode_store.select_path(str(payload.get("path") or ""))
            else:
                index = int(payload.get("index"))
                selected = episode_store.select(index)
        except (TypeError, ValueError, IndexError, FileNotFoundError, PermissionError) as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(selected)

    @app.route("/api/directory/open", methods=["POST"])
    def api_open_directory():
        payload = request.get_json(silent=True) or {}
        try:
            selected = episode_store.open_directory(payload.get("path"))
        except (NotADirectoryError, FileNotFoundError, PermissionError) as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(selected)

    @app.route("/api/episode/delete", methods=["POST"])
    def api_delete_episode_entry():
        payload = request.get_json(silent=True) or {}
        try:
            result = episode_store.move_to_deleted(str(payload.get("path") or ""))
        except (FileNotFoundError, PermissionError, OSError) as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(result)

    @app.route("/api/frame/<int:idx>")
    def api_frame(idx: int):
        try:
            return jsonify(episode_store.current().frame(idx))
        except FileNotFoundError as exc:
            return jsonify({"error": str(exc)}), 404

    @app.route("/api/frame_bundle/<int:idx>")
    def api_frame_bundle(idx: int):
        """Single round-trip: frame summary + base64-encoded image data URLs.

        Query params:
          tabs=rgb,depth,occupancy,box2d,box3d   (default: rgb)
          include_tactile=0|1                    (default: 0)

        Frontend uses this to preload the whole episode into memory and play
        without per-frame HTTP latency.
        """
        try:
            episode = episode_store.current()
            frame_data = episode.frame(idx)
            summary = episode.summary()
        except FileNotFoundError as exc:
            return jsonify({"error": str(exc)}), 404

        tabs_param = request.args.get("tabs", "rgb")
        tabs = [t for t in (tab.strip() for tab in tabs_param.split(",")) if t]
        include_tactile = request.args.get("include_tactile", "0") == "1"

        cam_ids = summary.get("camera_ids", [])
        images: dict[str, dict[str, str | None]] = {}
        for tab in tabs:
            bucket: dict[str, str | None] = {}
            for cam in cam_ids:
                bucket[cam] = _encode_data_url(_render_tab_image(episode, tab, cam, idx))
            images[tab] = bucket

        tactile: dict[str, str | None] = {}
        if include_tactile:
            for site in summary.get("tactile_sites", []):
                try:
                    payload = episode.tactile_image(site, idx)
                except Exception:
                    payload = None
                tactile[site] = _encode_data_url(payload)

        return jsonify({"frame": frame_data, "images": images, "tactile": tactile})

    @app.route("/api/tactile/contact_summary")
    def api_contact_summary():
        try:
            return jsonify(episode_store.current().contact_summary())
        except FileNotFoundError as exc:
            return jsonify({"error": str(exc)}), 404

    @app.route("/image/rgb/<cam_id>/<int:idx>")
    def image_rgb(cam_id: str, idx: int):
        return _send_image(*_guard_image(lambda: episode_store.current().rgb_image(cam_id, idx)))

    @app.route("/image/depth/<cam_id>/<int:idx>.png")
    def image_depth(cam_id: str, idx: int):
        return _send_image(*_guard_image(lambda: episode_store.current().depth_image(cam_id, idx)))

    @app.route("/image/occupancy/<cam_id>/<int:idx>.png")
    def image_occupancy(cam_id: str, idx: int):
        return _send_image(*_guard_image(lambda: episode_store.current().occupancy_image(cam_id, idx)))

    @app.route("/image/tactile/<site>/<int:idx>.png")
    def image_tactile(site: str, idx: int):
        return _send_image(*_guard_image(lambda: episode_store.current().tactile_image(site, idx)))

    @app.route("/image/box/<cam_id>/<int:idx>.png")
    def image_box(cam_id: str, idx: int):
        return _send_image(*_guard_image(lambda: episode_store.current().box_image(cam_id, idx)))

    @app.route("/image/box/<mode>/<cam_id>/<int:idx>.png")
    def image_box_mode(mode: str, cam_id: str, idx: int):
        return _send_image(*_guard_image(lambda: episode_store.current().box_image(cam_id, idx, mode=mode)))

    return app


def _guard_image(fn) -> tuple[bytes, str]:
    try:
        return fn()
    except Exception as exc:
        return placeholder_png("Render error", str(exc)), "image/png"


def _render_tab_image(episode, tab: str, cam: str, idx: int) -> tuple[bytes, str] | None:
    try:
        if tab == "rgb":
            return episode.rgb_image(cam, idx)
        if tab == "depth":
            return episode.depth_image(cam, idx)
        if tab == "occupancy":
            return episode.occupancy_image(cam, idx)
        if tab == "box2d":
            return episode.box_image(cam, idx, mode="box2d")
        if tab == "box3d":
            return episode.box_image(cam, idx, mode="box3d")
    except Exception:
        return None
    return None


def _encode_data_url(payload: tuple[bytes, str] | None) -> str | None:
    if payload is None:
        return None
    data, mime = payload
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def _send_image(data: bytes, mimetype: str):
    return send_file(io.BytesIO(data), mimetype=mimetype)


def main() -> None:
    args = parse_args()
    app = create_app(args.hdf5)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
