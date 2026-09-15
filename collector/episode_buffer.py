"""In-memory episode buffer with optional frame cap to prevent OOM."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Dict, List
import numpy as np

DEFAULT_MAX_FRAMES = 50_000
_MEMORY_WARN_INTERVAL = 5_000


@dataclass
class FrameRecord:
    """Dense frame storage: each frame has its own observation and optional action."""
    frame_index: int
    timestamp_ns: int
    sim_step: int
    valid: bool
    errors: List[str]
    camera: Dict
    robot: Dict | None
    objects: Dict
    labels: Dict
    # New fields for v2
    action_commanded: np.ndarray | None = None  # [joint_dim]
    action_valid: bool = True
    action_source: str | None = None  # "scripted" | "teleop"
    teleop_snapshot: dict | None = None


@dataclass
class EpisodeBuffer:
    camera_ids: list[str]
    object_ids: list[str]
    max_frames: int = DEFAULT_MAX_FRAMES
    frames: list[FrameRecord] = field(default_factory=list)
    _dropped_count: int = field(default=0, repr=False)

    def add_frame(self, frame: FrameRecord) -> None:
        if 0 < self.max_frames <= len(self.frames):
            if self._dropped_count == 0:
                print(
                    f"[WARN] EpisodeBuffer reached max_frames={self.max_frames}. "
                    "Subsequent frames will be dropped to prevent OOM.",
                    file=sys.stderr,
                )
            self._dropped_count += 1
            if self._dropped_count % _MEMORY_WARN_INTERVAL == 0:
                print(
                    f"[WARN] EpisodeBuffer: {self._dropped_count} frames dropped so far.",
                    file=sys.stderr,
                )
            return
        self.frames.append(frame)

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def dropped_count(self) -> int:
        return self._dropped_count

    @property
    def error_count(self) -> int:
        return sum(1 for frame in self.frames if not frame.valid)

    # Helper methods for v2 schema
    def build_frame_index(self) -> np.ndarray:
        """Build frame index array [0, 1, 2, ..., n-1]."""
        return np.arange(len(self.frames), dtype=np.int32)

    def build_is_first(self) -> np.ndarray:
        """Build is_first array - True only for first frame."""
        n = len(self.frames)
        flags = np.zeros(n, dtype=np.bool_)
        if n > 0:
            flags[0] = True
        return flags

    def build_is_last(self) -> np.ndarray:
        """Build is_last array - True only for last frame."""
        n = len(self.frames)
        flags = np.zeros(n, dtype=np.bool_)
        if n > 0:
            flags[-1] = True
        return flags

    def build_action_valid(self) -> np.ndarray:
        """Build action_valid: False for frames with no action and for the last frame."""
        n = len(self.frames)
        flags = np.array(
            [f.action_commanded is not None for f in self.frames],
            dtype=np.bool_,
        )
        if n > 0:
            flags[-1] = False  # last frame: no subsequent state to observe
        return flags

