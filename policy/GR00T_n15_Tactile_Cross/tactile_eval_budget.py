"""Pure episode-budget resolution for tactile policy evaluation."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Optional, Union


@dataclass(frozen=True)
class EpisodeBudget:
    episode_steps: int
    max_physics_steps: int
    source: str


_TIMING_KEY_PATTERN = re.compile(
    r"^\s*(expert_time_step|expert_time_s)\b"
)
_TIMING_LINE_PATTERN = re.compile(
    r"^\s*(expert_time_step|expert_time_s)\s*:(.*)$"
)
_NUMBER_PATTERN = re.compile(
    r"[+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|\.?(?:inf(?:inity)?|nan))",
    re.IGNORECASE,
)


def _validate_positive_integer(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be positive and an integer")


def _validate_finite_positive_number(value: object, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a finite positive number")


def _parse_scene_timing(line: str) -> tuple[str, float] | None:
    key_match = _TIMING_KEY_PATTERN.match(line)
    if key_match is None:
        return None

    key = key_match.group(1)
    timing_match = _TIMING_LINE_PATTERN.fullmatch(line.rstrip("\r\n"))
    if timing_match is None:
        raise ValueError(f"{key} must be a finite positive number")

    raw_value = timing_match.group(2)
    comment_marker = re.search(r"(?<=[ \t])#", raw_value)
    if comment_marker is not None:
        raw_value = raw_value[:comment_marker.start()]
    raw_value = raw_value.strip()
    if not raw_value:
        return None
    if _NUMBER_PATTERN.fullmatch(raw_value) is None:
        raise ValueError(f"{key} must be a finite positive number")

    value = float(raw_value)
    _validate_finite_positive_number(value, key)
    return key, value


def resolve_episode_budget(
    scene_path: Union[str, Path],
    *,
    explicit_episode_steps: Optional[int],
    policy_stride: int = 3,
    default_episode_steps: int = 800,
    scale: float = 1.5,
    physics_hz: float = 60.0,
) -> EpisodeBudget:
    """Resolve policy and physics step limits from CLI or scene metadata."""

    _validate_positive_integer(policy_stride, "policy_stride")
    _validate_positive_integer(default_episode_steps, "default_episode_steps")
    _validate_finite_positive_number(scale, "scale")
    _validate_finite_positive_number(physics_hz, "physics_hz")

    if explicit_episode_steps is not None:
        _validate_positive_integer(
            explicit_episode_steps,
            "explicit_episode_steps",
        )
        return EpisodeBudget(
            episode_steps=explicit_episode_steps,
            max_physics_steps=explicit_episode_steps * policy_stride,
            source="explicit",
        )

    expert_time_step = None
    expert_time_s = None
    path = Path(scene_path)
    if path.is_file():
        with path.open("r", encoding="utf-8") as scene_file:
            for line in scene_file:
                timing = _parse_scene_timing(line)
                if timing is None:
                    continue
                key, value = timing
                if key == "expert_time_step" and expert_time_step is None:
                    expert_time_step = value
                elif key == "expert_time_s" and expert_time_s is None:
                    expert_time_s = value
                if expert_time_step is not None and expert_time_s is not None:
                    break

    if expert_time_step is not None:
        target_physics_steps = math.ceil(expert_time_step * scale)
        episode_steps = math.ceil(target_physics_steps / policy_stride)
        source = "expert_time_step"
    elif expert_time_s is not None:
        target_physics_steps = math.ceil(expert_time_s * physics_hz * scale)
        episode_steps = math.ceil(target_physics_steps / policy_stride)
        source = "expert_time_s"
    else:
        episode_steps = default_episode_steps
        source = "default"

    return EpisodeBudget(
        episode_steps=episode_steps,
        max_physics_steps=episode_steps * policy_stride,
        source=source,
    )
