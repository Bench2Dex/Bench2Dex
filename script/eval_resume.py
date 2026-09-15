"""Resume planning helpers for split policy evaluation launchers."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


class ResumePlanError(ValueError):
    """Raised when an existing eval output cannot be resumed safely."""


@dataclass(frozen=True)
class ChannelResumeState:
    channel: str
    status: str
    start_episode: int
    num_episodes: int
    append: bool
    completed_in_target: int
    channel_dir: Path


def load_episode_indices(path: Path | str, *, strict: bool = False) -> list[int]:
    """Load completed episode_index values from a per_episode.jsonl file.

    Args:
        path: Path to per_episode.jsonl.
        strict: If True, raise on duplicates or gaps (legacy behavior).
                If False (default), warn and deduplicate/gap-fill gracefully,
                so that a previously-corrupted resume can still be recovered.
    """
    jsonl_path = Path(path)
    if not jsonl_path.exists():
        return []

    raw_indices: list[int] = []
    seen: set[int] = set()
    dup_count = 0
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ResumePlanError(f"{jsonl_path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(payload, dict):
                raise ResumePlanError(f"{jsonl_path}:{line_no}: expected JSON object")
            if "episode_index" not in payload:
                raise ResumePlanError(f"{jsonl_path}:{line_no}: missing episode_index")
            try:
                episode_index = int(payload["episode_index"])
            except (TypeError, ValueError) as exc:
                raise ResumePlanError(
                    f"{jsonl_path}:{line_no}: invalid episode_index={payload['episode_index']!r}"
                ) from exc
            if episode_index < 1:
                raise ResumePlanError(f"{jsonl_path}:{line_no}: episode_index must be >= 1")
            if episode_index in seen:
                dup_count += 1
                if strict:
                    raise ResumePlanError(f"{jsonl_path}:{line_no}: duplicate episode_index={episode_index}")
                print(
                    f"[resume] WARNING: {jsonl_path}:{line_no}: duplicate episode_index={episode_index}, "
                    f"keeping first occurrence",
                    file=sys.stderr,
                )
                continue
            seen.add(episode_index)
            raw_indices.append(episode_index)

    if dup_count > 0:
        print(
            f"[resume] WARNING: {jsonl_path}: {dup_count} duplicate line(s) ignored; "
            f"re-run tools/fix_resume_duplicates.py to clean up permanently",
            file=sys.stderr,
        )

    raw_indices.sort()
    # Check for gaps
    gaps: list[tuple[int, int]] = []
    for previous, current in zip(raw_indices, raw_indices[1:]):
        if current != previous + 1:
            gaps.append((previous, current))

    if gaps:
        gap_str = ", ".join(f"{p}→{c}" for p, c in gaps)
        if strict:
            raise ResumePlanError(
                f"{jsonl_path}: non-contiguous episode_index values: {gap_str}"
            )
        print(
            f"[resume] WARNING: {jsonl_path}: non-contiguous episode_index values: {gap_str}; "
            f"resume will only count the longest contiguous prefix",
            file=sys.stderr,
        )

    # Return the longest contiguous prefix starting from the first index.
    # This is the safe set of episodes that are guaranteed complete.
    if not raw_indices:
        return []
    contiguous: list[int] = [raw_indices[0]]
    for idx in raw_indices[1:]:
        if idx == contiguous[-1] + 1:
            contiguous.append(idx)
        else:
            break
    return contiguous


def find_latest_resume_root(
    eval_root_parent: Path | str,
    *,
    name_prefix: str,
    robot_suffix: str,
) -> Path:
    """Return the newest eval root matching the launcher naming pattern."""
    parent = Path(eval_root_parent)
    if not parent.is_dir():
        raise ResumePlanError(f"Eval root parent does not exist: {parent}")

    suffix = f"_{robot_suffix}" if robot_suffix else ""
    candidates = [
        child
        for child in parent.iterdir()
        if child.is_dir()
        and child.name.startswith(name_prefix)
        and (not suffix or child.name.endswith(suffix))
    ]
    if not candidates:
        raise ResumePlanError(
            f"No resume output found under {parent} matching prefix={name_prefix!r} "
            f"suffix={suffix!r}"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _channel_dir(root: Path, channel: str, *, multi_channel: bool) -> Path:
    return root / channel if multi_channel else root


def inspect_channel(
    channel: str,
    channel_dir: Path | str,
    *,
    target_start: int,
    target_count: int,
) -> ChannelResumeState:
    if target_start < 1:
        raise ResumePlanError("target_start must be >= 1")
    if target_count < 1:
        raise ResumePlanError("target_count must be >= 1")

    out_dir = Path(channel_dir)
    target_end = target_start + target_count - 1
    indices = load_episode_indices(out_dir / "per_episode.jsonl")
    if not indices:
        return ChannelResumeState(
            channel=channel,
            status="missing",
            start_episode=target_start,
            num_episodes=target_count,
            append=False,
            completed_in_target=0,
            channel_dir=out_dir,
        )

    first_index = indices[0]
    last_index = indices[-1]
    if first_index > target_start:
        raise ResumePlanError(
            f"{out_dir / 'per_episode.jsonl'} starts at episode {first_index}, "
            f"but resume target starts at {target_start}"
        )

    completed_in_target = max(0, min(last_index, target_end) - target_start + 1)
    if last_index >= target_end:
        return ChannelResumeState(
            channel=channel,
            status="complete",
            start_episode=target_end + 1,
            num_episodes=0,
            append=True,
            completed_in_target=target_count,
            channel_dir=out_dir,
        )

    next_start = max(target_start, last_index + 1)
    return ChannelResumeState(
        channel=channel,
        status="partial",
        start_episode=next_start,
        num_episodes=target_end - next_start + 1,
        append=True,
        completed_in_target=completed_in_target,
        channel_dir=out_dir,
    )


def build_resume_plan(
    root: Path | str,
    channels: Iterable[str],
    *,
    target_start: int,
    target_count: int,
    multi_channel: bool,
) -> list[ChannelResumeState]:
    eval_root = Path(root)
    return [
        inspect_channel(
            channel,
            _channel_dir(eval_root, channel, multi_channel=multi_channel),
            target_start=target_start,
            target_count=target_count,
        )
        for channel in channels
    ]


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="Print a TSV resume plan for shell launchers.")
    plan.add_argument("--eval-root-parent", required=True)
    plan.add_argument("--name-prefix", required=True)
    plan.add_argument("--robot-suffix", required=True)
    plan.add_argument("--resume-dir", default=None)
    plan.add_argument("--target-start", type=int, required=True)
    plan.add_argument("--target-count", type=int, required=True)
    plan.add_argument("--multi-channel", action="store_true")
    plan.add_argument("--channels", nargs="+", required=True)
    return parser.parse_args(argv)


def _cmd_plan(args: argparse.Namespace) -> int:
    if args.resume_dir:
        root = Path(args.resume_dir).expanduser().resolve()
        if not root.is_dir():
            raise ResumePlanError(f"Resume dir does not exist: {root}")
    else:
        root = find_latest_resume_root(
            args.eval_root_parent,
            name_prefix=args.name_prefix,
            robot_suffix=args.robot_suffix,
        ).resolve()

    print(f"ROOT\t{root}")
    for state in build_resume_plan(
        root,
        args.channels,
        target_start=args.target_start,
        target_count=args.target_count,
        multi_channel=bool(args.multi_channel),
    ):
        print(
            "\t".join(
                [
                    "CHANNEL",
                    state.channel,
                    state.status,
                    str(state.start_episode),
                    str(state.num_episodes),
                    "true" if state.append else "false",
                    str(state.completed_in_target),
                    str(state.channel_dir),
                ]
            )
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    try:
        if args.command == "plan":
            return _cmd_plan(args)
    except ResumePlanError as exc:
        print(f"[resume] {exc}", file=sys.stderr)
        return 2
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
