#!/usr/bin/env python3
"""Compute robustness metrics from four-channel eval outputs.

Usage:
    python tools/compute_robustness.py /path/to/eval_root
    python tools/compute_robustness.py /path/to/eval_root --allow-partial --no-write

Expected layout:
    eval_root/
      none/per_episode.jsonl
      cov_only/per_episode.jsonl
      inv_only/per_episode.jsonl
      inv_cov/per_episode.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.statistics import bootstrap_ci  # noqa: E402


CHANNELS: dict[str, dict[str, str]] = {
    "none": {"category": "none", "label": "baseline"},
    "cov_only": {"category": "cov", "label": "covariate"},
    "inv_only": {"category": "inv", "label": "invariance"},
    "inv_cov": {"category": "inv+cov", "label": "combined"},
}

PERTURBED_CHANNELS = ("cov_only", "inv_only", "inv_cov")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected object, got {type(row).__name__}")
            rows.append(row)
    return rows


def _coerce_success(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n", ""}:
            return False
    return bool(value)


def _is_success(row: dict[str, Any]) -> bool:
    metrics = row.get("metrics") or {}
    if isinstance(metrics, dict) and "success" in metrics:
        return _coerce_success(metrics["success"])
    return _coerce_success(row.get("success"))


def _episode_index(row: dict[str, Any], fallback: int) -> int:
    value = row.get("episode_index")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return fallback


def _success_values(rows: list[dict[str, Any]]) -> list[float]:
    return [1.0 if _is_success(row) else 0.0 for row in rows]


def _rate(successes: int, total: int) -> float | None:
    if total <= 0:
        return None
    return successes / total


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def _ci(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    mean, low, high = bootstrap_ci(values, n_resamples=10000, seed=0)
    return {"mean": mean, "ci_low": low, "ci_high": high}


def _rows_by_episode(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Index rows by episode, retaining the first row for duplicate diagnostics."""
    indexed: dict[int, dict[str, Any]] = {}
    for fallback, row in enumerate(rows):
        indexed.setdefault(_episode_index(row, fallback), row)
    return indexed


def _generalization_sample(row: dict[str, Any]) -> dict[str, Any] | None:
    sample = row.get("scene_generalization_sample")
    return sample if isinstance(sample, dict) else None


def _covariate_signature(row: dict[str, Any]) -> dict[str, Any] | None:
    """Realized task-geometry factors that inv_only must preserve."""
    sample = _generalization_sample(row)
    if sample is None:
        return None
    spatial = sample.get("spatial") or {}
    placements = sample.get("resolved_object_placements")
    table_height = spatial.get("table_height") if isinstance(spatial, dict) else None
    if not isinstance(placements, dict) or not isinstance(table_height, dict):
        return None
    return {
        "resolved_object_placements": placements,
        "table_height_offset_m": table_height.get("height_offset_m"),
    }


def _invariant_signature(row: dict[str, Any]) -> dict[str, Any] | None:
    """Realized nuisance factors that cov_only must preserve."""
    sample = _generalization_sample(row)
    if sample is None:
        return None
    spatial = sample.get("spatial") or {}
    if not isinstance(spatial, dict):
        return None
    return {
        "appearance": sample.get("appearance"),
        "clutter": sample.get("clutter"),
        "camera": spatial.get("camera"),
    }


def _paired_signature_check(
    baseline_rows: list[dict[str, Any]],
    channel_rows: list[dict[str, Any]],
    *,
    signature_fn,
    name: str,
) -> dict[str, Any]:
    baseline_by_episode = _rows_by_episode(baseline_rows)
    channel_by_episode = _rows_by_episode(channel_rows)
    common = sorted(set(baseline_by_episode) & set(channel_by_episode))
    mismatches: list[int] = []
    missing: list[int] = []
    mismatch_field_counts: Counter[str] = Counter()
    for episode_index in common:
        baseline_signature = signature_fn(baseline_by_episode[episode_index])
        channel_signature = signature_fn(channel_by_episode[episode_index])
        if baseline_signature is None or channel_signature is None:
            missing.append(episode_index)
        elif baseline_signature != channel_signature:
            mismatches.append(episode_index)
            _count_difference_paths(
                baseline_signature,
                channel_signature,
                counts=mismatch_field_counts,
            )
    return {
        "name": name,
        "verifiable": bool(common) and not missing,
        "valid": bool(common) and not missing and not mismatches,
        "common_episode_count": len(common),
        "mismatch_count": len(mismatches),
        "mismatch_episode_indices_first20": mismatches[:20],
        "mismatch_field_counts": dict(sorted(mismatch_field_counts.items())),
        "missing_signature_count": len(missing),
        "missing_signature_episode_indices_first20": missing[:20],
    }


def _count_difference_paths(
    left: Any,
    right: Any,
    *,
    counts: Counter[str],
    prefix: str = "",
) -> None:
    """Attribute paired protocol mismatches to compact JSON field paths."""
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                counts[path] += 1
            else:
                _count_difference_paths(left[key], right[key], counts=counts, prefix=path)
        return
    if isinstance(left, list) and isinstance(right, list):
        for index in range(max(len(left), len(right))):
            path = f"{prefix}[{index}]"
            if index >= len(left) or index >= len(right):
                counts[path] += 1
            else:
                _count_difference_paths(left[index], right[index], counts=counts, prefix=path)
        return
    if left != right:
        counts[prefix or "<root>"] += 1


def _channel_summary(channel: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = _success_values(rows)
    successes = int(sum(values))
    episode_indices = [_episode_index(row, idx) for idx, row in enumerate(rows)]
    episode_counts = Counter(episode_indices)
    duplicate_episode_indices = sorted(idx for idx, count in episode_counts.items() if count > 1)
    return {
        "channel": channel,
        "category": CHANNELS[channel]["category"],
        "label": CHANNELS[channel]["label"],
        "n_episodes": len(rows),
        "successes": successes,
        "success_rate": _rate(successes, len(rows)),
        "success_rate_ci": _ci(values),
        "episode_index_min": min(episode_indices) if episode_indices else None,
        "episode_index_max": max(episode_indices) if episode_indices else None,
        "duplicate_episode_indices": duplicate_episode_indices,
    }


def _success_index_set(rows: list[dict[str, Any]]) -> set[int]:
    return {
        _episode_index(row, idx)
        for idx, row in enumerate(rows)
        if _is_success(row)
    }


def _episode_index_set(rows: list[dict[str, Any]]) -> set[int]:
    return {_episode_index(row, idx) for idx, row in enumerate(rows)}


def _first_items(values: set[int], limit: int = 20) -> list[int]:
    return sorted(values)[:limit]


def compute_robustness(
    root: Path,
    *,
    expected_n: int | None,
    allow_partial: bool,
) -> dict[str, Any]:
    root = root.resolve()
    issues: list[str] = []
    rows_by_channel: dict[str, list[dict[str, Any]]] = {}

    for channel in CHANNELS:
        path = root / channel / "per_episode.jsonl"
        if not path.exists():
            issues.append(f"missing {channel}/per_episode.jsonl")
            continue
        rows = _read_jsonl(path)
        rows_by_channel[channel] = rows
        if expected_n is not None and len(rows) != expected_n:
            issues.append(f"{channel}: expected {expected_n} episodes, got {len(rows)}")

    if not allow_partial:
        missing = sorted(set(CHANNELS) - set(rows_by_channel))
        if missing:
            issues.append(
                "incomplete robustness set: need all four channels "
                f"({', '.join(CHANNELS)}), missing {', '.join(missing)}"
            )

    channels = {
        channel: _channel_summary(channel, rows)
        for channel, rows in rows_by_channel.items()
    }

    for channel, summary in channels.items():
        duplicates = summary["duplicate_episode_indices"]
        if duplicates:
            issues.append(
                f"{channel}: duplicate episode indices {duplicates[:20]}"
            )

    baseline = channels.get("none")
    baseline_rate = baseline.get("success_rate") if baseline else None
    baseline_rows = rows_by_channel.get("none", [])
    baseline_success_indices = _success_index_set(baseline_rows)

    protocol_checks: dict[str, dict[str, Any]] = {}
    if "none" in rows_by_channel and "cov_only" in rows_by_channel:
        protocol_checks["cov_only_preserves_invariant_factors"] = _paired_signature_check(
            rows_by_channel["none"],
            rows_by_channel["cov_only"],
            signature_fn=_invariant_signature,
            name="cov_only preserves baseline appearance/clutter/camera",
        )
    if "none" in rows_by_channel and "inv_only" in rows_by_channel:
        protocol_checks["inv_only_preserves_covariant_factors"] = _paired_signature_check(
            rows_by_channel["none"],
            rows_by_channel["inv_only"],
            signature_fn=_covariate_signature,
            name="inv_only preserves baseline object placements/table height",
        )
    for check_name, check in protocol_checks.items():
        if not check["valid"]:
            issues.append(
                f"protocol check failed ({check_name}): "
                f"mismatches={check['mismatch_count']}, "
                f"missing_signatures={check['missing_signature_count']}"
            )

    episode_sets = {
        channel: _episode_index_set(rows)
        for channel, rows in rows_by_channel.items()
    }
    union_indices = set().union(*episode_sets.values()) if episode_sets else set()
    common_indices = set.intersection(*episode_sets.values()) if episode_sets else set()
    missing_by_channel = {
        channel: {
            "count": len(union_indices - indices),
            "episode_indices_first20": _first_items(union_indices - indices),
        }
        for channel, indices in episode_sets.items()
    }
    all_channels_present = set(rows_by_channel) == set(CHANNELS)
    episode_sets_aligned = (
        all_channels_present and len(common_indices) == len(union_indices)
    )
    if all_channels_present and not episode_sets_aligned:
        issues.append(
            "episode index sets are not aligned across all four channels: "
            f"common={len(common_indices)}, union={len(union_indices)}"
        )
    expected_counts_complete = expected_n is None or (
        all_channels_present
        and all(len(rows) == expected_n for rows in rows_by_channel.values())
    )
    duplicate_free = all(
        not summary["duplicate_episode_indices"] for summary in channels.values()
    )
    data_complete = (
        all_channels_present
        and expected_counts_complete
        and duplicate_free
        and episode_sets_aligned
    )

    per_category: dict[str, dict[str, Any]] = {}
    perturbed_values: list[float] = []
    perturbed_successes = 0
    perturbed_n = 0

    for channel in PERTURBED_CHANNELS:
        if channel not in rows_by_channel:
            continue
        channel_rows = rows_by_channel[channel]
        summary = channels[channel]
        category = CHANNELS[channel]["category"]
        values = _success_values(channel_rows)
        perturbed_values.extend(values)
        perturbed_successes += int(sum(values))
        perturbed_n += len(values)

        channel_success_indices = _success_index_set(channel_rows)
        channel_indices = _episode_index_set(channel_rows)
        paired_denominator = len(baseline_success_indices)
        kept_success_indices = baseline_success_indices & channel_success_indices
        missing_baseline_success_indices = baseline_success_indices - channel_indices
        # Category ratios are official only after the entire four-channel data
        # gate passes.  ``--allow-partial`` may expose observed values for
        # debugging, but must not turn them into publishable robustness.
        category_protocol_valid = data_complete
        if category == "cov":
            category_protocol_valid = category_protocol_valid and bool(
                protocol_checks.get("cov_only_preserves_invariant_factors", {}).get("valid", False)
            )
        elif category == "inv":
            category_protocol_valid = category_protocol_valid and bool(
                protocol_checks.get("inv_only_preserves_covariant_factors", {}).get("valid", False)
            )
        observed_ratio = _ratio(summary["success_rate"], baseline_rate)
        per_category[category] = {
            "channel": channel,
            "protocol_valid": category_protocol_valid,
            "n_episodes": summary["n_episodes"],
            "successes": summary["successes"],
            "success_rate": summary["success_rate"],
            "success_rate_ci": summary["success_rate_ci"],
            "robust_ratio_vs_baseline": observed_ratio if category_protocol_valid else None,
            "observed_ratio_unvalidated": observed_ratio if not category_protocol_valid else None,
            "paired_retention": {
                "kept_successes": len(kept_success_indices),
                "baseline_successes": paired_denominator,
                "retention_rate": _rate(len(kept_success_indices), paired_denominator),
                "overlap_episode_count": len(_episode_index_set(baseline_rows) & channel_indices),
                "missing_baseline_success_episode_count": len(missing_baseline_success_indices),
                "missing_baseline_success_episode_indices_first20": _first_items(
                    missing_baseline_success_indices
                ),
            },
        }

    perturbed_rate = _rate(perturbed_successes, perturbed_n)

    all_values: list[float] = []
    for rows in rows_by_channel.values():
        all_values.extend(_success_values(rows))
    mixed_successes = int(sum(all_values))
    mixed_n = len(all_values)

    protocol_valid = data_complete and not issues and all(
        check.get("valid", False) for check in protocol_checks.values()
    )
    observed_overall_ratio = _ratio(perturbed_rate, baseline_rate)
    return {
        "root": str(root),
        "complete": protocol_valid,
        "data_complete": data_complete,
        "protocol_valid": protocol_valid,
        "allow_partial": allow_partial,
        "expected_n": expected_n,
        "issues": issues,
        "channels": channels,
        "protocol_audit": {
            "valid": protocol_valid,
            "checks": protocol_checks,
        },
        "robustness": {
            "definition": (
                "Robustness is computed after all four channels finish: perturbation "
                "success rate divided by baseline none success rate. The preferred "
                "overall value uses only cov_only, inv_only, and inv_cov in the numerator."
            ),
            "baseline_channel": "none",
            "baseline_n": baseline["n_episodes"] if baseline else 0,
            "baseline_successes": baseline["successes"] if baseline else 0,
            "baseline_success_rate": baseline_rate,
            "baseline_success_rate_ci": baseline.get("success_rate_ci") if baseline else None,
            "per_category": per_category,
            "overall_perturbed_successes": perturbed_successes,
            "overall_perturbed_n": perturbed_n,
            "overall_perturbed_success_rate": perturbed_rate,
            "overall_perturbed_success_rate_ci": _ci(perturbed_values),
            "overall_robust_ratio_vs_baseline": (
                observed_overall_ratio if protocol_valid else None
            ),
            "observed_overall_ratio_unvalidated": (
                observed_overall_ratio if not protocol_valid else None
            ),
        },
        "diagnostics": {
            "mixed_four_channel_successes": mixed_successes,
            "mixed_four_channel_n": mixed_n,
            "mixed_four_channel_success_rate": _rate(mixed_successes, mixed_n),
            "mixed_four_channel_note": (
                "This mixes baseline ability with perturbation performance and should "
                "not be reported as robustness."
            ),
            "episode_alignment": {
                "union_episode_count": len(union_indices),
                "common_episode_count": len(common_indices),
                "missing_by_channel": missing_by_channel,
            },
        },
    }


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.2f}%"


def _fmt_ratio(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def _fmt_ci(ci: dict[str, float] | None) -> str:
    if not ci:
        return "[n/a]"
    return f"[{_fmt_pct(ci['ci_low'])}, {_fmt_pct(ci['ci_high'])}]"


def print_report(summary: dict[str, Any]) -> None:
    print(f"Robustness summary: {summary['root']}")
    if not summary["data_complete"]:
        status = "INCOMPLETE"
    elif summary["protocol_valid"]:
        status = "VALID"
    else:
        status = "INVALID"
    print(f"Status: {status}")
    if summary["issues"]:
        print("Issues:")
        for issue in summary["issues"]:
            print(f"  - {issue}")

    print("\nChannel success rates:")
    print("channel   category  successes/n  success_rate  95% CI              robust_ratio")
    baseline_rate = summary["robustness"]["baseline_success_rate"]
    for channel in CHANNELS:
        data = summary["channels"].get(channel)
        if not data:
            print(f"{channel:<9} {CHANNELS[channel]['category']:<8} missing")
            continue
        category = CHANNELS[channel]["category"]
        category_data = summary["robustness"]["per_category"].get(category, {})
        ratio = category_data.get("robust_ratio_vs_baseline") if channel != "none" else None
        print(
            f"{channel:<9} {data['category']:<8} "
            f"{data['successes']:>3}/{data['n_episodes']:<3}      "
            f"{_fmt_pct(data['success_rate']):<12} "
            f"{_fmt_ci(data['success_rate_ci']):<19} "
            f"{_fmt_ratio(ratio)}"
        )

    robustness = summary["robustness"]
    print("\nOverall perturbation robustness:")
    print(
        "perturbed "
        f"{robustness['overall_perturbed_successes']}/"
        f"{robustness['overall_perturbed_n']} = "
        f"{_fmt_pct(robustness['overall_perturbed_success_rate'])}, "
        f"RR={_fmt_ratio(robustness['overall_robust_ratio_vs_baseline'])}"
    )
    if robustness.get("observed_overall_ratio_unvalidated") is not None:
        print(
            "observed ratio (UNVALIDATED; do not report) = "
            f"{_fmt_ratio(robustness['observed_overall_ratio_unvalidated'])}"
        )

    if summary["protocol_audit"]["checks"]:
        print("\nProtocol purity checks:")
        for check in summary["protocol_audit"]["checks"].values():
            print(
                f"  {'PASS' if check['valid'] else 'FAIL'}: {check['name']} "
                f"(mismatches={check['mismatch_count']}, "
                f"missing={check['missing_signature_count']})"
            )
            if not check["valid"] and check.get("mismatch_field_counts"):
                top_fields = sorted(
                    check["mismatch_field_counts"].items(),
                    key=lambda item: (-item[1], item[0]),
                )[:8]
                print(
                    "    differing fields: "
                    + ", ".join(f"{field} ({count})" for field, count in top_fields)
                )

    if robustness["per_category"]:
        print("\nPaired retention among baseline successes:")
        print("channel    kept/baseline_successes  retention  missing_baseline_success")
        category_to_channel = {
            CHANNELS[channel]["category"]: channel for channel in PERTURBED_CHANNELS
        }
        for category in ("cov", "inv", "inv+cov"):
            data = robustness["per_category"].get(category)
            if not data:
                continue
            paired = data["paired_retention"]
            print(
                f"{category_to_channel[category]:<10} "
                f"{paired['kept_successes']:>3}/{paired['baseline_successes']:<3}                 "
                f"{_fmt_pct(paired['retention_rate']):<10} "
                f"{paired['missing_baseline_success_episode_count']}"
            )

    diagnostics = summary["diagnostics"]
    print("\nDiagnostic only:")
    print(
        "mixed four-channel success rate = "
        f"{diagnostics['mixed_four_channel_successes']}/"
        f"{diagnostics['mixed_four_channel_n']} = "
        f"{_fmt_pct(diagnostics['mixed_four_channel_success_rate'])}"
    )
    print("Do not use the mixed four-channel rate as robustness.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Eval root containing four channel directories")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON output path. Defaults to <root>/robustness_summary.json",
    )
    parser.add_argument("--no-write", action="store_true", help="Only print report")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Compute from available channels instead of requiring all four",
    )
    parser.add_argument(
        "--expected-n",
        type=int,
        default=None,
        help="Expected episode count per channel; mismatches are reported as issues",
    )
    parser.add_argument("--indent", type=int, default=2, help="JSON indentation")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = compute_robustness(
        args.root,
        expected_n=args.expected_n,
        allow_partial=args.allow_partial,
    )
    print_report(summary)

    if not args.no_write:
        output = args.output or (args.root / "robustness_summary.json")
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=args.indent, ensure_ascii=False)
            f.write("\n")
        print(f"\nWrote {output}")

    if summary["complete"] or args.allow_partial:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
