"""Shared command-line logging and profiling controls."""

from __future__ import annotations

import argparse
import logging


LOG_LEVELS = ("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG")


def add_logging_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the release-wide logging and opt-in profiling flags."""

    parser.add_argument(
        "--log-level",
        type=str.upper,
        choices=LOG_LEVELS,
        default="INFO",
        help="Python log verbosity (default: INFO).",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable diagnostic dumps and periodic performance reports.",
    )


def configure_logging(level: str = "INFO", *, force: bool = False) -> None:
    """Configure a compact process-wide logging format."""

    normalized = str(level).upper()
    if normalized not in LOG_LEVELS:
        raise ValueError(f"Unsupported log level: {level!r}")
    logging.basicConfig(
        level=getattr(logging, normalized),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        force=force,
    )
