"""Isaac Sim rendering launch helpers shared by dex2bench entrypoints."""

from __future__ import annotations

import argparse
import os


HEADLESS_CAMERA_PARITY_EXPERIENCE = "isaaclab.python.rendering.kit"
DISABLE_HEADLESS_ACTIVE_VIEWPORT_ENV = "DEX2BENCH_HEADLESS_DISABLE_ACTIVE_VIEWPORT"

_TRUE_VALUES = {"1", "true", "yes", "on"}


def is_env_flag_enabled(name: str) -> bool:
    """Return whether an environment flag is truthy."""
    return os.environ.get(name, "").strip().lower() in _TRUE_VALUES


def configure_headless_camera_parity_experience(
    args: argparse.Namespace,
    *,
    enable_cameras: bool | None = None,
    log_prefix: str,
) -> bool:
    """Make headless camera runs use the same rendering experience as GUI runs.

    IsaacLab normally chooses ``isaaclab.python.headless.rendering.kit`` for
    ``--headless --enable_cameras``.  For dex2bench camera capture, the GUI
    rendering experience is the better default because it matches fisheye camera
    projection and RTX settings more closely to the data collected with UI.

    The run still remains headless from SimulationApp's perspective; this helper
    only changes the Kit experience and, by default, keeps the active viewport
    render path enabled for camera parity.
    """
    if getattr(args, "experience", ""):
        return False

    headless = bool(getattr(args, "headless", False) or is_env_flag_enabled("HEADLESS"))
    cameras = bool(
        enable_cameras
        if enable_cameras is not None
        else getattr(args, "enable_cameras", False) or is_env_flag_enabled("ENABLE_CAMERAS")
    )
    xr = bool(getattr(args, "xr", False) or is_env_flag_enabled("XR"))

    if not (headless and cameras and not xr):
        return False

    prefix = f"[{log_prefix}] " if log_prefix else ""
    if not is_env_flag_enabled(DISABLE_HEADLESS_ACTIVE_VIEWPORT_ENV):
        setattr(args, "video", True)
        print(
            f"{prefix}Headless camera parity: keeping the active viewport enabled "
            f"(set {DISABLE_HEADLESS_ACTIVE_VIEWPORT_ENV}=1 to use pure offscreen rendering).",
            flush=True,
        )
    args.experience = HEADLESS_CAMERA_PARITY_EXPERIENCE
    print(
        f"{prefix}Headless camera parity: using {HEADLESS_CAMERA_PARITY_EXPERIENCE} "
        "instead of IsaacLab's default headless rendering kit.",
        flush=True,
    )
    return True
