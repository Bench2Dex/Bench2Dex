"""Batch screenshot from a custom 6D camera pose for selected HDF5 episodes.

Launches ``screenshot_6d.py`` once per episode to capture a single PNG
from a user-specified 6D camera pose at frame 0.

Generalization is RESTORED from each HDF5 — original HDF5 files are
NEVER modified.  Only PNG screenshots are written to the output directory.

Usage:
    python tools/replay/batch_screenshot_6d.py \\
        --origin-dir /path/to/replay-generalization \\
        --output-dir  /path/to/screenshots \\
        --episode-select standard_50

Episode selection (--episode-select):
    standard_50  – episodes 0-24 without _1 suffix + 25_1 to 49_1 with _1 suffix

Environment variables:
    OVERWRITE=1   Overwrite existing PNG files (default 1)
    OVERWRITE=0   Skip episodes that already have a PNG matching the episode name

Display:
    Default is headless.
    --gui   Run with a real UI on the current DISPLAY.
    --xvfb  Run on a managed Xvfb virtual display.
"""

import argparse
import os
import re
import shlex
import signal
import subprocess
import sys
from pathlib import Path


# ── Parse arguments ────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description="Batch screenshot selected HDF5 episodes from a custom 6D camera pose."
)
parser.add_argument(
    "--origin-dir",
    type=str,
    required=True,
    help="Directory containing episode_*.hdf5 files (e.g. replay-generalization/).",
)
parser.add_argument(
    "--output-dir",
    type=str,
    required=True,
    help="Output directory for episode_*.png screenshot files.",
)
parser.add_argument(
    "--episode-select",
    type=str,
    default=None,
    choices=["standard_50"],
    help=(
        "Episode selection preset. "
        "standard_50: episodes 0-24 (no _1 suffix) + 25_1 to 49_1 (with _1 suffix)."
    ),
)
parser.add_argument(
    "--camera-pos",
    type=str,
    default="0.0,-1.36,2.48,50.0,0.0,0.0",
    help="6D camera pose: Tx,Ty,Tz,Rx,Ry,Rz (m, deg).",
)
parser.add_argument("--width", type=int, default=1920, help="Screenshot width (default: 1920).")
parser.add_argument("--height", type=int, default=1080, help="Screenshot height (default: 1080).")
parser.add_argument(
    "--focal-length", type=float, default=10.5,
    help="Camera focal length in mm.",
)
parser.add_argument(
    "--scene", type=str, default=None, help="Override scene YAML path."
)
parser.add_argument(
    "--clean-background", action="store_true",
    help="Forward to screenshot_6d.py: use Isaac's solid-color dome background instead of the recorded USD room / env texture.",
)
parser.add_argument(
    "--zero-wrist", action="store_true",
    help="Forward to screenshot_6d.py: zero joint5/l_joint5 after setting qpos "
         "(fixes hands pointing up, e.g. multi_rm_65_with_revo2).",
)
display_group = parser.add_mutually_exclusive_group()
display_group.add_argument(
    "--gui", action="store_true",
    help="Run with a real GUI on the current DISPLAY.",
)
display_group.add_argument(
    "--xvfb", action="store_true",
    help="Run on a managed Xvfb virtual display.",
)
display_group.add_argument(
    "--headless", action="store_true",
    help="Run headless (the default). Accepted for clarity.",
)
parser.add_argument(
    "--no-validate", action="store_true",
    help="Skip post-capture validation of output PNGs.",
)
args = parser.parse_args()

REPO_DIR = Path(__file__).resolve().parents[2]  # dex2bench repo root
ORIGIN_DIR = Path(args.origin_dir)
OUTPUT_DIR = Path(args.output_dir)

# ── Helpers ────────────────────────────────────────────────────────────

def _build_episode_list(origin_dir: Path, select_mode: str | None) -> list[str]:
    """Return the sorted list of HDF5 filenames to process.

    When *select_mode* is None, all episode_*.hdf5 files are returned.
    """
    if select_mode is None:
        all_files = sorted(origin_dir.glob("episode_*.hdf5"))
        return [p.name for p in all_files]

    if select_mode == "standard_50":
        selected: list[str] = []
        # Episodes 0–24: no _1 suffix
        for i in range(0, 25):
            fname = f"episode_{i:06d}.hdf5"
            if (origin_dir / fname).exists():
                selected.append(fname)
            else:
                print(f"[WARN] Expected episode not found: {fname}", flush=True)
        # Episodes 25–49: with _1 suffix
        for i in range(25, 50):
            fname = f"episode_{i:06d}_1.hdf5"
            if (origin_dir / fname).exists():
                selected.append(fname)
            else:
                print(f"[WARN] Expected episode not found: {fname}", flush=True)
        return selected

    return []


def _hdf5_name_to_png_name(hdf5_name: str) -> str:
    """Convert episode_XXXXXX[_X].hdf5 → episode_XXXXXX[_X].png."""
    return Path(hdf5_name).stem + ".png"


def _resolve_display_mode(namespace: argparse.Namespace) -> str:
    if namespace.xvfb:
        return "xvfb"
    if namespace.gui:
        return "gui"
    return "headless"


def _display_mode_label(display_mode: str) -> str:
    if display_mode == "xvfb":
        return "headed+xvfb"
    if display_mode == "gui":
        return "headed real UI"
    return "headless"


def _build_screenshot_command(
    hdf5_path: Path,
    output_png: Path,
    *,
    camera_pos: str,
    width: int,
    height: int,
    focal_length: float,
    display_mode: str,
    scene: str | None,
    clean_background: bool = False,
    zero_wrist: bool = False,
) -> list[str]:
    script_path = REPO_DIR / "tools" / "replay" / "screenshot_6d.py"
    parts = [
        sys.executable,
        str(script_path),
        "--hdf5", str(hdf5_path),
        "--output-png", str(output_png),
        "--camera-pos", camera_pos,
        "--width", str(width),
        "--height", str(height),
        "--focal-length", str(focal_length),
    ]
    if display_mode == "headless":
        parts.append("--headless")
    if scene:
        parts.extend(["--scene", scene])
    if clean_background:
        parts.append("--clean-background")
    if zero_wrist:
        parts.append("--zero-wrist")
    return parts


def _kill_process_group(proc: subprocess.Popen, grace: float = 3.0) -> None:
    """Send SIGTERM then SIGKILL to the entire process group of *proc*."""
    if proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except OSError:
        pass
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        print("[batch] Graceful shutdown timed out — sending SIGKILL", flush=True)
        try:
            os.killpg(pgid, signal.SIGKILL)
        except OSError:
            pass
        proc.wait()


def _validate_png(png_path: Path) -> tuple[bool, str]:
    """Return (is_valid, reason). Checks that the PNG exists and is valid."""
    if not png_path.exists():
        return False, "PNG file missing"
    if png_path.stat().st_size < 100:
        return False, f"PNG too small ({png_path.stat().st_size} bytes)"
    try:
        from PIL import Image
        with Image.open(png_path) as img:
            img.verify()
        return True, "ok"
    except Exception as exc:
        return False, f"corrupt: {exc}"


# ── Main ───────────────────────────────────────────────────────────────

def main() -> int:
    if not ORIGIN_DIR.is_dir():
        print(f"[ERROR] Origin directory not found: {ORIGIN_DIR}")
        return 1

    # Build episode list
    episode_names = _build_episode_list(ORIGIN_DIR, args.episode_select)
    if not episode_names:
        print(f"[ERROR] No matching episodes found in {ORIGIN_DIR}")
        if args.episode_select:
            print(f"  Selection mode: {args.episode_select}")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    os.chdir(REPO_DIR)

    overwrite = os.environ.get("OVERWRITE", "1") == "1"
    display_mode = _resolve_display_mode(args)

    # ── Xvfb virtual display ───────────────────────────────────────────
    _xvfb_proc = None
    _xvfb_pgid = None
    _xvfb_display = None
    _cleanup_done = False

    def _stop_xvfb() -> None:
        nonlocal _xvfb_proc, _xvfb_pgid, _cleanup_done
        if _cleanup_done or _xvfb_proc is None:
            return
        _cleanup_done = True
        if _xvfb_proc.poll() is not None:
            _xvfb_proc = None
            return
        for _sig in (signal.SIGTERM, signal.SIGKILL):
            if _xvfb_proc.poll() is not None:
                break
            if _xvfb_pgid is not None:
                try:
                    os.killpg(_xvfb_pgid, _sig)
                except OSError:
                    pass
            try:
                _xvfb_proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
        _xvfb_proc = None
        print("\n[batch] Xvfb stopped", flush=True)

    import atexit as _atexit
    _atexit.register(_stop_xvfb)

    _active_proc = None

    def _kill_active() -> None:
        nonlocal _active_proc
        if _active_proc is not None:
            _kill_process_group(_active_proc)
            _active_proc = None

    if display_mode == "xvfb":
        import shutil as _shutil
        _xvfb_bin = _shutil.which("Xvfb")
        if _xvfb_bin is None:
            print("[batch][ERROR] --xvfb requested but Xvfb was not found.", flush=True)
            return 1
        _xvfb_display = os.environ.get("DEX2BENCH_XVFB_DISPLAY", ":99")
        _xvfb_proc = subprocess.Popen(
            [_xvfb_bin, _xvfb_display, "-screen", "0", "1920x1080x24", "-ac"],
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        _xvfb_pgid = os.getpgid(_xvfb_proc.pid)
        import time as _time
        _socket_path = f"/tmp/.X11-unix/X{_xvfb_display.lstrip(':')}"
        for _ in range(50):
            if os.path.exists(_socket_path):
                break
            _time.sleep(0.1)
        else:
            print("[batch][ERROR] Xvfb did not start.", flush=True)
            _stop_xvfb()
            return 1

    _subprocess_env = os.environ.copy()
    if _xvfb_display is not None:
        _subprocess_env["DISPLAY"] = _xvfb_display
        print(f"[batch] Xvfb virtual display: {_xvfb_display}", flush=True)

    # ── Batch loop ─────────────────────────────────────────────────────
    ok_count = 0
    fail_count = 0
    skip_count = 0
    failed: list[str] = []
    total = len(episode_names)

    print(f"[batch] Origin  : {ORIGIN_DIR}")
    print(f"[batch] Output  : {OUTPUT_DIR}")
    print(f"[batch] Episodes: {total}")
    print(f"[batch] Select  : {args.episode_select or 'all'}")
    print(f"[batch] OVERWRITE={1 if overwrite else 0}")
    print(f"[batch] Camera  : {args.camera_pos}")
    print(f"[batch] Resolution: {args.width}x{args.height}")
    print(f"[batch] Display : {_display_mode_label(display_mode)}")
    print(f"[batch] Generalization: RESTORE from HDF5 (no resampling)")
    print(f"[batch] Original HDF5 files are NEVER modified")

    interrupted = False
    try:
        for idx, hdf5_name in enumerate(episode_names, start=1):
            hdf5_path = ORIGIN_DIR / hdf5_name
            png_name = _hdf5_name_to_png_name(hdf5_name)
            output_png = OUTPUT_DIR / png_name

            if output_png.exists() and not overwrite:
                print(f"[SKIP] {idx}/{total} {png_name} (already exists)")
                skip_count += 1
                continue

            cmd_list = _build_screenshot_command(
                hdf5_path,
                output_png,
                camera_pos=args.camera_pos,
                width=args.width,
                height=args.height,
                focal_length=args.focal_length,
                display_mode=display_mode,
                scene=args.scene,
                clean_background=args.clean_background,
                zero_wrist=args.zero_wrist,
            )
            cmd_str = " ".join(shlex.quote(p) for p in cmd_list)
            print(f"\n{'=' * 80}")
            print(f"[episode {idx}/{total}] {hdf5_name}")
            print(f"  → {png_name}")
            print(f"CMD: {cmd_str}")
            print(f"{'=' * 80}\n")

            _active_proc = subprocess.Popen(
                cmd_list, start_new_session=True, env=_subprocess_env,
            )
            try:
                ret = _active_proc.wait()
            except KeyboardInterrupt:
                print(f"\n[batch] Interrupted — stopping {hdf5_name} ...",
                      flush=True)
                signal.signal(signal.SIGINT, signal.SIG_IGN)
                _kill_active()
                _stop_xvfb()
                interrupted = True
                break
            finally:
                _active_proc = None

            if ret != 0:
                print(f"\n[ERROR] Screenshot failed: {hdf5_name}, exit={ret}")
                fail_count += 1
                failed.append(hdf5_name)
                # Remove partial output if any
                if output_png.exists():
                    output_png.unlink()
                continue

            if output_png.exists():
                size_kb = output_png.stat().st_size / 1024
                print(f"[OK] {hdf5_name} → {png_name} ({size_kb:.0f} KB)")
                ok_count += 1
            else:
                print(f"[ERROR] Output PNG missing: {output_png}")
                fail_count += 1
                failed.append(hdf5_name)
    except KeyboardInterrupt:
        interrupted = True
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        _kill_active()

    _stop_xvfb()

    # ── Post-capture validation ────────────────────────────────────────
    if not args.no_validate and not interrupted:
        print(f"\n{'=' * 80}")
        print("[validate] Checking output PNGs ...")
        bad_count = 0
        for png_path in sorted(OUTPUT_DIR.glob("*.png")):
            valid, reason = _validate_png(png_path)
            if valid:
                continue
            print(f"[validate] BAD: {png_path.name} — {reason}")
            bad_count += 1
            if png_path.name not in failed:
                failed.append(png_path.name)
        if bad_count:
            fail_count += bad_count
            print(f"[validate] {bad_count} PNG(s) are missing or corrupt")
        else:
            print("[validate] All PNGs look valid.")

    # ── Summary ────────────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    if interrupted:
        print("[batch] Interrupted by user")
    print("[batch] Done")
    print(f"  Output:   {OUTPUT_DIR}")
    print(f"  OK:       {ok_count}")
    print(f"  Skipped:  {skip_count}")
    print(f"  Failed:   {fail_count}")
    if failed:
        print("[batch] Failed episodes:")
        for name in failed:
            print(f"  {name}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
