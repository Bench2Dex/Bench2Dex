"""Batch replay HDF5 episodes with selectable replay capture modalities.

Usage (via shell wrapper):
    bash tools/replay/batch_replay.sh
    bash tools/replay/batch_replay.sh --enable-rgb --enable-depth
    bash tools/replay/batch_replay.sh --enable-rgb --enable-depth --enable-tactile

Or directly:
    python tools/replay/batch_replay.py \
        --origin-dir outputs/ur5_rh56dfx/scenes/06_fruit_bowl_loading/origin \
        --replay-dir  outputs/ur5_rh56dfx/scenes/06_fruit_bowl_loading/replay
    python tools/replay/batch_replay.py \
        --origin-dir outputs/.../origin \
        --replay-dir  outputs/.../replay \
        --enable-depth
    python tools/replay/batch_replay.py \
        --origin-dir outputs/.../origin \
        --replay-dir  outputs/.../replay \
        --enable-rgb --enable-depth
    python tools/replay/batch_replay.py \
        --origin-dir outputs/.../origin \
        --replay-dir  outputs/.../replay \
        --enable-tactile
    python tools/replay/batch_replay.py \
        --origin-dir outputs/.../origin \
        --replay-dir  outputs/.../replay \
        --enable-rgb --enable-depth --enable-tactile
    python tools/replay/batch_replay.py \
        --origin-dir outputs/.../origin \
        --replay-dir  outputs/.../replay \
        --tactile-only
    python tools/replay/batch_replay.py \
        --origin-dir outputs/.../origin-generalization \
        --replay-dir  outputs/.../replay-generalization-seen \
        --enable-rgb \
        --resample-groups background,table_surface,light \
        --generalization-split seen

Environment variables:
    OVERWRITE=1   Overwrite existing replay files (default 1)
    OVERWRITE=0   Skip episodes that already have a replay file

Capture options:
    No capture flags defaults to --enable-rgb for backward compatibility.
    --enable-rgb      Write replayed RGB camera data.
    --enable-depth    Write replayed depth camera data.
    --enable-tactile  Write replayed TacMap tactile data.
    --tactile-only    Capture tactile only; disables RGB/depth and implies tactile.
    Flags can be combined, for example --enable-rgb --enable-depth --enable-tactile.

Generalization options:
    Default behavior restores the exact generalization stored in each HDF5.
    --resample-groups switches replay to current config selective resampling.
    Safe visual groups are: background, table_surface, light, camera.

Display:
    Default is headless: passes --headless to replay.py, which then selects
    isaaclab.python.rendering.kit for headless camera parity.
    --gui   Run with a real UI on the current DISPLAY.
    --xvfb  Run on a managed Xvfb virtual display.
    --headless  Explicitly request headless (the default, accepted for clarity).
"""

import argparse
import os
import shlex
import signal
import subprocess
import sys
from pathlib import Path


# ── Parse arguments ──────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Batch replay HDF5 episodes with selectable capture.")
parser.add_argument(
    "--origin-dir",
    type=str,
    required=True,
    help="Directory containing original episode_*.hdf5 files.",
)
parser.add_argument(
    "--replay-dir",
    type=str,
    required=True,
    help="Output directory for replayed episode_*.hdf5 files.",
)
parser.add_argument("--enable-rgb", action="store_true", help="Capture RGB frames during replay.")
parser.add_argument("--enable-depth", action="store_true", help="Capture depth frames during replay.")
parser.add_argument("--enable-tactile", action="store_true", help="Capture TacMap tactile frames during replay.")
parser.add_argument(
    "--tactile-only",
    action="store_true",
    help="Capture tactile data only; disables RGB/depth and implies --enable-tactile.",
)
parser.add_argument(
    "--resample-groups",
    type=str,
    default=None,
    help=(
        "Comma-separated scene generalization groups to resample from the current config. "
        "When omitted, replay restores the HDF5 generalization sample."
    ),
)
parser.add_argument(
    "--generalization-config",
    type=str,
    default=None,
    help="Path to scene generalization YAML. Defaults to configs/scene/generalization.yaml.",
)
parser.add_argument(
    "--generalization-split",
    type=str,
    default=None,
    choices=["seen", "unseen", "all"],
    help="Discrete visual asset split for resampled scene generalization.",
)
display_group = parser.add_mutually_exclusive_group()
display_group.add_argument(
    "--gui",
    action="store_true",
    help="Run replay with a real GUI on the current DISPLAY (does not pass --headless).",
)
display_group.add_argument(
    "--xvfb",
    action="store_true",
    help="Run replay as a headed Isaac app on a managed Xvfb display (does not pass --headless).",
)
display_group.add_argument(
    "--headless",
    action="store_true",
    help="Run replay headless (the default). Accepted for clarity; has no effect.",
)
parser.add_argument(
    "--no-validate",
    action="store_true",
    help="Skip post-replay RGB validation. By default, missing camera data is detected and auto-retried.",
)
args = parser.parse_args()

REPO_DIR = Path(__file__).resolve().parents[2]  # dex2bench repo root
ORIGIN_DIR = Path(args.origin_dir)
REPLAY_DIR = Path(args.replay_dir)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resolve_capture_flags(namespace: argparse.Namespace) -> tuple[bool, bool, bool, bool]:
    enable_rgb = bool(namespace.enable_rgb)
    enable_depth = bool(namespace.enable_depth)
    enable_tactile = bool(namespace.enable_tactile)
    tactile_only = bool(namespace.tactile_only)

    if tactile_only:
        return False, False, True, True
    if not (enable_rgb or enable_depth or enable_tactile):
        enable_rgb = True
    return enable_rgb, enable_depth, enable_tactile, False


def _resolve_display_mode(namespace: argparse.Namespace) -> str:
    if namespace.xvfb:
        return "xvfb"
    if namespace.gui:
        return "gui"
    return "headless"


def _display_mode_label(display_mode: str) -> str:
    if display_mode == "xvfb":
        return "headed+xvfb (no --headless)"
    if display_mode == "gui":
        return "headed real UI (no --headless)"
    return "headless"


def _build_replay_command(
    hdf5_path: Path,
    output_path: Path,
    *,
    enable_rgb: bool,
    enable_depth: bool,
    enable_tactile: bool,
    tactile_only: bool,
    display_mode: str,
    resample_groups: str | None,
    generalization_config: str | None,
    generalization_split: str | None,
    background_asset_index: int | None,
) -> list[str]:
    parts = [
        sys.executable,
        "replay.py",
        "--hdf5",
        str(hdf5_path),
        "--output",
        str(output_path),
    ]
    if display_mode == "headless":
        parts.append("--headless")
    if resample_groups:
        parts.extend(["--enable-generalization", "--resample-groups", resample_groups])
        if generalization_config:
            parts.extend(["--generalization-config", generalization_config])
        if generalization_split:
            parts.extend(["--generalization-split", generalization_split])
        if background_asset_index is not None:
            parts.extend(["--background-asset-index", str(background_asset_index)])
    else:
        parts.append("--restore-generalization")  # restore scene params (table height, etc.) from HDF5
    if tactile_only:
        parts.append("--tactile-only")
    else:
        if enable_rgb:
            parts.append("--enable-rgb")
        if enable_depth:
            parts.append("--enable-depth")
        if enable_tactile:
            parts.append("--enable-tactile")
    return parts


def _resamples_background(resample_groups: str | None) -> bool:
    groups = {g.strip() for g in str(resample_groups or "").split(",") if g.strip()}
    return "background" in groups


def _kill_process_group(proc: subprocess.Popen, grace: float = 3.0) -> None:
    """Send SIGTERM then SIGKILL to the entire process group of *proc*."""
    if proc.poll() is not None:
        return  # already exited
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    # Graceful shutdown first
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


# ── Post-replay validation ────────────────────────────────────────────────────

_REQUIRED_RGB_CAMERAS = (
    "cam_wrist_right",
    "cam_wrist_left",
    "cam_stereo_left",
    "cam_stereo_right",
)


def _validate_replay_output(
    hdf5_path: Path,
    *,
    enable_rgb: bool,
    enable_depth: bool,
) -> tuple[bool, str]:
    """Return (is_valid, reason).  Checks that expected camera data exists."""
    if not enable_rgb and not enable_depth:
        return True, "no camera capture requested"
    try:
        import h5py
        with h5py.File(hdf5_path, "r") as f:
            cameras = f.get("cameras")
            if cameras is None:
                return False, "no /cameras group"
            missing: list[str] = []
            for cam_id in cameras.keys():
                grp = cameras.get(cam_id)
                if enable_rgb and "rgb" not in grp:
                    missing.append(f"{cam_id}/rgb")
                elif enable_rgb and grp["rgb"].shape[0] == 0:
                    missing.append(f"{cam_id}/rgb (empty)")
                if enable_depth and "depth" not in grp:
                    missing.append(f"{cam_id}/depth")
            if missing:
                return False, f"missing: {', '.join(sorted(missing))}"
            return True, "ok"
    except Exception as exc:
        return False, f"read error: {exc}"


def _auto_retry_broken_episodes(
    broken: list[Path],
    replay_dir: Path,
    *,
    enable_rgb: bool,
    enable_depth: bool,
    enable_tactile: bool,
    tactile_only: bool,
    display_mode: str,
    subprocess_env: dict,
) -> tuple[int, int, list[str]]:
    """Re-replay broken episodes using --restore-generalization from the broken file.

    The broken file already contains the correct generalization metadata from the
    batch replay pass; using it as the source HDF5 with --restore-generalization
    preserves the coordinated resampling (background index cycling, etc.).
    """
    import shutil as _shutil

    fixed = 0
    still_failed = 0
    failed_names: list[str] = []

    for broken_path in broken:
        name = broken_path.name
        temp_output = broken_path.with_name(broken_path.stem + "_retry.hdf5")
        backup_dir = replay_dir / ".bad"
        backup_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[validate] Retrying {name} with --restore-generalization ...")
        retry_cmd = _build_replay_command(
            broken_path,
            temp_output,
            enable_rgb=enable_rgb,
            enable_depth=enable_depth,
            enable_tactile=enable_tactile,
            tactile_only=tactile_only,
            display_mode=display_mode,
            resample_groups=None,          # triggers --restore-generalization
            generalization_config=None,
            generalization_split=None,
            background_asset_index=None,
        )
        print(f"[validate] CMD: {' '.join(shlex.quote(p) for p in retry_cmd)}")

        proc = subprocess.Popen(
            retry_cmd,
            start_new_session=True,
            env=subprocess_env,
        )
        ret = proc.wait()

        if ret != 0 or not temp_output.exists():
            print(f"[validate] FAILED: {name} replay exit={ret}")
            still_failed += 1
            failed_names.append(name)
            if temp_output.exists():
                temp_output.unlink()
            continue

        valid, reason = _validate_replay_output(
            temp_output,
            enable_rgb=enable_rgb,
            enable_depth=enable_depth,
        )
        if not valid:
            print(f"[validate] FAILED: {name} retry still invalid: {reason}")
            still_failed += 1
            failed_names.append(name)
            temp_output.unlink()
            continue

        # Backup the broken file, replace with the fixed one
        backup_path = backup_dir / name
        _shutil.move(str(broken_path), str(backup_path))
        _shutil.move(str(temp_output), str(broken_path))
        print(f"[validate] FIXED: {name} (broken backup → .bad/{name})")
        fixed += 1

    return fixed, still_failed, failed_names


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    REPLAY_DIR.mkdir(parents=True, exist_ok=True)
    os.chdir(REPO_DIR)

    episodes = sorted(ORIGIN_DIR.glob("episode_*.hdf5"))
    if not episodes:
        print(f"[ERROR] No episode_*.hdf5 files found in {ORIGIN_DIR}")
        return 1

    overwrite = os.environ.get("OVERWRITE", "1") == "1"

    display_mode = _resolve_display_mode(args)

    # Xvfb virtual display for headed-on-virtual-display mode.
    # This is intentionally separate from headless mode: --xvfb does not pass
    # --headless to replay.py, so IsaacLab treats it as a GUI render path.
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

    # Track the currently-running replay subprocess for cleanup.
    _active_proc = None

    def _kill_active_replay() -> None:
        nonlocal _active_proc
        if _active_proc is not None:
            _kill_process_group(_active_proc)
            _active_proc = None

    if display_mode == "xvfb":
        import shutil as _shutil
        _xvfb_bin = _shutil.which("Xvfb")
        if _xvfb_bin is None:
            print("[batch][ERROR] --xvfb requested but Xvfb was not found. Use --gui for the current DISPLAY.", flush=True)
            return 1
        else:
            _xvfb_display = os.environ.get("DEX2BENCH_XVFB_DISPLAY", ":99")
            _xvfb_proc = subprocess.Popen(
                [_xvfb_bin, _xvfb_display, "-screen", "0", "1280x720x24", "-ac"],
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
                print("[batch][ERROR] Xvfb did not start. Use --gui for the current DISPLAY.", flush=True)
                _stop_xvfb()
                return 1
    _subprocess_env = os.environ.copy()
    if _xvfb_display is not None:
        _subprocess_env["DISPLAY"] = _xvfb_display
        print(f"[batch] Xvfb virtual display: {_xvfb_display}", flush=True)
    # ── end Xvfb setup ─────────────────────────────────────────────────

    ok_count = 0
    fail_count = 0
    skip_count = 0
    failed: list[str] = []
    enable_rgb, enable_depth, enable_tactile, tactile_only = _resolve_capture_flags(args)

    print(f"[batch] Origin : {ORIGIN_DIR}")
    print(f"[batch] Replay : {REPLAY_DIR}")
    print(f"[batch] Episodes: {len(episodes)}")
    print(f"[batch] OVERWRITE={1 if overwrite else 0}")
    print(
        "[batch] Capture: "
        f"rgb={enable_rgb}, depth={enable_depth}, tactile={enable_tactile}, tactile_only={tactile_only}"
    )
    print(f"[batch] Display : {_display_mode_label(display_mode)}")
    if display_mode == "headless":
        print("[batch] Rendering: replay.py --headless uses isaaclab.python.rendering.kit for camera parity")
    elif display_mode == "xvfb":
        print("[batch] Rendering: Xvfb provides a virtual display; replay.py is launched without --headless")
    else:
        print("[batch] Rendering: real UI path on the current DISPLAY; replay.py is launched without --headless")
    if args.resample_groups:
        print(
            "[batch] Generalization: "
            f"resample_groups={args.resample_groups}, "
            f"split={args.generalization_split or '<config>'}, "
            f"config={args.generalization_config or '<default>'}"
        )
        if _resamples_background(args.resample_groups):
            print("[batch] Background sequence: episode order maps to filtered background index, cycling if needed")
    else:
        print("[batch] Generalization: restore from HDF5")

    interrupted = False
    proc: subprocess.Popen | None = None
    try:
        for idx, hdf5_path in enumerate(episodes, start=1):
            output_path = REPLAY_DIR / hdf5_path.name
            if output_path.exists() and not overwrite:
                print(f"[SKIP] {idx}/{len(episodes)} {hdf5_path.name}")
                skip_count += 1
                continue

            cmd_list = _build_replay_command(
                hdf5_path,
                output_path,
                enable_rgb=enable_rgb,
                enable_depth=enable_depth,
                enable_tactile=enable_tactile,
                tactile_only=tactile_only,
                display_mode=display_mode,
                resample_groups=args.resample_groups,
                generalization_config=args.generalization_config,
                generalization_split=args.generalization_split,
                background_asset_index=(idx - 1) if _resamples_background(args.resample_groups) else None,
            )
            cmd_str = " ".join(shlex.quote(p) for p in cmd_list)
            print(f"\n{'=' * 80}")
            print(f"[episode {idx}/{len(episodes)}] {hdf5_path.name}")
            print(f"CMD: {cmd_str}")
            print(f"{'=' * 80}\n")

            # ── Launch replay.py in its own process group ──────────────
            # start_new_session=True creates a new process group (PGID ==
            # child PID) so os.killpg() kills the entire tree — Isaac Sim
            # and all its subprocesses — with a single call.
            _active_proc = subprocess.Popen(cmd_list, start_new_session=True,
                                            env=_subprocess_env)
            try:
                ret = _active_proc.wait()
            except KeyboardInterrupt:
                print(f"\n[batch] Interrupted — stopping {hdf5_path.name} ...",
                      flush=True)
                signal.signal(signal.SIGINT, signal.SIG_IGN)
                _kill_active_replay()
                _stop_xvfb()
                interrupted = True
                break
            finally:
                _active_proc = None
            if ret != 0:
                print(f"\n[ERROR] Replay failed: {hdf5_path.name}, exit={ret}")
                fail_count += 1
                failed.append(hdf5_path.name)
                continue

            if output_path.exists():
                print(f"[OK] {hdf5_path.name} -> {output_path}")
                ok_count += 1
            else:
                print(f"[ERROR] Output file missing: {output_path}")
                fail_count += 1
                failed.append(hdf5_path.name)
    except KeyboardInterrupt:
        interrupted = True
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        _kill_active_replay()

    _stop_xvfb()

    # ── Post-replay validation: detect & auto-retry episodes missing RGB ──
    validate_fixed = 0
    validate_still_failed: list[str] = []
    if enable_rgb and not args.no_validate and not interrupted:
        print(f"\n{'=' * 80}")
        print("[validate] Scanning output files for missing camera data ...")
        broken: list[Path] = []
        for hdf5_path in sorted(REPLAY_DIR.glob("episode_*.hdf5")):
            # Skip temp/backup files from previous retries
            if "_retry" in hdf5_path.stem:
                continue
            valid, reason = _validate_replay_output(
                hdf5_path,
                enable_rgb=enable_rgb,
                enable_depth=enable_depth,
            )
            if valid:
                continue
            print(f"[validate] BROKEN: {hdf5_path.name} — {reason}")
            broken.append(hdf5_path)

        if broken:
            print(
                f"\n[validate] {len(broken)} episode(s) missing camera data. "
                "Auto-retrying with --restore-generalization ..."
            )
            validate_fixed, still_broken, validate_still_failed = _auto_retry_broken_episodes(
                broken,
                REPLAY_DIR,
                enable_rgb=enable_rgb,
                enable_depth=enable_depth,
                enable_tactile=enable_tactile,
                tactile_only=tactile_only,
                display_mode=display_mode,
                subprocess_env=_subprocess_env,
            )
            fail_count += still_broken
            ok_count += validate_fixed
            # Remove fixed episodes from the failed list if they were there
            fixed_names = {p.name for p in broken} - {n for n in validate_still_failed}
            failed = [n for n in failed if n not in fixed_names]
            failed.extend(validate_still_failed)
        else:
            print("[validate] All episodes have camera data.")

    print(f"\n{'=' * 80}")
    if interrupted:
        print("[batch] Interrupted by user")
    print("[batch] Done")
    print(f"  OK:      {ok_count}")
    print(f"  Skipped: {skip_count}")
    print(f"  Failed:  {fail_count}")
    if validate_fixed:
        print(f"  Auto-fixed: {validate_fixed} (broken backup → .bad/)")
    if failed:
        print("[batch] Failed episodes:")
        for name in failed:
            print(f"  {name}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
