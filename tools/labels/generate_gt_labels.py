"""Generate GT labels (occupancy + 3D/2D boxes) for an episode directory.

Pipeline (defaults: inplace write, overwrite, cuda, 2 workers):
  1. discover base files `episode_*.hdf5` (excluding `*_1.hdf5`) in <dir>;
  2. generate `labels/occupancy_gt` for each base file using the per-episode
     *auto* bounds (height-generalization aware: z floor = actual tabletop -
     slab thickness, z_max = 1.50, y_max widened for the microwave). Live
     progress + ETA are printed to the console.
  3. for each base that succeeded, verify its `_1` sibling has identical
     object assets + per-frame pose/qpos (safety guard); if so, byte-copy the
     `labels/occupancy_gt` group into the `_1` file (overwrite).
  4. (on by default) generate `labels/box3d` (+`labels/box2d`) for EVERY file
     (base and `_1`) directly — box labels are mesh-free and ~350x cheaper
     than occupancy, so they are computed per-file rather than copied.

Usage:
    python tools/labels/generate_gt_labels.py <episode_dir>
    python tools/labels/generate_gt_labels.py <dir> --workers 4
    python tools/labels/generate_gt_labels.py <dir> --no-copy

Note: do NOT add a `--bounds` option here. The per-episode auto bounds handle
the table-height generalization; a fixed bounds box would break episodes whose
table height has a generalization offset.
"""

from __future__ import annotations

import argparse
import functools
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import h5py
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from collector.config import normalize_occupancy_gt_config
from tools.labels.generate_occupancy_gt import generate_occupancy_gt_for_file
from tools.labels.generate_box_labels import generate_box_labels_for_file


# ---------------------------------------------------------------------------
# discovery + helpers
# ---------------------------------------------------------------------------

def find_base_files(directory: str) -> list[str]:
    """Return sorted base episode paths (excludes `*_1.hdf5` siblings)."""
    out: list[str] = []
    for name in sorted(os.listdir(directory)):
        if name.endswith(".hdf5") and not name.endswith("_1.hdf5") \
                and name.startswith("episode_"):
            out.append(os.path.join(directory, name))
    return out


def find_all_files(directory: str) -> list[str]:
    """Return sorted base + `_1` sibling paths (every episode_*.hdf5)."""
    out: list[str] = []
    for name in sorted(os.listdir(directory)):
        if name.endswith(".hdf5") and name.startswith("episode_"):
            out.append(os.path.join(directory, name))
    return out


def sibling_1_path(base_path: str) -> str:
    """episode_000000.hdf5 -> episode_000000_1.hdf5"""
    return base_path[:-5] + "_1.hdf5"


def _fmt_dur(seconds: float) -> str:
    s = int(max(0.0, seconds))
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _record_done(log_path: str, fields: dict) -> None:
    """Append one line summarizing a finished/failed directory run.

    Called from main()'s finally, so it runs on success, interruption, and
    exceptions alike. One line per invocation: timestamp + key=val pairs.
    """
    try:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = " | ".join([ts] + [f"{k}={v}" for k, v in fields.items()])
        d = os.path.dirname(log_path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        print(f"[log] recorded -> {log_path}", flush=True)
    except Exception as exc:  # logging must never mask the real result
        print(f"[log] FAILED to write {log_path}: {exc}", flush=True)


def _read_meta_str(f: h5py.File, key: str) -> str | None:
    m = f.get("meta")
    if m is None or key not in m:
        return None
    v = m[key][()]
    if isinstance(v, bytes):
        return v.decode()
    if hasattr(v, "item"):
        return str(v.item())
    return str(v)


def verify_identical_scene(a: h5py.File, c: h5py.File) -> bool:
    """Safety guard: base and `_1` must share identical object assets + poses.

    Occupancy is a function of (object meshes, per-frame poses). Only when
    assets + every pose/qpos frame are byte-identical is a copy valid.
    """
    if _read_meta_str(a, "object_asset_paths") != _read_meta_str(c, "object_asset_paths"):
        return False
    if _read_meta_str(a, "object_body_types") != _read_meta_str(c, "object_body_types"):
        return False
    try:
        objs = list(a["objects"].keys())
        if objs != list(c["objects"].keys()):
            return False
        for o in objs:
            pa = a["objects"][o]["pose_world"][:]
            pc = c["objects"][o]["pose_world"][:]
            if pa.shape != pc.shape or not np.array_equal(pa, pc):
                return False
            if "qpos" in a["objects"][o] and "qpos" in c["objects"][o]:
                qa = a["objects"][o]["qpos"][:]
                qc = c["objects"][o]["qpos"][:]
                if not np.array_equal(qa, qc):
                    return False
    except Exception:
        return False
    return True


# ---------------------------------------------------------------------------
# generation phase (parallel, with progress + ETA)
# ---------------------------------------------------------------------------

def _eta(t0: float, done: int, total: int) -> tuple[float, float]:
    elapsed = time.time() - t0
    if done <= 0:
        return elapsed, 0.0
    throughput = done / elapsed
    return elapsed, (total - done) / throughput if throughput > 0 else 0.0


def run_generation(
    base_files: list[str],
    *,
    workers: int,
    device: str,
    voxel_size: float,
    geometry_source: str,
    overwrite: bool,
    heartbeat: float,
) -> tuple[list[str], int, int]:
    cfg = normalize_occupancy_gt_config({
        "voxel_size": float(voxel_size),
        "geometry_source": geometry_source,
        "label_version": "occupancy_gt_v1",
    })
    fn = functools.partial(
        generate_occupancy_gt_for_file,
        occupancy_config=cfg,
        write_mode="inplace",
        overwrite=overwrite,
        device=device,
        bounds_override=None,        # use per-episode auto bounds
    )

    total = len(base_files)
    t0 = time.time()
    print(f"[gen] start {total} base episodes | device={device} workers={workers} "
          f"bounds=auto(per-episode) overwrite={overwrite}", flush=True)

    succeeded: list[str] = []
    n_ok = n_fail = 0
    interrupted = False

    # Sliding-window submit: at most `workers` futures in flight at once, so
    # `in_flight` is exactly the set of episodes currently being generated.
    # Using wait(timeout=heartbeat, FIRST_COMPLETED) wakes us both on each
    # completion AND on the heartbeat (to print running names + ETA when an
    # episode takes long). We deliberately do NOT use a `with` block: its
    # __exit__ calls shutdown(wait=True), which blocks until running episodes
    # finish (~minutes), making Ctrl-C unresponsive.
    it = iter(base_files)
    in_flight: dict = {}
    pool = ProcessPoolExecutor(max_workers=workers)
    try:
        for _ in range(workers):
            try:
                p = next(it)
            except StopIteration:
                break
            in_flight[pool.submit(fn, p)] = p

        while in_flight:
            done, _ = wait(list(in_flight.keys()), timeout=heartbeat,
                           return_when=FIRST_COMPLETED)
            if not done:
                elapsed, eta = _eta(t0, n_ok + n_fail, total)
                running = [os.path.basename(in_flight[f]) for f in in_flight]
                print(f"[gen] {n_ok + n_fail}/{total} done | running {running} | "
                      f"elapsed {_fmt_dur(elapsed)} | eta ~{_fmt_dur(eta)}", flush=True)
                continue

            for fut in done:
                path = in_flight.pop(fut)
                try:
                    ok = bool(fut.result())
                except Exception as exc:
                    ok = False
                    print(f"[gen] ERROR {os.path.basename(path)}: {exc}", flush=True)
                if ok:
                    n_ok += 1
                    succeeded.append(path)
                else:
                    n_fail += 1
                elapsed, eta = _eta(t0, n_ok + n_fail, total)
                tag = "OK  " if ok else "FAIL"
                print(f"[gen] {n_ok + n_fail}/{total} {tag} "
                      f"{os.path.basename(path)} | elapsed {_fmt_dur(elapsed)} | "
                      f"eta ~{_fmt_dur(eta)}", flush=True)
                # refill the window with the next queued episode
                try:
                    np_ = next(it)
                    in_flight[pool.submit(fn, np_)] = np_
                except StopIteration:
                    pass
    except KeyboardInterrupt:
        interrupted = True
        print("\n[gen] interrupted (Ctrl-C): cancelling queued + killing workers...",
              flush=True)
        for f in list(in_flight):
            f.cancel()
        # Force-kill running worker processes so we don't block on an
        # in-flight ~minutes-long episode. _processes is CPython-private but
        # stable; guard with getattr in case it changes.
        for proc in list(getattr(pool, "_processes", {}).values()):
            try:
                proc.kill()
            except Exception:
                pass
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    elapsed = time.time() - t0
    if interrupted:
        print(f"[gen] stopped: ok={n_ok} fail={n_fail} "
              f"(interrupted, {_fmt_dur(elapsed)})", flush=True)
    else:
        print(f"[gen] finished: ok={n_ok} fail={n_fail} ({_fmt_dur(elapsed)})",
              flush=True)
    return succeeded, n_ok, n_fail, interrupted


# ---------------------------------------------------------------------------
# copy phase (base -> _1, guarded)
# ---------------------------------------------------------------------------

def run_copy(succeeded: list[str], *, overwrite: bool) -> int:
    total = len(succeeded)
    print(f"[copy] start: propagate occupancy_gt base->_1 for {total} pairs "
          f"(safety guard on, overwrite={overwrite})", flush=True)
    t0 = time.time()
    n_copied = n_skipped = n_no_sib = 0
    for i, b0 in enumerate(succeeded, 1):
        b1 = sibling_1_path(b0)
        name = os.path.basename(b0)
        if not os.path.isfile(b1):
            print(f"[copy] {i}/{total} {name}: NO _1 sibling, skip", flush=True)
            n_no_sib += 1
            continue
        try:
            with h5py.File(b0, "r") as a, h5py.File(b1, "r+") as c:
                if "labels/occupancy_gt" not in a:
                    print(f"[copy] {i}/{total} {name}: base has no occupancy_gt, skip",
                          flush=True)
                    n_skipped += 1
                    continue
                if not verify_identical_scene(a, c):
                    print(f"[copy] {i}/{total} {name}: GUARD FAIL (assets/pose differ) "
                          f"-> skip, regenerate this pair separately", flush=True)
                    n_skipped += 1
                    continue
                dest = c.require_group("labels")
                if "occupancy_gt" in dest:
                    if not overwrite:
                        print(f"[copy] {i}/{total} {name}: _1 has occupancy_gt & "
                              f"overwrite off, skip", flush=True)
                        n_skipped += 1
                        continue
                    del dest["occupancy_gt"]
                # H5Ocopy across files: preserves chunked + gzip(opts=9) layout.
                dest.copy(a["labels/occupancy_gt"], "occupancy_gt")
                c.flush()
                n_copied += 1
                print(f"[copy] {i}/{total} {name} -> {os.path.basename(b1)}: copied",
                      flush=True)
        except Exception as exc:
            print(f"[copy] {i}/{total} {name}: ERROR {exc}", flush=True)
            n_skipped += 1
    print(f"[copy] finished: copied={n_copied} skipped={n_skipped} "
          f"no_sibling={n_no_sib} ({_fmt_dur(time.time() - t0)})", flush=True)
    return n_copied


# ---------------------------------------------------------------------------
# box-label phase (box3d + optional box2d), run on every file (base + _1)
# ---------------------------------------------------------------------------

def run_boxes(
    all_files: list[str],
    *,
    device: str,
    overwrite: bool,
    box2d: bool,
    heartbeat: float,
) -> None:
    """Generate box3d (+box2d) for every file (base and _1 alike).

    Box labels depend only on each file's own poses + local_bboxes (+ cameras
    for box2d), not on meshes, so they are ~350x cheaper than occupancy
    (seconds, not minutes, per episode). They are computed per-file (no
    base->_1 copy) because `_1` generalization variants may carry different
    camera intrinsics/extrinsics. Run sequentially on one GPU (no fork) to
    avoid CUDA-context sharing issues.
    """
    total = len(all_files)
    mode = "box3d+box2d" if box2d else "box3d"
    print(f"[box] start {total} files | mode={mode} device={device} "
          f"overwrite={overwrite} (sequential, ~1.4s/ep)", flush=True)
    t0 = time.time()
    n_ok = n_err = 0
    for i, p in enumerate(all_files, 1):
        name = os.path.basename(p)
        t_ep = time.time()
        try:
            ok = generate_box_labels_for_file(
                p, box2d=box2d, write_mode="inplace", overwrite=overwrite,
                device=device,
            )
            ok = bool(ok)
        except Exception as exc:
            ok = False
            print(f"[box] ERROR {name}: {exc}", flush=True)
        if ok:
            n_ok += 1
        else:
            n_err += 1
        elapsed = time.time() - t0
        avg = elapsed / i
        eta = avg * (total - i)
        tag = "OK  " if ok else "FAIL"
        print(f"[box] {i}/{total} {tag} {name} ({time.time()-t_ep:.1f}s) | "
              f"elapsed {_fmt_dur(elapsed)} | eta ~{_fmt_dur(eta)}", flush=True)
    print(f"[box] finished: ok={n_ok} err={n_err} ({_fmt_dur(time.time()-t0)})",
          flush=True)
    return n_ok, n_err


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate occupancy for base episodes and copy to _1 siblings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("directory", help="directory with episode_*.hdf5 + _1 siblings")
    ap.add_argument("--workers", type=int, default=2,
                    help="parallel generation workers (2 already saturates one GPU)")
    ap.add_argument("--device", default="cuda", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--voxel-size", type=float, default=0.01)
    ap.add_argument("--geometry-source", choices=["collision", "visual"],
                    default="collision")
    ap.add_argument("--no-overwrite", action="store_true",
                    help="do not overwrite existing occupancy_gt")
    ap.add_argument("--no-copy", action="store_true",
                    help="only generate, skip base->_1 copy")
    ap.add_argument("--no-boxes", action="store_true",
                    help="skip box3d/box2d generation (on by default)")
    ap.add_argument("--no-box2d", action="store_true",
                    help="generate box3d only, skip 2D box projection")
    ap.add_argument("--heartbeat", type=float, default=60.0,
                    help="seconds between progress prints while idle")
    ap.add_argument(
        "--log",
        default="../jcy/occupancy/done.txt",
        help="append a one-line record per finished/failed directory run",
    )
    args = ap.parse_args()

    directory = os.path.abspath(args.directory)
    if not os.path.isdir(directory):
        print(f"[ERROR] not a directory: {directory}", file=sys.stderr)
        return 2
    base = find_base_files(directory)
    if not base:
        print(f"[ERROR] no base episode_*.hdf5 found in {directory}", file=sys.stderr)
        return 2

    sibs = sum(1 for b in base if os.path.isfile(sibling_1_path(b)))
    print(f"[plan] {len(base)} base episodes | {sibs}/{len(base)} _1 siblings "
          f"present for copy", flush=True)
    print(f"[plan] defaults: inplace, overwrite={not args.no_overwrite}, "
          f"device={args.device}, workers={args.workers}, "
          f"voxel_size={args.voxel_size}, geometry={args.geometry_source}, "
          f"boxes={'on' if not args.no_boxes else 'off'} "
          f"box2d={'on' if not args.no_boxes and not args.no_box2d else 'off'}",
          flush=True)

    overwrite = not args.no_overwrite
    log_path = args.log
    t_start = time.time()
    rec: dict = {
        "dir": directory, "base": len(base), "sibs": sibs,
        "workers": args.workers, "device": args.device,
    }
    status = "completed"
    rc = 0
    try:
        succeeded, g_ok, g_fail, g_int = run_generation(
            base,
            workers=args.workers,
            device=args.device,
            voxel_size=args.voxel_size,
            geometry_source=args.geometry_source,
            overwrite=overwrite,
            heartbeat=args.heartbeat,
        )
        rec["gen_ok"] = g_ok
        rec["gen_fail"] = g_fail
        rec["gen_interrupted"] = g_int

        if g_int:
            # Ctrl-C during generation: skip copy/box, record as interrupted.
            status = "interrupted"
            rc = 130
            rec["copy"] = "skipped(interrupted)"
            rec["box"] = "skipped(interrupted)"
        elif g_ok == 0:
            status = "failed:no_episode_succeeded"
            rc = 1
            rec["copy"] = "skipped"
            rec["box"] = "skipped"
        else:
            if not args.no_copy:
                rec["copy"] = run_copy(succeeded, overwrite=overwrite)
            else:
                rec["copy"] = "skipped(--no-copy)"
            if not args.no_boxes:
                all_files = find_all_files(directory)
                b_ok, b_err = run_boxes(
                    all_files,
                    device=args.device,
                    overwrite=overwrite,
                    box2d=not args.no_box2d,
                    heartbeat=args.heartbeat,
                )
                rec["box_ok"] = b_ok
                rec["box_err"] = b_err
            else:
                rec["box"] = "skipped(--no-boxes)"
    except KeyboardInterrupt:
        # Ctrl-C during copy/box phases (run_generation handles its own).
        status = "interrupted"
        rc = 130
    except Exception as exc:
        status = f"error:{type(exc).__name__}"
        rec["error"] = str(exc)[:120]
        rc = 1
    finally:
        rec["status"] = status
        rec["elapsed_s"] = int(time.time() - t_start)
        _record_done(log_path, rec)
        print(f"[done] status={status} elapsed={_fmt_dur(time.time()-t_start)} rc={rc}",
              flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
