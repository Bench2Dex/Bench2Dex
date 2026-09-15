"""Generate RGB-augmented copies of origin HDF5 episodes.

For each episode in --origin-dir, creates N copies in --output-dir (N = --multiplier),
each with a new randomly-sampled appearance (background, table surface, lighting).
Spatial layout (object positions, camera offsets, table height), clutter,
embodiment, and robot motions are preserved unchanged so the physical task
remains identical but the visual appearance is different.

Usage:
    python tools/gen_rgb_augmented_episodes.py \
        --origin-dir outputs/ur5_rh56dfx/scenes/06_fruit_bowl_loading/origin \
        --output-dir outputs/ur5_rh56dfx/scenes/06_fruit_bowl_loading/origin_generate \
        --multiplier 2 \
        [--seed 42]

  With --multiplier 2 and 52 source episodes, output will be
  episode_000000.hdf5 ... episode_000103.hdf5 (104 total).

After running, replay origin_generate with batch_replay.py to capture new RGB:
    python tools/replay/batch_replay.py \
        --origin-dir outputs/.../origin_generate \
        --replay-dir outputs/.../replay_generate
"""

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import h5py

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Generate RGB-only augmented episode copies.")
parser.add_argument("--origin-dir", required=True, help="Directory with original episode_*.hdf5 files.")
parser.add_argument("--output-dir", required=True, help="Output directory for augmented copies.")
parser.add_argument("--multiplier", type=int, default=1,
                    help="Number of augmented copies to generate per source episode (default 1).")
parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
args = parser.parse_args()

if args.multiplier < 1:
    print("[ERROR] --multiplier must be >= 1")
    sys.exit(1)

_seed = args.seed if args.seed is not None else random.randint(0, 2**31 - 1)
rng = random.Random(_seed)
print(f"[INFO] RNG seed: {_seed}")

ORIGIN_DIR = Path(args.origin_dir)
OUTPUT_DIR = Path(args.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Helpers ────────────────────────────────────────────────────────────────────

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _sample_sym(half_range: float) -> float:
    return rng.uniform(-half_range, half_range)


def _rpy_deg_to_quat(roll_deg: float, pitch_deg: float, yaw_deg: float) -> list:
    """Convert roll-pitch-yaw (degrees) to quaternion [qw, qx, qy, qz].
    Matches build/generalization.py's convention: _rpy_to_quat((0, pitch, yaw)).
    """
    import math
    cr = math.cos(math.radians(roll_deg) * 0.5)
    sr = math.sin(math.radians(roll_deg) * 0.5)
    cp = math.cos(math.radians(pitch_deg) * 0.5)
    sp = math.sin(math.radians(pitch_deg) * 0.5)
    cy = math.cos(math.radians(yaw_deg) * 0.5)
    sy = math.sin(math.radians(yaw_deg) * 0.5)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return [qw, qx, qy, qz]


def _sample_new_appearance(gen_config: dict, orig_appearance: dict) -> dict:
    """Sample a new appearance dict from gen_config, keeping only visual randomisation.

    - background:   pick a random asset from config assets list
    - table_surface: pick a random asset from config assets list
    - light:        re-sample intensity/color/direction/angle within config ranges
    """
    app_cfg = gen_config.get("appearance", {})

    # ── Background ────────────────────────────────────────────────────────────
    bg_cfg = app_cfg.get("background", {})
    bg_assets = bg_cfg.get("assets", [])
    if bg_assets:
        chosen_bg = rng.choice(bg_assets)
    else:
        chosen_bg = None

    orig_bg = orig_appearance.get("background", {})
    if chosen_bg:
        new_bg = {
            "enabled": True,
            "clean_background": False,
            "intensity": float(bg_cfg.get("intensity", orig_bg.get("intensity", 1500.0))),
            "yaw_deg": float(chosen_bg.get("yaw_deg", 0.0)),
            "base_yaw_deg": float(chosen_bg.get("yaw_deg", 0.0)),
            "flip_180_applied": False,
            "physics_enabled": bool(chosen_bg.get("physics_enabled", False)),
            "fill_light_intensity": float(bg_cfg.get("fill_light_intensity", orig_bg.get("fill_light_intensity", 800.0))),
            "fill_light_jitter_ratio": float(bg_cfg.get("fill_light_jitter_ratio", orig_bg.get("fill_light_jitter_ratio", 0.2))),
            "asset_kind": chosen_bg.get("kind", "image"),
            "asset_category": chosen_bg.get("category", "Indoor"),
            "asset_uri": chosen_bg.get("uri", orig_bg.get("asset_uri", "")),
            "scene_offset": list(chosen_bg.get("scene_offset", [0.0, 0.0, 0.0])),
        }
    else:
        new_bg = dict(orig_bg)

    # ── Table surface ─────────────────────────────────────────────────────────
    ts_cfg = app_cfg.get("table_surface", {})
    ts_assets = ts_cfg.get("assets", [])
    orig_ts = orig_appearance.get("table_surface", {})
    if ts_assets:
        chosen_ts = rng.choice(ts_assets)
        new_ts = {
            "enabled": True,
            "clean_surface": False,
            "asset_name": chosen_ts.get("name", orig_ts.get("asset_name", "")),
            "asset_uri": chosen_ts.get("uri", orig_ts.get("asset_uri", "")),
        }
    else:
        new_ts = dict(orig_ts)

    # ── Light ─────────────────────────────────────────────────────────────────
    light_cfg = app_cfg.get("light", {})
    base_intensity = float(light_cfg.get("intensity", 3600.0))
    intensity_jitter = float(light_cfg.get("intensity_jitter_ratio", 0.15))
    base_color = light_cfg.get("color", [0.8, 0.8, 0.8])
    color_jitter = float(light_cfg.get("color_jitter_abs", 0.06))
    base_yaw = float(light_cfg.get("direction_yaw_deg", 0.0))
    base_pitch = float(light_cfg.get("direction_pitch_deg", -60.0))
    yaw_jitter = float(light_cfg.get("direction_yaw_jitter_deg", 12.0))
    pitch_jitter = float(light_cfg.get("direction_pitch_jitter_deg", 8.0))
    base_angle = float(light_cfg.get("angle_deg", 1.0))
    angle_jitter = float(light_cfg.get("angle_jitter_deg", 0.5))
    extreme_rate = float(light_cfg.get("extreme_light_sampling_rate", 0.02))

    is_extreme = rng.random() < extreme_rate
    if is_extreme:
        ep = light_cfg.get("extreme_profile", {})
        intensity = rng.uniform(
            ep.get("intensity_range", [6000, 18000])[0],
            ep.get("intensity_range", [6000, 18000])[1],
        )
        color = [
            rng.uniform(ep.get("color_min", [0.35]*3)[i], ep.get("color_max", [1.0]*3)[i])
            for i in range(3)
        ]
        yaw_deg = rng.uniform(*ep.get("direction_yaw_range_deg", [-180, 180]))
        pitch_deg = rng.uniform(*ep.get("direction_pitch_range_deg", [-85, -10]))
        angle_deg = rng.uniform(*ep.get("angle_range_deg", [0.2, 8.0]))
    else:
        intensity = base_intensity * rng.uniform(1.0 - intensity_jitter, 1.0 + intensity_jitter)
        color = [_clamp(base_color[i] + _sample_sym(color_jitter), 0.0, 1.0) for i in range(3)]
        yaw_deg = base_yaw + _sample_sym(yaw_jitter)
        pitch_deg = base_pitch + _sample_sym(pitch_jitter)
        angle_deg = max(0.0, base_angle + _sample_sym(angle_jitter))

    new_light = {
        "enabled": True,
        "follow_background_yaw": bool(light_cfg.get("follow_background_yaw", False)),
        "applied_background_yaw_deg": 0.0,
        "is_extreme": is_extreme,
        "intensity": float(intensity),
        "color": color,
        "direction_quat": _rpy_deg_to_quat(0.0, pitch_deg, yaw_deg),
        "angle_deg": float(angle_deg),
    }

    return {"background": new_bg, "table_surface": new_ts, "light": new_light}


def _read_gen_sample_and_config(hdf5_path: Path):
    with h5py.File(hdf5_path, "r") as f:
        raw_sample = f["meta/scene_generalization_sample"][()]
        raw_config = f["meta/scene_generalization_config"][()]
    sample = json.loads(raw_sample.decode() if isinstance(raw_sample, bytes) else raw_sample)
    config = json.loads(raw_config.decode() if isinstance(raw_config, bytes) else raw_config)
    return sample, config


def _write_gen_sample(hdf5_path: Path, new_sample: dict):
    with h5py.File(hdf5_path, "a") as f:
        key = "meta/scene_generalization_sample"
        if key in f:
            del f[key]
        new_bytes = json.dumps(new_sample, ensure_ascii=False).encode("utf-8")
        f.create_dataset(key, data=new_bytes)


# ── Main ───────────────────────────────────────────────────────────────────────

origin_files = sorted(ORIGIN_DIR.glob("episode_*.hdf5"))
if not origin_files:
    print(f"[ERROR] No episode_*.hdf5 files found in {ORIGIN_DIR}")
    sys.exit(1)

multiplier = args.multiplier
total_out = len(origin_files) * multiplier
print(f"[gen_rgb_augmented] {len(origin_files)} source episodes × {multiplier} = {total_out} output episodes")
print(f"[gen_rgb_augmented] {ORIGIN_DIR} -> {OUTPUT_DIR}")

ok = err = skip = 0
out_idx = 0  # global sequential index for episode_XXXXXX.hdf5 naming
for src_path in origin_files:
    # Pre-read gen config once per source file (shared across all copies)
    try:
        orig_sample, gen_config = _read_gen_sample_and_config(src_path)
    except Exception as exc:
        print(f"  [ERROR] Cannot read {src_path.name}: {exc}")
        err += multiplier
        out_idx += multiplier
        continue

    orig_appearance = orig_sample.get("appearance", {})

    for _copy_i in range(multiplier):
        dst_name = f"episode_{out_idx:06d}.hdf5"
        dst_path = OUTPUT_DIR / dst_name
        out_idx += 1

        if dst_path.exists() and not args.overwrite:
            print(f"  [skip] {dst_name} (already exists, use --overwrite to replace)")
            skip += 1
            continue

        try:
            # Sample new appearance (background / table surface / light only)
            new_appearance = _sample_new_appearance(gen_config, orig_appearance)

            # Build new sample: keep everything except appearance
            new_sample = dict(orig_sample)
            new_sample["appearance"] = new_appearance
            # Mark as restored so replay uses these exact settings
            new_sample["_replay_generalization_mode"] = "restored"

            # Copy HDF5 and patch the generalization sample
            shutil.copy2(src_path, dst_path)
            _write_gen_sample(dst_path, new_sample)

            bg_uri = new_appearance["background"].get("asset_uri", "?")
            ts_name = new_appearance["table_surface"].get("asset_name", "?")
            print(f"  [OK] {src_path.name} -> {dst_name}  bg={Path(bg_uri).name}  table={Path(ts_name).stem}")
            ok += 1
        except Exception as exc:
            print(f"  [ERROR] {src_path.name} copy {_copy_i}: {exc}")
            import traceback; traceback.print_exc()
            err += 1

print(f"\n[done] OK={ok}  skip={skip}  error={err}  (total output: {out_idx})")
print(f"Now replay with:\n  python tools/replay/batch_replay.py \\\n    --origin-dir {OUTPUT_DIR} \\\n    --replay-dir <your_replay_generate_dir>")
