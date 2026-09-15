"""Run a trained tactile GR00T N1.5 policy in Isaac Sim for evaluation.

This standalone runner mirrors ``run_policy.py`` while adding real-time TacMap
capture. It consumes four RGB cameras plus checkpoint-defined tactile sites and
does not modify the standard GR00T/ACT/DP evaluation paths.

Usage (manus env):
    cd .
    python tactile_run_policy.py \
        --task scenes/06_fruit_bowl_loading.yaml \
        --ckpt-dir outputs/logs/tactile/task06-sharpa \
        --enable-rgb

    # headless mode:
    python tactile_run_policy.py \
        --task scenes/06_fruit_bowl_loading.yaml \
        --ckpt-dir outputs/logs/tactile/task06-sharpa \
        --enable-rgb --headless

Notes:
    - --enable-rgb enables RGB rendering (required for policy inference).
    - Default cameras (stereo 鍙岀洰 + wrist 鍙岃厱 only):
        cam_wrist_right, cam_wrist_left, cam_stereo_left, cam_stereo_right.
      cam_overhead and cam_chest are excluded by default regardless of policy type.
    - TacMap inference requires a CUDA simulation device.
    - Camera and tactile channel order comes from the checkpoint metadata.
    - Each episode runs for --episode-steps steps, then resets and loops forever.
"""

from __future__ import annotations

import argparse
import copy
import logging
import os
import sys
import threading
from collections import deque

import numpy as np

# Pre-import pinocchio before Isaac Sim
try:
    import pinocchio  # noqa: F401
except ImportError:
    pass

from isaaclab.app import AppLauncher
from utils.isaac_rendering import configure_headless_camera_parity_experience
from utils.logging_config import add_logging_arguments, configure_logging


logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(
    description="Run a four-RGB + checkpoint-defined-TacMap tactile GR00T policy in Isaac Sim."
)
parser.add_argument("--task", type=str, required=True,
                    help="Path to task YAML (e.g. scenes/06_fruit_bowl_loading.yaml).")
parser.add_argument("--policy-type", type=str, default="GR00T_TACTILE",
                    choices=["GR00T_TACTILE", "gr00t_tactile"],
                    help="Standalone policy backend (fixed to GR00T_TACTILE).")
parser.add_argument("--policy-name", type=str, default=None,
                    help="Policy name used in output summaries. Default: derived from ckpt-dir or 'act'.")
parser.add_argument("--policy-display-name", type=str, default=None,
                    help="Display name for policy in eval summaries (e.g. ckpt-relative path). "
                         "Takes priority over --policy-name for summary metadata only.")
parser.add_argument("--ckpt-dir", type=str, default=None,
                    help="Compatibility argument; the tactile GR00T model is served remotely.")
parser.add_argument("--host", type=str, default="127.0.0.1",
                    help="Tactile GR00T policy-server host.")
parser.add_argument("--port", type=int, default=9000,
                    help="Tactile GR00T policy-server port.")
parser.add_argument("--record-dir", type=str, default=None,
                    help="Directory to save HDF5 snapshots of successful episodes (cleared at startup, low-bitrate JPG).")
parser.add_argument("--record-all", action="store_true", default=False,
                    help="Save ALL episodes (not just successes). Failures go to failure/ subdirectory.")
parser.add_argument("--ckpt-name", type=str, default="policy_best.ckpt",
                    help="Checkpoint filename inside --ckpt-dir.")
parser.add_argument("--chunk-size", type=int, default=16,
                    help="Maximum GR00T action chunk requested from the remote server (default: 16).")
parser.add_argument("--state-dim", type=int, default=36,
                    help="Robot joint dimension. Auto-overridden from --robot-key and --active-dof when available.")
parser.add_argument("--robot-key", type=str, default=None,
                    help="Robot registry key (e.g. multi_ur5_schunk_hand_with_flange). "
                         "Auto-detected from ckpt_dir parent directory if not specified. "
                         "Falls back to the first robot subdirectory under ckpt_dir.")
parser.add_argument("--active-dof", action=argparse.BooleanOptionalAction, default=True,
                    help="Use active-only qpos/action dimensions and expand actions through mimic rules (default: enabled).")
parser.add_argument("--episode-steps", type=int, default=None,
                    help="Explicit policy-step override; by default use scene expert timing, "
                         "falling back to 800 policy steps.")
parser.add_argument("--warmup-steps", type=int, default=60,
                    help="Steps to home robot before querying policy.")
parser.add_argument("--temporal-agg", action="store_true",
                    help="Enable ACT temporal aggregation.")
parser.add_argument("--temporal-agg-k", type=float, default=0.1,
                    help="Temporal aggregation decay factor k. Higher=less smoothing. Default 0.1 (medium).")
parser.add_argument("--seed", type=int, default=100000000,
                    help="Evaluation seed namespace base. Task and episode offsets are added deterministically.")
parser.add_argument("--task-seed-id", type=int, default=None,
                    help="Numeric task id for seed partitioning when the task filename has no numeric prefix.")
parser.add_argument("--collect-config", type=str, default=None,
                    help="Collect config YAML. Defaults to configs/collect/default.yaml.")
parser.add_argument("--enable-rgb", action="store_true",
                    help="Enable RGB capture (required for policy inference).")
parser.add_argument("--tactile-resolution-step", type=int, default=None,
                    help="TacMap sampling step; defaults to checkpoint metadata and must match it.")
parser.add_argument("--tactile-max-distance", type=float, default=None,
                    help="TacMap ray distance; defaults to checkpoint metadata and must match it.")
parser.add_argument("--enable-generalization", action="store_true",
                    help="Enable scene generalization (object/background/lighting randomization).")
parser.add_argument("--generalization-config", type=str, default=None,
                    help="Path to scene generalization YAML. Defaults to configs/scene/generalization.yaml.")
parser.add_argument("--generalization-profile", type=str, default="none",
                    choices=[
                        "none",
                        "cov_only", "cov-only", "cov",
                        "inv_only", "inv-only", "inv",
                        "inv_cov", "inv+cov", "full",
                    ],
                    help="Evaluation protocol slice. cov_only enables trajectory-covariant axes only; "
                         "inv_only enables invariant visual/context axes only; "
                         "inv_cov enables invariant axes plus covariant axes.")
parser.add_argument("--generalization-split", type=str, default=None,
                    choices=["seen", "unseen", "all"],
                    help="Asset/range split to apply when --generalization-profile is set. Defaults to the YAML asset_split.")
parser.add_argument("--anchor-hdf5", type=str, default=None,
                    help="HDF5 episode to anchor generalization for none/cov_only/inv_only profiles. "
                         "Axes not resampled will use this episode's exact generalization sample.")
parser.add_argument("--anchor-dir", type=str, default=None,
                    help="Directory of replay HDF5 episodes. "
                         "Selects main 0-24 + _1 25-49 for full 50-background coverage "
                         "(main and _1 are disjoint background sets); "
                         "falls back to all main episodes when _1 variants are absent. "
                         "Each inference episode cycles through anchors so generalization samples "
                         "are not tied to a single episode. "
                         "Overrides --anchor-hdf5 when both are given.")
parser.add_argument("--early-stop", action=argparse.BooleanOptionalAction, default=True,
                    help="Enable early stopping on stable success (default: enabled; use --no-early-stop for fixed-horizon runs).")
parser.add_argument("--output-dir", type=str, default=None,
                    help="Directory to write benchmark results (per_episode.jsonl, per_task.json, summary.json).")
parser.add_argument("--append-output", action="store_true",
                    help="Append to an existing --output-dir and include prior per_episode.jsonl rows in summaries.")
parser.add_argument("--num-episodes", type=int, default=0,
                    help="Number of episodes to run (0 = infinite loop).")
parser.add_argument("--start-episode", type=int, default=0,
                    help="Skip to episode N (seeds are computed from --seed as if starting from 0).")
parser.add_argument("--append-record-dir", action="store_true",
                    help="Do not clear existing --record-dir HDF5 files at startup.")
parser.add_argument("--realtime", action="store_true",
                    help="Throttle simulation to real-time pace (sleep per step). "
                         "Without this flag the sim runs at maximum speed (default).")
parser.add_argument("--render-every-physics-step", action="store_true",
                    help="Compatibility/debug mode: render at 60Hz after every physics step. "
                         "By default, render only at the 20Hz camera/policy sampling rate; "
                         "physics, control, and metrics remain at 60Hz.")
parser.add_argument("--dump-tactile-attention", action="store_true",
                    help="Log tactile attention-pooling weights (query over finger sites) "
                         "to <output-dir>/tactile_attention.jsonl for post-hoc visualization.")
AppLauncher.add_app_launcher_args(parser)
add_logging_arguments(parser)
args_cli = parser.parse_args()
configure_logging(args_cli.log_level)

# Camera stack must be on for RGB rendering
args_cli.enable_cameras = True
configure_headless_camera_parity_experience(args_cli, log_prefix="tactile_run_policy")

logger.info("Launching AppLauncher")
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
logger.info("AppLauncher ready")

# Post-launch imports
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.utils.io import load_yaml  # noqa: E402

SCRIPT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from policy.GR00T_n15_Tactile.tactile_eval_budget import (  # noqa: E402
    resolve_episode_budget,
)

RUN_POLICY_SEED_POLICY = "episode_seed = base_seed + episode_index - 1"
INV_COV_EVAL_SEED = 200_000_000  # fixed seed for the inv_cov channel to decouple from anchor channels

_PROFILE_NONE = {"", "none"}
_PROFILE_COV_ONLY = {"cov_only", "cov-only", "cov"}
_PROFILE_INV_ONLY = {"inv_only", "inv-only", "inv"}
_PROFILE_INV_COV = {"inv_cov", "inv+cov", "full"}

def _normalize_generalization_profile(profile: str | None) -> str:
    key = str(profile or "none").strip().lower()
    if key in _PROFILE_NONE:
        return "none"
    if key in _PROFILE_COV_ONLY:
        return "cov_only"
    if key in _PROFILE_INV_ONLY:
        return "inv_only"
    if key in _PROFILE_INV_COV:
        return "inv_cov"
    raise ValueError(f"Unsupported generalization profile: {profile!r}")


def _mapping(raw: dict, *path: str) -> dict:
    cur = raw
    for key in path:
        nxt = cur.get(key)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[key] = nxt
        cur = nxt
    return cur


def _normalize_generalization_split(split: str | None, *, default: str = "seen") -> str:
    value = str(split or default).strip().lower()
    if value not in {"seen", "unseen", "all"}:
        raise ValueError(f"Unsupported generalization split: {split!r}")
    return value


def _apply_generalization_split(raw: dict, *, split: str | None) -> dict:
    """Apply an explicit split to every asset-backed generalization axis."""
    cfg = copy.deepcopy(raw or {})
    if split is None:
        return cfg

    split = _normalize_generalization_split(split)
    cfg["asset_split"] = split
    background = _mapping(cfg, "appearance", "background")
    table_surface = _mapping(cfg, "appearance", "table_surface")
    clutter = _mapping(cfg, "clutter", "tabletop")
    background["asset_split"] = split
    table_surface["asset_split"] = split
    clutter["asset_split"] = split
    return cfg


def _apply_generalization_profile(raw: dict, *, profile: str) -> dict:
    """Return a config copy filtered to a benchmark protocol slice.

    - none:     all axes OFF (anchor HDF5 provides everything)
    - cov_only: covariant axes ON (unseen) — object_pose, table_height;
                invariant axes OFF (anchor)
    - inv_only: invariant axes ON (unseen) — background, table_surface,
                light, clutter, camera; covariant axes OFF (anchor)
    - inv_cov:  all axes ON (unseen)
    """
    profile = _normalize_generalization_profile(profile)
    cfg = copy.deepcopy(raw or {})

    background = _mapping(cfg, "appearance", "background")
    table_surface = _mapping(cfg, "appearance", "table_surface")
    light = _mapping(cfg, "appearance", "light")
    object_pose = _mapping(cfg, "spatial", "object_pose")
    table_height = _mapping(cfg, "spatial", "table_height")
    camera = _mapping(cfg, "spatial", "camera")
    clutter = _mapping(cfg, "clutter", "tabletop")

    inv_axes = [background, table_surface, light, clutter, camera]
    cov_axes = [object_pose, table_height]
    all_axes = inv_axes + cov_axes

    if profile == "inv_cov":
        cfg["asset_split"] = "unseen"
        for m in all_axes:
            m["enabled"] = True
        for m in inv_axes:
            m["asset_split"] = "unseen"
        object_pose["position_jitter_cm"] = 2.0
        object_pose["yaw_jitter_deg"] = 10.0
        table_height["offset_m_range"] = [-0.05, 0.05]
        return cfg

    # none / cov_only / inv_only: start with all axes OFF (anchor provides defaults)
    for m in all_axes:
        m["enabled"] = False

    if profile == "cov_only":
        cfg["asset_split"] = "unseen"
        for m in cov_axes:
            m["enabled"] = True
        object_pose["position_jitter_cm"] = 2.0
        object_pose["yaw_jitter_deg"] = 10.0
        table_height["offset_m_range"] = [-0.05, 0.05]

    elif profile == "inv_only":
        cfg["asset_split"] = "unseen"
        for m in inv_axes:
            m["enabled"] = True
            m["asset_split"] = "unseen"

    # profile == "none": all OFF, fully determined by anchor
    return cfg


def _seed_everything(seed: int) -> None:
    seed_everything(seed)


def _episode_seed(base_seed: int, episode_idx: int) -> int:
    return episode_seed(int(base_seed), int(episode_idx))


def _robustness_category(
    sample,
    *,
    generalization_enabled: bool,
    generalization_profile: str,
) -> str:
    """Return robustness protocol bucket: none, cov, inv, or inv+cov."""
    profile = _normalize_generalization_profile(generalization_profile)
    if profile == "none":
        return "none"
    if not generalization_enabled or sample is None:
        axes = active_perturbation_axes(sample, generalization_enabled=generalization_enabled)
        if axes == "none":
            return "none"
        axis_set = {axis.strip() for axis in axes.split(",") if axis.strip()}
        invariant_axes = {"background", "table_surface", "light", "clutter", "camera"}
        covariant_axes = {"object_pose", "table_height"}
        has_inv = bool(axis_set & invariant_axes)
        has_cov = bool(axis_set & covariant_axes)
        if has_inv and has_cov:
            return "inv+cov"
        if has_inv:
            return "inv"
        if has_cov:
            return "cov"
        return "inv+cov"
    if profile == "cov_only":
        return "cov"
    if profile == "inv_only":
        return "inv"
    return "inv+cov"


from build import (  # noqa: E402
    build_scene,
    active_perturbation_axes,
    load_scene_generalization_config,
    merge_scene_generalization_overrides,
    parse_scene_generalization_config,
    relativize_sample_paths,
    collect_task_asset_codes,
    dict_to_generalization_sample,
    merge_generalization_samples,
    sample_scene_generalization,
    scene_generalization_sample_debug_lines,
    scene_generalization_sample_to_dict,
    validate_resolved_object_placement_keys,
)
from collector import load_collect_config  # noqa: E402
from collector.cameras import CameraRig  # noqa: E402
from collector.tacmap_rig import (  # noqa: E402
    TacMapRig,
    build_tacmap_raycast_targets,
    tacmap_supported,
)
from collector.state_reader import read_joint_limits, read_joint_state, read_object_states  # noqa: E402
from policy.GR00T_n15_Tactile.tactile_deployment import (  # noqa: E402
    GR00TTactileDeployment,
    RemoteGR00TTactileDeployment,
)
from utils.episode_runtime import (  # noqa: E402
    clear_scene_prims_preserve_robot,
    initialize_scene_runtime_state,
    move_controlled_articulations_home,
)
from utils.policy_timing import should_query_policy  # noqa: E402
from robots.active_dof_utils import get_active_dof_info, get_active_dof_info_for_runtime, select_active, expand_to_full  # noqa: E402

_active_dof_info = None  # set once at init when --active-dof is enabled
from utils.runtime_helpers import (  # noqa: E402
    classify_interactive_objects,
    write_articulation_targets,
    update_sim_objects,
)
from utils.seed_policy import (  # noqa: E402
    episode_seed,
    resolve_task_seed_id,
    seed_everything,
    task_base_seed,
)
print("[tactile_run_policy] All imports OK.", flush=True)

# Camera mapping: collect cam_id -> ACT obs key.
# _build_policy_obs builds obs for all available cameras; ACT.get_action()
# reads only the cameras matching its training camera_names.
_CAM_TO_OBS: dict[str, str] = {
    "cam_overhead":    "head_cam",
    "cam_wrist_right": "right_cam",
    "cam_wrist_left":  "left_cam",
    "cam_stereo_left": "stereo_left_cam",
    "cam_stereo_right": "stereo_right_cam",
}

# Tactile checkpoint camera name -> CameraRig id.
_TACTILE_CAMERA_TO_RIG: dict[str, str] = {
    "cam_right_wrist": "cam_wrist_right",
    "cam_left_wrist": "cam_wrist_left",
    "cam_stereo_left": "cam_stereo_left",
    "cam_stereo_right": "cam_stereo_right",
}

# cam_chest is excluded because it's not used by any policy.
_DEFAULT_INFERENCE_CAMERAS: set[str] = {
    "cam_wrist_right",
    "cam_wrist_left",
    "cam_stereo_left",
    "cam_stereo_right",
}
_IMG_H, _IMG_W = 480, 640


def _validate_camera_config(camera_cfgs, policy: GR00TTactileDeployment) -> None:
    unsupported = [
        name for name in policy.camera_names if name not in _TACTILE_CAMERA_TO_RIG
    ]
    if unsupported:
        raise ValueError(f"unsupported checkpoint camera_names: {unsupported}")
    configured_list = [cfg.camera_id for cfg in camera_cfgs]
    if len(set(configured_list)) != len(configured_list):
        raise ValueError(f"collect config contains duplicate RGB cameras: {configured_list}")
    configured = set(configured_list)
    required = {_TACTILE_CAMERA_TO_RIG[name] for name in policy.camera_names}
    missing = sorted(required - configured)
    if missing:
        raise ValueError(f"collect config is missing required RGB cameras: {missing}")


def _format_mib(num_bytes: int | float | None) -> str:
    if num_bytes is None:
        return "n/a"
    return f"{float(num_bytes) / (1024.0 * 1024.0):.1f}MiB"


def _resolve_robot_key_from_ckpt_dir(ckpt_dir: str) -> str | None:
    """Auto-detect robot key from ckpt_dir structure.

    Checks (in order):
      1. Parent directory name (e.g. .../multi_ur5_schunk_hand_with_flange/act_active)
      2. Grandparent directory name (e.g. .../44/multi_ur5_schunk_hand_with_flange/act_active)
      3. First valid robot subdirectory under ckpt_dir

    Returns None if no valid robot key is found.
    """
    from robots import ROBOT_SPAWNERS
    ckpt_path = os.path.normpath(str(ckpt_dir)).rstrip("/")
    for _ in range(2):
        candidate = os.path.basename(ckpt_path)
        if candidate in ROBOT_SPAWNERS:
            return candidate
        ckpt_path = os.path.dirname(ckpt_path)
    import glob
    for entry in sorted(glob.glob(os.path.join(ckpt_dir, "*"))):
        name = os.path.basename(entry)
        if name in ROBOT_SPAWNERS and os.path.isdir(entry):
            return name
    return None


def _process_rss_bytes() -> int | None:
    """Return current process RSS from /proc without adding a psutil dependency."""
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) * 1024
    except Exception:
        return None
    return None


def _log_memory(label: str, device: str | None = None) -> None:
    rss = _process_rss_bytes()
    msg = f"[mem] {label}: rss={_format_mib(rss)}"

    if torch.cuda.is_available():
        try:
            cuda_device = torch.device(device or "cuda")
            if cuda_device.type != "cuda":
                cuda_device = torch.device("cuda")
            try:
                torch.cuda.synchronize(cuda_device)
            except Exception:
                pass
            free_bytes, total_bytes = torch.cuda.mem_get_info(cuda_device)
            reserved = torch.cuda.memory_reserved(cuda_device)
            allocated = torch.cuda.memory_allocated(cuda_device)
            msg += (
                f" cuda_free={_format_mib(free_bytes)}"
                f" cuda_total={_format_mib(total_bytes)}"
                f" torch_reserved={_format_mib(reserved)}"
                f" torch_allocated={_format_mib(allocated)}"
            )
        except Exception as exc:
            msg += f" cuda_query_failed={exc}"
    else:
        msg += " cuda=unavailable"

    print(msg, flush=True)


def _compact_episode_result(result):
    """Drop large per-step payloads from the in-memory result list."""
    from dataclasses import replace

    return replace(result, timeseries=None)


def _require_tactile_gpu(device: str) -> None:
    """TacMap ray casting is only supported by the CUDA simulation pipeline."""

    if not str(device).strip().lower().startswith("cuda"):
        raise RuntimeError(
            "tactile_run_policy.py requires a CUDA simulation device for TacMap; "
            f"got --device={device!r}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "tactile_run_policy.py requires CUDA, but torch.cuda.is_available() is false"
        )


def _load_tactile_policy(args: argparse.Namespace) -> RemoteGR00TTactileDeployment:
    robot_key = args.robot_key or (
        _resolve_robot_key_from_ckpt_dir(args.ckpt_dir) if args.ckpt_dir else None
    )
    if not robot_key:
        raise ValueError("--robot-key is required for tactile GR00T evaluation")
    task_path = args.task if os.path.isabs(args.task) else os.path.join(SCRIPT_DIR, args.task)
    task_cfg = load_yaml(task_path)
    prompt = str(task_cfg.get("description") or f"perform task: {os.path.basename(args.task)}")
    model = RemoteGR00TTactileDeployment(
        args.host,
        args.port,
        robot_key=robot_key,
        prompt=prompt,
        action_horizon=args.chunk_size,
        tactile_resolution_step=args.tactile_resolution_step or 1,
        tactile_max_distance=args.tactile_max_distance or 0.015,
    )
    print(f"[policy] Connected to tactile GR00T server at {args.host}:{args.port}", flush=True)
    print(
        f"[policy] Inputs: cameras={list(model.camera_names)} "
        f"tactile_sites={list(model.site_names)} state_dim={model.state_dim} "
        f"chunk_size={model.chunk_size}",
        flush=True,
    )
    return model


def _load_act_policy(args: argparse.Namespace):
    """Load ACT model from checkpoint directory."""
    act_dir = os.path.join(SCRIPT_DIR, "policy", "ACT")
    if act_dir not in sys.path:
        sys.path.insert(0, act_dir)

    from act_policy import ACT  # type: ignore  # noqa: E402

    ckpt_dir = args.ckpt_dir
    ckpt_path = os.path.join(ckpt_dir, args.ckpt_name)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint '{args.ckpt_name}' not found in '{ckpt_dir}'")

    act_args = {
        "kl_weight": 10,
        "chunk_size": args.chunk_size,
        "state_dim": args.state_dim,
        "hidden_dim": 512,
        "dim_feedforward": 3200,
        "lr": 1e-5,
        "lr_backbone": 1e-5,
        "backbone": "resnet18",
        "enc_layers": 4,
        "dec_layers": 7,
        "nheads": 8,
        "dropout": 0.1,
        "pre_norm": False,
        "camera_names": ["cam_right_wrist", "cam_left_wrist",
                          "cam_stereo_left", "cam_stereo_right"],
        "temporal_agg": args.temporal_agg,
        "temporal_agg_k": args.temporal_agg_k,
        "device": args.device,
        "ckpt_dir": ckpt_dir,
    }

    # DETR's build_ACT_model_and_optimizer calls parser.parse_args() which reads
    # sys.argv.  Isaac Sim's args are not valid DETR args, so we temporarily
    # replace sys.argv with the minimal required DETR arguments.
    _saved_argv = sys.argv
    sys.argv = [
        "run_policy.py",
        "--ckpt_dir", ckpt_dir,
        "--policy_class", "ACT",
        "--task_name", "deploy",
        "--seed", "0",
        "--num_epochs", "1",
        "--state_dim", str(args.state_dim),
    ]
    try:
        model = ACT(act_args, None)
    finally:
        sys.argv = _saved_argv

    # Override with the exact requested checkpoint
    state_dict = torch.load(ckpt_path, map_location=args.device)
    model.policy.load_state_dict(state_dict)
    model.policy.eval()
    print(f"[policy] Loaded ACT weights from {ckpt_path}", flush=True)
    return model


def _load_dp_policy(args: argparse.Namespace):
    """Load DP model from checkpoint and training config."""
    dp_dir = os.path.join(SCRIPT_DIR, "policy", "DP")
    if dp_dir not in sys.path:
        sys.path.insert(0, dp_dir)

    from dp_model import DP  # type: ignore  # noqa: E402
    import yaml

    ckpt_path = os.path.join(args.ckpt_dir, args.ckpt_name)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint '{args.ckpt_name}' not found in '{args.ckpt_dir}'")

    config_path = args.training_config or os.path.join(
        "policy", "DP", "diffusion_policy", "config", "robot_dp_36_dex2scene.yaml"
    )
    if not os.path.isabs(config_path):
        config_path = os.path.join(SCRIPT_DIR, config_path)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"DP training config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model = DP(
        ckpt_path,
        n_obs_steps=cfg["n_obs_steps"],
        n_action_steps=cfg["n_action_steps"],
        device=args.device,
    )
    print(f"[policy] Loaded DP weights from {ckpt_path}", flush=True)
    print(f"[policy] Loaded DP training config from {config_path}", flush=True)
    return model


def _load_remote_policy(args: argparse.Namespace):
    client = RemotePolicyClient(args.remote_host, args.remote_port)
    print(f"[policy] Connected to remote policy server at {args.remote_host}:{args.remote_port}", flush=True)
    return client


def _load_policy(args: argparse.Namespace):
    if args.policy_type.upper() != "GR00T_TACTILE":
        raise ValueError(f"Unsupported policy type: {args.policy_type}")
    return _load_tactile_policy(args)


def _preprocess_image(rgb_hwc: np.ndarray, *, scale_to_unit: bool = False) -> np.ndarray:
    """HWC uint8 -> CHW float32, resized to _IMG_H x _IMG_W."""
    import cv2
    if rgb_hwc is None or rgb_hwc.ndim != 3 or rgb_hwc.shape[0] == 0 or rgb_hwc.shape[1] == 0:
        raise ValueError(f"Invalid RGB image shape: {None if rgb_hwc is None else rgb_hwc.shape}")
    img = cv2.resize(rgb_hwc, (_IMG_W, _IMG_H), interpolation=cv2.INTER_LINEAR)
    chw = np.moveaxis(img.astype(np.float32), -1, 0)
    if scale_to_unit:
        chw = chw / 255.0
    return chw




def _policy_qpos(robot_art, robot_state: dict | None = None) -> np.ndarray:
    """Return policy qpos, reusing the post-physics metric read when present."""
    if robot_state is not None and robot_state.get("qpos") is not None:
        qpos = np.asarray(robot_state["qpos"], dtype=np.float32)
    else:
        qpos = robot_art.data.joint_pos[0].cpu().numpy().astype(np.float32)
    if _active_dof_info is not None:
        qpos = select_active(qpos, _active_dof_info)
    return qpos


def _joint_command_context(robot_art, raw_action: np.ndarray, full_target: np.ndarray) -> dict:
    """Build a sparse-diagnostic snapshot of the command sent to Isaac.

    This is intentionally only serialized if MetricTracker observes the first
    physical hard-limit violation.  It adds no per-step output payload.
    """
    runtime_names = list(getattr(getattr(robot_art, "data", None), "joint_names", None) or robot_art.joint_names)
    raw = np.asarray(raw_action, dtype=np.float32).reshape(-1)
    full = np.asarray(full_target, dtype=np.float32).reshape(-1)
    active_names: list[str]
    mimic_source_by_joint: dict[str, str] = {}
    if _active_dof_info is not None and raw.size == _active_dof_info.active_dof:
        active_names = list(_active_dof_info.active_joint_names)
        for mimic_idx, source_idx, _multiplier, _offset in _active_dof_info.mimic_rules:
            if mimic_idx < len(runtime_names) and source_idx < len(runtime_names):
                mimic_source_by_joint[runtime_names[mimic_idx]] = runtime_names[source_idx]
    else:
        active_names = list(runtime_names) if raw.size == len(runtime_names) else []
    return {
        "active_joint_names": active_names,
        "raw_active_action": raw,
        "full_joint_names": runtime_names,
        "expanded_full_target": full,
        # No command shield exists yet.  Keeping this explicit prevents a
        # diagnostic reader from mistaking the raw target for a clipped one.
        "shielded_full_target": full,
        "shield_applied": False,
        "mimic_source_by_joint": mimic_source_by_joint,
    }


def _build_policy_obs(
    frames: dict,
    robot_art,
    policy_type: str = "ACT",
    *,
    robot_state: dict | None = None,
) -> dict | None:
    """Build obs dict for ACT/DP policy inference.

    Missing camera frames are replaced with a black (zero) image so the policy
    can still run even when auxiliary cameras (chest / stereo) fail to render.
    The model only actively uses the cameras matching its training camera_names,
    so black frames for unused cameras are harmless.
    Returns None only when the robot joint state is unavailable.
    """
    policy_type_upper = policy_type.upper()
    scale_to_unit = policy_type_upper == "DP"
    qpos = _policy_qpos(robot_art, robot_state)
    obs: dict = {"qpos": qpos, "agent_pos": qpos}
    _missing: list[str] = []
    for cam_id, obs_key in _CAM_TO_OBS.items():
        frame = frames.get(cam_id)
        if (
            frame is None
            or frame.rgb is None
            or frame.rgb.ndim != 3
            or frame.rgb.shape[0] == 0
            or frame.rgb.shape[1] == 0
        ):
            _missing.append(cam_id)
            # Use black frame so policy can still run
            img = np.zeros((_IMG_H, _IMG_W, 3), dtype=np.uint8)
        else:
            img = frame.rgb  # HWC uint8
        obs[obs_key] = _preprocess_image(img, scale_to_unit=scale_to_unit)
    return obs


def _validate_tactile_rig(policy: GR00TTactileDeployment, tactile_rig: TacMapRig) -> None:
    runtime_sites = tuple(str(name) for name in tactile_rig.site_names)
    if len(set(runtime_sites)) != len(runtime_sites):
        raise ValueError(f"TacMapRig site_names contains duplicates: {runtime_sites}")
    missing = [name for name in policy.site_names if name not in runtime_sites]
    extra = [name for name in runtime_sites if name not in policy.site_names]
    if missing or extra:
        raise ValueError(
            "TacMapRig sites do not match checkpoint site_names: "
            f"missing={missing}, extra={extra}"
        )
    if tactile_rig.resolution_step != policy.tactile_resolution_step:
        raise ValueError("TacMapRig resolution_step does not match checkpoint")
    if tactile_rig.image_size != policy.tactile_image_size:
        raise ValueError("TacMapRig image_size does not match checkpoint")
    if not np.isclose(
        tactile_rig.max_distance,
        policy.tactile_max_distance_m,
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError("TacMapRig max_distance does not match checkpoint")


def _close_sensor_rigs(camera_rig, tactile_rig) -> None:
    """Best-effort deterministic cleanup for all episode setup/rollout paths."""

    if tactile_rig is not None:
        try:
            tactile_rig.close()
        except Exception as exc:
            print(f"  [WARN] Failed to close TacMapRig: {exc}", flush=True)
    if camera_rig is not None:
        try:
            camera_rig.close()
        except Exception as exc:
            print(f"  [WARN] Failed to close CameraRig: {exc}", flush=True)


class _EpisodeSensorOwner:
    """Own both sensor rigs so one outer finally covers their full lifetime."""

    def __init__(self) -> None:
        self.camera_rig = None
        self.tactile_rig = None
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        _close_sensor_rigs(self.camera_rig, self.tactile_rig)


def _build_tactile_policy_obs(
    frames: dict,
    tactile_capture: dict,
    robot_art,
    policy: GR00TTactileDeployment,
    *,
    robot_state: dict | None = None,
) -> dict:
    """Build the checkpoint-ordered four-RGB plus dynamic-TacMap online observation."""

    images: dict[str, np.ndarray] = {}
    for camera_name in policy.camera_names:
        camera_id = _TACTILE_CAMERA_TO_RIG.get(camera_name)
        if camera_id is None:
            raise ValueError(f"unsupported tactile checkpoint camera name: {camera_name}")
        frame = frames.get(camera_id)
        issue = _camera_frame_issue(frame)
        if issue is not None:
            raise RuntimeError(f"camera '{camera_id}' unavailable: {issue}")
        images[camera_name] = frame.rgb

    if not isinstance(tactile_capture, dict) or not isinstance(
        tactile_capture.get("tacmap"), dict
    ):
        raise RuntimeError("TacMapRig capture did not return a tactile 'tacmap' mapping")
    return {
        "qpos": _policy_qpos(robot_art, robot_state),
        "images": images,
        "tactile": tactile_capture["tacmap"],
    }


def _camera_frame_issue(frame) -> str | None:
    if frame is None:
        return "missing frame"
    rgb = getattr(frame, "rgb", None)
    if rgb is None:
        return "missing RGB"
    if not isinstance(rgb, np.ndarray):
        return f"RGB type={type(rgb).__name__}"
    if rgb.ndim != 3 or rgb.shape[0] == 0 or rgb.shape[1] == 0 or rgb.shape[2] < 3:
        return f"invalid RGB shape={rgb.shape}"
    return None


def _capture_complete_camera_set(camera_rig, sim, physics_dt: float, *, attempts: int = 5) -> dict:
    required_ids = tuple(camera_rig.camera_ids)
    last_issues: dict[str, str] = {}
    for attempt in range(1, attempts + 1):
        frames = camera_rig.capture(dt=physics_dt)
        last_issues = {
            camera_id: issue
            for camera_id in required_ids
            if (issue := _camera_frame_issue(frames.get(camera_id))) is not None
        }
        if not last_issues:
            return frames
        print(
            f"[WARN] Camera capture incomplete (attempt {attempt}/{attempts}): {last_issues}; "
            f"capture_errors={camera_rig.last_capture_errors}",
            flush=True,
        )
        sim.render()
    raise RuntimeError(
        f"Camera preflight failed after {attempts} attempts: {last_issues}; "
        f"capture_errors={camera_rig.last_capture_errors}"
    )


def _render_camera_sample(sim, camera_rig) -> None:
    """Align mounted cameras and render the exact state about to be observed.

    Camera observations are consumed at ``POLICY_STRIDE`` (20Hz), whereas
    physics advances at 60Hz.  Syncing immediately before this render keeps
    wrist RGB/extrinsics aligned while avoiding two unused renders between
    policy observations.
    """
    camera_rig.sync_mounted_camera_poses()
    sim.render()


def _step_sim_with_mounted_camera_sync(
    sim,
    camera_rig,
    interactive_objects,
    physics_dt: float,
    *,
    render: bool = False,
) -> None:
    """Advance one 60Hz physics step, rendering only when explicitly requested.

    Normal policy evaluation renders at the 20Hz observation boundary via
    :func:`_render_camera_sample`.  ``render=True`` preserves the previous
    every-physics-step behavior for visual/debug regression comparisons.
    """
    sim.step(render=False)
    update_sim_objects(interactive_objects.values(), physics_dt)
    if render:
        _render_camera_sample(sim, camera_rig)


def _split_legacy_joint_action(qpos: np.ndarray) -> dict[str, object]:
    result = {
        "vector": qpos,
        "qpos": qpos,
        "left_arm": [],
        "left_gripper": 0.0,
        "right_arm": [],
        "right_gripper": 0.0,
    }
    if qpos.size == 14:
        result["left_arm"] = qpos[:6].tolist()
        result["left_gripper"] = float(qpos[6])
        result["right_arm"] = qpos[7:13].tolist()
        result["right_gripper"] = float(qpos[13])
    return result


def _build_remote_policy_obs(
    frames: dict,
    robot_art,
    *,
    instruction: str | None = None,
    robot_state: dict | None = None,
) -> dict | None:
    qpos = _policy_qpos(robot_art, robot_state)

    def _frame_rgb(camera_id: str, fallback_camera_id: str | None = None) -> np.ndarray:
        for candidate in (camera_id, fallback_camera_id):
            if not candidate:
                continue
            frame = frames.get(candidate)
            if frame is not None and frame.rgb is not None:
                return frame.rgb
        return np.zeros((_IMG_H, _IMG_W, 3), dtype=np.uint8)

    observation = {
        "observation": {
            "head_camera": {"rgb": _frame_rgb("cam_overhead")},
            "right_camera": {"rgb": _frame_rgb("cam_wrist_right")},
            "left_camera": {"rgb": _frame_rgb("cam_wrist_left")},
            "front_camera": {"rgb": _frame_rgb("cam_chest", "cam_overhead")},
            "cam_overhead": {"rgb": _frame_rgb("cam_overhead")},
            "cam_wrist_right": {"rgb": _frame_rgb("cam_wrist_right")},
            "cam_wrist_left": {"rgb": _frame_rgb("cam_wrist_left")},
            "cam_chest": {"rgb": _frame_rgb("cam_chest", "cam_overhead")},
            "cam_stereo_left": {"rgb": _frame_rgb("cam_stereo_left", "cam_overhead")},
            "cam_stereo_right": {"rgb": _frame_rgb("cam_stereo_right", "cam_overhead")},
        },
        "joint_action": _split_legacy_joint_action(qpos),
        "pointcloud": None,
        "language": instruction or "",
    }
    return observation


def _refresh_observation_joint_state(
    obs: dict,
    robot_art,
    *,
    robot_state: dict | None = None,
) -> dict:
    qpos = _policy_qpos(robot_art, robot_state)
    refreshed = dict(obs)
    if "agent_pos" in refreshed:
        refreshed["qpos"] = qpos
        refreshed["agent_pos"] = qpos
        return refreshed

    refreshed["joint_action"] = _split_legacy_joint_action(qpos)
    return refreshed


def _load_success_checker(task: dict):
    """Load check_success function from task YAML success_conditions."""
    conditions = task.get("success_conditions", [])
    for cond in conditions:
        if cond.get("type") == "custom":
            evaluator = cond.get("evaluator")
            if evaluator:
                import importlib
                mod = importlib.import_module(evaluator)
                return getattr(mod, "check_success", None)
    return None


def _get_object_states(interactive_objects: dict, object_ids: list[str] | None = None) -> dict:
    """Read complete task object state from simulation."""
    if object_ids is None:
        object_ids = sorted(obj_id for obj_id in interactive_objects.keys() if obj_id != "global_robot")
    return read_object_states(interactive_objects, object_ids)


def _metrics_spec_for_episode(metrics_spec: dict | None, runtime: dict) -> dict:
    """Return an episode-local metrics spec with runtime geometry injected.

    Scene YAML can contain nominal table_z values, but table-height
    generalization changes the actual table surface per episode.  The tracker
    uses table_z for drop detection and grasp lift proxies, so runtime table_z
    must be authoritative.
    """
    spec = copy.deepcopy(metrics_spec or {})
    runtime_table_z = runtime.get("table_z")
    if runtime_table_z is None:
        return spec

    table_z = float(runtime_table_z)
    safety = spec.setdefault("safety", {})
    if isinstance(safety, dict):
        safety["table_z"] = table_z

    grasp = spec.get("grasp")
    if isinstance(grasp, dict):
        grasp["table_z"] = table_z
    return spec


def _reset_policy(policy, *, seed: int | None = None) -> None:
    """Reset policy internal timestep/history and remote policy RNG."""
    if hasattr(policy, "reset"):
        policy.reset()
        return
    if hasattr(policy, "reset_model"):
        if seed is not None and getattr(policy, "uses_raw_observation", False):
            policy.reset_model(seed=seed)
        else:
            policy.reset_model()
        return
    if hasattr(policy, "reset_obs"):
        policy.reset_obs()
        return
    policy.t = 0
    if policy.temporal_agg:
        policy.all_time_actions = torch.zeros(
            [policy.max_timesteps, policy.max_timesteps + policy.num_queries, policy.state_dim],
        ).to(policy.device)


_skip_episode_event = threading.Event()


def _start_stdin_skip_listener() -> None:
    """Start a daemon thread that sets _skip_episode_event when user types '1'."""
    def _listener():
        try:
            for line in sys.stdin:
                if line.strip() == "1":
                    _skip_episode_event.set()
                    print("[INFO] Episode skip requested 鈥?finishing current step and moving to next episode.", flush=True)
        except Exception:
            pass

    t = threading.Thread(target=_listener, name="stdin-skip-listener", daemon=True)
    t.start()


def _log_tactile_attention(
    path: str,
    episode_idx: int,
    policy_step: int,
    weights: np.ndarray,
) -> None:
    """Append one attention-weight row (JSONL) for post-hoc visualization."""
    import json as _json
    row = {
        "episode": int(episode_idx),
        "step": int(policy_step),
        "weights": [float(value) for value in weights],
    }
    with open(path, "a") as handle:
        handle.write(_json.dumps(row) + "\n")


def _log_modality_attention(
    path: str,
    episode_idx: int,
    policy_step: int,
    proportions: np.ndarray,
) -> None:
    """Append one cross-attention modality-proportion row (JSONL)."""
    import json as _json
    row = {
        "episode": int(episode_idx),
        "step": int(policy_step),
        "modalities": {
            "latent": float(proportions[0]),
            "proprio": float(proportions[1]),
            "visual": float(proportions[2]),
            "tactile": float(proportions[3]),
        },
    }
    with open(path, "a") as handle:
        handle.write(_json.dumps(row) + "\n")


def _run_episode_impl(
    sim,
    runtime,
    camera_cfgs,
    camera_generalization_sample,
    policy,
    physics_dt,
    args,
    episode_idx,
    episode_seed: int,
    base_seed: int,
    scene_generalization_sample: dict | None,
    seed_policy: str = RUN_POLICY_SEED_POLICY,
    check_success_fn=None,
    metrics_spec=None,
    perturbation_axis=None,
    robot_key=None,
    task_instruction: str | None = None,
    _sensor_owner: _EpisodeSensorOwner | None = None,
):
    """Run one policy episode. Returns EpisodeResult or None if sim shut down."""
    from benchmark.metrics import EpisodeResult
    if _sensor_owner is None:
        raise ValueError("_run_episode_impl requires an episode sensor owner")
    interactive_objects = runtime["interactive_objects"]
    pre_step_hooks = list(
        (runtime.get("robot_runtime") or {}).get("pre_step_hooks", [])
    )
    camera_rig = None
    tactile_rig = None
    object_ids = sorted((runtime.get("asset_local_bbox") or {}).keys())
    if not object_ids:
        object_ids = sorted(obj_id for obj_id in interactive_objects.keys() if obj_id != "global_robot")

    # Initialize metric tracker if metrics spec is available
    metric_tracker = None
    no_early_stop = not getattr(args, 'early_stop', False)
    if metrics_spec:
        metrics_spec = _metrics_spec_for_episode(metrics_spec, runtime)
        try:
            from benchmark.metric_tracker import MetricTracker
            _table_height_offset = 0.0
            if scene_generalization_sample:
                _th_sample = scene_generalization_sample.get("spatial", {}).get("table_height", {})
                if _th_sample and _th_sample.get("enabled"):
                    _table_height_offset = float(_th_sample.get("height_offset_m", 0.0) or 0.0)
            metric_tracker = MetricTracker(
                metrics_spec,
                dt=physics_dt,
                table_height_offset=_table_height_offset,
                robot_key=robot_key,
            )
        except ImportError:
            print("[WARN] benchmark.metric_tracker not available; metrics disabled.", flush=True)

    pre_reset_groups = classify_interactive_objects(interactive_objects)
    robot_art = pre_reset_groups.robot_articulation
    if robot_art is None:
        print("[WARN] No robot articulation found before sim.reset - skipping episode.")
        return None

    camera_rig = CameraRig(
        sim,
        camera_cfgs,
        enable_rgb=args.enable_rgb,
        enable_depth=False,
        robot_articulation=robot_art,
        camera_generalization_sample=camera_generalization_sample,
    )
    _sensor_owner.camera_rig = camera_rig
    print(f"[ep {episode_idx}] Camera rig IDs: {camera_rig.camera_ids}", flush=True)

    object_prim_paths = runtime.get("object_prim_paths", {})
    object_body_types = runtime.get("object_body_types", {})
    tacmap_object_prim_paths, tacmap_object_body_types = build_tacmap_raycast_targets(
        object_prim_paths,
        object_body_types,
    )
    robot_prim_path = str(
        runtime.get("robot_pose", {}).get("prim_path", "/World/Objects/GlobalRobot")
    )
    try:
        tactile_rig = TacMapRig(
            robot_key=robot_key,
            object_prim_paths=tacmap_object_prim_paths,
            object_body_types=tacmap_object_body_types,
            robot_prim_path=robot_prim_path,
            resolution_step=args.tactile_resolution_step,
            max_distance=args.tactile_max_distance,
        )
        _sensor_owner.tactile_rig = tactile_rig
        _validate_tactile_rig(policy, tactile_rig)
    except Exception:
        _sensor_owner.close()
        raise
    print(f"[ep {episode_idx}] TacMap sites: {list(tactile_rig.site_names)}", flush=True)

    _log_memory(f"ep {episode_idx} before initialize_scene_runtime_state/sim.reset", args.device)
    try:
        groups, controlled_articulations, _, _, articulation_hold_targets = (
            initialize_scene_runtime_state(
                sim=sim,
                physics_dt=physics_dt,
                interactive_objects=interactive_objects,
                object_prim_paths=runtime.get("object_prim_paths", {}),
                object_display_colors=runtime.get("object_display_colors", {}),
                collector=None,
                app_running_state_fn=lambda: simulation_app.is_running(),
            )
        )
    except Exception:
        _sensor_owner.close()
        tactile_rig = None
        camera_rig = None
        _log_memory(f"ep {episode_idx} initialize_scene_runtime_state/sim.reset failed", args.device)
        raise
    _log_memory(f"ep {episode_idx} after initialize_scene_runtime_state/sim.reset", args.device)

    robot_art = groups.robot_articulation
    if robot_art is None:
        print("[WARN] No robot articulation found - skipping episode.")
        _sensor_owner.close()
        tactile_rig = None
        camera_rig = None
        return None

    # Read joint limits once (invariant within an episode)
    _joint_limits = None
    if robot_art is not None:
        try:
            _joint_limits = read_joint_limits(robot_art)
        except Exception:
            pass

    def _record_joint_limit_baseline(phase: str) -> None:
        """Check reset/home state without contaminating rollout safety rates."""
        if metric_tracker is None or robot_art is None:
            return
        try:
            baseline_state = read_joint_state(robot_art)
            baseline = metric_tracker.record_joint_limit_baseline(
                phase,
                robot_state=baseline_state,
                joint_limits=_joint_limits,
            )
            if baseline.get("hard_violation"):
                names = [item.get("joint") for item in baseline.get("violating_joints", [])]
                print(
                    f"  [WARN] Joint-limit baseline '{phase}' already exceeds physical hard limits: {names}",
                    flush=True,
                )
        except Exception as exc:
            print(f"  [WARN] Joint-limit baseline '{phase}' check failed: {exc}", flush=True)

    global _active_dof_info
    try:
        camera_rig.set_robot_articulation(robot_art)
        camera_rig.initialize_after_reset()
        tactile_rig.initialize_after_reset(robot_art)

        # Reindex active DOF info to match the runtime joint order from Isaac Sim.
        # The URDF parse order may differ from the articulation joint order.
        if _active_dof_info is not None and robot_art is not None:
            runtime_joint_names = list(robot_art.data.joint_names)
            _active_dof_info = get_active_dof_info_for_runtime(
                robot_key or _active_dof_info.robot_key,
                runtime_joint_names,
            )
            policy.validate_runtime(
                robot_key,
                _active_dof_info.active_indices,
                _active_dof_info.active_joint_names,
            )
        else:
            raise ValueError("tactile policy evaluation requires --active-dof")
    except Exception:
        _sensor_owner.close()
        raise

    _record_joint_limit_baseline("post_reset")

    # Create ContactSensorReader after sim.reset() if contact_pairs configured
    _contact_reader = None
    _contact_pairs = runtime.get("contact_pairs", [])
    if _contact_pairs:
        try:
            from collector.contact_sensor_reader import ContactSensorReader
            _object_prim_paths = runtime.get("object_prim_paths", {})
            _contact_reader = ContactSensorReader(_contact_pairs, object_prim_paths=_object_prim_paths)
            if _contact_reader.available:
                print(f"  [contact] ContactSensorReader initialized for {len(_contact_pairs)} pair(s)", flush=True)
            else:
                _contact_reader = None
        except Exception as _e:
            print(f"  [WARN] Failed to create ContactSensorReader: {_e}", flush=True)
            _contact_reader = None

    # ---------- diagnose overhead camera pose ----------
    _oh_cam = camera_rig._cameras.get("cam_overhead")
    if _oh_cam is not None:
        _oh_pos  = _oh_cam.data.pos_w[0].cpu().tolist()
        _oh_quat = _oh_cam.data.quat_w_world[0].cpu().tolist()
        print(f"  [cam_overhead pose] pos={[f'{v:.3f}' for v in _oh_pos]}  "
              f"quat_wxyz={[f'{v:.3f}' for v in _oh_quat]}", flush=True)
    # ---------------------------------------------------

    # Render warmup: prime camera buffers BEFORE physics warmup.
    # sim.step() triggers physics+render but replicator annotators may not flush
    # until an explicit sim.render() after reset.  Mirrors replay.py step-11.
    _RENDER_WARMUP = 5
    print(f"[ep {episode_idx}] Render warmup ({_RENDER_WARMUP} renders)...", flush=True)
    try:
        for _ in range(_RENDER_WARMUP):
            sim.render()
    except Exception:
        _sensor_owner.close()
        raise

    # Warm up: move robot home
    # teleport=True instantly resets joint state + velocity so the PD controller
    # does not have to fight large residual velocities from the previous episode.
    print(f"[ep {episode_idx}] Homing robot ({args.warmup_steps} warm-up steps)...")
    try:
        move_controlled_articulations_home(controlled_articulations, articulation_hold_targets, teleport=True)
        for _ in range(args.warmup_steps):
            for _hook in pre_step_hooks:
                _hook()
            write_articulation_targets(controlled_articulations, articulation_hold_targets)
            _step_sim_with_mounted_camera_sync(
                sim,
                camera_rig,
                interactive_objects,
                physics_dt,
                render=args.render_every_physics_step,
            )
    except Exception:
        _sensor_owner.close()
        raise

    _record_joint_limit_baseline("post_home")

    print(f"[ep {episode_idx}] Camera preflight...", flush=True)
    # Capture reads the already-rendered frame.  Sync and render once after
    # homing so the preflight RGB and mounted-camera extrinsics are current.
    try:
        _render_camera_sample(sim, camera_rig)
        preflight_frames = _capture_complete_camera_set(camera_rig, sim, physics_dt)
        preflight_tactile = tactile_rig.capture(dt=physics_dt)
        _build_tactile_policy_obs(
            preflight_frames,
            preflight_tactile,
            robot_art,
            policy,
        )
    except Exception:
        _sensor_owner.close()
        raise
    print(
        f"[ep {episode_idx}] TacMap preflight: "
        f"{len(preflight_tactile['tacmap'])} sites",
        flush=True,
    )
    # Save preflight images if DEX2BENCH_PREFLIGHT_DIR is set
    _preflight_dir = os.environ.get("DEX2BENCH_PREFLIGHT_DIR", "")
    if _preflight_dir:
        os.makedirs(_preflight_dir, exist_ok=True)
        import cv2 as _cv2
        for camera_id in camera_rig.camera_ids:
            frame = preflight_frames[camera_id]
            if frame is not None and frame.rgb is not None:
                img_bgr = _cv2.cvtColor(frame.rgb, _cv2.COLOR_RGB2BGR)
                _cv2.imwrite(os.path.join(_preflight_dir, f"inference_preflight_{camera_id}.png"), img_bgr)
        with open(os.path.join(_preflight_dir, "inference_preflight_extrinsics.txt"), "w") as _ef:
            for camera_id in camera_rig.camera_ids:
                frame = preflight_frames[camera_id]
                if frame is not None:
                    p = np.asarray(frame.cam_pos_w).round(6).tolist()
                    q = np.asarray(frame.cam_quat_wxyz).round(6).tolist()
                    _ef.write(f"{camera_id} pos_w={p} quat_wxyz={q}\n")
    for camera_id in camera_rig.camera_ids:
        frame = preflight_frames[camera_id]
        pos = None if frame.cam_pos_w is None else np.asarray(frame.cam_pos_w).round(4).tolist()
        quat = None if frame.cam_quat_wxyz is None else np.asarray(frame.cam_quat_wxyz).round(4).tolist()
        print(
            f"  [camera] {camera_id}: rgb={tuple(frame.rgb.shape)} pos_w={pos} quat_wxyz={quat}",
            flush=True,
        )

    _reset_policy(policy, seed=episode_seed)

    POLICY_STRIDE = 3  # 60Hz physics / 20Hz policy
    # Budget logic aligned with eval_budget.sh: read expert_time_step/expert_time_s
    # from scene YAML, apply 1.5x scale, and allow --episode-steps to override it.
    scene_path = (
        args.task
        if os.path.isabs(args.task)
        else os.path.join(SCRIPT_DIR, args.task)
    )
    budget = resolve_episode_budget(
        scene_path,
        explicit_episode_steps=args.episode_steps,
        policy_stride=POLICY_STRIDE,
    )
    episode_steps = budget.episode_steps
    max_steps = budget.max_physics_steps
    print(
        f"[ep {episode_idx}] Budget source={budget.source}: "
        f"{episode_steps} policy steps, max_steps={max_steps}",
        flush=True,
    )
    rgb_ok = args.enable_rgb
    # Collect config: fps=20, step_stride=3 (60Hz physics, 20Hz policy)
    # ACT is trained at 20Hz: policy.t increments 20 times per second.
    # Physics runs at 60Hz; camera capture and ACT timestep advance at 20Hz.
    print(f"[ep {episode_idx}] Running policy for up to {max_steps} steps (policy @ every {POLICY_STRIDE} phys steps)...", flush=True)
    _current_action: np.ndarray | None = None
    _queued_actions = deque()
    _last_obs = None  # last valid obs dict (reused between camera captures)
    _success = False
    _success_step = -1
    _success_ctx: dict = {}
    _policy_query_count = 0
    _policy_query_count_at_stable_success: int | None = None
    _policy_exception: str | None = None
    _last_step = -1
    _latest_joint_command = None

    # Per-episode recording for --record-dir
    _recorder = getattr(args_cli, "_recorder", None)
    if _recorder is not None:
        _recorder.start_episode(
            object_ids,
            camera_rig.camera_ids,
            episode_idx=episode_idx,
            metadata={
                "scene_generalization_sample": relativize_sample_paths(
                    dict(scene_generalization_sample or {})
                ),
                "scene_generalization_config": relativize_sample_paths(
                    dict(runtime.get("scene_generalization_config") or {})
                ),
            },
        )

    # Realtime throttle: only active with --realtime flag
    import time as _time
    _realtime = getattr(args, "realtime", False)
    _next_step_wall = _time.monotonic() if _realtime else None

    # --dump-tactile-attention: JSONL sinks inside the episode's output dir
    _attn_path = None
    _modality_path = None
    if getattr(args, "dump_tactile_attention", False) and getattr(args, "output_dir", None):
        _attn_path = os.path.join(args.output_dir, "tactile_attention.jsonl")
        _modality_path = os.path.join(args.output_dir, "modality_attention.jsonl")

    frames = None
    obs = None
    # Reuse the post-step metric state for the next policy observation to avoid
    # a second qpos GPU-to-CPU transfer.
    _latest_metric_robot_state = None
    _sim_shutdown = False

    try:
        for step in range(max_steps):
            if not simulation_app.is_running():
                _sim_shutdown = True
                break

            # Check if user requested skip to next episode
            if _skip_episode_event.is_set():
                print(f"[ep {episode_idx}] Skipping to next episode (user pressed '1').", flush=True)
                break

            # Capture cameras at 20Hz; reuse last obs at 60Hz.  Rendering is
            # deliberately here (rather than after every physics step): this
            # frame represents the state reached by the preceding physics
            # step, exactly the state used for this policy observation.
            if step % POLICY_STRIDE == 0:
                if not args.render_every_physics_step:
                    _render_camera_sample(sim, camera_rig)
                frames = _capture_complete_camera_set(camera_rig, sim, physics_dt)
                tactile_capture = tactile_rig.capture(dt=physics_dt)
                if rgb_ok:
                    obs = _build_tactile_policy_obs(
                        frames,
                        tactile_capture,
                        robot_art,
                        policy,
                        robot_state=_latest_metric_robot_state,
                    )
                    if obs is not None:
                        _last_obs = obs
            elif _last_obs is not None and rgb_ok:
                _last_obs = _refresh_observation_joint_state(
                    _last_obs,
                    robot_art,
                    robot_state=_latest_metric_robot_state,
                )

            # Tactile ACT returns action chunks. Pop one entry at 20Hz and hold
            # it for the intervening 60Hz physics steps.
            policy_type_upper = args.policy_type.upper()
            uses_chunks = getattr(policy, "returns_action_chunks", False)
            if _last_obs is not None and rgb_ok:
                if uses_chunks:
                    if should_query_policy(step, stride=POLICY_STRIDE):
                        if _queued_actions:
                            if hasattr(policy, "update_obs"):
                                policy.update_obs(_last_obs)
                        else:
                            actions = policy.get_action(_last_obs)
                            if _attn_path is not None and hasattr(policy, "last_tactile_attention_weights"):
                                _attn_weights = policy.last_tactile_attention_weights()
                                if _attn_weights is not None:
                                    _log_tactile_attention(_attn_path, episode_idx, _policy_query_count, _attn_weights)
                                if _modality_path is not None and hasattr(policy, "last_modality_attention"):
                                    _mod_attn = policy.last_modality_attention()
                                    if _mod_attn is not None:
                                        _log_modality_attention(_modality_path, episode_idx, _policy_query_count, _mod_attn)
                            if actions is not None:
                                actions = np.asarray(actions, dtype=np.float32)
                                if actions.ndim == 1:
                                    _queued_actions.append(actions)
                                else:
                                    actions = actions.reshape(-1, actions.shape[-1])
                                    _queued_actions.extend(actions)
                                _policy_query_count += 1
                        if _queued_actions:
                            _current_action = np.asarray(_queued_actions.popleft(), dtype=np.float32)
                else:
                    # ACT: query at capture rate (20Hz), matching training data rate
                    if should_query_policy(step, stride=POLICY_STRIDE):
                        action = policy.get_action(_last_obs)
                        if _attn_path is not None and hasattr(policy, "last_tactile_attention_weights"):
                            _attn_weights = policy.last_tactile_attention_weights()
                            if _attn_weights is not None:
                                _log_tactile_attention(_attn_path, episode_idx, _policy_query_count, _attn_weights)
                            if _modality_path is not None and hasattr(policy, "last_modality_attention"):
                                _mod_attn = policy.last_modality_attention()
                                if _mod_attn is not None:
                                    _log_modality_attention(_modality_path, episode_idx, _policy_query_count, _mod_attn)
                        if action is not None:
                            _current_action = np.squeeze(action).astype(np.float32)
                            if _current_action.ndim > 1:
                                _current_action = _current_action.reshape(-1, _current_action.shape[-1])[0]
                            _policy_query_count += 1

            # --record-dir: collect object poses, qpos, actions, and cam frames
            if _recorder is not None and step % POLICY_STRIDE == 0 and _current_action is not None:
                _cam_frames = {}
                for _cid in camera_rig.camera_ids:
                    _cf = frames.get(_cid)
                    _cam_frames[_cid] = _cf.get("rgb") if isinstance(_cf, dict) else getattr(_cf, "rgb", None)
                _recorder.record_step(
                    qpos=robot_art.data.joint_pos[0].detach().cpu().numpy(),
                    action=_current_action,
                    object_states=_get_object_states(interactive_objects, _recorder.current_object_ids()),
                    camera_frames=_cam_frames,
                )

            # Apply current action target every physical step
            if _current_action is not None:
                target = articulation_hold_targets.get(id(robot_art))
                raw_action_vec = np.asarray(_current_action, dtype=np.float32).reshape(-1)
                action_vec = raw_action_vec
                if _active_dof_info is not None and len(action_vec) == _active_dof_info.active_dof:
                    action_vec = expand_to_full(action_vec, _active_dof_info)
                if target is not None and len(action_vec) == target.shape[1]:
                    target[0, :] = torch.from_numpy(action_vec).to(
                        dtype=target.dtype, device=target.device,
                    )
                    _latest_joint_command = _joint_command_context(
                        robot_art,
                        raw_action_vec,
                        action_vec,
                    )

            for _hook in pre_step_hooks:
                _hook()
            write_articulation_targets(controlled_articulations, articulation_hold_targets)
            _step_sim_with_mounted_camera_sync(
                sim,
                camera_rig,
                interactive_objects,
                physics_dt,
                render=args.render_every_physics_step,
            )
            _last_step = step

            # Throttle to real-time pace when --realtime is set
            if _next_step_wall is not None:
                _next_step_wall += physics_dt
                _sleep = _next_step_wall - _time.monotonic()
                if _sleep > 0:
                    _time.sleep(_sleep)

            # Update metric tracker every physics step
            if metric_tracker is not None and metric_tracker.available:
                states = _get_object_states(interactive_objects, object_ids)
                if _contact_reader is not None and _contact_reader.available:
                    _contact_reader.update(physics_dt)
                    _contact_data = _contact_reader.read()
                    for _obj_id, _forces in _contact_data.items():
                        if _obj_id in states:
                            states[_obj_id]["contact_forces"] = _forces
                _robot_state = None
                if robot_art is not None:
                    try:
                        _robot_state = read_joint_state(robot_art)
                    except Exception:
                        pass
                if _robot_state is not None:
                    _latest_metric_robot_state = _robot_state
                metric_tracker.update(
                    states,
                    sim_step=step,
                    dt=physics_dt,
                    robot_state=_robot_state,
                    joint_limits=_joint_limits,
                    joint_command=_latest_joint_command,
                )
                if metric_tracker.success:
                    if not _success:
                        _success = True
                        _success_step = step
                        _policy_query_count_at_stable_success = _policy_query_count
                        print(f"  [ep {episode_idx}] STABLE SUCCESS at step {step}!", flush=True)
                    if not no_early_stop:
                        break
            elif not _success and check_success_fn is not None and step % 30 == 0:
                # Legacy fallback: check success every 30 steps
                states = _get_object_states(interactive_objects, object_ids)
                if check_success_fn(states, _success_ctx, {}):
                    _success = True
                    _success_step = step
                    print(f"  [ep {episode_idx}] SUCCESS at step {step}!", flush=True)

            if (step + 1) % 200 == 0:
                policy_step = (step + 1) // POLICY_STRIDE
                cur_qpos = robot_art.data.joint_pos[0].cpu().numpy()
                policy_t = getattr(policy, "t", _policy_query_count)
                print(f"  [ep {episode_idx}] phys_step {step+1}/{max_steps} (policy_t={policy_t})", flush=True)

                # ---- save step-200 camera & ACT input images ----
                _save_step_images = os.environ.get("SAVE_STEP_IMAGES", "")
                if _save_step_images:
                    import cv2 as _cv2
                    _step_img_dir = os.path.join(_save_step_images, f"ep{episode_idx:04d}")
                    os.makedirs(_step_img_dir, exist_ok=True)
                    step_tag = f"step{step+1:04d}"

                    # 1) raw camera frames (as seen by renderer)
                    for _cid, _f in frames.items():
                        _rgb = _f.rgb if hasattr(_f, 'rgb') else _f.get('rgb')
                        if _rgb is not None and _rgb.ndim == 3:
                            _cv2.imwrite(
                                os.path.join(_step_img_dir, f"{step_tag}_{_cid}_cam.jpg"),
                                _cv2.cvtColor(_rgb, _cv2.COLOR_RGB2BGR),
                            )

                    # 2) ACT input images (resized & preprocessed)
                    if _last_obs is not None:
                        for _obs_key in _CAM_TO_OBS.values():
                            _img_chw = _last_obs.get(_obs_key)
                            if _img_chw is not None and isinstance(_img_chw, np.ndarray) and _img_chw.ndim == 3:
                                _img_hwc = np.moveaxis(_img_chw, 0, -1)
                                # ACT uses float32 [0,255]; DP uses float32 [0,1]
                                if _img_hwc.max() <= 1.0:
                                    _img_hwc = (_img_hwc * 255.0).astype(np.uint8)
                                else:
                                    _img_hwc = _img_hwc.astype(np.uint8)
                                _cv2.imwrite(
                                    os.path.join(_step_img_dir, f"{step_tag}_{_obs_key}_act.jpg"),
                                    _cv2.cvtColor(_img_hwc, _cv2.COLOR_RGB2BGR),
                                )
    except Exception as exc:
        _policy_exception = str(exc)
        print(
            f"[ERROR] Episode {episode_idx} aborted at phys_step={_last_step + 1} "
            f"before policy_query_count={_policy_query_count}: {exc}",
            flush=True,
        )

    if _sim_shutdown:
        episode_result = None
        terminated_reason = "shutdown"
    elif _policy_exception is not None:
        terminated_reason = "error"
    elif _skip_episode_event.is_set():
        terminated_reason = "skipped"
    elif _success and not no_early_stop:
        terminated_reason = "stable_success"
    else:
        terminated_reason = "max_steps"

    steps_completed = max(0, _last_step + 1)
    result_str = "SUCCESS" if _success else "FAIL"
    step_info = f" at step {_success_step}" if _success else ""

    if not _sim_shutdown and metric_tracker is not None and metric_tracker.available:
        mr = metric_tracker.finalize(
            steps=steps_completed,
            terminated_reason=terminated_reason,
            policy_query_count=_policy_query_count,
            policy_query_count_at_stable_success=_policy_query_count_at_stable_success,
            max_steps=max_steps,
            policy_stride=POLICY_STRIDE,
            evaluation_protocol="reach_and_stop" if getattr(args, "early_stop", True) else "fixed_horizon",
        )
        print(f"[ep {episode_idx}] Episode finished: {result_str}{step_info}  "
              f"stable_success={mr.stable_success} ever_instant={mr.ever_instant_success} "
              f"at_end={mr.at_end_success_observed} current_stage_completion_rate={mr.current_stage_completion_rate:.2f} "
              f"latched_stage_completion_rate={mr.latched_stage_completion_rate:.2f} "
              f"task_efficiency={mr.task_efficiency} safety_violation_step_rate={mr.safety_violation_step_rate or 0:.3f}", flush=True)
        _early_stop_enabled = bool(getattr(args, "early_stop", True))
        _episode_success = (
            bool(mr.stable_success)
            if _early_stop_enabled
            else bool(mr.at_end_success_budget)
        )
        episode_result = EpisodeResult(
            scene=args.task,
            episode_index=episode_idx,
            seed=episode_seed,
            episode_seed=episode_seed,
            base_seed=base_seed,
            seed_policy=seed_policy,
            scene_generalization_sample=scene_generalization_sample,
            success=_episode_success,
            steps=steps_completed,
            error=_policy_exception,
            task_family=metric_tracker.task_family,
            robot_key=robot_key,
            perturbation_axis=perturbation_axis,
            stable_success=mr.stable_success,
            ever_instant_success=mr.ever_instant_success,
            at_end_success=mr.at_end_success,
            at_end_success_observed=mr.at_end_success_observed,
            at_end_success_budget=mr.at_end_success_budget,
            first_success_step=mr.first_success_step,
            stable_success_step=mr.stable_success_step,
            first_stable_success_step=mr.first_stable_success_step,
            steps_to_stable_success=mr.steps_to_stable_success,
            policy_steps_to_stable_success=mr.policy_steps_to_stable_success,
            policy_queries_to_stable_success=mr.policy_queries_to_stable_success,
            time_to_stable_success_s=mr.time_to_stable_success_s,
            expert_time_s=mr.expert_time_s,
            expert_time_step=mr.expert_time_step,
            success_hold_s=mr.success_hold_s,
            terminal_success_rate=mr.terminal_success_rate,
            stage_completion_rate=mr.stage_completion_rate,
            normalized_progress_score=mr.normalized_progress_score,
            current_stage_completion_rate=mr.current_stage_completion_rate,
            current_normalized_progress_score=mr.current_normalized_progress_score,
            current_chain_depth=mr.current_chain_depth,
            latched_stage_completion_rate=mr.latched_stage_completion_rate,
            latched_normalized_progress_score=mr.latched_normalized_progress_score,
            latched_chain_depth=mr.latched_chain_depth,
            task_efficiency=mr.task_efficiency,
            kinematic_grasp_stability_index=mr.kinematic_grasp_stability_index,
            mean_kinematic_grasp_stability=mr.mean_kinematic_grasp_stability,
            per_object_best_kinematic_grasp_stability=mr.per_object_best_kinematic_grasp_stability,
            grasp_gsi_diagnostics=mr.grasp_gsi_diagnostics,
            robot_motion_metrics=mr.robot_motion_metrics,
            tool_selection_accuracy=mr.tool_selection_accuracy,
            tool_switch_success_rate=mr.tool_switch_success_rate,
            tool_switch_total=mr.tool_switch_total,
            tool_switch_successful=mr.tool_switch_successful,
            safety_violation_rate=mr.safety_violation_rate,
            safety_violation_rate_time=mr.safety_violation_rate_time,
            safety_violation_rate_event=mr.safety_violation_rate_event,
            safety_violation_step_rate=mr.safety_violation_step_rate,
            safety_violation_events_per_step=mr.safety_violation_events_per_step,
            episode_violation_rate=mr.episode_violation_rate,
            safety_hard_violation=mr.safety_hard_violation,
            drop_violation=mr.drop_violation,
            high_speed_violation=mr.high_speed_violation,
            excessive_impact_violation=mr.excessive_impact_violation,
            robot_constraint_violation=mr.robot_constraint_violation,
            finger_joint_limit_saturation=mr.finger_joint_limit_saturation,
            impact_metric_available=mr.impact_metric_available,
            robot_constraint_diagnostics=mr.robot_constraint_diagnostics,
            joint_limit_active_steps=mr.joint_limit_active_steps,
            joint_limit_active_step_rate=mr.joint_limit_active_step_rate,
            finger_joint_limit_saturation_steps=mr.finger_joint_limit_saturation_steps,
            finger_joint_limit_saturation_step_rate=mr.finger_joint_limit_saturation_step_rate,
            max_joint_limit_excess_rad=mr.max_joint_limit_excess_rad,
            max_joint_limit_excess_joint=mr.max_joint_limit_excess_joint,
            chain_depth=mr.chain_depth,
            stage_completion=mr.stage_completion,
            current_stage_completion=mr.current_stage_completion,
            stage_first_completion_step=mr.stage_first_completion_step,
            violation_counts=mr.violation_counts,
            policy_query_count=mr.policy_query_count,
            terminated_reason=mr.terminated_reason,
            timeseries=mr.timeseries,
            evaluation_protocol="reach_and_stop" if _early_stop_enabled else "fixed_horizon",
            early_stop=_early_stop_enabled,
            max_episode_steps=max_steps,
            dwell_time_s=metric_tracker.dwell_time_s,
        )
    elif not _sim_shutdown:
        print(f"[ep {episode_idx}] Episode finished: {result_str}{step_info}", flush=True)
        episode_result = EpisodeResult(
            scene=args.task,
            episode_index=episode_idx,
            seed=episode_seed,
            episode_seed=episode_seed,
            base_seed=base_seed,
            seed_policy=seed_policy,
            scene_generalization_sample=scene_generalization_sample,
            success=_success,
            steps=steps_completed,
            error=_policy_exception,
            robot_key=robot_key,
            perturbation_axis=perturbation_axis,
            terminated_reason=terminated_reason,
            evaluation_protocol="reach_and_stop" if getattr(args, "early_stop", True) else "fixed_horizon",
            early_stop=bool(getattr(args, "early_stop", True)),
            max_episode_steps=max_steps,
        )
    else:
        print(f"[ep {episode_idx}] Episode stopped because simulation_app is no longer running.", flush=True)

    try:
        if _contact_reader is not None:
            _contact_reader.close()
    except Exception as exc:
        print(f"  [WARN] Failed to close ContactSensorReader: {exc}", flush=True)
    _sensor_owner.close()
    try:
        del frames, obs, _last_obs, _current_action, metric_tracker, _contact_reader, tactile_rig, camera_rig
    except UnboundLocalError:
        pass
    import gc as _gc
    _gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    # --record-dir: write successful episode as minimal HDF5
    if _recorder is not None:
        _path = _recorder.finish_episode(_success)
        if _path is not None:
            print(f"[ep {episode_idx}] Recorded successful episode to {_path}", flush=True)

    _log_memory(f"ep {episode_idx} after episode cleanup", args.device)
    return episode_result


def _run_episode(*args, **kwargs):
    """Run one episode with unconditional CameraRig/TacMapRig cleanup."""

    _sensor_owner = _EpisodeSensorOwner()
    try:
        return _run_episode_impl(*args, _sensor_owner=_sensor_owner, **kwargs)
    finally:
        _sensor_owner.close()


def main() -> None:
    import traceback
    print("[main] started", flush=True)
    try:
        if not args_cli.enable_rgb:
            raise ValueError("--enable-rgb is required for tactile policy evaluation")
        if not args_cli.active_dof:
            raise ValueError("tactile checkpoints require --active-dof")
        _require_tactile_gpu(args_cli.device)

        task_path = os.path.join(SCRIPT_DIR, args_cli.task) if not os.path.isabs(args_cli.task) else args_cli.task
        if not os.path.isfile(task_path):
            raise FileNotFoundError(f"Task YAML not found: {task_path}")
        namespace_base_seed = int(args_cli.seed)
        # Channel-level seed decoupling: inv_cov uses a fixed seed so its
        # full random sampling is independent of the anchor-dependent channels.
        _gp = _normalize_generalization_profile(args_cli.generalization_profile)
        if _gp == "inv_cov":
            namespace_base_seed = INV_COV_EVAL_SEED
            print(f"[main] inv_cov channel: overriding seed to {namespace_base_seed}", flush=True)
        task_seed_id = resolve_task_seed_id(task_path, args_cli.task_seed_id)
        base_seed = task_base_seed(namespace_base_seed, task_seed_id)
        _seed_everything(base_seed)
        print(
            f"[main] Set eval seed namespace_base={namespace_base_seed} "
            f"task_seed_id={task_seed_id} base_seed={base_seed}",
            flush=True,
        )
        task = load_yaml(task_path)
        task_dir = os.path.dirname(task_path)
        print(f"[init] Loading TACTILE from {args_cli.ckpt_dir} ...", flush=True)
        policy = _load_policy(args_cli)
        (
            args_cli.tactile_resolution_step,
            args_cli.tactile_max_distance,
        ) = policy.resolve_tactile_sensor_settings(
            args_cli.tactile_resolution_step,
            args_cli.tactile_max_distance,
        )
        _log_memory("after TACTILE load", args_cli.device)
        if getattr(args_cli, "dump_tactile_attention", False):
            if hasattr(policy, "enable_tactile_attention_capture"):
                policy.enable_tactile_attention_capture()
                print("[init] Tactile attention capture ENABLED "
                      f"(sites: {list(policy.site_names)})", flush=True)
            else:
                print("[WARN] --dump-tactile-attention not supported by this policy", flush=True)

        default_collect_cfg = os.path.join(SCRIPT_DIR, "configs", "collect", "default.yaml")
        collect_cfg_path = args_cli.collect_config or default_collect_cfg
        # The checkpoint robot is authoritative unless explicitly repeated on CLI.
        from robots import get_robot_keys  # noqa: E402
        default_robot_key = args_cli.robot_key or policy.robot_key
        if default_robot_key:
            source = "--robot-key" if args_cli.robot_key else "checkpoint"
            print(f"[init] Robot key: {default_robot_key} (from {source})", flush=True)
        if default_robot_key != policy.robot_key:
            raise ValueError(
                f"--robot-key '{default_robot_key}' does not match checkpoint "
                f"robot_key '{policy.robot_key}'"
            )
        if not tacmap_supported(default_robot_key):
            raise ValueError(f"TacMap is not configured for robot '{default_robot_key}'")
        args_cli.resolved_robot_key = default_robot_key

        collect_cfg = load_collect_config(collect_cfg_path, robot_key=default_robot_key)
        # cam_overhead and cam_chest are excluded regardless of policy type.
        collect_cfg.cameras = [c for c in collect_cfg.cameras
                               if c.camera_id in _DEFAULT_INFERENCE_CAMERAS]
        _validate_camera_config(collect_cfg.cameras, policy)
        collect_cfg_by_robot = {default_robot_key: collect_cfg}
        print(f"[init] Collect config: {collect_cfg_path}", flush=True)
        print(f"[init] Camera robot profile: {default_robot_key}", flush=True)
        print(f"[init] Cameras: {[c.camera_id for c in collect_cfg.cameras]}", flush=True)

        # Load scene generalization config
        default_gen_cfg = os.path.join(SCRIPT_DIR, "configs", "scene", "generalization.yaml")
        gen_cfg_path = args_cli.generalization_config or default_gen_cfg
        generalization_raw = load_yaml(gen_cfg_path) or {}
        overrides = task.get("generalization_overrides") if isinstance(task, dict) else None
        merged_gen_raw = merge_scene_generalization_overrides(generalization_raw, overrides) if overrides else dict(generalization_raw)
        generalization_profile = _normalize_generalization_profile(args_cli.generalization_profile)
        merged_gen_raw = _apply_generalization_profile(
            merged_gen_raw,
            profile=generalization_profile,
        )
        # Re-apply task YAML overrides so per-scene settings (e.g.
        # microwave pose frozen with position_jitter_cm=0.0) survive
        # the profile's hard-coded defaults.
        if overrides:
            merged_gen_raw = merge_scene_generalization_overrides(merged_gen_raw, overrides)
        merged_gen_raw = _apply_generalization_split(
            merged_gen_raw,
            split=args_cli.generalization_split,
        )
        generalization_cfg = parse_scene_generalization_config(merged_gen_raw, config_path=gen_cfg_path)
        generalization_effective_split = getattr(generalization_cfg, "asset_split", args_cli.generalization_split or "seen")

        # Load anchor HDF5(s) for none/cov_only/inv_only profiles
        _anchor_sample = None
        _anchor_samples: list = []  # multiple anchors from --anchor-dir

        if args_cli.anchor_dir:
            # --anchor-dir: load main 0-24 + _1 25-49, cycle through them per episode.
            # This gives 50 unique backgrounds (main and _1 are disjoint sets across the
            # seen pool) and 50 unique demonstrations.  Falls back to all main episodes
            # when _1 variants are absent (legacy datasets).
            _anchor_dir = args_cli.anchor_dir
            if not os.path.isdir(_anchor_dir):
                raise FileNotFoundError(f"--anchor-dir is not a directory: {_anchor_dir}")
            import glob as _glob
            import re as _re
            _all_files = sorted([
                p for p in _glob.glob(os.path.join(_anchor_dir, "episode_*.hdf5"))
                if "_replay." not in os.path.basename(p)
            ])
            # Separate main and _1 variant episodes
            _main_episodes = []
            _var_episodes = []
            for _p in _all_files:
                _bn = os.path.basename(_p)
                _m = _re.match(r'episode_(\d+)(?:_1)?\.hdf5', _bn)
                if _m:
                    _ep_num = int(_m.group(1))
                    if "_1." in _bn:
                        _var_episodes.append((_ep_num, _p))
                    else:
                        _main_episodes.append((_ep_num, _p))
            _main_episodes.sort(key=lambda x: x[0])
            _var_episodes.sort(key=lambda x: x[0])
            # main 0-24 (25 unique demos + 25 backgrounds) +
            # _1 25-49 (25 different demos + 25 disjoint backgrounds) = 50 + 50
            _selected = [_p for _num, _p in _main_episodes if _num < 25] + \
                        [_p for _num, _p in _var_episodes if _num >= 25]
            if not _selected:
                # Legacy datasets without _1 variants fall back to all main episodes.
                _selected = [_p for _num, _p in _main_episodes]
            _all_episodes = _selected
            if not _all_episodes:
                raise ValueError(f"--anchor-dir has no episode_*.hdf5 files: {_anchor_dir}")
            for _ep_path in _all_episodes:
                import json as _json_anchor, h5py as _h5py_anchor
                with _h5py_anchor.File(_ep_path, "r") as _hf:
                    if "meta/scene_generalization_sample" not in _hf:
                        raise ValueError(
                            f"--anchor-dir episode has no scene_generalization_sample: {_ep_path}")
                    _raw = _hf["meta/scene_generalization_sample"][()]
                    _sample_json = _json_anchor.loads(_raw.decode() if isinstance(_raw, bytes) else str(_raw))
                    try:
                        from build.generalization import _find_dataset_root, resolve_sample_paths
                        _sample_json = resolve_sample_paths(_sample_json, _find_dataset_root(SCRIPT_DIR))
                    except Exception as _exc:
                        print(f"[WARN] Failed to resolve anchor paths: {_exc}", flush=True)
                    try:
                        validate_resolved_object_placement_keys(
                            _sample_json,
                            (obj["id"] for obj in task.get("objects", [])),
                            required=True,
                        )
                    except (TypeError, ValueError) as _exc:
                        raise ValueError(
                            f"Invalid anchor metadata in {_ep_path}: {_exc}"
                        ) from _exc
                    _anchor_samples.append(dict_to_generalization_sample(_sample_json))
            print(f"[init] Anchor dir loaded: {_anchor_dir} ({len(_anchor_samples)} episodes)",
                  flush=True)
            # Set first anchor as the initial _anchor_sample for compatibility
            if _anchor_samples:
                _anchor_sample = _anchor_samples[0]

        elif args_cli.anchor_hdf5:
            if not os.path.isfile(args_cli.anchor_hdf5):
                raise FileNotFoundError(f"--anchor-hdf5 file not found: {args_cli.anchor_hdf5}")
            import json as _json_anchor, h5py as _h5py_anchor
            with _h5py_anchor.File(args_cli.anchor_hdf5, "r") as _hf:
                if "meta/scene_generalization_sample" not in _hf:
                    raise ValueError(f"--anchor-hdf5 {args_cli.anchor_hdf5} has no scene_generalization_sample")
                _raw = _hf["meta/scene_generalization_sample"][()]
                _sample_json = _json_anchor.loads(_raw.decode() if isinstance(_raw, bytes) else str(_raw))
                try:
                    from build.generalization import _find_dataset_root, resolve_sample_paths
                    _sample_json = resolve_sample_paths(_sample_json, _find_dataset_root(SCRIPT_DIR))
                except Exception as _exc:
                    print(f"[WARN] Failed to resolve anchor HDF5 generalization paths: {_exc}", flush=True)
                try:
                    validate_resolved_object_placement_keys(
                        _sample_json,
                        (obj["id"] for obj in task.get("objects", [])),
                        required=True,
                    )
                except (TypeError, ValueError) as _exc:
                    raise ValueError(
                        f"Invalid anchor metadata in {args_cli.anchor_hdf5}: {_exc}"
                    ) from _exc
                _anchor_sample = dict_to_generalization_sample(_sample_json)
            print(f"[init] Anchor HDF5 loaded: {args_cli.anchor_hdf5}", flush=True)

        # Validate: none/cov_only/inv_only require --anchor-hdf5 or --anchor-dir
        if generalization_profile in ("none", "cov_only", "inv_only") and _anchor_sample is None:
            raise ValueError(
                f"--anchor-hdf5 or --anchor-dir is required for --generalization-profile={generalization_profile}. "
                "Only inv_cov can run without an anchor episode."
            )

        generalization_enabled = True
        if generalization_enabled:
            print(f"[INFO] Scene generalization enabled: {gen_cfg_path}")
            print(
                f"[INFO] Scene generalization protocol: "
                f"profile={generalization_profile} split={generalization_effective_split}",
                flush=True,
            )

        global _active_dof_info
        _active_dof_info = None
        dof_info = get_active_dof_info(default_robot_key)
        if args_cli.active_dof:
            _active_dof_info = dof_info
            args_cli.state_dim = dof_info.active_dof
            print(
                f"[init] Active DOF: robot={default_robot_key} "
                f"full_dof={dof_info.full_dof} active_dof={dof_info.active_dof} "
                f"(will reindex to runtime joint order each episode)",
                flush=True,
            )
        else:
            args_cli.state_dim = dof_info.full_dof
            print(f"[init] Full DOF: robot={default_robot_key} state_dim={dof_info.full_dof}", flush=True)
        if args_cli.state_dim != policy.state_dim:
            raise ValueError(
                f"checkpoint state_dim={policy.state_dim} does not match "
                f"robot active_dof={args_cli.state_dim}"
            )
        args_cli.chunk_size = policy.chunk_size

        print("[init] Building SimulationContext ...", flush=True)
        sim_cfg = sim_utils.SimulationCfg(
            dt=0.0166666,
            device=args_cli.device,
            physx=sim_utils.PhysxCfg(
                enable_ccd=True,
                enable_stabilization=True,
                bounce_threshold_velocity=0.01,
                gpu_max_rigid_contact_count=2**23,
                gpu_max_rigid_patch_count=2**22,
            ),
        )
        sim = sim_utils.SimulationContext(sim_cfg)
        physics_dt = sim.get_physics_dt()
        sim.set_camera_view([0.0, 0.0, 2.20], [0.0, 0.0, 0.82])
        runtime = None
        camera_rig = None
        scene_generalization_sample = None
        robot_key = default_robot_key

        # Load success checker from task YAML
        check_success_fn = _load_success_checker(task)
        if check_success_fn is not None:
            print(f"[init] Success checker loaded from task YAML.", flush=True)
        else:
            print(f"[init] No success checker found - success rate will not be tracked.", flush=True)

        if args_cli.record_dir is not None:
            from utils.inference_recorder import InferenceRecorder
            args_cli._recorder = InferenceRecorder(
                args_cli.record_dir,
                record_all=args_cli.record_all,
                clear=not bool(getattr(args_cli, "append_record_dir", False)),
            )

        episode_idx = max(0, int(getattr(args_cli, "start_episode", 0) or 0) - 1)
        _episodes_run = 0
        _all_results: list = []
        _max_episodes = getattr(args_cli, 'num_episodes', 0) or 0
        _output_dir = None
        _policy_name = None
        _append_episode_result = None

        out_dir_path = getattr(args_cli, 'output_dir', None)
        if not out_dir_path:
            import datetime

            # Build descriptive folder name: <task>_<hand>_<model>_<profile>_<timestamp>
            # e.g. 60_rh5dg2_gr00t_none_0630_1930

            # 1. Task identifier: "60_breadbasket_fast_food_loading" -> "60"
            task_path = getattr(args_cli, 'task', '')
            task_base = os.path.splitext(os.path.basename(task_path or ''))[0]
            task_parts = task_base.split('_', 1)
            task_id = task_parts[0] if task_parts and task_parts[0].isdigit() else task_base

            # 2. Robot short name: "multi_ur5_rh5dg2_with_flange" -> "rh5dg2"
            robot_key = getattr(args_cli, 'resolved_robot_key', '') or 'unknown'
            robot_parts = robot_key.replace('multi_', '').split('_')
            robot_short = '_'.join(p for p in robot_parts if p not in ('with', 'flange', 'ur5', 'ur5e'))
            if len(robot_short) > 30:
                robot_short = '_'.join(p for p in robot_parts[2:4] if p not in ('with', 'flange')) or robot_key.split('_')[-1]

            # 3. Model name: ckpt_name stripped of trailing version suffix
            model_name = os.path.splitext(args_cli.ckpt_name or 'model')[0]
            model_name = model_name.rstrip('_n15') if model_name.endswith('_n15') else model_name

            # 4. Generalization profile: "none", "cov", "inv", "inv_cov"
            profile = _normalize_generalization_profile(args_cli.generalization_profile)

            # 5. Timestamp (MMDD_HHMM, no year)
            timestamp = datetime.datetime.now().strftime("%m%d_%H%M")

            folder_name = f"{task_id}_{robot_short}_{model_name}_{profile}_{timestamp}"
            output_root = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "output", "metric"))
            out_dir_path = os.path.join(output_root, folder_name)

        if out_dir_path:
            args_cli.output_dir = out_dir_path
            from benchmark.results import (
                append_episode_result as _append_episode_result,
                initialize_incremental_results as _initialize_incremental_results,
                load_episode_results as _load_episode_results,
            )
            from pathlib import Path as _Path
            _output_dir = _Path(out_dir_path)
            # --policy-display-name takes priority (ckpt-relative path);
            # fall back to --policy-name, then ckpt_dir/ckpt_name, then policy type.
            _policy_name = (
                getattr(args_cli, 'policy_display_name', None)
                or args_cli.policy_name
                or (os.path.join(args_cli.ckpt_dir, args_cli.ckpt_name) if getattr(args_cli, 'ckpt_dir', None) and getattr(args_cli, 'ckpt_name', None) else None)
                or args_cli.policy_type.upper()
            )
            if getattr(args_cli, "append_output", False):
                _all_results.extend(_load_episode_results(_output_dir))
                print(f"[RESULTS] Loaded {len(_all_results)} existing episode(s) from {_output_dir}", flush=True)
            _initialize_incremental_results(_output_dir, append=bool(getattr(args_cli, "append_output", False)))
            print(f"[RESULTS] Incremental output enabled: {_output_dir}", flush=True)
        if _max_episodes > 0:
            print(f"[init] Running {_max_episodes} episode(s).\n", flush=True)
        else:
            print("[init] Starting infinite evaluation loop. Press Ctrl+C to stop.", flush=True)
        print("[init] Type '1' + Enter in this terminal to skip to the next episode.\n", flush=True)
        _start_stdin_skip_listener()
        while simulation_app.is_running():
            episode_idx += 1
            _skip_episode_event.clear()  # reset skip flag at start of each episode

            # --start-episode: skip episodes before the requested index
            if episode_idx < args_cli.start_episode:
                print(f"[ep {episode_idx}] Skipped (--start-episode={args_cli.start_episode})", flush=True)
                continue

            print(f"\n{'='*60}\n  Episode {episode_idx}\n{'='*60}", flush=True)
            _log_memory(f"ep {episode_idx} start", args_cli.device)

            episode_seed = _episode_seed(base_seed, episode_idx)
            _seed_everything(episode_seed)
            print(
                f"[ep {episode_idx}] episode_seed={episode_seed} "
                f"(base_seed={base_seed}; {RUN_POLICY_SEED_POLICY})",
                flush=True,
            )

            # Rotate anchor when --anchor-dir is used (episode_idx is 1-indexed)
            _ep_anchor = _anchor_sample
            _anchor_idx = None
            if _anchor_samples:
                _anchor_idx = (episode_idx - 1) % len(_anchor_samples)
                _ep_anchor = _anchor_samples[_anchor_idx]

            if generalization_profile == "none" and _ep_anchor is not None:
                scene_generalization_sample = copy.deepcopy(_ep_anchor)
                scene_generalization_sample.robot_key = default_robot_key
            elif generalization_profile in ("cov_only", "inv_only") and _ep_anchor is not None:
                fresh_sample = sample_scene_generalization(
                    generalization_cfg,
                    enabled=generalization_enabled,
                    task_asset_codes=collect_task_asset_codes(task),
                    asset_split="unseen",
                    available_robot_keys=get_robot_keys(),
                )
                if generalization_profile == "cov_only":
                    resample_groups = {"object_pose", "table_height"}
                else:
                    resample_groups = {"background", "table_surface", "light", "clutter", "camera"}
                scene_generalization_sample = merge_generalization_samples(
                    copy.deepcopy(_ep_anchor), fresh_sample, resample_groups,
                )
                scene_generalization_sample.robot_key = default_robot_key
            else:
                scene_generalization_sample = sample_scene_generalization(
                    generalization_cfg,
                    enabled=generalization_enabled,
                    task_asset_codes=collect_task_asset_codes(task),
                    asset_split="unseen" if generalization_profile == "inv_cov" else args_cli.generalization_split,
                    available_robot_keys=get_robot_keys(),
                )
            robot_key = str(getattr(scene_generalization_sample, "robot_key", "") or default_robot_key)
            if robot_key != policy.robot_key:
                raise ValueError(
                    f"episode robot '{robot_key}' does not match tactile checkpoint "
                    f"robot '{policy.robot_key}'; disable robot_asset randomization"
                )
            if not tacmap_supported(robot_key):
                raise ValueError(f"TacMap is not configured for robot '{robot_key}'")
            episode_collect_cfg = collect_cfg_by_robot.get(robot_key)
            if episode_collect_cfg is None:
                episode_collect_cfg = load_collect_config(collect_cfg_path, robot_key=robot_key)
                # Apply the same stereo+wrist-only default as the initial load.
                episode_collect_cfg.cameras = [c for c in episode_collect_cfg.cameras
                                               if c.camera_id in _DEFAULT_INFERENCE_CAMERAS]
                _validate_camera_config(episode_collect_cfg.cameras, policy)
                collect_cfg_by_robot[robot_key] = episode_collect_cfg
            if args_cli.active_dof:
                _active_dof_info = get_active_dof_info(robot_key)
                if _active_dof_info.active_dof != args_cli.state_dim:
                    raise ValueError(
                        f"Episode robot {robot_key} active_dof={_active_dof_info.active_dof} "
                        f"does not match loaded policy state_dim={args_cli.state_dim}. "
                        "Use a checkpoint trained for this robot or disable robot_asset randomization."
                    )
            else:
                _active_dof_info = None
            if generalization_enabled:
                for line in scene_generalization_sample_debug_lines(scene_generalization_sample):
                    print(line)
            _sample_dict = (
                scene_generalization_sample_to_dict(scene_generalization_sample)
                if scene_generalization_sample is not None else None
            )
            if _sample_dict is not None:
                _sample_dict["_generalization_profile"] = generalization_profile
                _sample_dict["_generalization_split"] = generalization_effective_split
                if _anchor_idx is not None:
                    _sample_dict["_anchor_source_index"] = _anchor_idx
                # Strip legacy light field (HDR deprecated; only usd_scene_light is used)
                _sample_dict.get("appearance", {}).pop("light", None)
            _current_gen_sample_dict = _sample_dict if generalization_enabled else None

            # Every episode gets a freshly built scene. Episode 1 starts from an
            # empty stage; later episodes also clear stale prims and camera sensors.
            if episode_idx > 1:
                print(f"[ep {episode_idx}] Rebuilding scene (full rebuild)...", flush=True)
                _log_memory(f"ep {episode_idx} before CameraRig close", args_cli.device)
                if camera_rig is not None:
                    camera_rig.close()
                    camera_rig = None
                runtime = None
                import gc as _gc
                _gc.collect()
                if torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                _log_memory(f"ep {episode_idx} after CameraRig close", args_cli.device)

                # Synchronize CUDA periodically before USD operations to reduce
                # long-run PhysX Fabric tensor-alignment failures.
                if episode_idx % 10 == 0 and torch.cuda.is_available():
                    try:
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                    except Exception:
                        pass

                _log_memory(f"ep {episode_idx} before scene clear", args_cli.device)
                clear_scene_prims_preserve_robot(sim=sim, preserve_robot=False)
                _log_memory(f"ep {episode_idx} after scene clear", args_cli.device)

                try:
                    from utils.usd_prims import delete_prim_compat as _del_prim
                    from utils.usd_prims import get_current_stage_compat as _get_stage
                    _stage = _get_stage()
                    _sensors_prim = _stage.GetPrimAtPath("/World/Sensors")
                    if _sensors_prim.IsValid():
                        for _child in list(_sensors_prim.GetChildren()):
                            try:
                                _del_prim(str(_child.GetPath()), stage=_stage)
                            except Exception:
                                pass
                except Exception as _e:
                    print(f"[WARN] Could not delete sensor prims: {_e}", flush=True)
                _gc.collect()
                if torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                _log_memory(f"ep {episode_idx} after scene cleanup", args_cli.device)

            print(f"[ep {episode_idx}] Building scene ...", flush=True)
            runtime = build_scene(
                task,
                task_dir,
                generalization_enabled=generalization_enabled,
                generalization_cfg=generalization_cfg,
                generalization_sample=scene_generalization_sample,
                robot_key=robot_key,
                existing_robot_runtime=None,
            )
            _log_memory(f"ep {episode_idx} after build_scene", args_cli.device)
            if generalization_enabled:
                _current_gen_sample_dict = dict(runtime.get("scene_generalization_sample") or _sample_dict or {})
                _current_gen_sample_dict["_generalization_profile"] = generalization_profile
                _current_gen_sample_dict["_generalization_split"] = generalization_effective_split
                if _anchor_idx is not None:
                    _current_gen_sample_dict["_anchor_source_index"] = _anchor_idx

            print(f"[ep {episode_idx}] Preparing CameraRig config ...", flush=True)
            print(f"[ep {episode_idx}] Camera robot profile: {robot_key}", flush=True)
            for camera_cfg in episode_collect_cfg.cameras:
                if camera_cfg.mount_type == "robot_link":
                    print(
                        f"  [camera mount] {camera_cfg.camera_id}: parent={camera_cfg.parent_link} "
                        f"offset_xyz={camera_cfg.offset_xyz} offset_rpy={camera_cfg.offset_rpy}",
                        flush=True,
                    )
            # Pass the same full generalization sample shape used by replay and
            # collection so wrist-camera offsets resolve identically.
            _cam_sample = _sample_dict or {}
            print(f"[ep {episode_idx}] CameraRig will be created before sim.reset()", flush=True)

            result = _run_episode(
                sim=sim,
                runtime=runtime,
                camera_cfgs=episode_collect_cfg.cameras,
                camera_generalization_sample=_cam_sample,
                policy=policy,
                physics_dt=physics_dt,
                args=args_cli,
                episode_idx=episode_idx,
                episode_seed=episode_seed,
                base_seed=base_seed,
                scene_generalization_sample=_current_gen_sample_dict,
                seed_policy=RUN_POLICY_SEED_POLICY,
                check_success_fn=check_success_fn,
                metrics_spec=task.get("metrics", {}),
                perturbation_axis=_robustness_category(
                    scene_generalization_sample,
                    generalization_enabled=generalization_enabled,
                    generalization_profile=generalization_profile,
                ),
                robot_key=robot_key,
                task_instruction=task.get("description", ""),
            )
            if result is not None:
                compact_result = _compact_episode_result(result)
                _all_results.append(compact_result)
                if _append_episode_result is not None and _output_dir is not None and _policy_name is not None:
                    _append_episode_result(_output_dir, _policy_name, result, _all_results)
                    print(f"  [RESULTS] Appended episode {episode_idx} to {_output_dir / 'per_episode.jsonl'}", flush=True)
                _success_count = sum(1 for r in _all_results if r.success)
                rate = _success_count / len(_all_results) * 100
                print(f"  [STATS] Success rate: {_success_count}/{len(_all_results)} = {rate:.1f}%", flush=True)
                result = None
                compact_result = None
                import gc as _gc
                _gc.collect()
                if torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                _log_memory(f"ep {episode_idx} after result write/compact", args_cli.device)

            _episodes_run += 1
            if _max_episodes > 0 and _episodes_run >= _max_episodes:
                print(f"\n[done] Reached target of {_max_episodes} episode(s). Terminating.", flush=True)
                os._exit(0)

        if camera_rig is not None:
            camera_rig.close()

        # Benchmark results are written incrementally after each completed episode.
        if _all_results and _output_dir is not None:
            from benchmark.metrics import summarize as _summarize_results
            _summary = _summarize_results(_all_results)
            print(f"\n[RESULTS] Final results available in {_output_dir}", flush=True)
            print(f"[RESULTS] Success rate: {_summary['success_rate']:.1%} "
                  f"({_summary['successes']}/{_summary['total']})", flush=True)
            _progress = _summary.get('latched_stage_completion_rate')
            _progress_str = f"{_progress:.3f}" if _progress is not None else "N/A"
            _safety_step_rate = _summary.get('safety_violation_step_rate')
            _safety_str = f"{_safety_step_rate:.3f}" if _safety_step_rate is not None else "N/A"
            print(f"[RESULTS] Safety Violation Step Rate: {_safety_str}  "
                  f"Progress: {_progress_str}", flush=True)

        print("\n[done] Simulation ended.")
    except Exception:
        print("[main] Exception in main():", flush=True)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    import traceback as _tb
    _exit_code = 0
    try:
        main()
    except Exception as _exc:
        _exit_code = 1
        print(f"[run_policy] FATAL ERROR: {_exc}", flush=True)
        _tb.print_exc()
    finally:
        simulation_app.close()
        os._exit(_exit_code)
