"""Train the tactile-aware ACT policy on Dex2Bench HDF5 episodes."""

from __future__ import annotations

import argparse
import json
import math
import pickle
import random
from pathlib import Path
from typing import Iterable, Mapping

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from robots.active_dof_utils import get_active_dof_info_for_hdf5
from .act.utils import get_norm_stats_dex2bench
from .act_tactile_dataset import ActTactileEpisodicDataset
from .dataset import (
    DEFAULT_CAMERA_NAMES,
    NormalizationStats,
    TactileEpisodeDataset,
    failed_demo_warnings,
    inspect_dataset_directory,
)
from .act_tactile_model import TactileACTConfig
from .tactile_policy import TactileACTPolicy


CHECKPOINT_VERSION = 4
SUPPORTED_CHECKPOINT_VERSIONS = {CHECKPOINT_VERSION}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _move_batch(batch, device: torch.device) -> dict[str, torch.Tensor]:
    if not isinstance(batch, Mapping):
        images, qpos, actions, is_pad, tactile = batch
        batch = {
            "images": images,
            "qpos": qpos,
            "actions": actions,
            "is_pad": is_pad,
            "tactile": tactile,
        }
    return {
        key: value.to(device, non_blocking=device.type == "cuda")
        for key, value in batch.items()
        if key in {"qpos", "images", "tactile", "actions", "is_pad"}
    }


def train_step(
    policy: TactileACTPolicy,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    *,
    max_grad_norm: float = 1.0,
) -> dict[str, float]:
    """Run one optimizer step and return scalar metrics."""

    policy.train()
    optimizer.zero_grad(set_to_none=True)
    losses = policy.compute_loss(_move_batch(batch, device))
    if not torch.isfinite(losses["loss"]):
        raise FloatingPointError(f"non-finite training loss: {losses['loss'].detach().cpu().item()}")
    losses["loss"].backward()
    if max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
    optimizer.step()
    return {
        key: float(losses[key].detach().cpu())
        for key in ("loss", "l1", "kl")
    }


@torch.no_grad()
def evaluate_loader(
    policy: TactileACTPolicy,
    loader: Iterable[dict[str, torch.Tensor]],
    device: torch.device,
) -> dict[str, float]:
    """Return sample-weighted loss metrics for one DataLoader."""

    policy.eval()
    totals = {"loss": 0.0, "l1": 0.0, "kl": 0.0}
    sample_count = 0
    for batch in loader:
        moved = _move_batch(batch, device)
        losses = policy.compute_loss(moved)
        batch_size = int(moved["qpos"].shape[0])
        sample_count += batch_size
        for key in totals:
            value = float(losses[key].detach().cpu())
            if not math.isfinite(value):
                raise FloatingPointError(f"non-finite validation {key}: {value}")
            totals[key] += value * batch_size
    if sample_count == 0:
        raise ValueError("cannot evaluate an empty DataLoader")
    return {key: value / sample_count for key, value in totals.items()}


def _stats_to_lists(stats: NormalizationStats) -> dict[str, list[float]]:
    return {key: value.tolist() for key, value in stats.as_dict().items()}


def _tactile_sensor_metadata(episode) -> dict[str, int | float]:
    """Read the TacMap encoding contract, with legacy-compatible defaults."""

    def _scalar(h5: h5py.File, key: str, default):
        dataset = h5.get(key)
        return default if dataset is None else dataset[()]

    with h5py.File(episode.path, "r") as h5:
        resolution_step = int(
            _scalar(h5, "robot/tactile/meta/resolution_step", 1)
        )
        image_size = int(
            _scalar(h5, "robot/tactile/meta/image_size", episode.tactile_shape[0])
        )
        native_resolution = int(
            _scalar(
                h5,
                "robot/tactile/meta/native_resolution",
                image_size * resolution_step,
            )
        )
        max_distance_m = float(
            _scalar(h5, "robot/tactile/meta/max_distance_m", 0.015)
        )
    return {
        "resolution_step": resolution_step,
        "native_resolution": native_resolution,
        "image_size": image_size,
        "max_distance_m": max_distance_m,
    }


def save_checkpoint(
    path: str | Path,
    *,
    policy: TactileACTPolicy,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    global_step: int,
    stats: NormalizationStats,
    camera_names: tuple[str, ...],
    site_names: tuple[str, ...],
    robot_key: str,
    active_indices: tuple[int, ...],
    active_joint_names: tuple[str, ...],
    tactile_sensor_metadata: Mapping[str, int | float],
    image_size: tuple[int, int] = (480, 640),
    tactile_size: tuple[int, int] | None = None,
) -> Path:
    """Save everything needed for offline reconstruction and evaluation."""

    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": CHECKPOINT_VERSION,
        "model_state_dict": policy.state_dict(),
        "optimizer_state_dict": None if optimizer is None else optimizer.state_dict(),
        "model_config": policy.config.to_dict(),
        "kl_weight": policy.kl_weight,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "normalization_stats": _stats_to_lists(stats),
        "camera_names": list(camera_names),
        "site_names": list(site_names),
        "robot_key": robot_key,
        "active_indices": list(active_indices),
        "active_joint_names": list(active_joint_names),
        "image_size": list(image_size),
        "tactile_size": None if tactile_size is None else list(tactile_size),
        "tactile_sensor_metadata": dict(tactile_sensor_metadata),
    }
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def _torch_load(path: Path, device: str | torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # PyTorch versions before the weights_only argument.
        return torch.load(path, map_location=device)


def load_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[TactileACTPolicy, dict]:
    """Reconstruct a policy and load a tactile checkpoint."""

    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    payload = _torch_load(checkpoint_path, device)
    if payload.get("format_version") not in SUPPORTED_CHECKPOINT_VERSIONS:
        raise ValueError(
            f"unsupported tactile checkpoint version {payload.get('format_version')}"
        )
    stored_config = TactileACTConfig.from_dict(payload["model_config"])
    policy = TactileACTPolicy(stored_config, kl_weight=float(payload["kl_weight"]))
    policy.load_state_dict(payload["model_state_dict"], strict=True)
    policy.config = stored_config
    policy.model.config = stored_config
    policy.to(device)
    return policy, payload


def _make_loader(
    dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    seed: int,
) -> DataLoader:
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
        "generator": torch.Generator().manual_seed(seed),
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = 1
    return DataLoader(dataset, **kwargs)


def _mean_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
    if not metrics:
        raise ValueError("training epoch produced no steps")
    return {
        key: float(np.mean([item[key] for item in metrics]))
        for key in ("loss", "l1", "kl")
    }


def run_training(args: argparse.Namespace) -> Path:
    """Train the tactile-only private ACT copy with ACT episode sampling."""

    set_seed(args.seed)
    device = torch.device(args.device if args.device != "auto" else (
        "cuda" if torch.cuda.is_available() else "cpu"
    ))
    camera_names = tuple(args.camera_names)
    episodes = inspect_dataset_directory(
        args.dataset_dir,
        camera_names=camera_names,
        use_active_dof=args.use_active_dof,
        robot_key=args.robot_key,
    )
    for warning in failed_demo_warnings(episodes):
        print(f"[tactile_policy] WARNING: {warning}", flush=True)
    paths = [str(episode.path) for episode in episodes]
    active_dof_info = get_active_dof_info_for_hdf5(paths[0]) if args.use_active_dof else None

    train_paths, val_paths = paths, []
    if args.val_ratio > 0 and len(paths) > 1:
        order = np.random.default_rng(args.seed).permutation(len(paths))
        val_count = min(max(1, int(len(paths) * args.val_ratio)), len(paths) - 1)
        val_paths = [paths[index] for index in order[:val_count]]
        train_paths = [paths[index] for index in order[val_count:]]

    norm_stats, _ = get_norm_stats_dex2bench(train_paths, active_dof_info=active_dof_info)
    stats = NormalizationStats.from_dict(norm_stats)

    dataset_args = (list(camera_names), norm_stats, args.chunk_size, active_dof_info)
    train_dataset = ActTactileEpisodicDataset(train_paths, *dataset_args)
    val_dataset = ActTactileEpisodicDataset(val_paths, *dataset_args) if val_paths else None
    pin_memory = device.type == "cuda"
    train_loader = _make_loader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                num_workers=args.num_workers, pin_memory=pin_memory, seed=args.seed)
    val_loader = None if val_dataset is None else _make_loader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=pin_memory, seed=args.seed,
    )

    schema = episodes[0]
    tactile_size = schema.tactile_shape if args.tactile_height is None else (
        args.tactile_height, args.tactile_width
    )
    active_joint_names = tuple(schema.joint_names[index] for index in schema.active_indices)
    sensor_metadata = _tactile_sensor_metadata(schema)
    model_config = TactileACTConfig(
        state_dim=schema.state_dim,
        camera_names=camera_names,
        site_names=schema.site_names,
        tactile_height=tactile_size[0],
        tactile_width=tactile_size[1],
        chunk_size=args.chunk_size,
        hidden_dim=args.hidden_dim,
        dim_feedforward=args.dim_feedforward,
        nheads=args.nheads,
        enc_layers=args.enc_layers,
        dec_layers=args.dec_layers,
        dropout=args.dropout,
        backbone=args.backbone,
        lr=args.lr,
        lr_backbone=args.backbone_lr,
        weight_decay=args.weight_decay,
    )
    policy = TactileACTPolicy(model_config, kl_weight=args.kl_weight).to(device)
    optimizer = policy.configure_optimizer()
    output_dir = Path(args.ckpt_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "dataset_stats.pkl").open("wb") as handle:
        pickle.dump(stats.as_dict(), handle)

    best_path, best_metric, global_step = output_dir / "policy_best.ckpt", math.inf, 0
    for epoch in range(args.epochs):
        epoch_metrics = []
        for batch_index, batch in enumerate(train_loader):
            epoch_metrics.append(train_step(policy, optimizer, batch, device, max_grad_norm=args.max_grad_norm))
            global_step += 1
            if args.max_steps_per_epoch and batch_index + 1 >= args.max_steps_per_epoch:
                break
        train_metrics = _mean_metrics(epoch_metrics)
        val_metrics = evaluate_loader(policy, val_loader, device) if val_loader is not None else None
        selection_metric = (val_metrics or train_metrics)["loss"]
        metadata = dict(
            policy=policy, optimizer=optimizer, epoch=epoch, global_step=global_step, stats=stats,
            camera_names=camera_names, site_names=schema.site_names, robot_key=schema.robot_key,
            active_indices=schema.active_indices, active_joint_names=active_joint_names,
            tactile_sensor_metadata=sensor_metadata, image_size=(args.image_height, args.image_width),
            tactile_size=tactile_size,
        )
        if (epoch + 1) % args.save_last_interval == 0 or epoch + 1 == args.epochs:
            save_checkpoint(output_dir / "policy_last.ckpt", **metadata)
        if selection_metric < best_metric:
            best_metric = selection_metric
            save_checkpoint(best_path, **metadata)
        print(f"[tactile_policy] epoch={epoch + 1}/{args.epochs} train_loss={train_metrics['loss']:.6f}", flush=True)
    return best_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument(
        "--robot-key",
        default=None,
        help="Expected HDF5 robot key; auto-detected when omitted",
    )
    parser.add_argument("--camera-names", nargs="+", default=list(DEFAULT_CAMERA_NAMES))
    parser.add_argument("--epochs", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--val-ratio", type=float, default=0.0)
    parser.add_argument("--chunk-size", type=int, default=30)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dim-feedforward", type=int, default=3200)
    parser.add_argument("--nheads", type=int, default=8)
    parser.add_argument("--enc-layers", type=int, default=4)
    parser.add_argument("--dec-layers", type=int, default=7)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--backbone", choices=("resnet18", "resnet34", "resnet50"), default="resnet18")
    parser.add_argument("--use-active-dof", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--tactile-height", type=int, default=None)
    parser.add_argument("--tactile-width", type=int, default=None)
    parser.add_argument("--kl-weight", type=float, default=10.0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-steps-per-epoch", type=int, default=0)
    parser.add_argument("--save-last-interval", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if (args.tactile_height is None) != (args.tactile_width is None):
        parser.error("--tactile-height and --tactile-width must be set together")
    if args.epochs <= 0 or args.batch_size <= 0 or args.num_workers < 0 or args.save_last_interval <= 0:
        parser.error("epochs/batch-size/save-last-interval must be positive and num-workers must be non-negative")
    best_path = run_training(args)
    print(f"[tactile_policy] best checkpoint: {best_path}", flush=True)


if __name__ == "__main__":
    main()
