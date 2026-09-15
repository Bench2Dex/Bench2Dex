"""Compare normal, zeroed, and temporally shuffled TacMap inputs offline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import torch
from torch.utils.data import DataLoader

from .dataset import (
    NormalizationStats,
    TactileEpisodeDataset,
    inspect_dataset_directory,
    make_frame_refs,
    split_frame_refs,
)
from .tactile_policy import TactileACTPolicy
from .train import evaluate_loader, load_checkpoint


def evaluate_ablation_loaders(
    policy: TactileACTPolicy,
    loaders: Mapping[str, DataLoader],
    device: torch.device,
    *,
    seed: int = 0,
) -> dict[str, dict[str, float | int]]:
    """Evaluate named DataLoaders with matched posterior random seeds."""

    required = {"normal", "zero", "shuffle"}
    if set(loaders) != required:
        raise ValueError(f"ablation loaders must be exactly {sorted(required)}")
    report: dict[str, dict[str, float | int]] = {}
    for mode in ("normal", "zero", "shuffle"):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        metrics = evaluate_loader(policy, loaders[mode], device)
        report[mode] = {
            **metrics,
            "sample_count": int(len(loaders[mode].dataset)),
        }
    return report


def _loader(dataset: TactileEpisodeDataset, batch_size: int, num_workers: int) -> DataLoader:
    kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": False,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = 1
    return DataLoader(dataset, **kwargs)


def evaluate_checkpoint(
    checkpoint: str | Path,
    dataset_dir: str | Path,
    *,
    batch_size: int = 2,
    num_workers: int = 0,
    val_ratio: float = 0.1,
    seed: int = 0,
    device: str = "auto",
) -> dict[str, dict[str, float | int]]:
    """Restore a checkpoint, validate its schema, and run three ablations."""

    resolved_device = torch.device(
        device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    policy, payload = load_checkpoint(checkpoint, device=resolved_device)
    camera_names = tuple(payload["camera_names"])
    site_names = tuple(payload["site_names"])
    active_indices = tuple(int(index) for index in payload["active_indices"])

    episodes = inspect_dataset_directory(
        dataset_dir,
        camera_names=camera_names,
        use_active_dof=True,
    )
    schema = episodes[0]
    if schema.robot_key != payload["robot_key"]:
        raise ValueError(
            f"checkpoint robot '{payload['robot_key']}' does not match dataset '{schema.robot_key}'"
        )
    if schema.site_names != site_names:
        raise ValueError("checkpoint tactile site order does not match the dataset")
    if schema.active_indices != active_indices:
        raise ValueError("checkpoint active joint indices do not match the dataset")
    if schema.state_dim != policy.config.state_dim:
        raise ValueError(
            f"checkpoint state_dim={policy.config.state_dim} does not match dataset {schema.state_dim}"
        )

    _, val_refs = split_frame_refs(
        episodes,
        val_ratio=val_ratio,
        chunk_size=policy.config.chunk_size,
        seed=seed,
    )
    refs = val_refs or make_frame_refs(episodes)
    stats = NormalizationStats.from_dict(payload["normalization_stats"])
    image_size = tuple(int(value) for value in payload.get("image_size", (480, 640)))
    stored_tactile_size = payload.get("tactile_size")
    tactile_size = (
        None
        if stored_tactile_size is None
        else tuple(int(value) for value in stored_tactile_size)
    )
    common = {
        "episodes": episodes,
        "frame_refs": refs,
        "stats": stats,
        "chunk_size": policy.config.chunk_size,
        "image_size": image_size,
        "tactile_size": tactile_size,
    }
    datasets = {
        mode: TactileEpisodeDataset(
            tactile_mode=mode,
            shuffle_seed=seed,
            **common,
        )
        for mode in ("normal", "zero", "shuffle")
    }
    loaders = {
        mode: _loader(dataset, batch_size=batch_size, num_workers=num_workers)
        for mode, dataset in datasets.items()
    }
    return evaluate_ablation_loaders(policy, loaders, resolved_device, seed=seed)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output", default=None, help="Optional JSON report path")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = evaluate_checkpoint(
        args.checkpoint,
        args.dataset_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        seed=args.seed,
        device=args.device,
    )
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
