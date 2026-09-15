#!/usr/bin/env python3
"""Visualize tactile attention-pooling weights dumped by --dump-tactile-attention.

Reads <output-dir>/tactile_attention.jsonl (one row per policy query with
head-averaged weights over finger sites) and renders:

  - a heatmap of attention weights over policy steps (one column per finger),
  - a per-finger mean-weight bar chart (episode-aggregated),
  - an optional per-episode heatmap grid.

Usage:
    python tools/vis/plot_tactile_attention.py <output-dir> [--site-names f0 f1 ...] [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load_rows(jsonl_path: Path) -> list[dict]:
    rows: list[dict] = []
    with jsonl_path.open("r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"no rows found in {jsonl_path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", help="Eval output dir containing tactile_attention.jsonl")
    parser.add_argument("--site-names", nargs="+", default=None,
                        help="Finger/site names in checkpoint order (labels heatmap columns)")
    parser.add_argument("--out-dir", default=None,
                        help="Where to write PNGs (default: <output_dir>)")
    args = parser.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("ERROR: matplotlib is required for visualization", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    jsonl_path = output_dir / "tactile_attention.jsonl"
    rows = load_rows(jsonl_path)

    episodes = sorted({int(row["episode"]) for row in rows})
    num_sites = len(rows[0]["weights"])
    site_names = args.site_names
    if site_names is None or len(site_names) != num_sites:
        site_names = [f"site{i}" for i in range(num_sites)]

    out_dir = Path(args.out_dir) if args.out_dir else output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Full heatmap: policy steps × sites ────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(6, num_sites * 0.9), max(4, len(rows) / 14)))
    matrix = np.array([row["weights"] for row in rows])
    im = ax.imshow(matrix.T, aspect="auto", interpolation="nearest", cmap="viridis",
                   vmin=0.0, vmax=max(0.15, float(matrix.max())))
    ax.set_xlabel("policy query index")
    ax.set_ylabel("finger site")
    ax.set_yticks(range(num_sites), site_names)
    ax.set_title(f"Tactile attention pooling weights\n({len(rows)} queries, {len(episodes)} episodes)")
    fig.colorbar(im, ax=ax, label="attention weight")
    fig.tight_layout()
    fig.savefig(out_dir / "tactile_attention_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"[plot] heatmap -> {out_dir / 'tactile_attention_heatmap.png'}")

    # ── 2. Per-finger mean bar chart ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(6, num_sites * 0.7), 4))
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0)
    colors = plt.cm.viridis(means / max(means.max(), 1e-9))
    ax.bar(range(num_sites), means, yerr=stds, color=colors, capsize=3)
    ax.set_xticks(range(num_sites), site_names, rotation=45, ha="right")
    ax.set_ylabel("mean attention weight")
    ax.set_title("Mean attention weight per finger site (heads averaged)")
    uniform = 1.0 / num_sites
    ax.axhline(uniform, color="red", linestyle="--", linewidth=1,
               label=f"uniform ({uniform:.2f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "tactile_attention_means.png", dpi=150)
    plt.close(fig)
    print(f"[plot] means -> {out_dir / 'tactile_attention_means.png'}")

    # ── 3. Per-episode mean weights (finger × episode) ───────────────────────
    if len(episodes) > 1:
        fig, ax = plt.subplots(figsize=(max(6, len(episodes) * 0.35), max(4, num_sites * 0.5)))
        ep_matrix = np.zeros((len(episodes), num_sites))
        for col, ep in enumerate(episodes):
            ep_rows = [row["weights"] for row in rows if int(row["episode"]) == ep]
            ep_matrix[col] = np.mean(ep_rows, axis=0)
        im = ax.imshow(ep_matrix.T, aspect="auto", cmap="viridis")
        ax.set_xlabel("episode")
        ax.set_xticks(range(len(episodes)), [str(ep) for ep in episodes])
        ax.set_yticks(range(num_sites), site_names)
        ax.set_title("Mean attention weight per finger per episode")
        fig.colorbar(im, ax=ax, label="mean attention weight")
        fig.tight_layout()
        fig.savefig(out_dir / "tactile_attention_per_episode.png", dpi=150)
        plt.close(fig)
        print(f"[plot] per-episode -> {out_dir / 'tactile_attention_per_episode.png'}")

    # ── 4. Modality proportions: stacked bars per episode ────────────────────
    modality_path = output_dir / "modality_attention.jsonl"
    if modality_path.is_file():
        import collections as _collections
        mod_rows = []
        with modality_path.open("r") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    mod_rows.append(json.loads(line))
        if mod_rows:
            mod_episodes = sorted({int(row["episode"]) for row in mod_rows})
            mod_names = ["latent", "proprio", "visual", "tactile"]
            per_ep_means = []
            for ep in mod_episodes:
                values = [row["modalities"] for row in mod_rows if int(row["episode"]) == ep]
                per_ep_means.append([np.mean([v[name] for v in values]) for name in mod_names])
            per_ep_means = np.array(per_ep_means)

            colors = {"latent": "#9e9e9e", "proprio": "#64b5f6",
                      "visual": "#4caf50", "tactile": "#ff7043"}
            fig, ax = plt.subplots(figsize=(max(8, len(mod_episodes) * 0.45), 5))
            bottom = np.zeros(len(mod_episodes))
            for i, name in enumerate(mod_names):
                ax.bar(range(len(mod_episodes)), per_ep_means[:, i], bottom=bottom,
                       label=name, color=colors[name], width=0.8)
                bottom += per_ep_means[:, i]
            ax.set_xticks(range(len(mod_episodes)), [str(ep) for ep in mod_episodes])
            ax.set_xlabel("episode")
            ax.set_ylabel("cross-attention proportion")
            ax.set_title("Decoder cross-attention share per modality (mean over layers/queries/steps)")
            ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.08))
            ax.axhline(1.0, color="black", linewidth=0.5)
            fig.tight_layout()
            fig.savefig(out_dir / "modality_attention_per_episode.png", dpi=150)
            plt.close(fig)
            print(f"[plot] modality per-episode -> {out_dir / 'modality_attention_per_episode.png'}")

            # overall mean pie
            fig, ax = plt.subplots(figsize=(5, 5))
            overall = per_ep_means.mean(axis=0)
            ax.pie(overall, labels=mod_names, colors=[colors[n] for n in mod_names],
                   autopct="%1.1f%%", startangle=90)
            ax.set_title("Overall cross-attention share per modality")
            fig.tight_layout()
            fig.savefig(out_dir / "modality_attention_pie.png", dpi=150)
            plt.close(fig)
            print(f"[plot] modality pie -> {out_dir / 'modality_attention_pie.png'}")

    print("[plot] done")


if __name__ == "__main__":
    main()
