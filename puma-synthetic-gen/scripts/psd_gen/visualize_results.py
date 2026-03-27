#!/usr/bin/env python3
"""
Visualize realized statistics for psd_gen datasets.

Produces a summary figure (psd_summary.png) with four panels:
  - Achieved vs target porosity (bar chart)
  - Slice-to-slice porosity std dev (line chart)
  - Realized statistics table
  - Sample mid-slice per dataset (if png_slices/ present)

Examples
--------
  # Specific datasets
  python psd_gen/visualize_results.py \\
      --datasets sigma_025 sigma_050 sigma_075 sigma_100 sigma_125

  # All datasets in the output directory
  python psd_gen/visualize_results.py --all

  # Custom data dir and output location
  python psd_gen/visualize_results.py --all \\
      --data-dir /path/to/output/psd_gen \\
      --out-dir /path/to/figures
"""

import argparse
import json
import warnings
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

try:
    from PIL import Image as PILImage
except ImportError:
    PILImage = None

DEFAULT_DATA = Path(__file__).parent.parent.parent / "output" / "psd_gen"

PALETTE = [
    "#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0",
    "#00BCD4", "#FF5722", "#8BC34A", "#9E9E9E", "#607D8B",
]
DARK_BG   = "#0f1117"
PANEL_BG  = "#1a1d27"
TABLE_HDR = "#2a2d37"
GRID_CLR  = "#444"


# ── data loading ──────────────────────────────────────────────────────────────

def _load_stats(dataset_dir: Path) -> dict | None:
    p = dataset_dir / "realized_stats.json"
    return json.loads(p.read_text()) if p.exists() else None


# ── plot helpers ──────────────────────────────────────────────────────────────

def _style_ax(ax, title: str) -> None:
    ax.set_facecolor(PANEL_BG)
    ax.set_title(title, color="white", fontweight="bold", fontsize=11)
    ax.tick_params(colors="white")
    ax.spines[:].set_color(GRID_CLR)


def _porosity_bars(ax, names, stats, colors, target) -> None:
    achieved = [stats[n]["overall_porosity"] for n in names]
    bars = ax.bar(names, achieved, color=colors, alpha=0.85,
                  edgecolor="white", linewidth=0.5)
    ax.axhline(target, color="white", linestyle="--", linewidth=1.5,
               alpha=0.7, label=f"Target ({target})")
    ax.set_ylabel("Achieved Porosity", color="white")
    ax.legend(facecolor=TABLE_HDR, labelcolor="white",
              edgecolor=GRID_CLR, fontsize=9)
    ax.set_ylim(min(achieved) - 0.02, max(achieved) + 0.02)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", color="white")
    for bar, val in zip(bars, achieved):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.001,
                f"{val:.3f}", ha="center", va="bottom",
                color="white", fontsize=8, fontweight="bold")


def _slice_std_line(ax, names, stats, colors) -> None:
    std_vals = [stats[n]["std_slice_porosity"] for n in names]
    ax.plot(names, std_vals, "o-", color="#FFD700", linewidth=2,
            markersize=8, markerfacecolor="white")
    ax.set_ylabel("Std of Slice Porosity", color="white")
    ax.grid(alpha=0.2, color="white")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", color="white")
    for x, y in zip(names, std_vals):
        ax.annotate(f"{y:.4f}", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", color="white", fontsize=8)


def _stats_table(ax, names, stats) -> None:
    ax.axis("off")
    col_labels = ["Dataset", "Porosity", "Std (slice)",
                  "Min slice", "Max slice", "Slices"]
    rows = [
        [
            n,
            f"{stats[n]['overall_porosity']:.4f}",
            f"{stats[n]['std_slice_porosity']:.4f}",
            f"{stats[n]['min_slice_porosity']:.4f}",
            f"{stats[n]['max_slice_porosity']:.4f}",
            str(stats[n].get("nslices", "?")),
        ]
        for n in names
    ]
    tbl = ax.table(cellText=rows, colLabels=col_labels,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor(TABLE_HDR if r == 0 else PANEL_BG)
        cell.set_text_props(color="white")
        cell.set_edgecolor(GRID_CLR)


def _slice_previews(gs, row, fig, names, data_dir, colors, ncols) -> None:
    for col, (name, color) in enumerate(zip(names, colors)):
        if col >= ncols:
            break
        ax = fig.add_subplot(gs[row, col])
        ax.set_facecolor(PANEL_BG)

        png_dir = data_dir / name / "png_slices"
        slices  = sorted(png_dir.glob("slice_*.png")) if png_dir.exists() else []

        if slices and PILImage is not None:
            img = np.array(PILImage.open(slices[len(slices) // 2]).convert("L"))
            ax.imshow(img, cmap="gray", aspect="auto")
            ax.set_title(f"{name}\n({len(slices) // 2}/{len(slices)})",
                         color=color, fontsize=8, fontweight="bold")
        else:
            ax.text(0.5, 0.5, "no image", transform=ax.transAxes,
                    ha="center", va="center", color="white", fontsize=9)
            ax.set_title(name, color=color, fontsize=8)

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        prog="visualize_results.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA),
                    help="Root directory containing dataset subdirs "
                         "(default: output/psd_gen/)")
    target = ap.add_mutually_exclusive_group(required=True)
    target.add_argument("--datasets", nargs="+", metavar="NAME",
                        help="Dataset names to include")
    target.add_argument("--all", action="store_true",
                        help="Include all subdirs in --data-dir")
    ap.add_argument("--porosity-target", type=float, default=0.492,
                    help="Reference line on porosity bar chart (default: 0.492)")
    ap.add_argument("--out-dir", type=str, default=None,
                    help="Where to save psd_summary.png "
                         "(default: same as --data-dir)")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        ap.error(f"--data-dir does not exist: {data_dir}")

    if args.all:
        names = sorted(d.name for d in data_dir.iterdir() if d.is_dir())
    else:
        names = args.datasets

    stats = {n: _load_stats(data_dir / n) for n in names}
    valid = [n for n in names if stats[n] is not None]

    if not valid:
        print("No realized_stats.json found. Run generate.py first.")
        return

    missing = [n for n in names if stats[n] is None]
    if missing:
        print(f"[WARN] No stats for: {missing} — skipping")

    ncols  = max(len(valid), 2)
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(valid))]

    fig = plt.figure(figsize=(min(4 * ncols + 4, 24), 14))
    fig.patch.set_facecolor(DARK_BG)
    gs = gridspec.GridSpec(
        3, ncols, figure=fig,
        hspace=0.55, wspace=0.35,
        top=0.91, bottom=0.07, left=0.06, right=0.97,
    )
    fig.suptitle("psd_gen — Synthetic Dataset Summary",
                 fontsize=15, fontweight="bold", color="white", y=0.97)

    half = max(ncols // 2, 1)

    ax_bar = fig.add_subplot(gs[0, :half])
    _style_ax(ax_bar, "Achieved vs Target Porosity")
    _porosity_bars(ax_bar, valid, stats, colors, args.porosity_target)

    ax_std = fig.add_subplot(gs[0, half:])
    _style_ax(ax_std, "Slice-to-Slice Porosity Variation")
    _slice_std_line(ax_std, valid, stats, colors)

    ax_tbl = fig.add_subplot(gs[1, :])
    _style_ax(ax_tbl, "Realized Statistics")
    _stats_table(ax_tbl, valid, stats)

    _slice_previews(gs, 2, fig, valid, data_dir, colors, ncols)

    out_dir  = Path(args.out_dir) if args.out_dir else data_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "psd_summary.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Saved: {out_path.resolve()}")
    plt.close()


if __name__ == "__main__":
    main()
