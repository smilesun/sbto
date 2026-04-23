"""
Side-by-side knot distribution comparison for 2+ solver result directories.

Reads knot_distribution.csv from each directory (already filtered for
cost < threshold and no-fall during the run).  For every joint, points from
each method are plotted at the same knot timestep positions with a small
horizontal offset and distinct colours so distributions can be compared at a
glance.

Usage
-----
    uv run python scripts/plot_knot_comparison.py \\
        DIR1 DIR2 [DIR3 ...] \\
        [--labels CEM PCBO Wishart] \\
        [--joints 0 1 2 3 4] \\
        [--max-joints 10] \\
        [--output /tmp/comparison.pdf] \\
        [--point-size 4] \\
        [--alpha 0.35] \\
        [--title "CEM vs PCBO — shared init"]

If a directory does not contain knot_distribution.csv (e.g. CEM with all
particles above cost threshold), that method is skipped with a warning.

Output
------
Saves a PDF (and PNG for quick viewing) to --output path.
If --output is omitted, saves to the current directory as knot_comparison.pdf.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

COLOURS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def _auto_label(directory: str) -> str:
    """Derive a short label from the directory name (strip timestamp prefix)."""
    name = Path(directory).name
    # Format: YYYY_MM_DD__HH_MM_SS__<description>
    parts = name.split("__", 2)
    return parts[-1] if len(parts) >= 3 else name


def load_knot_csv(directory: str) -> pd.DataFrame | None:
    path = os.path.join(directory, "knot_distribution.csv")
    if not os.path.exists(path):
        print(f"  [WARN] No knot_distribution.csv in {directory} — skipping.")
        return None
    df = pd.read_csv(path)
    return df


def plot_comparison(
    directories: list[str],
    labels: list[str],
    joints: list[int] | None,
    max_joints: int,
    output: str,
    point_size: float,
    alpha: float,
    title: str,
    offset_frac: float = 0.3,
) -> None:
    # Load data for each method
    data: list[tuple[str, pd.DataFrame]] = []
    for d, lbl in zip(directories, labels):
        df = load_knot_csv(d)
        if df is not None:
            data.append((lbl, df))

    if not data:
        print("No valid knot_distribution.csv files found. Exiting.")
        sys.exit(1)

    # Determine joints to plot
    all_joints = sorted(data[0][1]["joint"].unique())
    if joints is not None:
        plot_joints = [j for j in joints if j in all_joints]
    else:
        plot_joints = all_joints[:max_joints]

    if not plot_joints:
        print("No joints to plot after filtering. Exiting.")
        sys.exit(1)

    # Knot timestep positions (use union across all methods)
    all_timesteps = sorted(
        set().union(*[set(df["knot_timestep"].unique()) for _, df in data])
    )
    t_arr = np.array(all_timesteps)
    gap = float(np.diff(t_arr).min()) if len(t_arr) > 1 else 1.0

    K = len(data)
    # Horizontal offsets: spread methods evenly within ±offset_frac*gap/2
    half = offset_frac * gap / 2
    if K == 1:
        offsets = [0.0]
    else:
        offsets = list(np.linspace(-half, half, K))

    n_joints = len(plot_joints)
    fig_h = max(3, 1.6 * n_joints)
    fig, axs = plt.subplots(n_joints, 1, figsize=(14, fig_h), sharex=False)
    if n_joints == 1:
        axs = [axs]

    rng = np.random.default_rng(0)

    for row_idx, joint in enumerate(plot_joints):
        ax = axs[row_idx]

        for method_idx, (label, df) in enumerate(data):
            colour = COLOURS[method_idx % len(COLOURS)]
            offset = offsets[method_idx]
            jitter_half = (half / K) * 0.6  # sub-jitter within each method's slot

            joint_df = df[df["joint"] == joint]
            for t in all_timesteps:
                vals = joint_df.loc[joint_df["knot_timestep"] == t, "value"].values
                if vals.size == 0:
                    continue
                jitter = rng.uniform(-jitter_half, jitter_half, size=vals.size)
                ax.scatter(
                    t + offset + jitter,
                    vals,
                    color=colour,
                    s=point_size,
                    alpha=alpha,
                    linewidths=0,
                    label=label if t == all_timesteps[0] else "_nolegend_",
                )

        ax.set_ylabel(f"joint {joint}", fontsize=7)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.tick_params(labelsize=7)
        if row_idx == 0:
            ax.legend(
                loc="upper right",
                fontsize=7,
                markerscale=2,
                framealpha=0.7,
            )

    axs[-1].set_xlabel("Knot timestep")
    method_str = " vs ".join(lbl for lbl, _ in data)
    n_str = " | ".join(f"{lbl}: {df['sample_rank'].nunique()}" for lbl, df in data)
    suptitle = title or f"Knot Distribution Comparison — {method_str}"
    fig.suptitle(f"{suptitle}\n({n_str} feasible samples)", fontsize=9)
    fig.tight_layout()

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)

    pdf_path = str(out.with_suffix(".pdf"))
    png_path = str(out.with_suffix(".png"))
    fig.savefig(pdf_path, format="pdf")
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Side-by-side knot distribution comparison")
    parser.add_argument("dirs", nargs="+", help="Result directories to compare")
    parser.add_argument("--labels", nargs="*", default=None,
                        help="Method labels (auto-derived from dir names if omitted)")
    parser.add_argument("--joints", nargs="*", type=int, default=None,
                        help="Specific joint indices to plot (default: all up to --max-joints)")
    parser.add_argument("--max-joints", type=int, default=10,
                        help="Maximum number of joints to show (default: 10)")
    parser.add_argument("--output", default="knot_comparison.pdf",
                        help="Output file path (PDF and PNG are both saved)")
    parser.add_argument("--point-size", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=0.35)
    parser.add_argument("--title", default="", help="Custom plot title")
    args = parser.parse_args()

    dirs = args.dirs
    labels = args.labels or [_auto_label(d) for d in dirs]
    if len(labels) < len(dirs):
        labels += [_auto_label(d) for d in dirs[len(labels):]]

    print(f"Comparing {len(dirs)} method(s):")
    for d, lbl in zip(dirs, labels):
        n_csv = os.path.exists(os.path.join(d, "knot_distribution.csv"))
        print(f"  [{lbl}]  {d}  csv={'yes' if n_csv else 'NO'}")

    plot_comparison(
        directories=dirs,
        labels=labels,
        joints=args.joints,
        max_joints=args.max_joints,
        output=args.output,
        point_size=args.point_size,
        alpha=args.alpha,
        title=args.title,
    )


if __name__ == "__main__":
    main()
