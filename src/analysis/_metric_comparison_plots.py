"""Shared scatter + delta-histogram plot for a pairwise metric comparison.

Used by compare_bpnet_cherimoya.py and compare_cherimoya_versions.py.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def plot_metric_comparison(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    label_a: str,
    label_b: str,
    metric_label: str,
    lower_is_better: bool,
    read_depth_col: str,
    out_path: Path,
) -> tuple[float, float]:
    """Scatter (A vs. B, colored by read depth) + histogram of B - A deltas.

    Points below (if lower_is_better) or above (if not) the y=x diagonal
    indicate B is better than A. Returns (median_delta, mean_delta).
    """
    delta = df[col_b] - df[col_a]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    ax = axes[0]
    sc = ax.scatter(
        df[col_a],
        df[col_b],
        c=np.log10(df[read_depth_col]),
        cmap="viridis",
        s=20,
        alpha=0.8,
        linewidths=0,
    )
    fig.colorbar(sc, ax=ax, label="log10(total reads)")

    lims = [
        min(df[col_a].min(), df[col_b].min()) - 0.01,
        max(df[col_a].max(), df[col_b].max()) + 0.01,
    ]
    ax.plot(lims, lims, color="gray", linestyle="dashed", linewidth=1, zorder=0)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    ax.set_xlabel(f"{label_a} {metric_label}")
    ax.set_ylabel(f"{label_b} {metric_label}")
    ax.set_title(metric_label)

    _, pval = wilcoxon(df[col_a], df[col_b])
    n_better = int((delta < 0).sum() if lower_is_better else (delta > 0).sum())
    ax.text(
        0.05,
        0.95,
        f"{label_b} better: {n_better}/{len(df)}\nWilcoxon p={pval:.2e}",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
    )

    ax = axes[1]
    ax.axvline(0, color="gray", linestyle="dashed", linewidth=1)
    ax.hist(delta, bins=30, color="steelblue", edgecolor="white", linewidth=0.5)
    ax.set_xlabel(f"Δ{metric_label} ({label_b} − {label_a})")
    ax.set_ylabel("Experiments")
    ax.set_title(f"Change in {metric_label}")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return delta.median(), delta.mean()
