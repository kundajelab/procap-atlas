#!/usr/bin/env python3
"""Plot observed versus predicted BPNet counts for one saved prediction NPZ.

The input NPZ is produced by:
    python src/bpnet/benchmark/benchmark_bpnet.py -e ENCSR882DWM --save-output

Examples:
    python src/bpnet/benchmark/plot_counts_prediction_scatter.py -e ENCSR882DWM
    python src/bpnet/benchmark/plot_counts_prediction_scatter.py -e ENCSR882DWM --scatter
"""

import argparse
import re
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_PREDICTIONS_DIR = REPO_ROOT / "predictions" / "bpnet"
DEFAULT_OUT_DIR = REPO_ROOT / "plots" / "bpnet"
FOLD_RE = re.compile(r"^(predict|expt)_fold(\d+)$")
POINTS_PER_INCH = 72
MAIN_PLOT_WIDTH_PT = 79.8
MAIN_PLOT_HEIGHT_PT = 79.3
COLORBAR_WIDTH_PT = 5.3
COLORBAR_HEIGHT_PT = 43.8
MAIN_PLOT_WIDTH_IN = MAIN_PLOT_WIDTH_PT / POINTS_PER_INCH
MAIN_PLOT_HEIGHT_IN = MAIN_PLOT_HEIGHT_PT / POINTS_PER_INCH
COLORBAR_WIDTH_IN = COLORBAR_WIDTH_PT / POINTS_PER_INCH
COLORBAR_HEIGHT_IN = COLORBAR_HEIGHT_PT / POINTS_PER_INCH
FIG_MARGIN_LEFT_IN = 0.46
FIG_MARGIN_RIGHT_IN = 0.08
FIG_MARGIN_BOTTOM_IN = 0.34
FIG_MARGIN_TOP_IN = 0.18
COLORBAR_GAP_IN = 0.08


def load_pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required to write plots. Install/update the "
            "procap-atlas environment from environment.yml."
        ) from exc
    return plt


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-e",
        "--experiment",
        default="ENCSR882DWM",
        help="experiment accession ID (default: ENCSR882DWM)",
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=DEFAULT_PREDICTIONS_DIR,
        help=f"directory containing saved BPNet prediction NPZs (default: {DEFAULT_PREDICTIONS_DIR})",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"directory for plot output (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=80,
        help="hexbin grid size for density plot (default: 80)",
    )
    parser.add_argument(
        "--scatter",
        action="store_true",
        help="draw raw points instead of a hexbin density plot",
    )
    parser.add_argument(
        "--format",
        action="append",
        default=None,
        choices=["png", "pdf", "svg"],
        help="output format; may be repeated (default: png and pdf)",
    )
    return parser.parse_args()


def fold_keys(npz: np.lib.npyio.NpzFile) -> list[int]:
    predict_folds = set()
    expt_folds = set()

    for key in npz.files:
        match = FOLD_RE.match(key)
        if match is None:
            continue
        prefix, fold = match.groups()
        if prefix == "predict":
            predict_folds.add(int(fold))
        else:
            expt_folds.add(int(fold))

    folds = sorted(predict_folds & expt_folds)
    if not folds:
        raise SystemExit(
            "No matching predict_foldN/expt_foldN arrays found in prediction NPZ."
        )
    return folds


def total_counts(array: np.ndarray, label: str) -> np.ndarray:
    values = np.asarray(array)
    if values.ndim < 2:
        raise SystemExit(f"{label} must have at least 2 dimensions, got shape {values.shape}.")

    counts = values.reshape(values.shape[0], -1).sum(axis=1)
    return np.asarray(counts, dtype=float)


def pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2:
        return float("nan")
    return float(np.corrcoef(x[valid], y[valid])[0, 1])


def load_counts(path: Path) -> tuple[np.ndarray, np.ndarray, list[int]]:
    with np.load(path) as npz:
        folds = fold_keys(npz)
        observed = []
        predicted = []

        for fold in folds:
            pred_key = f"predict_fold{fold}"
            expt_key = f"expt_fold{fold}"
            predicted.append(total_counts(npz[pred_key], pred_key))
            observed.append(total_counts(np.abs(npz[expt_key]), expt_key))

    return np.concatenate(observed), np.concatenate(predicted), folds


def fixed_scatter_figure(plt):
    fig_width = (
        FIG_MARGIN_LEFT_IN
        + MAIN_PLOT_WIDTH_IN
        + COLORBAR_GAP_IN
        + COLORBAR_WIDTH_IN
        + FIG_MARGIN_RIGHT_IN
    )
    fig_height = FIG_MARGIN_BOTTOM_IN + MAIN_PLOT_HEIGHT_IN + FIG_MARGIN_TOP_IN
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = fig.add_axes(
        [
            FIG_MARGIN_LEFT_IN / fig_width,
            FIG_MARGIN_BOTTOM_IN / fig_height,
            MAIN_PLOT_WIDTH_IN / fig_width,
            MAIN_PLOT_HEIGHT_IN / fig_height,
        ]
    )

    colorbar_bottom = FIG_MARGIN_BOTTOM_IN + (
        MAIN_PLOT_HEIGHT_IN - COLORBAR_HEIGHT_IN
    ) / 2
    cax = fig.add_axes(
        [
            (FIG_MARGIN_LEFT_IN + MAIN_PLOT_WIDTH_IN + COLORBAR_GAP_IN)
            / fig_width,
            colorbar_bottom / fig_height,
            COLORBAR_WIDTH_IN / fig_width,
            COLORBAR_HEIGHT_IN / fig_height,
        ]
    )
    return fig, ax, cax


def nice_tick_step(raw_step: float) -> float:
    if raw_step <= 0:
        return 1.0
    exponent = np.floor(np.log10(raw_step))
    fraction = raw_step / (10**exponent)
    if fraction <= 1:
        nice_fraction = 1
    elif fraction <= 2:
        nice_fraction = 2
    elif fraction <= 2.5:
        nice_fraction = 2.5
    elif fraction <= 5:
        nice_fraction = 5
    else:
        nice_fraction = 10
    return nice_fraction * (10**exponent)


def shared_axis_ticks(lims: list[float], target_ticks: int = 5) -> np.ndarray:
    step = nice_tick_step((lims[1] - lims[0]) / max(target_ticks - 1, 1))
    start = np.ceil(lims[0] / step) * step
    stop = np.floor(lims[1] / step) * step
    ticks = np.arange(start, stop + step * 0.5, step)
    ticks = np.round(ticks, 10)
    ticks[np.isclose(ticks, 0)] = 0
    return ticks


def plot_counts(
    observed_counts: np.ndarray,
    predicted_counts: np.ndarray,
    folds: list[int],
    experiment: str,
    out_dir: Path,
    formats: list[str],
    bins: int,
    scatter: bool,
):
    plt = load_pyplot()
    x = np.log1p(observed_counts)
    y = np.log1p(predicted_counts)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) == 0:
        raise SystemExit("No finite log1p observed/predicted count pairs to plot.")
    r = pearsonr(x, y)

    fig, ax, cax = fixed_scatter_figure(plt)
    if scatter:
        ax.scatter(x, y, s=1, alpha=0.12, color="#243B6B", linewidths=0)
        cax.set_visible(False)
    else:
        hb = ax.hexbin(x, y, gridsize=bins, bins="log", mincnt=1, cmap="viridis")
        cbar = fig.colorbar(hb, cax=cax)
        cbar.set_label("log10(n points)")
        cbar.ax.tick_params(labelsize=4.5, length=1.5, width=0.4, pad=1)
        cbar.ax.yaxis.label.set_size(5)
        cbar.outline.set_linewidth(0.4)

    lim_min = float(min(x.min(), y.min()))
    lim_max = float(max(x.max(), y.max()))
    pad = 0.03 * (lim_max - lim_min) if lim_max > lim_min else 0.1
    lims = [lim_min - pad, lim_max + pad]
    ax.plot(lims, lims, color="0.35", linestyle="dashed", linewidth=1, zorder=0)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ticks = shared_axis_ticks(lims)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlabel("log1p observed counts")
    ax.set_ylabel("log1p predicted counts")
    ax.set_title(f"{experiment} BPNet count predictions", fontsize=6, pad=2)
    ax.text(
        0.04,
        0.96,
        f"r = {r:.3f}\nn = {len(x):,}\nfolds = {','.join(map(str, folds))}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=4.8,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1},
    )
    ax.grid(color="0.92", linewidth=0.4)
    ax.tick_params(axis="both", labelsize=5, length=2, width=0.5, pad=1)
    ax.xaxis.label.set_size(5.5)
    ax.yaxis.label.set_size(5.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "scatter" if scatter else "hexbin"
    for fmt in formats:
        out_path = out_dir / f"{experiment}_counts_prediction_{suffix}.{fmt}"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"Wrote {out_path}")
    plt.close(fig)


def main():
    args = parse_args()
    formats = args.format or ["png", "pdf"]
    pred_path = args.predictions_dir / f"{args.experiment}.npz"
    if not pred_path.exists():
        raise SystemExit(
            f"Prediction NPZ not found: {pred_path}\n"
            "Create it with: python src/bpnet/benchmark/benchmark_bpnet.py "
            f"-e {args.experiment} --save-output"
        )

    observed_counts, predicted_counts, folds = load_counts(pred_path)
    plot_counts(
        observed_counts,
        predicted_counts,
        folds,
        args.experiment,
        args.out_dir,
        formats,
        args.bins,
        args.scatter,
    )


if __name__ == "__main__":
    main()
