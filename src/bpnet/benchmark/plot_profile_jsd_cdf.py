#!/usr/bin/env python3
"""Plot a profile JSD CDF for one BPNet experiment.

The input NPZ is produced by:
    python src/bpnet/benchmark/benchmark_bpnet.py -e ENCSR882DWM --save-output

Examples:
    python src/bpnet/benchmark/plot_profile_jsd_cdf.py -e ENCSR882DWM
    python src/bpnet/benchmark/plot_profile_jsd_cdf.py -e ENCSR882DWM --bounds-npz performance_metrics/bpnet_bounds/per_locus/ENCSR882DWM.npz
    python src/bpnet/benchmark/plot_profile_jsd_cdf.py -e ENCSR882DWM --format png
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_PREDICTIONS_DIR = REPO_ROOT / "predictions" / "bpnet"
DEFAULT_OUT_DIR = REPO_ROOT / "plots" / "bpnet"
FOLD_RE = re.compile(r"^(predict|expt)_fold(\d+)$")
POINTS_PER_INCH = 72
# Wider than the scatter panel because the CDF has no color legend.
MAIN_PLOT_WIDTH_PT = 95.0
MAIN_PLOT_HEIGHT_PT = 79.3
MAIN_PLOT_WIDTH_IN = MAIN_PLOT_WIDTH_PT / POINTS_PER_INCH
MAIN_PLOT_HEIGHT_IN = MAIN_PLOT_HEIGHT_PT / POINTS_PER_INCH
FIG_MARGIN_LEFT_IN = 0.46
FIG_MARGIN_RIGHT_IN = 0.08
FIG_MARGIN_BOTTOM_IN = 0.34
FIG_MARGIN_TOP_IN = 0.18


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-e",
        "--experiment",
        required=True,
        help="experiment accession ID (e.g. ENCSR882DWM)",
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
        "--bounds-npz",
        type=Path,
        default=None,
        help="optional per-locus bounds NPZ from generate_profile_jsd_bounds.py",
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


def normalize_profiles(values: np.ndarray, label: str) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    if values.ndim < 2:
        raise SystemExit(f"{label} must have at least 2 dimensions, got shape {values.shape}.")

    flat = values.reshape(values.shape[0], -1)
    totals = flat.sum(axis=1, keepdims=True)
    valid = np.isfinite(flat).all(axis=1) & np.isfinite(totals[:, 0]) & (totals[:, 0] > 0)
    normalized = np.zeros_like(flat, dtype=float)
    normalized[valid] = flat[valid] / totals[valid]
    return normalized, valid


def jensen_shannon_distance(predicted: np.ndarray, observed: np.ndarray) -> np.ndarray:
    pred, pred_valid = normalize_profiles(predicted, "predicted profiles")
    obs, obs_valid = normalize_profiles(np.abs(observed), "observed profiles")
    valid = pred_valid & obs_valid

    jsd = np.full(pred.shape[0], np.nan, dtype=float)
    if not valid.any():
        return jsd

    p = pred[valid]
    q = obs[valid]
    m = 0.5 * (p + q)

    with np.errstate(divide="ignore", invalid="ignore"):
        p_kl = np.where(p > 0, p * np.log(p / m), 0.0).sum(axis=1)
        q_kl = np.where(q > 0, q * np.log(q / m), 0.0).sum(axis=1)

    jsd[valid] = np.sqrt(0.5 * (p_kl + q_kl))
    return jsd


def load_profile_jsd(path: Path) -> tuple[np.ndarray, list[int]]:
    jsds = []
    with np.load(path) as npz:
        folds = fold_keys(npz)
        for fold in folds:
            jsds.append(
                jensen_shannon_distance(
                    npz[f"predict_fold{fold}"],
                    npz[f"expt_fold{fold}"],
                )
            )

    values = np.concatenate(jsds)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        raise SystemExit("No finite profile JSD values found in prediction NPZ.")
    return values, folds


def load_bound_values(path: Path) -> dict[str, np.ndarray]:
    arrays = {
        "replicate": [],
        "average": [],
    }
    with np.load(path) as npz:
        for key in npz.files:
            if key.startswith("replicate_js_distance_fold"):
                arrays["replicate"].append(npz[key])
            elif key.startswith("average_profile_js_distance_fold"):
                arrays["average"].append(npz[key])

    values = {}
    for label, parts in arrays.items():
        if not parts:
            continue
        combined = np.concatenate(parts)
        values[label] = combined[np.isfinite(combined)]
    return values


def draw_cdf(ax, values: np.ndarray, **kwargs):
    sorted_values = np.sort(values)
    cumulative = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
    ax.plot(sorted_values, cumulative, **kwargs)


def fixed_cdf_figure(plt):
    fig_width = FIG_MARGIN_LEFT_IN + MAIN_PLOT_WIDTH_IN + FIG_MARGIN_RIGHT_IN
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
    return fig, ax


def plot_cdf(
    values: np.ndarray,
    folds: list[int],
    experiment: str,
    out_dir: Path,
    formats: list[str],
    bounds_npz: Path | None,
):
    sorted_values = np.sort(values)
    median = float(np.median(sorted_values))

    fig, ax = fixed_cdf_figure(plt)
    draw_cdf(ax, values, color="#243B6B", linewidth=0.8, label="Model")
    ax.axvline(median, color="black", linestyle="dashed", linewidth=0.55)

    if bounds_npz is not None:
        if not bounds_npz.exists():
            raise SystemExit(f"bounds NPZ not found: {bounds_npz}")
        bounds = load_bound_values(bounds_npz)
        if "replicate" in bounds:
            draw_cdf(
                ax,
                bounds["replicate"],
                color="#25824F",
                linewidth=0.65,
                label="Replicate",
            )
        if "average" in bounds:
            draw_cdf(
                ax,
                bounds["average"],
                color="#B86B00",
                linewidth=0.65,
                label="Average profile",
            )

    ax.set_xlabel("Profile JSD")
    ax.set_ylabel("Cumulative fraction")
    ax.set_title(f"{experiment} BPNet profile JSD CDF", fontsize=6, pad=2)
    ax.set_ylim(0, 1)
    ax.grid(color="0.9", linewidth=0.4)
    ax.text(
        0.04,
        0.96,
        f"n = {len(sorted_values):,}\nmedian = {median:.3f}\nfolds = {','.join(map(str, folds))}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=4.8,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1},
    )
    ax.legend(frameon=False, fontsize=4.8, handlelength=1.2, loc="lower right")
    ax.tick_params(axis="both", labelsize=5, length=2, width=0.5, pad=1)
    ax.xaxis.label.set_size(5.5)
    ax.yaxis.label.set_size(5.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        out_path = out_dir / f"{experiment}_profile_jsd_cdf.{fmt}"
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

    values, folds = load_profile_jsd(pred_path)
    plot_cdf(values, folds, args.experiment, args.out_dir, formats, args.bounds_npz)


if __name__ == "__main__":
    main()
