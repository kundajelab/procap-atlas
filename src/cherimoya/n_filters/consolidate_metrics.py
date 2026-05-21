#!/usr/bin/env python3
"""Consolidate and plot Cherimoya n_filters sweep benchmark metrics."""

import argparse
import json
import re
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_METRICS_DIR = REPO_ROOT / "performance_metrics" / "cherimoya"
DEFAULT_OUT_DIR = REPO_ROOT / "performance_metrics" / "cherimoya_n_filters"
DEFAULT_PLOT_DIR = DEFAULT_OUT_DIR / "plots"
N_READS_PATH = REPO_ROOT / "configs" / "n_reads.txt"
SWEEP_N_FILTERS = [16, 24, 36, 48, 64, 96, 196, 256]
BASELINE_N_FILTERS = 128
PLOT_N_FILTERS = [16, 24, 36, 48, 64, 96, 128, 196, 256]
METRICS = [
    "profile_pearson",
    "profile_jsd",
    "log_counts_pearson",
    "counts_spearman",
]
METRIC_LABELS = {
    "profile_pearson": "Profile Pearson",
    "profile_jsd": "Profile JSD",
    "log_counts_pearson": "Log-counts Pearson",
    "counts_spearman": "Counts Spearman",
}
METRIC_RE = re.compile(r"^(?P<experiment>ENCSR[0-9A-Z]+)_nf(?P<n_filters>\d+)\.json$")


def load_pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required to write n_filters sweep plots. "
            "Install/update the procap-atlas environment from environment.yml."
        ) from exc
    return plt


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=DEFAULT_METRICS_DIR,
        help=f"directory containing benchmark JSON files (default: {DEFAULT_METRICS_DIR})",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"directory for consolidated TSV output (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=DEFAULT_PLOT_DIR,
        help=f"directory for plot output (default: {DEFAULT_PLOT_DIR})",
    )
    parser.add_argument(
        "--use-genome-wide",
        action="store_true",
        help="plot genome-wide metrics instead of per-fold averages",
    )
    return parser.parse_args()


def fold_average(data: dict, metric: str) -> float:
    folds = list(data["per_fold"].values())
    return sum(fold[metric] for fold in folds) / len(folds)


def load_rows(metrics_dir: Path) -> list[dict]:
    rows = []
    valid_filters = set(SWEEP_N_FILTERS)

    for path in sorted(metrics_dir.glob("*_nf*.json")):
        match = METRIC_RE.match(path.name)
        if match is None:
            continue

        n_filters = int(match.group("n_filters"))
        if n_filters not in valid_filters:
            continue

        with open(path) as f:
            data = json.load(f)

        row = {
            "experiment": data["experiment"],
            "biosample": data["biosample"],
            "n_filters": n_filters,
            "model_source": "sweep",
            "model_dir": data["model_dir"],
            "metrics_json": str(path),
        }
        for metric in METRICS:
            row[metric] = fold_average(data, metric)
            row[f"genome_wide_{metric}"] = data["genome_wide"][metric]

        rows.append(row)

    return rows


def load_baseline_rows(metrics_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(metrics_dir.glob("ENCSR*.json")):
        if METRIC_RE.match(path.name) is not None:
            continue

        with open(path) as f:
            data = json.load(f)

        row = {
            "experiment": data["experiment"],
            "biosample": data["biosample"],
            "n_filters": BASELINE_N_FILTERS,
            "model_source": "baseline",
            "model_dir": data["model_dir"],
            "metrics_json": str(path),
        }
        for metric in METRICS:
            row[metric] = fold_average(data, metric)
            row[f"genome_wide_{metric}"] = data["genome_wide"][metric]

        rows.append(row)

    return rows


def merge_read_counts(df: pd.DataFrame) -> pd.DataFrame:
    n_reads = pd.read_csv(N_READS_PATH, sep="\t")
    read_cols = ["experiment", "biosample", "pl_reads", "mn_reads", "total_reads"]
    return n_reads[read_cols].merge(
        df.drop(columns="biosample"), on="experiment", how="inner"
    )


def plot_metric(df: pd.DataFrame, metric: str, path_prefix: Path):
    plt = load_pyplot()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for _, group in df.sort_values("n_filters").groupby("experiment"):
        if len(group) < 2:
            continue
        ax.plot(
            group["n_filters"],
            group[metric],
            color="0.65",
            alpha=0.25,
            linewidth=0.8,
        )

    median = df.groupby("n_filters", as_index=False)[metric].median()
    ax.plot(
        median["n_filters"],
        median[metric],
        color="black",
        marker="o",
        linewidth=2,
        label="Median",
    )

    ax.set_xscale("log", base=2)
    ax.set_xticks(PLOT_N_FILTERS)
    ax.set_xticklabels([str(n) for n in PLOT_N_FILTERS])
    ax.set_xlabel("n_filters")
    ax.set_ylabel(METRIC_LABELS[metric])
    ax.set_title(f"Cherimoya n_filters sweep: {METRIC_LABELS[metric]}")
    ax.grid(axis="y", color="0.9", linewidth=0.8)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path_prefix.with_suffix(".png"), dpi=200)
    fig.savefig(path_prefix.with_suffix(".pdf"))
    plt.close(fig)


def plot_summary(df: pd.DataFrame, metrics: list[str], path_prefix: Path):
    plt = load_pyplot()

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    axes = axes.flatten()

    for ax, metric in zip(axes, metrics):
        for _, group in df.sort_values("n_filters").groupby("experiment"):
            if len(group) < 2:
                continue
            ax.plot(
                group["n_filters"],
                group[metric],
                color="0.65",
                alpha=0.2,
                linewidth=0.7,
            )

        median = df.groupby("n_filters", as_index=False)[metric].median()
        ax.plot(
            median["n_filters"],
            median[metric],
            color="black",
            marker="o",
            linewidth=2,
        )

        ax.set_xscale("log", base=2)
        ax.set_xticks(PLOT_N_FILTERS)
        ax.set_xticklabels([str(n) for n in PLOT_N_FILTERS])
        ax.set_title(METRIC_LABELS[metric])
        ax.grid(axis="y", color="0.9", linewidth=0.8)

    for ax in axes[2:]:
        ax.set_xlabel("n_filters")

    fig.suptitle("Cherimoya n_filters sweep test metrics", y=0.995)
    fig.tight_layout()
    fig.savefig(path_prefix.with_suffix(".png"), dpi=200)
    fig.savefig(path_prefix.with_suffix(".pdf"))
    plt.close(fig)


def make_plots(df: pd.DataFrame, plot_dir: Path, use_genome_wide: bool):
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_df = df.copy()
    metric_names = METRICS

    if use_genome_wide:
        for metric in METRICS:
            plot_df[metric] = plot_df[f"genome_wide_{metric}"]

    suffix = "genome_wide" if use_genome_wide else "per_fold_average"
    for metric in metric_names:
        plot_metric(plot_df, metric, plot_dir / f"{metric}_{suffix}")
    plot_summary(plot_df, metric_names, plot_dir / f"summary_{suffix}")


def main():
    args = parse_args()
    rows = load_rows(args.metrics_dir) + load_baseline_rows(args.metrics_dir)
    if not rows:
        raise SystemExit(
            f"No n_filters sweep or baseline metrics found in {args.metrics_dir}. "
            "Expected files like ENCSR882DWM_nf64.json or baseline ENCSR882DWM.json."
        )

    df = pd.DataFrame(rows)
    df = merge_read_counts(df)
    df = df.sort_values(["experiment", "n_filters"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "procap-atlas_n_filters_metrics.tsv"
    df.to_csv(out_path, sep="\t", index=False)

    make_plots(df, args.plot_dir, args.use_genome_wide)
    print(f"Wrote {len(df)} model metrics to {out_path}")
    print(f"Wrote plots to {args.plot_dir}")


if __name__ == "__main__":
    main()
