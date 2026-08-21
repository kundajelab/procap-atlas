#!/usr/bin/env python3
"""
Compare Cherimoya benchmark metrics across archived model versions.

Cherimoya has been retrained multiple times as the upstream package and this
repo's training script evolved. Per-experiment benchmark JSONs (and
consolidated TSVs) from earlier runs are archived under
performance_metrics/cherimoya/{version}/, alongside the current run at the
top level of performance_metrics/cherimoya/ -- see src/cherimoya/README.md's
Historical Notes for what each version is.

For every pair of versions and every shared metric, produces one figure with
a scatterplot (older vs. newer, one point per experiment, colored by read
depth, with a y=x reference line and a Wilcoxon signed-rank test) and a
histogram of per-experiment deltas (newer minus older).

Usage:
    python src/analysis/compare_cherimoya_versions.py
    python src/analysis/compare_cherimoya_versions.py --metrics profile_jsd
    python src/analysis/compare_cherimoya_versions.py --min-reads 10000000
"""

import argparse
import itertools
import sys
from pathlib import Path

import pandas as pd

from _metric_comparison_plots import plot_metric_comparison

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CHERIMOYA_METRICS_DIR = REPO_ROOT / "performance_metrics" / "cherimoya"
DEFAULT_OUT_DIR = REPO_ROOT / "plots" / "cherimoya_versions"

# Oldest to newest; see src/cherimoya/README.md's Historical Notes. The
# current run lives at the top level of CHERIMOYA_METRICS_DIR, not a
# version-named subdirectory.
DEFAULT_VERSIONS = {
    "v0.0.1": CHERIMOYA_METRICS_DIR / "v0.0.1" / "procap-atlas_performance_metrics.tsv",
    "c0cbabe26c": (
        CHERIMOYA_METRICS_DIR
        / "c0cbabe26cabfb5012f4fc5328af832e32f9ed04"
        / "procap-atlas_performance_metrics.tsv"
    ),
    "v0.2.0": CHERIMOYA_METRICS_DIR / "procap-atlas_performance_metrics.tsv",
}

# lower_is_better controls delta sign convention (newer - older) and which
# side of zero/the diagonal counts as an improvement.
METRIC_INFO = {
    "profile_pearson": {"label": "profile Pearson", "lower_is_better": False},
    "profile_jsd": {"label": "profile JSD", "lower_is_better": True},
    "log_counts_pearson": {"label": "log-counts Pearson", "lower_is_better": False},
    "counts_spearman": {"label": "counts Spearman", "lower_is_better": False},
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"directory for figure output (default: {DEFAULT_OUT_DIR.relative_to(REPO_ROOT)})",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=sorted(METRIC_INFO),
        default=sorted(METRIC_INFO),
        help="metrics to compare (default: all)",
    )
    parser.add_argument(
        "--min-reads",
        type=float,
        default=0,
        help="skip experiments with fewer than N total reads (default: 0)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    versions = {}
    for name, path in DEFAULT_VERSIONS.items():
        if not path.exists():
            print(f"WARNING: {name} metrics TSV not found, skipping: {path}", file=sys.stderr)
            continue
        versions[name] = pd.read_csv(path, sep="\t")

    if len(versions) < 2:
        print("ERROR: fewer than 2 version TSVs found", file=sys.stderr)
        sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for name_a, name_b in itertools.combinations(versions, 2):
        df_a, df_b = versions[name_a], versions[name_b]

        metrics = [m for m in args.metrics if m in df_a.columns and m in df_b.columns]
        missing = sorted(set(args.metrics) - set(metrics))
        if missing:
            print(
                f"WARNING: {name_a} vs {name_b}: metrics not present in both, "
                f"skipping: {missing}",
                file=sys.stderr,
            )
        if not metrics:
            continue

        # Inner join: keeps only experiments benchmarked for both versions
        cols = ["experiment", "biosample", "total_reads", *metrics]
        df = df_a[cols].merge(
            df_b[["experiment", *metrics]],
            on="experiment",
            suffixes=(f"_{name_a}", f"_{name_b}"),
        )
        df = df[df["total_reads"] >= args.min_reads]

        if len(df) < 2:
            print(
                f"WARNING: {name_a} vs {name_b}: fewer than 2 shared experiments "
                f"after filtering ({len(df)}), skipping",
                file=sys.stderr,
            )
            continue

        print(f"{name_a} vs {name_b}: {len(df)} shared experiments", file=sys.stderr)
        for metric in metrics:
            info = METRIC_INFO[metric]
            out_path = args.out_dir / f"{name_a}_vs_{name_b}_{metric}.pdf"
            median_delta, mean_delta = plot_metric_comparison(
                df,
                col_a=f"{metric}_{name_a}",
                col_b=f"{metric}_{name_b}",
                label_a=name_a,
                label_b=name_b,
                metric_label=info["label"],
                lower_is_better=info["lower_is_better"],
                read_depth_col="total_reads",
                out_path=out_path,
            )
            print(
                f"  {info['label']}: median delta={median_delta:.4f}, "
                f"mean delta={mean_delta:.4f}",
                file=sys.stderr,
            )
            print(f"  Saved {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
