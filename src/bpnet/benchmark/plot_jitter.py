#!/usr/bin/env python3
"""Plot fold-averaged BPNet benchmark metrics across experiments.

Examples:
    python src/bpnet/benchmark/plot_jitter.py
    python src/bpnet/benchmark/plot_jitter.py --metric profile_jsd
    python src/bpnet/benchmark/plot_jitter.py --metric profile_jsd --sqrt-values
    python src/bpnet/benchmark/plot_jitter.py --min-reads 10000000
    python src/bpnet/benchmark/plot_jitter.py --use-genome-wide
"""

import argparse
import csv
import fnmatch
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_METRICS_DIR = REPO_ROOT / "performance_metrics" / "bpnet"
DEFAULT_OUT_DIR = REPO_ROOT / "plots" / "bpnet"
DEFAULT_BOUNDS_TSV = (
    REPO_ROOT / "performance_metrics" / "bpnet_bounds" / "procap-atlas_profile_jsd_bounds.tsv"
)
N_READS_PATH = REPO_ROOT / "configs" / "n_reads.txt"
BASELINE_RE = re.compile(r"^ENCSR[0-9A-Z]+\.json$")
POINTS_PER_INCH = 72
MAIN_PLOT_WIDTH_PT = 98.9
MAIN_PLOT_HEIGHT_PT = 26.5
MAIN_PLOT_WIDTH_IN = MAIN_PLOT_WIDTH_PT / POINTS_PER_INCH
MAIN_PLOT_HEIGHT_IN = MAIN_PLOT_HEIGHT_PT / POINTS_PER_INCH
FIG_MARGIN_LEFT_IN = 0.42
FIG_MARGIN_RIGHT_IN = 0.06
FIG_MARGIN_BOTTOM_IN = 0.26
FIG_MARGIN_TOP_IN = 0.22
METRIC_INFO = {
    "log_counts_pearson": {
        "label": "Log-counts Pearson r",
        "title": "log-counts Pearson",
        "stem": "log_counts_pearson",
    },
    "profile_jsd": {
        "label": "Profile Jensen-Shannon divergence",
        "title": "profile Jensen-Shannon divergence",
        "stem": "profile_jsd",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=DEFAULT_METRICS_DIR,
        help=f"directory containing BPNet benchmark JSON files (default: {DEFAULT_METRICS_DIR})",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"directory for plot output (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--metric",
        choices=sorted(METRIC_INFO),
        default="log_counts_pearson",
        help="benchmark metric to plot (default: log_counts_pearson)",
    )
    parser.add_argument(
        "--sqrt-values",
        action="store_true",
        help=(
            "plot the square root of metric values. Useful for plotting "
            "profile_jsd on Jensen-Shannon distance scale."
        ),
    )
    parser.add_argument(
        "--bounds-tsv",
        type=Path,
        nargs="?",
        const=DEFAULT_BOUNDS_TSV,
        default=None,
        help=(
            "optional profile JSD bounds TSV from generate_profile_jsd_bounds.py "
            f"(default if no path is supplied: {DEFAULT_BOUNDS_TSV})"
        ),
    )
    parser.add_argument(
        "--min-reads",
        type=float,
        default=0,
        help="skip experiments with fewer total reads than this (default: 0, disabled)",
    )
    parser.add_argument(
        "--use-genome-wide",
        action="store_true",
        help="plot the genome-wide value per experiment instead of fold averages",
    )
    parser.add_argument(
        "--include-pattern",
        action="append",
        default=None,
        metavar="GLOB",
        help=(
            "include metric JSON files matching this filename glob; may be repeated. "
            "Default: baseline ENCSR*.json files only"
        ),
    )
    parser.add_argument(
        "--format",
        action="append",
        default=None,
        choices=["png", "pdf", "svg"],
        help="output format; may be repeated (default: png and pdf)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="random seed for y jitter (default: 1)",
    )
    return parser.parse_args()


def iter_metric_paths(metrics_dir: Path, include_patterns: list[str] | None):
    for path in sorted(metrics_dir.glob("*.json")):
        if include_patterns is None:
            if BASELINE_RE.match(path.name):
                yield path
            continue

        if any(fnmatch.fnmatch(path.name, pattern) for pattern in include_patterns):
            yield path


def load_rows(
    metrics_dir: Path,
    include_patterns: list[str] | None,
    min_reads: float,
    metric: str,
) -> list[dict]:
    read_counts = {}
    if N_READS_PATH.exists():
        with open(N_READS_PATH) as f:
            header = next(f).rstrip("\n").split("\t")
            try:
                exp_idx = header.index("experiment")
                reads_idx = header.index("total_reads")
            except ValueError:
                exp_idx = None
                reads_idx = None

            if exp_idx is not None and reads_idx is not None:
                for line in f:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) <= max(exp_idx, reads_idx):
                        continue
                    try:
                        read_counts[parts[exp_idx]] = float(parts[reads_idx])
                    except ValueError:
                        continue
    rows = []

    for path in iter_metric_paths(metrics_dir, include_patterns):
        with open(path) as f:
            data = json.load(f)

        exp_id = data["experiment"]
        total_reads = read_counts.get(exp_id)
        if min_reads > 0 and (total_reads is None or total_reads < min_reads):
            continue

        common = {
            "experiment": exp_id,
            "biosample": data.get("biosample", ""),
            "metrics_json": str(path),
            "total_reads": total_reads,
        }

        genome_wide = data.get("genome_wide", {})
        if metric in genome_wide:
            rows.append(
                {
                    **common,
                    "fold": "genome_wide",
                    "value": float(genome_wide[metric]),
                    "value_type": "genome_wide",
                }
            )

        for fold, fold_metrics in data.get("per_fold", {}).items():
            if metric not in fold_metrics:
                continue
            rows.append(
                {
                    **common,
                    "fold": str(fold),
                    "value": float(fold_metrics[metric]),
                    "value_type": "per_fold",
                }
            )

    return rows


def summarize(rows: list[dict]) -> dict[str, float]:
    values = np.array([row["value"] for row in rows], dtype=float)
    return {
        "median": float(np.nanmedian(values)),
        "q25": float(np.nanpercentile(values, 25)),
        "q75": float(np.nanpercentile(values, 75)),
    }


def average_per_fold_rows(rows: list[dict]) -> list[dict]:
    averaged = []
    per_fold_rows = [row for row in rows if row["value_type"] == "per_fold"]

    for exp_id in sorted({row["experiment"] for row in per_fold_rows}):
        exp_rows = [row for row in per_fold_rows if row["experiment"] == exp_id]
        values = np.array([row["value"] for row in exp_rows], dtype=float)
        first = exp_rows[0]
        averaged.append(
            {
                **first,
                "fold": "mean",
                "value": float(np.nanmean(values)),
                "n_folds": len(exp_rows),
                "value_type": "fold_average",
            }
        )

    return averaged


def sqrt_transform_rows(rows: list[dict]) -> list[dict]:
    values = np.array([row["value"] for row in rows], dtype=float)
    if np.any(values < 0):
        raise SystemExit("--sqrt-values cannot be used when any plotted value is negative.")

    transformed = []
    for row in rows:
        transformed.append({**row, "value": float(np.sqrt(row["value"]))})
    return transformed


def display_metric_info(metric: str, sqrt_values: bool) -> dict[str, str]:
    metric_info = dict(METRIC_INFO[metric])
    if not sqrt_values:
        return metric_info

    if metric == "profile_jsd":
        metric_info["label"] = "Profile Jensen-Shannon distance"
        metric_info["title"] = "profile Jensen-Shannon distance"
    else:
        metric_info["label"] = f"sqrt({metric_info['label']})"
        metric_info["title"] = f"sqrt {metric_info['title']}"
    metric_info["stem"] = f"{metric_info['stem']}_sqrt"
    return metric_info


def load_bounds_rows(
    bounds_tsv: Path,
    use_genome_wide: bool,
    sqrt_values: bool,
) -> dict[str, list[dict]]:
    suffix = "genome_wide" if use_genome_wide else "fold_average"
    value_cols = {
        "replicate": f"replicate_profile_jsd_{suffix}",
        "average": f"average_profile_jsd_{suffix}",
    }
    values = {"replicate": [], "average": []}
    with open(bounds_tsv, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            for label, col in value_cols.items():
                raw = row.get(col)
                if raw in (None, ""):
                    continue
                value = float(raw)
                if sqrt_values:
                    value = float(np.sqrt(value))
                values[label].append(
                    {
                        "experiment": row["experiment"],
                        "value": value,
                    }
                )
    return values


def fixed_main_plot_figure(plt):
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


def plot_jitter(
    rows: list[dict],
    out_dir: Path,
    formats: list[str],
    use_genome_wide: bool,
    seed: int,
    metric: str,
    sqrt_values: bool,
    bounds_tsv: Path | None,
):
    metric_info = display_metric_info(metric, sqrt_values)
    value_label = "genome-wide" if use_genome_wide else "fold-averaged"
    if use_genome_wide:
        plot_rows = [row for row in rows if row["value_type"] == "genome_wide"]
    else:
        plot_rows = average_per_fold_rows(rows)
    if not plot_rows:
        raise SystemExit(f"No {value_label} {metric_info['title']} values found.")
    if sqrt_values:
        plot_rows = sqrt_transform_rows(plot_rows)

    rng = np.random.default_rng(seed)
    x = np.array([row["value"] for row in plot_rows], dtype=float)
    y = rng.uniform(-0.25, 0.25, size=len(plot_rows))

    fig, ax = fixed_main_plot_figure(plt)

    ax.boxplot(
        x,
        vert=False,
        positions=[0],
        widths=0.42,
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": "#DCE6F2", "edgecolor": "0.35", "linewidth": 0.6},
        medianprops={"color": "black", "linewidth": 0.8},
        whiskerprops={"color": "0.45", "linewidth": 0.6},
        capprops={"color": "0.45", "linewidth": 0.6},
        zorder=0,
    )

    ax.scatter(
        x,
        y,
        s=5,
        color="#243B6B",
        alpha=0.45,
        linewidths=0,
    )

    stats = summarize(plot_rows)
    ax.axvline(stats["median"], color="black", linewidth=0.7, label="Median")
    ax.axvline(stats["q25"], color="0.45", linestyle="dashed", linewidth=0.45, label="IQR")
    ax.axvline(stats["q75"], color="0.45", linestyle="dashed", linewidth=0.45)

    if bounds_tsv is not None:
        if metric != "profile_jsd":
            raise SystemExit("--bounds-tsv is only supported with --metric profile_jsd.")
        if not bounds_tsv.exists():
            raise SystemExit(f"bounds TSV not found: {bounds_tsv}")
        bounds = load_bounds_rows(bounds_tsv, use_genome_wide, sqrt_values)
        bound_specs = [
            ("replicate", -0.41, "#25824F", "Replicate"),
            ("average", 0.41, "#B86B00", "Average profile"),
        ]
        for key, y_pos, color, label in bound_specs:
            bound_values = np.array([row["value"] for row in bounds[key]], dtype=float)
            if len(bound_values) == 0:
                continue
            ax.scatter(
                bound_values,
                np.full(len(bound_values), y_pos),
                marker="|",
                s=10,
                color=color,
                alpha=0.65,
                linewidths=0.5,
                label=label,
            )
            ax.axvline(
                np.nanmedian(bound_values),
                color=color,
                linewidth=0.55,
                alpha=0.75,
            )

    ax.set_xlabel(metric_info["label"])
    ax.set_ylabel("")
    title_suffix = value_label if use_genome_wide else "fold-averaged across model folds"
    ax.set_title(f"BPNet {metric_info['title']} ({title_suffix})", fontsize=6, pad=2)
    ax.grid(axis="x", color="0.9", linewidth=0.4)
    ax.set_ylim(-0.55, 0.55)
    ax.set_yticks([])
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.tick_params(axis="x", labelsize=5, length=2, width=0.5, pad=1)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.label.set_size(5.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    ax.text(
        0.02,
        0.98,
        f"experiments={len(plot_rows)}\nmedian={stats['median']:.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=4.8,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1},
    )
    ax.legend(frameon=False, loc="lower right", fontsize=4.8, handlelength=1.2)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{metric_info['stem']}_genome_wide" if use_genome_wide else f"{metric_info['stem']}_jitter"
    for fmt in formats:
        out_path = out_dir / f"{stem}.{fmt}"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"Wrote {out_path}")
    plt.close(fig)


def main():
    args = parse_args()
    formats = args.format or ["png", "pdf"]
    rows = load_rows(args.metrics_dir, args.include_pattern, args.min_reads, args.metric)
    if not rows:
        raise SystemExit(
            f"No BPNet metric JSON files found in {args.metrics_dir}. "
            "Expected baseline files like ENCSR882DWM.json."
        )
    plot_jitter(
        rows,
        args.out_dir,
        formats,
        args.use_genome_wide,
        args.seed,
        args.metric,
        args.sqrt_values,
        args.bounds_tsv,
    )


if __name__ == "__main__":
    main()
