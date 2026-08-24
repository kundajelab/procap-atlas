"""Generate a Fi-NeMo hit-calling report and filter hits by CWM similarity.

Runs `finemo report` on an existing `call_hits_bpnet.py` output directory and
uses its per-motif `cwm_similarity` metric -- the normalized dot product
between the CWM built from Fi-NeMo's own called hits and the reference motif
CWM -- as a quality gate. This follows the same principle as the Human
Development Multiomic Atlas fetal-atlas paper's hit-calling QC
(https://github.com/GreenleafLab/HDMA/blob/main/code/03-chrombpnet/02-compendium/06b-reconcile_hits.py),
which drops all hits for any motif whose hit-derived CWM correlates poorly
with the reference CWM (they used a 0.9 threshold on their `cwm_correlation`,
the equivalent metric in the older Fi-NeMo release they used).

`cwm_similarity` is computed from `regions.npz` + `hits.tsv` + the motif h5's
own CWMs, independent of TF-MoDISco seqlets, so `--no-recall` is always used
here: the shared MotifCompendium cluster-average motif h5
(`motifcompendium_{head}_cluster_averages.h5`) only stores averaged
`contrib_scores` per cluster, not seqlets -- unlike HDMA's own merged motif
h5, which is a real TF-MoDISco-lite pattern collapse (via
`modiscolite.aggregator.SimilarPatternsCollapser`) that keeps seqlets -- so
seqlet-recall metrics aren't available to us here. Even if MotifCompendium's
data model retained seqlets (it discards them on load, keeping only an
averaged CWM per source pattern), pooling seqlets across experiments into one
shared cluster would make per-experiment recall structurally biased low for
any motif mostly discovered in other experiments' peaks. cwm_similarity has
no such issue, since it only compares against this experiment's own called
hits.

Note: if call_hits_bpnet.py was run with --cwm-trim-thresholds/
--cwm-trim-coords overrides (e.g. from compute_trim_floor.py), `finemo
report` -- which only exposes a single global --cwm-trim-threshold, no
per-motif override -- can't exactly reproduce that trimming, so
cwm_similarity for those specific motifs may be computed against a slightly
different template width than was actually used to call hits.

`finemo report`'s own report.html visualizes the pre-filter hit set (from
`hits_unique.tsv`) only. This script additionally plots the cwm_similarity
distribution with the threshold marked, and re-runs Fi-NeMo's own
hit-stat/peak-distribution/co-occurrence plotting functions directly on the
post-filter hit set (`hits_filtered.tsv`), so the two are visually comparable
side by side rather than only having a pre-filter report and a plain
post-filter TSV.

Usage:
    python src/bpnet/hitcall/report_bpnet.py -e ENCSR882DWM
    python src/bpnet/hitcall/report_bpnet.py -e ENCSR882DWM --head count
    python src/bpnet/hitcall/report_bpnet.py -e ENCSR882DWM --cwm-similarity-threshold 0.85
"""

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl
from finemo.evaluation import get_motif_occurences
from finemo.visualization import (
    plot_hit_peak_distributions,
    plot_hit_stat_distributions,
    plot_peak_motif_indicator_heatmap,
)

from call_hits_bpnet import DEFAULT_CWM_TRIM_THRESHOLD, trim_suffix

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def run(cmd, verbose):
    if verbose:
        print(" ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True)


def plot_similarity_distribution(motif_report, threshold, out_path):
    """Histogram of per-motif cwm_similarity with the drop threshold marked."""
    similarities = motif_report["cwm_similarity"].to_numpy()
    n_dropped = int((similarities <= threshold).sum())

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(similarities, bins=30)
    ax.axvline(threshold, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("cwm_similarity")
    ax.set_ylabel("Number of motifs")
    ax.set_title(
        f"{n_dropped}/{motif_report.height} motifs at or below threshold "
        f"({threshold})"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_hit_comparison(hits, motif_names, out_dir):
    """Re-run Fi-NeMo's own hit-stat/peak-distribution/co-occurrence plots on
    an arbitrary hits DataFrame (used for both the pre- and post-filter sets),
    so the two are visually comparable side by side.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    hits_lazy = hits.lazy()
    occ_df, coocc = get_motif_occurences(hits_lazy, motif_names)
    plot_hit_stat_distributions(hits_lazy, motif_names, str(out_dir))
    plot_hit_peak_distributions(occ_df, motif_names, str(out_dir))
    plot_peak_motif_indicator_heatmap(coocc, motif_names, str(out_dir))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-e",
        "--experiment",
        type=str,
        required=True,
        help="experiment accession ID (e.g. ENCSR882DWM)",
    )
    parser.add_argument(
        "-m",
        "--model-dir",
        type=str,
        default=None,
        help="override model directory, default derived from config",
    )
    parser.add_argument(
        "--head",
        type=str,
        default="profile",
        choices=["profile", "count"],
        help="attribution/motif head the hits were called against (default: profile)",
    )
    parser.add_argument(
        "--modisco-h5",
        type=str,
        default=None,
        help=(
            "motif CWMs the hits were called against; must match what "
            "call_hits_bpnet.py used. Default is the shared MotifCompendium "
            "cluster-average file for --head, motifcompendium/bpnet/"
            "motifcompendium_{head}_cluster_averages.h5"
        ),
    )
    parser.add_argument(
        "--cwm-trim-threshold",
        type=float,
        default=DEFAULT_CWM_TRIM_THRESHOLD,
        help=(
            "motif trim threshold; should match call_hits_bpnet.py's "
            "--cwm-trim-threshold (default: 0.3)"
        ),
    )
    parser.add_argument(
        "--cwm-trim-thresholds",
        type=str,
        default=None,
        help=(
            "path to the same -T/--cwm-trim-thresholds file call_hits_bpnet.py "
            "was run with, if any -- needed only to locate that run's output "
            "directory (encoded into the dir name via trim_suffix()); finemo "
            "report itself has no per-motif threshold override, so it can't "
            "exactly reproduce this trimming (see module docstring)"
        ),
    )
    parser.add_argument(
        "--cwm-trim-coords",
        type=str,
        default=None,
        help=(
            "path to the same -R/--cwm-trim-coords file call_hits_bpnet.py "
            "was run with, if any -- needed only to locate that run's output "
            "directory (encoded into the dir name via trim_suffix())"
        ),
    )
    parser.add_argument(
        "--cwm-similarity-threshold",
        type=float,
        default=0.9,
        help=(
            "drop all hits for motifs with cwm_similarity at or below this "
            "value (default: 0.9, from the HDMA fetal atlas paper's "
            "cwm_correlation filter)"
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    model_dir_name = Path(args.model_dir).name if args.model_dir else args.experiment
    suffix = trim_suffix(
        args.cwm_trim_threshold, args.cwm_trim_thresholds, args.cwm_trim_coords
    )
    # regions.npz is trim-independent and lives in the plain
    # {model_dir_name}_{head}/ cache dir; hits themselves live in that dir's
    # trim-suffixed subdirectory (see call_hits_bpnet.py).
    exp_dir = REPO_ROOT / "hitcalls" / "bpnet" / f"{model_dir_name}_{args.head}"
    hits_dir = exp_dir / suffix.lstrip("_") if suffix else exp_dir
    regions_npz = exp_dir / "regions.npz"
    hits_tsv = hits_dir / "hits_unique.tsv"

    if args.modisco_h5:
        modisco_h5 = Path(args.modisco_h5)
    else:
        modisco_h5 = (
            REPO_ROOT
            / "motifcompendium"
            / "bpnet"
            / f"motifcompendium_{args.head}_cluster_averages.h5"
        )

    for path, label in [
        (regions_npz, "regions.npz"),
        (hits_tsv, "hits_unique.tsv"),
        (modisco_h5, "motif CWMs"),
    ]:
        if not Path(path).exists():
            print(f"Error: {label} not found: {path}", file=sys.stderr)
            print("Run call_hits_bpnet.py first.", file=sys.stderr)
            sys.exit(1)

    report_dir = hits_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    run(
        [
            "finemo",
            "report",
            "-r",
            str(regions_npz),
            "-H",
            str(hits_dir),
            "-m",
            str(modisco_h5),
            "-o",
            str(report_dir),
            "-t",
            str(args.cwm_trim_threshold),
            "-n",
        ],
        args.verbose,
    )

    motif_report_path = report_dir / "motif_report.tsv"
    motif_report = pl.read_csv(motif_report_path, separator="\t")
    low_similarity = motif_report.filter(
        pl.col("cwm_similarity") <= args.cwm_similarity_threshold
    )
    drop_motifs = set(low_similarity["motif_name"].to_list())
    motif_names = motif_report["motif_name"].to_list()

    plot_similarity_distribution(
        motif_report,
        args.cwm_similarity_threshold,
        report_dir / "cwm_similarity_distribution.png",
    )

    print(
        f"{len(drop_motifs)}/{motif_report.height} motifs at or below "
        f"cwm_similarity {args.cwm_similarity_threshold}:"
    )
    for row in low_similarity.sort("cwm_similarity").iter_rows(named=True):
        print(
            f"  {row['motif_name']}: cwm_similarity={row['cwm_similarity']:.3f} "
            f"num_hits_total={row['num_hits_total']}"
        )

    hits = pl.read_csv(hits_tsv, separator="\t")
    hits_filtered = hits.filter(~pl.col("motif_name").is_in(drop_motifs))
    hits_filtered_path = hits_dir / "hits_filtered.tsv"
    hits_filtered.write_csv(hits_filtered_path, separator="\t")

    print(
        f"\nKept {hits_filtered.height}/{hits.height} hits "
        f"({hits.height - hits_filtered.height} dropped from "
        f"{len(drop_motifs)} low-similarity motifs)"
    )
    print(f"Wrote {hits_filtered_path}")

    comparison_dir = hits_dir / "comparison"
    plot_hit_comparison(hits, motif_names, comparison_dir / "pre_filter")
    plot_hit_comparison(hits_filtered, motif_names, comparison_dir / "post_filter")
    print(
        f"Wrote pre/post-filter comparison plots (hit-stat distributions, "
        f"hits-per-peak distributions, motif co-occurrence heatmap) to "
        f"{comparison_dir}"
    )


if __name__ == "__main__":
    main()
