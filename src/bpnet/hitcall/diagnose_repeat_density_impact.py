#!/usr/bin/env python3
"""Report filter_repeat_density.py's per-motif hit-drop rate, across every
motif, not just the ones a specific investigation happened to be looking at.

filter_repeat_density.py is deliberately motif-identity-agnostic (see its
own module docstring): it drops all hits of a motif within any
--cluster-window bp span containing --min-cluster-hits+ same-motif hits in
the same peak, generalizing Kelly Cochran's hand-coded TATA/TATATA-only
repeat filter to every motif. The only real-data validation this got before
being trusted atlas-wide was two specific motifs (GATA, TA-Inr in K562
ENCSR220XSM), both showing <1% hits dropped -- reassuring for those two,
but not a check of whether the filter is similarly gentle for the other
~40+ motifs in the same experiment, or whether some other motif is being
hit much harder. filter_low_confidence_hits.py's hit_seqlet_confidence
score column made exactly this mistake once already: identity-agnostic
plus a couple of clean spot-checks looked safe, until the actual full
per-motif breakdown showed it touching 28/45 motifs at 0%-89% drop rates,
including motifs with no known contamination problem at all (see that
script's module docstring for the full story). This script runs the same
kind of full breakdown for the repeat-density filter specifically, using
data that's already there -- no recomputation needed, just comparing
hits_unique.tsv (before) against hits_dedensified.tsv (after) per motif.

Usage:
    python src/bpnet/hitcall/diagnose_repeat_density_impact.py -e ENCSR220XSM
    python src/bpnet/hitcall/diagnose_repeat_density_impact.py -e ENCSR220XSM --min-trim-len 6
    python src/bpnet/hitcall/diagnose_repeat_density_impact.py -e ENCSR220XSM --min-trim-len 6 --min-hits 50
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

from call_hits_bpnet import DEFAULT_CWM_TRIM_THRESHOLD, trim_suffix

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "experiment_config.yaml"
DEFAULT_MIN_HITS = 20


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-e", "--experiment", type=str, required=True,
        help="experiment accession ID (e.g. ENCSR882DWM)",
    )
    parser.add_argument(
        "-m", "--model-dir", type=str, default=None,
        help="override model directory, default derived from config",
    )
    parser.add_argument(
        "--head", type=str, default="profile", choices=["profile", "count"],
        help="attribution/motif head the hits were called against (default: profile)",
    )
    parser.add_argument(
        "--min-trim-len", type=int, default=None, metavar="BP",
        help=(
            "must match the value hitcall/launch.py was run with, if any -- "
            "resolves the same per-experiment trim-coords-suffixed output "
            "directory rather than the plain {model_dir_name}_{head}/ one."
        ),
    )
    parser.add_argument(
        "--min-hits", type=int, default=DEFAULT_MIN_HITS,
        help=(
            "skip a motif entirely if it has fewer than this many hits "
            f"before filtering (default: {DEFAULT_MIN_HITS}, too few for a "
            "meaningful drop-rate estimate)"
        ),
    )
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    if args.experiment not in config["experiments"]:
        print(f"Error: {args.experiment} not found in config", file=sys.stderr)
        sys.exit(1)

    model_dir_name = Path(args.model_dir).name if args.model_dir else args.experiment
    modisco_dir = REPO_ROOT / "modisco" / "bpnet"
    trim_coords = (
        modisco_dir / f"{args.experiment}_{args.head}_trim_coords_min{args.min_trim_len}bp.tsv"
        if args.min_trim_len is not None
        else None
    )
    suffix = trim_suffix(DEFAULT_CWM_TRIM_THRESHOLD, None, trim_coords)
    exp_dir = REPO_ROOT / "hitcalls" / "bpnet" / f"{model_dir_name}_{args.head}"
    hits_dir = exp_dir / suffix.lstrip("_") if suffix else exp_dir

    before_path = hits_dir / "hits_unique.tsv"
    after_path = hits_dir / "hits_dedensified.tsv"
    for path, label in [(before_path, "hits_unique.tsv"), (after_path, "hits_dedensified.tsv")]:
        if not path.exists():
            print(f"Error: {label} not found: {path}", file=sys.stderr)
            sys.exit(1)

    print(f"Before (pre-filter):  {before_path}")
    print(f"After (post-filter):  {after_path}\n")

    before_counts = pd.read_csv(before_path, sep="\t", usecols=["motif_name"])["motif_name"].value_counts()
    after_counts = pd.read_csv(after_path, sep="\t", usecols=["motif_name"])["motif_name"].value_counts()

    all_motifs = before_counts.index.union(after_counts.index)
    before_counts = before_counts.reindex(all_motifs, fill_value=0)
    after_counts = after_counts.reindex(all_motifs, fill_value=0)

    eligible = before_counts[before_counts >= args.min_hits].index
    n_skipped = len(all_motifs) - len(eligible)

    frac_dropped = (1 - after_counts / before_counts).reindex(eligible)
    frac_dropped = frac_dropped.sort_values(ascending=False)

    print(f"{'motif_name':<28} {'n_before':>10} {'n_after':>10} {'frac_dropped':>13}")
    for motif_name in frac_dropped.index:
        print(
            f"{motif_name:<28} {before_counts[motif_name]:>10} "
            f"{after_counts[motif_name]:>10} {frac_dropped[motif_name]:>13.1%}"
        )

    thresholds = [0.50, 0.10, 0.01]
    print(f"\n{len(eligible)} motif(s) checked ({n_skipped} skipped, fewer than {args.min_hits} hits):")
    for t in thresholds:
        n_above = int((frac_dropped > t).sum())
        print(f"  {n_above}/{len(eligible)} motif(s) with >{t:.0%} of hits dropped")


if __name__ == "__main__":
    main()
