#!/usr/bin/env python3
"""Check whether CA-Inr's "excess" hits (peaks where it's called 2-4x --
see diagnose_hit_signal_metaplot.py's module docstring for the investigation
this follows on from) fail tangermeme.seqlet.recursive_seqlets corroboration
at a much higher rate than "normal" hits (peaks with exactly 1 hit) -- the
same check that originally found the TATA/TA-Inr defect in K562
(filter_low_confidence_hits.py's hit_seqlet_confidence).

Every other per-hit statistic checked so far (hit_correlation,
hit_coefficient, hit_importance, position relative to peak midpoint, strand
majority) fails to distinguish excess hits from normal ones. Those are all,
directly or indirectly, downstream of Fi-NeMo's global per-motif L1-penalty/
correlation-floor thresholding, which has no local-prominence check
anywhere -- exactly the root cause established for TATA/TA-Inr. If CA-Inr's
excess hits are the same defect, they should fail seqlet corroboration
(hit_seqlet_confidence == 0.0, no overlapping recursive_seqlets call at all)
far more often than normal hits, even though hit_correlation/hit_coefficient
can't tell them apart.

This does NOT filter anything or write output -- it's the same diagnostic
step that, for TATA, preceded actually building the corroboration filter
into the pipeline. --seqlet-low-similarity-only's cwm_similarity-based
scoping was never applied to CA-Inr (its cwm_similarity stays >0.9
regardless of this problem -- see the investigation writeup), so this
checks the corroboration signal directly rather than through that gate.

Usage:
    python src/bpnet/hitcall/diagnose_seqlet_confidence_by_group.py -e ENCSR342WAR --min-trim-len 6 \\
        --motif-name pos_patterns.pattern_2
    python src/bpnet/hitcall/diagnose_seqlet_confidence_by_group.py -e ENCSR342WAR --min-trim-len 6 \\
        --motif-name pos_patterns.pattern_2 --split-excess-by-strand-mix
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from finemo.data_io import load_regions_npz

from call_hits_bpnet import DEFAULT_CWM_TRIM_THRESHOLD, resolve_hits_path, trim_suffix
from filter_by_seqlet_importance import build_peak_row_index, project_contribs
from filter_low_confidence_hits import (
    DEFAULT_SEQLET_ADDITIONAL_FLANKS,
    DEFAULT_SEQLET_CLIP_PERCENTILE,
    DEFAULT_SEQLET_THRESHOLD,
    compute_seqlet_confidence,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_NORMAL_COUNT = 1
DEFAULT_EXCESS_MIN_COUNT = 2
DEFAULT_EXCESS_MAX_COUNT = 4


def summarize(label, confidence):
    n = len(confidence)
    n_corroborated = int((confidence > 0.0).sum())
    frac = n_corroborated / n if n else float("nan")
    median_all = float(np.median(confidence))
    median_corroborated = (
        float(np.median(confidence[confidence > 0.0])) if n_corroborated else float("nan")
    )
    print(
        f"  {label:<45} n={n:>7}  corroborated={n_corroborated:>7} ({frac:>6.1%})  "
        f"median(all)={median_all:>6.3f}  median(corroborated only)={median_corroborated:>6.3f}"
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-e", "--experiment", type=str, required=True)
    parser.add_argument("-m", "--model-dir", type=str, default=None)
    parser.add_argument(
        "--head", type=str, default="profile", choices=["profile", "count"]
    )
    parser.add_argument("--min-trim-len", type=int, default=None, metavar="BP")
    parser.add_argument("--motif-name", type=str, required=True)
    parser.add_argument("--normal-count", type=int, default=DEFAULT_NORMAL_COUNT, metavar="N")
    parser.add_argument("--excess-min-count", type=int, default=DEFAULT_EXCESS_MIN_COUNT, metavar="N")
    parser.add_argument("--excess-max-count", type=int, default=DEFAULT_EXCESS_MAX_COUNT, metavar="N")
    parser.add_argument("--split-excess-by-strand-mix", action="store_true")
    parser.add_argument("--seqlet-threshold", type=float, default=DEFAULT_SEQLET_THRESHOLD)
    parser.add_argument("--seqlet-clip-percentile", type=float, default=DEFAULT_SEQLET_CLIP_PERCENTILE)
    parser.add_argument("--seqlet-additional-flanks", type=int, default=DEFAULT_SEQLET_ADDITIONAL_FLANKS)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

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

    hits_path = resolve_hits_path(hits_dir, verbose=args.verbose)
    if hits_path is None:
        print(f"Error: no hits found in {hits_dir}", file=sys.stderr)
        sys.exit(1)
    regions_npz = exp_dir / "regions.npz"
    if not regions_npz.exists():
        print(f"Error: regions.npz not found: {regions_npz}", file=sys.stderr)
        sys.exit(1)

    if args.verbose:
        print(f"Reading hits from {hits_path}")
        print(f"Reading regions from {regions_npz}")

    hits = pd.read_csv(hits_path, sep="\t")
    hits = hits[hits["motif_name"] == args.motif_name].copy()
    if hits.empty:
        print(f"Error: no hits for motif {args.motif_name}", file=sys.stderr)
        sys.exit(1)

    sequences, contribs, peaks_df, _ = load_regions_npz(str(regions_npz))
    contribs = project_contribs(contribs, sequences)
    peak_row_index = build_peak_row_index(peaks_df)
    peak_region_starts = peaks_df["peak_region_start"].to_numpy()

    print(f"Computing hit_seqlet_confidence for {len(hits)} hits of {args.motif_name}...")
    hits["hit_seqlet_confidence"] = compute_seqlet_confidence(
        hits,
        contribs,
        peak_row_index,
        peak_region_starts,
        threshold=args.seqlet_threshold,
        additional_flanks=args.seqlet_additional_flanks,
        clip_percentile=args.seqlet_clip_percentile,
    )

    counts = hits.groupby("peak_id").size()
    normal_peaks = counts[counts == args.normal_count].index
    excess_peaks = counts[
        (counts >= args.excess_min_count) & (counts <= args.excess_max_count)
    ].index

    print(
        f"\n{args.experiment} {args.head} {args.motif_name} "
        f"(recursive_seqlets threshold={args.seqlet_threshold}):"
    )
    summarize(
        f"normal (n={args.normal_count} hit/peak)",
        hits.loc[hits["peak_id"].isin(normal_peaks), "hit_seqlet_confidence"].to_numpy(),
    )

    excess_hits = hits[hits["peak_id"].isin(excess_peaks)]
    if args.split_excess_by_strand_mix:
        strand_nunique = excess_hits.groupby("peak_id")["strand"].nunique()
        mixed_strand_peaks = strand_nunique[strand_nunique > 1].index
        same_strand_peaks = strand_nunique[strand_nunique == 1].index
        summarize(
            f"excess mixed-strand ({args.excess_min_count}-{args.excess_max_count}/peak)",
            excess_hits.loc[
                excess_hits["peak_id"].isin(mixed_strand_peaks), "hit_seqlet_confidence"
            ].to_numpy(),
        )
        summarize(
            f"excess same-strand ({args.excess_min_count}-{args.excess_max_count}/peak)",
            excess_hits.loc[
                excess_hits["peak_id"].isin(same_strand_peaks), "hit_seqlet_confidence"
            ].to_numpy(),
        )
    else:
        summarize(
            f"excess ({args.excess_min_count}-{args.excess_max_count} hits/peak)",
            excess_hits["hit_seqlet_confidence"].to_numpy(),
        )


if __name__ == "__main__":
    main()
