#!/usr/bin/env python3
"""Consolidate every experiment's final report_bpnet.py motif_report.tsv
into one atlas-wide table -- the "final Fi-NeMo results" once
launch_post_hoc_pipeline.py has run atlas-wide, since nothing else in this
repo aggregates hit-calling QC across experiments (only per-experiment
scripts read one motif_report.tsv at a time).

motif_report.tsv is written by finemo's own `finemo report` CLI (not by this
repo's code), one row per motif, with columns motif_name, num_hits_total,
num_hits_restricted, cwm_similarity. Reads the *final* pass's report
(hits_dir/report/motif_report.tsv, i.e. after both filter_repeat_density.py
and filter_low_confidence_hits.py), not the baseline pass report_bpnet.py
also writes for --seqlet-low-similarity-only scoping -- that one reflects
pre-corroboration-filter hit counts and would overstate what actually
survived.

Usage:
    python src/bpnet/hitcall/consolidate_motif_reports.py
    python src/bpnet/hitcall/consolidate_motif_reports.py --head profile --head count
    python src/bpnet/hitcall/consolidate_motif_reports.py --min-trim-len 6
    python src/bpnet/hitcall/consolidate_motif_reports.py --min-reads 20000000
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

import compressed_io
from call_hits_bpnet import resolve_experiment_paths

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "experiment_config.yaml"
N_READS_PATH = REPO_ROOT / "configs" / "n_reads.txt"
DEFAULT_OUT = REPO_ROOT / "figures" / "motif_atlas" / "hitcall_motif_report_consolidated.tsv"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--head", type=str, action="append", choices=["profile", "count"],
        default=None, metavar="HEAD",
        help="attribution/motif head(s) to consolidate; repeatable (default: profile)",
    )
    parser.add_argument(
        "--min-reads", type=int, default=0,
        help="skip experiments with fewer total reads than this (default: 0, disabled)",
    )
    parser.add_argument(
        "--min-trim-len", type=int, default=None, metavar="BP",
        help="must match the value hitcall/launch.py was run with, if any",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, metavar="PATH")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    heads = args.head if args.head is not None else ["profile"]

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    experiments = list(config["experiments"].keys())

    read_counts_df = pd.read_csv(
        N_READS_PATH, sep="\t", usecols=["experiment", "total_reads"]
    )
    read_counts = dict(zip(read_counts_df["experiment"], read_counts_df["total_reads"]))

    rows = []
    missing = []
    skipped_reads = 0
    for exp_id in experiments:
        if read_counts.get(exp_id, 0) < args.min_reads:
            skipped_reads += 1
            continue
        for head in heads:
            _, hits_dir, _, _ = resolve_experiment_paths(exp_id, head, args.min_trim_len)
            report_path = hits_dir / "report" / "motif_report.tsv"
            if not compressed_io.exists(report_path):
                missing.append((exp_id, head))
                continue
            df = pd.read_csv(compressed_io.resolve(report_path), sep="\t")
            df.insert(0, "head", head)
            df.insert(0, "experiment", exp_id)
            rows.append(df)
            if args.verbose:
                print(f"{exp_id} {head}: {len(df)} motif(s)")

    if not rows:
        print("Error: no motif_report.tsv found for any experiment/head", file=sys.stderr)
        sys.exit(1)

    consolidated = pd.concat(rows, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    consolidated.to_csv(args.out, sep="\t", index=False)

    n_exp = consolidated["experiment"].nunique()
    total = len(experiments) * len(heads)
    print(
        f"Consolidated {len(consolidated)} motif row(s) across {n_exp} "
        f"experiment(s), {len(missing)} missing, {skipped_reads} skipped "
        f"(<{args.min_reads:,} reads) of {total} experiment/head pair(s)"
    )
    if missing and args.verbose:
        print("Missing (experiment, head):", missing)

    print("\nAtlas-wide totals:")
    print(f"  total hits (num_hits_total, all motifs summed): {consolidated['num_hits_total'].sum():,}")
    if "num_hits_restricted" in consolidated.columns:
        print(f"  total hits (num_hits_restricted): {consolidated['num_hits_restricted'].sum():,}")
    print(f"  median cwm_similarity: {consolidated['cwm_similarity'].median():.3f}")
    low_sim = (consolidated["cwm_similarity"] <= 0.8).mean()
    print(f"  motif rows at or below cwm_similarity 0.8: {low_sim:.1%}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
