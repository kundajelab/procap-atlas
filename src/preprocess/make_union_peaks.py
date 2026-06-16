#!/usr/bin/env python3
"""
Build a union/merged peak set across all experiments.

Reads each experiment's processed peak BED file, pools the intervals,
sorts by chromosome and start position, and merges overlapping or adjacent
intervals into a non-overlapping union peak set.

Only canonical chromosomes (chr1–22, X, Y) are included. Peaks can optionally
be extended by --slop bp on each side before merging.

Output: data/processed/peaks/union_peaks.bed.gz (3 columns: chrom, start, end)

Usage:
    python src/preprocess/make_union_peaks.py
    python src/preprocess/make_union_peaks.py --min-reads 10000000
    python src/preprocess/make_union_peaks.py --slop 100
    python src/preprocess/make_union_peaks.py --dry-run
"""

import argparse
import gzip
import sys
from pathlib import Path

import yaml
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "experiment_config.yaml"
N_READS_PATH = REPO_ROOT / "configs" / "n_reads.txt"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "processed" / "peaks" / "union_peaks.bed.gz"

CANONICAL_CHROMS = [f"chr{i}" for i in list(range(1, 23)) + ["X", "Y"]]
CHROM_ORDER = {c: i for i, c in enumerate(CANONICAL_CHROMS)}


def read_peaks(bed_gz_path: Path, slop: int) -> list[tuple[int, int, int]]:
    """Read (chrom_index, start, end) tuples from a gzipped BED file.

    Only canonical chromosomes are included. start/end are adjusted by slop.
    """
    intervals = []
    with gzip.open(bed_gz_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                continue
            chrom = fields[0]
            if chrom not in CHROM_ORDER:
                continue
            start = max(0, int(fields[1]) - slop)
            end = int(fields[2]) + slop
            intervals.append((CHROM_ORDER[chrom], start, end))
    return intervals


def merge_intervals(
    intervals: list[tuple[int, int, int]],
) -> list[tuple[str, int, int]]:
    """Sort and merge overlapping/adjacent intervals.

    Input: list of (chrom_index, start, end)
    Output: list of (chrom, start, end) merged non-overlapping intervals
    """
    if not intervals:
        return []

    intervals.sort()  # sorts by (chrom_index, start, end)

    merged = []
    cur_chrom_idx, cur_start, cur_end = intervals[0]

    for chrom_idx, start, end in intervals[1:]:
        if chrom_idx == cur_chrom_idx and start <= cur_end:
            cur_end = max(cur_end, end)
        else:
            merged.append((CANONICAL_CHROMS[cur_chrom_idx], cur_start, cur_end))
            cur_chrom_idx, cur_start, cur_end = chrom_idx, start, end

    merged.append((CANONICAL_CHROMS[cur_chrom_idx], cur_start, cur_end))
    return merged


def main():
    parser = argparse.ArgumentParser(description="Build union peak set across experiments")
    parser.add_argument(
        "--min-reads",
        type=float,
        default=0,
        metavar="N",
        help="skip experiments with fewer than N total reads (default: 0)",
    )
    parser.add_argument(
        "--slop",
        type=int,
        default=0,
        metavar="BP",
        help="extend each peak by BP on each side before merging (default: 0)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        metavar="PATH",
        help=f"output path (default: {DEFAULT_OUTPUT.relative_to(REPO_ROOT)})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print summary stats without writing output",
    )
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    n_reads_map = {}
    if N_READS_PATH.exists():
        with open(N_READS_PATH) as f:
            next(f)
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 5:
                    n_reads_map[parts[0]] = float(parts[4])

    all_intervals: list[tuple[int, int, int]] = []
    n_included = 0

    for exp_id, exp in tqdm(config["experiments"].items(), unit="exp"):
        total_reads = n_reads_map.get(exp_id, 0)
        if total_reads < args.min_reads:
            continue

        processed = exp.get("processed", {})
        peaks_path = REPO_ROOT / processed.get("peaks", "")
        if not peaks_path.exists():
            print(f"WARNING: {exp_id}: peak file not found, skipping", file=sys.stderr)
            continue

        intervals = read_peaks(peaks_path, args.slop)
        all_intervals.extend(intervals)
        n_included += 1

    print(
        f"\nLoaded {len(all_intervals):,} peaks from {n_included} experiments",
        file=sys.stderr,
    )

    merged = merge_intervals(all_intervals)
    total_bp = sum(e - s for _, s, e in merged)
    print(f"Merged to {len(merged):,} union peaks ({total_bp / 1e6:.1f} Mbp)", file=sys.stderr)

    if args.dry_run:
        print("Dry run — no output written.", file=sys.stderr)
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output, "wt") as f:
        for chrom, start, end in merged:
            f.write(f"{chrom}\t{start}\t{end}\n")

    print(f"Written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
