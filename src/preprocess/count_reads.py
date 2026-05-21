#!/usr/bin/env python3
"""Count total reads in the processed BigWig files for each experiment.

Read counts are computed as the sum over all chromosomes of the BigWig signal
using pybigtools. The minus-strand BigWigs store counts as negative values,
so their total is taken as an absolute value.

Usage:
    python src/preprocess/count_reads.py                    # writes to configs/n_reads.txt
    python src/preprocess/count_reads.py --tsv reads.tsv   # write TSV to custom path
"""

import argparse
import sys
from pathlib import Path

import pybigtools
import yaml
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "experiment_config.yaml"
OUTPUT_PATH = REPO_ROOT / "configs" / "n_reads.txt"


def count_bigwig_reads(bigwig_path: Path) -> float:
    """Return total read count for a BigWig file."""
    with pybigtools.open(str(bigwig_path)) as bw:
        return sum(
            value * (end - start)
            for chrom, size in bw.chroms().items()
            for start, end, value in bw.records(chrom, 0, size)
        )


def main():
    parser = argparse.ArgumentParser(description="Count reads per experiment BigWig")
    parser.add_argument(
        "--tsv",
        metavar="FILE",
        default=OUTPUT_PATH,
        help=f"output TSV path (default: {OUTPUT_PATH})",
    )
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    header = "\t".join(
        ["experiment", "biosample", "pl_reads", "mn_reads", "total_reads"]
    )

    rows = []
    experiments = config["experiments"]
    for exp_id, exp in tqdm(experiments.items(), total=len(experiments), unit="exp"):
        biosample = exp.get("biosample", "")
        processed = exp.get("processed", {})

        pl_path = REPO_ROOT / processed["pl_bigwig"]
        mn_path = REPO_ROOT / processed["mn_bigwig"]

        missing = [p for p in (pl_path, mn_path) if not p.exists()]
        if missing:
            print(
                f"WARNING: {exp_id}: missing files {[p.name for p in missing]}, skipping",
                file=sys.stderr,
            )
            continue

        pl_reads = count_bigwig_reads(pl_path)
        mn_reads = abs(count_bigwig_reads(mn_path))
        total = pl_reads + mn_reads

        row = f"{exp_id}\t{biosample}\t{pl_reads:.0f}\t{mn_reads:.0f}\t{total:.0f}"
        rows.append(row)

    with open(args.tsv, "w") as f:
        f.write(header + "\n")
        f.write("\n".join(rows) + "\n")
    print(f"\nResults written to {args.tsv}", file=sys.stderr)


if __name__ == "__main__":
    main()
