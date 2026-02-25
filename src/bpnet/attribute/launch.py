#!/usr/bin/env python3
"""Submit SLURM jobs to compute DeepLIFT/SHAP attributions for trained BPNet models.

Reads experiment IDs from configs/experiment_config.yaml and submits one
sbatch job per (experiment, head) pair via attribute_bpnet.py.

Jobs are skipped if the output attribution file already exists or if any
fold model is missing (meaning training is incomplete). Experiments with
fewer total reads than --min-reads (default: 10_000_000) are also skipped.

Usage:
    python src/bpnet/attribute/launch.py                    # submit all experiments, profile head
    python src/bpnet/attribute/launch.py --dry-run           # print sbatch scripts without submitting
    python src/bpnet/attribute/launch.py --head count        # count head only
    python src/bpnet/attribute/launch.py --head profile --head count  # both heads
    python src/bpnet/attribute/launch.py --time 24:00:00 --mem 64G --partition gpu
    python src/bpnet/attribute/launch.py --min-reads 20000000  # only well-covered experiments
"""

import argparse
import subprocess
import sys
import textwrap
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "experiment_config.yaml"
CHROM_SPLITS_PATH = REPO_ROOT / "configs" / "chrom_splits.yaml"
N_READS_PATH = REPO_ROOT / "configs" / "n_reads.txt"
ATTRIBUTE_SCRIPT = REPO_ROOT / "src" / "bpnet" / "attribute" / "attribute_bpnet.py"


def load_n_reads():
    """Parse n_reads.txt and return {experiment_id: total_reads} dict."""
    df = pd.read_csv(N_READS_PATH, sep="\t", usecols=["experiment", "total_reads"])
    return dict(zip(df["experiment"], df["total_reads"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print sbatch scripts without submitting",
    )
    parser.add_argument(
        "--head",
        type=str,
        action="append",
        choices=["profile", "count"],
        default=None,
        metavar="HEAD",
        help="attribution head(s) to run; repeatable (default: profile)",
    )
    # SLURM resource flags
    parser.add_argument("--partition", type=str, default="akundaje")
    parser.add_argument("--gpus", type=str, default="1")
    parser.add_argument("--cpus-per-task", type=int, default=1)
    parser.add_argument("--mem", type=str, default="64G")
    parser.add_argument("--time", type=str, default="48:00:00")
    parser.add_argument(
        "--min-reads",
        type=int,
        default=10_000_000,
        help="skip experiments with fewer total reads than this (default: 10000000)",
    )
    # Extra args forwarded to attribute_bpnet.py
    parser.add_argument(
        "--attribute-args",
        type=str,
        default="",
        help="extra arguments forwarded to attribute_bpnet.py (e.g. '--batch-size 32')",
    )
    args = parser.parse_args()

    heads = args.head if args.head is not None else ["profile"]

    # Load experiment list
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    experiments = list(config["experiments"].keys())

    # Load fold count
    with open(CHROM_SPLITS_PATH) as f:
        chrom_splits = yaml.safe_load(f)
    n_folds = len(chrom_splits["folds"])

    # Load read counts for filtering
    read_counts = load_n_reads()

    log_dir = REPO_ROOT / "logs" / "bpnet_attr"
    log_dir.mkdir(parents=True, exist_ok=True)

    submitted = 0
    skipped_done = 0
    skipped_untrained = 0
    skipped_reads = 0
    for exp_id in experiments:
        # Skip experiments with too few reads
        n_reads = read_counts.get(exp_id, 0)
        if n_reads < args.min_reads:
            skipped_reads += 1
            continue

        # Skip if any fold model is missing (training incomplete)
        model_dir = REPO_ROOT / "models" / "bpnet" / exp_id
        missing = [
            model_dir / f"{exp_id}.fold{fold}.torch"
            for fold in range(n_folds)
            if not (model_dir / f"{exp_id}.fold{fold}.torch").exists()
        ]
        if missing:
            skipped_untrained += 1
            continue

        for head in heads:
            # Skip if attribution output already exists
            out_path = REPO_ROOT / "attributions" / "bpnet" / f"{exp_id}_{head}.npz"
            if out_path.exists():
                skipped_done += 1
                continue

            job_name = f"bpnet_attr_{exp_id}_{head}"
            attr_cmd = f"python {ATTRIBUTE_SCRIPT} -e {exp_id} --head {head} -v {args.attribute_args}"

            sbatch_script = textwrap.dedent(f"""\
                #!/bin/bash
                #SBATCH --job-name={job_name}
                #SBATCH --partition={args.partition}
                #SBATCH --gpus={args.gpus}
                #SBATCH --cpus-per-task={args.cpus_per_task}
                #SBATCH --mem={args.mem}
                #SBATCH --time={args.time}
                #SBATCH --output={log_dir}/{job_name}.out
                #SBATCH --error={log_dir}/{job_name}.err

                ml openblas/0.3.28
                ml xsimd/8.1.0
                ml xz/5.8.1
                ml hdf5/1.14.4
                ml arrow/22.0.0
                ml load py-pyarrow/18.1.0_py312
                ml lz4/1.8.0
                ml biology
                ml htslib
                ml ucsc-utils

                mamba activate torch
                {attr_cmd}
            """)

            if args.dry_run:
                print(f"--- {job_name} ---")
                print(sbatch_script)
                submitted += 1
                continue

            result = subprocess.run(
                ["sbatch"], input=sbatch_script, capture_output=True, text=True
            )
            if result.returncode == 0:
                print(f"{job_name}: {result.stdout.strip()}")
                submitted += 1
            else:
                print(
                    f"ERROR submitting {job_name}: {result.stderr.strip()}",
                    file=sys.stderr,
                )

    action = "Would submit" if args.dry_run else "Submitted"
    total = len(experiments) * len(heads)
    print(
        f"\n{action} {submitted} jobs, skipped {skipped_reads} experiments "
        f"with <{args.min_reads:,} reads, skipped {skipped_untrained} with incomplete "
        f"training, skipped {skipped_done} already attributed ({total} total)"
    )


if __name__ == "__main__":
    main()
