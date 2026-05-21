#!/usr/bin/env python3
"""Submit SLURM jobs to convert BPNet attribution NPZ files to BigWigs.

Reads experiment IDs from configs/experiment_config.yaml and submits one
sbatch job per (experiment, head) pair via attribution_to_bigwig.py.

Jobs are skipped if the output BigWig already exists or if the required
attribution/OHE NPZ files are missing.

Usage:
    python src/bpnet/attribute/launch_bigwig_conversion.py
    python src/bpnet/attribute/launch_bigwig_conversion.py --dry-run
    python src/bpnet/attribute/launch_bigwig_conversion.py --head profile
    python src/bpnet/attribute/launch_bigwig_conversion.py --head profile --head count
    python src/bpnet/attribute/launch_bigwig_conversion.py --min-reads 20000000
"""

import argparse
import shlex
import subprocess
import sys
import textwrap
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "experiment_config.yaml"
N_READS_PATH = REPO_ROOT / "configs" / "n_reads.txt"
CONVERSION_SCRIPT = REPO_ROOT / "src" / "bpnet" / "attribute" / "attribution_to_bigwig.py"


def load_n_reads():
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
        help="attribution head(s) to run; repeatable (default: profile count)",
    )
    parser.add_argument(
        "--min-reads",
        type=int,
        default=0,
        help="skip experiments with fewer total reads than this (default: 0, disabled)",
    )
    parser.add_argument(
        "--conversion-args",
        type=str,
        default="",
        help="extra arguments forwarded to attribution_to_bigwig.py",
    )
    # SLURM resource flags, matching modisco/launch_report.py defaults.
    parser.add_argument("--partition", type=str, default="normal,akundaje,owners")
    parser.add_argument("--cpus-per-task", type=int, default=1)
    parser.add_argument("--mem", type=str, default="16G")
    parser.add_argument("--time", type=str, default="2:00:00")
    args = parser.parse_args()

    heads = args.head if args.head is not None else ["profile", "count"]

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    experiments = list(config["experiments"].keys())

    read_counts = load_n_reads()

    attr_dir = REPO_ROOT / "attributions" / "bpnet"
    out_dir = attr_dir / "bigwigs"
    log_dir = REPO_ROOT / "logs" / "bpnet_attr_bigwig"
    log_dir.mkdir(parents=True, exist_ok=True)

    submitted = 0
    skipped_done = 0
    skipped_no_attr = 0
    skipped_reads = 0

    for exp_id in experiments:
        n_reads = read_counts.get(exp_id, 0)
        if n_reads < args.min_reads:
            skipped_reads += 1
            continue

        ohe_path = attr_dir / f"{exp_id}_ohe.npz"
        if not ohe_path.exists():
            skipped_no_attr += len(heads)
            continue

        for head in heads:
            attr_path = attr_dir / f"{exp_id}_{head}.npz"
            if not attr_path.exists():
                skipped_no_attr += 1
                continue

            output_path = out_dir / f"{exp_id}_{head}.bigWig"
            if output_path.exists():
                skipped_done += 1
                continue

            job_name = f"attr_bigwig_{exp_id}_{head}"
            conversion_cmd = (
                f"python {shlex.quote(str(CONVERSION_SCRIPT))}"
                f" -e {shlex.quote(exp_id)}"
                f" --head {shlex.quote(head)}"
            )
            if args.conversion_args:
                conversion_cmd += f" {args.conversion_args}"

            sbatch_script = textwrap.dedent(f"""\
                #!/bin/bash -l
                #SBATCH --job-name={job_name}
                #SBATCH --ntasks=1
                #SBATCH --ntasks-per-node=1
                #SBATCH --nodes=1
                #SBATCH --cpus-per-task={args.cpus_per_task}
                #SBATCH --mem={args.mem}
                #SBATCH --partition={args.partition}
                #SBATCH --time={args.time}
                #SBATCH --output={log_dir}/{job_name}.out
                #SBATCH --error={log_dir}/{job_name}.err
                #SBATCH -C NO_GPU

                ml gh # github CLI
                ml gcc/12.4.0
                ml cmake/3.31.4
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

                cd {REPO_ROOT}
                mkdir -p {out_dir}
                time {conversion_cmd}
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
        f"with <{args.min_reads:,} reads, skipped {skipped_no_attr} missing "
        f"attribution/OHE files, skipped {skipped_done} already done ({total} total)"
    )


if __name__ == "__main__":
    main()
