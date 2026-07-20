#!/usr/bin/env python3
"""Submit SLURM jobs to run `modisco report` on completed modisco .h5 outputs.

Reads experiment IDs from configs/experiment_config.yaml and submits one
sbatch job per (experiment, head) pair. Jobs are skipped if the report
directory already exists or if the .h5 file is missing (run launch.py first).

Usage:
    python src/bpnet/modisco/launch_report.py                    # submit all experiments, both heads
    python src/bpnet/modisco/launch_report.py --dry-run           # print sbatch scripts without submitting
    python src/bpnet/modisco/launch_report.py --head count        # count head only
    python src/bpnet/modisco/launch_report.py --head profile      # profile head only
    python src/bpnet/modisco/launch_report.py --min-reads 20000000
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
N_READS_PATH = REPO_ROOT / "configs" / "n_reads.txt"
JASPAR_PATH = (
    REPO_ROOT / "data" / "JASPAR2026_CORE_vertebrates_non-redundant_pfms_meme.txt"
)


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
    # SLURM resource flags
    parser.add_argument("--partition", type=str, default="normal,akundaje,owners")
    parser.add_argument("--cpus-per-task", type=int, default=1)
    parser.add_argument("--mem", type=str, default="16G")
    parser.add_argument("--time", type=str, default="2:00:00")
    parser.add_argument(
        "--min-reads",
        type=int,
        default=0,
        help="skip experiments with fewer total reads than this (default: 0, disabled)",
    )
    args = parser.parse_args()

    heads = args.head if args.head is not None else ["profile", "count"]

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    experiments = list(config["experiments"].keys())

    read_counts_df = pd.read_csv(
        N_READS_PATH, sep="\t", usecols=["experiment", "total_reads"]
    )
    read_counts = dict(zip(read_counts_df["experiment"], read_counts_df["total_reads"]))

    out_dir = REPO_ROOT / "modisco" / "bpnet"
    log_dir = REPO_ROOT / "logs" / "bpnet_modisco"
    log_dir.mkdir(parents=True, exist_ok=True)

    submitted = 0
    skipped_done = 0
    skipped_no_h5 = 0
    skipped_reads = 0

    for exp_id in experiments:
        n_reads = read_counts.get(exp_id, 0)
        if n_reads < args.min_reads:
            skipped_reads += 1
            continue

        for head in heads:
            out_h5 = out_dir / f"{exp_id}_{head}.modisco.h5"
            if not out_h5.exists():
                skipped_no_h5 += 1
                continue

            report_dir = out_dir / f"{exp_id}_{head}.modisco"
            if report_dir.exists():
                skipped_done += 1
                continue

            job_name = f"modisco_report_{exp_id}_{head}"

            modisco_report_cmd = (
                f"uv run --project {REPO_ROOT} --extra sherlock --frozen modisco report"
                f" -i {out_h5}"
                f" -o {exp_id}_{head}.modisco"
                f" -m {JASPAR_PATH}"
                f" --lite"
            )

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
                
                mamba activate "${{PROCAP_ATLAS_ENV:-procap-atlas}}"

                cd {out_dir}
                time {modisco_report_cmd}
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
        f"with <{args.min_reads:,} reads, skipped {skipped_no_h5} missing .h5, "
        f"skipped {skipped_done} already done ({total} total)"
    )


if __name__ == "__main__":
    main()
