#!/usr/bin/env python3
"""Submit SLURM jobs to compute compute_trim_floor.py's minimum
motif-trim-length floor per experiment.

Reads experiment IDs from configs/experiment_config.yaml and submits one
sbatch job per (experiment, head) pair via compute_trim_floor.py -e, so
hitcall/launch.py --min-trim-len and hitcall/launch_report.py/launch_link.py
--min-trim-len have a per-experiment trim-coords file to find.

Jobs are skipped if the output TSV already exists or if the per-experiment
modisco.h5 is missing (run modisco/launch.py first).

Usage:
    python src/bpnet/hitcall/launch_trim_floor.py                    # submit all experiments, profile head
    python src/bpnet/hitcall/launch_trim_floor.py --dry-run           # print sbatch scripts without submitting
    python src/bpnet/hitcall/launch_trim_floor.py --head profile --head count
    python src/bpnet/hitcall/launch_trim_floor.py --min-len 8
    python src/bpnet/hitcall/launch_trim_floor.py --min-reads 20000000
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
TRIM_FLOOR_SCRIPT = REPO_ROOT / "src" / "bpnet" / "hitcall" / "compute_trim_floor.py"


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
        help="motif head(s) to compute the floor for; repeatable (default: profile)",
    )
    parser.add_argument(
        "--min-len",
        type=int,
        default=6,
        help="minimum trimmed motif length in bp (default: 6, Kelly Cochran's "
        "ProCapNet Inr floor)",
    )
    # SLURM resource flags
    parser.add_argument("--partition", type=str, default="normal,akundaje,owners")
    parser.add_argument("--cpus-per-task", type=int, default=1)
    parser.add_argument("--mem", type=str, default="8G")
    parser.add_argument("--time", type=str, default="15:00")
    parser.add_argument(
        "--min-reads",
        type=int,
        default=0,
        help="skip experiments with fewer total reads than this (default: 0, disabled)",
    )
    parser.add_argument(
        "--trim-floor-args",
        type=str,
        default="",
        help="extra arguments forwarded to compute_trim_floor.py (e.g. '--cwm-trim-threshold 0.25')",
    )
    args = parser.parse_args()

    heads = args.head if args.head is not None else ["profile"]

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    experiments = list(config["experiments"].keys())

    read_counts_df = pd.read_csv(
        N_READS_PATH, sep="\t", usecols=["experiment", "total_reads"]
    )
    read_counts = dict(zip(read_counts_df["experiment"], read_counts_df["total_reads"]))

    modisco_dir = REPO_ROOT / "modisco" / "bpnet"
    log_dir = REPO_ROOT / "logs" / "bpnet_hitcall_trim_floor"
    log_dir.mkdir(parents=True, exist_ok=True)

    submitted = 0
    skipped_done = 0
    skipped_no_modisco_h5 = 0
    skipped_reads = 0
    for exp_id in experiments:
        n_reads = read_counts.get(exp_id, 0)
        if n_reads < args.min_reads:
            skipped_reads += 1
            continue

        for head in heads:
            modisco_h5 = modisco_dir / f"{exp_id}_{head}.modisco.h5"
            if not modisco_h5.exists():
                skipped_no_modisco_h5 += 1
                continue

            out_path = (
                modisco_dir / f"{exp_id}_{head}_trim_coords_min{args.min_len}bp.tsv"
            )
            if out_path.exists():
                skipped_done += 1
                continue

            job_name = f"bpnet_hitcall_trim_floor_{exp_id}_{head}"
            trim_floor_cmd = (
                f"uv run --project {REPO_ROOT} --extra sherlock --frozen python {TRIM_FLOOR_SCRIPT} "
                f"-e {exp_id} --head {head} --min-len {args.min_len} {args.trim_floor_args}"
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

                ml biology
                ml htslib

                mamba activate "${{PROCAP_ATLAS_ENV:-procap-atlas}}"
                {trim_floor_cmd}
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
        f"with <{args.min_reads:,} reads, skipped {skipped_no_modisco_h5} "
        f"missing the per-experiment MoDISco h5, skipped {skipped_done} "
        f"already computed ({total} total)"
    )


if __name__ == "__main__":
    main()
