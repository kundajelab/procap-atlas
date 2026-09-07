#!/usr/bin/env python3
"""Submit SLURM jobs to drop each motif's low-confidence hit mode, when one
is detectable.

Reads experiment IDs from configs/experiment_config.yaml and submits one
sbatch job per (experiment, head) pair via filter_low_confidence_hits.py.
Run after hitcall/launch_filter_repeat_density.py (this reads
hits_dedensified.tsv if present, else hits_unique.tsv) and before
hitcall/launch_report.py, which prefers this step's output
(hits_confidence_filtered.tsv) when present.

Jobs are skipped if hits_confidence_filtered.tsv already exists or if
neither hits_dedensified.tsv nor hits_unique.tsv exists (run
call_hits_bpnet.py/hitcall/launch.py, and optionally
filter_repeat_density.py/hitcall/launch_filter_repeat_density.py, first).
This step does not use a GPU, so it runs as its own cheap CPU-only launcher,
like launch_report.py.

Usage:
    python src/bpnet/hitcall/launch_low_confidence_hits.py                    # submit all experiments, profile head
    python src/bpnet/hitcall/launch_low_confidence_hits.py --dry-run           # print sbatch scripts without submitting
    python src/bpnet/hitcall/launch_low_confidence_hits.py --head profile --head count
    python src/bpnet/hitcall/launch_low_confidence_hits.py --min-reads 20000000
    python src/bpnet/hitcall/launch_low_confidence_hits.py --min-trim-len 6  # match hitcall/launch.py's floor
"""

import argparse
import subprocess
import sys
import textwrap
from pathlib import Path

import pandas as pd
import yaml

from call_hits_bpnet import DEFAULT_CWM_TRIM_THRESHOLD, trim_suffix

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "experiment_config.yaml"
N_READS_PATH = REPO_ROOT / "configs" / "n_reads.txt"
FILTER_SCRIPT = (
    REPO_ROOT / "src" / "bpnet" / "hitcall" / "filter_low_confidence_hits.py"
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
        help="attribution/motif head(s) to filter; repeatable (default: profile)",
    )
    # SLURM resource flags
    parser.add_argument("--partition", type=str, default="normal,akundaje,owners")
    parser.add_argument("--cpus-per-task", type=int, default=1)
    parser.add_argument("--mem", type=str, default="16G")
    parser.add_argument("--time", type=str, default="30:00")
    parser.add_argument(
        "--min-reads",
        type=int,
        default=0,
        help="skip experiments with fewer total reads than this (default: 0, disabled)",
    )
    parser.add_argument(
        "--filter-args",
        type=str,
        default="",
        help="extra arguments forwarded to filter_low_confidence_hits.py (e.g. '--min-rise-frac 1.25')",
    )
    parser.add_argument(
        "--min-trim-len",
        type=int,
        default=None,
        metavar="BP",
        help=(
            "must match the value hitcall/launch.py was run with, if any -- "
            "resolves the same per-experiment trim-coords-suffixed output "
            "directory rather than the plain {model_dir_name}_{head}/ one."
        ),
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

    hitcalls_dir = REPO_ROOT / "hitcalls" / "bpnet"
    modisco_dir = REPO_ROOT / "modisco" / "bpnet"
    log_dir = REPO_ROOT / "logs" / "bpnet_hitcall_low_confidence"
    log_dir.mkdir(parents=True, exist_ok=True)

    submitted = 0
    skipped_done = 0
    skipped_missing = 0
    skipped_reads = 0
    for exp_id in experiments:
        n_reads = read_counts.get(exp_id, 0)
        if n_reads < args.min_reads:
            skipped_reads += 1
            continue

        model_dir_name = exp_id

        for head in heads:
            cwm_trim_coords = (
                modisco_dir
                / f"{exp_id}_{head}_trim_coords_min{args.min_trim_len}bp.tsv"
                if args.min_trim_len is not None
                else None
            )
            suffix = trim_suffix(DEFAULT_CWM_TRIM_THRESHOLD, None, cwm_trim_coords)
            exp_dir = hitcalls_dir / f"{model_dir_name}_{head}"
            hits_dir = exp_dir / suffix.lstrip("_") if suffix else exp_dir

            hits_dedensified = hits_dir / "hits_dedensified.tsv"
            hits_unique = hits_dir / "hits_unique.tsv"
            if not hits_dedensified.exists() and not hits_unique.exists():
                skipped_missing += 1
                continue

            hits_confidence_filtered = hits_dir / "hits_confidence_filtered.tsv"
            if hits_confidence_filtered.exists():
                skipped_done += 1
                continue

            job_name = f"bpnet_hitcall_low_confidence_{exp_id}_{head}{suffix}"
            filter_cmd = (
                f"uv run --project {REPO_ROOT} --extra sherlock --frozen python {FILTER_SCRIPT} "
                f"-e {exp_id} --head {head} -v"
            )
            if args.min_trim_len is not None:
                filter_cmd += f" --min-trim-len {args.min_trim_len}"
            filter_cmd += f" {args.filter_args}"

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
                {filter_cmd}
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
        f"with <{args.min_reads:,} reads, skipped {skipped_missing} missing "
        f"hits_dedensified.tsv/hits_unique.tsv, skipped {skipped_done} "
        f"already filtered ({total} total)"
    )


if __name__ == "__main__":
    main()
