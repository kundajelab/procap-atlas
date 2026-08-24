#!/usr/bin/env python3
"""Submit SLURM jobs to run report_bpnet.py on completed Fi-NeMo hit calls.

Reads experiment IDs from configs/experiment_config.yaml and submits one
sbatch job per (experiment, head) pair via report_bpnet.py, which runs
`finemo report --no-recall` and filters hits by per-motif cwm_similarity.

Jobs are skipped if hits_filtered.tsv already exists or if hits_unique.tsv is
missing (run call_hits_bpnet.py/hitcall/launch.py first). This step does not
use a GPU, unlike hit calling itself, so it runs as a separate, cheaper
launcher -- mirroring modisco/launch.py vs modisco/launch_report.py. If
hitcall/launch.py was run with --min-trim-len, pass the same value here so
this launcher looks in the matching trim-coords output directory.

Usage:
    python src/bpnet/hitcall/launch_report.py                    # submit all experiments, profile head
    python src/bpnet/hitcall/launch_report.py --dry-run           # print sbatch scripts without submitting
    python src/bpnet/hitcall/launch_report.py --head count        # count head only
    python src/bpnet/hitcall/launch_report.py --head profile --head count  # both heads
    python src/bpnet/hitcall/launch_report.py --min-reads 20000000
    python src/bpnet/hitcall/launch_report.py --report-args '--cwm-similarity-threshold 0.85'
    python src/bpnet/hitcall/launch_report.py --min-trim-len 6  # match hitcall/launch.py's floor
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
REPORT_SCRIPT = REPO_ROOT / "src" / "bpnet" / "hitcall" / "report_bpnet.py"


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
        help="attribution/motif head(s) to report on; repeatable (default: profile)",
    )
    # SLURM resource flags
    parser.add_argument("--partition", type=str, default="normal,akundaje,owners")
    parser.add_argument("--cpus-per-task", type=int, default=4)
    parser.add_argument("--mem", type=str, default="64G")
    parser.add_argument("--time", type=str, default="2:00:00")
    parser.add_argument(
        "--min-reads",
        type=int,
        default=0,
        help="skip experiments with fewer total reads than this (default: 0, disabled)",
    )
    parser.add_argument(
        "--report-args",
        type=str,
        default="",
        help="extra arguments forwarded to report_bpnet.py (e.g. '--cwm-similarity-threshold 0.85')",
    )
    parser.add_argument(
        "--min-trim-len",
        type=int,
        default=None,
        metavar="BP",
        help=(
            "must match the value hitcall/launch.py was run with, if any -- "
            "resolves the same trim-coords-suffixed output directory rather "
            "than the plain {model_dir_name}_{head}/ one."
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
    mc_dir = REPO_ROOT / "motifcompendium" / "bpnet"
    log_dir = REPO_ROOT / "logs" / "bpnet_hitcall_report"
    log_dir.mkdir(parents=True, exist_ok=True)

    trim_coords_by_head = {}
    if args.min_trim_len is not None:
        for head in heads:
            trim_coords_by_head[head] = (
                mc_dir
                / f"motifcompendium_{head}_trim_coords_min{args.min_trim_len}bp.tsv"
            )

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
                trim_coords_by_head[head] if args.min_trim_len is not None else None
            )
            suffix = trim_suffix(DEFAULT_CWM_TRIM_THRESHOLD, None, cwm_trim_coords)
            exp_dir = hitcalls_dir / f"{model_dir_name}_{head}"
            hits_dir = exp_dir / suffix.lstrip("_") if suffix else exp_dir
            hits_tsv = hits_dir / "hits_unique.tsv"
            if not hits_tsv.exists():
                skipped_missing += 1
                continue

            hits_filtered = hits_dir / "hits_filtered.tsv"
            if hits_filtered.exists():
                skipped_done += 1
                continue

            job_name = f"bpnet_hitcall_report_{exp_id}_{head}"
            report_cmd = (
                f"uv run --project {REPO_ROOT} --extra sherlock --frozen python {REPORT_SCRIPT} "
                f"-e {exp_id} --head {head} -v"
            )
            if cwm_trim_coords is not None:
                report_cmd += f" --cwm-trim-coords {cwm_trim_coords}"
            report_cmd += f" {args.report_args}"

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
                {report_cmd}
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
        f"hits_unique.tsv, skipped {skipped_done} already reported ({total} total)"
    )


if __name__ == "__main__":
    main()
