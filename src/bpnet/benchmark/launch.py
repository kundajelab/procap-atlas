#!/usr/bin/env python3
"""Submit SLURM jobs to benchmark trained BPNet models for all experiments.

Reads experiment IDs from configs/experiment_config.yaml and submits one
sbatch job per experiment via benchmark_bpnet.py. Jobs are skipped if the
metrics JSON already exists or if any fold model is missing.

Usage:
    python src/bpnet/benchmark/launch.py
    python src/bpnet/benchmark/launch.py --dry-run
    python src/bpnet/benchmark/launch.py --save-output
    python src/bpnet/benchmark/launch.py --min-reads 20000000
    python src/bpnet/benchmark/launch.py --benchmark-args '--batch-size 32'
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
CHROM_SPLITS_PATH = REPO_ROOT / "configs" / "chrom_splits.yaml"
N_READS_PATH = REPO_ROOT / "configs" / "n_reads.txt"
BENCHMARK_SCRIPT = REPO_ROOT / "src" / "bpnet" / "benchmark" / "benchmark_bpnet.py"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--save-output",
        action="store_true",
        help="forward --save-output to benchmark_bpnet.py",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="submit even when the metrics JSON already exists",
    )
    parser.add_argument(
        "--model-dir-root",
        type=Path,
        default=REPO_ROOT / "models" / "bpnet",
        help="directory containing one model subdirectory per experiment",
    )
    parser.add_argument(
        "--min-reads",
        type=int,
        default=0,
        help="skip experiments with fewer total reads than this (default: 0, disabled)",
    )
    parser.add_argument(
        "--benchmark-args",
        type=str,
        default="",
        help="extra arguments forwarded to benchmark_bpnet.py",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="GPU_GEN:AMP|GPU_GEN:LOV|GPU_GEN:HPR",
    )
    parser.add_argument("--partition", type=str, default="akundaje,owners")
    parser.add_argument("--cpus-per-task", type=int, default=4)
    parser.add_argument("--mem", type=str, default="32G")
    parser.add_argument("--time", type=str, default="6:00:00")
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    experiments = list(config["experiments"].keys())

    with open(CHROM_SPLITS_PATH) as f:
        chrom_splits = yaml.safe_load(f)
    n_folds = len(chrom_splits["folds"])

    read_counts_df = pd.read_csv(
        N_READS_PATH, sep="\t", usecols=["experiment", "total_reads"]
    )
    read_counts = dict(zip(read_counts_df["experiment"], read_counts_df["total_reads"]))

    log_dir = REPO_ROOT / "logs" / "bpnet_benchmark"
    log_dir.mkdir(parents=True, exist_ok=True)

    submitted = 0
    skipped_reads = 0
    skipped_untrained = 0
    skipped_done = 0

    for exp_id in experiments:
        if read_counts.get(exp_id, 0) < args.min_reads:
            skipped_reads += 1
            continue

        model_dir = args.model_dir_root / exp_id
        missing = [
            model_dir / f"{exp_id}.fold{fold}.torch"
            for fold in range(n_folds)
            if not (model_dir / f"{exp_id}.fold{fold}.torch").exists()
        ]
        if missing:
            skipped_untrained += 1
            continue

        metrics_path = REPO_ROOT / "performance_metrics" / "bpnet" / f"{model_dir.name}.json"
        if metrics_path.exists() and not args.force:
            skipped_done += 1
            continue

        job_name = f"bpnet_bench_{exp_id}"
        benchmark_cmd = (
            f"uv run --project {shlex.quote(str(REPO_ROOT))} --extra sherlock --frozen "
            f"python {shlex.quote(str(BENCHMARK_SCRIPT))}"
            f" -e {shlex.quote(exp_id)}"
            f" --model-dir {shlex.quote(str(model_dir))}"
            " -v"
        )
        if args.save_output:
            benchmark_cmd += " --save-output"
        if args.benchmark_args:
            benchmark_cmd += f" {args.benchmark_args}"

        sbatch_script = textwrap.dedent(f"""\
            #!/bin/bash -l
            #SBATCH --job-name={job_name}
            #SBATCH --ntasks=1
            #SBATCH --ntasks-per-node=1
            #SBATCH --nodes=1
            #SBATCH --gpus=1
            #SBATCH -C {args.gpus}
            #SBATCH --cpus-per-task={args.cpus_per_task}
            #SBATCH --mem={args.mem}
            #SBATCH --partition={args.partition}
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

            mamba activate "${{PROCAP_ATLAS_ENV:-procap-atlas}}"
            nvidia-smi -L
            cd {REPO_ROOT}
            time {benchmark_cmd}
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
            print(f"ERROR submitting {job_name}: {result.stderr.strip()}", file=sys.stderr)

    action = "Would submit" if args.dry_run else "Submitted"
    print(
        f"\n{action} {submitted} jobs, skipped {skipped_reads} experiments "
        f"with <{args.min_reads:,} reads, skipped {skipped_untrained} with "
        f"incomplete training, skipped {skipped_done} already benchmarked"
    )


if __name__ == "__main__":
    main()
