#!/usr/bin/env python3
"""Submit Cherimoya n_filters sweep benchmark jobs.

Each submitted job is an 8-task SLURM array, one task for each trained
n_filters model directory for a single experiment.

Usage:
    python src/cherimoya/n_filters/launch_benchmark.py
    python src/cherimoya/n_filters/launch_benchmark.py --dry-run
    python src/cherimoya/n_filters/launch_benchmark.py --min-reads 20000000
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
BENCHMARK_SCRIPT = (
    REPO_ROOT / "src" / "cherimoya" / "benchmark" / "benchmark_cherimoya.py"
)
MODEL_ROOT = REPO_ROOT / "models" / "cherimoya_n_filters"
METRICS_DIR = REPO_ROOT / "performance_metrics" / "cherimoya"
APPTAINER_IMAGE = Path("/scratch/users/ayhe/apptainer/cherimoya.sif")
NUMBA_CACHE_DIR = Path("/scratch/users/ayhe/numba_cache")
N_FILTERS = [16, 24, 36, 48, 64, 96, 196, 256]


def all_benchmarks_exist(exp_id: str) -> bool:
    """Return True if every n_filters benchmark JSON exists for an experiment."""
    return all(
        (METRICS_DIR / f"{exp_id}_nf{n_filters}.json").exists()
        for n_filters in N_FILTERS
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print sbatch scripts without submitting",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="|".join(
            [
                "GPU_SKU:A100_PCIE",
                "GPU_SKU:A100_SXM4",
                "GPU_SKU:A40",
                "GPU_SKU:H100_SXM5",
                "GPU_SKU:H200_SXM5",
                "GPU_SKU:L40S",
            ]
        ),
    )
    parser.add_argument("--partition", type=str, default="akundaje,owners")
    parser.add_argument("--cpus-per-task", type=int, default=1)
    parser.add_argument("--mem", type=str, default="32G")
    parser.add_argument("--time", type=str, default="6:00:00")
    parser.add_argument(
        "--apptainer-image",
        type=Path,
        default=APPTAINER_IMAGE,
        help=f"Apptainer image used to run benchmarks (default: {APPTAINER_IMAGE})",
    )
    parser.add_argument(
        "--apptainer-bind",
        action="append",
        default=["/oak/stanford/groups/akundaje/ayhe", "/scratch/users/ayhe"],
        help=(
            "path to bind into the Apptainer container; may be repeated "
            "(default: /oak/stanford/groups/akundaje/ayhe and /scratch/users/ayhe)"
        ),
    )
    parser.add_argument(
        "--min-reads",
        type=int,
        default=0,
        help="skip experiments with fewer total reads than this (default: 0)",
    )
    parser.add_argument(
        "--benchmark-args",
        type=str,
        default="",
        help="extra arguments forwarded to benchmark_cherimoya.py (e.g. '-b 128')",
    )
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

    log_dir = REPO_ROOT / "logs" / "cherimoya_n_filters_benchmark"
    log_dir.mkdir(parents=True, exist_ok=True)

    bind_args = " ".join(f"--bind {shlex.quote(path)}" for path in args.apptainer_bind)
    n_filters_bash = " ".join(str(n) for n in N_FILTERS)
    extra_benchmark_args = f" {args.benchmark_args}" if args.benchmark_args else ""

    submitted = 0
    skipped_benchmarked = 0
    skipped_reads = 0
    for exp_id in experiments:
        n_reads = read_counts.get(exp_id, 0)
        if n_reads < args.min_reads:
            skipped_reads += 1
            continue

        if all_benchmarks_exist(exp_id):
            skipped_benchmarked += 1
            continue

        job_name = f"cherimoya_nf_bench_{exp_id}"
        benchmark_cmd = (
            f'apptainer exec --nv {bind_args} "$APPTAINER_IMAGE" '
            f'python "$BENCHMARK_SCRIPT" -e "$EXP_ID" --model-dir "$MODEL_DIR" -v'
            f"{extra_benchmark_args}"
        )

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
            #SBATCH --array=0-{len(N_FILTERS) - 1}
            #SBATCH --output={log_dir}/{job_name}_%A_%a.out
            #SBATCH --error={log_dir}/{job_name}_%A_%a.err

            set -euo pipefail

            N_FILTERS_VALUES=({n_filters_bash})
            TASK_ID="${{SLURM_ARRAY_TASK_ID}}"
            N_FILTERS_VALUE="${{N_FILTERS_VALUES[$TASK_ID]}}"

            EXP_ID={shlex.quote(exp_id)}
            REPO_ROOT={shlex.quote(str(REPO_ROOT))}
            MODEL_ROOT={shlex.quote(str(MODEL_ROOT))}
            METRICS_DIR={shlex.quote(str(METRICS_DIR))}
            BENCHMARK_SCRIPT={shlex.quote(str(BENCHMARK_SCRIPT))}
            APPTAINER_IMAGE={shlex.quote(str(args.apptainer_image))}
            NUMBA_CACHE_DIR={shlex.quote(str(NUMBA_CACHE_DIR))}
            MODEL_DIR="$MODEL_ROOT/${{EXP_ID}}_nf${{N_FILTERS_VALUE}}"
            METRICS_PATH="$METRICS_DIR/${{EXP_ID}}_nf${{N_FILTERS_VALUE}}.json"

            mkdir -p "$NUMBA_CACHE_DIR"
            export NUMBA_CACHE_DIR
            export APPTAINERENV_NUMBA_CACHE_DIR="$NUMBA_CACHE_DIR"

            if [[ -f "$METRICS_PATH" ]]; then
                echo "Skipping existing benchmark: $METRICS_PATH"
                exit 0
            fi

            if [[ ! -d "$MODEL_DIR" ]]; then
                echo "Skipping missing model directory: $MODEL_DIR"
                exit 0
            fi

            missing=0
            for fold in $(seq 0 {n_folds - 1}); do
                model_path="$MODEL_DIR/${{EXP_ID}}.fold${{fold}}.torch"
                if [[ ! -f "$model_path" ]]; then
                    echo "Missing model checkpoint: $model_path"
                    missing=1
                fi
            done
            if [[ "$missing" -ne 0 ]]; then
                echo "Skipping incomplete model directory: $MODEL_DIR"
                exit 0
            fi

            cd "$REPO_ROOT"
            nvidia-smi -L
            echo "Benchmarking $EXP_ID with n_filters=$N_FILTERS_VALUE"
            {benchmark_cmd}
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
                f"ERROR submitting {job_name}: {result.stderr.strip()}", file=sys.stderr
            )

    action = "Would submit" if args.dry_run else "Submitted"
    total_tasks = len(experiments) * len(N_FILTERS)
    print(
        f"\n{action} {submitted} experiment benchmark arrays "
        f"({submitted * len(N_FILTERS)} tasks), skipped {skipped_reads} experiments "
        f"with <{args.min_reads:,} reads, skipped {skipped_benchmarked} fully "
        f"benchmarked experiments ({total_tasks} possible tasks)"
    )


if __name__ == "__main__":
    main()
