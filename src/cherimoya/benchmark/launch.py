#!/usr/bin/env python3
"""Submit SLURM jobs to benchmark trained Cherimoya models for all experiments.

Reads experiment IDs from configs/experiment_config.yaml and submits one
sbatch job per experiment via benchmark_cherimoya.py. Jobs are skipped if the
metrics JSON already exists or if any fold model is missing.

Jobs run natively on Sherlock using SRCC's py-pytorch/py-triton modules (see
src/cherimoya/sherlock_native/), not the Apptainer image, which cannot run on
Sherlock's GPU driver (see src/cherimoya/apptainer/README.md).

--local skips SLURM entirely and runs each experiment directly in the
foreground with inherited stdout/stderr, via `uv run --extra cherimoya`
instead of the Sherlock modules/venv -- useful on a GPU box you already have
a shell on (e.g. a lab cluster). SLURM resource flags
(--gpus/--partition/--cpus-per-task/--mem/--time) are ignored in this mode.

Usage:
    python src/cherimoya/benchmark/launch.py
    python src/cherimoya/benchmark/launch.py --dry-run
    python src/cherimoya/benchmark/launch.py --save-output
    python src/cherimoya/benchmark/launch.py --min-reads 20000000
    python src/cherimoya/benchmark/launch.py --benchmark-args '-b 128'
    python src/cherimoya/benchmark/launch.py --local
    python src/cherimoya/benchmark/launch.py --local --dry-run
"""

import argparse
import os
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
BENCHMARK_SCRIPT = REPO_ROOT / "src" / "cherimoya" / "benchmark" / "benchmark_cherimoya.py"
# Resolved by bash at job runtime (not by Python at submission time) so these
# work under any Sherlock username, matching setup_env.sh/test_install.sh.
NUMBA_CACHE_DIR = "${NUMBA_CACHE_DIR:-/scratch/users/${USER}/numba_cache}"
VENV_DIR = "${CHERIMOYA_VENV_DIR:-/scratch/users/${USER}/venvs/cherimoya-sherlock}"
PYTORCH_MODULE = "py-pytorch/2.9.1_py314"
TRITON_MODULE = "py-triton/3.5.1_py314"
# --local doesn't run on Sherlock, so it has no reason to default into
# Sherlock's scratch layout; a repo-relative cache dir works anywhere.
LOCAL_NUMBA_CACHE_DIR = REPO_ROOT / ".cache" / "numba"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print sbatch scripts (or local commands, with --local) without running them",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help=(
            "run each experiment directly in the foreground with inherited "
            "stdout/stderr via `uv run --extra cherimoya`, instead of "
            "submitting SLURM jobs; ignores SLURM resource flags "
            "(--gpus/--partition/--cpus-per-task/--mem/--time)"
        ),
    )
    parser.add_argument(
        "--save-output",
        action="store_true",
        help="forward --save-output to benchmark_cherimoya.py",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="submit even when the metrics JSON already exists",
    )
    parser.add_argument(
        "--model-dir-root",
        type=Path,
        default=REPO_ROOT / "models" / "cherimoya",
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
        help="extra arguments forwarded to benchmark_cherimoya.py",
    )
    # SLURM resource flags
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
                "GPU_SKU:RTX_3090",
            ]
        ),
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

    log_dir = REPO_ROOT / "logs" / "cherimoya_benchmark"
    log_dir.mkdir(parents=True, exist_ok=True)

    local_env = None
    if args.local:
        local_env = dict(os.environ)
        local_env.setdefault("NUMBA_CACHE_DIR", str(LOCAL_NUMBA_CACHE_DIR))
        Path(local_env["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

    submitted = 0
    skipped_reads = 0
    skipped_untrained = 0
    skipped_done = 0
    local_failures = []

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

        metrics_path = (
            REPO_ROOT / "performance_metrics" / "cherimoya" / f"{model_dir.name}.json"
        )
        if metrics_path.exists() and not args.force:
            skipped_done += 1
            continue

        if args.local:
            cmd = [
                "uv",
                "run",
                "--project",
                str(REPO_ROOT),
                "--extra",
                "cherimoya",
                "--frozen",
                "python3",
                str(BENCHMARK_SCRIPT),
                "-e",
                exp_id,
                "--model-dir",
                str(model_dir),
                "-v",
            ]
            if args.save_output:
                cmd.append("--save-output")
            if args.benchmark_args:
                cmd.extend(shlex.split(args.benchmark_args))

            if args.dry_run:
                print(shlex.join(cmd))
                submitted += 1
                continue

            print(f"=== Benchmarking {exp_id} ===")
            result = subprocess.run(cmd, cwd=REPO_ROOT, env=local_env)
            if result.returncode != 0:
                local_failures.append((exp_id, result.returncode))
                print(
                    f"ERROR: {exp_id} exited with {result.returncode}",
                    file=sys.stderr,
                )
            submitted += 1
            continue

        job_name = f"cherimoya_bench_{exp_id}"
        benchmark_cmd = (
            f'python3 "$BENCHMARK_SCRIPT" -e {shlex.quote(exp_id)}'
            f" --model-dir {shlex.quote(str(model_dir))} -v"
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

            mkdir -p "{NUMBA_CACHE_DIR}"
            export NUMBA_CACHE_DIR="{NUMBA_CACHE_DIR}"
            BENCHMARK_SCRIPT={shlex.quote(str(BENCHMARK_SCRIPT))}

            ml load math
            ml load {PYTORCH_MODULE} {TRITON_MODULE}
            source "{VENV_DIR}/bin/activate"

            nvidia-smi -L
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

    if args.local:
        action = "Would run" if args.dry_run else "Ran"
        unit = "experiments"
    else:
        action = "Would submit" if args.dry_run else "Submitted"
        unit = "jobs"
    print(
        f"\n{action} {submitted} {unit}, skipped {skipped_reads} experiments "
        f"with <{args.min_reads:,} reads, skipped {skipped_untrained} with "
        f"incomplete training, skipped {skipped_done} already benchmarked"
    )
    if args.local and not args.dry_run:
        if local_failures:
            print(f"{len(local_failures)} experiment(s) failed:")
            for exp_id, returncode in local_failures:
                print(f"  {exp_id} (exit {returncode})")
        else:
            print("All experiments completed successfully.")


if __name__ == "__main__":
    main()
