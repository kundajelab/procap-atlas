#!/usr/bin/env python3
"""Submit SLURM jobs to generate final BPNet predicted PRO-cap tracks.

Reads experiment IDs from configs/experiment_config.yaml and submits one GPU job
per experiment via generate_predicted_tracks.py. Jobs are skipped if filtered
peaks or any fold checkpoint is missing, or if both output BigWigs already
exist unless --force is provided.

Usage:
    python src/bpnet/predict/launch.py
    python src/bpnet/predict/launch.py --dry-run
    python src/bpnet/predict/launch.py --min-reads 10000000
    python src/bpnet/predict/launch.py --prediction-args '--batch-size 32'
"""

from __future__ import annotations

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
PREDICTION_SCRIPT = (
    REPO_ROOT / "src" / "bpnet" / "predict" / "generate_predicted_tracks.py"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "predictions" / "bpnet" / "bigwigs"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="submit even when both predicted BigWigs already exist",
    )
    parser.add_argument(
        "--model-dir-root",
        type=Path,
        default=REPO_ROOT / "models" / "bpnet",
        help="directory containing one model subdirectory per experiment",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="directory for predicted plus/minus BigWigs",
    )
    parser.add_argument(
        "--min-reads",
        type=int,
        default=0,
        help="skip experiments with fewer total reads than this (default: 0, disabled)",
    )
    parser.add_argument(
        "--prediction-args",
        type=str,
        default="",
        help="extra arguments forwarded to generate_predicted_tracks.py",
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
    experiments = config["experiments"]

    with open(CHROM_SPLITS_PATH) as f:
        chrom_splits = yaml.safe_load(f)
    n_folds = len(chrom_splits["folds"])

    read_counts_df = pd.read_csv(
        N_READS_PATH, sep="\t", usecols=["experiment", "total_reads"]
    )
    read_counts = dict(
        zip(read_counts_df["experiment"], read_counts_df["total_reads"])
    )
    log_dir = REPO_ROOT / "logs" / "bpnet_predict_tracks"
    log_dir.mkdir(parents=True, exist_ok=True)

    submitted = 0
    skipped_reads = 0
    skipped_missing_peaks = 0
    skipped_untrained = 0
    skipped_done = 0

    for exp_id, exp in experiments.items():
        if read_counts.get(exp_id, 0) < args.min_reads:
            skipped_reads += 1
            continue

        processed = exp.get("processed", {})
        peaks_rel = processed.get("filtered_peaks")
        if peaks_rel is None or not (REPO_ROOT / peaks_rel).exists():
            skipped_missing_peaks += 1
            continue

        model_dir = args.model_dir_root / exp_id
        missing_models = [
            model_dir / f"{exp_id}.fold{fold}.torch"
            for fold in range(n_folds)
            if not (model_dir / f"{exp_id}.fold{fold}.torch").exists()
        ]
        if missing_models:
            skipped_untrained += 1
            continue

        plus_output = args.output_dir / f"{model_dir.name}_pl.bigWig"
        minus_output = args.output_dir / f"{model_dir.name}_mn.bigWig"
        if plus_output.exists() and minus_output.exists() and not args.force:
            skipped_done += 1
            continue

        job_name = f"bpnet_predict_{exp_id}"
        prediction_cmd = (
            f"uv run --project {shlex.quote(str(REPO_ROOT))} --extra sherlock --frozen "
            f"python {shlex.quote(str(PREDICTION_SCRIPT))}"
            f" -e {shlex.quote(exp_id)}"
            f" --model-dir {shlex.quote(str(model_dir))}"
            f" --output-dir {shlex.quote(str(args.output_dir))}"
            " -v"
        )
        if args.prediction_args:
            prediction_cmd += f" {args.prediction_args}"

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
            mkdir -p {args.output_dir}
            time {prediction_cmd}
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
        f"with <{args.min_reads:,} reads, skipped {skipped_missing_peaks} missing "
        f"filtered peaks, skipped {skipped_untrained} with incomplete training, "
        f"skipped {skipped_done} already done"
    )


if __name__ == "__main__":
    main()
