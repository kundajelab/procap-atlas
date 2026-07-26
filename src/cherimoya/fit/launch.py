#!/usr/bin/env python3
"""Submit SLURM jobs to train Cherimoya models for all experiments and folds.

Reads experiment IDs from configs/experiment_config.yaml and submits one
sbatch job per experiment via fit_cherimoya.py, running that experiment's
folds sequentially within the job.

Experiments with fewer total reads than --min-reads (default: 10_000_000) are
skipped, as low-coverage experiments tend to produce poorly calibrated models.
Folds with an already-trained model file are also skipped automatically; an
experiment with every fold already trained is not submitted at all.

Usage:
    python src/cherimoya/fit/launch.py                    # submit one job per experiment, 7 folds each
    python src/cherimoya/fit/launch.py --dry-run           # print sbatch scripts without submitting
    python src/cherimoya/fit/launch.py --time 48:00:00 --mem 32G --partition gpu
    python src/cherimoya/fit/launch.py --min-reads 20000000  # only well-covered experiments
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
FIT_SCRIPT = REPO_ROOT / "src" / "cherimoya" / "fit" / "fit_cherimoya.py"
APPTAINER_IMAGE = Path("/scratch/users/ayhe/apptainer/cherimoya.sif")
NUMBA_CACHE_DIR = Path("/scratch/users/ayhe/numba_cache")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print sbatch scripts without submitting",
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
            ]
        ),
    )
    parser.add_argument("--partition", type=str, default="akundaje,owners")
    parser.add_argument("--cpus-per-task", type=int, default=4)
    parser.add_argument("--mem", type=str, default="32G")
    parser.add_argument(
        "--time",
        type=str,
        default="48:00:00",
        help=(
            "time budget for the whole job (HH:MM:SS), regardless of how "
            "many folds it trains; the owners partition caps jobs at "
            "48:00:00 (default: 48:00:00)"
        ),
    )
    parser.add_argument(
        "--apptainer-image",
        type=Path,
        default=APPTAINER_IMAGE,
        help=f"Apptainer image used to run Cherimoya training (default: {APPTAINER_IMAGE})",
    )
    parser.add_argument(
        "--apptainer-bind",
        action="append",
        default=["/oak/stanford/groups/akundaje/ayhe", "/scratch/users/ayhe"],
        help=(
            "path to bind into the Apptainer container; may be repeated "
            "(default: /oak and /scratch)"
        ),
    )
    parser.add_argument("--apptainer-env", action="append", default=[""])
    parser.add_argument(
        "--min-reads",
        type=int,
        default=0,
        help="skip experiments with fewer total reads than this (default: 0)",
    )
    # Extra args forwarded to fit_cherimoya.py
    parser.add_argument(
        "--fit-args",
        type=str,
        default="",
        help="extra arguments forwarded to fit_cherimoya.py (e.g. '--max-epochs 100')",
    )
    args = parser.parse_args()

    # Load experiment list
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    experiments = list(config["experiments"].keys())

    # Load fold count
    with open(CHROM_SPLITS_PATH) as f:
        chrom_splits = yaml.safe_load(f)
    n_folds = len(chrom_splits["folds"])

    # Load read counts for filtering
    read_counts_df = pd.read_csv(
        N_READS_PATH, sep="\t", usecols=["experiment", "total_reads"]
    )
    read_counts = dict(zip(read_counts_df["experiment"], read_counts_df["total_reads"]))

    log_dir = REPO_ROOT / "logs" / "cherimoya_fit"
    log_dir.mkdir(parents=True, exist_ok=True)

    submitted = 0
    skipped_trained = 0
    skipped_reads = 0
    skipped_complete = 0
    for exp_id in experiments:
        # Skip experiments with too few reads
        n_reads = read_counts.get(exp_id, 0)
        if n_reads < args.min_reads:
            skipped_reads += 1
            continue

        # Skip folds that are already trained
        model_dir = REPO_ROOT / "models" / "cherimoya" / exp_id
        folds_to_run = [
            fold
            for fold in range(n_folds)
            if not (model_dir / f"{exp_id}.fold{fold}.torch").exists()
        ]
        skipped_trained += n_folds - len(folds_to_run)
        if not folds_to_run:
            skipped_complete += 1
            continue

        job_name = f"cherimoya_{exp_id}"
        bind_args = " ".join(
            f"--bind {shlex.quote(path)}" for path in args.apptainer_bind
        )
        extra_fit_args = ""
        if args.fit_args:
            extra_fit_args = " " + args.fit_args
        apptainer_cmds = "\n".join(
            f'apptainer exec --nv {bind_args} "$APPTAINER_IMAGE" '
            f'python "$FIT_SCRIPT" -e {shlex.quote(exp_id)} --fold {fold} -v'
            f"{extra_fit_args}"
            for fold in folds_to_run
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
            #SBATCH --output={log_dir}/{job_name}.out
            #SBATCH --error={log_dir}/{job_name}.err

            mkdir -p {NUMBA_CACHE_DIR}
            FIT_SCRIPT={shlex.quote(str(FIT_SCRIPT))}
            APPTAINER_IMAGE={shlex.quote(str(args.apptainer_image))}

            nvidia-smi -L
            {apptainer_cmds}
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
    total = len(experiments) * n_folds
    print(
        f"\n{action} {submitted} jobs, skipped {skipped_reads} experiments "
        f"with <{args.min_reads:,} reads, skipped {skipped_complete} fully-trained "
        f"experiments, skipped {skipped_trained} already-trained folds "
        f"({total} total folds)"
    )


if __name__ == "__main__":
    main()
