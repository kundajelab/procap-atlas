#!/usr/bin/env python3
"""Submit SLURM jobs to train BNBPNet models for all experiments and folds.

Reads experiment IDs from configs/experiment_config.yaml and submits one
sbatch job per (experiment, fold) pair.

Usage:
    python src/bpnet/fit/launch.py                    # submit all experiments x 7 folds
    python src/bpnet/fit/launch.py --dry-run           # print sbatch scripts without submitting
    python src/bpnet/fit/launch.py --time 12:00:00 --mem 32G --partition gpu
"""

import argparse
import subprocess
import sys
import textwrap
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "experiment_config.yaml"
CHROM_SPLITS_PATH = REPO_ROOT / "configs" / "chrom_splits.yaml"
FIT_SCRIPT = REPO_ROOT / "src" / "bpnet" / "fit" / "fit.py"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print sbatch scripts without submitting",
    )
    # SLURM resource flags
    parser.add_argument("--partition", type=str, default="akundaje")
    parser.add_argument("--gpus", type=str, default="1")
    parser.add_argument("--cpus-per-task", type=int, default=1)
    parser.add_argument("--mem", type=str, default="64G")
    parser.add_argument("--time", type=str, default="24:00:00")
    # Extra args forwarded to fit.py
    parser.add_argument(
        "--fit-args",
        type=str,
        default="",
        help="extra arguments forwarded to fit.py (e.g. '--max-epochs 50 --lr 0.001')",
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

    log_dir = REPO_ROOT / "logs" / "bpnet"
    log_dir.mkdir(parents=True, exist_ok=True)

    submitted = 0
    skipped = 0
    for exp_id in experiments:
        for fold in range(n_folds):
            # Skip if model already trained
            model_dir = REPO_ROOT / "models" / "bpnet" / exp_id
            model_path = model_dir / f"{exp_id}.fold{fold}.final.torch"
            if model_path.exists():
                skipped += 1
                continue

            job_name = f"bpnet_{exp_id}_f{fold}"
            fit_cmd = f"python {FIT_SCRIPT} -e {exp_id} --fold {fold}"
            if args.fit_args:
                fit_cmd += f" {args.fit_args}"

            sbatch_script = textwrap.dedent(f"""\
                #!/bin/bash
                #SBATCH --job-name={job_name}
                #SBATCH --partition={args.partition}
                #SBATCH --gpus={args.gpus}
                #SBATCH --cpus-per-task={args.cpus_per_task}
                #SBATCH --mem={args.mem}
                #SBATCH --time={args.time}
                #SBATCH --output={log_dir}/{job_name}.out
                #SBATCH --error={log_dir}/{job_name}.err

                {fit_cmd}
            """)

            if args.dry_run:
                print(f"--- {job_name} ---")
                print(sbatch_script)
                submitted += 1
                continue

            result = subprocess.run(
                ["sbatch"],
                input=sbatch_script,
                capture_output=True,
                text=True,
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
    print(f"\n{action} {submitted} jobs, skipped {skipped} already trained ({total} total)")


if __name__ == "__main__":
    main()
