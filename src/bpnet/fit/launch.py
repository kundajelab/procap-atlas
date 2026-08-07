#!/usr/bin/env python3
"""Submit SLURM jobs to train BPNet models for all experiments and folds.

Reads experiment IDs from configs/experiment_config.yaml and submits one
sbatch job per (experiment, fold) pair via fit_bpnet.py.

Experiments with fewer total reads than --min-reads (default: 10_000_000) are
skipped, as low-coverage experiments tend to produce poorly calibrated models.
Folds with a completed model are also skipped automatically.

Jobs are submitted with --requeue, so a pre-empted job (the akundaje/owners
partitions are preemptible) is automatically resubmitted by SLURM instead of
needing a manual resubmit. Since each job trains a single (experiment, fold)
pair from scratch (bpnet-lite's fit() has no resume support), a requeued job
just retrains that fold from epoch 0 -- but it still re-checks for a
completed model right before doing so, in case the job actually finished
just before being marked pre-empted. "Completed" means a
`{exp_id}.fold{fold}.final.torch` file, which bpnet-lite's fit() writes
exactly once, at the very end of training -- `{exp_id}.fold{fold}.torch` (no
`.final`) is the wrong artifact for this, since it's overwritten throughout
training whenever validation loss improves and can already exist after a
single epoch.

Usage:
    python src/bpnet/fit/launch.py                    # submit all experiments x 7 folds
    python src/bpnet/fit/launch.py --dry-run           # print sbatch scripts without submitting
    python src/bpnet/fit/launch.py --time 12:00:00 --mem 32G --partition gpu
    python src/bpnet/fit/launch.py --min-reads 20000000  # only well-covered experiments
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
FIT_SCRIPT = REPO_ROOT / "src" / "bpnet" / "fit" / "fit_bpnet.py"


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
        default="GPU_GEN:AMP|GPU_GEN:LOV|GPU_GEN:HPR",
    )
    parser.add_argument("--partition", type=str, default="akundaje,owners")
    parser.add_argument("--cpus-per-task", type=int, default=4)
    parser.add_argument("--mem", type=str, default="32G")
    parser.add_argument("--time", type=str, default="24:00:00")
    parser.add_argument(
        "--min-reads",
        type=int,
        default=10_000_000,
        help="skip experiments with fewer total reads than this (default: 10000000)",
    )
    # Extra args forwarded to fit_bpnet.py
    parser.add_argument(
        "--fit-args",
        type=str,
        default="",
        help="extra arguments forwarded to fit.py (e.g. '--max-epochs 100')",
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

    log_dir = REPO_ROOT / "logs" / "bpnet_fit"
    log_dir.mkdir(parents=True, exist_ok=True)

    submitted = 0
    skipped_trained = 0
    skipped_reads = 0
    for exp_id in experiments:
        # Skip experiments with too few reads
        n_reads = read_counts.get(exp_id, 0)
        if n_reads < args.min_reads:
            skipped_reads += 1
            continue

        for fold in range(n_folds):
            # Skip if model already trained. `.final.torch` is written
            # exactly once, at the very end of bpnet-lite's fit() -- unlike
            # `.torch` (no `.final`), which is overwritten throughout
            # training whenever validation loss improves, so checking that
            # one would treat a fold pre-empted after a single epoch as
            # complete.
            model_dir = REPO_ROOT / "models" / "bpnet" / exp_id
            model_path = model_dir / f"{exp_id}.fold{fold}.final.torch"
            if model_path.exists():
                skipped_trained += 1
                continue

            job_name = f"bpnet_{exp_id}_f{fold}"
            fit_cmd = (
                f"uv run --project {REPO_ROOT} --extra sherlock --frozen python "
                f"{FIT_SCRIPT} -e {exp_id} --fold {fold} -v"
            )
            if args.fit_args:
                fit_cmd += f" {args.fit_args}"

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
                #SBATCH --requeue
                #SBATCH --open-mode=append
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

                MODEL_PATH={shlex.quote(str(model_path))}
                if [[ -f "$MODEL_PATH" ]]; then
                    echo "Already completed: $MODEL_PATH"
                    exit 0
                fi

                nvidia-smi -L
                {fit_cmd}
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
        f"with <{args.min_reads:,} reads, skipped {skipped_trained} already trained "
        f"({total} total)"
    )


if __name__ == "__main__":
    main()
