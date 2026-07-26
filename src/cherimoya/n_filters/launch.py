#!/usr/bin/env python3
"""Submit Cherimoya n_filters sweep jobs as one SLURM job per experiment.

Each submitted job loops over all n_filters values and folds for a single
experiment. This keeps the number of queued SLURM jobs bounded by the number of
experiments instead of the number of experiment/filter/fold combinations.

Jobs run natively on Sherlock using SRCC's py-pytorch/py-triton modules (see
src/cherimoya/sherlock_native/), not the Apptainer image, which cannot run on
Sherlock's GPU driver (see src/cherimoya/apptainer/README.md).

Usage:
    python src/cherimoya/n_filters/launch.py
    python src/cherimoya/n_filters/launch.py --dry-run
    python src/cherimoya/n_filters/launch.py --min-reads 20000000
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
MODEL_ROOT = REPO_ROOT / "models" / "cherimoya_n_filters"
NUMBA_CACHE_DIR = Path("/scratch/users/ayhe/numba_cache")
# Resolved by bash at job runtime (not by Python at submission time) so it
# works under any Sherlock username, matching setup_env.sh/test_install.sh.
VENV_DIR = "${CHERIMOYA_VENV_DIR:-/scratch/users/${USER}/venvs/cherimoya-sherlock}"
PYTORCH_MODULE = "py-pytorch/2.9.1_py314"
TRITON_MODULE = "py-triton/3.5.1_py314"
N_FILTERS = [16, 24, 36, 48, 64, 96, 196, 256]


def all_checkpoints_exist(exp_id: str, n_folds: int) -> bool:
    """Return True if every n_filters/fold checkpoint exists for an experiment."""
    for n_filters in N_FILTERS:
        model_dir = MODEL_ROOT / f"{exp_id}_nf{n_filters}"
        for fold in range(n_folds):
            model_path = model_dir / f"{exp_id}.fold{fold}.final.torch"
            if not model_path.exists():
                return False
    return True


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
    parser.add_argument("--partition", type=str, default="akundaje")
    parser.add_argument("--cpus-per-task", type=int, default=4)
    parser.add_argument("--mem", type=str, default="64G")
    parser.add_argument("--time", type=str, default="7-00:00:00")
    parser.add_argument(
        "--min-reads",
        type=int,
        default=0,
        help="skip experiments with fewer total reads than this (default: 0)",
    )
    parser.add_argument(
        "--fit-args",
        type=str,
        default="",
        help="extra arguments forwarded to fit_cherimoya.py (e.g. '--max-epochs 100')",
    )
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    experiments = list(config["experiments"].keys())

    with open(CHROM_SPLITS_PATH) as f:
        chrom_splits = yaml.safe_load(f)
    n_folds = len(chrom_splits["folds"])
    n_tasks = len(N_FILTERS) * n_folds

    read_counts_df = pd.read_csv(
        N_READS_PATH, sep="\t", usecols=["experiment", "total_reads"]
    )
    read_counts = dict(zip(read_counts_df["experiment"], read_counts_df["total_reads"]))

    log_dir = REPO_ROOT / "logs" / "cherimoya_n_filters"
    log_dir.mkdir(parents=True, exist_ok=True)

    n_filters_bash = " ".join(str(n) for n in N_FILTERS)
    extra_fit_args = f" {args.fit_args}" if args.fit_args else ""

    submitted = 0
    skipped_complete = 0
    skipped_reads = 0
    for exp_id in experiments:
        n_reads = read_counts.get(exp_id, 0)
        if n_reads < args.min_reads:
            skipped_reads += 1
            continue

        if all_checkpoints_exist(exp_id, n_folds):
            skipped_complete += 1
            continue

        job_name = f"cherimoya_nf_{exp_id}"
        fit_cmd = (
            f'python3 "$FIT_SCRIPT" -e "$EXP_ID" --fold "$FOLD" -v '
            f'--n-filters "$N_FILTERS_VALUE" --output-dir "$OUTPUT_DIR"'
            f"{extra_fit_args}"
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
            #SBATCH --output={log_dir}/{job_name}_%j.out
            #SBATCH --error={log_dir}/{job_name}_%j.err

            set -euo pipefail

            N_FILTERS_VALUES=({n_filters_bash})
            N_FOLDS={n_folds}

            EXP_ID={shlex.quote(exp_id)}
            REPO_ROOT={shlex.quote(str(REPO_ROOT))}
            MODEL_ROOT={shlex.quote(str(MODEL_ROOT))}
            FIT_SCRIPT={shlex.quote(str(FIT_SCRIPT))}
            NUMBA_CACHE_DIR={shlex.quote(str(NUMBA_CACHE_DIR))}

            mkdir -p "$NUMBA_CACHE_DIR"
            export NUMBA_CACHE_DIR

            ml load math
            ml load {PYTORCH_MODULE} {TRITON_MODULE}
            source "{VENV_DIR}/bin/activate"

            cd "$REPO_ROOT"
            nvidia-smi -L

            for N_FILTERS_VALUE in "${{N_FILTERS_VALUES[@]}}"; do
                OUTPUT_DIR="$MODEL_ROOT/${{EXP_ID}}_nf${{N_FILTERS_VALUE}}"

                for FOLD in $(seq 0 $((N_FOLDS - 1))); do
                    MODEL_PATH="$OUTPUT_DIR/${{EXP_ID}}.fold${{FOLD}}.final.torch"

                    if [[ -f "$MODEL_PATH" ]]; then
                        echo "Skipping existing model: $MODEL_PATH"
                        continue
                    fi

                    echo "Training $EXP_ID fold $FOLD with n_filters=$N_FILTERS_VALUE"
                    {fit_cmd}
                done
            done
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
    total_tasks = len(experiments) * n_tasks
    print(
        f"\n{action} {submitted} experiment jobs ({submitted * n_tasks} serial "
        f"training commands), skipped {skipped_reads} experiments with "
        f"<{args.min_reads:,} reads, skipped {skipped_complete} fully trained "
        f"experiments ({total_tasks} possible commands)"
    )


if __name__ == "__main__":
    main()
