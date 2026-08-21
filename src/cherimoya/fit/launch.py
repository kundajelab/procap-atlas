#!/usr/bin/env python3
"""Submit SLURM jobs to train Cherimoya models for all experiments and folds.

Reads experiment IDs from configs/experiment_config.yaml and submits one
sbatch job per experiment via fit_cherimoya.py, running that experiment's
folds sequentially within the job.

Experiments with fewer total reads than --min-reads (default: 10_000_000) are
skipped, as low-coverage experiments tend to produce poorly calibrated models.
Folds with a completed model are also skipped automatically; an experiment
with every fold already trained is not submitted at all.

SLURM jobs are submitted with --requeue, so a pre-empted job (the
akundaje/owners partitions are preemptible) is automatically resubmitted by
SLURM rather than needing a manual resubmit. The submitted script re-checks
each fold for a completed model at the start of every run (not just once at
submission time), so a requeued job skips whatever folds finished before
pre-emption and only (re)trains the rest -- cherimoya trains fast enough
(~30s/epoch) that redoing a partially-trained fold from scratch is cheap.
"Completed" is judged by `{exp_id}.fold{fold}.final.torch`, which
Cherimoya's fit() writes exactly once, at the very end of training --
`{exp_id}.fold{fold}.torch` (no `.final`) is the wrong artifact for this,
since it's overwritten throughout training whenever validation correlation
improves and can already exist after a single epoch.

By default, jobs run via `apptainer exec --nv` against the Cherimoya
Apptainer image (see src/cherimoya/apptainer/) -- its
pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime base image bundles Python 3.12
and torch 2.13.0, so it trains with torch.compile enabled. check_gpu.py has
confirmed this path works on real Sherlock hardware across every GPU SKU
tested (see src/cherimoya/apptainer/README.md).

--native runs each fold natively instead, using SRCC's py-pytorch/py-triton
Sherlock modules (see src/cherimoya/sherlock_native/). That module pair is
Python 3.14-only, which makes torch.compile() unconditionally raise (see the
comment above `compile_supported` in fit_cherimoya.py) until Sherlock ships
a torch>=2.10 build -- prefer the Apptainer default unless you have a
specific reason to fall back to native modules.

--local skips SLURM entirely and runs each fold directly in the foreground
with inherited stdout/stderr, instead of submitting SLURM jobs -- useful on
a GPU box you already have a shell on (e.g. a lab cluster). It still uses
Apptainer by default (add --native for a `uv run --extra cherimoya`
invocation instead, e.g. if that box has no Apptainer image built). SLURM
resource flags (--gpus/--partition/--cpus-per-task/--mem/--time) are ignored
in this mode. Combined with --dry-run, --local prints one runnable command
per fold instead of executing them -- pipe that into something like
`simple_gpu_scheduler` to fan a personal multi-GPU box out in parallel
instead of running folds one at a time.

Usage:
    python src/cherimoya/fit/launch.py                    # submit one job per experiment, 7 folds each
    python src/cherimoya/fit/launch.py --dry-run           # print sbatch scripts without submitting
    python src/cherimoya/fit/launch.py --time 48:00:00 --mem 32G --partition gpu
    python src/cherimoya/fit/launch.py --min-reads 20000000  # only well-covered experiments
    python src/cherimoya/fit/launch.py --native            # run via SRCC's native modules instead of Apptainer
    python src/cherimoya/fit/launch.py --local             # run in the foreground, no SLURM
    python src/cherimoya/fit/launch.py --local --dry-run   # print one command per fold, e.g. for simple_gpu_scheduler
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
FIT_SCRIPT = REPO_ROOT / "src" / "cherimoya" / "fit" / "fit_cherimoya.py"
# Resolved by bash at job runtime (not by Python at submission time) so these
# work under any Sherlock username, matching setup_env.sh/test_install.sh.
NUMBA_CACHE_DIR = "${NUMBA_CACHE_DIR:-/scratch/users/${USER}/numba_cache}"
VENV_DIR = "${CHERIMOYA_VENV_DIR:-/scratch/users/${USER}/venvs/cherimoya-sherlock}"
PYTORCH_MODULE = "py-pytorch/2.9.1_py314"
TRITON_MODULE = "py-triton/3.5.1_py314"
APPTAINER_IMAGE = "${CHERIMOYA_APPTAINER_IMAGE:-/scratch/users/${USER}/apptainer/cherimoya.sif}"
# --local doesn't run on Sherlock, so it has no reason to default into
# Sherlock's scratch layout; a repo-relative cache dir works anywhere.
LOCAL_NUMBA_CACHE_DIR = REPO_ROOT / ".cache" / "numba"


def resolve_local_apptainer_image(value):
    """Resolve APPTAINER_IMAGE's bash ${VAR:-default} template to a real path.

    --local runs commands via subprocess.run() directly, with no shell to
    expand that syntax (unlike the sbatch script, which bash interprets), so
    passing it through unresolved would make apptainer look for a file
    literally named "${CHERIMOYA_APPTAINER_IMAGE:-...}".
    """
    if value != APPTAINER_IMAGE:
        return value
    default = f"/scratch/users/{os.environ.get('USER', '')}/apptainer/cherimoya.sif"
    return os.environ.get("CHERIMOYA_APPTAINER_IMAGE", default)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print sbatch scripts (or local commands, with --local) without running them",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help=(
            "run each fold directly in the foreground, instead of "
            "submitting SLURM jobs; ignores SLURM resource flags "
            "(--gpus/--partition/--cpus-per-task/--mem/--time); combined "
            "with --dry-run, prints one command per fold instead of "
            "running them (e.g. to pipe into simple_gpu_scheduler)"
        ),
    )
    parser.add_argument(
        "--native",
        action="store_true",
        help=(
            "use SRCC's py-pytorch/py-triton Sherlock modules instead of "
            "the Apptainer image (default) -- either for SLURM submission "
            "or, combined with --local, for foreground execution via "
            "`uv run --extra cherimoya`. The native module pair is "
            "Python-3.14-only, which disables torch.compile (see "
            "fit_cherimoya.py); prefer the Apptainer default unless you "
            "have a specific reason to fall back"
        ),
    )
    parser.add_argument(
        "--apptainer-image",
        type=str,
        default=APPTAINER_IMAGE,
        help=f"path to the Cherimoya Apptainer .sif image (default: {APPTAINER_IMAGE})",
    )
    parser.add_argument(
        "--apptainer-bind",
        action="append",
        default=None,
        help=(
            "path to bind into the Apptainer container; may be repeated "
            "(default: the repo root -- add more if models/data live "
            "outside it, e.g. on scratch or oak)"
        ),
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
    parser.add_argument("--mem", type=str, default="64G")
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
    use_apptainer = not args.native
    local_apptainer_image = (
        resolve_local_apptainer_image(args.apptainer_image)
        if args.local and use_apptainer
        else None
    )

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

    apptainer_binds = args.apptainer_bind or [str(REPO_ROOT)]
    apptainer_bind_args = " ".join(f"--bind {shlex.quote(path)}" for path in apptainer_binds)

    # Only the native path needs a host-side NUMBA_CACHE_DIR override; the
    # Apptainer image sets its own via %environment.
    local_env = None
    if args.local and not use_apptainer:
        local_env = dict(os.environ)
        local_env.setdefault("NUMBA_CACHE_DIR", str(LOCAL_NUMBA_CACHE_DIR))
        Path(local_env["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

    submitted = 0
    skipped_trained = 0
    skipped_reads = 0
    skipped_complete = 0
    local_failures = []
    for exp_id in experiments:
        # Skip experiments with too few reads
        n_reads = read_counts.get(exp_id, 0)
        if n_reads < args.min_reads:
            skipped_reads += 1
            continue

        # Skip folds with a completed model. `.final.torch` is written
        # exactly once, at the very end of Cherimoya's fit() -- unlike
        # `.torch` (no `.final`), which is overwritten throughout training
        # whenever validation correlation improves, so checking that one
        # would treat a fold pre-empted after a single epoch as complete.
        model_dir = REPO_ROOT / "models" / "cherimoya" / exp_id
        folds_to_run = [
            fold
            for fold in range(n_folds)
            if not (model_dir / f"{exp_id}.fold{fold}.final.torch").exists()
        ]
        skipped_trained += n_folds - len(folds_to_run)
        if not folds_to_run:
            skipped_complete += 1
            continue

        extra_fit_args = ""
        if args.fit_args:
            extra_fit_args = " " + args.fit_args

        if args.local:
            for fold in folds_to_run:
                if use_apptainer:
                    cmd = [
                        "apptainer",
                        "exec",
                        "--nv",
                        *[
                            part
                            for path in apptainer_binds
                            for part in ("--bind", path)
                        ],
                        local_apptainer_image,
                        "python",
                        str(FIT_SCRIPT),
                        "-e",
                        exp_id,
                        "--fold",
                        str(fold),
                        "-v",
                        *shlex.split(args.fit_args),
                    ]
                else:
                    cmd = [
                        "uv",
                        "run",
                        "--project",
                        str(REPO_ROOT),
                        "--extra",
                        "cherimoya",
                        "--frozen",
                        "python3",
                        str(FIT_SCRIPT),
                        "-e",
                        exp_id,
                        "--fold",
                        str(fold),
                        "-v",
                        *shlex.split(args.fit_args),
                    ]
                if args.dry_run:
                    print(shlex.join(cmd))
                    continue

                print(f"=== Training {exp_id} fold {fold} ===")
                result = subprocess.run(cmd, cwd=REPO_ROOT, env=local_env)
                if result.returncode != 0:
                    local_failures.append((exp_id, fold, result.returncode))
                    print(
                        f"ERROR: {exp_id} fold {fold} exited with "
                        f"{result.returncode}",
                        file=sys.stderr,
                    )
            submitted += 1
            continue

        job_name = f"cherimoya_{exp_id}"
        folds_bash = " ".join(str(fold) for fold in folds_to_run)

        # Built as a flat list of lines (rather than a nested triple-quoted
        # block) and joined with the same 12-space indent as the rest of the
        # template below, so the final textwrap.dedent() still finds a
        # common prefix across every line -- a mismatched indent here leaves
        # the whole script indented, which sbatch rejects as "not a batch
        # script" (see the historical version of this comment in git log).
        if use_apptainer:
            setup_lines = [
                f"FIT_SCRIPT={shlex.quote(str(FIT_SCRIPT))}",
                f'APPTAINER_IMAGE="{args.apptainer_image}"',
                f"EXP_ID={shlex.quote(exp_id)}",
                f"MODEL_DIR={shlex.quote(str(model_dir))}",
                f"FOLDS=({folds_bash})",
                "",
            ]
            run_fold_cmd = (
                f'apptainer exec --nv {apptainer_bind_args} "$APPTAINER_IMAGE" '
                f'python "$FIT_SCRIPT" -e "$EXP_ID" --fold "$FOLD" -v{extra_fit_args}'
            )
        else:
            setup_lines = [
                f'mkdir -p "{NUMBA_CACHE_DIR}"',
                f'export NUMBA_CACHE_DIR="{NUMBA_CACHE_DIR}"',
                f"FIT_SCRIPT={shlex.quote(str(FIT_SCRIPT))}",
                f"EXP_ID={shlex.quote(exp_id)}",
                f"MODEL_DIR={shlex.quote(str(model_dir))}",
                f"FOLDS=({folds_bash})",
                "",
                "ml load math",
                f"ml load {PYTORCH_MODULE} {TRITON_MODULE}",
                f'source "{VENV_DIR}/bin/activate"',
                "",
            ]
            run_fold_cmd = f'python3 "$FIT_SCRIPT" -e "$EXP_ID" --fold "$FOLD" -v{extra_fit_args}'
        setup_block = "\n            ".join(setup_lines)

        # The fold loop below re-checks each fold's .final.torch at runtime
        # (not just once at submission time, in `folds_to_run` above), so a
        # SLURM-requeued rerun of this same script skips whatever folds
        # finished before pre-emption instead of blindly retraining all of
        # them. --requeue is what makes a pre-empted job actually retry
        # automatically; --open-mode=append keeps prior attempts' output
        # instead of truncating the log on each retry.
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

            {setup_block}
            nvidia-smi -L

            for FOLD in "${{FOLDS[@]}}"; do
                MODEL_PATH="$MODEL_DIR/${{EXP_ID}}.fold${{FOLD}}.final.torch"
                if [[ -f "$MODEL_PATH" ]]; then
                    echo "Skipping already-completed fold $FOLD: $MODEL_PATH"
                    continue
                fi
                echo "Training $EXP_ID fold $FOLD"
                {run_fold_cmd}
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
                f"ERROR submitting {job_name}: {result.stderr.strip()}",
                file=sys.stderr,
            )

    if args.local:
        action = "Would run" if args.dry_run else "Ran"
        unit = "experiments"
    else:
        action = "Would submit" if args.dry_run else "Submitted"
        unit = "jobs"
    total = len(experiments) * n_folds
    print(
        f"\n{action} {submitted} {unit}, skipped {skipped_reads} experiments "
        f"with <{args.min_reads:,} reads, skipped {skipped_complete} fully-trained "
        f"experiments, skipped {skipped_trained} already-trained folds "
        f"({total} total folds)"
    )
    if args.local and not args.dry_run:
        if local_failures:
            print(f"{len(local_failures)} fold(s) failed:")
            for exp_id, fold, returncode in local_failures:
                print(f"  {exp_id} fold {fold} (exit {returncode})")
        else:
            print("All folds completed successfully.")


if __name__ == "__main__":
    main()
