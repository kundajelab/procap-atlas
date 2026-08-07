#!/usr/bin/env python3
"""Submit SLURM jobs to benchmark trained Cherimoya models for all experiments.

Reads experiment IDs from configs/experiment_config.yaml and submits one
sbatch job per experiment via benchmark_cherimoya.py. Jobs are skipped if the
metrics JSON already exists or if any fold model is missing.

By default, jobs run via `apptainer exec --nv` against the Cherimoya
Apptainer image (see src/cherimoya/apptainer/). check_gpu.py has confirmed
this path works on real Sherlock hardware across every GPU SKU tested (see
src/cherimoya/apptainer/README.md).

--native runs each experiment natively instead, using SRCC's py-pytorch/
py-triton Sherlock modules (see src/cherimoya/sherlock_native/).

--local skips SLURM entirely and runs each experiment directly in the
foreground with inherited stdout/stderr, instead of submitting SLURM jobs.
It still uses Apptainer by default (add --native for a `uv run --extra
cherimoya` invocation instead). SLURM resource flags
(--gpus/--partition/--cpus-per-task/--mem/--time) are ignored in this mode.
Combined with --dry-run, --local prints one runnable command per experiment
instead of executing them -- pipe that into something like
`simple_gpu_scheduler` to fan a personal multi-GPU box out in parallel.

Usage:
    python src/cherimoya/benchmark/launch.py
    python src/cherimoya/benchmark/launch.py --dry-run
    python src/cherimoya/benchmark/launch.py --save-output
    python src/cherimoya/benchmark/launch.py --min-reads 20000000
    python src/cherimoya/benchmark/launch.py --benchmark-args '-b 128'
    python src/cherimoya/benchmark/launch.py --native
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
            "run each experiment directly in the foreground, instead of "
            "submitting SLURM jobs; ignores SLURM resource flags "
            "(--gpus/--partition/--cpus-per-task/--mem/--time); combined "
            "with --dry-run, prints one command per experiment instead of "
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
            "`uv run --extra cherimoya`"
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
    use_apptainer = not args.native
    local_apptainer_image = (
        resolve_local_apptainer_image(args.apptainer_image)
        if args.local and use_apptainer
        else None
    )

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
            if use_apptainer:
                cmd = [
                    "apptainer",
                    "exec",
                    "--nv",
                    *[part for path in apptainer_binds for part in ("--bind", path)],
                    local_apptainer_image,
                    "python",
                    str(BENCHMARK_SCRIPT),
                    "-e",
                    exp_id,
                    "--model-dir",
                    str(model_dir),
                    "-v",
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
        extra_benchmark_args = ""
        if args.save_output:
            extra_benchmark_args += " --save-output"
        if args.benchmark_args:
            extra_benchmark_args += f" {args.benchmark_args}"

        # Built as a flat list of lines (rather than a nested triple-quoted
        # block) and joined with the same 12-space indent as the rest of the
        # template below, so the final textwrap.dedent() still finds a
        # common prefix across every line -- a mismatched indent here leaves
        # the whole script indented, which sbatch rejects as "not a batch
        # script".
        if use_apptainer:
            setup_lines = [
                f"BENCHMARK_SCRIPT={shlex.quote(str(BENCHMARK_SCRIPT))}",
                f'APPTAINER_IMAGE="{args.apptainer_image}"',
                "",
            ]
            benchmark_cmd = (
                f'apptainer exec --nv {apptainer_bind_args} "$APPTAINER_IMAGE" '
                f'python "$BENCHMARK_SCRIPT" -e {shlex.quote(exp_id)} '
                f"--model-dir {shlex.quote(str(model_dir))} -v{extra_benchmark_args}"
            )
        else:
            setup_lines = [
                f'mkdir -p "{NUMBA_CACHE_DIR}"',
                f'export NUMBA_CACHE_DIR="{NUMBA_CACHE_DIR}"',
                f"BENCHMARK_SCRIPT={shlex.quote(str(BENCHMARK_SCRIPT))}",
                "",
                "ml load math",
                f"ml load {PYTORCH_MODULE} {TRITON_MODULE}",
                f'source "{VENV_DIR}/bin/activate"',
                "",
            ]
            benchmark_cmd = (
                f'python3 "$BENCHMARK_SCRIPT" -e {shlex.quote(exp_id)} '
                f"--model-dir {shlex.quote(str(model_dir))} -v{extra_benchmark_args}"
            )
        setup_block = "\n            ".join(setup_lines)

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

            {setup_block}
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
