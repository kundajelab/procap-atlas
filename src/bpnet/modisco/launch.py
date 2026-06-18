#!/usr/bin/env python3
"""Submit SLURM jobs to run `modisco motifs` on BPNet attributions.

Reads experiment IDs from configs/experiment_config.yaml and submits one
sbatch job per (experiment, head) pair. Jobs are skipped if the modisco .h5
output already exists or if the required attribution/OHE npz files are missing
(run attribute/launch.py first).

Run launch_report.py separately after motifs completes.

Usage:
    python src/bpnet/modisco/launch.py                    # submit all experiments, both heads
    python src/bpnet/modisco/launch.py --dry-run           # print sbatch scripts without submitting
    python src/bpnet/modisco/launch.py --head count        # count head only
    python src/bpnet/modisco/launch.py --head profile      # profile head only
    python src/bpnet/modisco/launch.py --time 6-23:00:00 --mem 64G
    python src/bpnet/modisco/launch.py --min-reads 20000000  # only well-covered experiments
"""

import argparse
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "experiment_config.yaml"
N_READS_PATH = REPO_ROOT / "configs" / "n_reads.txt"
N_PEAKS_PATH = REPO_ROOT / "configs" / "n_peaks.txt"

EXPERIMENT_RE = re.compile(r"^(ENCSR\w+)_")
DEFAULT_RELAUNCH_PARTITION = "akundaje"
DEFAULT_RELAUNCH_TIME = "6-23:00:00"


def load_peak_counts(path: Path) -> dict[str, int]:
    """Load experiment peak counts from configs/n_peaks.txt."""
    peak_counts = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            peak_path, count_text = line.rsplit(":", maxsplit=1)
            match = EXPERIMENT_RE.match(Path(peak_path).name)
            if match is None:
                continue
            peak_counts[match.group(1)] = int(count_text.strip())
    return peak_counts


def largest_peak_experiments(
    experiments: list[str], peak_counts: dict[str, int], n_experiments: int
) -> set[str]:
    """Return the experiment IDs with the largest peak sets."""
    if n_experiments <= 0:
        return set()
    ranked = sorted(
        (
            (exp_id, peak_counts[exp_id])
            for exp_id in experiments
            if exp_id in peak_counts
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    return {exp_id for exp_id, _ in ranked[:n_experiments]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print sbatch scripts without submitting",
    )
    parser.add_argument(
        "--head",
        type=str,
        action="append",
        choices=["profile", "count"],
        default=None,
        metavar="HEAD",
        help="attribution head(s) to run; repeatable (default: profile count)",
    )
    # Modisco parameters
    parser.add_argument(
        "-n",
        "--n-seqlets",
        type=int,
        default=1_000_000,
        help="max number of seqlets to use (default: 1_000_000)",
    )
    parser.add_argument(
        "-l",
        "--leiden",
        type=int,
        default=50,
        help="number of leiden clusters to use (default: 50)",
    )
    parser.add_argument(
        "-w",
        "--window",
        type=int,
        default=1000,
        help="window to get seqlets from (default: 1000)",
    )
    # SLURM resource flags
    parser.add_argument("--partition", type=str, default="normal,akundaje,owners")
    parser.add_argument("--cpus-per-task", type=int, default=32)
    parser.add_argument("--mem", type=str, default="64G")
    parser.add_argument("--time", type=str, default="2-00:00:00")
    parser.add_argument(
        "--large-peak-top-n",
        type=int,
        default=30,
        help=(
            "number of largest peak-set experiments to launch with relaunch-style "
            "SLURM defaults (default: 30; use 0 to disable)"
        ),
    )
    parser.add_argument(
        "--large-peak-partition",
        type=str,
        default=DEFAULT_RELAUNCH_PARTITION,
        help=(
            "partition for large peak-set jobs "
            f"(default: {DEFAULT_RELAUNCH_PARTITION})"
        ),
    )
    parser.add_argument(
        "--large-peak-time",
        type=str,
        default=DEFAULT_RELAUNCH_TIME,
        help=(
            "time limit for large peak-set jobs "
            f"(default: {DEFAULT_RELAUNCH_TIME})"
        ),
    )
    parser.add_argument(
        "--min-reads",
        type=int,
        default=0,
        help="skip experiments with fewer total reads than this (default: 0, disabled)",
    )
    args = parser.parse_args()

    heads = args.head if args.head is not None else ["profile", "count"]

    # Load experiment list
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    experiments = list(config["experiments"].keys())

    # Load read counts for filtering
    read_counts_df = pd.read_csv(
        N_READS_PATH, sep="\t", usecols=["experiment", "total_reads"]
    )
    read_counts = dict(zip(read_counts_df["experiment"], read_counts_df["total_reads"]))
    peak_counts = load_peak_counts(N_PEAKS_PATH)
    large_peak_experiments = largest_peak_experiments(
        experiments, peak_counts, args.large_peak_top_n
    )

    attr_dir = REPO_ROOT / "attributions" / "bpnet"
    out_dir = REPO_ROOT / "modisco" / "bpnet"
    log_dir = REPO_ROOT / "logs" / "bpnet_modisco"
    log_dir.mkdir(parents=True, exist_ok=True)

    submitted = 0
    skipped_done = 0
    skipped_no_attr = 0
    skipped_reads = 0
    for exp_id in experiments:
        # Skip experiments with too few reads
        n_reads = read_counts.get(exp_id, 0)
        if n_reads < args.min_reads:
            skipped_reads += 1
            continue

        # OHE sequences file is shared across heads
        ohe_path = attr_dir / f"{exp_id}_ohe.npz"
        if not ohe_path.exists():
            skipped_no_attr += len(heads)
            continue

        for head in heads:
            attr_path = attr_dir / f"{exp_id}_{head}.npz"
            if not attr_path.exists():
                skipped_no_attr += 1
                continue

            # Skip if modisco .h5 output already exists
            out_h5 = out_dir / f"{exp_id}_{head}.modisco.h5"
            if out_h5.exists():
                skipped_done += 1
                continue

            is_large_peak_job = exp_id in large_peak_experiments
            partition = (
                args.large_peak_partition if is_large_peak_job else args.partition
            )
            time = args.large_peak_time if is_large_peak_job else args.time
            job_name = f"modisco_{exp_id}_{head}"

            modisco_motifs_cmd = (
                f"uv run --project {REPO_ROOT} --frozen --extra bpnet modisco motifs"
                f" -s {ohe_path}"
                f" -a {attr_path}"
                f" -o {out_h5}"
                f" -n {args.n_seqlets} -l {args.leiden} -w {args.window} -v"
            )

            sbatch_script = textwrap.dedent(f"""\
                #!/bin/bash -l
                #SBATCH --job-name={job_name}
                #SBATCH --ntasks=1
                #SBATCH --ntasks-per-node=1
                #SBATCH --nodes=1
                #SBATCH --cpus-per-task={args.cpus_per_task}
                #SBATCH --mem={args.mem}
                #SBATCH --partition={partition}
                #SBATCH --time={time}
                #SBATCH --output={log_dir}/{job_name}.out
                #SBATCH --error={log_dir}/{job_name}.err
                #SBATCH -C NO_GPU
                
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
                export NUMBA_NUM_THREADS={args.cpus_per_task}

                mkdir -p {out_dir}
                cd {out_dir}
                time {modisco_motifs_cmd}
            """)

            if args.dry_run:
                peak_note = ""
                if is_large_peak_job:
                    peak_note = (
                        f" (large peak set: {peak_counts.get(exp_id, 0):,} peaks, "
                        f"partition: {partition}, time: {time})"
                    )
                print(f"--- {job_name}{peak_note} ---")
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
    total = len(experiments) * len(heads)
    print(
        f"\n{action} {submitted} jobs, skipped {skipped_reads} experiments "
        f"with <{args.min_reads:,} reads, skipped {skipped_no_attr} missing attributions, "
        f"skipped {skipped_done} already done ({total} total). "
        f"Large peak-set split: top {len(large_peak_experiments)} experiments use "
        f"{args.large_peak_partition} for {args.large_peak_time}."
    )


if __name__ == "__main__":
    main()
