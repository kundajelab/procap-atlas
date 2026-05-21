#!/usr/bin/env python3
"""Run save_ohe.py for all experiments asynchronously.

Reads experiment IDs from configs/experiment_config.yaml and runs
save_ohe.py for each, up to --jobs processes concurrently.

Jobs are skipped if the output OHE file already exists.

Usage:
    python src/bpnet/attribute/run_ohe.py                   # all experiments, 4 concurrent
    python src/bpnet/attribute/run_ohe.py -j 8              # 8 concurrent jobs
    python src/bpnet/attribute/run_ohe.py --dry-run         # print commands without running
    python src/bpnet/attribute/run_ohe.py --min-reads 10000000  # skip low-coverage experiments
"""

import argparse
import asyncio
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "experiment_config.yaml"
N_READS_PATH = REPO_ROOT / "configs" / "n_reads.txt"
SAVE_OHE_SCRIPT = REPO_ROOT / "src" / "bpnet" / "attribute" / "save_ohe.py"


def load_n_reads():
    df = pd.read_csv(N_READS_PATH, sep="\t", usecols=["experiment", "total_reads"])
    return dict(zip(df["experiment"], df["total_reads"]))


async def run_experiment(exp_id, semaphore, verbose):
    async with semaphore:
        cmd = ["python", str(SAVE_OHE_SCRIPT), "-e", exp_id]
        if verbose:
            cmd.append("-v")
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        return exp_id, proc.returncode, stdout.decode().strip(), stderr.decode().strip()


async def main_async(args):
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    experiments = list(config["experiments"].keys())

    read_counts = load_n_reads()
    out_dir = REPO_ROOT / "attributions" / "bpnet"

    to_run = []
    skipped_reads = 0
    skipped_done = 0
    for exp_id in experiments:
        if read_counts.get(exp_id, 0) < args.min_reads:
            skipped_reads += 1
            continue
        if (out_dir / f"{exp_id}_ohe.npz").exists():
            skipped_done += 1
            continue
        to_run.append(exp_id)

    print(
        f"Queued {len(to_run)} experiments, skipped {skipped_reads} with "
        f"<{args.min_reads:,} reads, {skipped_done} already done"
    )

    if args.dry_run:
        for exp_id in to_run:
            print(f"  python {SAVE_OHE_SCRIPT} -e {exp_id}")
        return

    if not to_run:
        return

    semaphore = asyncio.Semaphore(args.jobs)
    tasks = [asyncio.create_task(run_experiment(exp_id, semaphore, args.verbose)) for exp_id in to_run]

    completed = 0
    failed = []
    for coro in asyncio.as_completed(tasks):
        exp_id, returncode, stdout, stderr = await coro
        completed += 1
        if returncode == 0:
            print(f"[{completed}/{len(to_run)}] {exp_id}: done")
        else:
            print(f"[{completed}/{len(to_run)}] {exp_id}: FAILED (exit {returncode})", file=sys.stderr)
            if stderr:
                print(stderr, file=sys.stderr)
            failed.append(exp_id)

    print(f"\nCompleted {len(to_run) - len(failed)}/{len(to_run)}")
    if failed:
        print(f"Failed: {', '.join(failed)}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print commands without running",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=4,
        help="max concurrent jobs (default: 4)",
    )
    parser.add_argument(
        "--min-reads",
        type=int,
        default=0,
        help="skip experiments with fewer total reads than this (default: 0)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
