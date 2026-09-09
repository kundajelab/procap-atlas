#!/usr/bin/env python3
"""Report launch_post_hoc_pipeline.py jobs that started but never reached
a genuinely-complete state, without having to check SLURM job states or
scan `.err` logs by hand.

Deliberately does NOT text-scan `.err` files for a generic "did this fail"
signal the way modisco/relaunch_timeout.py does for its specific TIME_LIMIT
case: that works there because SLURM's own time-limit cancellation message
is a fixed, reliable string, but this pipeline's own dependencies (finemo,
numpy, matplotlib) routinely print non-fatal UserWarning/RuntimeWarning
text to stderr even on a fully successful run (confirmed directly -- every
real report_bpnet.py run this session printed warnings like "Passing a
hits.tsv file to `finemo report` is deprecated" and "invalid value
encountered in divide" on success). A naive non-empty-stderr check would
flag nearly everything.

Instead, reuses launch_post_hoc_pipeline.py's own completion check as
ground truth: hits_filtered.tsv must exist AND be at least as new as
hits_confidence_filtered.tsv, confirming the final report_bpnet.py pass
(step 4) genuinely ran after the corroboration filter (step 3), not just
after the baseline pass (step 2) alone. An experiment is flagged only if:

1. It has a job log (job was actually submitted at some point), and
2. It is not yet complete by that check, and
3. Its job name doesn't currently appear in `squeue` (so it isn't just
   still legitimately running/pending/requeued -- flagging an in-progress
   job as "failed" would be a false alarm).

Since launch_post_hoc_pipeline.py is submitted with --requeue, most
preemptions resolve themselves automatically without ever needing this
script -- what's left to catch here is everything --requeue doesn't help
with: a real bug or bad data in one of the four underlying scripts, an
OOM/node failure, hitting the --time limit, etc. This script only reports;
it doesn't resubmit anything. Since launch_post_hoc_pipeline.py's own skip
check is the same completion check used here, simply rerunning it with the
same arguments is always safe and will only resubmit the flagged
experiments, not the already-completed ones.

Usage:
    python src/bpnet/hitcall/check_post_hoc_pipeline_failures.py
    python src/bpnet/hitcall/check_post_hoc_pipeline_failures.py --head profile --head count
    python src/bpnet/hitcall/check_post_hoc_pipeline_failures.py --min-trim-len 6
    python src/bpnet/hitcall/check_post_hoc_pipeline_failures.py --tail-lines 40
    python src/bpnet/hitcall/check_post_hoc_pipeline_failures.py --no-squeue  # skip the live-queue check
"""

import argparse
import getpass
import subprocess
from pathlib import Path

import pandas as pd
import yaml

from call_hits_bpnet import DEFAULT_CWM_TRIM_THRESHOLD, trim_suffix

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "experiment_config.yaml"
N_READS_PATH = REPO_ROOT / "configs" / "n_reads.txt"


def get_running_job_names(no_squeue: bool) -> set[str] | None:
    """Job names currently RUNNING/PENDING/etc. per squeue, or None if
    squeue wasn't available/queryable (caller should then treat every
    incomplete-with-a-log experiment as flagged, with a caveat printed).
    """
    if no_squeue:
        return None
    try:
        result = subprocess.run(
            ["squeue", "-u", getpass.getuser(), "-h", "-o", "%j"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return set(line.strip() for line in result.stdout.splitlines() if line.strip())


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--head",
        type=str,
        action="append",
        choices=["profile", "count"],
        default=None,
        metavar="HEAD",
        help="attribution/motif head(s) to check; repeatable (default: profile)",
    )
    parser.add_argument(
        "--min-reads",
        type=int,
        default=0,
        help="skip experiments with fewer total reads than this (default: 0, disabled)",
    )
    parser.add_argument(
        "--min-trim-len",
        type=int,
        default=None,
        metavar="BP",
        help="must match the value hitcall/launch.py was run with, if any",
    )
    parser.add_argument(
        "--tail-lines",
        type=int,
        default=20,
        help="lines of each flagged experiment's .err log to print (default: 20)",
    )
    parser.add_argument(
        "--no-squeue",
        action="store_true",
        help=(
            "skip the live-queue check entirely (e.g. if squeue isn't "
            "available) -- WARNING: without it, a job that is still "
            "legitimately running/requeued will be reported as flagged too"
        ),
    )
    args = parser.parse_args()

    heads = args.head if args.head is not None else ["profile"]

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    experiments = list(config["experiments"].keys())

    read_counts_df = pd.read_csv(
        N_READS_PATH, sep="\t", usecols=["experiment", "total_reads"]
    )
    read_counts = dict(zip(read_counts_df["experiment"], read_counts_df["total_reads"]))

    hitcalls_dir = REPO_ROOT / "hitcalls" / "bpnet"
    modisco_dir = REPO_ROOT / "modisco" / "bpnet"
    log_dir = REPO_ROOT / "logs" / "bpnet_hitcall_post_hoc_pipeline"

    running_job_names = get_running_job_names(args.no_squeue)
    if running_job_names is None and not args.no_squeue:
        print(
            "WARNING: could not query squeue (not available, timed out, or "
            "errored) -- a job that is still legitimately running/requeued "
            "may be incorrectly reported as flagged below.\n"
        )

    flagged = []
    still_running = 0
    not_submitted = 0
    checked = 0
    for exp_id in experiments:
        n_reads = read_counts.get(exp_id, 0)
        if n_reads < args.min_reads:
            continue

        model_dir_name = exp_id

        for head in heads:
            cwm_trim_coords = (
                modisco_dir
                / f"{exp_id}_{head}_trim_coords_min{args.min_trim_len}bp.tsv"
                if args.min_trim_len is not None
                else None
            )
            suffix = trim_suffix(DEFAULT_CWM_TRIM_THRESHOLD, None, cwm_trim_coords)
            exp_dir = hitcalls_dir / f"{model_dir_name}_{head}"
            hits_dir = exp_dir / suffix.lstrip("_") if suffix else exp_dir

            hits_unique = hits_dir / "hits_unique.tsv"
            if not hits_unique.exists():
                continue

            job_name = f"bpnet_hitcall_post_hoc_pipeline_{exp_id}_{head}{suffix}"
            err_path = log_dir / f"{job_name}.err"
            out_path = log_dir / f"{job_name}.out"
            if not err_path.exists() and not out_path.exists():
                not_submitted += 1
                continue

            checked += 1

            hits_filtered = hits_dir / "hits_filtered.tsv"
            hits_confidence_filtered = hits_dir / "hits_confidence_filtered.tsv"
            already_done = (
                hits_filtered.exists()
                and hits_confidence_filtered.exists()
                and hits_filtered.stat().st_mtime >= hits_confidence_filtered.stat().st_mtime
            )
            if already_done:
                continue

            if running_job_names is not None and job_name in running_job_names:
                still_running += 1
                continue

            flagged.append((exp_id, head, job_name, err_path))

    if not flagged:
        print(
            f"No flagged jobs ({checked} submitted-and-checked, "
            f"{still_running} still running, {not_submitted} never submitted)."
        )
        return

    print(f"{len(flagged)} job(s) started but not genuinely complete:\n")
    for exp_id, head, job_name, err_path in flagged:
        print(f"=== {exp_id} / {head} ({job_name}) ===")
        if err_path.exists():
            lines = err_path.read_text(errors="replace").splitlines()
            tail = lines[-args.tail_lines:] if args.tail_lines > 0 else lines
            for line in tail:
                print(f"  {line}")
        else:
            print(f"  (no .err file at {err_path})")
        print()

    print(
        f"{len(flagged)} flagged, {checked - len(flagged) - still_running} "
        f"complete, {still_running} still running, {not_submitted} never "
        f"submitted ({checked} total submitted-and-checked).\n"
        "Rerun launch_post_hoc_pipeline.py with the same arguments to "
        "resubmit only the flagged experiments -- its own skip check "
        "matches the one used here, so already-complete experiments won't "
        "be resubmitted."
    )


if __name__ == "__main__":
    main()
