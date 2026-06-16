#!/usr/bin/env python3
"""Relaunch timed-out modisco SLURM jobs with a longer runtime.

Scans logs/bpnet_modisco/*.err for SLURM TIME LIMIT cancellations and
resubmits the affected (experiment, head) jobs, skipping any whose output
already exists (completed on retry).

Usage:
    python src/bpnet/modisco/relaunch_timeout.py                        # relaunch timed-out jobs
    python src/bpnet/modisco/relaunch_timeout.py --dry-run              # preview without submitting
    python src/bpnet/modisco/relaunch_timeout.py --time 4-00:00:00      # custom time limit
    python src/bpnet/modisco/relaunch_timeout.py --list                  # just list timed-out jobs
"""

import argparse
import re
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
JASPAR_PATH = (
    REPO_ROOT / "data" / "JASPAR2026_CORE_vertebrates_non-redundant_pfms_meme.txt"
)

# Pattern: modisco_{exp_id}_{head}.err
LOG_NAME_RE = re.compile(r"^modisco_(\w+)_(profile|count)\.err$")
RELOG_NAME_RE = re.compile(r"^relaunch_modisco_(\w+)_(profile|count)\.err$")
TIMEOUT_RE = re.compile(
    r"(CANCELLED\s+AT\s+\S+\s+DUE\s+TO\s+TIME\s+LIMIT"
    r"|DUE TO TIME LIMIT"
    r"|TIME_LIMIT"
    r"|TIMEOUT)",
    re.IGNORECASE,
)


def find_timed_out(log_dir: Path) -> list[tuple[str, str, Path]]:
    """Return list of (exp_id, head, err_path) for jobs that hit the time limit."""
    timed_out = []
    for err_path in sorted(log_dir.glob("modisco_*.err")):
        m = LOG_NAME_RE.match(err_path.name)
        if not m:
            continue
        exp_id, head = m.group(1), m.group(2)
        text = err_path.read_text(errors="replace")
        if TIMEOUT_RE.search(text):
            timed_out.append((exp_id, head, err_path))
    for err_path in sorted(log_dir.glob("relaunch_modisco_*.err")):
        m = RELOG_NAME_RE.match(err_path.name)
        if not m:
            continue
        exp_id, head = m.group(1), m.group(2)
        timed_out.append((exp_id, head, err_path))
    return timed_out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print sbatch scripts without submitting",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list timed-out jobs and exit",
    )
    # Modisco parameters (keep defaults matching launch.py)
    parser.add_argument("-n", "--n-seqlets", type=int, default=1_000_000)
    parser.add_argument("-l", "--leiden", type=int, default=50)
    parser.add_argument("-w", "--window", type=int, default=1000)
    # SLURM resource flags
    parser.add_argument("--partition", type=str, default="akundaje")
    parser.add_argument("--cpus-per-task", type=int, default=32)
    parser.add_argument("--mem", type=str, default="64G")
    parser.add_argument(
        "--time",
        type=str,
        default="6-23:00:00",
        help="new time limit for relaunched jobs (default: 6-23:00:00)",
    )
    args = parser.parse_args()

    log_dir = REPO_ROOT / "logs" / "bpnet_modisco"
    attr_dir = REPO_ROOT / "attributions" / "bpnet"
    out_dir = REPO_ROOT / "modisco" / "bpnet"

    timed_out = find_timed_out(log_dir)

    if not timed_out:
        print("No timed-out jobs found in", log_dir)
        return

    if args.list:
        print(f"Timed-out jobs ({len(timed_out)}):")
        for exp_id, head, err_path in timed_out:
            out_h5 = out_dir / f"{exp_id}_{head}.modisco.h5"
            status = "DONE" if out_h5.exists() else "PENDING"
            print(f"  [{status}] {exp_id}_{head}  ({err_path.name})")
        return

    submitted = 0
    skipped_done = 0
    skipped_no_attr = 0

    for exp_id, head, err_path in timed_out:
        out_h5 = out_dir / f"{exp_id}_{head}.modisco.h5"
        if out_h5.exists():
            print(f"WARNING: {out_h5} already exists, skipping", file=sys.stderr)
            skipped_done += 1
            continue

        ohe_path = attr_dir / f"{exp_id}_ohe.npz"
        attr_path = attr_dir / f"{exp_id}_{head}.npz"
        if not ohe_path.exists() or not attr_path.exists():
            print(
                f"WARNING: missing attribution/OHE for {exp_id}_{head}, skipping",
                file=sys.stderr,
            )
            skipped_no_attr += 1
            continue

        job_name = f"relaunch_modisco_{exp_id}_{head}"

        modisco_motifs_cmd = (
            f"modisco motifs"
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
            #SBATCH --partition={args.partition}
            #SBATCH --time={args.time}
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
            print(f"--- {job_name} (was: {err_path.name}, new time: {args.time}) ---")
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

    action = "Would resubmit" if args.dry_run else "Resubmitted"
    print(
        f"\n{action} {submitted} jobs, skipped {skipped_done} already done, "
        f"skipped {skipped_no_attr} missing attributions "
        f"(of {len(timed_out)} timed-out total)"
    )


if __name__ == "__main__":
    main()
