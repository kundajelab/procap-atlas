#!/usr/bin/env python3
"""Submit SLURM jobs to run diagnose_background_energy_ratio.py --out-tsv
across every experiment in configs/experiment_config.yaml, for reviewing
detect_elbow_count's cutoffs by hand across many experiments/heads before
trusting its defaults atlas-wide (see filter_low_confidence_hits.py's
--seqlet-background-excess-only, which wires this same metric into the
actual corroboration-floor scoping).

Reads experiment IDs from configs/experiment_config.yaml and submits one
sbatch job per (experiment, head) pair. Jobs are skipped if the per-
experiment/head output TSV already exists (see --out-dir) or if
regions.npz/a resolvable hits file is missing (run call_hits_bpnet.py/
hitcall/launch.py first). This step does not use a GPU and doesn't run
tangermeme's recursive_seqlets, so it's cheap -- similar resource needs to
report_bpnet.py's own launcher (launch_report.py), which this mirrors.

Jobs are submitted with --requeue: the default --partition includes
`owners`, which is preemptible (`normal`/`akundaje`/`gpu` are not), and
without --requeue a preempted job just dies with no automatic
resubmission. A requeued job reruns diagnose_background_energy_ratio.py
from scratch, which is safe -- it deterministically overwrites its own
--out-tsv from the same inputs every time.

Usage:
    python src/bpnet/hitcall/launch_background_energy_ratio.py                    # all experiments, profile head
    python src/bpnet/hitcall/launch_background_energy_ratio.py --dry-run           # print sbatch scripts without submitting
    python src/bpnet/hitcall/launch_background_energy_ratio.py --head profile --head count
    python src/bpnet/hitcall/launch_background_energy_ratio.py --min-trim-len 6     # match hitcall/launch.py's floor
    python src/bpnet/hitcall/launch_background_energy_ratio.py --force              # rerun even if the TSV already exists
"""

import argparse
import subprocess
import sys
import textwrap
from pathlib import Path

import pandas as pd
import yaml

from call_hits_bpnet import DEFAULT_CWM_TRIM_THRESHOLD, resolve_hits_path, trim_suffix

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "experiment_config.yaml"
N_READS_PATH = REPO_ROOT / "configs" / "n_reads.txt"
DIAGNOSE_SCRIPT = REPO_ROOT / "src" / "bpnet" / "hitcall" / "diagnose_background_energy_ratio.py"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dry-run", action="store_true", help="print sbatch scripts without submitting")
    parser.add_argument(
        "--head", type=str, action="append", choices=["profile", "count"], default=None,
        metavar="HEAD", help="attribution/motif head(s) to run; repeatable (default: profile)",
    )
    # SLURM resource flags
    parser.add_argument("--partition", type=str, default="normal,akundaje,owners")
    parser.add_argument("--cpus-per-task", type=int, default=4)
    parser.add_argument("--mem", type=str, default="64G")
    parser.add_argument("--time", type=str, default="2:00:00")
    parser.add_argument(
        "--min-reads", type=int, default=0,
        help="skip experiments with fewer total reads than this (default: 0, disabled)",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=REPO_ROOT / "background_excess" / "bpnet",
        help="directory for per-experiment/head output TSVs (default: background_excess/bpnet/)",
    )
    parser.add_argument(
        "--diagnose-args", type=str, default="",
        help="extra arguments forwarded to diagnose_background_energy_ratio.py (e.g. '--min-gap-ratio 0.15')",
    )
    parser.add_argument(
        "--min-trim-len", type=int, default=None, metavar="BP",
        help=(
            "must match the value hitcall/launch.py was run with, if any -- "
            "resolves the same per-experiment trim-coords-suffixed output "
            "directory rather than the plain {model_dir_name}_{head}/ one."
        ),
    )
    parser.add_argument(
        "--force", action="store_true",
        help="resubmit even if the output TSV already exists",
    )
    args = parser.parse_args()

    heads = args.head if args.head is not None else ["profile"]

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    experiments = list(config["experiments"].keys())

    read_counts_df = pd.read_csv(N_READS_PATH, sep="\t", usecols=["experiment", "total_reads"])
    read_counts = dict(zip(read_counts_df["experiment"], read_counts_df["total_reads"]))

    hitcalls_dir = REPO_ROOT / "hitcalls" / "bpnet"
    modisco_dir = REPO_ROOT / "modisco" / "bpnet"
    log_dir = REPO_ROOT / "logs" / "bpnet_hitcall_background_excess"
    log_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    submitted = 0
    skipped_done = 0
    skipped_missing = 0
    skipped_reads = 0
    for exp_id in experiments:
        n_reads = read_counts.get(exp_id, 0)
        if n_reads < args.min_reads:
            skipped_reads += 1
            continue

        for head in heads:
            cwm_trim_coords = (
                modisco_dir / f"{exp_id}_{head}_trim_coords_min{args.min_trim_len}bp.tsv"
                if args.min_trim_len is not None
                else None
            )
            suffix = trim_suffix(DEFAULT_CWM_TRIM_THRESHOLD, None, cwm_trim_coords)
            exp_dir = hitcalls_dir / f"{exp_id}_{head}"
            hits_dir = exp_dir / suffix.lstrip("_") if suffix else exp_dir
            regions_npz = exp_dir / "regions.npz"
            if not regions_npz.exists() or resolve_hits_path(hits_dir) is None:
                skipped_missing += 1
                continue

            out_tsv = args.out_dir / f"{exp_id}_{head}.tsv"
            if out_tsv.exists() and not args.force:
                skipped_done += 1
                continue

            job_name = f"bpnet_hitcall_background_excess_{exp_id}_{head}{suffix}"
            diagnose_cmd = (
                f"uv run --project {REPO_ROOT} --extra sherlock --frozen python {DIAGNOSE_SCRIPT} "
                f"-e {exp_id} --head {head} -v --out-tsv {out_tsv}"
            )
            if args.min_trim_len is not None:
                diagnose_cmd += f" --min-trim-len {args.min_trim_len}"
            diagnose_cmd += f" {args.diagnose_args}"

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
                #SBATCH --requeue

                ml biology
                ml htslib

                mamba activate "${{PROCAP_ATLAS_ENV:-procap-atlas}}"
                {diagnose_cmd}
            """)

            if args.dry_run:
                print(f"--- {job_name} ---")
                print(sbatch_script)
                submitted += 1
                continue

            result = subprocess.run(["sbatch"], input=sbatch_script, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"{job_name}: {result.stdout.strip()}")
                submitted += 1
            else:
                print(f"ERROR submitting {job_name}: {result.stderr.strip()}", file=sys.stderr)

    action = "Would submit" if args.dry_run else "Submitted"
    total = len(experiments) * len(heads)
    print(
        f"\n{action} {submitted} jobs, skipped {skipped_reads} experiments "
        f"with <{args.min_reads:,} reads, skipped {skipped_missing} missing "
        f"regions.npz/hits, skipped {skipped_done} already written ({total} total)"
    )


if __name__ == "__main__":
    main()
