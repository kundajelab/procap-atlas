#!/usr/bin/env python3
"""Submit SLURM jobs to call Fi-NeMo motif hits from BPNet attributions.

Reads experiment IDs from configs/experiment_config.yaml and submits one
sbatch job per (experiment, head) pair via call_hits_bpnet.py, calling hits
against the shared MotifCompendium cluster-average motif set for each head.

Jobs are skipped if the output hits.tsv already exists or if the required
OHE/attribution files or the MotifCompendium cluster-average h5 are missing
(run attribute/launch.py and motifcompendium/cluster_motifs_all.py first).

Usage:
    python src/bpnet/hitcall/launch.py                    # submit all experiments, profile head
    python src/bpnet/hitcall/launch.py --dry-run           # print sbatch scripts without submitting
    python src/bpnet/hitcall/launch.py --head count        # count head only
    python src/bpnet/hitcall/launch.py --head profile --head count  # both heads
    python src/bpnet/hitcall/launch.py --time 12:00:00 --mem 32G
    python src/bpnet/hitcall/launch.py --min-reads 20000000  # only well-covered experiments
"""

import argparse
import subprocess
import sys
import textwrap
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "experiment_config.yaml"
N_READS_PATH = REPO_ROOT / "configs" / "n_reads.txt"
CALL_HITS_SCRIPT = REPO_ROOT / "src" / "bpnet" / "hitcall" / "call_hits_bpnet.py"


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
        help="attribution/motif head(s) to call hits against; repeatable (default: profile)",
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
    parser.add_argument("--time", type=str, default="12:00:00")
    parser.add_argument(
        "--min-reads",
        type=int,
        default=0,
        help="skip experiments with fewer total reads than this (default: 0, disabled)",
    )
    parser.add_argument(
        "--call-hits-args",
        type=str,
        default="",
        help="extra arguments forwarded to call_hits_bpnet.py (e.g. '--global-lambda 0.6')",
    )
    args = parser.parse_args()

    heads = args.head if args.head is not None else ["profile"]

    # Load experiment list
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    experiments = list(config["experiments"].keys())

    # Load read counts for filtering
    read_counts_df = pd.read_csv(
        N_READS_PATH, sep="\t", usecols=["experiment", "total_reads"]
    )
    read_counts = dict(zip(read_counts_df["experiment"], read_counts_df["total_reads"]))

    attr_dir = REPO_ROOT / "attributions" / "bpnet"
    mc_dir = REPO_ROOT / "motifcompendium" / "bpnet_all_motifs"
    out_dir = REPO_ROOT / "hitcalls" / "bpnet"
    log_dir = REPO_ROOT / "logs" / "bpnet_hitcall"
    log_dir.mkdir(parents=True, exist_ok=True)

    submitted = 0
    skipped_done = 0
    skipped_missing = 0
    skipped_reads = 0
    for exp_id in experiments:
        # Skip experiments with too few reads
        n_reads = read_counts.get(exp_id, 0)
        if n_reads < args.min_reads:
            skipped_reads += 1
            continue

        model_dir_name = exp_id
        ohe_path = attr_dir / f"{exp_id}_ohe.npz"

        for head in heads:
            attr_path = attr_dir / f"{model_dir_name}_{head}.npz"
            motif_h5 = mc_dir / f"motifcompendium_{head}_cluster_averages.h5"
            if not (ohe_path.exists() and attr_path.exists() and motif_h5.exists()):
                skipped_missing += 1
                continue

            hits_path = out_dir / f"{model_dir_name}_{head}" / "hits.tsv"
            if hits_path.exists():
                skipped_done += 1
                continue

            job_name = f"bpnet_hitcall_{exp_id}_{head}"
            call_hits_cmd = (
                f"uv run --project {REPO_ROOT} --extra sherlock --frozen python {CALL_HITS_SCRIPT} "
                f"-e {exp_id} --head {head} -v {args.call_hits_args}"
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
                #SBATCH --output={log_dir}/{job_name}.out
                #SBATCH --error={log_dir}/{job_name}.err

                ml biology
                ml htslib

                mamba activate "${{PROCAP_ATLAS_ENV:-procap-atlas}}"
                nvidia-smi -L
                {call_hits_cmd}
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
    total = len(experiments) * len(heads)
    print(
        f"\n{action} {submitted} jobs, skipped {skipped_reads} experiments "
        f"with <{args.min_reads:,} reads, skipped {skipped_missing} missing "
        f"attributions/MotifCompendium outputs, skipped {skipped_done} already "
        f"called ({total} total)"
    )


if __name__ == "__main__":
    main()
