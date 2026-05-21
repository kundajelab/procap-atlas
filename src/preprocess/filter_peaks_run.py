#!/usr/bin/env python3
"""Filter non-ACGT regions from processed peak BED files for all experiments.

Reads processed paths from configs/experiment_config.yaml and calls
filter_nonACGT_regions for each experiment, writing filtered BED output
to data/processed/peaks/.
"""

import argparse
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

import yaml
from _filter_nonACGT_regions import filter_nonACGT_regions

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "experiment_config.yaml"
FASTA = REPO_ROOT / "data" / "hg38.fa"


def process_experiment(task: dict) -> dict:
    """Filter non-ACGT regions for a single experiment."""
    exp_id = task["exp_id"]
    peaks_path = task["peaks_path"]
    output_path = task["output_path"]
    in_window = task["in_window"]

    try:
        filtered = filter_nonACGT_regions(
            str(peaks_path), str(FASTA), in_window=in_window
        )
        # Write uncompressed BED then bgzip
        tmp_path = output_path.parent / output_path.stem
        filtered.to_csv(tmp_path, sep="\t", index=False, header=False)
        subprocess.run(["bgzip", str(tmp_path)], check=True)
        return {"status": "ok", "exp_id": exp_id}
    except Exception as e:
        return {"status": "error", "exp_id": exp_id, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="Filter non-ACGT regions from processed peaks for all experiments"
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=4,
        help="number of parallel workers (default: 4)",
    )
    parser.add_argument(
        "-w",
        "--window",
        type=int,
        default=2114,
        help="window size around peak center to check (default: 2114)",
    )
    args = parser.parse_args()

    if not FASTA.exists():
        print(
            f"Error: {FASTA} not found. Run src/download/download_genome.sh first.",
            file=sys.stderr,
        )
        sys.exit(1)

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    experiments = config["experiments"]
    total = len(experiments)

    # Build task list
    tasks = []
    skipped_exists = 0
    skipped_missing = 0

    for exp_id, exp in experiments.items():
        processed = exp.get("processed", {})
        peaks_rel = processed.get("peaks", "")
        filtered_rel = processed.get("filtered_peaks", "")

        if not peaks_rel or not filtered_rel:
            continue

        peaks_path = REPO_ROOT / peaks_rel
        output_path = REPO_ROOT / filtered_rel

        if output_path.exists():
            skipped_exists += 1
            continue

        if not peaks_path.exists():
            print(f"{exp_id}: peaks not found ({peaks_path}), skipping")
            skipped_missing += 1
            continue

        output_path.parent.mkdir(parents=True, exist_ok=True)

        tasks.append(
            {
                "exp_id": exp_id,
                "peaks_path": peaks_path,
                "output_path": output_path,
                "in_window": args.window,
            }
        )

    to_process = len(tasks)
    if to_process == 0:
        print(
            f"Nothing to do ({skipped_exists} already existed, "
            f"{skipped_missing} missing peaks)"
        )
        return

    print(
        f"Processing {to_process}/{total} experiments with {args.jobs} workers "
        f"({skipped_exists} already existed, {skipped_missing} missing peaks)"
    )

    processed_count = 0
    errors = 0

    with Pool(processes=args.jobs) as pool:
        for result in pool.imap_unordered(process_experiment, tasks):
            if result["status"] == "ok":
                processed_count += 1
                print(
                    f"[{processed_count + errors}/{to_process}] "
                    f"{result['exp_id']}: done"
                )
            else:
                errors += 1
                print(
                    f"[{processed_count + errors}/{to_process}] "
                    f"ERROR: {result['exp_id']}: {result['error']}"
                )

    print(f"\nDone: {processed_count} processed, {errors} errors")


if __name__ == "__main__":
    main()
