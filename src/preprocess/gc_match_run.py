#!/usr/bin/env python3
"""Generate GC-matched negative regions for each experiment's peak calls.

Reads processed paths from configs/experiment_config.yaml and calls
gc_match.extract_matching_loci for each experiment, writing bgzip-compressed
BED output to data/processed/negatives/.
"""

import argparse
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

import yaml
from _gc_match import extract_matching_loci

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "experiment_config.yaml"
FASTA = REPO_ROOT / "data" / "hg38.fa"
OUTPUT_DIR = REPO_ROOT / "data" / "processed" / "negatives"

CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX"]


def process_experiment(task: dict) -> dict:
    """Run GC-matched negative extraction for a single experiment."""
    exp_id = task["exp_id"]
    peaks_path = task["peaks_path"]
    bigwigs = task["bigwigs"]
    output_path = task["output_path"]

    try:
        matched_loci = extract_matching_loci(
            loci=str(peaks_path),
            fasta=str(FASTA),
            bigwig=bigwigs,
            chroms=CHROMS,
            n_jobs=1,
        )

        # Write uncompressed BED then bgzip
        tmp_path = output_path.parent / output_path.stem
        matched_loci.to_csv(tmp_path, header=False, sep="\t", index=False)
        subprocess.run(["bgzip", str(tmp_path)], check=True)
        return {"status": "ok", "exp_id": exp_id}
    except Exception as e:
        return {"status": "error", "exp_id": exp_id, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="Generate GC-matched negatives for all experiments"
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        help="number of parallel workers (default: 4)",
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

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    experiments = config["experiments"]
    total = len(experiments)

    # Build task list, filtering out experiments that should be skipped
    tasks = []
    skipped_exists = 0
    skipped_no_peaks = 0
    skipped_missing_peaks = 0

    for exp_id, exp in experiments.items():
        processed = exp.get("processed", {})
        peaks_rel = processed.get("peaks", "")
        pl_bw_rel = processed.get("pl_bigwig", "")
        mn_bw_rel = processed.get("mn_bigwig", "")
        gc_neg_rel = processed.get("gc_negatives", "")

        if not peaks_rel or not gc_neg_rel:
            skipped_no_peaks += 1
            continue

        peaks_path = REPO_ROOT / peaks_rel
        pl_bw_path = REPO_ROOT / pl_bw_rel
        mn_bw_path = REPO_ROOT / mn_bw_rel
        output_path = REPO_ROOT / gc_neg_rel

        if output_path.exists():
            skipped_exists += 1
            continue

        if not peaks_path.exists():
            print(f"{exp_id}: peaks not found ({peaks_path}), skipping")
            skipped_missing_peaks += 1
            continue

        bigwigs = None
        if pl_bw_path.exists() and mn_bw_path.exists():
            bigwigs = [str(pl_bw_path), str(mn_bw_path)]
        else:
            print(
                f"{exp_id}: WARNING: bigwig(s) not found, running without signal filter"
            )

        tasks.append(
            {
                "exp_id": exp_id,
                "peaks_path": peaks_path,
                "bigwigs": bigwigs,
                "output_path": output_path,
            }
        )

    to_process = len(tasks)
    if to_process == 0:
        print(
            f"Nothing to do ({skipped_exists} already existed, "
            f"{skipped_missing_peaks} missing peaks, {skipped_no_peaks} no peaks in config)"
        )
        return

    print(
        f"Processing {to_process}/{total} experiments with {args.jobs} workers "
        f"({skipped_exists} already existed, {skipped_missing_peaks} missing peaks, "
        f"{skipped_no_peaks} no peaks in config)"
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
