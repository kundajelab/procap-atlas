#!/usr/bin/env python3
"""Merge bidirectional and unidirectional peak BED files into data/processed/peaks/."""

import argparse
import shutil
import sys
from multiprocessing import Pool
from pathlib import Path

import yaml

from _merge_uni_bi_peaks import merge_uni_bi_peaks, write_to_tsv

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "experiment_config.yaml"
PEAKS_INPUT_DIR = REPO_ROOT / "data" / "raw" / "peaks"
OUTPUT_DIR = REPO_ROOT / "data" / "processed" / "peaks"


def process_experiment(task: dict) -> dict:
    """Merge or copy peak files for a single experiment."""
    exp_id = task["exp_id"]
    bi_path = task["bi_path"]
    uni_path = task["uni_path"]
    output_path = task["output_path"]

    try:
        if bi_path and uni_path:
            all_peaks = merge_uni_bi_peaks(str(uni_path), str(bi_path))
            write_to_tsv(str(output_path), all_peaks)
            return {"status": "merged", "exp_id": exp_id}
        elif bi_path:
            shutil.copy2(str(bi_path), str(output_path))
            return {"status": "copied", "exp_id": exp_id}
        elif uni_path:
            shutil.copy2(str(uni_path), str(output_path))
            return {"status": "copied", "exp_id": exp_id}
    except Exception as e:
        return {"status": "error", "exp_id": exp_id, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="Merge bidirectional and unidirectional peaks for all experiments"
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=4,
        help="number of parallel workers (default: 4)",
    )
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    experiments = config["experiments"]
    total = len(experiments)

    # Build task list, filtering out experiments that should be skipped
    tasks = []
    skipped = 0

    for exp_id, exp in experiments.items():
        output_path = REPO_ROOT / exp["processed"]["peaks"]
        if output_path.exists():
            skipped += 1
            continue

        bi_filenames = exp.get("peaks", [])
        uni_filenames = exp.get("unidirectional_peaks", [])

        if not bi_filenames and not uni_filenames:
            continue

        if len(bi_filenames) > 1:
            print(
                f"WARNING: {exp_id} has {len(bi_filenames)} bidirectional peak files — "
                f"this is unexpected and may indicate duplicate archived peaks "
                f"that should be added to archive_blacklist.txt: {bi_filenames}",
                file=sys.stderr,
            )

        if len(uni_filenames) > 1:
            print(
                f"WARNING: {exp_id} has {len(uni_filenames)} unidirectional peak files — "
                f"this is unexpected and may indicate duplicate archived peaks "
                f"that should be added to archive_blacklist.txt: {uni_filenames}",
                file=sys.stderr,
            )

        bi_path = PEAKS_INPUT_DIR / bi_filenames[0] if bi_filenames else None
        uni_path = PEAKS_INPUT_DIR / uni_filenames[0] if uni_filenames else None

        # Check that input files exist
        missing = False
        if bi_path and not bi_path.exists():
            print(
                f"WARNING: {exp_id}: missing {bi_filenames[0]}, skipping",
                file=sys.stderr,
            )
            missing = True
        if uni_path and not uni_path.exists():
            print(
                f"WARNING: {exp_id}: missing {uni_filenames[0]}, skipping",
                file=sys.stderr,
            )
            missing = True
        if missing:
            continue

        tasks.append(
            {
                "exp_id": exp_id,
                "bi_path": bi_path,
                "uni_path": uni_path,
                "output_path": output_path,
            }
        )

    to_process = len(tasks)
    if to_process == 0:
        print(f"Nothing to do ({skipped} already existed)")
        return

    print(
        f"Processing {to_process}/{total} experiments with {args.jobs} workers "
        f"({skipped} already existed)"
    )

    merged = 0
    copied = 0
    errors = 0

    with Pool(processes=args.jobs) as pool:
        for result in pool.imap_unordered(process_experiment, tasks):
            done = merged + copied + errors + 1
            if result["status"] == "merged":
                merged += 1
                print(f"[{done}/{to_process}] {result['exp_id']}: merged")
            elif result["status"] == "copied":
                copied += 1
                print(f"[{done}/{to_process}] {result['exp_id']}: copied")
            else:
                errors += 1
                print(
                    f"[{done}/{to_process}] ERROR: {result['exp_id']}: {result['error']}"
                )

    print(
        f"\nDone: {merged} merged, {copied} copied, {skipped} already existed, "
        f"{errors} errors"
    )


if __name__ == "__main__":
    main()
