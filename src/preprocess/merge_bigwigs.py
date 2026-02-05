#!/usr/bin/env python3
"""Merge replicate BigWig files per experiment, keeping strands separate.

For each experiment in the YAML config, merges all plus-strand replicate
bigwigs into one file and all minus-strand replicate bigwigs into another.
Single-replicate experiments are moved/renamed rather than reprocessed.

Requires UCSC Kent tools: bigWigMerge, bedGraphToBigWig

Usage:
    python src/preprocess/merge_bigwigs.py            # default 4 workers
    python src/preprocess/merge_bigwigs.py -j 8        # 8 workers
    python src/preprocess/merge_bigwigs.py -j 1        # sequential
"""

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "experiment_config.yaml"
BIGWIG_INPUT_DIR = REPO_ROOT / "data" / "raw" / "bigwigs"
OUTPUT_DIR = REPO_ROOT / "data" / "processed" / "bigwigs"
CHROM_SIZES = REPO_ROOT / "data" / "hg38.chrom.sizes"


def check_dependencies():
    """Verify required tools are on PATH."""
    missing = []
    for tool in ("bigWigMerge", "bedGraphToBigWig"):
        if shutil.which(tool) is None:
            missing.append(tool)
    if missing:
        print(
            f"Error: missing required UCSC tools: {', '.join(missing)}\n"
            "Install from https://hgdownload.soe.ucsc.edu/admin/exe/",
            file=sys.stderr,
        )
        sys.exit(1)


def merge_bigwigs(input_paths: list[Path], output_path: Path):
    """Merge BigWig files into a single BigWig via bigWigMerge + bedGraphToBigWig."""
    if len(input_paths) == 1:
        shutil.copy2(str(input_paths[0]), str(output_path))
        return

    with tempfile.TemporaryDirectory(dir=OUTPUT_DIR) as tmpdir:
        bg_path = Path(tmpdir) / "merged.bedGraph"
        bg_sorted = Path(tmpdir) / "merged.sorted.bedGraph"

        subprocess.run(
            [
                "bigWigMerge",
                "-threshold=-10000000",
                *[str(p) for p in input_paths],
                str(bg_path),
            ],
            check=True,
        )
        if bg_path.stat().st_size == 0:
            raise RuntimeError("bigWigMerge produced an empty bedGraph")
        # bedGraphToBigWig requires LC_COLLATE=C sorted input
        subprocess.run(
            f"LC_COLLATE=C sort -k1,1 -k2,2n '{bg_path}' > '{bg_sorted}'",
            shell=True,
            check=True,
        )
        subprocess.run(
            ["bedGraphToBigWig", str(bg_sorted), str(CHROM_SIZES), str(output_path)],
            check=True,
        )


def process_task(task: dict) -> dict:
    """Run a single merge task. Returns a result dict with status info."""
    try:
        merge_bigwigs(task["input_paths"], task["output_path"])
        return {"status": "ok", "label": task["label"]}
    except Exception as e:
        return {"status": "error", "label": task["label"], "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Merge replicate BigWig files")
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=4,
        help="number of parallel workers (default: 4)",
    )
    args = parser.parse_args()

    check_dependencies()

    if not CHROM_SIZES.exists():
        print(
            f"Error: {CHROM_SIZES} not found. "
            "Run src/download/download_genome.sh first.",
            file=sys.stderr,
        )
        sys.exit(1)

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build task list
    tasks = []
    skipped = 0
    errors = 0
    experiments = config["experiments"]

    for exp_id, exp in experiments.items():
        biosample = re.sub(r"[^\w-]", "_", exp.get("biosample", "")).strip("_")
        for strand, key in [("pl", "pl_bigwigs"), ("mn", "mn_bigwigs")]:
            filenames = exp[key]
            if not filenames:
                continue

            output_path = OUTPUT_DIR / f"{exp_id}_{biosample}_{strand}.bigWig"
            if output_path.exists():
                skipped += 1
                continue

            input_paths = [BIGWIG_INPUT_DIR / fn for fn in filenames]
            missing = [p for p in input_paths if not p.exists()]
            if missing:
                print(
                    f"WARNING: {exp_id} {strand}: "
                    f"missing inputs: {[p.name for p in missing]}, skipping",
                    file=sys.stderr,
                )
                errors += 1
                continue

            n = len(input_paths)
            action = "moving" if n == 1 else f"merging {n} replicates"
            label = f"{exp_id}_{biosample}_{strand}"
            tasks.append(
                {
                    "input_paths": input_paths,
                    "output_path": output_path,
                    "label": label,
                    "action": action,
                }
            )

    total = len(tasks)
    if total == 0:
        print(
            f"Nothing to do ({skipped} already existed, {errors} skipped due to errors)"
        )
        return

    print(
        f"Processing {total} tasks with {args.jobs} workers "
        f"({skipped} already existed, {errors} skipped due to errors)"
    )

    completed = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(process_task, t): t for t in tasks}
        for future in as_completed(futures):
            task = futures[future]
            result = future.result()
            completed += 1
            if result["status"] == "ok":
                print(f"[{completed}/{total}] {task['label']}: {task['action']}")
            else:
                print(
                    f"[{completed}/{total}] ERROR: {task['label']}: {result['error']}",
                    file=sys.stderr,
                )
                errors += 1

    print(f"\nDone: {completed} processed, {skipped} already existed, {errors} errors")


if __name__ == "__main__":
    main()
