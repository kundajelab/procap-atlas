#!/usr/bin/env python3
"""Move and rename peak BED files to data/processed/peaks/ with experiment and biosample names."""

import re
import shutil
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "experiment_config.yaml"
PEAKS_INPUT_DIR = REPO_ROOT / "data" / "raw" / "peaks"
OUTPUT_DIR = REPO_ROOT / "data" / "processed" / "peaks"


def main():
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    experiments = config["experiments"]
    total = len(experiments)
    moved = 0
    skipped = 0
    errors = 0

    for i, (exp_id, exp) in enumerate(experiments.items(), 1):
        biosample = re.sub(r"[^\w-]", "_", exp.get("biosample", "")).strip("_")
        peak_type = exp.get("peak_type", "bidirectional")
        filenames = exp.get("peaks", [])
        if not filenames:
            continue

        if len(filenames) > 1:
            print(
                f"WARNING: {exp_id} has {len(filenames)} peak files — "
                f"this is unexpected and may indicate duplicate archived peaks "
                f"that should be added to archive_blacklist.txt: {filenames}",
                file=sys.stderr,
            )

        for j, fn in enumerate(filenames):
            suffix = f"_{j + 1}" if len(filenames) > 1 else ""
            output_path = (
                OUTPUT_DIR / f"{exp_id}_{biosample}_{peak_type}{suffix}.bed.gz"
            )
            if output_path.exists():
                skipped += 1
                continue

            input_path = PEAKS_INPUT_DIR / fn
            if not input_path.exists():
                print(
                    f"[{i}/{total}] WARNING: {exp_id}: missing {fn}, skipping",
                    file=sys.stderr,
                )
                errors += 1
                continue

            print(f"[{i}/{total}] {fn} -> {output_path.name}")
            shutil.move(str(input_path), str(output_path))
            moved += 1

    print(
        f"\nDone: {moved} moved, {skipped} already existed, {errors} skipped due to errors"
    )


if __name__ == "__main__":
    main()
