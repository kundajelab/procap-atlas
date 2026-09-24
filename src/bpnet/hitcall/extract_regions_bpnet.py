#!/usr/bin/env python3
"""Rebuild just peaks.narrowPeak/regions.npz for an experiment, without
re-running Fi-NeMo call-hits.

call_hits_bpnet.py caches these two files (trim-independent, derived only
from the experiment's own filtered peaks + saved OHE/attribution arrays) and
already regenerates them automatically if missing or corrupt -- but only as
a side effect of also unconditionally re-running `finemo call-hits`, which
has no skip-if-unchanged logic of its own. That overwrites hits.tsv/
hits_unique.tsv (and would require redoing every downstream filter/report
stage) even when only regions.npz itself needs replacing -- e.g. after
regions.npz was deleted directly (or a disk cleanup removed it) while
hits.tsv and every later-stage file were left alone.

Calls call_hits_bpnet.py's own ensure_regions_npz/build_peaks_narrowpeak
directly rather than reimplementing them, so this always stays in sync with
whatever call_hits_bpnet.py itself does to build these two files.

Usage:
    python src/bpnet/hitcall/extract_regions_bpnet.py -e ENCSR882DWM
    python src/bpnet/hitcall/extract_regions_bpnet.py -e ENCSR882DWM --head count
    python src/bpnet/hitcall/extract_regions_bpnet.py -e ENCSR882DWM --model-dir models/bpnet/ENCSR882DWM_gc0.1
    python src/bpnet/hitcall/extract_regions_bpnet.py -e ENCSR882DWM --force  # rebuild even if a valid regions.npz already exists
"""

import argparse
import sys
from pathlib import Path

import yaml

from call_hits_bpnet import IN_WINDOW, ensure_regions_npz

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "experiment_config.yaml"
CHROM_SPLITS_PATH = REPO_ROOT / "configs" / "chrom_splits.yaml"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-e", "--experiment", type=str, required=True)
    parser.add_argument(
        "-m", "--model-dir", type=str, default=None,
        help="override model directory, default derived from config",
    )
    parser.add_argument(
        "--head", type=str, default="profile", choices=["profile", "count"],
    )
    parser.add_argument(
        "--region-width", type=int, default=IN_WINDOW,
        help="must match the value call_hits_bpnet.py was run with (default: 2114)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="rebuild even if a valid regions.npz already exists",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    experiments = config["experiments"]
    if args.experiment not in experiments:
        print(f"Error: {args.experiment} not found in config", file=sys.stderr)
        sys.exit(1)
    peaks_path = str(REPO_ROOT / experiments[args.experiment]["processed"]["filtered_peaks"])

    with open(CHROM_SPLITS_PATH) as f:
        chrom_splits = {int(k): v for k, v in yaml.safe_load(f)["folds"].items()}

    model_dir_name = Path(args.model_dir).name if args.model_dir else args.experiment
    attr_dir = REPO_ROOT / "attributions" / "bpnet"
    ohe_path = attr_dir / f"{args.experiment}_ohe.npz"
    attr_path = attr_dir / f"{model_dir_name}_{args.head}.npz"

    for path, label in [
        (peaks_path, "filtered_peaks"),
        (ohe_path, "OHE sequences"),
        (attr_path, "attributions"),
    ]:
        if not Path(path).exists():
            print(f"Error: {label} not found: {path}", file=sys.stderr)
            sys.exit(1)

    out_dir = REPO_ROOT / "hitcalls" / "bpnet" / f"{model_dir_name}_{args.head}"
    out_dir.mkdir(parents=True, exist_ok=True)

    regions_npz = out_dir / "regions.npz"
    if args.force and regions_npz.exists():
        regions_npz.unlink()

    ensure_regions_npz(
        peaks_path, chrom_splits, ohe_path, attr_path, out_dir,
        args.region_width, args.verbose,
    )


if __name__ == "__main__":
    main()
