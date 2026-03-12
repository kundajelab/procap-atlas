"""
A small script to just extract and save OHE sequences for a given experiment.

Usage:
    python src/bpnet/attribute/save_ohe.py -e ENCSR261KBX
"""

import argparse
import sys
from itertools import chain
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tangermeme.io import extract_loci

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "experiment_config.yaml"
CHROM_SPLITS_PATH = REPO_ROOT / "configs" / "chrom_splits.yaml"
FASTA = str(REPO_ROOT / "data" / "hg38.fa")
BLACKLIST = str(REPO_ROOT / "data" / "hg38.blacklist.bed.gz")


def load_chrom_splits():
    """Load chromosome fold assignments from chrom_splits.yaml.

    Returns a dict mapping fold number (int) to list of chromosome names.
    """
    with open(CHROM_SPLITS_PATH) as f:
        data = yaml.safe_load(f)
    return {int(k): v for k, v in data["folds"].items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-e",
        "--experiment",
        type=str,
        required=True,
        help="experiment accession ID (e.g. ENCSR882DWM)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    # Load experiment config
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    experiments = config["experiments"]
    if args.experiment not in experiments:
        print(f"Error: {args.experiment} not found in config", file=sys.stderr)
        sys.exit(1)

    exp = experiments[args.experiment]
    processed = exp.get("processed", {})

    # Chromosome splits: fold i = test, fold (i+1)%n = validation, rest = train
    chrom_splits = load_chrom_splits()

    # Resolve paths from config
    peaks_path = str(REPO_ROOT / processed["filtered_peaks"])

    # Output directory
    out_dir = REPO_ROOT / "attributions" / "bpnet"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Verify data files exist
    for path, label in [(peaks_path, "filtered_peaks")]:
        if not Path(path).exists():
            print(f"Error: {label} not found: {path}", file=sys.stderr)
            sys.exit(1)

    # Build params with defaults, then apply CLI overrides
    params = {
        "sequences": FASTA,
        "loci": peaks_path,
        "blacklist": [BLACKLIST],
        "in_window": 2114,
        "verbose": False,
    }

    if args.verbose:
        params["verbose"] = True

    # Load peaks
    loci = pd.read_csv(
        params["loci"],
        sep="\t",
        usecols=[0, 1, 2],
        header=None,
        index_col=False,
        names=["chrom", "start", "end"],
        dtype={"chrom": str},
    )

    print(str(REPO_ROOT / processed["filtered_peaks"]))

    # Load sequences to OHE
    all_chrom = list(chain.from_iterable(chrom_splits.values()))
    X = extract_loci(
        loci=loci,
        sequences=params["sequences"],
        chroms=all_chrom,
        in_window=params["in_window"],
        verbose=params["verbose"],
        ignore=list("QWERYUIOPSDFHJKLZXVBNM"),
        exclusion_lists=params["blacklist"],
    )

    # Save OHE
    ohe_path = out_dir / f"{args.experiment}_ohe.npz"
    np.savez_compressed(ohe_path, X)
    print(f"\nOne-hot-encoded sequences saved to {ohe_path}")


if __name__ == "__main__":
    main()
