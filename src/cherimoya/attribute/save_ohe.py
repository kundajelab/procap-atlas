"""Extract and save one-hot-encoded sequences for a Cherimoya experiment.

Usage:
    python src/cherimoya/attribute/save_ohe.py -e ENCSR261KBX
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


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-e",
        "--experiment",
        type=str,
        required=True,
        help="experiment accession ID (e.g. ENCSR882DWM)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    experiments = config["experiments"]
    if args.experiment not in experiments:
        print(f"Error: {args.experiment} not found in config", file=sys.stderr)
        sys.exit(1)

    exp = experiments[args.experiment]
    processed = exp.get("processed", {})
    peaks_path = str(REPO_ROOT / processed["filtered_peaks"])
    if not Path(peaks_path).exists():
        print(f"Error: filtered_peaks not found: {peaks_path}", file=sys.stderr)
        sys.exit(1)

    with open(CHROM_SPLITS_PATH) as f:
        data = yaml.safe_load(f)
    chrom_splits = {int(k): v for k, v in data["folds"].items()}

    loci = pd.read_csv(
        peaks_path,
        sep="\t",
        usecols=[0, 1, 2],
        header=None,
        index_col=False,
        names=["chrom", "start", "end"],
        dtype={"chrom": str},
    )

    all_chroms = list(chain.from_iterable(chrom_splits.values()))
    X = extract_loci(
        loci=loci,
        sequences=FASTA,
        chroms=all_chroms,
        in_window=2114,
        verbose=args.verbose,
        ignore=list("QWERYUIOPSDFHJKLZXVBNM"),
        exclusion_lists=[BLACKLIST],
    )

    out_dir = REPO_ROOT / "attributions" / "cherimoya"
    out_dir.mkdir(parents=True, exist_ok=True)
    ohe_path = out_dir / f"{args.experiment}_ohe.npz"
    np.savez_compressed(ohe_path, X.numpy())
    print(f"One-hot-encoded sequences saved to {ohe_path}")


if __name__ == "__main__":
    main()
