"""Compute DeepLIFT/SHAP attributions for a trained BPNet model across all folds.

Loads trained models for each fold, computes hypothetical attributions on all
peaks genome-wide, averages across folds, and saves to attributions/bpnet/.

The --background arguments must match those used during training so that the
correct model directory is resolved. Use --model-dir to specify the path
directly instead.

Usage:
    python src/bpnet/attribute/attribute_bpnet.py -e ENCSR261KBX
    python src/bpnet/attribute/attribute_bpnet.py -e ENCSR261KBX --background gc:0.1
    python src/bpnet/attribute/attribute_bpnet.py -e ENCSR261KBX --model-dir models/bpnet/ENCSR261KBX_dnase --head count --ohe
"""

import argparse
import gc
import sys
from itertools import chain
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from bpnetlite.attribute import deep_lift_shap
from bpnetlite.bpnet import CountWrapper, ProfileWrapper
from tangermeme.io import extract_loci

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "experiment_config.yaml"
CHROM_SPLITS_PATH = REPO_ROOT / "configs" / "chrom_splits.yaml"
FASTA = str(REPO_ROOT / "data" / "hg38.fa")
BLACKLIST = str(REPO_ROOT / "data" / "hg38.blacklist.bed.gz")

VALID_BACKGROUNDS = {"ccre", "gc"}


def parse_background(value: str) -> tuple[str, float]:
    """Parse a 'NAME:RATIO' background argument."""
    try:
        name, ratio_str = value.split(":")
        ratio = float(ratio_str)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid background '{value}': expected NAME:RATIO (e.g. ccre:0.05)"
        )
    if name not in VALID_BACKGROUNDS:
        raise argparse.ArgumentTypeError(
            f"Unknown background '{name}': must be one of {sorted(VALID_BACKGROUNDS)}"
        )
    if ratio <= 0:
        raise argparse.ArgumentTypeError(f"Ratio must be positive, got {ratio}")
    return name, ratio


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
    parser.add_argument(
        "--background",
        metavar="NAME:RATIO",
        type=parse_background,
        action="append",
        dest="backgrounds",
        default=None,
        help="background config used during training (repeatable); "
             "default: ccre:0.0714 gc:0.0714",
    )
    parser.add_argument(
        "-m", "--model-dir", type=str, default=None,
        help="override model directory (default: derived from --background)",
    )
    parser.add_argument(
        "--head",
        type=str,
        default="profile",
        choices=["profile", "count"],
        help="type of prediction to make (profile or count)",
    )
    parser.add_argument(
        "--ohe",
        action="store_true",
        help="whether to save one-hot-encoded sequences to disk",
    )
    parser.add_argument("-bs", "--batch-size", type=int, default=64)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.backgrounds is None:
        args.backgrounds = [("ccre", 1 / 14), ("gc", 1 / 14)]

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
    n_folds = len(chrom_splits)

    # Resolve paths from config
    peaks_path = str(REPO_ROOT / processed["filtered_peaks"])
    pl_bw_path = str(REPO_ROOT / processed["pl_bigwig"])
    mn_bw_path = str(REPO_ROOT / processed["mn_bigwig"])

    # Output directory
    out_dir = REPO_ROOT / "attributions" / "bpnet"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Verify data files exist
    for path, label in [
        (peaks_path, "filtered_peaks"),
        (pl_bw_path, "plus bigwig"),
        (mn_bw_path, "minus bigwig"),
    ]:
        if not Path(path).exists():
            print(f"Error: {label} not found: {path}", file=sys.stderr)
            sys.exit(1)

    # Resolve model directory from background config or explicit override
    if args.model_dir:
        model_dir = Path(args.model_dir)
    else:
        bg_suffix = "_".join(
            f"{name}{ratio:g}" for name, ratio in sorted(args.backgrounds)
        )
        model_dir = REPO_ROOT / "models" / "bpnet" / f"{args.experiment}_{bg_suffix}"
    model_paths = [
        model_dir / f"{args.experiment}.fold{fold}.torch" for fold in range(n_folds)
    ]
    for model_path in model_paths:
        if not model_path.exists():
            print(f"Error: model not found: {model_path}", file=sys.stderr)
            sys.exit(1)

    # Build params with defaults, then apply CLI overrides
    params = {
        "sequences": FASTA,
        "signals": [pl_bw_path, mn_bw_path],
        "loci": peaks_path,
        "blacklist": [BLACKLIST],
        "in_window": 2114,
        "out_window": 1000,
        "n_filters": 512,
        "n_layers": 8,
        "batch_size": 64,
        "verbose": False,
    }

    # Apply CLI overrides
    cli_overrides = {"batch_size": args.batch_size}
    for k, v in cli_overrides.items():
        if v is not None:
            params[k] = v

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

    # Calculate attributions for each fold, and store
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

    # Save ohe if requested
    if args.ohe:
        ohe_path = out_dir / f"{args.experiment}_ohe.npz"
        np.savez_compressed(ohe_path, X)
        print(f"\nOne-hot-encoded sequences saved to {ohe_path}")

    # Calculate attributions, looping over folds
    attributions = []
    for fold in range(n_folds):
        # Load model
        model = torch.load(
            model_paths[fold], weights_only=False, map_location=torch.device("cpu")
        )
        if args.head == "profile":
            model = ProfileWrapper(model)
        elif args.head == "count":
            model = CountWrapper(model)
        # Calculate attributions
        attributions.append(
            deep_lift_shap(
                model=model,
                X=X,
                verbose=params["verbose"],
                device="cuda",
                batch_size=params["batch_size"],
                warning_threshold=0.01,
                hypothetical=True,
            )
        )
        del model
        gc.collect()
        torch.cuda.empty_cache()

    out_path = out_dir / f"{model_dir.name}_{args.head}.npz"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, np.stack(attributions).mean(axis=0))
    print(f"\nAttributions saved to {out_path}")


if __name__ == "__main__":
    main()
