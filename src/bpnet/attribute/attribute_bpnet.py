"""Compute DeepLIFT/SHAP attributions for a trained BPNet model across all folds.

Loads trained models for each fold, computes hypothetical attributions on all
peaks genome-wide, averages across folds, and saves to attributions/bpnet/.

Usage:
    python src/bpnet/attribute/attribute_bpnet.py -e ENCSR261KBX
    python src/bpnet/attribute/attribute_bpnet.py -e ENCSR261KBX --head count
    python src/bpnet/attribute/attribute_bpnet.py -e ENCSR261KBX --model-dir models/bpnet/ENCSR261KBX_dnase
    python src/bpnet/attribute/attribute_bpnet.py -e ENCSR261KBX --reference-mode dinucleotide
"""

import argparse
import gc
import sys
from itertools import chain
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import torch
from bpnetlite.bpnet import CountWrapper, ProfileWrapper
from tangermeme.io import extract_loci

try:
    from src.bpnet.attribute.deeplift import deep_lift_shap
except ModuleNotFoundError:
    from deeplift import deep_lift_shap

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
    
def nucleotide_frequency_references(X):
    """Return one soft reference per sequence using observed base frequencies."""
    frequencies = X.float().mean(dim=-1, keepdim=True)
    return frequencies.expand_as(X).unsqueeze(1).clone()


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
        "-m",
        "--model-dir",
        type=str,
        default=None,
        help="override model directory, default derived from config",
    )
    parser.add_argument(
        "--head",
        type=str,
        default="profile",
        choices=["profile", "count"],
        help="type of prediction to make (profile or count)",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=64)
    parser.add_argument(
        "--reference-mode",
        choices=("frequency", "dinucleotide"),
        default="frequency",
        help=(
            "DeepLIFT reference baseline. 'frequency' uses one soft "
            "input-wide nucleotide-frequency reference per sequence; "
            "'dinucleotide' uses bpnet-lite/tangermeme's dinucleotide shuffles."
        ),
    )
    parser.add_argument(
        "--n-shuffles",
        type=int,
        default=20,
        help="number of dinucleotide shuffles when --reference-mode dinucleotide",
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
    with open(CHROM_SPLITS_PATH) as f:
        data = yaml.safe_load(f)
    chrom_splits = {int(k): v for k, v in data["folds"].items()}
    n_folds = len(chrom_splits)

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

    # Resolve model directory from background config or explicit override
    if args.model_dir:
        model_dir = Path(args.model_dir)
    else:
        model_dir = REPO_ROOT / "models" / "bpnet" / args.experiment
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

    # Load sequences
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
        if args.reference_mode == "frequency":
            references = nucleotide_frequency_references(X)
            n_shuffles = 1
        else:
            references = None
            n_shuffles = args.n_shuffles

        # Calculate attributions
        attribution_kwargs = {
            "model": model,
            "X": X,
            "verbose": params["verbose"],
            "device": "cuda",
            "batch_size": params["batch_size"],
            "warning_threshold": 0.01,
            "hypothetical": True,
            "n_shuffles": n_shuffles,
        }
        if references is not None:
            attribution_kwargs["references"] = references
        attributions.append(
            deep_lift_shap(**attribution_kwargs)
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
