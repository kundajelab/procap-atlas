"""Compute DeepLIFT/SHAP attributions for a trained Cherimoya model across all folds.

Loads trained models for each fold, computes hypothetical attributions on all
peaks genome-wide, averages across folds, and saves to attributions/cherimoya/.

Uses the genomic nucleotide-frequency reference by default (matching
attribute_bpnet.py). Use --reference-mode dinucleotide for dinucleotide-shuffled
references.

This wrapper exists instead of calling `cherimoya attribute` directly because
the CLI does not support custom reference callables (only dinucleotide
shuffles via n_shuffles), attributes a single model file rather than averaging
across folds, and crops attributions to attr_window (400bp) rather than saving
full-width.

Usage:
    python src/cherimoya/attribute/attribute_cherimoya.py -e ENCSR882DWM
    python src/cherimoya/attribute/attribute_cherimoya.py -e ENCSR882DWM --head profile
    python src/cherimoya/attribute/attribute_cherimoya.py -e ENCSR882DWM --reference-mode dinucleotide
    python src/cherimoya/attribute/attribute_cherimoya.py -e ENCSR882DWM --model-dir models/cherimoya/ENCSR882DWM_gc0.1
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
from cherimoya import Cherimoya, ControlWrapper, LogCountWrapper, ProfileWrapper
from cherimoya.deep_lift_shap import attribution_ops
from tangermeme.deep_lift_shap import deep_lift_shap
from tangermeme.io import extract_loci

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "experiment_config.yaml"
CHROM_SPLITS_PATH = REPO_ROOT / "configs" / "chrom_splits.yaml"
FASTA = str(REPO_ROOT / "data" / "hg38.fa")
BLACKLIST = str(REPO_ROOT / "data" / "hg38.blacklist.bed.gz")


def nucleotide_frequency_references(X, n=1, random_state=None):
    """Return soft PFM references from each sequence's observed base frequencies."""
    if n < 1:
        raise ValueError("n must be at least 1")
    frequencies = X.float().mean(dim=-1, keepdim=True)
    return (
        frequencies.expand(-1, -1, X.shape[-1])
        .unsqueeze(1)
        .expand(-1, n, -1, -1)
        .clone()
    )


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
        help="type of prediction to attribute (default: profile)",
    )
    parser.add_argument(
        "--group",
        type=int,
        default=0,
        help=(
            "signal group index to attribute (default: 0). Atlas models "
            "have one stranded group, so 0 is the only valid value."
        ),
    )
    parser.add_argument("-b", "--batch-size", type=int, default=64)
    parser.add_argument(
        "--reference-mode",
        choices=("frequency", "dinucleotide"),
        default="frequency",
        help=(
            "DeepLIFT reference baseline. 'frequency' uses one soft "
            "input-wide nucleotide-frequency reference per sequence; "
            "'dinucleotide' uses tangermeme's dinucleotide shuffles."
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
    n_folds = len(chrom_splits)

    if args.model_dir:
        model_dir = Path(args.model_dir)
    else:
        model_dir = REPO_ROOT / "models" / "cherimoya" / args.experiment
    model_paths = [
        model_dir / f"{args.experiment}.fold{fold}.torch" for fold in range(n_folds)
    ]
    for path in model_paths:
        if not path.exists():
            print(f"Error: model not found: {path}", file=sys.stderr)
            sys.exit(1)

    print(f"Experiment: {args.experiment} ({exp['biosample']})")
    print(f"Model dir:  {model_dir}")
    print(f"Head:       {args.head}")
    print(f"Reference:  {args.reference_mode}")

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

    if args.reference_mode == "frequency":
        n_shuffles = 1
        references = nucleotide_frequency_references
    else:
        references = None
        n_shuffles = args.n_shuffles

    attributions = []
    for fold in range(n_folds):
        print(f"\nFold {fold}: {model_paths[fold].name}")
        model = Cherimoya.load(
            model_paths[fold], device="cpu", compile=False
        )
        model = ControlWrapper(model)
        if args.head == "count":
            wrapper = LogCountWrapper(model, group=args.group)
        else:
            wrapper = ProfileWrapper(model, group=args.group)

        kwargs = {
            "model": wrapper,
            "X": X,
            "hypothetical": True,
            "n_shuffles": n_shuffles,
            "batch_size": args.batch_size,
            "warning_threshold": 1e-3,
            "additional_nonlinear_ops": attribution_ops(),
            "device": "cuda",
            "verbose": args.verbose,
        }
        if references is not None:
            kwargs["references"] = references
        attributions.append(deep_lift_shap(**kwargs))

        del model, wrapper
        gc.collect()
        torch.cuda.empty_cache()

    out_dir = REPO_ROOT / "attributions" / "cherimoya"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{model_dir.name}_{args.head}.npz"
    np.savez_compressed(out_path, np.stack(attributions).mean(axis=0))
    print(f"\nAttributions saved to {out_path}")

    ohe_path = out_dir / f"{args.experiment}_ohe.npz"
    if not ohe_path.exists():
        np.savez_compressed(ohe_path, X.numpy())
        print(f"OHE saved to {ohe_path}")


if __name__ == "__main__":
    main()
