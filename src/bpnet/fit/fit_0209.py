"""
Wrapper script to fit a BNBPNet model on PRO-cap data.

Reads experiment paths from configs/experiment_config.yaml and trains a BNBPNet
model using bpnetlite's PeakGenerator for data loading. Supports checkpoint
resumption.

Chromosome splits are read from configs/chrom_splits.yaml. Fold i holds out
fold i for testing and validates on fold (i+1) %% 7. Remaining folds are used
for training.

Usage:
    python src/bpnet/fit/fit.py -e ENCSR882DWM --fold 0
    python src/bpnet/fit/fit.py -e ENCSR882DWM --fold 3 --max-epochs 50 --lr 0.001 ...
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml
from bpnetlite.bpnet import BPNet
from data_loader import PeakGenerator
from tangermeme.io import extract_loci
from torch.optim import AdamW

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "experiment_config.yaml"
CHROM_SPLITS_PATH = REPO_ROOT / "configs" / "chrom_splits.yaml"
FASTA = str(REPO_ROOT / "data" / "hg38.fa")
MAPPABILITY = str(REPO_ROOT / "data" / "k36.Umap.MultiTrackMappability.bw")
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
    parser.add_argument(
        "--fold",
        type=int,
        required=True,
        help="fold to hold out for testing (validation = (fold+1) %% 7)",
    )
    parser.add_argument("-o", "--output-dir", type=str, default=None)
    parser.add_argument("--n-filters", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--count_loss_weight", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--early-stopping", type=int, default=None)
    parser.add_argument("--max-jitter", type=int, default=None)
    parser.add_argument("--negatives-ratio", type=float, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
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

    # Resolve paths from config
    peaks_path = str(REPO_ROOT / processed["peaks"])
    pl_bw_path = str(REPO_ROOT / processed["pl_bigwig"])
    mn_bw_path = str(REPO_ROOT / processed["mn_bigwig"])
    negatives_path = str(REPO_ROOT / processed["gc_negatives"])

    # Verify files exist
    for path, label in [
        (peaks_path, "peaks"),
        (pl_bw_path, "plus bigwig"),
        (mn_bw_path, "minus bigwig"),
    ]:
        if not Path(path).exists():
            print(f"Error: {label} not found: {path}", file=sys.stderr)
            sys.exit(1)

    has_negatives = Path(negatives_path).exists()
    if not has_negatives:
        print(
            f"WARNING: negatives not found ({negatives_path}), training without negatives"
        )

    # Chromosome splits: fold i = test, fold (i+1)%n = validation, rest = train
    chrom_splits = load_chrom_splits()
    n_folds = len(chrom_splits)
    test_fold = args.fold
    valid_fold = (args.fold + 1) % n_folds
    train_folds = [f for f in range(n_folds) if f not in (test_fold, valid_fold)]

    test_chroms = chrom_splits[test_fold]
    valid_chroms = chrom_splits[valid_fold]
    train_chroms = []
    for f in train_folds:
        train_chroms.extend(chrom_splits[f])

    # Output directory
    output_dir = Path(
        args.output_dir or str(REPO_ROOT / "models" / "bpnet" / args.experiment)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build params with defaults, then apply CLI overrides
    params = {
        "name": str(output_dir / f"{args.experiment}.fold{args.fold}"),
        "sequences": FASTA,
        "signals": [pl_bw_path, mn_bw_path],
        "loci": peaks_path,
        "negatives": negatives_path if has_negatives else None,
        # "controls": [MAPPABILITY],
        "blacklist": [BLACKLIST],
        "checkpoint": None,
        "in_window": 2114,
        "out_window": 1000,
        "max_jitter": 200,
        "n_filters": 512,
        "n_layers": 8,
        "count_loss_weight": 100,
        "reverse_complement": True,
        "shuffle": True,
        "batch_size": 64,
        "learning_rate": 0.0005,
        "max_epochs": 50,
        "early_stopping": None,
        "negatives_ratio": 1 / 10,
        "training_chroms": train_chroms,
        "validation_chroms": valid_chroms,
        "random_state": 47,
        "verbose": False,
    }

    # Apply CLI overrides
    cli_overrides = {
        "n_filters": args.n_filters,
        "n_layers": args.n_layers,
        "count_loss_weight": args.count_loss_weight,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "max_epochs": args.max_epochs,
        "early_stopping": args.early_stopping,
        "max_jitter": args.max_jitter,
        "negatives_ratio": args.negatives_ratio,
        "checkpoint": args.checkpoint,
    }
    for k, v in cli_overrides.items():
        if v is not None:
            params[k] = v
    if args.verbose:
        params["verbose"] = True

    # Load peaks
    peaks = pd.read_csv(
        params["loci"],
        sep="\t",
        usecols=[0, 1, 2],
        header=None,
        index_col=False,
        names=["chrom", "start", "end"],
        dtype={"chrom": str},
    )

    # Load negatives
    negatives = None
    if params["negatives"] is not None:
        negatives = pd.read_csv(
            params["negatives"],
            sep="\t",
            usecols=[0, 1, 2],
            header=None,
            index_col=False,
            names=["chrom", "start", "end"],
            dtype={"chrom": str},
        )

    # Training DataLoader
    print(f"Experiment: {args.experiment} ({exp['biosample']})")
    print(f"Fold {args.fold}: test={test_chroms}, valid={valid_chroms}")
    print(f"Loading training data (chroms: {train_chroms})...")
    train_data_loader = PeakGenerator(
        peaks=peaks,
        negatives=negatives,
        sequences=params["sequences"],
        signals=params["signals"],
        # controls=params["controls"],
        chroms=params["training_chroms"],
        in_window=params["in_window"],
        out_window=params["out_window"],
        max_jitter=params["max_jitter"],
        reverse_complement=params["reverse_complement"],
        shuffle=params["shuffle"],
        random_state=params["random_state"],
        batch_size=params["batch_size"],
        verbose=params["verbose"],
        negative_ratio=params["negatives_ratio"],
        exclusion_lists=params["blacklist"],
        min_counts=None,
        max_counts=None,
        pin_memory=True,
        num_workers=0,
    )

    # Validation DataLoader
    print(f"Loading validation data (chroms: {params['validation_chroms']})...")
    val = extract_loci(
        loci=peaks,
        sequences=params["sequences"],
        signals=params["signals"],
        chroms=params["validation_chroms"],
        in_window=params["in_window"],
        out_window=params["out_window"],
        max_jitter=0,
        ignore=list("QWERYUIOPSDFHJKLZXVBNM"),
        exclusion_lists=params["blacklist"],
        verbose=params["verbose"],
    )
    X_valid, y_valid = val
    y_valid = torch.abs(y_valid)

    # Initialize model
    n_outputs = len(params["signals"])
    model = BPNet(
        name=params["name"],
        n_filters=params["n_filters"],
        n_outputs=n_outputs,
        n_control_tracks=0,
        count_loss_weight=params["count_loss_weight"],
        n_layers=params["n_layers"],
        trimming=(params["in_window"] - params["out_window"]) // 2,
        verbose=params["verbose"],
    )
    model = model.to("cuda")
    optimizer = AdamW(model.parameters(), lr=params["learning_rate"])

    # Fit model
    model.fit(
        training_data=train_data_loader,
        optimizer=optimizer,
        X_valid=X_valid,
        y_valid=y_valid,
        max_epochs=params["max_epochs"],
        batch_size=params["batch_size"],
        early_stopping=params["early_stopping"],
        dtype=torch.float,
    )

    print(f"\nModel saved to {output_dir}/")


if __name__ == "__main__":
    main()
