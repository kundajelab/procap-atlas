"""
Train on ENCSR261KBX tracks but ENCSR220XSM peaks.

Usage:
    python src/bpnet/fit/fit_bpnet_K562_peak_test.py -f 0
    python src/bpnet/fit/fit_bpnet_K562_peak_test.py -e  -f 0 --background ccre:0.05 --background gc:0.05
"""
# ENCSR220XSM
# ENCSR261KBX

import argparse
import sys
import warnings
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
BLACKLIST = str(REPO_ROOT / "data" / "hg38.blacklist.bed.gz")
CCRES = REPO_ROOT / "data" / "GRCh38-cCREs.bed.gz"

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


def load_bed(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep="\t",
        usecols=[0, 1, 2],
        header=None,
        index_col=False,
        names=["chrom", "start", "end"],
        dtype={"chrom": str},
    )


def build_negatives_pool(
    sources: list[tuple[str, float, pd.DataFrame]],
    random_state: None | int = None,
) -> tuple[pd.DataFrame | None, float]:
    """Combine negatives from multiple sources sampled proportionally by ratio.

    Limits each source so that no source runs out before the others given their
    relative ratios. Returns (combined_df, total_ratio).
    """
    if not sources:
        return None, 0.0

    # k = max samples-per-unit-ratio limited by the source that runs out first
    k = min(len(df) / ratio for _, ratio, df in sources)

    dfs = []
    for name, ratio, df in sources:
        n = int(k * ratio)
        sampled = df.sample(n=n, random_state=random_state) if n < len(df) else df
        print(f"  '{name}': {len(sampled):,} / {len(df):,} negatives")
        dfs.append(sampled)

    return pd.concat(dfs, ignore_index=True), sum(r for _, r, _ in sources)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-f",
        "--fold",
        type=int,
        required=True,
        help="fold to hold out for testing (validation = (fold+1) %% 7)",
    )
    parser.add_argument(
        "--background",
        metavar="NAME:RATIO",
        type=parse_background,
        action="append",
        dest="backgrounds",
        default=None,
        help="background source and per-batch ratio (repeatable); "
        "default: gc:0.1429 (1/7 negatives:positives, GC-matched only)",
    )

    parser.add_argument("-o", "--output-dir", type=str, default=None)
    parser.add_argument("--n-filters", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--count-loss-weight", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--early-stopping", type=int, default=None)
    parser.add_argument("--max-jitter", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    # Load experiment config
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    # Validate experiment
    experiments = config["experiments"]
    experiment = "ENCSR220XSMtracks_ENCSR261KBXpeaks"

    # Fetch experiment data paths
    exp_new = experiments["ENCSR261KBX"]
    exp_old = experiments["ENCSR220XSM"]
    processed_new = exp_new.get("processed", {})
    processed_old = exp_old.get("processed", {})

    peaks_path = str(REPO_ROOT / processed_new["peaks"])
    pl_bw_path = str(REPO_ROOT / processed_old["pl_bigwig"])
    mn_bw_path = str(REPO_ROOT / processed_old["mn_bigwig"])

    # Validate experiment data paths
    for path, label in [
        (peaks_path, "peaks"),
        (pl_bw_path, "plus bigwig"),
        (mn_bw_path, "minus bigwig"),
    ]:
        if not Path(path).exists():
            print(f"Error: {label} not found: {path}", file=sys.stderr)
            sys.exit(1)

    # Fetch background data paths
    background_paths = {
        "ccre": CCRES,
        "gc": REPO_ROOT / processed_new["gc_negatives"],
    }

    # Chromosome splits: fold i = test, fold (i+1)%n = validation, rest = train
    with open(CHROM_SPLITS_PATH) as f:
        data = yaml.safe_load(f)
    chrom_splits = {int(k): v for k, v in data["folds"].items()}
    n_folds = len(chrom_splits)
    test_fold = args.fold
    valid_fold = (args.fold + 1) % n_folds
    train_folds = [f for f in range(n_folds) if f not in (test_fold, valid_fold)]
    test_chroms = chrom_splits[test_fold]
    valid_chroms = chrom_splits[valid_fold]
    train_chroms = [c for f in train_folds for c in chrom_splits[f]]

    # Check whether we're using the default background ratios
    using_default_backgrounds = args.backgrounds is None
    if using_default_backgrounds:
        args.backgrounds = [("gc", 1 / 7)]

    # Output directory name encodes background sources and ratios only when
    # non-default backgrounds are specified
    if not using_default_backgrounds:
        raise NotImplementedError
    else:
        output_dir = REPO_ROOT / "models" / "bpnet" / experiment

    output_dir.mkdir(parents=True, exist_ok=True)

    params = {
        "name": str(output_dir / f"{experiment}.fold{args.fold}"),
        "sequences": FASTA,
        "signals": [pl_bw_path, mn_bw_path],
        "loci": peaks_path,
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
        "training_chroms": train_chroms,
        "validation_chroms": valid_chroms,
        "random_state": None,
        "verbose": False,
    }

    cli_overrides = {
        "n_filters": args.n_filters,
        "n_layers": args.n_layers,
        "count_loss_weight": args.count_loss_weight,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "max_epochs": args.max_epochs,
        "early_stopping": args.early_stopping,
        "max_jitter": args.max_jitter,
        "random_state": args.random_state,
    }
    for k, v in cli_overrides.items():
        if v is not None:
            params[k] = v
    if args.verbose:
        params["verbose"] = True

    # Load peaks
    peaks = load_bed(params["loci"])

    # Load background sources and build negatives pool
    print(f"Fold {args.fold}: test={test_chroms}, valid={valid_chroms}")

    sources = []
    for name, ratio in args.backgrounds:
        path = background_paths[name]
        if not Path(path).exists():
            warnings.warn(f"Background '{name}' not found at {path}, skipping.")
            continue
        sources.append((name, ratio, load_bed(path)))

    if args.backgrounds and not sources:
        warnings.warn(
            "No background sources could be loaded. Training without negatives."
        )

    print(f"Building negatives pool ({len(sources)} source(s)):")
    negatives, negatives_ratio = build_negatives_pool(sources, params["random_state"])

    # Training DataLoader
    print(f"Loading training data (chroms: {train_chroms})...")
    train_data_loader = PeakGenerator(
        peaks=peaks,
        negatives=negatives,
        sequences=params["sequences"],
        signals=params["signals"],
        chroms=params["training_chroms"],
        in_window=params["in_window"],
        out_window=params["out_window"],
        max_jitter=params["max_jitter"],
        reverse_complement=params["reverse_complement"],
        shuffle=params["shuffle"],
        random_state=params["random_state"],
        batch_size=params["batch_size"],
        verbose=params["verbose"],
        negative_ratio=negatives_ratio,
        exclusion_lists=params["blacklist"],
        min_counts=None,
        max_counts=None,
        pin_memory=True,
        num_workers=0,
    )

    # Validation data
    print(f"Loading validation data (chroms: {valid_chroms})...")
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

    # Initialize and train model
    model = BPNet(
        name=params["name"],
        n_filters=params["n_filters"],
        n_outputs=len(params["signals"]),
        n_control_tracks=0,
        count_loss_weight=params["count_loss_weight"],
        n_layers=params["n_layers"],
        trimming=(params["in_window"] - params["out_window"]) // 2,
        verbose=params["verbose"],
    )
    model = model.to("cuda")
    optimizer = AdamW(model.parameters(), lr=params["learning_rate"])

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
