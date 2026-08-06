"""
Consolidated Cherimoya training script with configurable background sampling.

Each --background NAME:RATIO argument adds a negative training source,
where RATIO is the fraction of each training batch drawn from that source.
Multiple --background arguments are combined into a single negatives pool
sampled proportionally by ratio. The output directory name encodes the
background configuration, e.g.:

  {experiment}_ccre0.05_gc0.05

if backgrounds are specified as:

  --background ccre:0.05 --background gc:0.05

The default background is 1/7 GC-matched negatives (no cCREs), giving 1/8 of each
batch as negatives. If no --background arguments are specified, no suffix will be
appended to the output directory name.

Available background sources:
  ccre   cCRE annotations (data/GRCh38-cCREs.bed.gz)
  gc     GC-matched negatives (from experiment config)

Chromosome splits are read from configs/chrom_splits.yaml. Fold i holds out
fold i for testing and validates on fold (i+1) %% 7. Remaining folds are used
for training.

Usage:
    python src/cherimoya/fit/fit_cherimoya.py -e ENCSR261KBX -f 0
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

# Must be set before `import torch` to take effect (read at torch's C++
# extension init time); matches cherimoya_cli's own fit command.
os.environ.setdefault("TORCH_CUDNN_V8_API_ENABLED", "1")

import pandas as pd
import torch
import yaml
from data_loader import PeakGenerator
from tangermeme.io import extract_loci
from torch.optim import SGD, AdamW, Muon
from torch.optim.lr_scheduler import (
    ConstantLR,
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
)

torch.backends.cudnn.benchmark = True

from cherimoya import Cherimoya

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
        "-e",
        "--experiment",
        type=str,
        required=True,
        help="experiment accession ID (e.g. ENCSR261KBX)",
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
        "default: gc:0.1 (1/10 negatives:positives, GC-matched only)",
    )

    parser.add_argument("-o", "--output-dir", type=str, default=None)
    parser.add_argument("--n-filters", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=9)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--early-stopping", type=int, default=None)
    parser.add_argument("--max-jitter", type=int, default=500)
    parser.add_argument("--random-state", type=int, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    # Load experiment config
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    # Validate experiment
    experiments = config["experiments"]
    if args.experiment not in experiments:
        print(f"Error: {args.experiment} not found in config", file=sys.stderr)
        sys.exit(1)

    # Fetch experiment data paths
    exp = experiments[args.experiment]
    processed = exp.get("processed", {})

    peaks_path = str(REPO_ROOT / processed["peaks"])
    pl_bw_path = str(REPO_ROOT / processed["pl_bigwig"])
    mn_bw_path = str(REPO_ROOT / processed["mn_bigwig"])

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
        "gc": REPO_ROOT / processed["gc_negatives"],
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
        args.backgrounds = [("gc", 1 / 10)]

    # Output directory name encodes background sources and ratios only when
    # non-default backgrounds are specified
    if not using_default_backgrounds:
        bg_suffix = "_".join(
            f"{name}{ratio:g}" for name, ratio in sorted(args.backgrounds)
        )
        default_dir = (
            REPO_ROOT / "models" / "cherimoya" / f"{args.experiment}_{bg_suffix}"
        )
    else:
        default_dir = REPO_ROOT / "models" / "cherimoya" / args.experiment

    output_dir = Path(args.output_dir or str(default_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    params = {
        "name": str(output_dir / f"{args.experiment}.fold{args.fold}"),
        "sequences": FASTA,
        "signals": [pl_bw_path, mn_bw_path],
        "loci": peaks_path,
        "blacklist": [BLACKLIST],
        "checkpoint": None,
        "in_window": 2114,
        "out_window": 1000,
        "max_jitter": 500,
        "n_filters": 128,
        "n_layers": 9,
        "reverse_complement": True,
        "shuffle": True,
        "batch_size": 64,
        "muon_lr": 0.025,
        "muon_wd": 0.03,
        "adam_lr": 0.001,
        "adam_wd": 0.0,
        "lw_lr": 0.001,
        "lw_wd": 0.0,
        "lw_momentum": 0.9,
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
        "batch_size": args.batch_size,
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
    print(f"Experiment: {args.experiment} ({exp['biosample']})")
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
        # Nested so cherimoya's normalize_signal_groups treats this as one
        # stranded 2-channel group (correct RC channel-swap behavior)
        # instead of two independent unstranded groups -- the latter is
        # what a flat 2-element list means as of cherimoya's signal-groups
        # refactor. params["signals"] itself stays flat: extract_loci
        # (used directly for validation below) and the model's
        # signal_groups=[len(params["signals"])] both need the flat form.
        signals=[params["signals"]],
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

    # Initialize and train model. torch.compile() unconditionally raises on
    # Python 3.14+ before torch 2.10 (see
    # https://github.com/pytorch/pytorch/issues/169875); Sherlock's only
    # torch>=2.9 build runs on py-pytorch/2.9.1_py314, so disable it there
    # automatically rather than crashing, but re-enable it once torch>=2.10
    # is available (e.g. a future Sherlock module). CheriBlock's fused
    # conv+norm kernel and inference megakernel dispatch on plain
    # `HAS_TRITON and x.is_cuda` checks independent of torch.compile, so
    # this only gives up the extra glue-op fusion torch.compile adds on
    # top, not Triton itself. torch.__version__ is a TorchVersion, which
    # supports PEP 440-aware comparison against a plain string.
    compile_supported = sys.version_info < (3, 14) or torch.__version__ >= "2.10"
    model = Cherimoya(
        name=params["name"],
        n_filters=params["n_filters"],
        signal_groups=[len(params["signals"])],
        n_control_tracks=0,
        n_layers=params["n_layers"],
        trimming=(params["in_window"] - params["out_window"]) // 2,
        verbose=params["verbose"],
        compile=compile_supported,
    )
    model = model.to("cuda")

    # Separate parameters for Muon (2D projection weights), AdamW (everything
    # else, including the 2D depth-wise conv_weight), and SGD (the lw0/lw1
    # Kendall uncertainty loss weights).
    muon_params, adam_params, lw_params = [], [], []
    for name, p in model.named_parameters():
        if name in ("lw0", "lw1"):
            lw_params.append(p)
        elif (
            p.ndim == 2
            and "weight" in name
            and name != "linear.weight"
            and "conv_weight" not in name
        ):
            muon_params.append(p)
        else:
            adam_params.append(p)

    muon_optimizer = Muon(
        muon_params, lr=params["muon_lr"], weight_decay=params["muon_wd"]
    )
    adam_optimizer = AdamW(
        adam_params, lr=params["adam_lr"], weight_decay=params["adam_wd"]
    )
    lw_optimizer = SGD(
        lw_params,
        lr=params["lw_lr"],
        weight_decay=params["lw_wd"],
        momentum=params["lw_momentum"],
    )

    # Warmup + cosine decay schedules
    num_warmup_epochs = 5
    max_epochs = params["max_epochs"]
    num_warmup_iters = len(train_data_loader) * num_warmup_epochs
    num_decay_iters = len(train_data_loader) * max(1, max_epochs - num_warmup_epochs)

    muon_scheduler = SequentialLR(
        muon_optimizer,
        schedulers=[
            LinearLR(muon_optimizer, start_factor=0.01, total_iters=num_warmup_iters),
            CosineAnnealingLR(muon_optimizer, T_max=num_decay_iters, eta_min=1e-5),
        ],
        milestones=[num_warmup_iters],
    )
    adam_scheduler = SequentialLR(
        adam_optimizer,
        schedulers=[
            LinearLR(adam_optimizer, start_factor=0.01, total_iters=num_warmup_iters),
            CosineAnnealingLR(adam_optimizer, T_max=num_decay_iters, eta_min=1e-5),
        ],
        milestones=[num_warmup_iters],
    )
    # Linear warmup then flat (no cosine decay) for the Kendall loss weights.
    lw_scheduler = SequentialLR(
        lw_optimizer,
        schedulers=[
            LinearLR(lw_optimizer, start_factor=0.01, total_iters=num_warmup_iters),
            ConstantLR(lw_optimizer, factor=1.0, total_iters=1),
        ],
        milestones=[num_warmup_iters],
    )

    # Train
    model.fit(
        training_data=train_data_loader,
        muon_optimizer=muon_optimizer,
        adam_optimizer=adam_optimizer,
        lw_optimizer=lw_optimizer,
        muon_scheduler=muon_scheduler,
        adam_scheduler=adam_scheduler,
        lw_scheduler=lw_scheduler,
        X_valid=X_valid,
        X_ctl_valid=None,
        y_valid=y_valid,
        max_epochs=params["max_epochs"],
        batch_size=params["batch_size"],
        early_stopping=params["early_stopping"],
        dtype=torch.float32,
    )

    print(f"\nModel saved to {output_dir}/")


if __name__ == "__main__":
    main()
