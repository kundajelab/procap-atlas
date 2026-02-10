# mkdir -p ../../../models/procapnet/ENCSR261KBX/
# wget https://www.encodeproject.org/files/ENCFF976FHE/@@download/ENCFF976FHE.tar.gz -O "../../../models/procapnet/models.tar.gz"
# tar -xvf "../../../models/procapnet/models.tar.gz" -C "../../../models/procapnet/ENCSR261KBX/"
# rm "../../../models/procapnet/models.tar.gz"

"""Benchmark a trained ProCapNet model across all folds.

Loads trained models for each fold, predicts on the held-out test chromosomes,
and reports profile and count prediction metrics.

Usage:
    python src/procapnet/benchmark/benchmark.py -e ENCSR261KBX -v
    python src/procapnet/benchmark/benchmark.py -e ENCSR261KBX -o -v # To save output
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from bpnetlite.performance import jensen_shannon_distance, pearson_corr, spearman_corr
from personal_bpnet.procapnet import ProCapNet
from tangermeme.io import extract_loci
from tangermeme.predict import predict

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
        "-o",
        "--save-output",
        action="store_true",
        help="whether to save predictions + signals to disk",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=None)
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
    n_folds = len(chrom_splits)

    # Resolve paths from config
    peaks_path = str(REPO_ROOT / processed["peaks"])
    pl_bw_path = str(REPO_ROOT / processed["pl_bigwig"])
    mn_bw_path = str(REPO_ROOT / processed["mn_bigwig"])

    # Verify data files exist
    for path, label in [
        (peaks_path, "peaks"),
        (pl_bw_path, "plus bigwig"),
        (mn_bw_path, "minus bigwig"),
    ]:
        if not Path(path).exists():
            print(f"Error: {label} not found: {path}", file=sys.stderr)
            sys.exit(1)

    # Model directory
    model_dir = Path(str(REPO_ROOT / "models" / "procapnet" / args.experiment))
    model_paths = [
        model_dir
        / f"fold_{fold}"
        / f"{args.experiment}.procapnet_model.fold{fold}.state_dict.torch"
        for fold in range(n_folds)
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
        "controls": [MAPPABILITY],
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

    # Predict on test sets
    signals = []
    preds = []
    for fold in range(n_folds):
        # Load data
        test_chrom = chrom_splits[fold]
        X, y = extract_loci(
            loci=loci,
            sequences=params["sequences"],
            chroms=test_chrom,
            signals=params["signals"],
            in_window=params["in_window"],
            out_window=params["out_window"],
            verbose=params["verbose"],
            ignore=list("QWERYUIOPSDFHJKLZXVBNM"),
            exclusion_lists=params["blacklist"],
        )
        signals.append(torch.abs(y))

        # Load model
        model = ProCapNet(
            n_filters=params["n_filters"],
            n_outputs=len(params["signals"]),
            n_layers=params["n_layers"],
        )
        model.load_state_dict(
            torch.load(
                model_paths[fold],
                weights_only=True,
                map_location=torch.device("cpu"),
            )
        )

        # Predict
        preds.append(
            predict(
                model=model,
                X=X,
                verbose=params["verbose"],
                device="cuda" if torch.cuda.is_available() else "cpu",
                batch_size=params["batch_size"],
            )
        )

    # Calculate performance metrics
    profile_corr = [
        pearson_corr(
            torch.nn.functional.softmax(pred[0].reshape(pred[0].shape[0], -1), dim=-1),
            signal.reshape(pred[0].shape[0], -1),
        ).numpy()
        for pred, signal in zip(preds, signals)
    ]
    profile_jsd = [
        jensen_shannon_distance(
            torch.nn.functional.log_softmax(
                pred[0].reshape(pred[0].shape[0], -1), dim=-1
            ),
            signal.reshape(pred[0].shape[0], -1),
        ).numpy()
        for pred, signal in zip(preds, signals)
    ]
    log_counts_pearson = [
        pearson_corr(pred[1].squeeze(), torch.log1p(signal.sum(dim=(-1, -2)))).item()
        for pred, signal in zip(preds, signals)
    ]
    counts_spearman = [
        spearman_corr(pred[1].squeeze(), signal.sum(dim=(-1, -2))).item()
        for pred, signal in zip(preds, signals)
    ]
    log_counts_pearson_all = pearson_corr(
        torch.cat([pred[1].squeeze() for pred in preds]),
        torch.cat([torch.log1p(signal.sum(dim=(-1, -2))) for signal in signals]),
    ).item()
    counts_spearman_all = spearman_corr(
        torch.cat([pred[1].squeeze() for pred in preds]),
        torch.cat([signal.sum(dim=(-1, -2)) for signal in signals]),
    ).item()

    # Output
    print("Per-fold results:\n----------------")
    print(
        f"Profile Pearson correlation: {[np.nanmedian(c).item() for c in profile_corr]}"
        f" (n_nan={[np.isnan(c).mean().item() for c in profile_corr]})"
    )
    print(
        f"Profile Jensen-Shannon distance: {[np.nanmedian(j).item() for j in profile_jsd]} "
        f"(n_nan={[np.isnan(j).mean().item() for j in profile_jsd]})"
    )
    print(f"Log Counts Pearson correlation: {log_counts_pearson}")
    print(f"Counts Spearman correlation: {counts_spearman}\n")

    print("\nGenome-wide results:\n----------------")
    print(f"Profile Pearson correlation: {np.nanmedian(np.concatenate(profile_corr))}")
    print(
        f"Profile Jensen-Shannon distance: {np.nanmedian(np.concatenate(profile_jsd))}"
    )
    print(f"Log Counts Pearson correlation: {log_counts_pearson_all}")
    print(f"Counts Spearman correlation: {counts_spearman_all}")

    if args.save_output:
        output_dir = str(REPO_ROOT) / "predictions" / "bpnet"
        output_path = output_dir / f"{args.experiment}.npz"
        np.savez_compressed(output_path, preds=preds, signals=signals)
        print(f"\nPredictions saved to {output_path}")


if __name__ == "__main__":
    main()
