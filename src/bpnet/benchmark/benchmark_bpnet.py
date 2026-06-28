"""
Benchmark a trained BPNet model across all folds.

Loads trained models for each fold, predicts on the held-out test chromosomes,
and reports profile and count prediction metrics.

Use --model-dir to specify the path directly if trained with custom negative ratios.

Usage:
    python src/bpnet/benchmark/benchmark_bpnet.py -e ENCSR261KBX
    python src/bpnet/benchmark/benchmark_bpnet.py -e ENCSR261KBX --model-dir models/bpnet/ENCSR261KBX_dnase
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from bpnetlite.performance import jensen_shannon_distance, pearson_corr, spearman_corr
from tangermeme.io import extract_loci
from tangermeme.predict import predict

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.modeling.profile import (
    count_scaled_profile,
    orientation_index,
    profile_log_probabilities,
    profile_probabilities,
)

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
    parser.add_argument(
        "-m",
        "--model-dir",
        type=str,
        default=None,
        help="override model directory, default derived from config",
    )
    parser.add_argument(
        "-o",
        "--save-output",
        action="store_true",
        help="save predictions and signals to disk",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=64)
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

    peaks_path = str(REPO_ROOT / processed["peaks"])
    pl_bw_path = str(REPO_ROOT / processed["pl_bigwig"])
    mn_bw_path = str(REPO_ROOT / processed["mn_bigwig"])

    for path, label in [
        (peaks_path, "peaks"),
        (pl_bw_path, "plus bigwig"),
        (mn_bw_path, "minus bigwig"),
    ]:
        if not Path(path).exists():
            print(f"Error: {label} not found: {path}", file=sys.stderr)
            sys.exit(1)

    # Resolve model directory from config or explicit override
    if args.model_dir:
        model_dir = Path(args.model_dir)
    else:
        model_dir = REPO_ROOT / "models" / "bpnet" / args.experiment

    with open(CHROM_SPLITS_PATH) as f:
        data = yaml.safe_load(f)
    chrom_splits = {int(k): v for k, v in data["folds"].items()}
    n_folds = len(chrom_splits)

    model_paths = [
        model_dir / f"{args.experiment}.fold{fold}.torch" for fold in range(n_folds)
    ]
    for model_path in model_paths:
        if not model_path.exists():
            print(f"Error: model not found: {model_path}", file=sys.stderr)
            sys.exit(1)

    params = {
        "sequences": FASTA,
        "signals": [pl_bw_path, mn_bw_path],
        "loci": peaks_path,
        "blacklist": [BLACKLIST],
        "in_window": 2114,
        "out_window": 1000,
        "batch_size": 64,
        "verbose": False,
    }

    if args.batch_size is not None:
        params["batch_size"] = args.batch_size
    if args.verbose:
        params["verbose"] = True

    loci = pd.read_csv(
        params["loci"],
        sep="\t",
        usecols=[0, 1, 2],
        header=None,
        index_col=False,
        names=["chrom", "start", "end"],
        dtype={"chrom": str},
    )

    print(f"Experiment: {args.experiment} ({exp['biosample']})")
    print(f"Model dir: {model_dir}")

    # Predict on each fold's test chromosomes
    signals = []
    preds = []
    for fold in range(n_folds):
        test_chroms = chrom_splits[fold]
        X, y = extract_loci(
            loci=loci,
            sequences=params["sequences"],
            chroms=test_chroms,
            signals=params["signals"],
            in_window=params["in_window"],
            out_window=params["out_window"],
            verbose=params["verbose"],
            ignore=list("QWERYUIOPSDFHJKLZXVBNM"),
            exclusion_lists=params["blacklist"],
        )
        signals.append(torch.abs(y))

        model = torch.load(
            model_paths[fold], weights_only=False, map_location=torch.device("cpu")
        )
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
            profile_probabilities(pred[0]).reshape(pred[0].shape[0], -1),
            signal.reshape(pred[0].shape[0], -1),
        ).numpy()
        for pred, signal in zip(preds, signals)
    ]
    profile_jsd = [
        jensen_shannon_distance(
            profile_log_probabilities(pred[0]).reshape(pred[0].shape[0], -1),
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
    orientation_index_pearson = [
        pearson_corr(
            orientation_index(pred[0], is_logit=True).squeeze(),
            orientation_index(signal).squeeze(),
        ).item()
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
    orientation_index_pearson_all = pearson_corr(
        torch.cat(
            [orientation_index(pred[0], is_logit=True).squeeze() for pred in preds]
        ),
        torch.cat([orientation_index(signal).squeeze() for signal in signals]),
    ).item()

    print("\nPer-fold results:\n----------------")
    print(
        f"Profile Pearson correlation: {[np.nanmedian(c).item() for c in profile_corr]}"
        f" (n_nan={[np.isnan(c).mean().item() for c in profile_corr]})"
    )
    print(
        f"Profile Jensen-Shannon distance: {[np.nanmedian(j).item() for j in profile_jsd]} "
        f"(n_nan={[np.isnan(j).mean().item() for j in profile_jsd]})"
    )
    print(f"Log counts Pearson correlation: {log_counts_pearson}")
    print(f"Counts Spearman correlation: {counts_spearman}")
    print(f"Orientation index Pearson correlation: {orientation_index_pearson}")

    print("\nGenome-wide results:\n----------------")
    print(f"Profile Pearson correlation: {np.nanmedian(np.concatenate(profile_corr))}")
    print(
        f"Profile Jensen-Shannon distance: {np.nanmedian(np.concatenate(profile_jsd))}"
    )
    print(f"Log counts Pearson correlation: {log_counts_pearson_all}")
    print(f"Counts Spearman correlation: {counts_spearman_all}")
    print(f"Orientation index Pearson correlation: {orientation_index_pearson_all}")

    # Save metrics to JSON
    metrics = {
        "experiment": args.experiment,
        "biosample": exp["biosample"],
        "model_dir": str(model_dir),
        "per_fold": {
            str(fold): {
                "profile_pearson": np.nanmedian(profile_corr[fold]).item(),
                "profile_jsd": np.nanmedian(profile_jsd[fold]).item(),
                "log_counts_pearson": log_counts_pearson[fold],
                "counts_spearman": counts_spearman[fold],
                "orientation_index_pearson": orientation_index_pearson[fold],
            }
            for fold in range(n_folds)
        },
        "genome_wide": {
            "profile_pearson": np.nanmedian(np.concatenate(profile_corr)).item(),
            "profile_jsd": np.nanmedian(np.concatenate(profile_jsd)).item(),
            "log_counts_pearson": log_counts_pearson_all,
            "counts_spearman": counts_spearman_all,
            "orientation_index_pearson": orientation_index_pearson_all,
        },
    }
    metrics_dir = REPO_ROOT / "performance_metrics" / "bpnet"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / f"{model_dir.name}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"\nMetrics saved to {metrics_path}")

    if args.save_output:
        output_dir = REPO_ROOT / "predictions" / "bpnet"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{args.experiment}.npz"
        scaled_preds = {
            f"predict_fold{i}": count_scaled_profile(pred[0], pred[1]).numpy()
            for i, pred in enumerate(preds)
        }
        expts = {f"expt_fold{i}": signal.numpy() for i, signal in enumerate(signals)}
        np.savez_compressed(output_path, **scaled_preds, **expts)
        print(f"\nPredictions saved to {output_path}")


if __name__ == "__main__":
    main()
