"""
Benchmark ENCSR220XSM on ENCSR261KBX tracks/peaks.

Loads trained models for each fold, predicts on the held-out test chromosomes,
and reports profile and count prediction metrics.

Use --model-dir to specify the path directly if trained with custom negative ratios.

Usage:
    python src/bpnet/benchmark/benchmark_bpnet_K562_peak_test.py
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
CONFIG_PATH = REPO_ROOT / "configs" / "experiment_config.yaml"
CHROM_SPLITS_PATH = REPO_ROOT / "configs" / "chrom_splits.yaml"
FASTA = str(REPO_ROOT / "data" / "hg38.fa")
BLACKLIST = str(REPO_ROOT / "data" / "hg38.blacklist.bed.gz")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-b", "--batch-size", type=int, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    # Load experiment config
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    model_name = "ENCSR261KBX"
    experiment = "ENCSR220XSM"
    experiments = config["experiments"]
    if experiment not in experiments:
        print(f"Error: {experiment} not found in config", file=sys.stderr)
        sys.exit(1)

    exp = experiments[experiment]
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

    model_dir = REPO_ROOT / "models" / "bpnet" / model_name

    with open(CHROM_SPLITS_PATH) as f:
        data = yaml.safe_load(f)
    chrom_splits = {int(k): v for k, v in data["folds"].items()}
    n_folds = len(chrom_splits)

    model_paths = [
        model_dir / f"{model_name}.fold{fold}.torch" for fold in range(n_folds)
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

    print(f"Experiment: {experiment} ({exp['biosample']})")
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

    print("\nGenome-wide results:\n----------------")
    print(f"Profile Pearson correlation: {np.nanmedian(np.concatenate(profile_corr))}")
    print(
        f"Profile Jensen-Shannon distance: {np.nanmedian(np.concatenate(profile_jsd))}"
    )
    print(f"Log counts Pearson correlation: {log_counts_pearson_all}")
    print(f"Counts Spearman correlation: {counts_spearman_all}")

    # Save metrics to JSON
    metrics = {
        "experiment": experiment,
        "biosample": exp["biosample"],
        "model_dir": str(model_dir),
        "per_fold": {
            str(fold): {
                "profile_pearson": np.nanmedian(profile_corr[fold]).item(),
                "profile_jsd": np.nanmedian(profile_jsd[fold]).item(),
                "log_counts_pearson": log_counts_pearson[fold],
                "counts_spearman": counts_spearman[fold],
            }
            for fold in range(n_folds)
        },
        "genome_wide": {
            "profile_pearson": np.nanmedian(np.concatenate(profile_corr)).item(),
            "profile_jsd": np.nanmedian(np.concatenate(profile_jsd)).item(),
            "log_counts_pearson": log_counts_pearson_all,
            "counts_spearman": counts_spearman_all,
        },
    }
    metrics_dir = REPO_ROOT / "performance_metrics" / "bpnet"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / f"{model_dir.name}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"\nMetrics saved to {metrics_path}")


if __name__ == "__main__":
    main()


# XSM model on KBX data
# Per-fold results:
# ----------------
# Profile Pearson correlation: [0.45464378595352173, 0.4578242599964142, 0.44409507513046265, 0.44443193078041077, 0.44513338804244995, 0.4777260720729828, 0.4725148379802704] (n_nan=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
# Profile Jensen-Shannon distance: [0.3506211042404175, 0.353196918964386, 0.3627933859825134, 0.3601803779602051, 0.36240914463996887, 0.3516295850276947, 0.3509453535079956] (n_nan=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
# Log counts Pearson correlation: [0.6975951790809631, 0.7304368615150452, 0.7222180366516113, 0.6959409713745117, 0.6804443001747131, 0.6934981942176819, 0.6191789507865906]
# Counts Spearman correlation: [0.7126717567443848, 0.721736490726471, 0.7263811826705933, 0.7170442342758179, 0.6875653266906738, 0.6932771801948547, 0.6368410587310791]
# Genome-wide results:
# ----------------
# Profile Pearson correlation: 0.45727628469467163
# Profile Jensen-Shannon distance: 0.35596099495887756
# Log counts Pearson correlation: 0.6839534640312195
# Counts Spearman correlation: 0.6922390460968018

# KBX model on XSM data
# Per-fold results:
# ----------------
# Profile Pearson correlation: [0.42537444829940796, 0.4225257635116577, 0.4089583158493042, 0.41842785477638245, 0.4058317244052887, 0.4277712106704712, 0.4178050756454468] (n_nan=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
# Profile Jensen-Shannon distance: [0.38009190559387207, 0.3753698766231537, 0.3898959159851074, 0.3836666941642761, 0.3917638659477234, 0.3821680247783661, 0.37783169746398926] (n_nan=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
# Log counts Pearson correlation: [0.7389988899230957, 0.7391630411148071, 0.7401285171508789, 0.7182667255401611, 0.7009422183036804, 0.7397207021713257, 0.6599981784820557]
# Counts Spearman correlation: [0.7558423280715942, 0.7552858591079712, 0.7544920444488525, 0.7370699048042297, 0.728195071220398, 0.7532427310943604, 0.6863449215888977]
# Genome-wide results:
# ----------------
# Profile Pearson correlation: 0.4184465706348419
# Profile Jensen-Shannon distance: 0.3833162486553192
# Log counts Pearson correlation: 0.7094368934631348
# Counts Spearman correlation: 0.7294664978981018
