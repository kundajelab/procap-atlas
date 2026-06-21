"""
Benchmark ENCSR220XSM/ENCSR261KBX track swap models.

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

    model = "ENCSR220XSMtracks_ENCSR261KBXpeaks"
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

    model_dir = REPO_ROOT / "models" / "bpnet" / model

    with open(CHROM_SPLITS_PATH) as f:
        data = yaml.safe_load(f)
    chrom_splits = {int(k): v for k, v in data["folds"].items()}
    n_folds = len(chrom_splits)

    model_paths = [model_dir / f"{model}.fold{fold}.torch" for fold in range(n_folds)]
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


# ENCSR261KBXtracks_ENCSR220XSMpeaks benchmarked on XSM peaks/tracks
# Per-fold results:
# ----------------
# Profile Pearson correlation: [0.4282265901565552, 0.42213183641433716, 0.4183763563632965, 0.42572301626205444, 0.41333407163619995, 0.4422926902770996, 0.4339592158794403] (n_nan=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
# Profile Jensen-Shannon distance: [0.38296306133270264, 0.37837737798690796, 0.3890886902809143, 0.38460275530815125, 0.39469921588897705, 0.3805086016654968, 0.3715485632419586] (n_nan=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
# Log counts Pearson correlation: [0.7635771036148071, 0.7589091658592224, 0.7573578953742981, 0.74910569190979, 0.7436556220054626, 0.7544538378715515, 0.6828385591506958]
# Counts Spearman correlation: [0.7700853943824768, 0.7638015747070312, 0.7627411484718323, 0.7565259337425232, 0.7496020793914795, 0.7639095783233643, 0.7036643028259277]
# Genome-wide results:
# ----------------
# Profile Pearson correlation: 0.4266435503959656
# Profile Jensen-Shannon distance: 0.3829578161239624
# Log counts Pearson correlation: 0.7386600971221924
# Counts Spearman correlation: 0.748038113117218

# ENCSR261KBXtracks_ENCSR220XSMpeaks benchmarked on KBX peaks/tracks
# Per-fold results:
# ----------------
# Profile Pearson correlation: [0.520988941192627, 0.5312654376029968, 0.5223436951637268, 0.539281964302063, 0.5147772431373596, 0.5471007823944092, 0.5394028425216675] (n_nan=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
# Profile Jensen-Shannon distance: [0.3328753709793091, 0.32546696066856384, 0.3357846736907959, 0.3329768180847168, 0.3436451256275177, 0.3290937542915344, 0.3298569321632385] (n_nan=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
# Log counts Pearson correlation: [0.7144080400466919, 0.7262429594993591, 0.7200379371643066, 0.7152741551399231, 0.704298734664917, 0.7092173099517822, 0.6286136507987976]
# Counts Spearman correlation: [0.7239834070205688, 0.7173106670379639, 0.7272437214851379, 0.7244733572006226, 0.7111468315124512, 0.7073975801467896, 0.6420050859451294]
# Genome-wide results:
# ----------------
# Profile Pearson correlation: 0.530531644821167
# Profile Jensen-Shannon distance: 0.3331270217895508
# Log counts Pearson correlation: 0.6920838356018066
# Counts Spearman correlation: 0.6955287456512451

# ENCSR220XSMtracks_ENCSR261KBXpeaks benchmarked on KBX peaks
# Per-fold results:
# ----------------
# Profile Pearson correlation: [0.4473073482513428, 0.4626322388648987, 0.42725712060928345, 0.4355049133300781, 0.446454256772995, 0.4541642665863037, 0.46331140398979187] (n_nan=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
# Profile Jensen-Shannon distance: [0.36051708459854126, 0.3511938154697418, 0.36803939938545227, 0.35772714018821716, 0.3625379800796509, 0.35916635394096375, 0.3519197106361389] (n_nan=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
# Log counts Pearson correlation: [0.6652905344963074, 0.7062722444534302, 0.6670526266098022, 0.6955056190490723, 0.6769235730171204, 0.6702952980995178, 0.5855225324630737]
# Counts Spearman correlation: [0.6854742765426636, 0.6977952718734741, 0.6807917356491089, 0.7091274261474609, 0.6879355311393738, 0.6678723692893982, 0.6054258942604065]
# Genome-wide results:
# ----------------
# Profile Pearson correlation: 0.44831573963165283
# Profile Jensen-Shannon distance: 0.35904210805892944
# Log counts Pearson correlation: 0.6489524245262146
# Counts Spearman correlation: 0.6572836637496948

# ENCSR220XSMtracks_ENCSR261KBXpeaks benchmarked on XSM peaks
# Per-fold results:
# ----------------
# Profile Pearson correlation: [0.5147372484207153, 0.5218952298164368, 0.5016355514526367, 0.5122641921043396, 0.5102304816246033, 0.5233112573623657, 0.5203038454055786] (n_nan=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
# Profile Jensen-Shannon distance: [0.3397788405418396, 0.33258119225502014, 0.34238845109939575, 0.3390665054321289, 0.34328100085258484, 0.3383597135543823, 0.32593074440956116] (n_nan=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
# Log counts Pearson correlation: [0.72304368019104, 0.745879590511322, 0.7061336636543274, 0.7342138886451721, 0.730441689491272, 0.7251958250999451, 0.6467440128326416]
# Counts Spearman correlation: [0.7406080365180969, 0.7580682635307312, 0.7274236679077148, 0.7496532797813416, 0.747827410697937, 0.7444878220558167, 0.6763299107551575]
# Genome-wide results:
# ----------------
# Profile Pearson correlation: 0.5150856971740723
# Profile Jensen-Shannon distance: 0.33752012252807617
# Log counts Pearson correlation: 0.7028396129608154
# Counts Spearman correlation: 0.7248238325119019
