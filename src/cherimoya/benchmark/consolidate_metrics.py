#!/usr/bin/env python3
"""Consolidate per-experiment Cherimoya benchmark JSON files into a single TSV.

Metrics are averaged across the 7 per-fold values rather than taken from the
genome_wide field, to match the approach used for the BPNet TSV.
"""

import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
METRICS_DIR = REPO_ROOT / "performance_metrics" / "cherimoya"
N_READS_PATH = REPO_ROOT / "configs" / "n_reads.txt"
OUT_PATH = METRICS_DIR / "procap-atlas_performance_metrics.tsv"

n_reads = pd.read_csv(N_READS_PATH, sep="\t")

rows = []
for path in sorted(METRICS_DIR.glob("*.json")):
    with open(path) as f:
        data = json.load(f)
    folds = list(data["per_fold"].values())
    rows.append({
        "experiment": data["experiment"],
        "biosample": data["biosample"],
        "profile_pearson":    sum(f["profile_pearson"]    for f in folds) / len(folds),
        "profile_jsd":        sum(f["profile_jsd"]        for f in folds) / len(folds),
        "log_counts_pearson": sum(f["log_counts_pearson"] for f in folds) / len(folds),
        "counts_spearman":    sum(f["counts_spearman"]    for f in folds) / len(folds),
    })

df = pd.DataFrame(rows)
df = n_reads[["experiment", "biosample", "pl_reads", "mn_reads", "total_reads"]].merge(
    df.drop(columns="biosample"), on="experiment", how="inner"
)
df.to_csv(OUT_PATH, sep="\t", index=False)
print(f"Wrote {len(df)} experiments to {OUT_PATH}")
