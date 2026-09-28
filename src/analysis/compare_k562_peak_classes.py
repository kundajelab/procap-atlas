"""Compare promoter vs. enhancer peak class breakdown for K562 experiments.

Quantifies how sequencing depth and negative-set choice affect GATA motif
recovery: classifies per-experiment peaks (and optionally DNase negatives)
against ENCODE SCREEN Registry V4 cCREs.

Run on the cluster where peak BED files exist:

    uv run python src/analysis/compare_k562_peak_classes.py

Reads from paths in configs/experiment_config.yaml.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from _peak_classes import (
    ENHANCER_LABELS,
    PROMOTER_LABELS,
    _contains,
    _merge_intervals,
    load_ccres,
    load_union_peak_midpoints,
)

ROOT = Path(__file__).resolve().parents[2]
CCRE_PATH = ROOT / "data" / "GRCh38-cCREs.bed.gz"

EXPERIMENTS = {
    "ENCSR220XSM": {
        "label": "XSM (44M reads, GC neg)",
        "peaks": ROOT / "data/processed/peaks/ENCSR220XSM_K562.bed.gz",
    },
    "ENCSR261KBX": {
        "label": "KBX (19M reads, GC neg)",
        "peaks": ROOT / "data/processed/peaks/ENCSR261KBX_K562.bed.gz",
    },
}

GC_NEG_PATH = ROOT / "data/processed/negatives/ENCSR261KBX_K562_gc_negatives.bed.gz"
DNASE_NEG_PATH = ROOT / "data/processed/negatives/ENCSR261KBX_K562_dnase.bed.gz"


def classify_bed(bed_path: Path, ccres: pd.DataFrame) -> dict:
    """Classify peaks/regions in a BED file as promoter, enhancer, or other."""
    peaks = load_union_peak_midpoints(bed_path)
    is_promoter_ccre = ccres["label"].isin(PROMOTER_LABELS)
    is_enhancer_ccre = ccres["label"].isin(ENHANCER_LABELS)

    n_promoter = 0
    n_enhancer = 0
    n_both = 0
    n_neither = 0

    for chrom, group in peaks.groupby("chrom", sort=False):
        chrom_ccres = ccres[ccres["chrom"] == chrom]
        promoter_iv = _merge_intervals(
            chrom_ccres.loc[is_promoter_ccre[chrom_ccres.index], "start"].to_numpy(),
            chrom_ccres.loc[is_promoter_ccre[chrom_ccres.index], "end"].to_numpy(),
        )
        enhancer_iv = _merge_intervals(
            chrom_ccres.loc[is_enhancer_ccre[chrom_ccres.index], "start"].to_numpy(),
            chrom_ccres.loc[is_enhancer_ccre[chrom_ccres.index], "end"].to_numpy(),
        )
        mids = group["mid"].to_numpy()
        in_p = _contains(*promoter_iv, mids)
        in_e = _contains(*enhancer_iv, mids)
        n_promoter += int(np.sum(in_p & ~in_e))
        n_enhancer += int(np.sum(in_e & ~in_p))
        n_both += int(np.sum(in_p & in_e))
        n_neither += int(np.sum(~in_p & ~in_e))

    total = n_promoter + n_enhancer + n_both + n_neither
    return {
        "total": total,
        "promoter": n_promoter,
        "enhancer": n_enhancer,
        "ambiguous": n_both,
        "other": n_neither,
        "pct_promoter": 100 * n_promoter / total if total else 0,
        "pct_enhancer": 100 * n_enhancer / total if total else 0,
    }


def main():
    ccres = load_ccres(CCRE_PATH)
    rows = []

    for exp_id, info in EXPERIMENTS.items():
        if not info["peaks"].exists():
            print(f"SKIP {info['label']}: {info['peaks']} not found")
            continue
        result = classify_bed(info["peaks"], ccres)
        result["experiment"] = info["label"]
        rows.append(result)

    for neg_path, neg_label in [
        (GC_NEG_PATH, "KBX GC negatives"),
        (DNASE_NEG_PATH, "KBX DNase negatives"),
    ]:
        if neg_path.exists():
            result = classify_bed(neg_path, ccres)
            result["experiment"] = neg_label
            rows.append(result)
        else:
            print(f"NOTE: {neg_path} not found — skipping {neg_label}")

    if not rows:
        print("No peak files found. Run this on the cluster.")
        return

    df = pd.DataFrame(rows).set_index("experiment")
    col_order = ["total", "promoter", "enhancer", "ambiguous", "other", "pct_promoter", "pct_enhancer"]
    df = df[[c for c in col_order if c in df.columns]]

    out_dir = ROOT / "figures" / "gata_investigation"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "peak_class_breakdown.csv"
    df.to_csv(csv_path)
    print(f"Saved to {csv_path}")

    print("\n=== Peak class breakdown (ENCODE SCREEN V4 cCREs) ===\n")
    print(df.to_string())

    if len(rows) >= 2:
        print("\n=== Key comparisons ===\n")
        xsm = next((r for r in rows if r["experiment"] == "XSM (44M reads, GC neg)"), None)
        kbx = next((r for r in rows if r["experiment"] == "KBX (19M reads, GC neg)"), None)
        if xsm and kbx:
            enh_ratio = xsm["enhancer"] / kbx["enhancer"] if kbx["enhancer"] else float("inf")
            pro_ratio = xsm["promoter"] / kbx["promoter"] if kbx["promoter"] else float("inf")
            print(f"XSM/KBX enhancer peaks: {xsm['enhancer']:,} / {kbx['enhancer']:,} = {enh_ratio:.1f}x")
            print(f"XSM/KBX promoter peaks: {xsm['promoter']:,} / {kbx['promoter']:,} = {pro_ratio:.1f}x")
            print(f"XSM/KBX total peaks:    {xsm['total']:,} / {kbx['total']:,} = {xsm['total']/kbx['total']:.1f}x")
            if enh_ratio > pro_ratio:
                print(f"\nEnhancer peaks scale {enh_ratio/pro_ratio:.1f}x faster than promoter peaks with depth")

        gc_neg = next((r for r in rows if r["experiment"] == "KBX GC negatives"), None)
        dnase = next((r for r in rows if r["experiment"] == "KBX DNase negatives"), None)
        if dnase:
            print(f"\nDNase negatives: {dnase['pct_enhancer']:.1f}% enhancer, {dnase['pct_promoter']:.1f}% promoter")
        if gc_neg:
            print(f"GC negatives:    {gc_neg['pct_enhancer']:.1f}% enhancer, {gc_neg['pct_promoter']:.1f}% promoter")
        if dnase and gc_neg:
            enh_fold = dnase["pct_enhancer"] / gc_neg["pct_enhancer"] if gc_neg["pct_enhancer"] else float("inf")
            print(f"\nDHS negatives are {enh_fold:.1f}x enriched for enhancer cCREs vs GC negatives")
            print("→ All DHS negatives are accessible chromatin, confounding GATA's chromatin-opening role")


if __name__ == "__main__":
    main()
