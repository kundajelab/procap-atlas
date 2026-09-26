"""Candidate promoter vs. candidate enhancer classification for union peaks.

Classifies each union peak by whether its midpoint falls in an ENCODE SCREEN
Registry V4 cCRE (`data/GRCh38-cCREs.bed.gz`, see
`src/download/download_genome.sh`) labelled PLS (promoter-like signature) or
pELS/dELS (proximal/distal enhancer-like signature). Peaks overlapping
neither, or landing in the same base pair as both a PLS and an ELS cCRE, are
left unclassified (NaN) rather than guessed -- which one wins would be an
invented rule, not a measurement.

The midpoint, not the full peak interval, decides class membership, matching
how peaks are treated everywhere else in this pipeline: `extract_observed
_counts` centers a fixed-width window on each peak's midpoint, and
`fit_bpnet.py`'s peak-vs-negative GC matching is midpoint-based too. Using the
full interval would double-count peaks wide enough to span both a promoter-
and an enhancer-like cCRE.

Peak indices are 0-based positions into `union_peaks.bed.gz`'s row order,
matching the column order `count_correlation.py` writes into
`observed_counts.tsv`/`predicted_counts.tsv` (see its `extract_observed_counts`
docstring) -- so the returned Series can be intersected directly against
`observed.columns` once those are read back in as strings.
"""

import gzip
from pathlib import Path

import numpy as np
import pandas as pd

CANONICAL_CHROMS = [f"chr{i}" for i in list(range(1, 23)) + ["X", "Y"]]

# Verified 2026-09 against the live file `download_genome.sh` fetches
# (https://downloads.wenglab.org/Registry-V4/GRCh38-cCREs.bed): the label
# column is always exactly one of these eight tokens, never comma-joined --
# {PLS: 19,952; pELS: 106,334; dELS: 569,312; CA: 86,861; CA-CTCF: 45,714;
# TF: 40,692; CA-H3K4me3: 27,975; CA-TF: 9,584} over a ~906k-row sample
# (chr1-17). Registry V3 compounded modifiers onto the primary label (e.g.
# "PLS,CTCF-bound"); V4 dropped that in favor of flat, mutually exclusive
# labels, including four new chromatin-accessible-only categories (CA,
# CA-CTCF, CA-TF, CA-H3K4me3) that are neither promoter- nor enhancer-like.
# Exact membership, not substring matching, since the real vocabulary is now
# known rather than assumed.
PROMOTER_LABELS = frozenset({"PLS"})
ENHANCER_LABELS = frozenset({"pELS", "dELS"})


def load_union_peak_midpoints(union_peaks_path: Path) -> pd.DataFrame:
    """chrom/mid for each union peak, in file row order (0-indexed).

    Row order is the join key back to the count matrices -- see module
    docstring -- so this must not sort, dedupe, or otherwise reorder rows.
    """
    df = pd.read_csv(
        union_peaks_path,
        sep="\t",
        header=None,
        usecols=[0, 1, 2],
        names=["chrom", "start", "end"],
        dtype={"chrom": str},
    )
    df["mid"] = (df["start"] + df["end"]) // 2
    return df


def load_ccres(ccre_bed_path: Path, label_column: int = -1) -> pd.DataFrame:
    """chrom/start/end/label from a Registry-V4-style cCRE bed(.gz).

    `label_column` defaults to the last column, matching the Registry V4
    layout (chrom, start, end, rDHS accession, cCRE accession, group label).
    Negative indices are resolved against the file's actual column count
    rather than hardcoded, so a schema with an extra or missing column fails
    loudly (via `usecols`) instead of silently reading the wrong field.
    """
    opener = gzip.open if str(ccre_bed_path).endswith(".gz") else open
    with opener(ccre_bed_path, "rt") as handle:
        n_cols = len(handle.readline().rstrip("\n").split("\t"))
    label_col = label_column if label_column >= 0 else n_cols + label_column
    return pd.read_csv(
        ccre_bed_path,
        sep="\t",
        header=None,
        usecols=[0, 1, 2, label_col],
        names=["chrom", "start", "end", "label"],
        dtype={"chrom": str, "label": str},
    )


def _merge_intervals(starts: np.ndarray, ends: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sort and merge overlapping/adjacent 0-based half-open intervals.

    Same algorithm as `make_union_peaks.py`'s `merge_intervals`, specialized
    to one chromosome and one class label so containment becomes a single
    `searchsorted` per peak instead of an O(peaks x cCREs) scan.
    """
    if len(starts) == 0:
        return starts, ends
    order = np.argsort(starts, kind="stable")
    starts, ends = starts[order], ends[order]
    merged_starts = [starts[0]]
    merged_ends = [ends[0]]
    for s, e in zip(starts[1:], ends[1:]):
        if s <= merged_ends[-1]:
            merged_ends[-1] = max(merged_ends[-1], e)
        else:
            merged_starts.append(s)
            merged_ends.append(e)
    return np.array(merged_starts), np.array(merged_ends)


def _contains(merged_starts: np.ndarray, merged_ends: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Boolean per point: does it fall in `[start, end)` of some merged interval."""
    out = np.zeros(len(points), dtype=bool)
    if len(merged_starts) == 0:
        return out
    idx = np.searchsorted(merged_starts, points, side="right") - 1
    valid = idx >= 0
    out[valid] = points[valid] < merged_ends[idx[valid]]
    return out


def classify_peaks_by_cre(
    union_peaks_path: Path, ccre_bed_path: Path, label_column: int = -1,
) -> pd.Series:
    """"promoter"/"enhancer" per union peak, indexed by 0-based position (as str).

    Indexed as strings rather than ints so it lines up directly with
    `observed.columns`/`predicted.columns`, which come back from
    `pd.read_csv(..., index_col=0)` as strings even though they started out as
    a bare `RangeIndex` (see `load_counts` in `cross_celltype_prediction.py`).
    """
    peaks = load_union_peak_midpoints(union_peaks_path)
    ccres = load_ccres(ccre_bed_path, label_column)
    is_promoter = ccres["label"].isin(PROMOTER_LABELS)
    is_enhancer = ccres["label"].isin(ENHANCER_LABELS)

    result = pd.Series(np.nan, index=peaks.index, dtype=object)
    for chrom, group in peaks.groupby("chrom", sort=False):
        chrom_ccres = ccres[ccres["chrom"] == chrom]
        promoter_iv = _merge_intervals(
            chrom_ccres.loc[is_promoter[chrom_ccres.index], "start"].to_numpy(),
            chrom_ccres.loc[is_promoter[chrom_ccres.index], "end"].to_numpy(),
        )
        enhancer_iv = _merge_intervals(
            chrom_ccres.loc[is_enhancer[chrom_ccres.index], "start"].to_numpy(),
            chrom_ccres.loc[is_enhancer[chrom_ccres.index], "end"].to_numpy(),
        )
        mids = group["mid"].to_numpy()
        in_promoter = _contains(*promoter_iv, mids)
        in_enhancer = _contains(*enhancer_iv, mids)
        labels = np.full(len(group), np.nan, dtype=object)
        labels[in_promoter & ~in_enhancer] = "promoter"
        labels[in_enhancer & ~in_promoter] = "enhancer"
        result.loc[group.index] = labels

    result.index = result.index.astype(str)
    return result
