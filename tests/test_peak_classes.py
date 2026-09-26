"""Tests for ENCODE SCREEN cCRE-based peak classification.

`classify_peaks_by_cre` is pure (bed files in, a Series out) and fully
testable without a reference genome or trained models, unlike the extraction
steps in `count_correlation.py`.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "analysis"))

from _peak_classes import (
    _contains,
    _merge_intervals,
    classify_peaks_by_cre,
    load_ccres,
    load_union_peak_midpoints,
)


def write_bed(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n")
    return path


# --- interval merge/containment ---------------------------------------------


def test_merge_intervals_joins_overlapping_spans():
    starts, ends = _merge_intervals(np.array([100, 150, 500]), np.array([200, 180, 600]))
    assert list(starts) == [100, 500]
    assert list(ends) == [200, 600]


def test_merge_intervals_joins_adjacent_spans_too():
    # Matches make_union_peaks.py's merge_intervals: "overlapping or adjacent."
    starts, ends = _merge_intervals(np.array([100, 200]), np.array([200, 300]))
    assert list(starts) == [100]
    assert list(ends) == [300]


def test_merge_intervals_handles_empty_input():
    starts, ends = _merge_intervals(np.array([]), np.array([]))
    assert len(starts) == 0 and len(ends) == 0


def test_contains_is_half_open():
    merged = (np.array([100]), np.array([200]))
    result = _contains(*merged, np.array([99, 100, 150, 199, 200]))
    assert list(result) == [False, True, True, True, False]


def test_contains_on_empty_intervals_is_all_false():
    result = _contains(np.array([]), np.array([]), np.array([50, 150]))
    assert list(result) == [False, False]


# --- loaders -----------------------------------------------------------------


def test_load_union_peak_midpoints_preserves_row_order(tmp_path):
    path = write_bed(tmp_path / "union.bed", ["chr2\t100\t200", "chr1\t500\t600"])
    df = load_union_peak_midpoints(path)
    assert list(df["chrom"]) == ["chr2", "chr1"]
    assert list(df["mid"]) == [150, 550]


def test_load_ccres_reads_the_last_column_as_label_by_default(tmp_path):
    path = write_bed(tmp_path / "ccres.bed", [
        "chr1\t100\t200\tEH38D1\tEH38E1\tPLS",
        "chr1\t500\t600\tEH38D2\tEH38E2\tCA-CTCF",
    ])
    df = load_ccres(path)
    assert list(df["label"]) == ["PLS", "CA-CTCF"]


def test_load_ccres_accepts_gzip(tmp_path):
    import gzip

    path = tmp_path / "ccres.bed.gz"
    with gzip.open(path, "wt") as handle:
        handle.write("chr1\t100\t200\tEH38D1\tEH38E1\tPLS\n")
    df = load_ccres(path)
    assert list(df["label"]) == ["PLS"]


# --- classification ------------------------------------------------------


def test_classify_peaks_by_cre_end_to_end(tmp_path):
    union = write_bed(tmp_path / "union.bed", [
        "chr1\t100\t200",   # mid 150 -> promoter
        "chr1\t500\t600",   # mid 550 -> enhancer (pELS)
        "chr1\t900\t1000",  # mid 950 -> neither
        "chr1\t1300\t1400", # mid 1350 -> both PLS and dELS -> ambiguous
        "chr2\t50\t150",    # mid 100 -> promoter, different chromosome
    ])
    ccre = write_bed(tmp_path / "ccres.bed", [
        "chr1\t100\t200\tEH38D1\tEH38E1\tPLS",
        "chr1\t520\t580\tEH38D2\tEH38E2\tpELS",
        "chr1\t1300\t1400\tEH38D3\tEH38E3\tPLS",
        "chr1\t1300\t1400\tEH38D4\tEH38E4\tdELS",
        "chr2\t0\t150\tEH38D5\tEH38E5\tPLS",
    ])
    result = classify_peaks_by_cre(union, ccre)
    assert result["0"] == "promoter"
    assert result["1"] == "enhancer"
    assert pd.isna(result["2"])
    assert pd.isna(result["3"])
    assert result["4"] == "promoter"


def test_classify_peaks_by_cre_index_is_stringified_position(tmp_path):
    union = write_bed(tmp_path / "union.bed", ["chr1\t100\t200"])
    ccre = write_bed(tmp_path / "ccres.bed", ["chr1\t100\t200\td\te\tPLS"])
    result = classify_peaks_by_cre(union, ccre)
    assert list(result.index) == ["0"]
    assert all(isinstance(v, str) for v in result.index)
    # Must line up with observed/predicted columns as read back from a TSV
    # (pd.read_csv(..., index_col=0) on a written RangeIndex header), not just
    # look stringlike -- get_indexer is what take_columns actually uses.
    from_tsv_columns = pd.Index(["0"])
    assert (from_tsv_columns.get_indexer(result.index) >= 0).all()


def test_classify_peaks_by_cre_handles_a_chromosome_with_no_ccres(tmp_path):
    union = write_bed(tmp_path / "union.bed", ["chrY\t100\t200"])
    ccre = write_bed(tmp_path / "ccres.bed", ["chr1\t100\t200\td\te\tPLS"])
    result = classify_peaks_by_cre(union, ccre)
    assert pd.isna(result["0"])


def test_pels_and_dels_both_classify_as_enhancer(tmp_path):
    union = write_bed(tmp_path / "union.bed", ["chr1\t100\t200", "chr1\t500\t600"])
    ccre = write_bed(tmp_path / "ccres.bed", [
        "chr1\t100\t200\td\te\tpELS",
        "chr1\t500\t600\td\te\tdELS",
    ])
    result = classify_peaks_by_cre(union, ccre)
    assert result["0"] == "enhancer"
    assert result["1"] == "enhancer"


def test_registry_v4_chromatin_accessible_categories_are_unclassified(tmp_path):
    """Registry V4 added four "chromatin accessible, otherwise unclassified"
    labels (CA, CA-CTCF, CA-TF, CA-H3K4me3) alongside PLS/pELS/dELS -- verified
    2026-09 against the live file at
    https://downloads.wenglab.org/Registry-V4/GRCh38-cCREs.bed. None of them
    is a promoter or an enhancer signature, and matching against exact labels
    (rather than a "PLS"/"ELS" substring) is what keeps them out without
    special-casing each one here.
    """
    union = write_bed(tmp_path / "union.bed", [
        f"chr1\t{i * 1000}\t{i * 1000 + 200}" for i in range(5)
    ])
    ccre = write_bed(tmp_path / "ccres.bed", [
        f"chr1\t{i * 1000}\t{i * 1000 + 200}\td{i}\te{i}\t{label}"
        for i, label in enumerate(["CA", "CA-CTCF", "CA-TF", "CA-H3K4me3", "TF"])
    ])
    result = classify_peaks_by_cre(union, ccre)
    assert result.isna().all(), result.to_dict()
