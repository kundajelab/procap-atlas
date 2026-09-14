"""Tests for compare_clusterings.py.

The tool exists to quantify how much MotifCompendium's k_centroids refinement
changes a partition, so the thing that must be right is the agreement metric:
it has to be invariant to cluster relabelling (ids are not stable across
builds) and it has to distinguish "renamed" from "actually moved".
"""

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "bpnet" / "motifcompendium"))
import compare_clusterings as cc  # noqa: E402


def mapping(assignments, path, posneg="pos"):
    """assignments = {(experiment, local_name): cluster_id}"""
    rows = []
    for (exp, local), cluster in assignments.items():
        rows.append({
            "experiment": exp,
            "local_motif_name": local,
            "compendium_motif_name": f"{posneg}_patterns.pattern_{cluster}",
        })
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
    return path


def test_relabelling_alone_counts_as_perfect_agreement(tmp_path):
    """Cluster ids are arbitrary across builds; only membership matters."""
    a = {("E1", "p0"): 0, ("E1", "p1"): 0, ("E2", "p0"): 1}
    b = {("E1", "p0"): 7, ("E1", "p1"): 7, ("E2", "p0"): 3}
    sa = cc.load_assignments(mapping(a, tmp_path / "a.tsv"))
    sb = cc.load_assignments(mapping(b, tmp_path / "b.tsv"))
    out = cc.compare_pair(sa, sb, "a", "b")
    assert out["frac_same_clustermates"] == 1.0
    assert out["n_patterns_moved"] == 0
    assert out["ari"] == 1.0


def test_a_single_moved_pattern_is_detected(tmp_path):
    a = {("E1", "p0"): 0, ("E1", "p1"): 0, ("E2", "p0"): 1, ("E2", "p1"): 1}
    b = {("E1", "p0"): 0, ("E1", "p1"): 0, ("E2", "p0"): 1, ("E2", "p1"): 0}
    sa = cc.load_assignments(mapping(a, tmp_path / "a.tsv"))
    sb = cc.load_assignments(mapping(b, tmp_path / "b.tsv"))
    out = cc.compare_pair(sa, sb, "a", "b")
    assert out["frac_same_clustermates"] < 1.0
    # the moved pattern and every pattern whose clustermates changed count
    assert out["n_patterns_moved"] == 4
    assert out["ari"] < 1.0


def test_a_split_is_reported_directionally(tmp_path):
    """One cluster in A becoming two in B is a split, not a merge."""
    a = {("E1", f"p{i}"): 0 for i in range(4)}
    b = {("E1", "p0"): 0, ("E1", "p1"): 0, ("E1", "p2"): 1, ("E1", "p3"): 1}
    sa = cc.load_assignments(mapping(a, tmp_path / "a.tsv"))
    sb = cc.load_assignments(mapping(b, tmp_path / "b.tsv"))
    out = cc.compare_pair(sa, sb, "a", "b")
    assert out["a_clusters_split_in_b"] == 1
    assert out["b_clusters_split_in_a"] == 0


def test_pos_and_neg_patterns_sharing_a_cluster_id_are_not_merged(tmp_path):
    """cluster_final is only unique within a posneg stratum; collapsing them
    would silently merge two distinct clusters."""
    p = tmp_path / "mixed.tsv"
    pd.DataFrame([
        {"experiment": "E1", "local_motif_name": "pos_patterns.pattern_0",
         "compendium_motif_name": "pos_patterns.pattern_5"},
        {"experiment": "E1", "local_motif_name": "neg_patterns.pattern_0",
         "compendium_motif_name": "neg_patterns.pattern_5"},
    ]).to_csv(p, sep="\t", index=False)
    s = cc.load_assignments(p)
    assert s.nunique() == 2, "pos:5 and neg:5 must stay distinct"


def test_only_shared_patterns_are_compared(tmp_path):
    """Builds can differ in experiment set; the comparison must intersect."""
    a = {("E1", "p0"): 0, ("E1", "p1"): 0, ("E2", "p0"): 1}
    b = {("E1", "p0"): 0, ("E1", "p1"): 0}
    sa = cc.load_assignments(mapping(a, tmp_path / "a.tsv"))
    sb = cc.load_assignments(mapping(b, tmp_path / "b.tsv"))
    out = cc.compare_pair(sa, sb, "a", "b")
    assert out["n_shared_patterns"] == 2
    assert out["n_clusters_a"] == 1, "counted on the shared patterns only"


def test_duplicate_pattern_rows_are_rejected(tmp_path):
    p = tmp_path / "dup.tsv"
    pd.DataFrame([
        {"experiment": "E1", "local_motif_name": "p0",
         "compendium_motif_name": "pos_patterns.pattern_0"},
        {"experiment": "E1", "local_motif_name": "p0",
         "compendium_motif_name": "pos_patterns.pattern_1"},
    ]).to_csv(p, sep="\t", index=False)
    with pytest.raises(ValueError, match="duplicate"):
        cc.load_assignments(p)


def test_missing_columns_are_named(tmp_path):
    p = tmp_path / "bad.tsv"
    pd.DataFrame([{"experiment": "E1"}]).to_csv(p, sep="\t", index=False)
    with pytest.raises(ValueError, match="local_motif_name"):
        cc.load_assignments(p)


def test_cli_reports_every_pair_and_writes_a_table(tmp_path):
    a = {("E1", f"p{i}"): i // 2 for i in range(6)}
    b = dict(a)
    b[("E1", "p5")] = 0
    c = {k: 0 for k in a}
    paths = {
        "leiden": mapping(a, tmp_path / "a.tsv"),
        "capped": mapping(b, tmp_path / "b.tsv"),
        "uncapped": mapping(c, tmp_path / "c.tsv"),
    }
    out_tsv = tmp_path / "cmp.tsv"
    result = subprocess.run(
        [sys.executable,
         str(REPO_ROOT / "src/bpnet/motifcompendium/compare_clusterings.py"),
         "--head", "count", "--out-tsv", str(out_tsv)]
        + [f"{k}={v}" for k, v in paths.items()],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    table = pd.read_csv(out_tsv, sep="\t")
    assert len(table) == 3, "3 builds -> 3 pairs"
    assert set(table["a"]) | set(table["b"]) == {"leiden", "capped", "uncapped"}
    assert "frac_same_clustermates" in result.stdout


def test_cli_rejects_a_single_build(tmp_path):
    a = mapping({("E1", "p0"): 0}, tmp_path / "a.tsv")
    result = subprocess.run(
        [sys.executable,
         str(REPO_ROOT / "src/bpnet/motifcompendium/compare_clusterings.py"),
         f"only={a}"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "at least two builds" in result.stderr
