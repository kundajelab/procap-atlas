"""Tests for src/bpnet/motifcompendium/group_cluster_families.py.

Weighted toward the three mistakes actually made while working this out, all
of which are silent and all of which bias toward "no redundancy":

  * leaving self-similarities in `within` inflates it, widening every gap so
    real duplicates read as separable;
  * a singleton reporting within=1.0 makes it the tightest cluster in the
    build;
  * verifying the guarantee with `nanmax` instead of `nanmin` reports the
    most separable pair of representatives, which any input satisfies.

The real measured numbers are used as fixtures, so a regression shows up as a
disagreement with the profile-head build rather than with an invented case.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent / "src" / "bpnet" / "motifcompendium"),
)

import group_cluster_families as gcf  # noqa: E402

# cluster_final 21, 8, 48 on the profile head: the TATA family.
TATA_WITHIN = np.array([0.985, 0.975, 0.966])
TATA_CROSS = np.array([
    [1.000, 0.961, 0.967],
    [0.961, 1.000, 0.932],
    [0.967, 0.932, 1.000],
])
# cluster_final 4, 15, 17: CA-Inr as picked by eye, which the numbers show is
# a chain rather than a clique -- 15 vs 17 is separable at +0.155.
CAINR_WITHIN = np.array([0.979, 0.984, 0.981])
CAINR_CROSS = np.array([
    [1.000, 0.922, 0.926],
    [0.922, 1.000, 0.826],
    [0.926, 0.826, 1.000],
])


def blocks(sizes, within_value, cross_value):
    """Similarity for len(sizes) clusters: constant within, constant across."""
    codes = np.concatenate([[c] * n for c, n in enumerate(sizes)])
    n = len(codes)
    S = np.full((n, n), float(cross_value))
    S[codes[:, None] == codes[None, :]] = float(within_value)
    np.fill_diagonal(S, 1.0)
    return S, codes


# --- cluster_cross_matrix ----------------------------------------------------


def test_within_excludes_self_similarity():
    """Three members at pairwise 0.8 give 0.8, not (6*0.8 + 3*1.0)/9 = 0.867."""
    S, codes = blocks([3], within_value=0.8, cross_value=0.0)
    _, within = gcf.cluster_cross_matrix(S, codes, 1)
    assert within[0] == pytest.approx(0.8)


def test_singleton_within_is_nan_not_one():
    S, codes = blocks([1, 3], within_value=0.9, cross_value=0.5)
    _, within = gcf.cluster_cross_matrix(S, codes, 2)
    assert np.isnan(within[0])
    assert within[1] == pytest.approx(0.9)


def test_cross_is_symmetric_and_is_the_pair_mean():
    S, codes = blocks([2, 3], within_value=0.9, cross_value=0.42)
    cross, _ = gcf.cluster_cross_matrix(S, codes, 2)
    assert cross[0, 1] == pytest.approx(0.42)
    assert cross[1, 0] == pytest.approx(0.42)


def test_unequal_sizes_divide_by_the_pair_count():
    """1 and 4 members give 4 cross pairs: divide by 4, not 5 or 16."""
    codes = np.array([0, 1, 1, 1, 1])
    S = np.eye(5)
    S[0, 1:] = [0.2, 0.4, 0.6, 0.8]
    S[1:, 0] = [0.2, 0.4, 0.6, 0.8]
    cross, _ = gcf.cluster_cross_matrix(S, codes, 2)
    assert cross[0, 1] == pytest.approx(0.5)


# --- gap_matrix --------------------------------------------------------------


def test_gap_uses_the_tighter_cluster_not_the_mean():
    """A loose cluster must not excuse a small gap to a tight one."""
    gap = gcf.gap_matrix(np.array([[1.0, 0.90], [0.90, 1.0]]),
                         np.array([0.99, 0.92]))
    assert gap[0, 1] == pytest.approx(0.02)  # 0.92 - 0.90, not 0.955 - 0.90


def test_gap_diagonal_is_masked():
    gap = gcf.gap_matrix(np.array([[1.0, 0.5], [0.5, 1.0]]),
                         np.array([0.9, 0.9]))
    assert np.isnan(gap[0, 0]) and np.isnan(gap[1, 1])


def test_gap_is_nan_against_a_singleton():
    gap = gcf.gap_matrix(np.array([[1.0, 0.95], [0.95, 1.0]]),
                         np.array([np.nan, 0.97]))
    assert np.isnan(gap[0, 1])


def test_observed_tata_gaps():
    gap = gcf.gap_matrix(TATA_CROSS, TATA_WITHIN)
    assert gap[0, 1] == pytest.approx(0.014, abs=1e-9)   # 21 vs 8
    assert gap[0, 2] == pytest.approx(-0.001, abs=1e-9)  # 21 vs 48
    assert gap[1, 2] == pytest.approx(0.034, abs=1e-9)   # 8 vs 48


def test_observed_cainr_gaps_include_a_separable_pair():
    gap = gcf.gap_matrix(CAINR_CROSS, CAINR_WITHIN)
    assert gap[0, 1] == pytest.approx(0.057, abs=1e-9)
    assert gap[0, 2] == pytest.approx(0.053, abs=1e-9)
    assert gap[1, 2] == pytest.approx(0.155, abs=1e-9)


# --- greedy_representatives --------------------------------------------------


def test_tata_collapses_at_the_chosen_threshold():
    gap = gcf.gap_matrix(TATA_CROSS, TATA_WITHIN)
    kept, rep = gcf.greedy_representatives(gap, [0, 1, 2], 0.05)
    assert len(kept) == 1
    assert list(rep) == [0, 0, 0]


def test_cainr_is_preserved_at_the_same_threshold():
    """The negative control: the one family the numbers call separable must
    survive the threshold that collapses TATA and AP-1."""
    gap = gcf.gap_matrix(CAINR_CROSS, CAINR_WITHIN)
    kept, _ = gcf.greedy_representatives(gap, [0, 1, 2], 0.05)
    assert len(kept) == 3


def test_absorbed_clusters_do_not_absorb_others():
    """A~B and B~C but not A~C: greedy keeps A and C, unlike single-linkage
    which percolated (largest component grew to 137 clusters at t=0.05)."""
    gap = np.array([
        [np.nan, 0.01, 0.40],
        [0.01, np.nan, 0.02],
        [0.40, 0.02, np.nan],
    ])
    kept, rep = gcf.greedy_representatives(gap, [0, 1, 2], 0.05)
    assert kept == [0, 2]
    assert rep[1] == 0  # absorbed by A, and cannot then pull in C


def test_absorbed_goes_to_its_tightest_representative():
    """Not the first kept cluster found, so the mapping does not depend on
    the iteration order of the kept list."""
    gap = np.array([
        [np.nan, 0.60, 0.04],
        [0.60, np.nan, 0.01],
        [0.04, 0.01, np.nan],
    ])
    kept, rep = gcf.greedy_representatives(gap, [0, 1, 2], 0.05)
    assert kept == [0, 1]
    assert rep[2] == 1  # gap 0.01 to B beats 0.04 to A


def test_order_decides_the_representative():
    gap = gcf.gap_matrix(TATA_CROSS, TATA_WITHIN)
    _, rep_a = gcf.greedy_representatives(gap, [1, 0, 2], 0.05)
    _, rep_b = gcf.greedy_representatives(gap, [2, 0, 1], 0.05)
    assert set(rep_a) == {1}
    assert set(rep_b) == {2}


def test_a_tighter_threshold_keeps_more():
    gap = gcf.gap_matrix(TATA_CROSS, TATA_WITHIN)
    # only 21 vs 48 (-0.001) is inside 0.01; 8 stays separate
    kept, _ = gcf.greedy_representatives(gap, [0, 1, 2], 0.01)
    assert len(kept) == 2


def test_singletons_are_never_absorbed():
    """Their gaps are all nan, so nanargmin must not be reached."""
    gap = gcf.gap_matrix(np.full((2, 2), 0.99), np.array([np.nan, np.nan]))
    kept, rep = gcf.greedy_representatives(gap, [0, 1], 0.05)
    assert kept == [0, 1]
    assert list(rep) == [0, 1]


# --- the guarantee -----------------------------------------------------------


def test_tightest_kept_pair_is_the_minimum_not_the_maximum():
    """The bug that made the check vacuous: nanmax returns the most separable
    pair, which is satisfied by any input."""
    gap = np.array([
        [np.nan, 0.06, 0.97],
        [0.06, np.nan, 0.80],
        [0.97, 0.80, np.nan],
    ])
    assert gcf.tightest_kept_pair(gap, [0, 1, 2]) == pytest.approx(0.06)


def test_tightest_kept_pair_is_nan_below_two_representatives():
    assert np.isnan(gcf.tightest_kept_pair(np.array([[np.nan]]), [0]))


@pytest.mark.parametrize("threshold", [0.0, 0.01, 0.02, 0.03, 0.05, 0.07])
def test_the_guarantee_holds_for_any_threshold(threshold):
    """No two representatives may be within the threshold, whatever it is."""
    rng = np.random.default_rng(0)
    k = 40
    cross = rng.uniform(0.5, 0.99, size=(k, k))
    cross = (cross + cross.T) / 2
    within = rng.uniform(0.95, 0.99, size=k)
    gap = gcf.gap_matrix(cross, within)
    kept, rep = gcf.greedy_representatives(gap, np.arange(k), threshold)
    tightest = gcf.tightest_kept_pair(gap, kept)
    assert np.isnan(tightest) or tightest > threshold
    # every cluster resolves to a representative, and representatives to self
    assert (rep >= 0).all()
    for i in kept:
        assert rep[i] == i


def test_end_to_end_on_planted_families():
    codes = np.concatenate([[0] * 4, [1] * 4, [2] * 4])
    n = len(codes)
    S = np.full((n, n), 0.30)
    for c in range(3):
        idx = np.flatnonzero(codes == c)
        S[np.ix_(idx, idx)] = 0.97
    a, b = np.flatnonzero(codes == 0), np.flatnonzero(codes == 1)
    S[np.ix_(a, b)] = 0.96
    S[np.ix_(b, a)] = 0.96
    np.fill_diagonal(S, 1.0)

    cross, within = gcf.cluster_cross_matrix(S, codes, 3)
    gap = gcf.gap_matrix(cross, within)
    assert gap[0, 1] == pytest.approx(0.01)
    assert gap[0, 2] == pytest.approx(0.67)
    kept, rep = gcf.greedy_representatives(gap, [0, 1, 2], 0.05)
    assert kept == [0, 2]
    assert rep[1] == 0
    assert gcf.tightest_kept_pair(gap, kept) > 0.05


# --- output table ------------------------------------------------------------


def test_every_cluster_appears_including_ungrouped_ones():
    """The mapping is non-destructive: an ineligible cluster is still a row,
    mapped to itself, so downstream joins never drop motifs."""
    import pandas as pd

    data = {
        "ids": np.array([10, 20, 30]),
        "all_ids": np.array([10, 20, 30, 40, 50]),
        "prevalence": pd.Series({10: 9, 20: 8, 30: 7, 40: 1, 50: 1}),
        "sizes": pd.Series({10: 5, 20: 4, 30: 3, 40: 1, 50: 1}),
    }
    gap = gcf.gap_matrix(TATA_CROSS, TATA_WITHIN)
    kept, rep = gcf.greedy_representatives(gap, [0, 1, 2], 0.05)
    table = gcf.build_table(data, rep, kept, TATA_WITHIN, gap)

    assert sorted(table.cluster_final) == [10, 20, 30, 40, 50]
    ungrouped = table[~table.grouped]
    assert sorted(ungrouped.cluster_final) == [40, 50]
    assert (ungrouped.family_rep == ungrouped.cluster_final).all()
    assert ungrouped.is_representative.all()
    # the three grouped clusters collapsed onto cluster 10
    grouped = table[table.grouped]
    assert set(grouped.family_rep) == {10}
    assert int(grouped.is_representative.sum()) == 1
    assert np.isnan(table.loc[table.cluster_final == 10, "gap_to_rep"].iloc[0])
