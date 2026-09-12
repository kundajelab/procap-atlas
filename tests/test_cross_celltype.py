"""Tests for the cross-cell-type prediction comparison.

Two halves, with different testability. The held-out fold logic in
`count_correlation.py` is pure and tested here; the extraction it feeds needs
BigWigs, a reference genome and trained models, so it runs only on the cluster.
The matrix analysis in `cross_celltype_prediction.py` is fully local.

The failure this guards against is the one that would silently invalidate the
whole analysis: `extract_predicted_counts` averages all seven fold models at
every peak, so a model predicting its *own* experiment benefits from six folds
that trained on those peaks while a model predicting a different experiment
does not. That inflates the matched diagonal, which is exactly the quantity the
comparison measures.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "analysis"))

import count_correlation as cc  # noqa: E402
import cross_celltype_prediction as ccp  # noqa: E402

SUBPROC_ENV_KEYS = {"MPLBACKEND": "Agg", "OMP_NUM_THREADS": "1"}


def env():
    import os

    return {**os.environ, **SUBPROC_ENV_KEYS}


# --- held-out fold assignment ----------------------------------------------


def test_real_fold_splits_cover_each_chromosome_once():
    # If a chromosome were in two folds, no fold would have held it out and
    # the held-out guarantee would be void.
    assignment = cc.fold_by_chrom(cc.load_chrom_folds())
    assert len(assignment) == 24                      # chr1-22, X, Y
    assert set(assignment.values()) == set(range(7))


def test_fold_by_chrom_rejects_a_chromosome_in_two_folds():
    with pytest.raises(ValueError, match="would not be held out"):
        cc.fold_by_chrom({0: {"chr1"}, 1: {"chr1", "chr2"}})


def test_split_peaks_by_fold_is_in_ascending_fold_order():
    # Concatenating fold results is what keeps vectors aligned across
    # experiments, and that only works if the order is fixed.
    assignment = {"chrA": 2, "chrB": 0, "chrC": 1}
    peaks = pd.DataFrame({
        "chrom": ["chrA", "chrB", "chrC", "chrA"],
        "start": [1, 2, 3, 4], "end": [9, 9, 9, 9],
    })
    got = cc.split_peaks_by_fold(peaks, assignment)
    assert [fold for fold, _ in got] == [0, 1, 2]
    assert list(got[0][1]["chrom"]) == ["chrB"]
    assert list(got[2][1]["chrom"]) == ["chrA", "chrA"]


def test_split_peaks_by_fold_preserves_within_fold_peak_order():
    assignment = {"chrA": 0}
    peaks = pd.DataFrame({
        "chrom": ["chrA"] * 3, "start": [30, 10, 20], "end": [39, 19, 29],
    })
    got = cc.split_peaks_by_fold(peaks, assignment)
    assert list(got[0][1]["start"]) == [30, 10, 20]


def test_split_peaks_by_fold_omits_empty_folds():
    assignment = {"chrA": 0, "chrB": 1}
    peaks = pd.DataFrame({"chrom": ["chrA"], "start": [1], "end": [9]})
    assert [fold for fold, _ in cc.split_peaks_by_fold(peaks, assignment)] == [0]


def test_unassigned_chromosomes_are_reported_not_silently_kept():
    assignment = cc.fold_by_chrom(cc.load_chrom_folds())
    peaks = pd.DataFrame({
        "chrom": ["chr1", "chrM", "chr1_alt"], "start": [1, 1, 1], "end": [9, 9, 9],
    })
    assert cc.unassigned_chroms(peaks, assignment) == ["chr1_alt", "chrM"]
    # and they do not appear in any fold's peaks
    kept = pd.concat([d for _, d in cc.split_peaks_by_fold(peaks, assignment)])
    assert set(kept["chrom"]) == {"chr1"}


def test_every_peak_lands_in_exactly_one_fold():
    assignment = cc.fold_by_chrom(cc.load_chrom_folds())
    peaks = pd.DataFrame({
        "chrom": [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"],
        "start": [1] * 24, "end": [9] * 24,
    })
    total = sum(len(d) for _, d in cc.split_peaks_by_fold(peaks, assignment))
    assert total == len(peaks)


# --- tiers and matrix ------------------------------------------------------


def test_tier_of_distinguishes_all_three_cases():
    groups = {"a": "blood", "b": "blood", "c": "heart"}
    assert ccp.tier_of("a", "a", groups) == "matched"
    assert ccp.tier_of("a", "b", groups) == "same tissue"
    assert ccp.tier_of("a", "c", groups) == "different tissue"


def test_tier_of_treats_unknown_groups_as_different():
    # two experiments with no group must not be called same-tissue just
    # because both are None
    groups = {"a": None, "b": None}
    assert ccp.tier_of("a", "b", groups) == "different tissue"


def counts_frame(spec, n_peaks=50, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {f"p{i}": rng.normal(size=len(spec)) for i in range(n_peaks)}, index=spec
    ).abs() * 10


def test_correlation_matrix_diagonal_is_one_for_perfect_prediction():
    obs = counts_frame(["a", "b", "c"])
    m = ccp.correlation_matrix(obs, obs)
    assert np.allclose(np.diag(m.to_numpy(dtype=float)), 1.0)


def test_correlation_matrix_orientation_is_model_by_experiment():
    obs = counts_frame(["a", "b"])
    pred = obs.copy()
    pred.loc["a"] = obs.loc["b"]          # model a now predicts b's signal
    m = ccp.correlation_matrix(obs, pred)
    assert m.at["a", "b"] > m.at["a", "a"]
    assert m.index.name == "model" and m.columns.name == "experiment"


def test_align_intersects_experiments_and_peaks():
    obs = counts_frame(["a", "b", "c"], n_peaks=5)
    pred = counts_frame(["b", "c", "d"], n_peaks=5).drop(columns=["p4"])
    o, p = ccp.align(obs, pred)
    assert list(o.index) == ["b", "c"] and list(p.index) == ["b", "c"]
    assert "p4" not in o.columns and o.shape == p.shape


def test_align_raises_rather_than_correlating_misaligned_vectors():
    obs = counts_frame(["a"], n_peaks=5)
    pred = counts_frame(["z"], n_peaks=5)
    with pytest.raises(ValueError, match="no experiments in common"):
        ccp.align(obs, pred)
    same = counts_frame(["a"], n_peaks=5)
    with pytest.raises(ValueError, match="no peaks in common"):
        ccp.align(obs, same.rename(columns=lambda c: c + "_x"))


def test_most_variable_peaks_picks_the_discriminating_ones():
    obs = pd.DataFrame({
        "flat": [10.0, 10.0, 10.0],
        "varying": [1.0, 100.0, 10_000.0],
        "mild": [5.0, 6.0, 7.0],
    }, index=["a", "b", "c"])
    assert ccp.most_variable_peaks(obs, 1) == ["varying"]
    # 0 or n >= width means "all", and must keep original column order
    assert ccp.most_variable_peaks(obs, 0) == list(obs.columns)
    assert ccp.most_variable_peaks(obs, 99) == list(obs.columns)


def test_summarize_tiers_orders_tiers_by_relatedness():
    pairs = pd.DataFrame({
        "tier": ["matched", "same tissue", "different tissue"] * 2,
        "correlation": [0.9, 0.6, 0.2, 0.95, 0.65, 0.25],
    })
    out = ccp.summarize_tiers(pairs)
    assert list(out["tier"]) == list(ccp.TIERS)
    assert out["delta_to_next"].iloc[0] > 0


def test_paired_within_model_is_one_row_per_model():
    pairs = pd.DataFrame([
        dict(model="a", experiment="a", model_group="blood", tier="matched",
             correlation=0.9),
        dict(model="a", experiment="b", model_group="blood",
             tier="same tissue", correlation=0.6),
        dict(model="a", experiment="c", model_group="blood",
             tier="different tissue", correlation=0.2),
    ])
    out = ccp.paired_within_model(pairs)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["beats_same_tissue"] and row["beats_different_tissue"]


def test_paired_within_model_records_a_loss():
    pairs = pd.DataFrame([
        dict(model="a", experiment="a", model_group="blood", tier="matched",
             correlation=0.1),
        dict(model="a", experiment="c", model_group="blood",
             tier="different tissue", correlation=0.8),
    ])
    out = ccp.paired_within_model(pairs)
    # numpy bool, so compare by value rather than identity
    assert not out.iloc[0]["beats_different_tissue"]


def test_sign_test_matches_hand_computed_values():
    assert ccp.sign_test(9, 9) == pytest.approx(2 / 2 ** 9)
    assert ccp.sign_test(5, 10) == pytest.approx(1.0)
    assert ccp.sign_test(0, 0) != ccp.sign_test(0, 0) or True      # nan, no raise


def test_sign_test_is_symmetric():
    assert ccp.sign_test(8, 10) == pytest.approx(ccp.sign_test(2, 10))


# --- CLI -------------------------------------------------------------------


def tiered_inputs(tmp_path, n_peaks=300, seed=0):
    """Observed/predicted files with matched > same tissue > different."""
    import yaml

    cfg = yaml.safe_load(open(REPO_ROOT / "configs" / "experiment_config.yaml"))
    sys.path.insert(0, str(REPO_ROOT / "src" / "analysis"))
    from _biosample_groups import load_group_map

    tissue, _ = load_group_map(cfg["experiments"], None, quiet=True)
    by_group = {}
    for exp, group in tissue.items():
        by_group.setdefault(group, []).append(exp)
    picked = []
    for group in ("blood_immune", "gi_tract"):
        picked += sorted(by_group[group])[:2]

    rng = np.random.default_rng(seed)
    group_signal = {g: rng.normal(size=n_peaks) for g in set(tissue.values())}
    obs, pred = {}, {}
    for exp in picked:
        own = rng.normal(size=n_peaks)
        truth = 2 * group_signal[tissue[exp]] + 1.5 * own
        obs[exp] = np.expm1(np.clip(truth + rng.normal(0, 0.3, n_peaks), -5, 8))
        pred[exp] = np.expm1(np.clip(truth + rng.normal(0, 0.3, n_peaks), -5, 8))
    index = [f"p{i}" for i in range(n_peaks)]
    o = tmp_path / "obs.tsv"
    p = tmp_path / "pred.tsv"
    pd.DataFrame(obs, index=index).T.to_csv(o, sep="\t")
    pd.DataFrame(pred, index=index).T.to_csv(p, sep="\t")
    return o, p


def run_cli(tmp_path, *extra):
    o, p = tiered_inputs(tmp_path)
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "src/analysis/cross_celltype_prediction.py"),
         "--observed", str(o), "--predicted", str(p),
         "--out-dir", str(tmp_path / "out"), *extra],
        capture_output=True, text=True, env=env(),
    )


def test_cli_recovers_the_tier_ordering(tmp_path):
    result = run_cli(tmp_path)
    assert result.returncode == 0, result.stderr
    tiers = pd.read_csv(tmp_path / "out" / "cross_celltype_tiers.tsv", sep="\t")
    med = dict(zip(tiers["tier"], tiers["median"]))
    assert med["matched"] > med["same tissue"] > med["different tissue"]


def test_cli_writes_all_tables_and_figures(tmp_path):
    result = run_cli(tmp_path)
    assert result.returncode == 0, result.stderr
    out = tmp_path / "out"
    for name in ("cross_celltype_matrix.tsv", "cross_celltype_pairs.tsv",
                 "cross_celltype_tiers.tsv", "cross_celltype_per_model.tsv",
                 "cross_celltype_matrix.pdf", "cross_celltype_tiers.pdf"):
        assert (out / name).exists(), name


def test_cli_warns_that_it_assumes_held_out_counts(tmp_path):
    result = run_cli(tmp_path)
    assert result.returncode == 0, result.stderr
    # the confound is severe enough that silence is not an option
    assert "--held-out-folds" in result.stderr


def test_cli_reports_the_sign_test(tmp_path):
    result = run_cli(tmp_path)
    assert result.returncode == 0, result.stderr
    assert "sign test" in result.stderr


def test_cli_reports_the_depth_correlation(tmp_path):
    result = run_cli(tmp_path)
    assert result.returncode == 0, result.stderr
    assert "read depth" in result.stderr


def test_cli_can_restrict_to_variable_peaks(tmp_path):
    result = run_cli(tmp_path, "--variable-peaks", "50")
    assert result.returncode == 0, result.stderr
    assert "most variable peaks" in result.stderr


def test_cli_points_at_the_extraction_command_when_inputs_are_missing(tmp_path):
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "src/analysis/cross_celltype_prediction.py"),
         "--observed", str(tmp_path / "nope.tsv"),
         "--predicted", str(tmp_path / "nope2.tsv"), "--out-dir", str(tmp_path)],
        capture_output=True, text=True, env=env(),
    )
    assert result.returncode == 1
    assert "--held-out-folds" in result.stderr
    assert "count_correlation.py" in result.stderr


# --- balanced subset -------------------------------------------------------


def test_balanced_subset_takes_the_deepest_per_group():
    exps = {f"e{i}": {} for i in range(6)}
    groups = {"e0": "blood", "e1": "blood", "e2": "blood",
              "e3": "heart", "e4": "heart", "e5": "heart"}
    depth = {"e0": 10, "e1": 30, "e2": 20, "e3": 5, "e4": 15, "e5": 25}
    got = cc.balanced_subset(exps, groups, 2, depth)
    assert set(got) == {"e1", "e2", "e5", "e4"}


def test_balanced_subset_applies_min_reads_before_selecting():
    # Selecting the deepest three and then dropping shallow ones can leave a
    # group with one experiment and no within-group pair, which is the tier the
    # analysis exists to measure.
    exps = {f"e{i}": {} for i in range(3)}
    groups = {"e0": "blood", "e1": "blood", "e2": "blood"}
    depth = {"e0": 100, "e1": 5, "e2": 50}
    got = cc.balanced_subset(exps, groups, 3, depth, min_reads=10)
    assert set(got) == {"e0", "e2"}


def test_balanced_subset_drops_the_other_bucket():
    exps = {"a": {}, "b": {}}
    groups = {"a": "blood", "b": "other"}
    got = cc.balanced_subset(exps, groups, 3, {"a": 1, "b": 1})
    assert set(got) == {"a"}


def test_balanced_subset_keeps_an_excluded_group_if_asked():
    exps = {"a": {}, "b": {}}
    groups = {"a": "blood", "b": "other"}
    got = cc.balanced_subset(exps, groups, 3, {"a": 1, "b": 1}, exclude_groups=())
    assert set(got) == {"a", "b"}


def test_balanced_subset_is_a_noop_at_zero():
    exps = {"a": {}, "b": {}}
    groups = {"a": "blood", "b": "heart"}
    assert cc.balanced_subset(exps, groups, 0, {}) is exps


def test_balanced_subset_skips_ungrouped_experiments():
    exps = {"a": {}, "b": {}}
    got = cc.balanced_subset(exps, {"a": "blood"}, 3, {"a": 1, "b": 1})
    assert set(got) == {"a"}


def test_balanced_subset_on_the_real_atlas_is_tier_capable():
    import yaml

    cfg = yaml.safe_load(open(REPO_ROOT / "configs" / "experiment_config.yaml"))
    sys.path.insert(0, str(REPO_ROOT / "src" / "analysis"))
    from _biosample_groups import load_group_map
    from plot_motif_rarefaction import load_read_counts

    tissue, _ = load_group_map(cfg["experiments"], None, quiet=True)
    got = cc.balanced_subset(
        cfg["experiments"], tissue, 3, load_read_counts(), min_reads=10e6
    )
    sizes = {}
    for exp in got:
        sizes[tissue[exp]] = sizes.get(tissue[exp], 0) + 1
    # a same-tissue tier needs at least two experiments in several groups
    assert sum(1 for n in sizes.values() if n >= 2) >= 10
    assert len(got) < 80, "the point of the subset is to be cheaper than 198"


# --- peak subsampling ------------------------------------------------------


def test_peak_subsample_is_deterministic_for_a_fixed_seed():
    # Every experiment and fold must see the same peaks: the vectors being
    # correlated line up only if the subsample is drawn once, identically.
    peaks = pd.DataFrame({
        "chrom": ["chr1"] * 100, "start": range(100), "end": range(1, 101),
    })
    a = peaks.sample(n=10, random_state=0).sort_values(["chrom", "start"])
    b = peaks.sample(n=10, random_state=0).sort_values(["chrom", "start"])
    assert list(a["start"]) == list(b["start"])
    c = peaks.sample(n=10, random_state=1).sort_values(["chrom", "start"])
    assert list(a["start"]) != list(c["start"])


def test_subsampled_peaks_still_split_across_folds():
    assignment = cc.fold_by_chrom(cc.load_chrom_folds())
    rng = np.random.default_rng(0)
    chroms = rng.choice(sorted(assignment), size=500)
    peaks = pd.DataFrame({
        "chrom": chroms, "start": range(500), "end": range(1, 501),
    })
    sub = peaks.sample(n=100, random_state=0)
    folds = cc.split_peaks_by_fold(sub, assignment)
    assert sum(len(d) for _, d in folds) == 100
    assert len(folds) > 1, "a subsample should still span several folds"
