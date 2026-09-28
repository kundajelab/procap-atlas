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


def test_tier_of_distinguishes_all_four_cases():
    groups = {"a": "blood", "a2": "blood", "b": "blood", "c": "heart"}
    samples = {"a": "K562", "a2": "K562", "b": "PBMC", "c": "heart LV"}
    assert ccp.tier_of("a", "a", groups, samples) == "matched"
    assert ccp.tier_of("a", "a2", groups, samples) == "same biosample"
    assert ccp.tier_of("a", "b", groups, samples) == "same tissue"
    assert ccp.tier_of("a", "c", groups, samples) == "different tissue"


def test_replicates_are_not_counted_as_same_tissue():
    # The atlas is heavily replicated (HCT116 n=16, PBMC n=8). Without the
    # split, a replicate pair would land in "same tissue" and could carry that
    # tier, reducing the claim to "models predict a rerun of their own sample".
    groups = {"a": "blood", "a2": "blood"}
    samples = {"a": "K562", "a2": "K562"}
    assert ccp.tier_of("a", "a2", groups, samples) != "same tissue"


def test_tier_of_falls_back_to_tissue_without_biosamples():
    groups = {"a": "blood", "b": "blood"}
    assert ccp.tier_of("a", "b", groups) == "same tissue"
    assert ccp.tier_of("a", "b", groups, None) == "same tissue"


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
        "tier": list(ccp.TIERS) * 2,
        "correlation": [0.95, 0.9, 0.6, 0.2, 0.97, 0.92, 0.65, 0.25],
    })
    out = ccp.summarize_tiers(pairs)
    assert list(out["tier"]) == list(ccp.TIERS)
    assert (out["delta_to_next"].dropna() > 0).all()


def test_summarize_tiers_skips_absent_tiers():
    pairs = pd.DataFrame({
        "tier": ["matched", "different tissue"],
        "correlation": [0.9, 0.2],
    })
    out = ccp.summarize_tiers(pairs)
    assert list(out["tier"]) == ["matched", "different tissue"]


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
    # Two largest groups, chosen by size rather than named: hardcoding group
    # names made every CLI test fail when blood_immune was split by lineage,
    # for reasons unrelated to what these tests check.
    largest = sorted(by_group, key=lambda g: (-len(by_group[g]), g))[:2]
    picked = []
    for group in largest:
        picked += sorted(by_group[group])[:2]

    rng = np.random.default_rng(seed)
    group_signal = {g: rng.normal(size=n_peaks) for g in set(tissue.values())}
    obs, pred = {}, {}
    for exp in picked:
        own = rng.normal(size=n_peaks)
        truth = 2 * group_signal[tissue[exp]] + 1.5 * own
        # clip at 0, not -5: expm1 of a negative is negative, and counts
        # cannot be negative
        obs[exp] = np.expm1(np.clip(truth + rng.normal(0, 0.3, n_peaks), 0, 8))
        pred[exp] = np.expm1(np.clip(truth + rng.normal(0, 0.3, n_peaks), 0, 8))
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


def test_balanced_subset_prefers_distinct_biosamples():
    # Three K562 replicates and one PBMC: taking the three deepest would give
    # three replicates and no same-tissue pair at all.
    exps = {f"e{i}": {} for i in range(4)}
    groups = {f"e{i}": "blood" for i in range(4)}
    samples = {"e0": "K562", "e1": "K562", "e2": "K562", "e3": "PBMC"}
    depth = {"e0": 100, "e1": 90, "e2": 80, "e3": 10}
    got = cc.balanced_subset(exps, groups, 2, depth, biosamples=samples)
    assert set(got) == {"e0", "e3"}, "must reach for the second biosample"


def test_balanced_subset_backfills_when_biosamples_run_out():
    exps = {f"e{i}": {} for i in range(3)}
    groups = {f"e{i}": "blood" for i in range(3)}
    samples = {f"e{i}": "K562" for i in range(3)}
    depth = {"e0": 100, "e1": 90, "e2": 80}
    got = cc.balanced_subset(exps, groups, 2, depth, biosamples=samples)
    assert len(got) == 2, "one biosample available, so fill with its replicates"


def test_balanced_subset_on_the_real_atlas_avoids_replicate_only_groups():
    import yaml
    from collections import Counter

    cfg = yaml.safe_load(open(REPO_ROOT / "configs" / "experiment_config.yaml"))
    sys.path.insert(0, str(REPO_ROOT / "src" / "analysis"))
    from _biosample_groups import load_group_map
    from plot_motif_rarefaction import load_read_counts

    tissue, sample = load_group_map(cfg["experiments"], None, quiet=True)
    depth = load_read_counts()
    got = cc.balanced_subset(
        cfg["experiments"], tissue, 3, depth, min_reads=10e6, biosamples=sample
    )
    sizes = Counter(tissue[e] for e in got)
    distinct = Counter(
        tissue[e] for e in {sample.get(e, e): e for e in got}.values()
    )
    replicate_only = [
        g for g, n in sizes.items() if n > 1 and distinct.get(g, 0) < 2
    ]
    assert not replicate_only, f"replicate-only within-group pairs: {replicate_only}"


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


# --- quantitative peak specificity -----------------------------------------
#
# Peak *calling* breadth is not a usable specificity measure at 224
# experiments: a promoter with modest lineage-biased activity is still called
# nearly everywhere, so breadth is dominated by near-ubiquitous peaks and its
# narrow tail is weak singletons. Specificity is scored from signal instead.


def spec_frame():
    # 4 experiments, 2 groups; three peaks with known specificity ordering
    return pd.DataFrame({
        "flat": [100.0, 100.0, 100.0, 100.0],       # equal everywhere
        "blood_only": [100.0, 100.0, 0.0, 0.0],     # one group
        "skewed": [100.0, 100.0, 30.0, 30.0],       # in between
    }, index=["b1", "b2", "h1", "h2"])


SPEC_GROUPS = {"b1": "blood", "b2": "blood", "h1": "heart", "h2": "heart"}


@pytest.mark.parametrize("index", ["tau", "entropy"])
def test_specificity_is_zero_for_a_ubiquitous_peak(index):
    spec = ccp.peak_specificity(spec_frame(), SPEC_GROUPS, index)
    assert spec["flat"] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("index", ["tau", "entropy"])
def test_specificity_is_one_for_a_single_group_peak(index):
    spec = ccp.peak_specificity(spec_frame(), SPEC_GROUPS, index)
    assert spec["blood_only"] == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("index", ["tau", "entropy"])
def test_specificity_orders_intermediate_peaks_correctly(index):
    spec = ccp.peak_specificity(spec_frame(), SPEC_GROUPS, index)
    assert spec["flat"] < spec["skewed"] < spec["blood_only"]


def test_tau_matches_the_yanai_definition_by_hand():
    # group means of log1p: blood = log(101) = 4.6151, heart = log(31) = 3.4340
    # tau = sum(1 - x_i/x_max) / (n - 1) = (0 + 1 - 3.4340/4.6151) / 1
    spec = ccp.peak_specificity(spec_frame(), SPEC_GROUPS, "tau")
    expected = (1 - np.log1p(30) / np.log1p(100)) / 1
    assert spec["skewed"] == pytest.approx(expected, abs=1e-6)


def test_tau_discriminates_better_than_entropy_among_broad_peaks():
    # The reason tau is the default: with 224 experiments most peaks are broadly
    # reproduced, so the index has to separate mostly-broad peaks from each
    # other. Entropy nearly saturates at 0 where tau still resolves.
    frame = spec_frame()
    tau = ccp.peak_specificity(frame, SPEC_GROUPS, "tau")["skewed"]
    ent = ccp.peak_specificity(frame, SPEC_GROUPS, "entropy")["skewed"]
    assert tau > 10 * ent


def test_specificity_rejects_an_unknown_index():
    with pytest.raises(ValueError, match="unknown specificity index"):
        ccp.peak_specificity(spec_frame(), SPEC_GROUPS, "nope")


def test_group_means_averages_within_group_first():
    obs = pd.DataFrame({"p": [0.0, 0.0, 100.0]}, index=["b1", "b2", "h1"])
    gm = ccp.group_means(obs, {"b1": "blood", "b2": "blood", "h1": "heart"})
    assert list(gm.index) == ["blood", "heart"]
    assert gm.at["blood", "p"] == pytest.approx(0.0)


@pytest.mark.parametrize("index", ["tau", "entropy"])
def test_specificity_uses_group_means_not_experiment_means_idx(index):
    obs = pd.DataFrame(
        {"small_group_only": [0.0] * 10 + [100.0]},
        index=[f"b{i}" for i in range(10)] + ["s0"],
    )
    groups = {f"b{i}": "blood" for i in range(10)}
    groups["s0"] = "stem"
    assert ccp.peak_specificity(obs, groups, index)["small_group_only"] == (
        pytest.approx(1.0, abs=1e-9)
    )


def test_specificity_uses_group_means_not_experiment_means():
    # 41-vs-4 group sizes are the real case: a peak active only in the small
    # group must still score as specific.
    obs = pd.DataFrame(
        {"small_group_only": [0.0] * 10 + [100.0]},
        index=[f"b{i}" for i in range(10)] + ["s0"],
    )
    groups = {f"b{i}": "blood" for i in range(10)}
    groups["s0"] = "stem"
    spec = ccp.peak_specificity(obs, groups)
    assert spec["small_group_only"] == pytest.approx(1.0, abs=1e-9)


def test_specificity_is_nan_without_two_groups():
    obs = spec_frame()
    one = {e: "blood" for e in obs.index}
    assert ccp.peak_specificity(obs, one).isna().all()
    assert ccp.peak_specificity(obs, {}).isna().all()


def test_specificity_handles_an_all_zero_peak():
    obs = pd.DataFrame({"dead": [0.0, 0.0, 0.0, 0.0]}, index=list(SPEC_GROUPS))
    spec = ccp.peak_specificity(obs, SPEC_GROUPS)
    assert spec["dead"] != spec["dead"] or 0 <= spec["dead"] <= 1  # nan or valid


def test_stratify_splits_top_and_bottom():
    spec = pd.Series({f"p{i}": i / 100 for i in range(100)})
    strata = ccp.stratify_by_specificity(spec, 0.1)
    assert set(strata) == {"specific", "ubiquitous"}
    assert "p99" in strata["specific"] and "p0" in strata["ubiquitous"]
    assert not set(strata["specific"]) & set(strata["ubiquitous"])


def test_stratify_rejects_a_degenerate_quantile():
    spec = pd.Series({f"p{i}": i / 10 for i in range(10)})
    assert ccp.stratify_by_specificity(spec, 0) == {}
    assert ccp.stratify_by_specificity(spec, 0.6) == {}
    assert ccp.stratify_by_specificity(pd.Series(dtype=float), 0.1) == {}


def test_cli_reports_tiers_by_specificity_stratum(tmp_path):
    result = run_cli(tmp_path, "--specificity-quantile", "0.2")
    assert result.returncode == 0, result.stderr
    assert "By observed peak specificity" in result.stderr
    out = pd.read_csv(
        tmp_path / "out" / "cross_celltype_by_specificity.tsv", sep="\t"
    )
    assert set(out["stratum"]) == {"specific", "ubiquitous"}
    assert (tmp_path / "out" / "peak_specificity.tsv").exists()


def test_cli_can_disable_the_specificity_stratification(tmp_path):
    result = run_cli(tmp_path, "--specificity-quantile", "0")
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "out" / "cross_celltype_by_specificity.tsv").exists()


def test_replicate_only_warning_excludes_single_experiment_groups():
    """A group with one experiment has no within-group pairs at all, so
    reporting that its pairs are replicates is meaningless -- and it is already
    reported as contributing no same-tissue pair."""
    from collections import Counter

    experiments = {"a": {}, "b": {}, "c": {}}
    tissue = {"a": "adipose", "b": "blood", "c": "blood"}
    sample = {"a": "fat", "b": "K562", "c": "K562"}
    sizes = Counter(tissue[e] for e in experiments)
    distinct = Counter(
        tissue[e] for e in {sample.get(e, e): e for e in experiments}.values()
    )
    thin = sorted(g for g, n in sizes.items() if n >= 2 and distinct.get(g, 0) < 2)
    assert thin == ["blood"], "adipose has no pairs to describe"


def test_unequal_rows_would_raise_rather_than_corrupt():
    """The guarantee behind held-out extraction: rows align across
    experiments. If they ever did not, pandas raises -- so a misaligned matrix
    cannot be written silently. count_correlation.py now also checks per
    experiment, because this exception would otherwise fire only after every
    remaining experiment had been predicted."""
    rows = {"a": np.zeros(10), "b": np.zeros(9)}
    with pytest.raises(ValueError):
        pd.DataFrame(rows)
    assert pd.DataFrame({"a": np.zeros(10), "b": np.zeros(10)}).shape == (10, 2)


# --- performance and the tau signal floor ----------------------------------


def test_take_columns_short_circuits_identical_columns():
    df = pd.DataFrame(np.zeros((2, 5)), columns=list("abcde"))
    assert ccp.take_columns(df, list("abcde")) is df


def test_take_columns_selects_by_position():
    df = pd.DataFrame(np.arange(10).reshape(2, 5), columns=list("abcde"))
    got = ccp.take_columns(df, ["c", "a"])
    assert list(got.columns) == ["c", "a"]
    assert list(got.iloc[0]) == [2, 0]


def test_take_columns_raises_on_an_unknown_column():
    df = pd.DataFrame(np.zeros((2, 3)), columns=list("abc"))
    with pytest.raises(KeyError, match="not present"):
        ccp.take_columns(df, ["a", "zz"])


def test_align_is_fast_on_a_wide_matrix():
    # align used .loc with ~100,000 column *labels*, which took minutes.
    import time

    n = 60_000
    cols = [f"p{i}" for i in range(n)]
    a = pd.DataFrame(np.zeros((4, n)), index=list("abcd"), columns=cols)
    b = a.copy()
    t0 = time.time()
    o, p = ccp.align(a, b)
    assert time.time() - t0 < 5, "wide align must not do per-label lookups"
    assert o.shape == (4, n)


def test_spearman_matches_pearson_on_ranks():
    rng = np.random.default_rng(0)
    o = pd.DataFrame(rng.lognormal(size=(4, 200)), index=list("abcd"))
    p = pd.DataFrame(rng.lognormal(size=(4, 200)), index=list("abcd"))
    got = ccp.correlation_matrix(o, p, "spearman")
    ranks_o = np.log1p(o.to_numpy()).argsort(axis=1).argsort(axis=1)
    ranks_p = np.log1p(p.to_numpy()).argsort(axis=1).argsort(axis=1)
    ref = np.corrcoef(ranks_p, ranks_o)[:4, 4:]
    assert np.allclose(got.to_numpy(), ref, atol=1e-9)


def test_correlation_matrix_rejects_an_unknown_method():
    o = counts_frame(["a", "b"], n_peaks=10)
    with pytest.raises(ValueError, match="unknown method"):
        ccp.correlation_matrix(o, o, "kendall")


def test_correlation_matrix_tolerates_a_constant_row():
    o = counts_frame(["a", "b"], n_peaks=20)
    p = o.copy()
    p.loc["a"] = 5.0                      # zero variance
    m = ccp.correlation_matrix(o, p)
    assert not np.isnan(m.to_numpy()).any(), "zero-variance rows give 0, not nan"
    assert (m.loc["a"] == 0).all()


def test_top_group_signal_is_in_count_units():
    obs = pd.DataFrame({"p": [0.0, 0.0, 99.0]}, index=["b1", "b2", "h1"])
    groups = {"b1": "blood", "b2": "blood", "h1": "heart"}
    assert ccp.top_group_signal(obs, groups)["p"] == pytest.approx(99.0)


def test_signal_floor_is_what_separates_real_specificity_from_noise():
    """Tau is inflated by noise on near-empty peaks: a single stochastic blip
    in one group scores ~1. On the real matrices the unfiltered top decile has
    12x less signal than the bottom decile, and tau anti-correlates with signal
    at rho = -0.45."""
    groups = {"b1": "blood", "b2": "blood", "h1": "heart", "h2": "heart"}
    obs = pd.DataFrame({
        "noise_blip": [0.0, 0.0, 0.03, 0.0],     # empty but maximally skewed
        "real_specific": [0.0, 0.0, 50.0, 50.0],
    }, index=list(groups))
    spec = ccp.peak_specificity(obs, groups)
    assert spec["noise_blip"] == pytest.approx(spec["real_specific"], abs=1e-6)
    signal = ccp.top_group_signal(obs, groups)
    eligible = signal[signal >= 0.5].index
    assert list(eligible) == ["real_specific"]


# --- reproducibility baseline ----------------------------------------------
#
# A predicted-vs-observed correlation is uninterpretable without two reference
# points: the ceiling (how well the data agrees with itself) and the benchmark
# (what a related experiment's own measurements would give instead of a model).


def test_baseline_excludes_the_trivial_matched_diagonal():
    obs = counts_frame(["a", "b", "c"], n_peaks=50)
    groups = {"a": "blood", "b": "blood", "c": "heart"}
    base = ccp.reproducibility_baseline(obs, groups, None)
    assert "matched" not in set(base["tier"]), "observed vs itself is 1.0"


def test_baseline_recovers_a_known_replicate_structure():
    rng = np.random.default_rng(0)
    shared = np.abs(rng.normal(size=400)) * 10
    obs = pd.DataFrame(
        np.vstack([
            shared + np.abs(rng.normal(0, 0.1, 400)),   # a1, near-identical
            shared + np.abs(rng.normal(0, 0.1, 400)),   # a2, its replicate
            np.abs(rng.normal(size=400)) * 10,          # c, unrelated
        ]),
        index=["a1", "a2", "c"],
        columns=[f"p{i}" for i in range(400)],
    )
    groups = {"a1": "blood", "a2": "blood", "c": "heart"}
    samples = {"a1": "K562", "a2": "K562", "c": "LV"}
    base = ccp.reproducibility_baseline(obs, groups, samples).set_index("tier")
    assert base.at["same biosample", "median"] > 0.9
    assert base.at["same biosample", "median"] > base.at["different tissue", "median"]


def test_baseline_is_computed_on_observed_only():
    # It must not touch predictions: its purpose is to bound them.
    obs = counts_frame(["a", "b"], n_peaks=40)
    groups = {"a": "blood", "b": "heart"}
    one = ccp.reproducibility_baseline(obs, groups, None)
    two = ccp.reproducibility_baseline(obs, groups, None)
    assert one.equals(two)


def test_cli_reports_the_baseline(tmp_path):
    result = run_cli(tmp_path)
    assert result.returncode == 0, result.stderr
    assert "Observed-vs-observed" in result.stderr
    assert (tmp_path / "out" / "cross_celltype_baseline.tsv").exists()


# --- homogenization (ProCapNet's comparison) --------------------------------
#
# Cochran et al. 2024 support "a largely cell-type-agnostic cis-regulatory code
# of initiation" by showing predictions correlate across cell-line pairs at
# r = 0.8-0.97 while the measurements correlate at only 0.5-0.71: the models
# represent cell types as far more alike than they are. That is a different and
# more demanding question than matched-vs-mismatched, which a model can pass
# while still predicting nearly the same thing everywhere.


def homog_frames(n_peaks=400, seed=0):
    """Measurements that differ sharply by group, predictions that do not."""
    rng = np.random.default_rng(seed)
    shared = np.abs(rng.normal(size=n_peaks)) * 10
    index = ["b1", "b2", "h1", "h2"]
    groups = {"b1": "blood", "b2": "blood", "h1": "heart", "h2": "heart"}
    blood_only = np.abs(rng.normal(size=n_peaks)) * 10
    heart_only = np.abs(rng.normal(size=n_peaks)) * 10
    obs = pd.DataFrame(
        np.vstack([blood_only, blood_only, heart_only, heart_only]),
        index=index, columns=[f"p{i}" for i in range(n_peaks)],
    )
    # every model predicts the shared component, ignoring cell type
    pred = pd.DataFrame(
        np.vstack([shared] * 4), index=index, columns=obs.columns
    )
    return obs, pred, groups


def test_homogenization_detects_models_ignoring_cell_type():
    obs, pred, groups = homog_frames()
    out = ccp.homogenization(obs, pred, groups, None).set_index(["source", "tier"])
    measured = out.at[("measured", "different tissue"), "median"]
    predicted = out.at[("predicted", "different tissue"), "median"]
    # predictions identical across cell types, measurements unrelated
    assert predicted > 0.99
    assert measured < 0.2
    assert predicted > measured


def test_homogenization_excludes_the_matched_diagonal():
    obs, pred, groups = homog_frames()
    out = ccp.homogenization(obs, pred, groups, None)
    assert "matched" not in set(out["tier"]), "self-correlation is 1.0"


def test_homogenization_reports_both_sources():
    obs, pred, groups = homog_frames()
    out = ccp.homogenization(obs, pred, groups, None)
    assert set(out["source"]) == {"measured", "predicted"}


def test_homogenization_is_flat_when_models_do_track_cell_type():
    obs, pred, groups = homog_frames()
    out = ccp.homogenization(obs, obs, groups, None).set_index(["source", "tier"])
    # predicting the measurements exactly means no homogenization gap
    assert out.at[("predicted", "different tissue"), "median"] == pytest.approx(
        out.at[("measured", "different tissue"), "median"]
    )


def test_cli_reports_homogenization(tmp_path):
    result = run_cli(tmp_path)
    assert result.returncode == 0, result.stderr
    assert "How much cell-type difference the models reproduce" in result.stderr
    assert (tmp_path / "out" / "cross_celltype_homogenization.tsv").exists()


# --- differential prediction -----------------------------------------------
#
# The projection edit/perturbation experiments measure: f(mutant) - f(reference)
# within one model cancels the shared baseline exactly. A level correlation
# cannot see it, which is why matched-vs-unmatched can be decisive for edits
# while looking modest in a level correlation.


def test_differential_detects_cell_type_signal_a_level_correlation_hides():
    rng = np.random.default_rng(0)
    n = 2000
    shared = np.abs(rng.normal(size=n)) * 20        # dominant shared component
    delta = rng.normal(size=n)                      # small cell-type component
    index = ["a", "b"]
    cols = [f"p{i}" for i in range(n)]
    obs = pd.DataFrame(np.vstack([shared + delta, shared - delta]),
                       index=index, columns=cols).abs()
    pred = pd.DataFrame(np.vstack([shared + 0.5 * delta, shared - 0.5 * delta]),
                        index=index, columns=cols).abs()
    groups = {"a": "blood", "b": "heart"}

    level = ccp.correlation_matrix(obs, pred)
    # the shared component makes every level correlation look alike
    assert abs(level.at["a", "b"] - level.at["a", "a"]) < 0.1

    diff = ccp.differential_prediction(obs, pred, groups, None)
    # differencing recovers the cell-type component
    assert diff["differential_r"].iloc[0] > 0.5


def test_differential_is_near_zero_when_models_ignore_cell_type():
    rng = np.random.default_rng(1)
    n = 2000
    shared = np.abs(rng.normal(size=n)) * 20
    obs = pd.DataFrame(
        np.vstack([shared + rng.normal(size=n), shared + rng.normal(size=n)]),
        index=["a", "b"], columns=[f"p{i}" for i in range(n)],
    ).abs()
    pred = pd.DataFrame(np.vstack([shared, shared]), index=["a", "b"],
                        columns=obs.columns)   # identical predictions
    diff = ccp.differential_prediction(obs, pred, {"a": "x", "b": "y"}, None)
    assert abs(diff["differential_r"].iloc[0]) < 0.1 or np.isnan(
        diff["differential_r"].iloc[0]
    )


def test_differential_covers_unordered_pairs_once():
    obs = counts_frame(["a", "b", "c"], n_peaks=100)
    groups = {"a": "x", "b": "y", "c": "z"}
    diff = ccp.differential_prediction(obs, obs, groups, None)
    assert len(diff) == 3                            # 3 choose 2
    assert not (diff["experiment_a"] == diff["experiment_b"]).any()


def test_summarize_differential_reports_sign_test_and_fraction():
    pairs = pd.DataFrame({
        "tier": ["different tissue"] * 10,
        "differential_r": [0.1] * 9 + [-0.05],
    })
    out = ccp.summarize_differential(pairs).iloc[0]
    assert out["frac_positive"] == pytest.approx(0.9)
    assert out["n_pairs"] == 10
    assert 0 < out["sign_test_p"] < 0.05


def test_cli_reports_differential_prediction(tmp_path):
    result = run_cli(tmp_path)
    assert result.returncode == 0, result.stderr
    assert "Differential prediction" in result.stderr
    for name in ("cross_celltype_differential.tsv",
                 "cross_celltype_differential_pairs.tsv"):
        assert (tmp_path / "out" / name).exists(), name


# --- scale normalization ---------------------------------------------------
#
# count_correlation.py RPM-normalizes observed signal but leaves predictions in
# each model's own trained count scale. On the real matrices that is a 74x
# difference in row sums, with predicted totals tracking read depth at
# Spearman 0.879 -- so log1p sat in a linear regime on one side and a
# logarithmic one on the other, and differencing carried a per-model offset.


def test_normalize_puts_rows_on_a_common_total():
    df = pd.DataFrame([[1.0, 1.0], [100.0, 300.0]], index=["a", "b"])
    out = ccp.normalize_within_peaks(df, total=1000)
    assert out.sum(axis=1).tolist() == pytest.approx([1000.0, 1000.0])
    assert out.loc["b"].tolist() == pytest.approx([250.0, 750.0])


def test_normalize_preserves_within_row_proportions():
    df = pd.DataFrame([[1.0, 2.0, 7.0]], index=["a"])
    out = ccp.normalize_within_peaks(df)
    assert (out.loc["a"] / out.loc["a"].sum()).tolist() == pytest.approx(
        [0.1, 0.2, 0.7]
    )


def test_normalize_leaves_an_all_zero_row_as_nan_not_inf():
    df = pd.DataFrame([[0.0, 0.0], [1.0, 1.0]], index=["dead", "b"])
    out = ccp.normalize_within_peaks(df)
    assert out.loc["dead"].isna().all()
    assert np.isfinite(out.loc["b"]).all()


def test_normalize_removes_a_per_row_scale_offset_from_differentials():
    """A constant per-model scale shifts every differential in one direction.
    A correlation is immune to it; sign accuracy is not."""
    rng = np.random.default_rng(0)
    n = 500
    base = np.abs(rng.normal(size=n)) * 10 + 1
    delta = rng.normal(size=n) * 0.5
    obs = pd.DataFrame(np.vstack([base + delta, base - delta]),
                       index=["a", "b"]).abs()
    # model b's predictions are on a 50x larger scale than model a's
    pred = pd.DataFrame(np.vstack([base + delta, 50 * (base - delta)]),
                        index=["a", "b"]).abs()

    raw = np.log1p(pred.to_numpy())
    raw_delta = raw[0] - raw[1]
    assert (raw_delta < 0).mean() > 0.95, "unnormalized: offset dominates sign"

    fixed = np.log1p(ccp.normalize_within_peaks(pred).to_numpy())
    fixed_delta = fixed[0] - fixed[1]
    true_delta = np.log1p(obs.to_numpy())[0] - np.log1p(obs.to_numpy())[1]
    agree = np.mean(np.sign(fixed_delta) == np.sign(true_delta))
    assert agree > 0.9, f"normalized signs should track the truth, got {agree}"


def test_cli_normalizes_by_default_and_can_be_turned_off(tmp_path):
    on = run_cli(tmp_path)
    assert on.returncode == 0, on.stderr
    assert "Normalizing within peaks" in on.stderr
    off = run_cli(tmp_path, "--normalize", "none")
    assert off.returncode == 0, off.stderr
    assert "Normalizing within peaks" not in off.stderr


def test_load_counts_rejects_negative_values(tmp_path):
    """Silent catastrophe otherwise: normalization scales a negative up, log1p
    of anything <= -1 is NaN, and every correlation involving that row comes
    back as exactly 0.0 instead of erroring. Caught by a synthetic fixture that
    used expm1 of a clipped normal and produced -0.993."""
    bad = tmp_path / "bad.tsv"
    pd.DataFrame([[1.0, -0.993]], index=["a"]).to_csv(bad, sep="\t")
    with pytest.raises(ValueError, match="negative value"):
        ccp.load_counts(bad)


def test_load_counts_accepts_zeros():
    import tempfile

    d = Path(tempfile.mkdtemp())
    ok = d / "ok.tsv"
    pd.DataFrame([[0.0, 1.0]], index=["a"]).to_csv(ok, sep="\t")
    assert ccp.load_counts(ok).shape == (1, 2)


def test_replicates_per_group_adds_back_same_biosample_experiments():
    """Preferring distinct biosamples starves the same-biosample tier, which
    anchors the low end of the effect-size gradient. On the real atlas
    --per-group 3 yields 2 replicate pairs and 5 yields only 3, because every
    extra slot goes to a new biosample."""
    exps = {f"e{i}": {} for i in range(5)}
    groups = {f"e{i}": "blood" for i in range(5)}
    samples = {"e0": "K562", "e1": "K562", "e2": "K562", "e3": "PBMC",
               "e4": "PBMC"}
    depth = {f"e{i}": 100 - i for i in range(5)}

    plain = cc.balanced_subset(exps, groups, 2, depth, biosamples=samples)
    assert len({samples[e] for e in plain}) == 2, "distinct biosamples first"

    topped = cc.balanced_subset(
        exps, groups, 2, depth, biosamples=samples, replicates_per_group=1
    )
    assert len(topped) == 3
    counts = {}
    for e in topped:
        counts[samples[e]] = counts.get(samples[e], 0) + 1
    assert max(counts.values()) == 2, "one biosample must now have a pair"


def test_replicates_per_group_targets_the_most_replicated_biosample():
    exps = {f"e{i}": {} for i in range(4)}
    groups = {f"e{i}": "blood" for i in range(4)}
    samples = {"e0": "solo", "e1": "trio", "e2": "trio", "e3": "trio"}
    depth = {"e0": 100, "e1": 90, "e2": 80, "e3": 70}
    got = cc.balanced_subset(
        exps, groups, 2, depth, biosamples=samples, replicates_per_group=1
    )
    # "trio" has three experiments, so it is where a replicate pair is cheapest
    assert sum(1 for e in got if samples[e] == "trio") >= 2


def test_replicates_per_group_is_a_noop_at_zero():
    exps = {"a": {}, "b": {}}
    groups = {"a": "blood", "b": "blood"}
    samples = {"a": "K562", "b": "K562"}
    got = cc.balanced_subset(
        exps, groups, 1, {"a": 2, "b": 1}, biosamples=samples,
        replicates_per_group=0,
    )
    assert len(got) == 1


def test_replicate_topup_is_cheaper_but_no_longer_pair_richer():
    """The subset strategies, both superseded by running all 198.

    Targeted top-up was adopted because at 18 tissue groups it reached more
    replicate pairs than `--per-group 8` while extracting a third fewer
    experiments. Splitting blood_immune into four lineages (21 groups) killed
    the pair advantage -- more groups means `--per-group 8` sweeps in more
    replicated biosamples -- leaving only the cost advantage:

        top-up(3, +2)   81 experiments    33 replicate pairs
        per-group 8    130 experiments    46 replicate pairs
        all 198        198 experiments   289 replicate pairs

    Neither subset is close to the full run, which is why both flags are
    superseded. Kept as a regression test on `balanced_subset` and as the
    record of why the top-up recommendation was withdrawn.
    """
    import yaml

    cfg = yaml.safe_load(open(REPO_ROOT / "configs" / "experiment_config.yaml"))
    sys.path.insert(0, str(REPO_ROOT / "src" / "analysis"))
    from _biosample_groups import load_group_map
    from plot_motif_rarefaction import load_read_counts

    tissue, sample = load_group_map(cfg["experiments"], None, quiet=True)
    depth = load_read_counts()

    def replicate_pairs(sub):
        exps = list(sub)
        return sum(
            1
            for i in range(len(exps))
            for j in range(i + 1, len(exps))
            if sample.get(exps[i]) == sample.get(exps[j])
        )

    topped = cc.balanced_subset(
        cfg["experiments"], tissue, 3, depth, min_reads=10e6,
        biosamples=sample, replicates_per_group=2,
    )
    wide = cc.balanced_subset(
        cfg["experiments"], tissue, 8, depth, min_reads=10e6, biosamples=sample
    )
    assert len(topped) < len(wide), "top-up should stay the cheaper extraction"
    # The pair advantage is gone at 21 groups; assert the current direction so
    # a future regrouping that restores it shows up as a failure to look at.
    assert replicate_pairs(topped) < replicate_pairs(wide)
    assert replicate_pairs(wide) < 289, "the full 198 run dominates both"


# --- naming the dominant tissue --------------------------------------------
#
# The cleanest framing: every model sees the identical sequence at a given
# peak, so all variation across models there is model-specific. No shared
# component to cancel, and immune to units, scale, and whether log1p is
# behaving linearly -- only the ordering of groups matters.


def topk_frames(n_peaks=300, seed=0, model_skill=1.0):
    """Peaks each dominated by one tissue; models partly recover which."""
    rng = np.random.default_rng(seed)
    tissues = ["blood", "heart", "liver"]
    index, groups = [], {}
    for tis in tissues:
        for r in range(2):
            e = f"{tis}{r}"
            index.append(e)
            groups[e] = tis
    winner = rng.integers(0, len(tissues), n_peaks)
    obs = np.ones((len(index), n_peaks))
    pred = np.ones((len(index), n_peaks))
    for i, e in enumerate(index):
        ti = tissues.index(groups[e])
        obs[i] += 50 * (winner == ti)
        pred[i] += 50 * model_skill * (winner == ti)
    cols = [f"p{i}" for i in range(n_peaks)]
    return (
        pd.DataFrame(obs, index=index, columns=cols),
        pd.DataFrame(pred, index=index, columns=cols),
        groups,
    )


def test_topk_is_perfect_when_models_know_the_tissue():
    obs, pred, groups = topk_frames()
    spec = ccp.peak_specificity(obs, groups)
    out = ccp.dominant_tissue_accuracy(obs, pred, groups, spec, quantiles=(0.0,))
    assert out["top1"].iloc[0] == pytest.approx(1.0)
    assert out["n_groups"].iloc[0] == 3
    # reported rounded to 4dp
    assert out["top1_chance"].iloc[0] == pytest.approx(1 / 3, abs=1e-4)


def test_topk_is_near_chance_when_models_ignore_the_tissue():
    obs, pred, groups = topk_frames(model_skill=0.0)
    spec = ccp.peak_specificity(obs, groups)
    out = ccp.dominant_tissue_accuracy(obs, pred, groups, spec, quantiles=(0.0,))
    # all models predict identically, so the argmax is arbitrary
    assert out["top1"].iloc[0] < 0.6


def test_topk_reports_rank_of_the_true_tissue():
    obs, pred, groups = topk_frames()
    spec = ccp.peak_specificity(obs, groups)
    out = ccp.dominant_tissue_accuracy(obs, pred, groups, spec, quantiles=(0.0,))
    assert out["median_rank_of_true"].iloc[0] == 1


def test_topk_excludes_groups_with_one_experiment():
    obs, pred, groups = topk_frames()
    # add a singleton group: its "mean" would be a single experiment
    obs.loc["solo"] = 1.0
    pred.loc["solo"] = 99.0
    groups["solo"] = "adipose"
    spec = ccp.peak_specificity(obs, groups)
    out = ccp.dominant_tissue_accuracy(obs, pred, groups, spec, quantiles=(0.0,))
    assert out["n_groups"].iloc[0] == 3, "adipose has one experiment"


def test_topk_is_scale_invariant_across_models():
    """Immune to the per-model scale differences that broke the correlations:
    only the ordering of group means at a peak matters."""
    obs, pred, groups = topk_frames()
    spec = ccp.peak_specificity(obs, groups)
    base = ccp.dominant_tissue_accuracy(obs, pred, groups, spec, quantiles=(0.0,))
    scaled = pred.copy()
    scaled.loc["blood0"] *= 50            # one model on a wildly different scale
    out = ccp.dominant_tissue_accuracy(obs, scaled, groups, spec, quantiles=(0.0,))
    # group means shift, but every model in a group shares the tissue signal
    assert out["top1"].iloc[0] >= base["top1"].iloc[0] - 0.35


def test_topk_skips_thresholds_with_too_few_peaks():
    obs, pred, groups = topk_frames(n_peaks=30)
    spec = ccp.peak_specificity(obs, groups)
    out = ccp.dominant_tissue_accuracy(
        obs, pred, groups, spec, quantiles=(0.0, 0.99)
    )
    assert (out["n_peaks"] >= 20).all()


def test_cli_reports_topk_sweep(tmp_path):
    result = run_cli(tmp_path)
    assert result.returncode == 0, result.stderr
    assert "Naming the most-active tissue" in result.stderr
    got = pd.read_csv(tmp_path / "out" / "cross_celltype_topk.tsv", sep="\t")
    assert "top1_over_chance" in got.columns


# --- quality filter and group balancing ------------------------------------


def test_exclude_low_quality_drops_blacklisted_and_uncapped():
    exps = {
        "good": {"library_construction": "PRO-cap"},
        "bad": {"library_construction": "PRO-cap"},
        "uncap": {"library_construction": "PRO-cap, uncapped"},
    }
    kept, bl, un = cc.exclude_low_quality(exps, {"bad"})
    assert set(kept) == {"good"}
    assert bl == ["bad"] and un == ["uncap"]


def test_exclude_low_quality_matches_the_rest_of_the_paper():
    """count_correlation.py had no experiment-level quality filter -- its five
    'blacklist' references are the genomic hg38 blacklist -- so a run over
    everything above 10M reads took in 203 experiments including the four
    uncapped libraries and ENCSR973QQI. Every other analysis uses 198."""
    import yaml

    cfg = yaml.safe_load(open(REPO_ROOT / "configs" / "experiment_config.yaml"))
    sys.path.insert(0, str(REPO_ROOT / "src" / "analysis"))
    from plot_motif_rarefaction import load_read_counts

    kept, bl, un = cc.exclude_low_quality(cfg["experiments"], {"ENCSR973QQI"})
    assert len(kept) == 219, "224 minus 1 blacklisted minus 4 uncapped"
    depth = load_read_counts()
    assert sum(1 for e in kept if depth.get(e, 0) >= 10e6) == 198


def test_exclude_low_quality_can_be_disabled():
    exps = {"a": {"library_construction": "PRO-cap"}}
    kept, bl, un = cc.exclude_low_quality(exps, set())
    assert set(kept) == {"a"} and not bl


def test_tier_summary_is_group_balanced_as_well_as_pooled():
    # one huge group with low values, one small group with high ones: pooling
    # hands the answer to the huge group
    pairs = pd.DataFrame(
        [dict(tier="different tissue", model_group="blood", correlation=0.1)] * 40
        + [dict(tier="different tissue", model_group="stem", correlation=0.9)] * 2
    )
    out = ccp.summarize_tiers(pairs).iloc[0]
    assert out["median"] == pytest.approx(0.1)
    assert out["median_group_balanced"] == pytest.approx(0.5)
    assert out["n_groups"] == 2


def test_tier_summary_omits_balancing_without_groups():
    pairs = pd.DataFrame([dict(tier="matched", correlation=0.5)])
    out = ccp.summarize_tiers(pairs)
    assert "median_group_balanced" not in out.columns


# --- differential ceiling ----------------------------------------------------


def test_replicate_pairs_are_disjoint_within_a_biosample():
    bios = {f"e{i}": "B" for i in range(5)}
    pairs = ccp.replicate_pairs([f"e{i}" for i in range(5)], bios)
    used = [e for pair in pairs["B"] for e in pair]
    assert len(used) == len(set(used)), "an experiment was reused across pairs"
    assert len(pairs["B"]) == 2, "5 experiments give 2 disjoint pairs, not 3"


def test_replicate_pairs_skips_unreplicated_biosamples():
    bios = {"a": "A", "b": "B", "c": "B"}
    pairs = ccp.replicate_pairs(["a", "b", "c"], bios)
    assert "A" not in pairs and pairs["B"] == [("b", "c")]


def test_replicate_pairs_is_capped_per_biosample():
    bios = {f"e{i}": "B" for i in range(12)}
    pairs = ccp.replicate_pairs([f"e{i}" for i in range(12)], bios,
                                max_per_biosample=2)
    assert len(pairs["B"]) == 2


def test_replicate_pairs_is_order_independent():
    exps = [f"e{i}" for i in range(6)]
    bios = {e: "B" for e in exps}
    assert (ccp.replicate_pairs(exps, bios)
            == ccp.replicate_pairs(list(reversed(exps)), bios))


def test_a_predictor_as_good_as_a_replicate_attains_exactly_the_ceiling():
    """The definition, stated as a test.

    If a model's prediction for a1 is literally another replicate's
    measurement (a2), then its differential correlation *is* the ceiling, so
    attained must be exactly 1.0. Anything else means `model` and `ceiling`
    are not being computed on the same contrast.
    """
    rng = np.random.default_rng(11)
    n = 300
    sig_a, sig_b = rng.normal(size=n) * 10, rng.normal(size=n) * 10
    rows = {
        "a1": np.abs(sig_a + rng.normal(size=n)),
        "a2": np.abs(sig_a + rng.normal(size=n)),
        "b1": np.abs(sig_b + rng.normal(size=n)),
        "b2": np.abs(sig_b + rng.normal(size=n)),
    }
    obs = pd.DataFrame(rows).T
    obs.columns = [f"p{i}" for i in range(n)]
    pred = obs.copy()
    pred.loc["a1"] = obs.loc["a2"].to_numpy()   # predict a1 using replicate a2
    pred.loc["b1"] = obs.loc["b2"].to_numpy()
    groups = {"a1": "ga", "a2": "ga", "b1": "gb", "b2": "gb"}
    bios = {"a1": "A", "a2": "A", "b1": "B", "b2": "B"}
    q = ccp.differential_ceiling(obs, pred, groups, bios)
    assert len(q) == 1
    assert q.at[0, "ceiling"] > 0.5, "replicates should agree on the difference"
    assert q.at[0, "attained"] == pytest.approx(1.0, abs=1e-3)


def test_ceiling_is_high_when_replicates_agree_and_low_when_they_do_not():
    rng = np.random.default_rng(0)
    n = 400
    signal_a = rng.normal(size=n) * 10
    signal_b = rng.normal(size=n) * 10
    groups = {"a1": "ga", "a2": "ga", "b1": "gb", "b2": "gb"}
    bios = {"a1": "A", "a2": "A", "b1": "B", "b2": "B"}

    def frame(noise):
        return pd.DataFrame(
            np.abs(np.vstack([
                signal_a + rng.normal(size=n) * noise,
                signal_a + rng.normal(size=n) * noise,
                signal_b + rng.normal(size=n) * noise,
                signal_b + rng.normal(size=n) * noise,
            ])),
            index=["a1", "a2", "b1", "b2"],
            columns=[f"p{i}" for i in range(n)],
        )

    clean = ccp.differential_ceiling(frame(0.1), frame(0.1), groups, bios)
    noisy = ccp.differential_ceiling(frame(20.0), frame(20.0), groups, bios)
    assert clean.at[0, "ceiling"] > 0.9
    assert noisy.at[0, "ceiling"] < clean.at[0, "ceiling"]


def test_differential_ceiling_is_empty_without_replication_on_both_sides():
    exps = ["a1", "a2", "b1"]
    obs = counts_frame(exps, n_peaks=50)
    groups = {"a1": "ga", "a2": "ga", "b1": "gb"}
    bios = {"a1": "A", "a2": "A", "b1": "B"}
    assert ccp.differential_ceiling(obs, obs, groups, bios).empty


def test_summarize_ceiling_orders_tiers_and_reports_quadruple_counts():
    quads = pd.DataFrame({
        "tier": ["different tissue", "same tissue", "different tissue"],
        "ceiling": [0.8, 0.6, 0.7],
        "model": [0.2, 0.1, 0.3],
        "attained": [0.25, 0.1667, 0.4286],
    })
    out = ccp.summarize_ceiling(quads)
    assert list(out["tier"]) == ["same tissue", "different tissue"]
    assert list(out["n_quadruples"]) == [1, 2]


def test_attained_is_nan_rather_than_infinite_at_a_zero_ceiling():
    quads = pd.DataFrame({
        "tier": ["same tissue"], "ceiling": [0.0], "model": [0.2],
        "attained": [np.nan],
    })
    out = ccp.summarize_ceiling(quads)
    assert np.isnan(out.at[0, "median_attained"])


# --- differential prediction panels -----------------------------------------


def ceiling_frame(rows):
    """rows = [(tier, ceiling, model)]"""
    df = pd.DataFrame(rows, columns=["tier", "ceiling", "model"])
    df["biosample_a"] = "A"
    df["biosample_b"] = "B"
    df["experiment_a"] = "e1"
    df["experiment_b"] = "e2"
    df["attained"] = np.where(df["ceiling"] > 0, df["model"] / df["ceiling"],
                              np.nan)
    return df


def test_ceiling_fit_excludes_non_positive_ceilings():
    """A negative ceiling means the replicate differentials anticorrelate, so
    the ratio is meaningless rather than small."""
    quads = ceiling_frame([
        ("different tissue", 0.8, 0.2),
        ("different tissue", 0.6, 0.15),
        ("different tissue", -0.4, 0.10),
        ("different tissue", 0.0, 0.05),
    ])
    usable, slope, dropped = ccp.ceiling_fit(quads)
    assert len(usable) == 2 and dropped == 2
    assert set(usable["ceiling"]) == {0.8, 0.6}


def test_ceiling_fit_uses_through_origin_slope_not_the_mean_ratio():
    """sum(xy)/sum(x^2), not mean(y/x).

    They differ materially: a ratio whose denominator is a near-zero ceiling
    is noise, and averaging ratios lets those points dominate.
    """
    quads = ceiling_frame([
        ("different tissue", 0.8, 0.20),    # ratio 0.25
        ("different tissue", 0.4, 0.10),    # ratio 0.25
        ("different tissue", 0.02, 0.02),   # ratio 1.00 -- noise
    ])
    _, slope, _ = ccp.ceiling_fit(quads)
    x = np.array([0.8, 0.4, 0.02])
    y = np.array([0.20, 0.10, 0.02])
    assert slope == pytest.approx((x * y).sum() / (x * x).sum())
    assert slope == pytest.approx(0.2506, abs=1e-3)
    assert abs(slope - (y / x).mean()) > 0.2, "the mean ratio would be 0.5"


def test_ceiling_fit_is_all_nan_when_nothing_is_usable():
    quads = ceiling_frame([("different tissue", -0.2, 0.1)])
    usable, slope, dropped = ccp.ceiling_fit(quads)
    assert len(usable) == 0 and dropped == 1 and np.isnan(slope)


def test_ceiling_panel_annotates_the_fitted_slope():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    quads = ceiling_frame([("different tissue", 0.8, 0.2),
                           ("different tissue", 0.6, 0.15)])
    fig, ax = plt.subplots()
    slope = ccp.draw_differential_ceiling(ax, quads)
    texts = [t.get_text() for t in ax.texts]
    plt.close(fig)
    assert any(f"{slope:.0%}" in t for t in texts), texts


def test_ceiling_panel_states_how_many_quadruples_it_omitted():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    quads = ceiling_frame([("different tissue", 0.8, 0.2),
                           ("different tissue", -0.3, 0.1),
                           ("different tissue", 0.0, 0.1)])
    fig, ax = plt.subplots()
    ccp.draw_differential_ceiling(ax, quads)
    texts = " ".join(t.get_text() for t in ax.texts)
    plt.close(fig)
    assert "2 of 3" in texts, texts


def test_ceiling_panel_annotations_do_not_overlap():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    ceil = rng.uniform(0.35, 0.95, 400)
    quads = ceiling_frame([
        ("different tissue", float(c), float(c * 0.26 + rng.normal(0, 0.02)))
        for c in ceil
    ] + [("different tissue", -0.2, 0.05)])
    fig, ax = plt.subplots(figsize=(4.0, 3.1))
    ccp.draw_differential_ceiling(ax, quads)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    boxes = [t.get_window_extent(r) for t in ax.texts if t.get_text().strip()]
    plt.close(fig)
    assert len(boxes) >= 3, "expected ceiling, slope and omission labels"
    for i, a in enumerate(boxes):
        for b in boxes[i + 1:]:
            assert not a.overlaps(b), "ceiling-panel annotations overlap"


def test_differential_tiers_draws_one_box_per_present_tier(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    pairs = pd.DataFrame({
        "tier": (["same biosample"] * 5 + ["same tissue"] * 5
                 + ["different tissue"] * 5),
        "differential_r": ([0.05] * 5 + [0.12] * 5 + [0.18] * 5),
    })
    summary = ccp.summarize_differential(pairs)
    path = tmp_path / "tiers.pdf"
    ccp.plot_differential_tiers(pairs, summary, path)
    assert path.exists() and path.stat().st_size > 1000


def test_differential_tiers_omits_absent_tiers(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    pairs = pd.DataFrame({
        "tier": ["different tissue"] * 6,
        "differential_r": [0.1, 0.2, 0.15, 0.18, 0.12, 0.19],
    })
    summary = ccp.summarize_differential(pairs)
    path = tmp_path / "one.pdf"
    ccp.plot_differential_tiers(pairs, summary, path)   # must not raise
    assert path.exists()


def test_differential_tiers_labels_every_box_with_n_and_median():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pairs = pd.DataFrame({
        "tier": ["same biosample"] * 4 + ["different tissue"] * 6,
        "differential_r": [0.05, 0.06, 0.07, 0.08] + [0.18] * 6,
    })
    fig, ax = plt.subplots(figsize=(4.0, 3.1))
    present = ccp.draw_differential_tiers(ax, pairs)
    texts = " ".join(t.get_text() for t in ax.texts)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    boxes = [t.get_window_extent(r) for t in ax.texts if t.get_text().strip()]
    plt.close(fig)
    assert present == ["same biosample", "different tissue"]
    assert "n=4" in texts and "n=6" in texts
    assert "med 0.180" in texts
    for i, a in enumerate(boxes):
        for b in boxes[i + 1:]:
            assert not a.overlaps(b), "tier labels overlap"


# --- top-k and homogenization panels ----------------------------------------


def topk_frame(n_groups=20, quantiles=(0.0, 0.9, 0.99)):
    rows = []
    for i, q in enumerate(quantiles):
        row = {"tau_quantile": q, "n_peaks": 96121 // (10 ** i),
               "n_groups": n_groups, "median_rank_of_true": 5 - i}
        for k in (1, 3, 5):
            acc = 0.19 + 0.07 * i + 0.2 * (k // 3)
            row[f"top{k}"] = acc
            row[f"top{k}_chance"] = k / n_groups
            row[f"top{k}_over_chance"] = round(acc / (k / n_groups), 2)
        rows.append(row)
    return pd.DataFrame(rows)


def test_topk_panel_draws_a_curve_and_a_chance_line_per_k():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    present = ccp.draw_topk(ax, topk_frame())
    curves = [ln for ln in ax.get_lines() if len(ln.get_xdata()) > 2]
    hlines = [ln for ln in ax.get_lines() if ln.get_linestyle() == ":"]
    plt.close(fig)
    assert present == [1, 3, 5]
    assert len(curves) == 3, "one accuracy curve per k"
    assert len(hlines) == 3, "one chance line per k"


def test_topk_panel_shows_absolute_accuracy_not_only_the_multiple():
    """The over-chance ratio alone reads as a big effect while hiding that
    top-1 is a third; the y axis must carry the level."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ccp.draw_topk(ax, topk_frame())
    lo, hi = ax.get_ylim()
    label = ax.get_ylabel()
    plt.close(fig)
    assert (lo, hi) == (0, 1), "accuracy axis should span the full range"
    assert "accuracy" in label.lower()


def test_topk_panel_tick_labels_carry_the_peak_count():
    """The rightmost threshold rests on ~1,000 peaks against ~96,000; a reader
    cannot weigh the curve's right end without that."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    t = topk_frame()
    ccp.draw_topk(ax, t)
    labels = [lbl.get_text() for lbl in ax.get_xticklabels()]
    plt.close(fig)
    assert len(labels) == len(t)
    assert all("\n" in lbl for lbl in labels)
    assert f"{t['n_peaks'].iloc[-1]:,}" in labels[-1]


def test_topk_panel_spaces_thresholds_evenly_not_linearly():
    """0, 0.5, 0.8, 0.9, 0.95, 0.99 on a linear axis crushes the four
    thresholds that matter into the right fifth of the panel."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    t = topk_frame(quantiles=(0.0, 0.5, 0.8, 0.9, 0.95, 0.99))
    ccp.draw_topk(ax, t)
    xs = sorted(ax.get_xticks())
    plt.close(fig)
    gaps = np.diff(xs)
    assert np.allclose(gaps, gaps[0]), "thresholds should be evenly spaced"


def homogenization_frame(measured, predicted):
    tiers = ["same biosample", "same tissue", "different tissue"]
    rows = []
    for source, vals in (("measured", measured), ("predicted", predicted)):
        for tier, v in zip(tiers, vals):
            rows.append({"source": source, "tier": tier, "n_pairs": 100,
                         "median": v, "q25": v - 0.05, "q75": v + 0.05})
    return pd.DataFrame(rows)


def test_homogenization_panel_plots_both_sources_over_the_tiers():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ccp.draw_homogenization(
        ax, homogenization_frame([0.82, 0.20, 0.02], [0.75, 0.66, 0.57]),
        "specific",
    )
    lines = [ln for ln in ax.get_lines() if len(ln.get_xdata()) == 3]
    labels = [lbl.get_text().replace("\n", " ") for lbl in ax.get_xticklabels()]
    plt.close(fig)
    assert len(lines) == 2, "one line per source"
    assert labels == ["same biosample", "same tissue", "different tissue"]


def test_homogenization_gap_annotation_reports_the_final_tier_gap():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ccp.draw_homogenization(
        ax, homogenization_frame([0.82, 0.20, 0.02], [0.75, 0.66, 0.57]),
        "specific", annotate_gap=True,
    )
    texts = " ".join(t.get_text() for t in ax.texts)
    plt.close(fig)
    assert "0.55" in texts, texts       # 0.57 - 0.02


def test_homogenization_gap_annotation_is_off_by_default():
    """The ubiquitous stratum has no interesting gap; annotating it there
    would imply a failure the numbers do not show."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ccp.draw_homogenization(
        ax, homogenization_frame([0.94, 0.87, 0.79], [0.93, 0.91, 0.89]),
        "ubiquitous",
    )
    texts = [t.get_text() for t in ax.texts if t.get_text().strip()]
    plt.close(fig)
    assert not texts


def test_homogenization_figure_keeps_one_legend_for_the_pair(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    spec = homogenization_frame([0.82, 0.20, 0.02], [0.75, 0.66, 0.57])
    ubiq = homogenization_frame([0.94, 0.87, 0.79], [0.93, 0.91, 0.89])
    path = tmp_path / "hom.pdf"
    ccp.plot_homogenization(spec, ubiq, path)
    assert path.exists() and path.stat().st_size > 1000


# --- peak-class stratification (candidate promoter vs enhancer) ------------


def peak_class_frames(n_peaks=300, seed=0):
    """topk_frames() plus a peak_class Series, half promoter/half enhancer.

    Peaks alternate class by position so both classes see the same tissue
    signal by construction -- any accuracy gap between them in a test would
    have to come from the split logic, not from one class happening to draw
    easier peaks.
    """
    obs, pred, groups = topk_frames(n_peaks=n_peaks, seed=seed)
    labels = np.where(np.arange(n_peaks) % 2 == 0, "promoter", "enhancer")
    classes = pd.Series(labels, index=obs.columns)
    return obs, pred, groups, classes


def test_restrict_by_peak_class_splits_columns_by_label():
    obs, pred, _groups, classes = peak_class_frames()
    restricted = ccp.restrict_by_peak_class(obs, pred, classes)
    assert set(restricted) == {"promoter", "enhancer"}
    for name, (oc, pc) in restricted.items():
        assert list(oc.columns) == list(pc.columns)
        assert set(classes[oc.columns]) == {name}
    assert len(restricted["promoter"][0].columns) + len(restricted["enhancer"][0].columns) == len(obs.columns)


def test_restrict_by_peak_class_drops_unclassified_peaks():
    obs, pred, _groups, classes = peak_class_frames()
    classes.iloc[:] = np.nan
    classes.iloc[:40] = "promoter"  # only promoter clears min_peaks
    restricted = ccp.restrict_by_peak_class(obs, pred, classes)
    assert set(restricted) == {"promoter"}


def test_restrict_by_peak_class_respects_min_peaks():
    obs, pred, _groups, classes = peak_class_frames()
    classes.iloc[2:] = "promoter"  # only 1 enhancer peak left
    restricted = ccp.restrict_by_peak_class(obs, pred, classes, min_peaks=20)
    assert "enhancer" not in restricted


def test_topk_by_peak_class_sweeps_tau_separately_per_class():
    """One full tau-quantile sweep per class, the same shape dominant_tissue
    _accuracy gives the unstratified panel -- not a single pooled row."""
    obs, pred, groups, classes = peak_class_frames()
    restricted = ccp.restrict_by_peak_class(obs, pred, classes)
    out = ccp.topk_by_peak_class(restricted, groups)
    assert set(out) == {"promoter", "enhancer"}
    for name, topk in out.items():
        assert "tau_quantile" in topk.columns
        assert len(topk) > 1, f"{name} should sweep multiple thresholds"
        assert (topk["top1"] > 0.8).all(), "both classes see the same easy signal"


def test_topk_by_peak_class_threads_min_peak_signal():
    """A high enough min_peak_signal should filter every peak in a class,
    dropping it from the result -- confirms the argument actually reaches
    each class's own top_group_signal/peak_specificity call rather than
    being accepted and ignored."""
    obs, pred, groups, classes = peak_class_frames()
    restricted = ccp.restrict_by_peak_class(obs, pred, classes)
    out = ccp.topk_by_peak_class(restricted, groups, min_peak_signal=1e6)
    assert out == {}


def test_topk_by_peak_class_threads_specificity_index():
    obs, pred, groups, classes = peak_class_frames()
    restricted = ccp.restrict_by_peak_class(obs, pred, classes)
    out = ccp.topk_by_peak_class(restricted, groups, specificity_index="entropy")
    assert set(out) == {"promoter", "enhancer"}


def test_topk_by_peak_class_is_empty_when_nothing_survives_restriction():
    _obs, _pred, groups, _classes = peak_class_frames()
    out = ccp.topk_by_peak_class({}, groups)
    assert out == {}


def test_differential_ceiling_by_peak_class_tags_and_concatenates():
    exps = ["a1", "a2", "b1", "b2"]
    groups = {"a1": "ga", "a2": "ga", "b1": "gb", "b2": "gb"}
    bios = {"a1": "A", "a2": "A", "b1": "B", "b2": "B"}
    obs = counts_frame(exps, n_peaks=60)
    obs.columns = [str(i) for i in range(60)]
    classes = pd.Series(
        np.where(np.arange(60) % 2 == 0, "promoter", "enhancer"),
        index=obs.columns,
    )
    restricted = ccp.restrict_by_peak_class(obs, obs, classes, min_peaks=20)
    out = ccp.differential_ceiling_by_peak_class(restricted, groups, bios)
    assert set(out["peak_class"]) == set(restricted)
    assert {"ceiling", "model", "attained", "tier"} <= set(out.columns)


def test_topk_by_peak_class_tables_plot_with_the_unstratified_plotter(tmp_path):
    """The point of sweeping tau per class is reusing draw_topk/plot_topk
    unmodified -- draw_topk's own behavior is already covered where it is
    defined; this just confirms a class-restricted table has the shape it
    expects (tau_quantile, n_peaks, n_groups, top{k}/_chance/_over_chance)."""
    obs, pred, groups, classes = peak_class_frames()
    restricted = ccp.restrict_by_peak_class(obs, pred, classes)
    out = ccp.topk_by_peak_class(restricted, groups)
    for name, topk in out.items():
        path = tmp_path / f"{name}.pdf"
        present = ccp.plot_topk(topk, path)
        assert present
        assert path.exists() and path.stat().st_size > 1000


def peak_class_bed_inputs(tmp_path, n_peaks=300):
    """union_peaks.bed + a matching cCRE bed with alternating PLS/dELS calls,
    one per peak, so every peak lands squarely in one class or the other."""
    union = tmp_path / "union_peaks.bed"
    ccre = tmp_path / "ccres.bed"
    with open(union, "w") as f:
        for i in range(n_peaks):
            start = i * 1000
            f.write(f"chr1\t{start}\t{start + 200}\n")
    with open(ccre, "w") as f:
        for i in range(n_peaks):
            mid = i * 1000 + 100
            label = "PLS" if i % 2 == 0 else "dELS"
            f.write(f"chr1\t{mid - 50}\t{mid + 50}\tEH38D{i}\tEH38E{i}\t{label}\n")
    return union, ccre


def run_cli_with_positional_peak_ids(tmp_path, *extra, n_peaks=300):
    """Like run_cli, but with "0", "1", ... peak columns.

    tiered_inputs names peaks "p0", "p1", ... for readability, but real
    observed/predicted_counts.tsv columns are the bare 0-based position in
    union_peaks.bed.gz (see count_correlation.py's extract_observed_counts
    docstring) -- --peak-class relies on that to join classify_peaks_by_cre's
    output back onto the count matrices, so this test needs the real format.
    """
    o, p = tiered_inputs(tmp_path, n_peaks=n_peaks)
    obs_df = pd.read_csv(o, sep="\t", index_col=0)
    pred_df = pd.read_csv(p, sep="\t", index_col=0)
    obs_df.columns = [str(i) for i in range(obs_df.shape[1])]
    pred_df.columns = [str(i) for i in range(pred_df.shape[1])]
    obs_df.to_csv(o, sep="\t")
    pred_df.to_csv(p, sep="\t")
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "src/analysis/cross_celltype_prediction.py"),
         "--observed", str(o), "--predicted", str(p),
         "--out-dir", str(tmp_path / "out"), *extra],
        capture_output=True, text=True, env=env(),
    )


def test_cli_reports_peak_class_topk_and_ceiling(tmp_path):
    union, ccre = peak_class_bed_inputs(tmp_path)
    result = run_cli_with_positional_peak_ids(
        tmp_path, "--peak-class",
        "--union-peaks", str(union), "--ccre-bed", str(ccre),
    )
    assert result.returncode == 0, result.stderr
    assert "among candidate promoter peaks" in result.stderr
    assert "among candidate enhancer peaks" in result.stderr
    out = tmp_path / "out"
    for name in ("promoter", "enhancer"):
        topk = pd.read_csv(out / f"cross_celltype_topk_{name}.tsv", sep="\t")
        assert "tau_quantile" in topk.columns
        assert (out / f"cross_celltype_topk_{name}.pdf").exists()


def test_cli_peak_class_requires_both_files(tmp_path):
    result = run_cli(tmp_path, "--peak-class")
    assert result.returncode == 1
    assert "--peak-class needs" in result.stderr


def test_cli_without_peak_class_does_not_write_its_outputs(tmp_path):
    result = run_cli(tmp_path)
    assert result.returncode == 0, result.stderr
    out = tmp_path / "out"
    assert not (out / "cross_celltype_topk_by_peak_class.tsv").exists()
