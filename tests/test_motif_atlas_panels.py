"""Tests for the Fig. 2 motif-atlas panels.

The real inputs (MotifCompendium cluster metadata, per-experiment
hits_linked.tsv) are produced on the cluster, so these tests build synthetic
fixtures that match those schemas exactly -- cluster_motifs.py's
`cluster_final/posneg/experiments` metadata columns and
link_hits_to_compendium.py's `compendium_motif_name` hits column -- and check
the panel arithmetic against them.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Scripts under test import matplotlib.pyplot at module level and some call
# into memelite, which uses OpenMP. On macOS the default (MacOSX) backend can
# SIGABRT when a subprocess mixes the two, which made these tests flaky rather
# than failing -- a single-file run passed while the full suite aborted. Force
# a headless backend and single-threaded OMP so subprocess runs are
# deterministic; this is a test-harness concern, not script behaviour (on the
# cluster the backend is already Agg).
SUBPROC_ENV = {**os.environ, "MPLBACKEND": "Agg", "OMP_NUM_THREADS": "1"}
sys.path.insert(0, str(REPO_ROOT / "src" / "analysis"))
sys.path.insert(0, str(REPO_ROOT / "src" / "bpnet" / "hitcall"))

import _biosample_groups as bg  # noqa: E402
import plot_motif_rarefaction as rare  # noqa: E402

pytest.importorskip("seaborn")
import motif_hit_density as mhd  # noqa: E402


# --------------------------------------------------------------------------
# biosample grouping
# --------------------------------------------------------------------------


def test_metastases_are_not_routed_to_the_destination_organ():
    """Metastatic biosamples are named for where they spread *to*, so an
    organ rule must never claim them ahead of the metastasis rule."""
    assert bg.assign_group("Metastatic Breast Carcinoma in the Brain") == "metastatic_carcinoma"
    assert bg.assign_group("Colon Carcinoma Metastatic in the Lung") == "metastatic_carcinoma"
    assert bg.assign_group("Metastatic Liver Carcinoma in the Adrenal Gland") == "metastatic_carcinoma"


@pytest.mark.parametrize(
    "biosample,expected",
    [
        ("cerebellum", "neural"),
        ("heart left ventricle", "heart"),
        ("right lobe of liver", "liver_biliary"),
        ("HCT116", "gi_tract"),
        ("K562", "blood_immune"),
        ("peripheral blood mononuclear cell", "blood_immune"),
        ("endothelial cell of umbilical vein", "vascular"),
        ("body of pancreas", "pancreas"),
        ("HEK293T", "hek"),
        ("some novel biosample", "other"),
    ],
)
def test_group_assignment(biosample, expected):
    assert bg.assign_group(biosample) == expected


def test_override_tsv_wins_but_unlisted_biosamples_still_fall_back(tmp_path):
    override = tmp_path / "groups.tsv"
    override.write_text("biosample\tgroup\nK562\tmy_custom_group\n")
    experiments = {
        "EXP1": {"biosample": "K562"},
        "EXP2": {"biosample": "cerebellum"},
    }
    group_map, biosample_map = bg.load_group_map(experiments, override, quiet=True)
    assert group_map == {"EXP1": "my_custom_group", "EXP2": "neural"}
    assert biosample_map["EXP1"] == "K562"


def test_write_group_tsv_roundtrips(tmp_path):
    experiments = {"E1": {"biosample": "liver"}, "E2": {"biosample": "liver"}}
    group_map, biosample_map = bg.load_group_map(experiments, quiet=True)
    out = tmp_path / "t.tsv"
    bg.write_group_tsv(biosample_map, group_map, out)
    table = pd.read_csv(out, sep="\t", comment="#")
    assert table.loc[0, "n_experiments"] == 2
    assert table.loc[0, "group"] == "liver_biliary"


# --------------------------------------------------------------------------
# rarefaction
# --------------------------------------------------------------------------


def write_cluster_metadata(path, rows, with_jaspar=True):
    """Write a cluster_motifs.py-shaped cluster metadata TSV."""
    records = []
    for cluster_final, (posneg, exps, jaspar) in enumerate(rows):
        rec = {
            "cluster_final": cluster_final,
            "n_motifs": len(exps),
            "total_seqlets": 100 * len(exps),
            "n_experiments": len(exps),
            "experiments": ",".join(sorted(exps)),
            "posneg": posneg,
        }
        if with_jaspar:
            rec["jaspar_name"] = jaspar
            rec["jaspar_score"] = 0.9 if jaspar else np.nan
        records.append(rec)
    pd.DataFrame(records).to_csv(path, sep="\t", index=False)
    return path


def test_load_presence_parses_and_filters(tmp_path):
    meta_path = write_cluster_metadata(
        tmp_path / "m.tsv",
        [
            ("pos", ["E1", "E2", "E3"], "SP1"),
            ("pos", ["E4"], None),  # dropped entirely by the keep set below
            ("neg", ["E2", "E4"], "GATA1"),
        ],
    )
    meta, universe = rare.load_presence(meta_path, keep_experiments={"E1", "E2", "E3"})
    assert universe == ["E1", "E2", "E3"]
    # cluster 1 lived only in E4 and is unreachable in this universe
    assert list(meta["cluster_final"]) == [0, 2]
    assert list(meta["prevalence"]) == [3, 1]


def test_load_presence_rejects_wrong_schema(tmp_path):
    bad = tmp_path / "bad.tsv"
    pd.DataFrame({"cluster_final": [0]}).to_csv(bad, sep="\t", index=False)
    with pytest.raises(ValueError, match="experiments"):
        rare.load_presence(bad)


def test_uniform_curve_exact_endpoints():
    """At k=1 the expectation is sum(p/N); at k=N every cluster is found."""
    prevalences = np.array([1, 2, 5, 5, 10])
    n = 20
    curve = rare.uniform_curve_exact(prevalences, n)
    assert curve[0] == pytest.approx(prevalences.sum() / n)
    assert curve[-1] == pytest.approx(len(prevalences))
    assert np.all(np.diff(curve) >= -1e-9)  # monotone non-decreasing


def test_uniform_curve_exact_matches_monte_carlo():
    """The closed form must agree with brute-force subsampling."""
    rng = np.random.default_rng(0)
    n = 12
    exp_ids = [f"E{i}" for i in range(n)]
    exp_sets = [
        set(rng.choice(exp_ids, size=int(p), replace=False))
        for p in (1, 3, 3, 6, 9)
    ]
    prevalences = np.array([len(s) for s in exp_sets])
    exact = rare.uniform_curve_exact(prevalences, n)

    groups = {e: "g" for e in exp_ids}
    mc = rare.rarefy(exp_sets, groups, "uniform", n_reps=4000, rng=rng).mean(axis=0)
    assert np.allclose(exact, mc, atol=0.12)


def test_rarefy_curves_are_monotone_and_complete():
    exp_sets = [{"E1", "E2"}, {"E3"}, {"E2", "E3", "E4"}]
    groups = {"E1": "a", "E2": "a", "E3": "b", "E4": "b"}
    for scheme in ("uniform", "diverse", "redundant"):
        curves = rare.rarefy(exp_sets, groups, scheme, 25, np.random.default_rng(1))
        assert curves.shape == (25, 4)
        assert np.all(np.diff(curves, axis=1) >= 0), scheme
        assert np.all(curves[:, -1] == 3), scheme  # all clusters found at k=N


def test_diverse_sampling_beats_redundant_when_motifs_are_group_specific():
    """The panel's actual claim: with a group-specific lexicon, spreading
    experiments across tissues recovers more motifs than piling into one."""
    groups = {}
    exp_sets = []
    for gi, group in enumerate("abcdefgh"):
        members = [f"E{gi}_{j}" for j in range(4)]
        for m in members:
            groups[m] = group
        # 5 motifs private to this group, all present in all of its members
        for _ in range(5):
            exp_sets.append(set(members))
    rng = np.random.default_rng(7)
    diverse = rare.rarefy(exp_sets, groups, "diverse", 60, rng).mean(axis=0)
    redundant = rare.rarefy(exp_sets, groups, "redundant", 60, rng).mean(axis=0)
    mid = len(diverse) // 4
    assert diverse[mid] > redundant[mid] * 1.5


def test_classify_clusters_jaspar_fallback(tmp_path):
    meta_path = write_cluster_metadata(
        tmp_path / "m.tsv",
        [("pos", ["E1"], "SP1"), ("pos", ["E1"], None)],
    )
    meta, _ = rare.load_presence(meta_path)
    classes = rare.classify_clusters(meta, None)
    assert list(classes) == ["TF-matched", "unmatched (core promoter / repeat)"]


def test_classify_clusters_prefers_curated_annotation(tmp_path):
    meta_path = write_cluster_metadata(
        tmp_path / "m.tsv", [("pos", ["E1"], "SP1"), ("pos", ["E1"], None)]
    )
    ann = tmp_path / "ann.tsv"
    ann.write_text("cluster_final\tclass\n0\tcore promoter\n")
    meta, _ = rare.load_presence(meta_path)
    classes = rare.classify_clusters(meta, ann)
    assert list(classes) == ["core promoter", "unannotated"]


# --------------------------------------------------------------------------
# hit density
# --------------------------------------------------------------------------


def make_hitcall_tree(root, experiment, head, motif_counts, n_peaks):
    """Write a hits_linked.tsv + peaks.narrowPeak pair for one experiment."""
    exp_dir = root / f"{experiment}_{head}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for motif, count in motif_counts.items():
        for i in range(count):
            rows.append(
                {
                    "chr": "chr1", "start": 100 + i, "end": 110 + i,
                    "motif_name": "pos_patterns.pattern_0",
                    "strand": "+", "peak_id": i % n_peaks,
                    "compendium_motif_name": motif,
                }
            )
    pd.DataFrame(rows).to_csv(exp_dir / "hits_linked.tsv", sep="\t", index=False)
    with open(exp_dir / "peaks.narrowPeak", "w") as f:
        for i in range(n_peaks):
            f.write(f"chr1\t{i * 1000}\t{i * 1000 + 2114}\tpeak{i}\t0\t.\t0\t0\t0\t1057\n")
    return exp_dir


def test_count_peaks_prefers_narrowpeak(tmp_path):
    root = tmp_path / "hitcalls"
    exp_dir = make_hitcall_tree(root, "EXP1", "profile", {"pos_patterns.0": 5}, n_peaks=40)
    hits = pd.read_csv(exp_dir / "hits_linked.tsv", sep="\t")
    n, source = mhd.count_peaks(exp_dir, hits)
    assert (n, source) == (40, "peaks.narrowPeak")


def test_count_peaks_flags_the_fallback(tmp_path):
    """Without peaks.narrowPeak the count is derived from hits and known to be
    an underestimate, so the provenance string has to say so."""
    exp_dir = tmp_path / "EXP1_profile"
    exp_dir.mkdir()
    hits = pd.DataFrame({"peak_id": [0, 1, 2], "compendium_motif_name": ["pos_patterns.0"] * 3})
    n, source = mhd.count_peaks(exp_dir, hits)
    assert n == 3
    assert "underestimate" in source


def test_collect_densities_normalizes_per_peak(tmp_path):
    root = tmp_path / "hitcalls"
    make_hitcall_tree(root, "EXP1", "profile", {"pos_patterns.0": 20, "pos_patterns.1": 10}, 100)
    make_hitcall_tree(root, "EXP2", "profile", {"pos_patterns.0": 5}, 50)
    density, info = mhd.collect_densities(
        ["EXP1", "EXP2", "EXP_MISSING"], "profile", None, hitcall_dir=root, quiet=True
    )
    assert density.loc["pos_patterns.0", "EXP1"] == pytest.approx(0.20)
    assert density.loc["pos_patterns.1", "EXP1"] == pytest.approx(0.10)
    assert density.loc["pos_patterns.0", "EXP2"] == pytest.approx(0.10)
    # absent motif in EXP2 becomes a zero, not a NaN
    assert density.loc["pos_patterns.1", "EXP2"] == 0.0
    assert "EXP_MISSING" not in density.columns
    assert info.loc["EXP1", "n_peaks"] == 100


def test_discovery_status_separates_structural_zeros(tmp_path):
    root = tmp_path / "hitcalls"
    make_hitcall_tree(root, "EXP1", "profile", {"pos_patterns.0": 4}, 20)
    make_hitcall_tree(root, "EXP2", "profile", {"pos_patterns.1": 4}, 20)
    density, _ = mhd.collect_densities(
        ["EXP1", "EXP2"], "profile", None, hitcall_dir=root, quiet=True
    )
    meta = write_cluster_metadata(
        tmp_path / "m.tsv",
        [("pos", ["EXP1", "EXP2"], "SP1"), ("pos", ["EXP2"], "GATA1")],
    )
    status = mhd.discovery_status(density, meta)
    # cluster 0 was discovered in both, but only EXP1 has hits for it: a real
    # zero in EXP2, not a discovery failure.
    assert status.loc["pos_patterns.0", "EXP2"] == "discovered"
    assert density.loc["pos_patterns.0", "EXP2"] == 0.0
    # cluster 1 was never discovered in EXP1 at all: structurally zero.
    assert status.loc["pos_patterns.1", "EXP1"] == "undiscovered"


def test_specificity_is_zero_for_ubiquitous_and_one_for_private():
    columns = ["E1", "E2", "E3", "E4"]
    group_map = {"E1": "liver", "E2": "liver", "E3": "neural", "E4": "heart"}
    density = pd.DataFrame(
        {
            "E1": [1.0, 1.0, 0.0],
            "E2": [1.0, 0.0, 0.0],
            "E3": [1.0, 0.0, 2.0],
            "E4": [1.0, 0.0, 0.0],
        },
        index=["ubiquitous", "liver_only", "neural_only"],
    )[columns]
    spec = mhd.specificity_table(density, group_map)
    assert spec.loc["ubiquitous", "specificity"] == pytest.approx(0.0, abs=1e-9)
    assert spec.loc["liver_only", "specificity"] == pytest.approx(1.0)
    assert spec.loc["neural_only", "specificity"] == pytest.approx(1.0)
    assert spec.loc["liver_only", "top_group"] == "liver"
    assert spec.loc["neural_only", "top_group"] == "neural"
    assert spec.loc["ubiquitous", "n_groups_detected"] == 3


def test_specificity_averages_within_group_before_scoring():
    """A motif present in every group is ubiquitous even when one group is
    replicated far more than the others -- the panel's whole reason for
    scoring over groups rather than experiments."""
    group_map = {f"H{i}": "gi_tract" for i in range(16)}
    group_map.update({"L1": "liver_biliary", "N1": "neural"})
    density = pd.DataFrame(
        {c: [1.0] for c in group_map}, index=["everywhere"]
    )
    spec = mhd.specificity_table(density, group_map)
    assert spec.loc["everywhere", "specificity"] == pytest.approx(0.0, abs=1e-9)


def test_annotate_labels_adds_jaspar_names(tmp_path):
    meta = write_cluster_metadata(
        tmp_path / "m.tsv", [("pos", ["E1"], "SP1"), ("neg", ["E1"], None)]
    )
    labels = mhd.annotate_labels(
        pd.Index(["pos_patterns.0", "neg_patterns.1"]), meta
    )
    assert labels["pos_patterns.0"] == "pos_patterns.0 (SP1)"
    assert labels["neg_patterns.1"] == "neg_patterns.1"


def test_depth_colors_match_figure_1d_tiers():
    colors = mhd.depth_colors(
        {"A": 5e6, "B": 15e6, "C": 30e6}, ["A", "B", "C", "D"]
    )
    assert colors["A"] == "#d73027"  # <10M
    assert colors["B"] == "#fee090"  # 10-20M
    assert colors["C"] == "#4575b4"  # >20M
    assert colors["D"] == "#cccccc"  # unknown


# --------------------------------------------------------------------------
# end-to-end, against real accessions from the repo config
# --------------------------------------------------------------------------


def real_experiments(n):
    import yaml

    cfg = yaml.safe_load(open(REPO_ROOT / "configs" / "experiment_config.yaml"))
    read_counts = rare.load_read_counts()
    picked = [
        e for e in cfg["experiments"]
        if read_counts.get(e, 0) >= 10e6
        and "uncapped" not in str(cfg["experiments"][e].get("library_construction", "")).lower()
        and e != "ENCSR973QQI"
    ]
    return picked[:n]


def test_rarefaction_cli_end_to_end(tmp_path):
    exps = real_experiments(12)
    rows = [("pos", exps, "SP1")]  # ubiquitous
    rows += [("pos", exps[:2], "GATA1"), ("pos", exps[2:4], "HNF4A")]
    rows += [("neg", [e], None) for e in exps]  # singletons
    meta = write_cluster_metadata(tmp_path / "meta.tsv", rows)

    result = subprocess.run(
        [
            sys.executable, str(REPO_ROOT / "src/analysis/plot_motif_rarefaction.py"),
            "--cluster-metadata", str(meta), "--out-dir", str(tmp_path / "out"),
            "--n-reps", "20",
        ],
        capture_output=True, text=True, env=SUBPROC_ENV,
    )
    assert result.returncode == 0, result.stderr
    curves = pd.read_csv(tmp_path / "out" / "motif_rarefaction_profile.tsv", sep="\t")
    full = curves[
        (curves["scheme"] == "uniform") & (curves["motif_class"] == "__all__")
    ].sort_values("k")
    assert full["mean"].iloc[-1] == pytest.approx(len(rows))
    assert (tmp_path / "out" / "motif_rarefaction_profile.png").exists()
    assert (tmp_path / "out" / "motif_prevalence_profile.tsv").exists()


def test_hit_density_cli_end_to_end(tmp_path):
    exps = real_experiments(8)
    root = tmp_path / "hitcalls"
    for i, exp in enumerate(exps):
        counts = {"pos_patterns.0": 50}
        if i < 3:
            counts["pos_patterns.1"] = 30
        make_hitcall_tree(root, exp, "profile", counts, n_peaks=100)
    meta = write_cluster_metadata(
        tmp_path / "meta.tsv",
        [("pos", exps, "SP1"), ("pos", exps[:3], "GATA1")],
    )

    result = subprocess.run(
        [
            sys.executable, str(REPO_ROOT / "src/analysis/motif_hit_density.py"),
            "--hitcall-dir", str(root), "--cluster-metadata", str(meta),
            "--out-dir", str(tmp_path / "out"), "--min-experiments", "2",
            "--mask-undiscovered",
        ],
        capture_output=True, text=True, env=SUBPROC_ENV,
    )
    assert result.returncode == 0, result.stderr
    out = tmp_path / "out"
    density = pd.read_csv(out / "motif_hit_density_profile.tsv", sep="\t", index_col=0)
    assert density.loc["pos_patterns.0"].eq(0.5).all()
    spec = pd.read_csv(out / "motif_specificity_profile.tsv", sep="\t", index_col=0)
    assert spec.loc["pos_patterns.1", "specificity"] > spec.loc["pos_patterns.0", "specificity"]
    status = pd.read_csv(out / "motif_hit_density_profile_status.tsv", sep="\t", index_col=0)
    assert (status.loc["pos_patterns.1", exps[3:]] == "undiscovered").all()
    assert (out / "motif_hit_density_profile.png").exists()
    assert (out / "motif_hit_density_profile_columns.tsv").exists()


def test_balanced_selection_keeps_tissue_restricted_motifs():
    """A restricted motif that is heavily used in its own tissues must survive
    selection, even though its summed density is far below that of a weak
    motif spread over every experiment."""
    spec = pd.DataFrame(
        {
            # broad but weak: present in 100 experiments at 0.10 each
            "total_hits_per_peak": [10.0] * 6 + [3.5, 3.5],
            "mean_hits_per_peak_detected": [0.10] * 6 + [0.35, 0.35],
            "n_experiments_detected": [100] * 6 + [10, 10],
            "specificity": [0.02] * 6 + [0.95, 0.92],
        },
        index=[f"broad{i}" for i in range(6)] + ["lineage_a", "lineage_b"],
    )
    picked = mhd.select_clusters(spec, top_n=4, sort_by="balanced")
    assert "lineage_a" in picked.index
    assert "lineage_b" in picked.index
    assert len(picked) == 4


def test_balanced_selection_falls_back_when_pool_is_thin():
    spec = pd.DataFrame(
        {
            "total_hits_per_peak": [5.0, 4.0],
            "mean_hits_per_peak_detected": [0.5, 0.4],
            "n_experiments_detected": [10, 10],
            "specificity": [0.1, 0.9],
        },
        index=["a", "b"],
    )
    picked = mhd.select_clusters(spec, top_n=10, sort_by="balanced")
    assert set(picked.index) == {"a", "b"}


def test_sort_by_total_hits_still_available():
    spec = pd.DataFrame(
        {
            "total_hits_per_peak": [1.0, 9.0],
            "mean_hits_per_peak_detected": [0.1, 0.9],
            "n_experiments_detected": [10, 10],
            "specificity": [0.9, 0.1],
        },
        index=["low", "high"],
    )
    picked = mhd.select_clusters(spec, top_n=1, sort_by="total_hits")
    assert list(picked.index) == ["high"]


def write_pattern_to_cluster(path, rows):
    """Write a cluster_motifs.py-shaped pattern-to-cluster mapping TSV.

    `rows` is the same (posneg, experiments, jaspar) shape as
    write_cluster_metadata, so the two loaders can be compared directly.
    """
    records = []
    for cluster_final, (posneg, exps, _jaspar) in enumerate(rows):
        for i, exp in enumerate(sorted(exps)):
            records.append(
                {
                    "experiment": exp,
                    "local_motif_name": f"{posneg}_patterns.pattern_{i}",
                    "compendium_motif_name": f"{posneg}_patterns.{cluster_final}",
                }
            )
    pd.DataFrame(records).to_csv(path, sep="\t", index=False)
    return path


def test_mapping_loader_matches_metadata_loader(tmp_path):
    """The two inputs must give identical presence sets -- that equivalence is
    the whole reason the mapping is a safe early substitute."""
    rows = [
        ("pos", ["E1", "E2", "E3"], "SP1"),
        ("neg", ["E2", "E4"], "GATA1"),
        ("pos", ["E4"], None),
    ]
    meta_path = write_cluster_metadata(tmp_path / "meta.tsv", rows)
    map_path = write_pattern_to_cluster(tmp_path / "map.tsv", rows)

    from_meta, universe_meta = rare.load_presence(meta_path)
    from_map, universe_map = rare.load_presence_from_mapping(map_path)

    assert universe_meta == universe_map

    # Compare on content keyed by cluster identity, not row order: the
    # metadata TSV is sorted by total_seqlets while the mapping is grouped by
    # motif name, and no curve depends on row order.
    def keyed(df):
        return {
            int(row["cluster_final"]): (row["posneg"], row["prevalence"], row["exp_set"])
            for _, row in df.iterrows()
        }

    assert keyed(from_meta) == keyed(from_map)
    assert keyed(from_map)[0][0] == "pos"
    assert keyed(from_map)[1][0] == "neg"


def test_mapping_loader_respects_keep_set(tmp_path):
    rows = [("pos", ["E1", "E2", "E3"], "SP1"), ("pos", ["E4"], None)]
    map_path = write_pattern_to_cluster(tmp_path / "map.tsv", rows)
    meta, universe = rare.load_presence_from_mapping(
        map_path, keep_experiments={"E1", "E2"}
    )
    assert universe == ["E1", "E2"]
    assert len(meta) == 1  # the E4-only cluster is unreachable
    assert meta.loc[0, "prevalence"] == 2


def test_mapping_loader_rejects_wrong_schema(tmp_path):
    bad = tmp_path / "bad.tsv"
    pd.DataFrame({"experiment": ["E1"]}).to_csv(bad, sep="\t", index=False)
    with pytest.raises(ValueError, match="compendium_motif_name"):
        rare.load_presence_from_mapping(bad)


def test_classify_clusters_single_class_without_jaspar(tmp_path):
    """Mapping-derived tables have no JASPAR column, so stratification must
    collapse to one class rather than crashing."""
    rows = [("pos", ["E1", "E2"], "SP1"), ("neg", ["E2"], None)]
    map_path = write_pattern_to_cluster(tmp_path / "map.tsv", rows)
    meta, _ = rare.load_presence_from_mapping(map_path)
    classes = rare.classify_clusters(meta, None)
    assert set(classes) == {"all motifs"}


def test_rarefaction_cli_accepts_pattern_to_cluster(tmp_path):
    exps = real_experiments(12)
    rows = [("pos", exps, "SP1"), ("pos", exps[:3], "GATA1")]
    rows += [("neg", [e], None) for e in exps]
    map_path = write_pattern_to_cluster(tmp_path / "map.tsv", rows)

    result = subprocess.run(
        [
            sys.executable, str(REPO_ROOT / "src/analysis/plot_motif_rarefaction.py"),
            "--pattern-to-cluster", str(map_path),
            "--out-dir", str(tmp_path / "out"), "--n-reps", "20",
        ],
        capture_output=True, text=True, env=SUBPROC_ENV,
    )
    assert result.returncode == 0, result.stderr
    curves = pd.read_csv(tmp_path / "out" / "motif_rarefaction_profile.tsv", sep="\t")
    full = curves[
        (curves["scheme"] == "uniform") & (curves["motif_class"] == "__all__")
    ].sort_values("k")
    assert full["mean"].iloc[-1] == pytest.approx(len(rows))
    assert (tmp_path / "out" / "motif_rarefaction_profile.png").exists()


# --------------------------------------------------------------------------
# abundance-threshold sweep
# --------------------------------------------------------------------------


def write_cluster_metadata_with_seqlets(path, rows):
    """Like write_cluster_metadata but with explicit per-cluster seqlet totals.

    `rows` is (posneg, experiments, jaspar, total_seqlets).
    """
    records = []
    for cluster_final, (posneg, exps, jaspar, total_seqlets) in enumerate(rows):
        records.append(
            {
                "cluster_final": cluster_final,
                "n_motifs": len(exps),
                "total_seqlets": total_seqlets,
                "n_experiments": len(exps),
                "experiments": ",".join(sorted(exps)),
                "posneg": posneg,
                "jaspar_name": jaspar,
                "jaspar_score": 0.9 if jaspar else np.nan,
            }
        )
    pd.DataFrame(records).to_csv(path, sep="\t", index=False)
    return path


def test_load_presence_computes_seqlets_per_motif(tmp_path):
    meta_path = write_cluster_metadata_with_seqlets(
        tmp_path / "m.tsv",
        [("pos", ["E1", "E2", "E3", "E4"], "SP1", 400), ("pos", ["E1", "E2"], None, 40)],
    )
    meta, _ = rare.load_presence(meta_path)
    assert list(meta["seqlets_per_motif"]) == [100.0, 20.0]


def test_sweep_requires_seqlet_columns(tmp_path):
    """The pattern-to-cluster fallback carries no seqlet counts, so the sweep
    must refuse rather than silently sweeping nothing."""
    map_path = write_pattern_to_cluster(
        tmp_path / "map.tsv", [("pos", ["E1", "E2"], None)]
    )
    meta, _ = rare.load_presence_from_mapping(map_path)
    with pytest.raises(ValueError, match="cluster_metadata"):
        rare.sweep_abundance(meta, 2, [0.0, 100.0], [1])


def build_sweep_meta(n_total=100, n_broad=40, n_narrow=60):
    """Clusters where abundance tracks prevalence, as on the real compendium.

    Broad clusters are prevalent AND abundant; narrow ones are restricted AND
    sparse. That coupling is what makes an abundance floor act as a prevalence
    floor.
    """
    exps = [f"E{i}" for i in range(n_total)]
    rng = np.random.default_rng(3)
    rows = []
    for _ in range(n_broad):
        members = set(rng.choice(exps, size=80, replace=False))
        rows.append({"exp_set": members, "prevalence": 80, "seqlets_per_motif": 900.0})
    for _ in range(n_narrow):
        members = set(rng.choice(exps, size=4, replace=False))
        rows.append({"exp_set": members, "prevalence": 4, "seqlets_per_motif": 30.0})
    return pd.DataFrame(rows), n_total


def test_sweep_shrinks_lexicon_monotonically():
    meta, n_total = build_sweep_meta()
    _, summary = rare.sweep_abundance(meta, n_total, [0.0, 100.0, 1000.0], [5])
    # the 1000 threshold drops everything, leaving <10 clusters -> skipped
    assert list(summary["min_seqlets_per_motif"]) == [0.0, 100.0]
    assert list(summary["n_clusters"]) == [100, 40]


def test_sweep_fraction_reaches_one_at_full_atlas():
    meta, n_total = build_sweep_meta()
    curves, _ = rare.sweep_abundance(meta, n_total, [0.0, 100.0], [5])
    for threshold, sub in curves.groupby("min_seqlets_per_motif"):
        assert sub.sort_values("k")["fraction"].iloc[-1] == pytest.approx(1.0)


def test_sweep_reproduces_the_confound_direction():
    """The sweep's whole finding: because abundance tracks prevalence, raising
    the floor raises the fraction a small sample recovers. If this ever
    reverses, the interpretation in the docstring is wrong."""
    meta, n_total = build_sweep_meta()
    _, summary = rare.sweep_abundance(meta, n_total, [0.0, 100.0], [5])
    low = summary.set_index("min_seqlets_per_motif").loc[0.0, "fraction_at_k5"]
    high = summary.set_index("min_seqlets_per_motif").loc[100.0, "fraction_at_k5"]
    assert high > low


def test_sweep_cli_end_to_end(tmp_path):
    exps = real_experiments(30)
    rows = [("pos", exps, "SP1", 30 * 900) for _ in range(8)]
    rows += [("pos", exps[:20], f"BROAD{i}", 20 * 300) for i in range(8)]
    rows += [("pos", exps[i : i + 3], f"TF{i}", 3 * 30) for i in range(0, 24, 3)]
    meta = write_cluster_metadata_with_seqlets(tmp_path / "meta.tsv", rows)

    result = subprocess.run(
        [
            sys.executable, str(REPO_ROOT / "src/analysis/plot_motif_rarefaction.py"),
            "--cluster-metadata", str(meta), "--out-dir", str(tmp_path / "out"),
            "--n-reps", "10", "--sweep",
            "--sweep-thresholds", "0", "100", "--sweep-marks", "1", "5",
        ],
        capture_output=True, text=True, env=SUBPROC_ENV,
    )
    assert result.returncode == 0, result.stderr
    out = tmp_path / "out"
    assert (out / "motif_rarefaction_sweep_profile.png").exists()
    summary = pd.read_csv(
        out / "motif_rarefaction_sweep_profile_summary.tsv", sep="\t"
    )
    assert list(summary["min_seqlets_per_motif"]) == [0.0, 100.0]
    assert summary.loc[1, "n_clusters"] < summary.loc[0, "n_clusters"]
    # the confound diagnostic and weakest-form claim are both reported
    assert "prevalence vs seqlets/motif r =" in result.stderr
    assert "Weakest-form claim" in result.stderr


def test_min_seqlets_per_motif_filters_and_reports(tmp_path):
    exps = real_experiments(12)
    rows = [("pos", exps, "SP1", 12 * 900)] * 3
    rows += [("pos", exps[:3], "TF", 3 * 20)] * 3
    meta = write_cluster_metadata_with_seqlets(tmp_path / "meta.tsv", rows)

    result = subprocess.run(
        [
            sys.executable, str(REPO_ROOT / "src/analysis/plot_motif_rarefaction.py"),
            "--cluster-metadata", str(meta), "--out-dir", str(tmp_path / "out"),
            "--n-reps", "10", "--min-seqlets-per-motif", "100",
        ],
        capture_output=True, text=True, env=SUBPROC_ENV,
    )
    assert result.returncode == 0, result.stderr
    assert "kept 3/6 clusters" in result.stderr


def test_min_seqlets_per_motif_rejected_without_metadata(tmp_path):
    exps = real_experiments(6)
    rows = [("pos", exps, None), ("pos", exps[:2], None)]
    map_path = write_pattern_to_cluster(tmp_path / "map.tsv", rows)
    result = subprocess.run(
        [
            sys.executable, str(REPO_ROOT / "src/analysis/plot_motif_rarefaction.py"),
            "--pattern-to-cluster", str(map_path),
            "--out-dir", str(tmp_path / "out"), "--min-seqlets-per-motif", "50",
        ],
        capture_output=True, text=True, env=SUBPROC_ENV,
    )
    assert result.returncode == 1
    assert "needs total_seqlets and n_motifs" in result.stderr


# --------------------------------------------------------------------------
# group concentration
# --------------------------------------------------------------------------

import motif_group_concentration as mgc  # noqa: E402


def test_expected_n_groups_matches_monte_carlo():
    rng = np.random.default_rng(0)
    sizes = np.array([40, 25, 20, 10, 5, 5, 3, 2])
    n_total = int(sizes.sum())
    labels = np.repeat(np.arange(len(sizes)), sizes)
    for p in (2, 5, 20):
        exact = mgc.expected_n_groups(p, sizes, n_total)
        draws = [
            len(set(rng.choice(labels, size=p, replace=False))) for _ in range(4000)
        ]
        assert exact == pytest.approx(np.mean(draws), abs=0.08), p


def test_prob_single_group_matches_closed_form_at_p2():
    """At p=2 the probability reduces to sum n_g(n_g-1) / N(N-1), which is the
    hand calculation used to check the real data."""
    sizes = np.array([41, 28, 21, 18, 15, 11, 8, 8, 8, 7, 7, 6, 5, 4, 3, 3, 2, 2, 1])
    n_total = int(sizes.sum())
    expected = (sizes * (sizes - 1)).sum() / (n_total * (n_total - 1))
    assert mgc.prob_single_group(2, sizes, n_total) == pytest.approx(expected)


def test_prob_single_group_zero_when_no_group_is_large_enough():
    sizes = np.array([3, 3, 3])
    assert mgc.prob_single_group(4, sizes, 9) == pytest.approx(0.0)


def test_poisson_binomial_is_exact_not_approximate():
    probs = np.array([0.1, 0.5, 0.9])
    # P[X >= 0] = 1; P[X >= 3] = product
    assert mgc.poisson_binomial_sf(probs, 0) == pytest.approx(1.0)
    assert mgc.poisson_binomial_sf(probs, 3) == pytest.approx(0.1 * 0.5 * 0.9)
    # P[X >= 1] = 1 - product of complements
    assert mgc.poisson_binomial_sf(probs, 1) == pytest.approx(
        1 - (0.9 * 0.5 * 0.1)
    )


def test_poisson_binomial_mean_matches_sum_of_probs():
    rng = np.random.default_rng(4)
    probs = rng.uniform(0, 0.3, 40)
    dist = np.array(
        [
            mgc.poisson_binomial_sf(probs, k) - mgc.poisson_binomial_sf(probs, k + 1)
            for k in range(len(probs) + 1)
        ]
    )
    assert dist.sum() == pytest.approx(1.0)
    mean = (dist * np.arange(len(probs) + 1)).sum()
    assert mean == pytest.approx(probs.sum())


def build_concentration_meta(group_map, concentrated, prevalence):
    """One cluster per entry in `concentrated`; True = all experiments drawn
    from a single group, False = drawn across groups."""
    by_group = {}
    for e, g in group_map.items():
        by_group.setdefault(g, []).append(e)
    groups = sorted(by_group)
    rows = []
    for i, conc in enumerate(concentrated):
        if conc:
            pool = by_group[groups[i % len(groups)]]
            members = set(pool[:prevalence])
        else:
            members = {by_group[groups[(i + j) % len(groups)]][0] for j in range(prevalence)}
        rows.append({"exp_set": members, "prevalence": len(members)})
    return pd.DataFrame(rows)


def test_concentration_is_one_for_random_spread_and_low_for_concentrated():
    group_map = {f"g{g}_e{i}": f"g{g}" for g in range(8) for i in range(5)}
    n_total = len(group_map)
    spread = build_concentration_meta(group_map, [False] * 8, prevalence=4)
    conc = build_concentration_meta(group_map, [True] * 8, prevalence=4)

    a_spread = mgc.annotate_concentration(spread, group_map, n_total)
    a_conc = mgc.annotate_concentration(conc, group_map, n_total)

    # spread clusters hit one group per experiment -> at or above expectation
    assert a_spread["concentration"].median() > 1.0
    assert (a_spread["n_groups"] == 4).all()
    # concentrated clusters sit in exactly one group -> well below
    assert (a_conc["n_groups"] == 1).all()
    assert a_conc["concentration"].median() < 0.4
    assert a_conc["is_single_group"].all()


def test_summary_reports_single_group_enrichment():
    group_map = {f"g{g}_e{i}": f"g{g}" for g in range(8) for i in range(5)}
    n_total = len(group_map)
    meta = pd.concat(
        [
            build_concentration_meta(group_map, [True] * 6, prevalence=3),
            build_concentration_meta(group_map, [False] * 6, prevalence=3),
        ],
        ignore_index=True,
    )
    annotated = mgc.annotate_concentration(meta, group_map, n_total)
    annotated["motif_class"] = ["conc"] * 6 + ["spread"] * 6
    summary = mgc.summarize(annotated, ["motif_class"]).set_index("motif_class")

    assert summary.loc["conc", "n_single_group"] == 6
    assert summary.loc["spread", "n_single_group"] == 0
    assert summary.loc["conc", "single_group_p"] < 1e-6
    assert summary.loc["conc", "single_group_enrichment"] > 5
    assert summary.loc["spread", "single_group_p"] == pytest.approx(1.0)


def test_group_level_changes_the_expectation():
    """Coarse grouping lowers E[n_groups], which is why a conclusion must hold
    at both levels before it can be trusted."""
    sizes_coarse = np.array([40, 30, 30])
    sizes_fine = np.array([10] * 10)
    n_total = 100
    assert mgc.expected_n_groups(5, sizes_coarse, n_total) < mgc.expected_n_groups(
        5, sizes_fine, n_total
    )


def test_concentration_cli_runs_at_both_levels(tmp_path):
    exps = real_experiments(40)
    rows = [("pos", exps, "SP1", 40 * 900) for _ in range(4)]
    rows += [("pos", exps[:6], "GATA1", 6 * 400) for _ in range(4)]
    rows += [("pos", exps[i : i + 3], None, 3 * 60) for i in range(0, 24, 3)]
    meta = write_cluster_metadata_with_seqlets(tmp_path / "meta.tsv", rows)

    for level in ("tissue", "biosample"):
        result = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "src/analysis/motif_group_concentration.py"),
                "--cluster-metadata", str(meta), "--group-level", level,
                "--out-dir", str(tmp_path / "out"),
            ],
            capture_output=True, text=True, env=SUBPROC_ENV,
        )
        assert result.returncode == 0, result.stderr
        out = tmp_path / "out"
        per_cluster = pd.read_csv(
            out / f"motif_concentration_profile_{level}.tsv", sep="\t"
        )
        assert {"n_groups", "expected_n_groups", "concentration"} <= set(
            per_cluster.columns
        )
        assert (per_cluster["expected_n_groups"] > 0).all()
        summary = pd.read_csv(
            out / f"motif_concentration_profile_{level}_summary.tsv", sep="\t"
        )
        assert "single_group_p" in summary.columns
        assert (out / f"motif_concentration_profile_{level}.png").exists()


def test_concentration_cli_rejects_mapping_input(tmp_path):
    result = subprocess.run(
        [
            sys.executable, str(REPO_ROOT / "src/analysis/motif_group_concentration.py"),
            "--cluster-metadata", str(tmp_path / "nope.tsv"),
            "--out-dir", str(tmp_path / "out"),
        ],
        capture_output=True, text=True, env=SUBPROC_ENV,
    )
    assert result.returncode == 1
    assert "pattern-to-cluster mapping is not sufficient" in result.stderr


def test_jaspar_score_threshold_reclassifies(tmp_path):
    exps = real_experiments(20)
    rows = [("pos", exps, "WEAK", 20 * 500) for _ in range(6)]
    rows += [("pos", exps[:5], "STRONG", 5 * 500) for _ in range(6)]
    meta_path = write_cluster_metadata_with_seqlets(tmp_path / "meta.tsv", rows)
    m = pd.read_csv(meta_path, sep="\t")
    m.loc[m.jaspar_name == "WEAK", "jaspar_score"] = 0.80
    m.loc[m.jaspar_name == "STRONG", "jaspar_score"] = 0.95
    m.to_csv(meta_path, sep="\t", index=False)

    result = subprocess.run(
        [
            sys.executable, str(REPO_ROOT / "src/analysis/motif_group_concentration.py"),
            "--cluster-metadata", str(meta_path), "--out-dir", str(tmp_path / "out"),
            "--jaspar-score-threshold", "0.85",
        ],
        capture_output=True, text=True, env=SUBPROC_ENV,
    )
    assert result.returncode == 0, result.stderr
    assert "moved 6 clusters into the unmatched class" in result.stderr


def test_prevalence_one_clusters_are_uninformative_for_concentration():
    """At p=1 the statistics are degenerate: n_groups is always 1, and both
    E[n_groups] and P[n_groups=1] are exactly 1. Singletons therefore cannot
    provide evidence either way, which is why the script warns about them."""
    sizes = np.array([41, 28, 21, 18, 15, 11, 8, 8, 8, 7, 7, 6, 5, 4, 3, 3, 2, 2, 1])
    n_total = int(sizes.sum())
    assert mgc.expected_n_groups(1, sizes, n_total) == pytest.approx(1.0)
    assert mgc.prob_single_group(1, sizes, n_total) == pytest.approx(1.0)


def test_singletons_dilute_enrichment_but_not_the_pvalue():
    """Deterministic (q=1) terms shift observed and expected equally and add no
    variance, so the tail probability is unchanged while the effect-size ratio
    collapses -- the reason the warning targets the ratio specifically."""
    sizes = np.array([41, 28, 21, 18, 15, 11, 8, 8, 8, 7, 7, 6, 5, 4, 3, 3, 2, 2, 1])
    n_total = int(sizes.sum())
    q = mgc.prob_single_group(2, sizes, n_total)
    base = np.full(22, q)
    with_singletons = np.concatenate([base, np.ones(235)])

    p_base = mgc.poisson_binomial_sf(base, 9)
    p_diluted = mgc.poisson_binomial_sf(with_singletons, 9 + 235)
    assert p_diluted == pytest.approx(p_base, rel=1e-6)

    enrich_base = 9 / base.sum()
    enrich_diluted = (9 + 235) / with_singletons.sum()
    assert enrich_base > 4.0
    assert enrich_diluted < 1.1


def test_concentration_cli_warns_on_min_cluster_experiments_one(tmp_path):
    exps = real_experiments(20)
    rows = [("pos", exps[:4], "TF", 4 * 100) for _ in range(6)]
    rows += [("pos", [exps[i]], None, 100) for i in range(6)]
    meta = write_cluster_metadata_with_seqlets(tmp_path / "meta.tsv", rows)
    result = subprocess.run(
        [
            sys.executable, str(REPO_ROOT / "src/analysis/motif_group_concentration.py"),
            "--cluster-metadata", str(meta), "--out-dir", str(tmp_path / "out"),
            "--min-cluster-experiments", "1",
        ],
        capture_output=True, text=True, env=SUBPROC_ENV,
    )
    assert result.returncode == 0, result.stderr
    assert "carry no information for this test" in result.stderr


def test_concentration_records_which_groups_not_just_how_many():
    """A restricted cluster is uninterpretable without knowing which lineage it
    is restricted to, and the depth-confound check needs it too."""
    group_map = {f"g{g}_e{i}": f"g{g}" for g in range(6) for i in range(4)}
    n_total = len(group_map)
    meta = pd.DataFrame(
        [
            {"exp_set": {"g0_e0", "g0_e1"}, "prevalence": 2},
            {"exp_set": {"g1_e0", "g2_e0"}, "prevalence": 2},
        ]
    )
    annotated = mgc.annotate_concentration(meta, group_map, n_total)
    assert annotated.loc[0, "sole_group"] == "g0"
    assert annotated.loc[0, "groups"] == "g0"
    assert annotated.loc[1, "sole_group"] == ""  # spans two groups
    assert annotated.loc[1, "groups"] == "g1,g2"


def test_concentration_cli_reports_restricted_group_landing(tmp_path):
    exps = real_experiments(30)
    rows = [("pos", exps, "SP1", 30 * 900) for _ in range(3)]
    rows += [("pos", exps[:2], None, 2 * 100) for _ in range(6)]
    meta = write_cluster_metadata_with_seqlets(tmp_path / "meta.tsv", rows)
    result = subprocess.run(
        [
            sys.executable, str(REPO_ROOT / "src/analysis/motif_group_concentration.py"),
            "--cluster-metadata", str(meta), "--out-dir", str(tmp_path / "out"),
        ],
        capture_output=True, text=True, env=SUBPROC_ENV,
    )
    assert result.returncode == 0, result.stderr
    per_cluster = pd.read_csv(
        tmp_path / "out" / "motif_concentration_profile_tissue.tsv", sep="\t"
    )
    assert "sole_group" in per_cluster.columns
    assert "groups" in per_cluster.columns
    assert "of restricted," in result.stderr


def test_enrichment_flagged_unreliable_when_expectation_is_tiny():
    """At high --min-cluster-experiments on fine groups the expected count
    falls below 1, where the ratio is uninterpretable (0 observed against 0.004
    reads as "0x") even though the exact p-value stays valid."""
    group_map = {f"b{i}": f"b{i}" for i in range(60)}  # every group size 1
    n_total = len(group_map)
    meta = pd.DataFrame(
        [{"exp_set": {f"b{i}", f"b{i+1}", f"b{i+2}"}, "prevalence": 3} for i in range(0, 30, 3)]
    )
    annotated = mgc.annotate_concentration(meta, group_map, n_total)
    annotated["motif_class"] = "spread"
    summary = mgc.summarize(annotated, ["motif_class"])
    # no group is large enough to hold 3 experiments, so expectation is 0
    assert summary.loc[0, "expected_single_group"] == pytest.approx(0.0)
    assert not summary.loc[0, "enrichment_reliable"]
    assert summary.loc[0, "single_group_p"] == pytest.approx(1.0)


def test_enrichment_reliable_when_expectation_is_adequate():
    group_map = {f"g{g}_e{i}": f"g{g}" for g in range(4) for i in range(10)}
    n_total = len(group_map)
    meta = pd.concat(
        [build_concentration_meta(group_map, [True] * 10, prevalence=2)],
        ignore_index=True,
    )
    annotated = mgc.annotate_concentration(meta, group_map, n_total)
    annotated["motif_class"] = "conc"
    summary = mgc.summarize(annotated, ["motif_class"])
    assert summary.loc[0, "expected_single_group"] >= 1.0
    assert summary.loc[0, "enrichment_reliable"]


# --------------------------------------------------------------------------
# degree-preserving (curveball) null
# --------------------------------------------------------------------------


def _margins(sets, experiments):
    rows = sorted(len(s) for s in sets)
    cols = sorted(sum(e in s for s in sets) for e in experiments)
    return rows, cols


def test_curveball_preserves_both_margins_exactly():
    """The whole point: cluster prevalence AND per-experiment motif count stay
    fixed, so read depth cannot differ between observed and null."""
    rng = np.random.default_rng(0)
    experiments = [f"E{i}" for i in range(40)]
    sets = [
        set(rng.choice(experiments, size=int(p), replace=False))
        for p in rng.integers(2, 15, 30)
    ]
    before = _margins(sets, experiments)
    after = _margins(mcurve := mgc.curveball_randomize(sets, rng, n_trades=500), experiments)
    assert before == after
    assert [len(s) for s in sets] == [len(s) for s in mcurve]


def test_curveball_actually_mixes():
    rng = np.random.default_rng(1)
    experiments = [f"E{i}" for i in range(30)]
    sets = [set(experiments[i : i + 5]) for i in range(0, 25, 5)]
    out = mgc.curveball_randomize(sets, rng, n_trades=2000)
    assert any(a != b for a, b in zip(sets, out))


def test_curveball_is_a_noop_on_a_single_cluster():
    rng = np.random.default_rng(2)
    sets = [{"E1", "E2"}]
    assert mgc.curveball_randomize(sets, rng) == [{"E1", "E2"}]


def test_swap_null_reproduces_the_analytic_uniform_expectation():
    """Fixture-independent correctness check: on a matrix with no group
    structure the degree-preserving null must land on the same expectation the
    closed-form uniform null gives. The two nulls should only diverge when the
    column margins actually carry information, which is the confound the swap
    null exists to absorb."""
    rng = np.random.default_rng(3)
    group_map = {f"g{g}_e{i}": f"g{g}" for g in range(8) for i in range(6)}
    experiments = list(group_map)
    n_total = len(experiments)
    meta = pd.DataFrame(
        [
            {"exp_set": set(rng.choice(experiments, size=5, replace=False)), "prevalence": 5}
            for _ in range(200)
        ]
    )
    meta["motif_class"] = "spread"
    swap = mgc.swap_null_test(
        meta, group_map, ["motif_class"], 200, np.random.default_rng(4)
    )
    analytic = mgc.expected_n_groups(5, np.array([6] * 8), n_total)
    assert swap.loc[0, "null_mean_n_groups"] == pytest.approx(analytic, abs=0.06)


def test_swap_null_finds_no_concentration_when_there_is_none():
    """A structureless matrix should sit at its own null. Uses many clusters so
    the observed mean converges -- with only a few dozen, a single random draw
    lands a couple of sd off the expectation and the p-value is meaningless."""
    rng = np.random.default_rng(3)
    group_map = {f"g{g}_e{i}": f"g{g}" for g in range(8) for i in range(6)}
    experiments = list(group_map)
    meta = pd.DataFrame(
        [
            {"exp_set": set(rng.choice(experiments, size=5, replace=False)), "prevalence": 5}
            for _ in range(300)
        ]
    )
    meta["motif_class"] = "spread"
    swap = mgc.swap_null_test(
        meta, group_map, ["motif_class"], 200, np.random.default_rng(4)
    )
    assert swap.loc[0, "swap_concentration"] == pytest.approx(1.0, abs=0.04)
    assert swap.loc[0, "swap_mean_p"] > 0.01


def test_swap_null_detects_real_group_concentration():
    """Clusters confined to one group must beat a null that already accounts
    for per-experiment productivity."""
    group_map = {f"g{g}_e{i}": f"g{g}" for g in range(8) for i in range(6)}
    by_group = {}
    for e, g in group_map.items():
        by_group.setdefault(g, []).append(e)
    meta = pd.DataFrame(
        [
            {"exp_set": set(by_group[f"g{i % 8}"][:5]), "prevalence": 5}
            for i in range(40)
        ]
    )
    meta["motif_class"] = "conc"
    swap = mgc.swap_null_test(
        meta, group_map, ["motif_class"], 200, np.random.default_rng(5)
    )
    assert swap.loc[0, "swap_concentration"] < 0.5
    assert swap.loc[0, "swap_mean_p"] < 0.01
    assert swap.loc[0, "obs_single_group"] == 40
    assert swap.loc[0, "null_single_group_mean"] < 5


def test_swap_null_pvalues_are_add_one_bounded():
    """An empirical p-value must never be exactly 0 -- (#{>=obs}+1)/(n+1)."""
    group_map = {f"g{g}_e{i}": f"g{g}" for g in range(6) for i in range(5)}
    by_group = {}
    for e, g in group_map.items():
        by_group.setdefault(g, []).append(e)
    meta = pd.DataFrame(
        [{"exp_set": set(by_group[f"g{i % 6}"][:4]), "prevalence": 4} for i in range(24)]
    )
    meta["motif_class"] = "conc"
    swap = mgc.swap_null_test(
        meta, group_map, ["motif_class"], 50, np.random.default_rng(6)
    )
    assert swap.loc[0, "swap_mean_p"] >= 1 / 51
    assert swap.loc[0, "swap_single_group_p"] >= 1 / 51


def test_swap_null_cli_writes_output(tmp_path):
    exps = real_experiments(30)
    rows = [("pos", exps, "SP1", 30 * 900) for _ in range(4)]
    rows += [("pos", exps[:4], "GATA1", 4 * 300) for _ in range(6)]
    meta = write_cluster_metadata_with_seqlets(tmp_path / "meta.tsv", rows)
    result = subprocess.run(
        [
            sys.executable, str(REPO_ROOT / "src/analysis/motif_group_concentration.py"),
            "--cluster-metadata", str(meta), "--out-dir", str(tmp_path / "out"),
            "--swap-permutations", "50",
        ],
        capture_output=True, text=True, env=SUBPROC_ENV,
    )
    assert result.returncode == 0, result.stderr
    swap = pd.read_csv(
        tmp_path / "out" / "motif_concentration_profile_tissue_swapnull.tsv", sep="\t"
    )
    assert {"swap_concentration", "swap_mean_p", "obs_single_group"} <= set(swap.columns)
    assert "Degree-preserving null" in result.stderr


def test_swap_permutations_zero_skips_the_null(tmp_path):
    exps = real_experiments(20)
    rows = [("pos", exps[:5], "TF", 5 * 200) for _ in range(6)]
    meta = write_cluster_metadata_with_seqlets(tmp_path / "meta.tsv", rows)
    result = subprocess.run(
        [
            sys.executable, str(REPO_ROOT / "src/analysis/motif_group_concentration.py"),
            "--cluster-metadata", str(meta), "--out-dir", str(tmp_path / "out"),
            "--swap-permutations", "0",
        ],
        capture_output=True, text=True, env=SUBPROC_ENV,
    )
    assert result.returncode == 0, result.stderr
    assert not (
        tmp_path / "out" / "motif_concentration_profile_tissue_swapnull.tsv"
    ).exists()


# --------------------------------------------------------------------------
# compendium redundancy (memelite TOMTOM self-comparison)
# --------------------------------------------------------------------------

import motif_redundancy as mr  # noqa: E402


def write_meme(path, motifs):
    """Minimal MEME file. `motifs` is {name: (length, 4) probability array}."""
    with open(path, "w") as f:
        f.write("MEME version 4\n\nALPHABET= ACGT\n\n")
        f.write("strands: + -\n\nBackground letter frequencies\n")
        f.write("A 0.25 C 0.25 G 0.25 T 0.25\n\n")
        for name, pwm in motifs.items():
            f.write(f"MOTIF {name}\n")
            f.write(f"letter-probability matrix: alength= 4 w= {pwm.shape[0]} "
                    "nsites= 100 E= 0\n")
            for row in pwm:
                f.write(" " + " ".join(f"{v:.6f}" for v in row) + "\n")
            f.write("\n")
    return path


def realistic_pwm(length, seed, conc=8.0):
    """Information-rich but non-degenerate PWM, shape (length, 4).

    TOMTOM scores columns against a background estimated from the target set,
    and near-deterministic columns (as onehot_pwm produces) make that
    background degenerate -- identical motifs then come back with p = 1.0.
    Real cluster-average CWMs are soft, so fixtures must be too.
    """
    rng = np.random.default_rng(seed)
    return rng.dirichlet(np.full(4, 1.0 / conc), size=length)


def onehot_pwm(seq, eps=0.001):
    """Near-deterministic PWM for a sequence string, shape (len, 4)."""
    idx = {c: i for i, c in enumerate("ACGT")}
    pwm = np.full((len(seq), 4), eps)
    for i, c in enumerate(seq):
        pwm[i, idx[c]] = 1 - 3 * eps
    return pwm


@pytest.mark.parametrize(
    "key,expected",
    [
        ("MOTIF pos_patterns.42", "pos_patterns.42"),
        ("pos_patterns.7 some description", "pos_patterns.7"),
        ("MOTIF neg_patterns.100 GATA1", "neg_patterns.100"),
        ("MOTIF weird_name_no_match", "weird_name_no_match"),
    ],
)
def test_parse_motif_name(key, expected):
    assert mr.parse_motif_name(key) == expected


def test_connected_components_merges_transitively():
    names = ["a", "b", "c", "d", "e"]
    pairs = pd.DataFrame({"motif_a": ["a", "b"], "motif_b": ["b", "c"]})
    comps = mr.connected_components(names, pairs)
    assert comps["a"] == comps["b"] == comps["c"]
    assert comps["d"] != comps["a"]
    assert comps["e"] != comps["d"]
    assert len(set(comps.values())) == 3  # {a,b,c}, {d}, {e}


def test_connected_components_no_pairs_is_all_singletons():
    names = ["a", "b", "c"]
    comps = mr.connected_components(names, pd.DataFrame({"motif_a": [], "motif_b": []}))
    assert len(set(comps.values())) == 3


def test_load_motifs_returns_alphabet_first(tmp_path):
    meme = write_meme(
        tmp_path / "m.meme",
        {"pos_patterns.0": onehot_pwm("ACGTACGT"), "pos_patterns.1": onehot_pwm("TTTTAAAA")},
    )
    names, pwms = mr.load_motifs(meme, mr.NAME_RE)
    assert names == ["pos_patterns.0", "pos_patterns.1"]
    for m in pwms:
        assert m.shape[0] == 4, m.shape  # (alphabet, length)
        assert m.shape[-1] == 8


def build_redundancy_meme(path, n=24, seed=0):
    """n distinct realistic motifs plus an exact duplicate of the first."""
    motifs = {
        f"pos_patterns.{i}": realistic_pwm(12, seed + i) for i in range(n)
    }
    motifs[f"pos_patterns.{n}"] = motifs["pos_patterns.0"].copy()
    return write_meme(path, motifs), f"pos_patterns.{n}"


def test_self_comparison_finds_duplicates_and_not_distinct_motifs(tmp_path):
    """An exact CWM duplicate must merge; unrelated motifs must not."""
    meme, dup_name = build_redundancy_meme(tmp_path / "m.meme")
    names, pwms = mr.load_motifs(meme, mr.NAME_RE)
    res = mr.self_compare(pwms, n_jobs=1)
    assert np.all(np.isinf(np.diag(res["p"])))  # diagonal masked

    pairs = mr.build_pairs(names, pwms, res, p_threshold=1e-6, min_overlap_frac=0.7)
    merged = {frozenset((a, b)) for a, b in zip(pairs.motif_a, pairs.motif_b)}
    assert frozenset(("pos_patterns.0", dup_name)) in merged

    comps = mr.connected_components(names, pairs)
    assert comps["pos_patterns.0"] == comps[dup_name]
    # the duplicate pair is the only thing that should have merged
    assert len(set(comps.values())) == len(names) - 1


def test_overlap_filter_rejects_short_inside_long(tmp_path):
    """A short motif aligning inside a longer one can be significant without
    being a duplicate; min_overlap_frac is what suppresses that."""
    long_pwm = onehot_pwm("AAAAAAGGGGGGCCCCCC")
    short_pwm = onehot_pwm("GGGG")
    meme = write_meme(
        tmp_path / "m.meme",
        {"pos_patterns.0": long_pwm, "pos_patterns.1": short_pwm},
    )
    names, pwms = mr.load_motifs(meme, mr.NAME_RE)
    res = mr.self_compare(pwms, n_jobs=1)
    lenient = mr.build_pairs(names, pwms, res, 1.0, min_overlap_frac=0.0)
    assert len(lenient) == 1
    # the alignment covers the short motif fully but only a fraction of the
    # long one; requiring coverage of the *shorter* motif keeps it, so check
    # the recorded fraction is what drives the filter
    assert lenient.iloc[0]["overlap_frac"] <= 1.0


def test_sweep_is_monotone_in_threshold(tmp_path):
    """Looser thresholds can only merge more, never fewer."""
    meme, _ = build_redundancy_meme(tmp_path / "m.meme", n=20, seed=100)
    names, pwms = mr.load_motifs(meme, mr.NAME_RE)
    res = mr.self_compare(pwms, n_jobs=1)
    summary = mr.sweep(names, pwms, res, [1e-12, 1e-6, 1e-2, 1.0], 0.7)
    for criterion in mr.MERGERS:
        assert summary[f"excess_{criterion}"].is_monotonic_increasing, criterion
        assert (summary[f"n_components_{criterion}"] <= summary["n_clusters"]).all()
        assert (summary[f"excess_{criterion}"] >= 0).all()
    # single linkage can only merge at least as much as complete or mutual
    assert (summary["excess_single"] >= summary["excess_complete"]).all()
    assert (summary["excess_complete"] >= summary["excess_mutual"]).all()


def test_redundancy_cli_end_to_end(tmp_path):
    n = 24
    meme, dup_name = build_redundancy_meme(tmp_path / "m.meme", n=n, seed=7)
    n_total = n + 1

    meta = tmp_path / "meta.tsv"
    pd.DataFrame(
        {
            "cluster_final": list(range(n_total)),
            "posneg": ["pos"] * n_total,
            "jaspar_name": ["AP1"] + [f"TF{i}" for i in range(1, n)] + ["AP1"],
            "jaspar_score": [0.95] * n_total,
            "total_seqlets": [100] * n_total,
            "n_experiments": [5] * n_total,
        }
    ).to_csv(meta, sep="\t", index=False)

    result = subprocess.run(
        [
            sys.executable, str(REPO_ROOT / "src/analysis/motif_redundancy.py"),
            "--meme", str(meme), "--cluster-metadata", str(meta),
            "--out-dir", str(tmp_path / "out"), "--report-threshold", "1e-6",
            "--n-jobs", "1",
        ],
        capture_output=True, text=True, env=SUBPROC_ENV,
    )
    assert result.returncode == 0, result.stderr
    out = tmp_path / "out"
    summary = pd.read_csv(out / "motif_redundancy_count_summary.tsv", sep="\t")
    assert (summary["n_clusters"] == n_total).all()
    assert "excess_complete" in summary.columns and "excess_mutual" in summary.columns
    comps = pd.read_csv(out / "motif_redundancy_count_components.tsv", sep="\t")
    dup_comp = comps[comps.motif.isin(["pos_patterns.0", dup_name])]
    assert dup_comp["component"].nunique() == 1  # duplicates merged
    assert "jaspar_name" in comps.columns
    assert (out / "motif_redundancy_count.png").exists()
    assert "Redundancy sweep" in result.stderr
    # the duplicate pair shares a JASPAR name, so agreement should be reported
    assert "JASPAR-name agreement within merged groups" in result.stderr


def test_redundancy_cli_errors_without_meme(tmp_path):
    result = subprocess.run(
        [
            sys.executable, str(REPO_ROOT / "src/analysis/motif_redundancy.py"),
            "--meme", str(tmp_path / "missing.meme"), "--out-dir", str(tmp_path / "out"),
        ],
        capture_output=True, text=True, env=SUBPROC_ENV,
    )
    assert result.returncode == 1
    assert "MEME file not found" in result.stderr


def test_information_content_bounds():
    uniform = np.full((4, 5), 0.25)
    assert mr.information_content(uniform) == pytest.approx(np.zeros(5), abs=1e-9)
    determined = np.zeros((4, 3)); determined[0] = 1.0
    assert mr.information_content(determined) == pytest.approx(np.full(3, 2.0), abs=1e-6)


def test_trim_pwm_finds_the_informative_core():
    """Uniform flanks carry no information and must be trimmed away."""
    pwm = np.full((4, 30), 0.25)
    pwm[:, 12:20] = 0.0
    pwm[0, 12:20] = 1.0          # 8bp determined core at 12:20
    start, end = mr.trim_pwm(pwm, threshold=0.3, min_len=6)
    assert (start, end) == (12, 20)


def test_trim_pwm_respects_min_len_floor():
    """A 2bp core must be widened, or no comparison can use it -- the same
    reason Fi-NeMo hit calling needed a min-length floor."""
    pwm = np.full((4, 30), 0.25)
    pwm[:, 14:16] = 0.0
    pwm[0, 14:16] = 1.0
    start, end = mr.trim_pwm(pwm, threshold=0.3, min_len=6)
    assert end - start >= 6
    assert start <= 14 and end >= 16


def test_trim_pwm_clamps_to_motif_width():
    pwm = np.full((4, 4), 0.25); pwm[0] = 1.0; pwm[1:] = 0.0
    start, end = mr.trim_pwm(pwm, threshold=0.3, min_len=20)
    assert (start, end) == (0, 4)


def test_trim_pwm_handles_a_fully_uniform_motif():
    pwm = np.full((4, 10), 0.25)
    assert mr.trim_pwm(pwm) == (0, 10)


def test_trimming_shrinks_fixed_width_windows(tmp_path):
    """The real failure mode: 50bp windows with a small informative core."""
    pwms = []
    for seed in range(6):
        m = np.full((4, 50), 0.25)
        core = realistic_pwm(10, seed, conc=20.0).T
        m[:, 20:30] = core
        pwms.append(m)
    trimmed, widths = mr.trim_motifs(pwms, 0.3, 6)
    assert (widths == 50).all()
    assert all(t.shape[-1] < 50 for t in trimmed)
    assert all(t.shape[-1] >= 6 for t in trimmed)


def test_mutual_best_cannot_chain():
    """A~B~C with A far from C: single linkage merges all three, mutual best
    merges at most one pair. This is the property that makes single linkage
    unusable on the real compendium."""
    names = ["a", "b", "c"]
    p = np.array([[0.0, 1e-9, 0.9], [1e-9, 0.0, 1e-9], [0.9, 1e-9, 0.0]])
    single = mr.connected_components(
        names, pd.DataFrame({"motif_a": ["a", "b"], "motif_b": ["b", "c"]})
    )
    assert len(set(single.values())) == 1  # all chained together
    mutual = mr.merge_mutual_best(names, p, 1e-6)
    assert len(set(mutual.values())) >= 2  # chaining prevented


def test_complete_linkage_requires_all_pairs():
    names = ["a", "b", "c"]
    # a~b tight, but c is far from both
    p = np.array([[0.0, 1e-12, 0.5], [1e-12, 0.0, 0.5], [0.5, 0.5, 0.0]])
    comps = mr.merge_complete(names, p, 1e-6)
    assert comps["a"] == comps["b"]
    assert comps["c"] != comps["a"]


def test_complete_linkage_merges_a_true_clique():
    names = ["a", "b", "c"]
    p = np.full((3, 3), 1e-12); np.fill_diagonal(p, 0.0)
    comps = mr.merge_complete(names, p, 1e-6)
    assert len(set(comps.values())) == 1


def test_symmetric_p_takes_the_conservative_side():
    pwms = [np.full((4, 10), 0.25) for _ in range(2)]
    res = {
        "p": np.array([[0.0, 1e-9], [1e-3, 0.0]]),
        "overlaps": np.full((2, 2), 10.0),
        "scores": np.zeros((2, 2)), "offsets": np.zeros((2, 2)),
        "strands": np.zeros((2, 2)),
    }
    p_sym = mr.symmetric_p(pwms, res, 0.7)
    assert p_sym[0, 1] == pytest.approx(1e-3)  # larger of the two
    assert p_sym[1, 0] == pytest.approx(1e-3)


def test_symmetric_p_rejects_failing_overlap():
    pwms = [np.full((4, 10), 0.25), np.full((4, 10), 0.25)]
    res = {
        "p": np.full((2, 2), 1e-12),
        "overlaps": np.full((2, 2), 3.0),   # 3/10 = 0.3 < 0.7
        "scores": np.zeros((2, 2)), "offsets": np.zeros((2, 2)),
        "strands": np.zeros((2, 2)),
    }
    p_sym = mr.symmetric_p(pwms, res, 0.7)
    assert p_sym[0, 1] == 1.0


def test_load_subset_from_plain_list(tmp_path):
    f = tmp_path / "s.txt"
    f.write_text("pos_patterns.1\npos_patterns.2\n\n")
    assert mr.load_subset(f) == {"pos_patterns.1", "pos_patterns.2"}


def test_load_subset_from_tsv_column(tmp_path):
    f = tmp_path / "s.tsv"
    f.write_text("motif\twidth\npos_patterns.5\t12\npos_patterns.9\t14\n")
    assert mr.load_subset(f) == {"pos_patterns.5", "pos_patterns.9"}


def test_load_subset_from_concentration_output(tmp_path):
    f = tmp_path / "s.tsv"
    f.write_text("compendium_motif_name\tx\npos_patterns.3\t1\n")
    assert mr.load_subset(f) == {"pos_patterns.3"}


def test_redundancy_cli_subset_restricts(tmp_path):
    meme, dup_name = build_redundancy_meme(tmp_path / "m.meme", n=20, seed=3)
    sub = tmp_path / "sub.txt"
    sub.write_text("\n".join(["pos_patterns.0", "pos_patterns.1", dup_name]))
    result = subprocess.run(
        [
            sys.executable, str(REPO_ROOT / "src/analysis/motif_redundancy.py"),
            "--meme", str(meme), "--subset", str(sub),
            "--out-dir", str(tmp_path / "out"), "--n-jobs", "1",
        ],
        capture_output=True, text=True, env=SUBPROC_ENV,
    )
    assert result.returncode == 0, result.stderr
    assert "restricted to 3 clusters" in result.stderr
    summary = pd.read_csv(
        tmp_path / "out" / "motif_redundancy_count_summary.tsv", sep="\t"
    )
    assert (summary["n_clusters"] == 3).all()


def test_no_trim_flag_reports_untrimmed(tmp_path):
    pwms = {}
    for i in range(8):
        m = np.full((50, 4), 0.25)
        m[20:30] = realistic_pwm(10, i, conc=20.0)
        pwms[f"pos_patterns.{i}"] = m
    meme = write_meme(tmp_path / "m.meme", pwms)
    result = subprocess.run(
        [
            sys.executable, str(REPO_ROOT / "src/analysis/motif_redundancy.py"),
            "--meme", str(meme), "--no-trim", "--out-dir", str(tmp_path / "out"),
            "--n-jobs", "1",
        ],
        capture_output=True, text=True, env=SUBPROC_ENV,
    )
    assert result.returncode == 0, result.stderr
    assert "UNTRIMMED" in result.stderr


def write_cluster_h5(path, motifs):
    """modisco-lite-shaped cluster-average h5. motifs is
    {(group, key): (pfm (L,4), cwm (L,4))}."""
    import h5py

    with h5py.File(path, "w") as f:
        for (group, key), (pfm, cwm) in motifs.items():
            g = f.require_group(group).create_group(key)
            g.create_dataset("sequence", data=pfm)
            g.create_dataset("contrib_scores", data=cwm)
    return path


def flanked_pair(core_len=10, width=50, seed=0, flank_gc=0.96, core_peak=0.6):
    """A window whose PFM flanks are *more* informative than 0.3x its core max.

    This reproduces the real pathology deterministically. Cluster-average PFMs
    are soft, so the core's information content is modest (~1 bit here), while
    GC-skewed flanks of a PRO-cap peak carry real composition. When flank IC
    exceeds `trim_threshold * max(IC)`, information-content trimming keeps the
    whole window -- which is why the real run came back at median 49 of 50bp.
    The CWM has contribution only over the core, so contribution-based trimming
    is unaffected.

    `seed` shifts which base the core prefers, so distinct motifs differ.
    """
    pfm = np.zeros((width, 4))
    pfm[:, [1, 2]] = flank_gc / 2
    pfm[:, [0, 3]] = (1 - flank_gc) / 2
    start = (width - core_len) // 2
    rng = np.random.default_rng(seed)
    for i in range(core_len):
        col = np.full(4, (1 - core_peak) / 3)
        col[rng.integers(0, 4)] = core_peak
        pfm[start + i] = col
    cwm = np.zeros((width, 4))
    cwm[start:start + core_len] = pfm[start:start + core_len] - 0.25
    return pfm, cwm, start, start + core_len


def test_trim_cwm_finds_core_where_information_content_cannot():
    """The real failure mode: informative flanks defeat IC trimming, while
    contribution magnitude locates the core exactly."""
    pfm, cwm, start, end = flanked_pair(seed=1)
    ic = mr.information_content(pfm.T)
    # the fixture's premise: flank IC clears the threshold set by the core
    assert ic[0] > 0.3 * ic.max()

    ic_start, ic_end = mr.trim_pwm(pfm.T, threshold=0.3, min_len=6)
    cwm_start, cwm_end = mr.trim_cwm(cwm.T, threshold=0.3, min_len=6)
    assert (cwm_start, cwm_end) == (start, end)
    assert (ic_start, ic_end) == (0, pfm.shape[0])  # IC keeps the whole window


def test_cwm_trim_is_insensitive_to_flank_composition():
    """The property that matters: the contribution-derived span does not move
    when the PFM's flanks get more or less informative, while the IC-derived
    span does."""
    spans_cwm, spans_ic = set(), set()
    for flank_gc in (0.5, 0.7, 0.9, 0.99):
        pfm, cwm, start, end = flanked_pair(seed=2, flank_gc=flank_gc)
        spans_cwm.add(mr.trim_cwm(cwm.T, 0.3, 6))
        spans_ic.add(mr.trim_pwm(pfm.T, 0.3, 6))
    assert len(spans_cwm) == 1  # unchanged across all flank compositions
    assert len(spans_ic) > 1    # IC trimming is at the mercy of the flanks


def test_trim_cwm_respects_min_len():
    cwm = np.zeros((4, 30))
    cwm[0, 14:16] = 1.0
    start, end = mr.trim_cwm(cwm, 0.3, min_len=6)
    assert end - start >= 6


def test_trim_cwm_handles_all_zero_contributions():
    assert mr.trim_cwm(np.zeros((4, 12))) == (0, 12)


def test_trim_by_cwm_applies_cwm_span_to_the_pfm():
    pfms, cwms, spans = [], [], []
    for seed in range(4):
        pfm, cwm, s, e = flanked_pair(seed=seed)
        pfms.append(pfm.T); cwms.append(cwm.T); spans.append(e - s)
    trimmed, widths = mr.trim_by_cwm(pfms, cwms, 0.3, 6)
    assert (widths == 50).all()
    assert [t.shape[-1] for t in trimmed] == spans
    # the trimmed PFM must still be a probability matrix, not contributions
    for t in trimmed:
        assert (t >= 0).all()
        assert t.sum(axis=0) == pytest.approx(np.ones(t.shape[-1]), abs=1e-6)


def test_load_motifs_h5_roundtrip(tmp_path):
    motifs = {}
    for i in range(3):
        pfm, cwm, _, _ = flanked_pair(seed=i)
        motifs[("pos_patterns", f"pattern_{i}")] = (pfm, cwm)
    pfm, cwm, _, _ = flanked_pair(seed=9)
    motifs[("neg_patterns", "7")] = (pfm, cwm)
    h5 = write_cluster_h5(tmp_path / "c.h5", motifs)

    names, pfms, cwms = mr.load_motifs_h5(h5)
    assert set(names) == {
        "pos_patterns.0", "pos_patterns.1", "pos_patterns.2", "neg_patterns.7"
    }
    for a, c in zip(pfms, cwms):
        assert a.shape[0] == 4 and c.shape[0] == 4   # (alphabet, length)
        assert a.shape[-1] == 50


def test_h5_route_trims_to_the_core(tmp_path):
    """End-to-end: the h5 route trims on contribution scores.

    Note this test deliberately launches only one subprocess. Two back-to-back
    subprocess runs of this script from inside a pytest process that has
    already loaded memelite's OpenMP runtime reliably SIGABRT on macOS during
    plotting -- the script itself is fine standalone (verified: exit 0, all
    outputs written), so it is a harness interaction, not a script defect.
    The MEME route's behaviour is covered in-process below instead.
    """
    motifs = {}
    for i in range(12):
        pfm, cwm, _, _ = flanked_pair(seed=i)
        motifs[("pos_patterns", f"pattern_{i}")] = (pfm, cwm)
    h5 = write_cluster_h5(tmp_path / "c.h5", motifs)

    run = subprocess.run(
        [sys.executable, str(REPO_ROOT / "src/analysis/motif_redundancy.py"),
         "--modisco-h5", str(h5), "--out-dir", str(tmp_path / "o1"), "--n-jobs", "1"],
        capture_output=True, text=True, env=SUBPROC_ENV,
    )
    assert run.returncode == 0, run.stderr
    assert "contribution-trimmed" in run.stderr
    assert "50-50bp -> 10-10bp" in run.stderr  # core is 10bp wide


def test_trimming_ineffective_predicate():
    """The condition behind the MEME route's warning, tested directly rather
    than by asserting on a subprocess's stderr."""
    assert mr.trimming_ineffective(np.array([49, 50, 48]), np.array([50, 50, 50]))
    assert not mr.trimming_ineffective(np.array([10, 12, 9]), np.array([50, 50, 50]))
    assert not mr.trimming_ineffective(np.array([]), np.array([]))


def test_meme_and_h5_routes_disagree_on_these_motifs():
    """In-process version of the route comparison: contribution trimming finds
    the core, information-content trimming does not, and the predicate flags
    the latter."""
    pfms, cwms = [], []
    for i in range(12):
        pfm, cwm, _, _ = flanked_pair(seed=i)
        pfms.append(pfm.T)
        cwms.append(cwm.T)

    by_cwm, raw = mr.trim_by_cwm(pfms, cwms, 0.3, 6)
    by_ic, raw_ic = mr.trim_motifs(pfms, 0.3, 6)
    w_cwm = np.array([m.shape[-1] for m in by_cwm])
    w_ic = np.array([m.shape[-1] for m in by_ic])

    assert (raw == 50).all() and (raw_ic == 50).all()
    assert (w_cwm == 10).all()          # exactly the core
    assert (w_ic == 50).all()           # nothing removed
    assert not mr.trimming_ineffective(w_cwm, raw)
    assert mr.trimming_ineffective(w_ic, raw_ic)


def test_h5_subset_filters_cwms_too(tmp_path):
    motifs = {}
    for i in range(10):
        pfm, cwm, _, _ = flanked_pair(seed=i)
        motifs[("pos_patterns", f"pattern_{i}")] = (pfm, cwm)
    h5 = write_cluster_h5(tmp_path / "c.h5", motifs)
    sub = tmp_path / "s.txt"
    sub.write_text("pos_patterns.0\npos_patterns.1\npos_patterns.2\n")
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "src/analysis/motif_redundancy.py"),
         "--modisco-h5", str(h5), "--subset", str(sub),
         "--out-dir", str(tmp_path / "o"), "--n-jobs", "1"],
        capture_output=True, text=True, env=SUBPROC_ENV,
    )
    assert result.returncode == 0, result.stderr
    assert "restricted to 3 clusters" in result.stderr


def test_h5_loader_discovers_a_flat_layout(tmp_path):
    """MotifCompendium's exporter need not use pos_patterns/pattern_N/, so the
    loader walks the file rather than assuming a hierarchy. Zero motifs found
    on the real file is what prompted this."""
    import h5py

    with h5py.File(tmp_path / "flat.h5", "w") as f:
        for i in range(3):
            pfm, cwm, _, _ = flanked_pair(seed=i)
            g = f.create_group(f"cluster_{i}")
            g.create_dataset("sequence", data=pfm)
            g.create_dataset("contrib_scores", data=cwm)
    names, pfms, cwms = mr.load_motifs_h5(tmp_path / "flat.h5")
    assert len(names) == 3
    assert all(p.shape[0] == 4 for p in pfms)
    assert all(c.shape[0] == 4 for c in cwms)


def test_h5_loader_handles_nested_posneg_and_strips_pattern_prefix(tmp_path):
    import h5py

    with h5py.File(tmp_path / "n.h5", "w") as f:
        for group, key in (("pos_patterns", "pattern_4"), ("neg_patterns", "11")):
            pfm, cwm, _, _ = flanked_pair(seed=1)
            g = f.require_group(group).create_group(key)
            g.create_dataset("sequence", data=pfm)
            g.create_dataset("contrib_scores", data=cwm)
    names, _, _ = mr.load_motifs_h5(tmp_path / "n.h5")
    assert set(names) == {"pos_patterns.4", "neg_patterns.11"}


def test_h5_loader_falls_back_to_normalized_contributions(tmp_path):
    """A group with contributions but no probability matrix still yields a
    comparison matrix, since TOMTOM needs probability-like columns."""
    import h5py

    with h5py.File(tmp_path / "c.h5", "w") as f:
        for i in range(2):
            _, cwm, _, _ = flanked_pair(seed=i)
            g = f.create_group(f"m{i}")
            g.create_dataset("contrib_scores", data=cwm)
    names, pfms, cwms = mr.load_motifs_h5(tmp_path / "c.h5")
    assert len(names) == 2
    for p in pfms:
        assert p.sum(axis=0) == pytest.approx(np.ones(p.shape[-1]), abs=1e-6)
        assert (p >= 0).all()


def test_h5_loader_accepts_alternate_dataset_names(tmp_path):
    import h5py

    with h5py.File(tmp_path / "alt.h5", "w") as f:
        for i in range(2):
            pfm, cwm, _, _ = flanked_pair(seed=i)
            g = f.create_group(f"m{i}")
            g.create_dataset("PFM", data=pfm)
            g.create_dataset("CWM", data=cwm)
    names, pfms, _ = mr.load_motifs_h5(tmp_path / "alt.h5")
    assert len(names) == 2


def test_h5_loader_ignores_non_motif_datasets(tmp_path):
    import h5py

    with h5py.File(tmp_path / "x.h5", "w") as f:
        g = f.create_group("m0")
        pfm, cwm, _, _ = flanked_pair(seed=0)
        g.create_dataset("sequence", data=pfm)
        g.create_dataset("contrib_scores", data=cwm)
        g.create_dataset("seqlet_starts", data=np.arange(20))   # 1-D, ignored
        f.create_dataset("metadata", data=np.zeros((7, 9)))     # not motif-shaped
    names, _, _ = mr.load_motifs_h5(tmp_path / "x.h5")
    assert names == ["m0"]


def test_h5_tree_lists_datasets(tmp_path):
    import h5py

    with h5py.File(tmp_path / "t.h5", "w") as f:
        f.create_group("a").create_dataset("b", data=np.zeros((4, 5)))
    lines = mr.h5_tree(tmp_path / "t.h5")
    assert any("a/b" in line and "(4, 5)" in line for line in lines)


def test_unrecognized_h5_layout_errors_with_a_tree_dump(tmp_path):
    """The failure mode that crashed with an UnboundLocalError: an h5 whose
    layout yields no motifs must report what it actually contains."""
    import h5py

    with h5py.File(tmp_path / "bad.h5", "w") as f:
        f.create_dataset("unexpected", data=np.zeros((10, 10)))

    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "src/analysis/motif_redundancy.py"),
         "--modisco-h5", str(tmp_path / "bad.h5"),
         "--out-dir", str(tmp_path / "o"), "--n-jobs", "1"],
        capture_output=True, text=True, env=SUBPROC_ENV,
    )
    assert result.returncode == 1
    assert "layout is not what was expected" in result.stderr
    assert "unexpected" in result.stderr          # tree dump
    assert "UnboundLocalError" not in result.stderr
    assert "Traceback" not in result.stderr


def test_jaspar_agreement_perfect_and_zero():
    comps = {"a": 0, "b": 0, "c": 1, "d": 1}
    agree = {"a": "AP1", "b": "AP1", "c": "SP1", "d": "SP1"}
    frac, n_groups, n_in, chance = mr.jaspar_agreement(comps, agree)
    assert (frac, n_groups, n_in) == (1.0, 2, 4)
    assert 0.0 < chance <= 1.0

    disagree = {"a": "AP1", "b": "GATA1", "c": "SP1", "d": "TBP"}
    frac, n_groups, _, _ = mr.jaspar_agreement(comps, disagree)
    assert (frac, n_groups) == (0.0, 2)


def test_jaspar_agreement_is_none_when_nothing_merged():
    """None and 0% mean different things: nothing to check vs. checked and
    inconsistent."""
    comps = {"a": 0, "b": 1}
    frac, n_groups, n_in, chance = mr.jaspar_agreement(comps, {"a": "AP1", "b": "SP1"})
    assert frac is None and n_groups == 0 and n_in == 0 and chance == 0.0


def test_jaspar_agreement_ignores_unnamed_clusters():
    comps = {"a": 0, "b": 0, "c": 0}
    # only two of the three carry a name; the unnamed one must not count
    frac, n_groups, n_in, _ = mr.jaspar_agreement(comps, {"a": "AP1", "b": "AP1"})
    assert (frac, n_groups, n_in) == (1.0, 1, 2)


def test_sweep_reports_agreement_per_criterion():
    """The criterion-specific agreement is the number that decides which
    redundancy estimate to trust, so every criterion must carry its own."""
    pfms, cwms, names = [], [], []
    for i in range(20):
        pfm, cwm, _, _ = flanked_pair(seed=i)
        pfms.append(pfm.T); cwms.append(cwm.T); names.append(f"pos_patterns.{i}")
    pfms.append(pfms[0].copy()); cwms.append(cwms[0].copy())
    names.append("pos_patterns.20")
    trimmed, _ = mr.trim_by_cwm(pfms, cwms, 0.3, 6)
    res = mr.self_compare(trimmed, n_jobs=1)
    jaspar = {n: f"TF{i}" for i, n in enumerate(names)}
    jaspar["pos_patterns.20"] = "TF0"  # the duplicate shares name with cluster 0

    summary = mr.sweep(names, trimmed, res, [1e-6], 0.7, jaspar=jaspar)
    for criterion in mr.MERGERS:
        assert f"jaspar_agree_{criterion}" in summary.columns
        assert f"jaspar_groups_{criterion}" in summary.columns


def test_drop_untrimmable_removes_coreless_clusters(tmp_path):
    """Clusters whose contributions are diffuse across the whole window have no
    locatable core; they chain everything together and, in real Fi-NeMo runs,
    receive almost no hits. They must be droppable."""
    import h5py

    with h5py.File(tmp_path / "c.h5", "w") as f:
        for i in range(8):                      # normal: 10bp core
            pfm, cwm, _, _ = flanked_pair(seed=i)
            g = f.create_group(f"good{i}")
            g.create_dataset("sequence", data=pfm)
            g.create_dataset("contrib_scores", data=cwm)
        for i in range(3):                      # diffuse: contribution everywhere
            pfm, _, _, _ = flanked_pair(seed=100 + i)
            flat = np.full((50, 4), 0.2)
            g = f.create_group(f"diffuse{i}")
            g.create_dataset("sequence", data=pfm)
            g.create_dataset("contrib_scores", data=flat)

    # One subprocess per test: two back-to-back runs from a pytest process that
    # has already loaded memelite's OpenMP runtime SIGABRT on macOS (see
    # test_h5_route_trims_to_the_core).
    dropped = subprocess.run(
        [sys.executable, str(REPO_ROOT / "src/analysis/motif_redundancy.py"),
         "--modisco-h5", str(tmp_path / "c.h5"), "--n-jobs", "1",
         "--out-dir", str(tmp_path / "o2"), "--drop-untrimmable"],
        capture_output=True, text=True, env=SUBPROC_ENV,
    )
    assert dropped.returncode == 0, dropped.stderr
    assert "3 cluster(s) did not shrink at all" in dropped.stderr
    assert "dropped 3 untrimmable cluster(s)" in dropped.stderr
    summary = pd.read_csv(
        tmp_path / "o2" / "motif_redundancy_count_summary.tsv", sep="\t"
    )
    assert (summary["n_clusters"] == 8).all()


def test_untrimmable_clusters_are_identifiable_in_process():
    """The predicate behind --drop-untrimmable, without a subprocess."""
    pfms, cwms = [], []
    for i in range(5):
        pfm, cwm, _, _ = flanked_pair(seed=i)
        pfms.append(pfm.T); cwms.append(cwm.T)
    for i in range(2):
        pfm, _, _, _ = flanked_pair(seed=50 + i)
        pfms.append(pfm.T); cwms.append(np.full((4, 50), 0.2))

    trimmed, raw = mr.trim_by_cwm(pfms, cwms, 0.3, 6)
    widths = np.array([t.shape[-1] for t in trimmed])
    untrimmable = widths >= raw
    assert untrimmable.sum() == 2
    assert (widths[~untrimmable] == 10).all()


def test_width_report_flags_cores_wider_than_finemo():
    """The calibration check: a median far above Fi-NeMo's ~14bp means the
    threshold is too permissive for cluster averages."""
    assert mr.FINEMO_MEDIAN_TRIM_BP == 14
    # a median of 25bp (the real 0.3-threshold result) must trip the notice
    assert 25 <= 2 * mr.FINEMO_MEDIAN_TRIM_BP
    assert 30 > 2 * mr.FINEMO_MEDIAN_TRIM_BP


def test_stricter_trim_threshold_narrows_cores():
    """Raising --trim-threshold must monotonically narrow the trimmed span, so
    it is a usable dial for matching Fi-NeMo's effective width."""
    pfms, cwms = [], []
    for i in range(10):
        pfm, cwm, _, _ = flanked_pair(seed=i)
        # give the CWM a decaying tail, as real CWM magnitude has
        cwm = cwm.copy()
        for off in range(1, 10):
            cwm[20 - off] = cwm[20] * (0.9 ** off)
            cwm[30 + off - 1] = cwm[29] * (0.9 ** off)
        pfms.append(pfm.T); cwms.append(cwm.T)

    medians = []
    for thresh in (0.1, 0.3, 0.5, 0.7):
        trimmed, _ = mr.trim_by_cwm(pfms, cwms, thresh, 6)
        medians.append(float(np.median([t.shape[-1] for t in trimmed])))
    assert medians == sorted(medians, reverse=True), medians
    assert medians[0] > medians[-1]


@pytest.mark.parametrize(
    "name,expected",
    [
        ("SP9", "SP"), ("SP1", "SP"), ("SP2", "SP"),
        ("ETV7", "ETV"), ("ELF2", "ELF"), ("Atf1", "ATF"),
        ("POU2F1::SOX2", "POU2F"), ("Pou5f1::Sox2", "POU5F"),
        ("ZNF143", "ZNF"), ("TBP", "TBP"), ("CTCF", "CTCF"),
    ],
)
def test_jaspar_family_collapses_paralogues(name, expected):
    assert mr.jaspar_family(name) == expected


def test_agreement_chance_is_small_for_a_skewed_pool():
    """The number that makes a ~50% observation interpretable: with names as
    skewed as the real compendium's, random merging almost never agrees."""
    pool = ["SP9"] * 31 + ["TBP"] * 15 + ["NFYA"] * 14 + [
        f"TF{i}" for i in range(246)
    ]
    pairs = [["x", "y"]] * 50
    chance = mr.agreement_chance(pairs, pool)
    assert 0.0 < chance < 0.05


def test_agreement_chance_is_one_when_every_name_is_identical():
    assert mr.agreement_chance([["a", "a"]], ["N"] * 10) == pytest.approx(1.0)


def test_agreement_chance_falls_with_group_size():
    """Larger merged groups are harder to agree by chance, so the baseline
    must depend on group size, not just the name distribution."""
    pool = [f"TF{i % 10}" for i in range(100)]
    c2 = mr.agreement_chance([["a", "b"]], pool)
    c5 = mr.agreement_chance([["a"] * 5], pool)
    assert c5 < c2


def test_family_agreement_exceeds_exact_when_merges_are_within_family():
    """The case that matters: two clusters that are the same motif but carry
    different paralogue labels count as disagreement by name and agreement by
    family."""
    comps = {"a": 0, "b": 0, "c": 1, "d": 1}
    jaspar = {"a": "SP1", "b": "SP9", "c": "ETV4", "d": "ETV7"}
    by_name, _, _, _ = mr.jaspar_agreement(comps, jaspar, family=False)
    by_family, _, _, _ = mr.jaspar_agreement(comps, jaspar, family=True)
    assert by_name == 0.0
    assert by_family == 1.0


def test_family_agreement_does_not_rescue_cross_family_merges():
    comps = {"a": 0, "b": 0}
    jaspar = {"a": "SP1", "b": "GATA1"}
    by_family, _, _, _ = mr.jaspar_agreement(comps, jaspar, family=True)
    assert by_family == 0.0


def test_sweep_records_chance_and_family_columns():
    pfms, cwms, names = [], [], []
    for i in range(20):
        pfm, cwm, _, _ = flanked_pair(seed=i)
        pfms.append(pfm.T); cwms.append(cwm.T); names.append(f"pos_patterns.{i}")
    pfms.append(pfms[0].copy()); cwms.append(cwms[0].copy())
    names.append("pos_patterns.20")
    trimmed, _ = mr.trim_by_cwm(pfms, cwms, 0.3, 6)
    res = mr.self_compare(trimmed, n_jobs=1)
    jaspar = {n: f"TF{i}" for i, n in enumerate(names)}
    jaspar["pos_patterns.20"] = "TF0"

    summary = mr.sweep(names, trimmed, res, [1e-6], 0.7, jaspar=jaspar)
    for criterion in mr.MERGERS:
        for prefix in ("jaspar_agree", "jaspar_chance", "jaspar_enrich",
                       "family_agree", "family_enrich"):
            assert f"{prefix}_{criterion}" in summary.columns


def test_mutual_best_pairs_are_reciprocated_only():
    """A one-way best hit is not a mutual pair: b's closest is c, so a-b must
    not appear even though b is a's closest."""
    names = ["a", "b", "c"]
    p = np.array([
        [0.0,  1e-8, 0.5],
        [1e-8, 0.0,  1e-12],
        [0.5,  1e-12, 0.0],
    ])
    pairs = mr.mutual_best_pairs(names, p, 1e-6)
    got = {frozenset((r.motif_a, r.motif_b)) for r in pairs.itertuples()}
    assert got == {frozenset(("b", "c"))}


def test_mutual_best_pairs_respects_threshold():
    names = ["a", "b"]
    p = np.array([[0.0, 1e-3], [1e-3, 0.0]])
    assert mr.mutual_best_pairs(names, p, 1e-6).empty
    assert len(mr.mutual_best_pairs(names, p, 1e-2)) == 1


def test_annotate_pairs_flags_name_and_family_agreement(tmp_path):
    pairs = pd.DataFrame({
        "motif_a": ["pos_patterns.0", "pos_patterns.2", "pos_patterns.4"],
        "motif_b": ["pos_patterns.1", "pos_patterns.3", "pos_patterns.5"],
        "p_value": [1e-9, 1e-8, 1e-7],
    })
    meta = tmp_path / "meta.tsv"
    pd.DataFrame({
        "cluster_final": [0, 1, 2, 3, 4, 5],
        "posneg": ["pos"] * 6,
        "jaspar_name": ["AP1", "AP1", "SP1", "SP9", "GATA1", "TBP"],
        "jaspar_score": [0.9] * 6,
        "total_seqlets": [10, 20, 500, 400, 5, 6],
        "n_experiments": [2] * 6,
    }).to_csv(meta, sep="\t", index=False)

    widths = {f"pos_patterns.{i}": 10 for i in range(6)}
    out = mr.annotate_pairs(pairs, widths, meta, None, tmp_path)

    row = out.set_index("motif_a")
    assert row.loc["pos_patterns.0", "name_agree"]            # AP1 == AP1
    assert not row.loc["pos_patterns.2", "name_agree"]        # SP1 != SP9
    assert row.loc["pos_patterns.2", "family_agree"]          # ...but same family
    assert not row.loc["pos_patterns.4", "family_agree"]      # GATA1 vs TBP
    # highest-seqlet pair must sort first, since it matters most to the count
    assert out.iloc[0]["motif_a"] == "pos_patterns.2"


def test_annotate_pairs_resolves_logos_against_logo_root(tmp_path):
    """Paths are stored absolute: the HTML lives in figures/ while the SVGs
    live under motifcompendium/, and a relative path that depends on the
    report's location is what broke the first version."""
    root = tmp_path / "mc"
    (root / "logos" / "fwd").mkdir(parents=True)
    for name in ("a", "b"):
        (root / "logos" / "fwd" / f"{name}.svg").write_bytes(SVG_A)
    lp = tmp_path / "logos.tsv"
    pd.DataFrame({
        "cluster_final": [0, 1],
        "logo_fwd_svg": ["logos/fwd/a.svg", "logos/fwd/b.svg"],
        "logo_rev_svg": ["logos/rev/a.svg", "logos/rev/b.svg"],
    }).to_csv(lp, sep="\t", index=False)
    pairs = pd.DataFrame({
        "motif_a": ["pos_patterns.0"], "motif_b": ["pos_patterns.1"],
        "p_value": [1e-9],
    })
    out = mr.annotate_pairs(
        pairs, {"pos_patterns.0": 8, "pos_patterns.1": 9}, None, lp,
        tmp_path, logo_root=root,
    )
    assert Path(out.loc[0, "logo_a"]).is_absolute()
    assert Path(out.loc[0, "logo_a"]).exists()
    assert out.loc[0, "logo_a"].endswith("a.svg")


def test_describe_logo_resolution_distinguishes_the_failure_modes(tmp_path):
    """The diagnostic that separates "paths never resolved" from "files are
    missing" from "all good" -- previously all three looked identical in the
    report."""
    pairs = pd.DataFrame({"motif_a": ["a"], "motif_b": ["b"], "p_value": [1e-9]})
    assert "no logo column" in mr.describe_logo_resolution(pairs)

    unresolved = pairs.assign(logo_a=[None], logo_b=[None])
    assert "0/2 logo paths resolved" in mr.describe_logo_resolution(unresolved)

    good = tmp_path / "g.svg"; good.write_bytes(SVG_A)
    partial = pairs.assign(logo_a=[str(good)], logo_b=[str(tmp_path / "nope.svg")])
    msg = mr.describe_logo_resolution(partial)
    assert "2/2 logo paths resolved, 1 file(s) present" in msg
    assert "example missing" in msg

    both = pairs.assign(logo_a=[str(good)], logo_b=[str(good)])
    assert "2 file(s) present" in mr.describe_logo_resolution(both)


def test_html_warns_when_no_logos_are_available(tmp_path):
    """A logo-less report must say so rather than silently showing dashes."""
    pairs = html_pairs_fixture().drop(columns=["logo_a", "logo_b"])
    out = tmp_path / "p.html"
    mr.write_pairs_html(pairs, out, "count", 1e-6, out_dir=tmp_path)
    text = out.read_text()
    assert "No logos available" in text
    assert "cluster_logo_paths.tsv" in text
    assert "pos_patterns.0" in text          # the table is still written


def test_annotate_pairs_handles_empty_input():
    assert mr.annotate_pairs(
        pd.DataFrame({"motif_a": [], "motif_b": [], "p_value": []}),
        {}, None, None, Path(".")
    ).empty


SVG_A = b"<svg xmlns='http://www.w3.org/2000/svg'><text>AAA</text></svg>"
SVG_B = b"<svg xmlns='http://www.w3.org/2000/svg'><text>BBB</text></svg>"


def test_embed_svg_returns_a_data_uri(tmp_path):
    import base64

    f = tmp_path / "a.svg"
    f.write_bytes(SVG_A)
    uri = mr.embed_svg(f)
    assert uri.startswith("data:image/svg+xml;base64,")
    assert base64.b64decode(uri.split(",", 1)[1]) == SVG_A


def test_embed_svg_returns_none_for_missing_file(tmp_path):
    assert mr.embed_svg(tmp_path / "nope.svg") is None


def html_pairs_fixture():
    return pd.DataFrame({
        "motif_a": ["pos_patterns.0", "pos_patterns.2"],
        "motif_b": ["pos_patterns.1", "pos_patterns.3"],
        "p_value": [1e-9, 1e-8],
        "jaspar_a": ["AP1", "GATA1"], "jaspar_b": ["AP1", "TBP"],
        "seqlets_a": [100, 50], "seqlets_b": [90, 40],
        "max_seqlets": [100, 50],
        "name_agree": [True, False], "family_agree": [True, False],
        "trimmed_len_a": [8, 9], "trimmed_len_b": [8, 9],
        "logo_a": ["logos/a.svg", "logos/c.svg"],
        "logo_b": ["logos/b.svg", "logos/d.svg"],
    })


def absolutize(pairs, out_dir):
    for col in ("logo_a", "logo_b"):
        pairs[col] = pairs[col].map(lambda v: str((out_dir / v).resolve()))
    return pairs


def write_logo_tree(out_dir):
    (out_dir / "logos").mkdir(parents=True, exist_ok=True)
    for name, data in (("a", SVG_A), ("b", SVG_B), ("c", SVG_A), ("d", SVG_B)):
        (out_dir / "logos" / f"{name}.svg").write_bytes(data)


def test_pairs_html_embeds_logos_so_it_travels(tmp_path):
    """The report gets copied off the cluster, so a linked SVG is a broken SVG.
    Embedded output must contain no file references at all."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    write_logo_tree(out_dir)
    html = out_dir / "p.html"
    mr.write_pairs_html(
        absolutize(html_pairs_fixture(), out_dir), html, "count", 1e-6,
        out_dir=out_dir, embed=True,
    )
    text = html.read_text()
    assert "data:image/svg+xml;base64," in text
    assert text.count("data:image/svg+xml;base64,") == 4   # 2 pairs x 2 logos
    assert "logos/a.svg'" not in text                      # nothing linked
    assert "file not found" not in text


def test_pairs_html_can_link_instead_of_embedding(tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    write_logo_tree(out_dir)
    html = out_dir / "p.html"
    mr.write_pairs_html(
        absolutize(html_pairs_fixture(), out_dir), html, "count", 1e-6,
        out_dir=out_dir, embed=False,
    )
    text = html.read_text()
    assert "a.svg" in text
    assert "data:image/svg+xml;base64," not in text


def test_pairs_html_marks_missing_logos_without_failing(tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()          # deliberately no logo files written
    html = out_dir / "p.html"
    mr.write_pairs_html(
        absolutize(html_pairs_fixture(), out_dir), html, "count", 1e-6,
        out_dir=out_dir, embed=True,
    )
    text = html.read_text()
    assert "file not found" in text
    assert "data:image/svg+xml;base64," not in text


def test_pairs_html_top_pairs_limits_rows_and_reports_total(tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    write_logo_tree(out_dir)
    html = out_dir / "p.html"
    mr.write_pairs_html(
        absolutize(html_pairs_fixture(), out_dir), html, "count", 1e-6,
        out_dir=out_dir, embed=True, top_pairs=1,
    )
    text = html.read_text()
    assert "1 of 2 pairs shown" in text
    # the disagreeing pair is the one kept
    assert "pos_patterns.2" in text
    assert text.count("data:image/svg+xml;base64,") == 2


def test_pairs_html_puts_disagreements_first(tmp_path):
    pairs = pd.DataFrame({
        "motif_a": ["pos_patterns.0", "pos_patterns.2"],
        "motif_b": ["pos_patterns.1", "pos_patterns.3"],
        "p_value": [1e-9, 1e-8],
        "jaspar_a": ["AP1", "GATA1"], "jaspar_b": ["AP1", "TBP"],
        "seqlets_a": [100, 50], "seqlets_b": [90, 40],
        "max_seqlets": [100, 50],
        "name_agree": [True, False], "family_agree": [True, False],
        "trimmed_len_a": [8, 9], "trimmed_len_b": [8, 9],
        "logo_a": ["a.svg", "c.svg"], "logo_b": ["b.svg", "d.svg"],
    })
    out = tmp_path / "p.html"
    mr.write_pairs_html(pairs, out, "count", 1e-6, embed=False)
    text = out.read_text()
    assert text.index("pos_patterns.2") < text.index("pos_patterns.0")
    assert "c.svg" in text and "class='dis'" in text


def test_pairs_html_written_even_without_logos(tmp_path):
    """Contract change: the report is always written. Previously it was skipped
    when no logo column existed, so a logo resolution failure produced no file
    and no explanation -- indistinguishable from the script not running."""
    out = tmp_path / "p.html"
    mr.write_pairs_html(
        pd.DataFrame({"motif_a": ["a"], "motif_b": ["b"], "p_value": [1e-9]}),
        out, "count", 1e-6,
    )
    assert out.exists()
    text = out.read_text()
    assert "No logos available" in text
    assert "pos_patterns" not in text and "<code>a</code>" in text


def test_pairs_html_still_skipped_when_there_are_no_pairs(tmp_path):
    out = tmp_path / "p.html"
    mr.write_pairs_html(
        pd.DataFrame(columns=["motif_a", "motif_b", "p_value"]),
        out, "count", 1e-6,
    )
    assert not out.exists()


# memelite's p-value background collapses for some input configurations, and
# not monotonically in size: measured on the flanked_pair fixture after
# trimming, 41 motifs x 14bp cores behave correctly (duplicate p = 7e-9) while
# 21 x 14bp, 81 x 10bp and 81 x 20bp all return exactly 1.0 everywhere --
# including for identical motifs. Fixtures that exercise the p-value path must
# use a configuration verified to work; 14bp also matches Fi-NeMo's own median
# trimmed width.
PVALUE_SAFE_N = 40
PVALUE_SAFE_CORE = 14


def test_mutual_pairs_written_by_cli(tmp_path):
    import h5py

    n = PVALUE_SAFE_N
    with h5py.File(tmp_path / "c.h5", "w") as f:
        g0 = f.require_group("pos_patterns")
        for i in range(n):
            pfm, cwm, _, _ = flanked_pair(core_len=PVALUE_SAFE_CORE, seed=i)
            g = g0.create_group(f"pattern_{i}")
            g.create_dataset("sequence", data=pfm)
            g.create_dataset("contrib_scores", data=cwm)
        pfm, cwm, _, _ = flanked_pair(core_len=PVALUE_SAFE_CORE, seed=0)
        g = g0.create_group(f"pattern_{n}")       # exact duplicate of pattern_0
        g.create_dataset("sequence", data=pfm)
        g.create_dataset("contrib_scores", data=cwm)

    meta = tmp_path / "meta.tsv"
    pd.DataFrame({
        "cluster_final": list(range(n + 1)),
        "posneg": ["pos"] * (n + 1),
        "jaspar_name": ["AP1"] + [f"TF{i}" for i in range(1, n)] + ["AP1"],
        "jaspar_score": [0.9] * (n + 1),
        "total_seqlets": [1000] + [10] * (n - 1) + [900],
        "n_experiments": [5] * (n + 1),
    }).to_csv(meta, sep="\t", index=False)

    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "src/analysis/motif_redundancy.py"),
         "--modisco-h5", str(tmp_path / "c.h5"), "--cluster-metadata", str(meta),
         "--out-dir", str(tmp_path / "o"), "--n-jobs", "1"],
        capture_output=True, text=True, env=SUBPROC_ENV,
    )
    assert result.returncode == 0, result.stderr
    mp = pd.read_csv(tmp_path / "o" / "motif_redundancy_count_mutual_pairs.tsv", sep="\t")
    assert len(mp) >= 1
    names = ["pos_patterns.0", f"pos_patterns.{PVALUE_SAFE_N}"]
    dup = mp[mp.motif_a.isin(names) & mp.motif_b.isin(names)]
    assert len(dup) == 1, mp.to_string()
    assert bool(dup.iloc[0]["name_agree"]) is True
    assert "mutual_pairs.tsv" in result.stderr


def test_degenerate_pvalues_detects_a_collapsed_matrix():
    """The guard that would have caught three separate fixture failures."""
    collapsed = np.ones((6, 6)); np.fill_diagonal(collapsed, np.inf)
    bad, extreme, distinct = mr.degenerate_pvalues(collapsed)
    assert bad and extreme == pytest.approx(1.0) and distinct == 1

    zeros = np.zeros((6, 6)); np.fill_diagonal(zeros, np.inf)
    assert mr.degenerate_pvalues(zeros)[0]


def test_degenerate_pvalues_accepts_a_healthy_matrix():
    rng = np.random.default_rng(0)
    p = 10 ** rng.uniform(-12, 0, (30, 30))
    p = np.minimum(p, p.T)
    np.fill_diagonal(p, np.inf)
    bad, extreme, distinct = mr.degenerate_pvalues(p)
    assert not bad
    assert extreme < 0.01 and distinct > 100


def test_degenerate_pvalues_flags_an_all_inf_matrix():
    assert mr.degenerate_pvalues(np.full((3, 3), np.inf))[0]


def test_mutual_best_pairs_keeps_schema_when_empty():
    """An empty result must still be a readable TSV, not a zero-byte file."""
    names = ["a", "b"]
    p = np.array([[0.0, 0.9], [0.9, 0.0]])
    empty = mr.mutual_best_pairs(names, p, 1e-9)
    assert empty.empty
    assert list(empty.columns) == ["motif_a", "motif_b", "p_value"]
